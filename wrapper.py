"""
wrapper.py — BIAFLOWS-compatible entrypoint for CIDeconvolve.

Parses BIAFLOWS job parameters (--infolder, --outfolder, --gtfolder, etc.)
via bioflows_local, then processes each input image through the
deconvolution pipeline in deconvolve.py and writes results to the output
folder.

Usage (inside Docker):
    python wrapper.py --infolder /data/in --outfolder /data/out --gtfolder /data/gt --local

Usage (local):
    python wrapper.py --infolder ./infolder --outfolder ./outfolder --gtfolder ./gtfolder --local --iterations 40 --method sdeconv_rl
"""
import csv
import glob
import json
import logging
import os
import platform
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from importlib import metadata as importlib_metadata

# Configure logging so deconvolve.py INFO messages are visible
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


class _TeeWriter:
    """Duplicate writes to the original stream and a log file."""

    def __init__(self, original, log_file):
        self._original = original
        self._log_file = log_file

    def write(self, data):
        self._original.write(data)
        try:
            self._log_file.write(data)
        except Exception:
            pass

    def flush(self):
        self._original.flush()
        try:
            self._log_file.flush()
        except Exception:
            pass

    def __getattr__(self, name):
        return getattr(self._original, name)

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent))

from bioflows_local import (
    CLASS_SPTCNT,
    BiaflowsJob,
    get_discipline,
    prepare_data,
)

# Import deconvolve first (handles torch-before-numpy DLL load order)
from deconvolve import (
    MAX_TILE_XY,
    MAX_TILE_Z,
    METHODS,
    deconvolve_image,
    load_image,
    save_result,
)

import numpy as np

# ---------------------------------------------------------------------------
# Benchmark method sets
# ---------------------------------------------------------------------------
_BENCH_BASE = [
    "sdeconv_rl",
    "pycudadecon_rl_cuda",
    "ci_rl",
    "ci_rl_tv",
    "ci_sparse_hessian",
]
_BENCH_BASE_RLF = _BENCH_BASE + [
    "redlionfish_rl",
]
_BENCH_BASE_DW_RLF = _BENCH_BASE_RLF + [
    "deconwolf_rl",
    "deconwolf_shb",
]
_BENCH_BASE_DW_DL2_RLF = _BENCH_BASE_DW_RLF + [
    "deconvlab2_rl",
    "deconvlab2_rltv",
    "deconvlab2_landweber",
    "deconvlab2_ista",
]
_BENCH_FAST = [
    "sdeconv_wiener",
    "pycudadecon_rl_cuda",
    "ci_rl",
    "ci_rl_tv",
    "sdeconv_rl",
    "skimage_cucim_rl",
    "deconvlab2_landweber",
]
_BENCH_ALL = list(METHODS.keys())

BENCH_METHOD_SETS = {
    "sdeconv_rl, pycudadecon_rl_cuda, ci_rl, ci_rl_tv, ci_sparse_hessian": _BENCH_BASE,
    "sdeconv_rl, pycudadecon_rl_cuda, ci_rl, ci_rl_tv, ci_sparse_hessian, redlionfish_rl": _BENCH_BASE_RLF,
    "sdeconv_rl, pycudadecon_rl_cuda, ci_rl, ci_rl_tv, ci_sparse_hessian, deconwolf_rl, deconwolf_shb, redlionfish_rl": _BENCH_BASE_DW_RLF,
    "sdeconv_wiener, pycudadecon_rl_cuda, ci_rl, ci_rl_tv, ci_sparse_hessian, sdeconv_rl, skimage_cucim_rl, deconvlab2_landweber": _BENCH_FAST,
    "all": _BENCH_ALL,
}

# ---------------------------------------------------------------------------
# RI lookup tables — value-choices in descriptor.json use "name (RI)" format
# ---------------------------------------------------------------------------
_IMMERSION_RI = {
    "air":   1.0003,
    "water": 1.333,
    "oil":   1.515,
}

_SAMPLE_RI = {
    "water":          1.333,
    "pbs":            1.334,
    "culture medium": 1.337,
    "vectashield":    1.45,
    "glycerol":       1.474,
    "oil":            1.515,
    "prolong glass":  1.52,
}

# Default sample RI when "auto" is chosen and metadata has no value
_SAMPLE_RI_DEFAULT = 1.45  # Vectashield

BENCHMARK_CSV_FIELDS = [
    "dataset_id",
    "image_filename",
    "method",
    "iterations",
    "device",
    "status",
    "status_reason",
    "crop_mode",
    "tiling_mode",
    "tile_limits",
    "time_s",
    "cpu_percent_avg",
    "cpu_percent_peak",
    "ram_total_mb",
    "ram_peak_mb",
    "ram_percent",
    "ram_avg_mb",
    "ram_delta_peak_mb",
    "gpu_util_avg",
    "gpu_util_peak",
    "gpu_total_mb",
    "gpu_mem_peak_mb",
    "gpu_mem_percent",
    "gpu_mem_avg_mb",
    "gpu_mem_delta_peak_mb",
    "torch_gpu_peak_mb",
    "torch_gpu_delta_mb",
    "gpu_spill_mb",
    "channels_compared",
    "sharpness_mean",
    "contrast_mean",
    "noise_proxy_mean",
    "result_shape",
]


def _to_bool(value) -> bool:
    """Convert a value to bool, handling string 'True'/'False' from CLI."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


def _parse_ri_choice(raw: str, lookup: dict[str, float]) -> float | None:
    """Parse a RI choice string like 'oil (1.515)' or a bare float.

    Returns None for 'auto' (meaning: use image metadata / fallback).
    """
    s = str(raw).strip().lower()
    if s == "auto":
        return None
    # Try "name (1.234)" format — extract the name part
    name = s.split("(")[0].strip()
    if name in lookup:
        return lookup[name]
    # Try bare float
    try:
        return float(s)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Human-readable byte formatting
# ---------------------------------------------------------------------------
def _format_bytes(mb):
    """Format megabytes as a human-readable string."""
    if mb >= 1024:
        return f"{mb / 1024:.1f} GB"
    return f"{mb:.0f} MB"


def _safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _shape_to_str(images: list[np.ndarray]) -> str:
    return " | ".join("x".join(str(dim) for dim in img.shape) for img in images)


def _normalise_image(arr: np.ndarray) -> np.ndarray:
    img = np.asarray(arr, dtype=np.float64)
    lo = float(np.min(img))
    hi = float(np.max(img))
    if hi <= lo:
        return np.zeros_like(img, dtype=np.float64)
    return (img - lo) / (hi - lo)


def _mean_or_zero(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _no_reference_metrics(arr: np.ndarray) -> dict[str, float]:
    img = _normalise_image(arr)
    p1 = float(np.percentile(img, 1))
    p99 = float(np.percentile(img, 99))
    q1 = float(np.percentile(img, 25))
    q3 = float(np.percentile(img, 75))

    try:
        from scipy import ndimage

        if img.ndim == 3:
            lap_vars = [float(np.var(ndimage.laplace(z))) for z in img]
            sharpness = float(np.mean(lap_vars)) if lap_vars else 0.0
        else:
            sharpness = float(np.var(ndimage.laplace(img)))
    except ImportError:
        grads = np.gradient(img.astype(np.float64))
        sharpness = float(np.mean([np.var(g) for g in grads])) if grads else 0.0

    return {
        "sharpness": sharpness,
        "contrast": p99 - p1,
        "noise_proxy": q3 - q1,
    }


def _quality_metrics(
    result_channels: list[np.ndarray],
) -> dict[str, float | int]:
    sharpness_vals: list[float] = []
    contrast_vals: list[float] = []
    noise_vals: list[float] = []

    for result in result_channels:
        nr = _no_reference_metrics(result)
        sharpness_vals.append(nr["sharpness"])
        contrast_vals.append(nr["contrast"])
        noise_vals.append(nr["noise_proxy"])
    return {
        "channels_compared": len(result_channels),
        "sharpness_mean": _mean_or_zero(sharpness_vals),
        "contrast_mean": _mean_or_zero(contrast_vals),
        "noise_proxy_mean": _mean_or_zero(noise_vals),
    }


def _stem(filename: str) -> str:
    name = filename
    for suffix in (".ome.tiff", ".ome.tif", ".tiff", ".tif", ".png", ".jpg", ".jpeg", ".bmp", ".npy", ".zarr"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def _collect_run_provenance(
    *,
    img_path: Path,
    data: dict,
    meta_overrides: dict,
    cli_overrides: set[str],
    bench_crop: bool,
    tiling: str,
    max_tile_xy: int,
    max_tile_z: int,
    benchmark_rows: list[dict],
) -> dict:
    meta = data["metadata"]

    cpu_name = platform.processor() or platform.machine()
    try:
        import psutil

        ram_total_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    except Exception:
        ram_total_gb = 0.0

    gpu_name = ""
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode("utf-8", errors="replace")
    except Exception:
        gpu_name = ""

    versions = {}
    for pkg in ("numpy", "scipy", "scikit-image", "torch", "tifffile"):
        try:
            versions[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            continue

    descriptor_image = ""
    try:
        descriptor = json.loads(Path(__file__).with_name("descriptor.json").read_text(encoding="utf-8"))
        descriptor_image = descriptor.get("container-image", {}).get("image", "")
    except Exception:
        pass

    version_txt = ""
    try:
        version_txt = Path(__file__).with_name("version.txt").read_text(encoding="utf-8").strip()
    except Exception:
        pass

    defaulted = set(meta.get("_defaulted_keys", set()))
    psf_fields = {}
    for field in (
        "na",
        "refractive_index",
        "sample_refractive_index",
        "microscope_type",
        "pixel_size_x",
        "pixel_size_y",
        "pixel_size_z",
    ):
        source = "image_metadata"
        if field in cli_overrides:
            source = "cli_override"
        elif field in defaulted:
            source = "fallback_default"
        psf_fields[field] = {"value": meta.get(field), "source": source}

    psf_fields["emission_wavelengths"] = {
        "value": [ch.get("emission_wavelength") for ch in meta.get("channels", [])],
        "source": "cli_override" if "emission_wavelength" in cli_overrides else (
            "fallback_default" if "emission_wavelength" in defaulted else "image_metadata"
        ),
    }
    psf_fields["excitation_wavelengths"] = {
        "value": [ch.get("excitation_wavelength") for ch in meta.get("channels", [])],
        "source": "cli_override" if "excitation_wavelength" in cli_overrides else "image_metadata",
    }

    return {
        "benchmark_version": 1,
        "repo_version": version_txt,
        "container_image": descriptor_image,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "python": platform.python_version(),
            "cpu": cpu_name,
            "gpu": gpu_name,
            "ram_total_gb": ram_total_gb,
        },
        "package_versions": versions,
        "dataset": {
            "image_filename": img_path.name,
            "shape": _shape_to_str(data["images"]),
            "n_channels": len(data["images"]),
            "is_3d": bool(data["images"] and data["images"][0].ndim == 3),
            "crop_mode": "center_crop" if bench_crop else "full_volume",
            "tiling_mode": tiling,
            "tile_limits": {"max_xy": max_tile_xy, "max_z": max_tile_z},
        },
        "psf_parameters": psf_fields,
        "meta_overrides": meta_overrides,
        "benchmark_rows": len(benchmark_rows),
    }


def _method_device(method: str) -> str:
    """Return the compute device label for a benchmark method."""
    _GPU_METHODS = {
        "pycudadecon_rl_cuda", "skimage_cucim_rl",
    }
    _CPU_METHODS = {
        "skimage_rl", "skimage_unsupervised_wiener",
        "deconvlab2_rl", "deconvlab2_rltv",
        "deconvlab2_landweber", "deconvlab2_ista",
        "deconwolf_rl", "deconwolf_shb",
    }
    if method in _GPU_METHODS:
        return "CUDA"
    if method in _CPU_METHODS:
        return "CPU"
    # ci_rl / ci_rl_tv / ci_sparse_hessian — PyTorch: CUDA when available, else CPU
    if method.startswith("ci_"):
        import torch
        return "CUDA" if torch.cuda.is_available() else "CPU"
    # redlionfish uses OpenCL — prefer GPU device, fall back to CPU device
    if method == "redlionfish_rl":
        try:
            import pyopencl
            has_gpu = False
            has_cpu = False
            for p in pyopencl.get_platforms():
                for d in p.get_devices():
                    if d.type & pyopencl.device_type.GPU:
                        has_gpu = True
                    if d.type & pyopencl.device_type.CPU:
                        has_cpu = True
            if has_gpu:
                return "CL/GPU"
            if has_cpu:
                return "CL/CPU"
        except Exception:
            pass
        return "CPU"
    return "?"


# ---------------------------------------------------------------------------
# Background metrics monitor
# ---------------------------------------------------------------------------
class _MetricsMonitor:
    """Daemon thread that samples CPU/RAM and GPU metrics during a run."""

    def __init__(self, interval=0.1):
        self._interval = interval
        self._stop_event = threading.Event()
        self._thread = None

        # Sampled data
        self._cpu_percent = []
        self._ram_bytes = []
        self._gpu_util = []
        self._gpu_mem_bytes = []

        # Baselines (set on start)
        self._ram_baseline = 0
        self._gpu_mem_baseline = 0

        # Timing
        self._t0 = 0.0
        self._t1 = 0.0

        # Detect capabilities
        self._proc = None
        try:
            import psutil
            self._proc = psutil.Process(os.getpid())
        except ImportError:
            pass

        self._nvml_handle = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            pass

    def start(self):
        """Record baselines and begin sampling."""
        self._cpu_percent.clear()
        self._ram_bytes.clear()
        self._gpu_util.clear()
        self._gpu_mem_bytes.clear()
        self._stop_event.clear()

        # CPU/RAM baseline
        if self._proc:
            self._proc.cpu_percent()  # prime the first call
            self._ram_baseline = self._proc.memory_info().rss
        else:
            self._ram_baseline = 0

        # GPU baseline
        if self._nvml_handle:
            import pynvml
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
            self._gpu_mem_baseline = mem_info.used
        else:
            self._gpu_mem_baseline = 0

        # Reset PyTorch peak stats and record baseline
        self._torch_baseline = 0
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                self._torch_baseline = torch.cuda.memory_allocated()
        except Exception:
            pass

        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _poll(self):
        """Sampling loop running in background thread."""
        while not self._stop_event.is_set():
            # CPU + RAM
            if self._proc:
                try:
                    self._cpu_percent.append(self._proc.cpu_percent())
                    self._ram_bytes.append(self._proc.memory_info().rss)
                except Exception:
                    pass

            # GPU
            if self._nvml_handle:
                try:
                    import pynvml
                    util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                    self._gpu_util.append(util.gpu)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                    self._gpu_mem_bytes.append(mem_info.used)
                except Exception:
                    pass

            self._stop_event.wait(self._interval)

    def stop(self):
        """Stop sampling and return metrics dict."""
        self._t1 = time.perf_counter()
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)

        elapsed = self._t1 - self._t0
        MB = 1024 * 1024

        m = {
            "time_s": elapsed,
            "cpu_percent_avg": 0.0,
            "cpu_percent_peak": 0.0,
            "ram_peak_mb": 0.0,
            "ram_avg_mb": 0.0,
            "ram_delta_peak_mb": 0.0,
            "gpu_util_avg": 0.0,
            "gpu_util_peak": 0.0,
            "gpu_mem_peak_mb": 0.0,
            "gpu_mem_avg_mb": 0.0,
            "gpu_mem_delta_peak_mb": 0.0,
            "torch_gpu_peak_mb": 0.0,
            "torch_gpu_delta_mb": 0.0,
            "gpu_total_mb": 0.0,
            "gpu_spill_mb": 0.0,
            "ram_total_mb": 0.0,
            "ram_percent": 0.0,
            "gpu_mem_percent": 0.0,
        }

        if self._cpu_percent:
            m["cpu_percent_avg"] = sum(self._cpu_percent) / len(self._cpu_percent)
            m["cpu_percent_peak"] = max(self._cpu_percent)

        if self._proc:
            import psutil
            m["ram_total_mb"] = psutil.virtual_memory().total / MB

        if self._ram_bytes:
            m["ram_peak_mb"] = max(self._ram_bytes) / MB
            m["ram_avg_mb"] = sum(self._ram_bytes) / len(self._ram_bytes) / MB
            m["ram_delta_peak_mb"] = (max(self._ram_bytes) - self._ram_baseline) / MB
            if m["ram_total_mb"] > 0:
                m["ram_percent"] = m["ram_peak_mb"] / m["ram_total_mb"] * 100

        if self._gpu_util:
            m["gpu_util_avg"] = sum(self._gpu_util) / len(self._gpu_util)
            m["gpu_util_peak"] = max(self._gpu_util)

        if self._gpu_mem_bytes:
            m["gpu_mem_peak_mb"] = max(self._gpu_mem_bytes) / MB
            m["gpu_mem_avg_mb"] = sum(self._gpu_mem_bytes) / len(self._gpu_mem_bytes) / MB
            m["gpu_mem_delta_peak_mb"] = (max(self._gpu_mem_bytes) - self._gpu_mem_baseline) / MB

        # Physical VRAM total
        if self._nvml_handle:
            try:
                import pynvml
                total_info = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                m["gpu_total_mb"] = total_info.total / MB
            except Exception:
                pass

        try:
            import torch
            if torch.cuda.is_available():
                m["torch_gpu_peak_mb"] = torch.cuda.max_memory_allocated() / MB
                m["torch_gpu_delta_mb"] = (
                    torch.cuda.max_memory_allocated() - self._torch_baseline
                ) / MB
        except Exception:
            pass

        # GPU memory percent
        if m["gpu_total_mb"] > 0 and m["gpu_mem_peak_mb"] > 0:
            m["gpu_mem_percent"] = m["gpu_mem_peak_mb"] / m["gpu_total_mb"] * 100

        # Spillover: when torch allocation exceeds physical VRAM
        if m["gpu_total_mb"] > 0 and m["torch_gpu_delta_mb"] > m["gpu_total_mb"]:
            m["gpu_spill_mb"] = m["torch_gpu_delta_mb"] - m["gpu_total_mb"]

        return m


def _check_method_available(method: str) -> tuple[bool, str]:
    """Return (available, reason) for a benchmark method."""
    import torch
    cuda = torch.cuda.is_available()

    # Methods requiring CUDA
    if method in ("pycudadecon_rl_cuda", "skimage_cucim_rl") and not cuda:
        return False, "no CUDA GPU"

    # sdeconv_rl uses CUDA when available, falls back to CPU
    if method == "pycudadecon_rl_cuda":
        try:
            import pycudadecon  # noqa: F401
        except ImportError:
            return False, "pycudadecon not installed"

    if method == "skimage_cucim_rl":
        try:
            import cupy  # noqa: F401
            import cucim  # noqa: F401
        except ImportError:
            return False, "cupy/cucim not installed"

    if method == "redlionfish_rl":
        try:
            import RedLionfishDeconv  # noqa: F401
        except ImportError:
            return False, "RedLionfishDeconv not installed"

    if method.startswith("deconwolf_"):
        from deconvolve import _DW_EXE
        if not _DW_EXE:
            return False, "dw executable not found"

    if method.startswith("deconvlab2_"):
        from deconvolve import _DECONVLAB2_JAR, _IJ_JAR
        if not (_DECONVLAB2_JAR.exists() and _IJ_JAR.exists() and shutil.which("java")):
            return False, "DeconvolutionLab2 JAR or java not found"

    return True, ""


def main(argv):
    with BiaflowsJob.from_cli(argv) as bj:
        parameters = getattr(bj, "parameters", SimpleNamespace())

        # Extract parameters with defaults from descriptor.json
        iterations = int(getattr(parameters, "iterations", 40))
        tiling_raw = getattr(parameters, "tiling", "custom")
        tile_limits_raw = str(getattr(parameters, "tile_limits", "512, 64"))
        method = getattr(parameters, "method", "sdeconv_rl")
        device_param = getattr(parameters, "device", "auto")
        device = None if device_param in (None, "auto") else device_param

        # PSF metadata overrides
        na_raw = getattr(parameters, "na", "auto")
        na_override = None if str(na_raw).strip().lower() == "auto" else float(na_raw)
        ri_raw = str(getattr(parameters, "refractive_index", "auto"))
        ri_override = _parse_ri_choice(ri_raw, _IMMERSION_RI)
        sample_ri_raw = str(getattr(parameters, "sample_ri", "auto"))
        sample_ri_parsed = _parse_ri_choice(sample_ri_raw, _SAMPLE_RI)
        sample_ri = sample_ri_parsed if sample_ri_parsed is not None else _SAMPLE_RI_DEFAULT
        micro_raw = getattr(parameters, "microscope_type", "auto")
        micro_override = None if str(micro_raw).strip().lower() == "auto" else str(micro_raw)
        em_raw = str(getattr(parameters, "emission_wl", "auto")).strip()
        em_override = (
            None if em_raw.lower() == "auto"
            else [float(x.strip()) for x in em_raw.split(",") if x.strip()]
        )
        ex_raw = str(getattr(parameters, "excitation_wl", "auto")).strip()
        ex_override = (
            None if ex_raw.lower() == "auto" or not ex_raw
            else [float(x.strip()) for x in ex_raw.split(",") if x.strip()]
        ) or None

        benchmark_mode = _to_bool(getattr(parameters, "benchmark", False))
        projection = str(getattr(parameters, "projection", "none")).lower()
        save_psf = _to_bool(getattr(parameters, "psf", False))
        log_to_file = _to_bool(getattr(parameters, "log", False))

        # Set up console-to-file tee if requested
        _log_fh = None
        if log_to_file:
            log_path = Path(bj.output_dir) / "cideconvolve.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            _log_fh = open(log_path, "w", encoding="utf-8")
            sys.stdout = _TeeWriter(sys.__stdout__, _log_fh)
            sys.stderr = _TeeWriter(sys.__stderr__, _log_fh)
            # Also send logging module output to the file
            _file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
            _file_handler.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            ))
            logging.getLogger().addHandler(_file_handler)

        # Benchmark-specific parameters
        bench_iter_raw = str(getattr(parameters, "bench_iterations", "20, 40, 60"))
        bench_iterations = [int(x.strip()) for x in bench_iter_raw.split(",") if x.strip()]
        bench_methods_key = str(getattr(parameters, "bench_methods", "fast")).lower()
        bench_methods = BENCH_METHOD_SETS.get(bench_methods_key)
        if bench_methods is None:
            # Key doesn't match a preset — try parsing as comma-separated method names
            parsed = [m.strip() for m in bench_methods_key.split(",") if m.strip()]
            valid = [m for m in parsed if m in METHODS]
            bench_methods = valid if valid else _BENCH_BASE
        bench_crop = _to_bool(getattr(parameters, "bench_crop", True))
        bench_one_image = _to_bool(getattr(parameters, "bench_one_image", True))
        bench_montage = _to_bool(getattr(parameters, "bench_montage", True))

        # Parse tiling
        tiling = str(tiling_raw).strip().lower()
        if tiling not in ("none", "custom"):
            tiling = "none"  # fallback

        # Parse tile limits (max_xy, max_z)
        _lim_parts = [s.strip() for s in tile_limits_raw.split(",") if s.strip()]
        max_tile_xy = int(_lim_parts[0]) if len(_lim_parts) >= 1 else MAX_TILE_XY
        max_tile_z = int(_lim_parts[1]) if len(_lim_parts) >= 2 else MAX_TILE_Z

        print("=" * 70)
        print("CIDeconvolve - BIAFLOWS Workflow")
        print("=" * 70)
        print(f"  Input dir    : {bj.input_dir}")
        print(f"  Output dir   : {bj.output_dir}")
        print(f"  Method       : {method}")
        print(f"  Iterations   : {iterations}")
        print(f"  Tiling       : {tiling}")
        if tiling == "custom":
            print(f"  Tile limits  : max_xy={max_tile_xy}, max_z={max_tile_z}")
        print(f"  Device       : {device_param}")
        print(f"  Benchmark    : {benchmark_mode}")
        print(f"  Projection   : {projection}")
        print(f"  Save PSF     : {save_psf}")
        if na_override is not None:
            print(f"  NA           : {na_override}")
        if ri_override is not None:
            print(f"  Immersion    : {ri_raw} -> RI {ri_override}")
        if sample_ri_parsed is not None:
            print(f"  Sample medium: {sample_ri_raw} -> RI {sample_ri}")
        else:
            print(f"  Sample medium: auto -> vectashield (RI {sample_ri})")
        if micro_override is not None:
            print(f"  Microscope   : {micro_override}")
        if em_override is not None:
            print(f"  Emission WL  : {em_override}")
        if ex_override is not None:
            print(f"  Excitation WL: {ex_override}")
        if benchmark_mode:
            print(f"  Bench iters  : {bench_iterations}")
            print(f"  Bench methods: {bench_methods_key} ({len(bench_methods)} methods)")
            print(f"  Bench crop   : {bench_crop}")
            print(f"  Bench 1 image: {bench_one_image}")
            print(f"  Bench montage: {bench_montage}")

        # Prepare data directories and collect input images
        in_imgs, _, in_path, _, out_path, tmp_path = prepare_data(
            get_discipline(bj, default=CLASS_SPTCNT), bj, is_2d=False, **bj.flags
        )

        if not in_imgs:
            print("No input images found. Exiting.")
            return

        print(f"\nFound {len(in_imgs)} input image(s).")

        # In benchmark mode with bench_one_image, keep only the first image
        if benchmark_mode and bench_one_image and len(in_imgs) > 1:
            in_imgs_sorted = sorted(in_imgs, key=lambda r: r.filename)
            in_imgs = [in_imgs_sorted[0]]
            print(f"Benchmark one image: using {in_imgs[0].filename}")

        for img_resource in in_imgs:
            img_path = Path(in_path) / img_resource.filename
            print(f"\n{'=' * 60}")
            print(f"Processing: {img_resource.filename}")
            print(f"{'=' * 60}")

            t0 = time.time()

            try:
                # Load image and extract metadata
                data = load_image(img_path)
                meta = data["metadata"]
                images = data["images"]

                print(f"  Channels: {len(images)}")
                for i, img in enumerate(images):
                    print(f"    Ch{i}: shape={img.shape}, dtype={img.dtype}")

                # Create a temp dir alongside outfolder for intermediate files
                tmp_work = Path(out_path) / "tmp"
                tmp_work.mkdir(parents=True, exist_ok=True)

                if benchmark_mode:
                    # ----- Benchmark path -----
                    _run_benchmark(
                        img_path,
                        data,
                        str(tmp_work),
                        stem=_stem(img_resource.filename),
                        bench_iterations=bench_iterations,
                        bench_methods=bench_methods,
                        bench_crop=bench_crop,
                        bench_montage=bench_montage,
                        tiling=tiling,
                        max_tile_xy=max_tile_xy,
                        max_tile_z=max_tile_z,
                        save_psf=save_psf,
                        na=na_override,
                        refractive_index=ri_override,
                        sample_refractive_index=sample_ri_parsed,
                        microscope_type=micro_override,
                        emission_wavelengths=em_override,
                        excitation_wavelengths=ex_override,
                    )
                    # Move final montage PNGs from tmp to output
                    for png in tmp_work.glob("decon_benchmark_*.png"):
                        dest = Path(out_path) / png.name
                        shutil.move(str(png), str(dest))
                        print(f"  -> {dest.name}")
                    # Move metrics CSV from tmp to output
                    for csvf in tmp_work.glob("benchmark_metrics_*.csv"):
                        dest = Path(out_path) / csvf.name
                        shutil.move(str(csvf), str(dest))
                        print(f"  -> {dest.name}")
                    for meta_json in tmp_work.glob("benchmark_provenance_*.json"):
                        dest = Path(out_path) / meta_json.name
                        shutil.move(str(meta_json), str(dest))
                        print(f"  -> {dest.name}")
                else:
                    # ----- Normal single-method path -----
                    result = deconvolve_image(
                        img_path,
                        method=method,
                        niter=iterations,
                        tiling=tiling,
                        max_tile_xy=max_tile_xy,
                        max_tile_z=max_tile_z,
                        device=device,
                        na=na_override,
                        refractive_index=ri_override,
                        sample_refractive_index=sample_ri,
                        microscope_type=micro_override,
                        emission_wavelengths=em_override,
                        excitation_wavelengths=ex_override,
                    )

                    if result is None:
                        print(f"  WARNING: deconvolve_image returned None for {img_resource.filename}")
                        shutil.rmtree(tmp_work, ignore_errors=True)
                        continue

                    stem = _stem(img_resource.filename)
                    is_3d = result["channels"][0].ndim == 3

                    if projection in ("mip", "sum") and is_3d:
                        out_name = f"{stem}_decon_{projection}.ome.tiff"
                        tmp_file = tmp_work / out_name
                        proj_result = dict(result)
                        if projection == "mip":
                            proj_result["channels"] = [
                                ch.max(axis=0) for ch in result["channels"]
                            ]
                            if result.get("source_channels"):
                                proj_result["source_channels"] = [
                                    ch.max(axis=0) for ch in result["source_channels"]
                                ]
                        else:  # sum
                            proj_result["channels"] = [
                                ch.astype(np.float32).sum(axis=0) for ch in result["channels"]
                            ]
                            if result.get("source_channels"):
                                proj_result["source_channels"] = [
                                    ch.astype(np.float32).sum(axis=0) for ch in result["source_channels"]
                                ]
                        save_result(proj_result, str(tmp_file))
                        print(f"  Saved {projection.upper()}: {out_name}")
                    else:
                        out_name = f"{stem}_decon.ome.tiff"
                        tmp_file = tmp_work / out_name
                        save_result(result, str(tmp_file))
                        print(f"  Saved: {out_name}")

                    # Move only the deconvolved TIFF to the output folder
                    dest = Path(out_path) / out_name
                    shutil.move(str(tmp_file), str(dest))

                # Clean up the temp working directory
                shutil.rmtree(tmp_work, ignore_errors=True)

            except Exception as exc:
                print(f"  ERROR processing {img_resource.filename}: {exc}")
                import traceback
                traceback.print_exc()
                # Clean up temp dir on failure so no partial files remain
                tmp_work = Path(out_path) / "tmp"
                shutil.rmtree(tmp_work, ignore_errors=True)
                continue

            elapsed = time.time() - t0
            print(f"  Time: {elapsed:.1f}s")

        print(f"\n{'=' * 70}")
        print("CIDeconvolve workflow complete.")
        print(f"{'=' * 70}")

        # Close log file tee if active
        if _log_fh is not None:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            _log_fh.close()
            print(f"Console log saved to: {Path(bj.output_dir) / 'cideconvolve.log'}")

        # Clean up tmp folder
        if tmp_path and Path(tmp_path).exists():
            shutil.rmtree(tmp_path, ignore_errors=True)
            print(f"Cleaned up tmp folder: {tmp_path}")
def _make_metadata_panel(meta, width, height, font):
    """Create an image panel showing PSF-relevant metadata.

    Values are annotated to show their source:
      (default)  – not in image metadata, fallback value used
      (override) – explicitly set via CLI parameter
      (no annotation) – from image metadata
    """
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (width, height), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)

    defaulted = meta.get("_defaulted_keys", set())
    cli = meta.get("_cli_overrides", set())

    def _tag(key):
        """Return a short annotation string for a metadata key."""
        if key in cli:
            return " (override)"
        if key in defaulted:
            return " (default)"
        return ""

    def _ri_label(ri_val, lookup):
        """Reverse-lookup a RI value to its medium name."""
        if ri_val is None:
            return "?"
        for name, val in lookup.items():
            if abs(val - ri_val) < 0.001:
                return f"{ri_val} ({name})"
        return str(ri_val)

    ri_imm = meta.get('refractive_index')
    ri_sam = meta.get('sample_refractive_index')

    lines = [
        f"Microscope: {meta.get('microscope_type', '?')}{_tag('microscope_type')}",
        f"NA: {meta.get('na', '?')}{_tag('na')}",
        f"RI (imm): {_ri_label(ri_imm, _IMMERSION_RI)}{_tag('refractive_index')}",
        f"RI (sample): {_ri_label(ri_sam, _SAMPLE_RI)}{_tag('sample_refractive_index')}",
        f"Pixel XY: {meta.get('pixel_size_x', '?')} \u00b5m{_tag('pixel_size_x')}",
        f"Pixel Z:  {meta.get('pixel_size_z', '?')} \u00b5m{_tag('pixel_size_z')}",
        f"Size: {meta.get('size_x', '?')}\u00d7{meta.get('size_y', '?')}\u00d7{meta.get('size_z', '?')}",
        "",
    ]
    em_tag = _tag("emission_wavelength")
    ex_tag = _tag("excitation_wavelength")
    for i, ch in enumerate(meta.get("channels", [])):
        em = ch.get("emission_wavelength") or "?"
        ex = ch.get("excitation_wavelength") or "—"
        lines.append(f"Ch{i}: Ex {ex}{ex_tag} / Em {em}{em_tag} nm")

    text = "\n".join(lines)
    draw.text((8, 8), text, fill=(255, 255, 255), font=font)
    return img


def _make_benchmark_montage(
    out_path,
    stem,
    available_methods,
    bench_iterations,
    all_metrics,
    metadata,
):
    """Create an RGB montage of all benchmark MIP PNGs and return its path."""
    from PIL import Image, ImageDraw, ImageFont

    out_dir = Path(out_path)

    font = None
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            font = ImageFont.truetype(name, 18)
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()

    # Column order — sdeconv methods use cuda if available, else cpu
    import torch as _torch
    _sdeconv_dev = "cuda" if _torch.cuda.is_available() else "cpu"
    col_order = []
    for m in available_methods:
        if m.startswith("sdeconv_"):
            col_order.append((m, _sdeconv_dev))
        else:
            col_order.append((m, None))

    # Row 0: source MIP
    rows = [[(out_dir / "mip_source.ome.png", "Source")]]

    # One row per iteration count
    for nit in bench_iterations:
        row = []
        for method, variant in col_order:
            if variant:
                fname = f"mip_{stem}_{method}_{variant}_{nit}i.ome.png"
                metrics_key = f"{method}_{variant}_{nit}i"
                label_name = f"{method}_{variant}"
            else:
                fname = f"mip_{stem}_{method}_{nit}i.ome.png"
                metrics_key = f"{method}_{nit}i"
                label_name = method

            met = all_metrics.get(metrics_key)
            if met is not None:
                label = (f"{label_name}\n"
                         f"{nit} iter  {met['time_s']:.1f}s")
            else:
                label = f"{label_name}\n{nit} iter"
            row.append((out_dir / fname, label))
        rows.append(row)

    # Load existing images, skip missing
    loaded_rows = []
    total = 0
    for row_entries in rows:
        row_images = []
        for path, label in row_entries:
            if path.exists():
                img = Image.open(path).convert("RGB")
                row_images.append((img, label))
                total += 1
        if row_images:
            loaded_rows.append(row_images)

    if total == 0:
        print("  No MIP PNG files found \u2014 skipping montage.")
        return None

    label_height = 78
    padding = 4

    all_imgs = [img for row in loaded_rows for img, _ in row]
    max_w = max(img.size[0] for img in all_imgs)
    max_h = max(img.size[1] for img in all_imgs)

    n_cols = max(len(row) for row in loaded_rows)
    n_rows = len(loaded_rows)
    cell_w = max_w + 2 * padding
    cell_h = max_h + label_height + 2 * padding

    # Metadata panel spans all columns to the right of Source
    span_cols = max(n_cols - 1, 1)
    meta_w = span_cols * cell_w - 2 * padding
    meta_panel = _make_metadata_panel(metadata, meta_w, max_h, font)

    montage_w = n_cols * cell_w
    montage_h = n_rows * cell_h
    montage = Image.new("RGB", (montage_w, montage_h), color=(0, 0, 0))
    draw = ImageDraw.Draw(montage)

    for row_idx, row_images in enumerate(loaded_rows):
        for col_idx, (img, label) in enumerate(row_images):
            x0 = col_idx * cell_w + padding
            y0 = row_idx * cell_h + padding
            x_off = (max_w - img.size[0]) // 2
            y_off = (max_h - img.size[1]) // 2
            montage.paste(img, (x0 + x_off, y0 + y_off))

            bbox = draw.textbbox((0, 0), label, font=font)
            tw = bbox[2] - bbox[0]
            tx = x0 + (max_w - tw) // 2
            ty = y0 + max_h + padding
            draw.text((tx, ty), label, fill=(255, 255, 255), font=font)

    # Paste wide metadata panel at row 0, starting at column 1
    meta_x = cell_w + padding
    meta_y = padding
    montage.paste(meta_panel, (meta_x, meta_y))

    montage_path = out_dir / f"decon_benchmark_{stem}.png"
    montage.save(str(montage_path))
    print(f"  Saved montage: {montage_path}  ({montage_w}x{montage_h})")
    return montage_path


def _make_per_channel_montages(
    out_path,
    stem,
    available_methods,
    bench_iterations,
    all_metrics,
    metadata,
):
    """Create one greyscale montage per channel from benchmark MIP TIFFs."""
    from PIL import Image, ImageDraw, ImageFont
    import tifffile

    out_dir = Path(out_path)
    n_ch = metadata.get("n_channels", 1)

    font = None
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            font = ImageFont.truetype(name, 18)
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()

    # Build column order — sdeconv methods use cuda if available, else cpu
    import torch as _torch
    _sdeconv_dev = "cuda" if _torch.cuda.is_available() else "cpu"
    col_order = []
    for m in available_methods:
        if m.startswith("sdeconv_"):
            col_order.append((m, _sdeconv_dev))
        else:
            col_order.append((m, None))

    # Build row definitions: row 0 = source, rows 1-N = per iteration count
    mip_rows = [
        [("Source", out_dir / "mip_source.ome.tiff")],
    ]
    for nit in bench_iterations:
        row = []
        for method, variant in col_order:
            if variant:
                fname = f"mip_{stem}_{method}_{variant}_{nit}i.ome.tiff"
                metrics_key = f"{method}_{variant}_{nit}i"
                label_name = f"{method}_{variant}"
            else:
                fname = f"mip_{stem}_{method}_{nit}i.ome.tiff"
                metrics_key = f"{method}_{nit}i"
                label_name = method

            met = all_metrics.get(metrics_key)
            if met is not None:
                label = (f"{label_name}\n"
                         f"{nit} iter  {met['time_s']:.1f}s")
            else:
                label = f"{label_name}\n{nit} iter"
            row.append((label, out_dir / fname))
        mip_rows.append(row)

    # Load existing TIFF arrays per row
    loaded_rows = []
    for row_entries in mip_rows:
        row_data = []
        for label, path in row_entries:
            if path.exists():
                arr = tifffile.imread(str(path))
                if arr.ndim == 2:
                    arr = arr[np.newaxis]
                row_data.append((label, arr))
        if row_data:
            loaded_rows.append(row_data)

    if not loaded_rows:
        return

    print(f"\n  Creating per-channel montages ({n_ch} channels)...")

    label_height = 78
    padding = 4

    for ch_idx in range(n_ch):
        panel_rows = []
        for row_data in loaded_rows:
            row_panels = []
            for label, arr in row_data:
                if ch_idx >= arr.shape[0]:
                    continue
                ch_img = arr[ch_idx].astype(np.float64)
                lo, hi = ch_img.min(), ch_img.max()
                if hi > lo:
                    ch_img = (ch_img - lo) / (hi - lo) * 255.0
                else:
                    ch_img = np.zeros_like(ch_img)
                grey = ch_img.clip(0, 255).astype(np.uint8)
                img = Image.fromarray(grey, mode="L").convert("RGB")

                # Channel label overlay
                img_draw = ImageDraw.Draw(img)
                img_draw.text((4, 2), f"Ch{ch_idx}", fill=(255, 255, 255), font=font)
                row_panels.append((img, label))
            if row_panels:
                panel_rows.append(row_panels)

        if not panel_rows:
            continue

        all_imgs = [img for row in panel_rows for img, _ in row]
        max_w = max(img.size[0] for img in all_imgs)
        max_h = max(img.size[1] for img in all_imgs)

        n_cols = max(len(row) for row in panel_rows)
        n_grid_rows = len(panel_rows)
        cell_w = max_w + 2 * padding
        cell_h = max_h + label_height + 2 * padding

        # Metadata panel spans all columns to the right of Source
        span_cols = max(n_cols - 1, 1)
        meta_w = span_cols * cell_w - 2 * padding
        meta_panel = _make_metadata_panel(metadata, meta_w, max_h, font)

        montage_w = n_cols * cell_w
        montage_h = n_grid_rows * cell_h
        montage = Image.new("RGB", (montage_w, montage_h), color=(0, 0, 0))
        draw = ImageDraw.Draw(montage)

        for row_idx, row_panels in enumerate(panel_rows):
            for col_idx, (img, label) in enumerate(row_panels):
                x0 = col_idx * cell_w + padding
                y0 = row_idx * cell_h + padding
                x_off = (max_w - img.size[0]) // 2
                y_off = (max_h - img.size[1]) // 2
                montage.paste(img, (x0 + x_off, y0 + y_off))

                bbox = draw.textbbox((0, 0), label, font=font)
                tw = bbox[2] - bbox[0]
                tx = x0 + (max_w - tw) // 2
                ty = y0 + max_h + padding
                draw.text((tx, ty), label, fill=(255, 255, 255), font=font)

        # Paste wide metadata panel at row 0, starting at column 1
        meta_x = cell_w + padding
        meta_y = padding
        montage.paste(meta_panel, (meta_x, meta_y))

        ch_path = out_dir / f"decon_benchmark_{stem}_ch{ch_idx}.png"
        montage.save(str(ch_path))
        print(f"    Ch{ch_idx}: {ch_path}  ({montage_w}x{montage_h})")


def _blank_benchmark_row(
    *,
    dataset_id: str,
    image_filename: str,
    method: str,
    iterations: int,
    crop_mode: str,
    tiling_mode: str,
    tile_limits: str,
) -> dict:
    row = {key: "" for key in BENCHMARK_CSV_FIELDS}
    row.update(
        {
            "dataset_id": dataset_id,
            "image_filename": image_filename,
            "method": method,
            "iterations": iterations,
            "crop_mode": crop_mode,
            "tiling_mode": tiling_mode,
            "tile_limits": tile_limits,
            "status": "pending",
        }
    )
    return row


def _write_metrics_csv(csv_path: Path, rows: list[dict]):
    """Write benchmark metrics to a CSV file."""
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=BENCHMARK_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            serialized = {}
            for key in BENCHMARK_CSV_FIELDS:
                value = row.get(key, "")
                if isinstance(value, float):
                    serialized[key] = f"{value:.6f}"
                elif value is None:
                    serialized[key] = ""
                else:
                    serialized[key] = value
            writer.writerow(serialized)
    print(f"\n  Metrics CSV saved -> {csv_path}")


def _write_provenance_json(json_path: Path, payload: dict) -> None:
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(f"  Benchmark provenance saved -> {json_path}")


def _run_benchmark(
    img_path: Path,
    data: dict,
    out_path: str,
    *,
    stem: str,
    bench_iterations: list[int],
    bench_methods: list[str],
    bench_crop: bool,
    bench_montage: bool,
    tiling: str,
    max_tile_xy: int,
    max_tile_z: int,
    save_psf: bool,
    na: float | None = None,
    refractive_index: float | None = None,
    sample_refractive_index: float | None = None,
    microscope_type: str | None = None,
    emission_wavelengths: list[float] | None = None,
    excitation_wavelengths: list[float] | None = None,
):
    """Run multi-method × multi-iteration benchmark for a single image."""
    del save_psf
    import gc
    import torch
    import tifffile

    meta = data["metadata"]
    images = data["images"]
    benchmark_rows: list[dict] = []
    meta_overrides: dict = {}
    work_path = img_path
    dataset_id = _stem(img_path.name)

    _cli_set: set[str] = set()
    if na is not None:
        _cli_set.add("na")
    if refractive_index is not None:
        _cli_set.add("refractive_index")
    if microscope_type is not None:
        _cli_set.add("microscope_type")
    if emission_wavelengths is not None:
        _cli_set.add("emission_wavelength")
    if excitation_wavelengths is not None:
        _cli_set.add("excitation_wavelength")
    if sample_refractive_index is not None:
        _cli_set.add("sample_refractive_index")
        meta["sample_refractive_index"] = sample_refractive_index
    else:
        sample_refractive_index = _SAMPLE_RI_DEFAULT
        meta["sample_refractive_index"] = sample_refractive_index

    if na is not None:
        meta["na"] = na
    if refractive_index is not None:
        meta["refractive_index"] = refractive_index
    if microscope_type is not None:
        meta["microscope_type"] = microscope_type
    if emission_wavelengths is not None:
        for i, wl in enumerate(emission_wavelengths):
            if i < len(meta.get("channels", [])):
                meta["channels"][i]["emission_wavelength"] = wl
    if excitation_wavelengths is not None:
        for i, wl in enumerate(excitation_wavelengths):
            if i < len(meta.get("channels", [])):
                meta["channels"][i]["excitation_wavelength"] = wl
    meta["_cli_overrides"] = _cli_set

    if bench_crop:
        cropped = []
        crop_slices = []
        for img in images:
            if img.ndim == 3:
                z, h, w = img.shape
                nz, ny, nx = min(z, max_tile_z), min(h, max_tile_xy), min(w, max_tile_xy)
                z0, y0, x0 = (z - nz) // 2, (h - ny) // 2, (w - nx) // 2
                crop_slices.append((slice(z0, z0 + nz), slice(y0, y0 + ny), slice(x0, x0 + nx)))
                img = img[z0:z0 + nz, y0:y0 + ny, x0:x0 + nx]
            else:
                h, w = img.shape
                ny, nx = min(h, max_tile_xy), min(w, max_tile_xy)
                y0, x0 = (h - ny) // 2, (w - nx) // 2
                crop_slices.append((slice(y0, y0 + ny), slice(x0, x0 + nx)))
                img = img[y0:y0 + ny, x0:x0 + nx]
            cropped.append(img)

        print(f"  Benchmark crop: {images[0].shape} -> {cropped[0].shape}")

        stack = np.stack(cropped, axis=0)
        axes = "CZYX" if cropped[0].ndim == 3 else "CYX"
        px_x = meta.get("pixel_size_x")
        px_y = meta.get("pixel_size_y")
        px_z = meta.get("pixel_size_z")
        resolution = (1.0 / px_x, 1.0 / px_y) if px_x and px_y else None
        crop_path = Path(out_path) / f"_bench_crop_{stem}.ome.tiff"
        tifffile.imwrite(
            str(crop_path),
            stack,
            ome=True,
            photometric="minisblack",
            resolution=resolution,
            resolutionunit=1,
            metadata={
                "axes": axes,
                "PhysicalSizeX": px_x,
                "PhysicalSizeY": px_y,
                "PhysicalSizeZ": px_z,
                "PhysicalSizeXUnit": "µm",
                "PhysicalSizeYUnit": "µm",
                "PhysicalSizeZUnit": "µm",
                "Channel": {
                    k: v
                    for k, v in {
                        "Name": [ch.get("name", f"Ch{i}") for i, ch in enumerate(meta.get("channels", []))],
                        "EmissionWavelength": [ch.get("emission_wavelength") for ch in meta.get("channels", [])]
                        if all(ch.get("emission_wavelength") is not None for ch in meta.get("channels", [])) else None,
                        "ExcitationWavelength": [ch.get("excitation_wavelength") for ch in meta.get("channels", [])]
                        if all(ch.get("excitation_wavelength") is not None for ch in meta.get("channels", [])) else None,
                    }.items()
                    if v is not None
                },
            },
        )
        work_path = crop_path
        meta_overrides = {
            "na": na if na is not None else meta.get("na"),
            "refractive_index": refractive_index if refractive_index is not None else meta.get("refractive_index"),
            "sample_refractive_index": sample_refractive_index,
            "microscope_type": microscope_type,
            "pixel_size_xy": meta.get("pixel_size_x"),
            "pixel_size_z": meta.get("pixel_size_z"),
            "emission_wavelengths": emission_wavelengths if emission_wavelengths is not None else (
                [ch.get("emission_wavelength") for ch in meta.get("channels", [])] or None
            ),
            "excitation_wavelengths": excitation_wavelengths if excitation_wavelengths is not None else (
                [ch.get("excitation_wavelength") for ch in meta.get("channels", [])] or None
            ),
        }
    else:
        meta_overrides = {
            "na": na,
            "refractive_index": refractive_index,
            "sample_refractive_index": sample_refractive_index,
            "microscope_type": microscope_type,
            "emission_wavelengths": emission_wavelengths,
            "excitation_wavelengths": excitation_wavelengths,
        }

    available_methods = []
    skip_reasons = {}
    for method in bench_methods:
        ok, reason = _check_method_available(method)
        if ok:
            available_methods.append(method)
        else:
            skip_reasons[method] = reason
            print(f"  Skipping {method}: {reason}")

    print(f"\n  Benchmarking {len(available_methods)} available method(s) x {len(bench_iterations)} iteration count(s)")
    for method in bench_methods:
        for nit in bench_iterations:
            row = _blank_benchmark_row(
                dataset_id=dataset_id,
                image_filename=img_path.name,
                method=method,
                iterations=nit,
                crop_mode="center_crop" if bench_crop else "full_volume",
                tiling_mode=tiling,
                tile_limits=f"{max_tile_xy},{max_tile_z}",
            )
            benchmark_rows.append(row)

            if method not in available_methods:
                row["status"] = "skipped"
                row["status_reason"] = skip_reasons.get(method, "method_unavailable")
                continue

            print(f"\n  -- {method}, {nit} iterations --")
            try:
                if method.startswith("sdeconv_"):
                    dev_val = "cuda" if torch.cuda.is_available() else "cpu"
                    monitor = _MetricsMonitor()
                    monitor.start()
                    result = deconvolve_image(
                        work_path,
                        method=method,
                        niter=nit,
                        device=dev_val,
                        tiling=tiling,
                        max_tile_xy=max_tile_xy,
                        max_tile_z=max_tile_z,
                        **meta_overrides,
                    )
                    out_name = f"{stem}_{method}_{dev_val}_{nit}i.ome.tiff"
                    out_file = Path(out_path) / out_name
                    save_result(result, str(out_file), mip_only=True)
                    metrics = monitor.stop()
                    row["device"] = dev_val.upper()
                    gpu_delta = metrics["torch_gpu_delta_mb"]
                    print(
                        f"    {dev_val}: {metrics['time_s']:.1f}s"
                        f"  RAM d{_format_bytes(metrics['ram_delta_peak_mb'])}"
                        f"  Alloc d{_format_bytes(gpu_delta)}"
                        f" -> {out_name}"
                    )
                else:
                    monitor = _MetricsMonitor()
                    monitor.start()
                    result = deconvolve_image(
                        work_path,
                        method=method,
                        niter=nit,
                        tiling=tiling,
                        max_tile_xy=max_tile_xy,
                        max_tile_z=max_tile_z,
                        **meta_overrides,
                    )
                    out_name = f"{stem}_{method}_{nit}i.ome.tiff"
                    out_file = Path(out_path) / out_name
                    save_result(result, str(out_file), mip_only=True)
                    metrics = monitor.stop()
                    row["device"] = _method_device(method)
                    print(
                        f"    {metrics['time_s']:.1f}s"
                        f"  RAM d{_format_bytes(metrics['ram_delta_peak_mb'])}"
                        f"  Alloc d{_format_bytes(metrics['gpu_mem_delta_peak_mb'])}"
                        f" -> {out_name}"
                    )

                row["status"] = "ok"
                row["status_reason"] = ""
                for key in (
                    "time_s",
                    "cpu_percent_avg",
                    "cpu_percent_peak",
                    "ram_total_mb",
                    "ram_peak_mb",
                    "ram_percent",
                    "ram_avg_mb",
                    "ram_delta_peak_mb",
                    "gpu_util_avg",
                    "gpu_util_peak",
                    "gpu_total_mb",
                    "gpu_mem_peak_mb",
                    "gpu_mem_percent",
                    "gpu_mem_avg_mb",
                    "gpu_mem_delta_peak_mb",
                    "torch_gpu_peak_mb",
                    "torch_gpu_delta_mb",
                    "gpu_spill_mb",
                ):
                    row[key] = _safe_float(metrics.get(key, 0.0))
                row["result_shape"] = _shape_to_str(result["channels"])
                row.update(_quality_metrics(result["channels"]))
                del result

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

                if torch.cuda.is_available():
                    try:
                        import pynvml

                        pynvml.nvmlInit()
                        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        prev_used = None
                        for _ in range(10):
                            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                            if prev_used is not None and mem.used == prev_used:
                                break
                            prev_used = mem.used
                            time.sleep(0.5)
                    except Exception:
                        time.sleep(3)
                else:
                    time.sleep(2)

            except ValueError as exc:
                print(f"    SKIPPED: {exc}")
                row["status"] = "skipped"
                row["status_reason"] = str(exc)
            except Exception as exc:
                print(f"    ERROR: {exc}")
                import traceback

                traceback.print_exc()
                row["status"] = "error"
                row["status_reason"] = str(exc)

    successful_rows = [row for row in benchmark_rows if row.get("status") == "ok"]
    if successful_rows:
        hdr = (
            f"  {'Method':<35} {'Device':>6} {'Time':>7} {'CPU%':>6}"
            f" {'RAM tot':>8} {'RAM pk':>8} {'RAM%':>5} {'RAM Δ':>8}"
            f" {'GPU%':>6} {'VRAM tot':>9} {'VRAM pk':>8} {'VRAM%':>5} {'Alloc Δ':>8} {'Spill':>8}"
        )
        sep = f"  {'─' * len(hdr.strip())}"
        print(f"\n{sep}")
        print("  Benchmark metrics summary:")
        print(f"{sep}")
        print(hdr)
        print(f"{sep}")
        for row in sorted(successful_rows, key=lambda r: (str(r['method']), int(r['iterations']))):
            label = f"{row['method']}_{row['iterations']}i"
            gpu_delta = row["torch_gpu_delta_mb"] if str(row["method"]).startswith("sdeconv_") else row["gpu_mem_delta_peak_mb"]
            spill = _safe_float(row.get("gpu_spill_mb", 0.0))
            spill_str = _format_bytes(spill) if spill > 0 else "—"
            print(
                f"  {label:<35} {row.get('device', '?'):>6}"
                f" {_safe_float(row['time_s']):>6.1f}s"
                f" {_safe_float(row['cpu_percent_avg']):>5.0f}%"
                f" {_format_bytes(_safe_float(row['ram_total_mb'])):>8}"
                f" {_format_bytes(_safe_float(row['ram_peak_mb'])):>8}"
                f" {_safe_float(row['ram_percent']):>4.0f}%"
                f" {_format_bytes(_safe_float(row['ram_delta_peak_mb'])):>8}"
                f" {_safe_float(row['gpu_util_avg']):>5.0f}%"
                f" {_format_bytes(_safe_float(row['gpu_total_mb'])):>9}"
                f" {_format_bytes(_safe_float(row['gpu_mem_peak_mb'])):>8}"
                f" {_safe_float(row['gpu_mem_percent']):>4.0f}%"
                f" {_format_bytes(_safe_float(gpu_delta)):>8}"
                f" {spill_str:>8}"
            )
        print(f"{sep}")

    csv_path = Path(out_path) / f"benchmark_metrics_{stem}.csv"
    _write_metrics_csv(csv_path, benchmark_rows)

    provenance_path = Path(out_path) / f"benchmark_provenance_{stem}.json"
    _write_provenance_json(
        provenance_path,
        _collect_run_provenance(
            img_path=img_path,
            data=data,
            meta_overrides=meta_overrides,
            cli_overrides=_cli_set,
            bench_crop=bench_crop,
            tiling=tiling,
            max_tile_xy=max_tile_xy,
            max_tile_z=max_tile_z,
            benchmark_rows=benchmark_rows,
        ),
    )

    montage_metrics = {}
    for row in successful_rows:
        key = (
            f"{row['method']}_{str(row['device']).lower()}_{row['iterations']}i"
            if str(row["method"]).startswith("sdeconv_")
            else f"{row['method']}_{row['iterations']}i"
        )
        montage_metrics[key] = {
            "device": row.get("device", ""),
            "time_s": _safe_float(row.get("time_s", 0.0)),
            "cpu_percent_avg": _safe_float(row.get("cpu_percent_avg", 0.0)),
            "ram_total_mb": _safe_float(row.get("ram_total_mb", 0.0)),
            "ram_peak_mb": _safe_float(row.get("ram_peak_mb", 0.0)),
            "ram_percent": _safe_float(row.get("ram_percent", 0.0)),
            "ram_delta_peak_mb": _safe_float(row.get("ram_delta_peak_mb", 0.0)),
            "gpu_util_avg": _safe_float(row.get("gpu_util_avg", 0.0)),
            "gpu_total_mb": _safe_float(row.get("gpu_total_mb", 0.0)),
            "gpu_mem_peak_mb": _safe_float(row.get("gpu_mem_peak_mb", 0.0)),
            "gpu_mem_percent": _safe_float(row.get("gpu_mem_percent", 0.0)),
            "gpu_mem_delta_peak_mb": _safe_float(row.get("gpu_mem_delta_peak_mb", 0.0)),
            "torch_gpu_peak_mb": _safe_float(row.get("torch_gpu_peak_mb", 0.0)),
            "torch_gpu_delta_mb": _safe_float(row.get("torch_gpu_delta_mb", 0.0)),
            "gpu_spill_mb": _safe_float(row.get("gpu_spill_mb", 0.0)),
        }
    if bench_montage:
        _make_benchmark_montage(out_path, stem, available_methods, bench_iterations, montage_metrics, meta)
        _make_per_channel_montages(out_path, stem, available_methods, bench_iterations, montage_metrics, meta)


if __name__ == "__main__":
    main(sys.argv[1:])
