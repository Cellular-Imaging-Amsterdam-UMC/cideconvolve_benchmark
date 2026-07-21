from __future__ import annotations

import importlib
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_LOCK = threading.Lock()
_MODULE: Any | None = None
_ERROR: str | None = None
_ATTEMPTED = False


def _safe_fragment(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")


def _toolkit_version(cuda_home: Path) -> str | None:
    nvcc = cuda_home / "bin" / ("nvcc.exe" if os.name == "nt" else "nvcc")
    if not nvcc.is_file():
        return None
    try:
        output = subprocess.check_output(
            [str(nvcc), "--version"], text=True, stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"release\s+(\d+\.\d+)", output)
    return match.group(1) if match else None


def _cuda_home() -> Path:
    configured = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    configured_path = Path(configured) if configured else Path()
    try:
        import torch

        runtime = torch.version.cuda
    except Exception:
        runtime = None
    if configured and (runtime is None or _toolkit_version(configured_path) == runtime):
        return configured_path
    if runtime:
        candidates = [
            Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
            / "NVIDIA GPU Computing Toolkit" / "CUDA" / f"v{runtime}",
            Path(f"/usr/local/cuda-{runtime}"),
            Path("/usr/local/cuda"),
        ]
        for candidate in candidates:
            if _toolkit_version(candidate) == runtime:
                return candidate
    return configured_path


def _compatibility() -> tuple[bool, str, dict[str, Any]]:
    try:
        import torch
    except Exception as exc:
        return False, f"PyTorch is unavailable: {exc}", {}
    details = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if not torch.cuda.is_available():
        return False, "CUDA is unavailable in PyTorch", details
    capability = torch.cuda.get_device_capability(torch.cuda.current_device())
    details["capability"] = capability
    if capability < (8, 6):
        return False, f"GPU compute capability sm_{capability[0]}{capability[1]} is below sm_86", details
    cuda_home = _cuda_home()
    toolkit = _toolkit_version(cuda_home)
    details["cuda_home"] = str(cuda_home) if str(cuda_home) else None
    details["toolkit_cuda"] = toolkit
    if toolkit is not None and toolkit != torch.version.cuda:
        return False, (
            f"CUDA Toolkit {toolkit} does not match PyTorch CUDA {torch.version.cuda or 'none'}"
        ), details
    return True, "compatible", details


def backend_status() -> dict[str, Any]:
    compatible, reason, details = _compatibility()
    details.update({
        "compatible": compatible,
        "reason": _ERROR or reason,
        "loaded": _MODULE is not None,
        "attempted": _ATTEMPTED,
    })
    return details


def _load_prebuilt() -> Any | None:
    for name in ("core._optimized_cuda", "cideconvolve_optimized_cuda"):
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError as exc:
            if exc.name != name:
                raise
    return None


def _jit_load(*, verbose: bool) -> Any:
    import torch

    cuda_home = _cuda_home()
    toolkit = _toolkit_version(cuda_home)
    if toolkit is None:
        raise RuntimeError(
            "no prebuilt optimized backend was found and a matching CUDA Toolkit/NVCC is unavailable"
        )
    capability = ".".join(
        str(part) for part in torch.cuda.get_device_capability(torch.cuda.current_device())
    )
    architecture = os.environ.get("TORCH_CUDA_ARCH_LIST", capability)
    signature = _safe_fragment(
        f"py{sys.version_info.major}{sys.version_info.minor}_torch{torch.__version__}_"
        f"cuda{toolkit}_arch{architecture}"
    )
    root = Path(__file__).resolve().parent
    cache_root = Path(os.environ.get(
        "CIDECONVOLVE_EXTENSION_CACHE",
        Path.home() / ".cache" / "cideconvolve" / "torch_extensions",
    ))
    build = cache_root / signature
    build.mkdir(parents=True, exist_ok=True)
    env_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
    env_cuda_home = os.environ.get("CUDA_HOME")
    env_cuda_path = os.environ.get("CUDA_PATH")
    env_path = os.environ.get("PATH", "")
    os.environ["TORCH_CUDA_ARCH_LIST"] = architecture
    os.environ["CUDA_HOME"] = str(cuda_home)
    os.environ["CUDA_PATH"] = str(cuda_home)
    python_scripts = Path(sys.executable).parent / ("Scripts" if os.name == "nt" else "bin")
    os.environ["PATH"] = os.pathsep.join(
        [str(Path(sys.executable).parent), str(python_scripts), str(cuda_home / "bin"), env_path]
    )
    try:
        if os.name == "nt" and not shutil.which("cl"):
            _activate_msvc_environment()
        import torch.utils.cpp_extension as cpp_extension

        cpp_extension.CUDA_HOME = str(cuda_home)
        if os.name == "nt":
            cflags = ["/O2", "/Zc:preprocessor"]
            cuda_flags = ["-O3", "-Xcompiler", "/Zc:preprocessor"]
            ldflags = ["cufft.lib"]
        else:
            cflags = ["-O3"]
            cuda_flags = ["-O3"]
            ldflags = ["-lcufft"]
        return cpp_extension.load(
            name=f"cideconvolve_optimized_cuda_{signature}",
            sources=[
                str(root / "cuda" / "fused_ops.cpp"),
                str(root / "cuda" / "fused_ops.cu"),
                str(root / "cuda" / "direct_fft.cpp"),
            ],
            build_directory=str(build),
            extra_cflags=cflags,
            extra_cuda_cflags=cuda_flags,
            extra_ldflags=ldflags,
            verbose=verbose,
        )
    finally:
        if env_arch is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = env_arch
        if env_cuda_home is None:
            os.environ.pop("CUDA_HOME", None)
        else:
            os.environ["CUDA_HOME"] = env_cuda_home
        if env_cuda_path is None:
            os.environ.pop("CUDA_PATH", None)
        else:
            os.environ["CUDA_PATH"] = env_cuda_path
        os.environ["PATH"] = env_path


def _activate_msvc_environment() -> None:
    """Import a Visual Studio x64 developer environment into this process."""
    program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
    candidates = [
        program_files / "Microsoft Visual Studio" / "2022" / edition
        / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
        for edition in ("Community", "BuildTools", "Professional", "Enterprise")
    ]
    script = next((candidate for candidate in candidates if candidate.is_file()), None)
    if script is None:
        return
    output = subprocess.check_output(
        f'cmd.exe /d /s /c ""{script}" >nul && set"',
        text=True,
        errors="replace",
    )
    for line in output.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key:
            os.environ[key] = value


def load_optimized_extension(*, required: bool = False, verbose: bool = False) -> Any | None:
    """Load a prebuilt extension or build a cache-isolated local extension.

    ``required=False`` is the fail-safe Auto behavior. ``required=True`` raises
    an actionable error for the explicit Optimized CUDA selection.
    """
    global _MODULE, _ERROR, _ATTEMPTED
    with _LOCK:
        if _MODULE is not None:
            return _MODULE
        if _ATTEMPTED:
            if required:
                raise RuntimeError(_ERROR or "optimized CUDA backend is unavailable")
            return None
        _ATTEMPTED = True
        compatible, reason, _ = _compatibility()
        if not compatible:
            _ERROR = reason
        else:
            try:
                _MODULE = _load_prebuilt()
                if _MODULE is None:
                    _MODULE = _jit_load(verbose=verbose)
                _ERROR = None
                log.info("Loaded optimized CUDA backend: %s", _MODULE.__name__)
            except Exception as exc:
                _ERROR = f"optimized CUDA backend unavailable: {type(exc).__name__}: {exc}"
                log.warning(_ERROR)
        if _MODULE is None and required:
            raise RuntimeError(_ERROR or "optimized CUDA backend is unavailable")
        return _MODULE


def reset_backend_state() -> None:
    """Reset cached detection state (primarily for tests and environment changes)."""
    global _MODULE, _ERROR, _ATTEMPTED
    with _LOCK:
        _MODULE = None
        _ERROR = None
        _ATTEMPTED = False
