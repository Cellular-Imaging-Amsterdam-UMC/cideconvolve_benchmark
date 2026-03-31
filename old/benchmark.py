"""
Benchmark: Richardson-Lucy deconvolution only — U2OS widefield data.

Loads the U2OS_R3D_WF_DV widefield image from the data/ folder, generates
channel-specific PSFs from OME metadata, and runs Richardson-Lucy
deconvolution (2D single-slice, full 3D, and plane-by-plane).
Also generates MIPs from the Huygens deconvolution for comparison.

Usage:
    python benchmark.py [image_filename] [blocks]

    image_filename  Optional image file in data/ folder.
                    Defaults to DividingCellcrop.ome.tiff.
                    The Huygens file (*_decon*) and output prefix
                    are derived automatically from the filename.
    blocks          Block tiling mode (default: auto).
                    'auto': calculate minimum blocks so each tile
                    stays within XY and Z limits.
                    0 or 1: no tiling.
                    >1: explicit number of blocks.
"""

import gc
import logging
import sys
import time
from pathlib import Path

# Configure logging so we can follow progress
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import deconvolve FIRST — it handles the torch-before-numpy DLL load
# order required on Windows with mixed conda/pip environments.
from deconvolve import (
    MAX_TILE_XY,
    MAX_TILE_Z,
    METHODS,
    _channel_color,
    deconvolve,
    deconvolve_image,
    generate_psf,
    generate_psf_deconwolf,
    load_image,
    save_mip_png,
    save_result,
)

import numpy as np

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Defaults — overridden by command-line argument in __main__
IMAGE_FILE = "DividingCellcrop.ome.tiff"
HUYGENS_FILE = "DividingCellcrop_decon.ome.tiff"
PREFIX = "DividingCellcrop.ome"
ORIGINAL_DATA_DIR = DATA_DIR  # preserved even when benchmark mode redirects DATA_DIR


def _derive_names(image_file: str):
    """Derive HUYGENS_FILE and PREFIX from an image filename."""
    p = Path(image_file)
    suffixes = "".join(p.suffixes)          # e.g. ".ome.tiff"
    base = p.name[:-len(suffixes)] if suffixes else p.name  # e.g. "DividingCellcrop"
    huygens = base + "_decon" + suffixes    # e.g. "DividingCellcrop_decon.ome.tiff"
    prefix = p.stem                         # e.g. "DividingCellcrop.ome"
    return huygens, prefix
NITERS = [20, 40, 60]  # iteration counts to test 
TIMINGS = {}  # key: output filename stem → elapsed seconds

# --- Toggle individual algorithms on/off ---
RUN_SDECONV_CPU_RL = True
RUN_SDECONV_CUDA_RL = True
RUN_SDECONV_PLANE_BY_PLANE_RL = True
RUN_PYCUDADECON_RL = True
RUN_DECONWOLF_RL = False
RUN_DECONWOLF_SHB = False
RUN_DECONVLAB2_RL = False
RUN_DECONVLAB2_RLTV = False
RUN_REDLIONFISH_RL = True
RUN_SKIMAGE_RL = False
RUN_SKIIMAGE_CUCIM_RL = True

# XY block-tiling: split each image into N_BLOCKS tiles before deconvolution.
# "auto" (default) calculates the minimum block count so each tile stays
# within XY Z max.  Set to 0 or 1 to disable tiling, or >1 for
# an explicit block count.
N_BLOCKS = "auto"

# Benchmark mode: crop image to tile-size limits and skip saving OME-TIFFs.
BENCHMARK_MODE = False

# Save PSF arrays as TIFF files (enabled by --psf).
SAVE_PSF = False

# Metadata overrides passed to deconvolve_image() so the cropped benchmark
# TIFF inherits the original pixel sizes, NA, wavelengths, etc.
_META_OVERRIDES: dict = {}


def benchmark_metadata():
    """Step 1: Load the image and inspect metadata."""
    print("\n" + "=" * 70)
    print("STEP 1: Load image and extract metadata")
    print("=" * 70)

    image_path = DATA_DIR / IMAGE_FILE
    data = load_image(image_path)
    meta = data["metadata"]
    images = data["images"]

    print(f"\nNumber of channels: {len(images)}")
    for i, img in enumerate(images):
        print(f"  Channel {i}: shape={img.shape}, dtype={img.dtype}, "
              f"min={img.min():.2f}, max={img.max():.2f}")

    print(f"\nMicroscope type: {meta['microscope_type']}")
    print(f"Numerical Aperture: {meta['na']}")
    print(f"Immersion RI: {meta['refractive_index']}")
    print(f"Pixel size XY: {meta['pixel_size_x']} µm")
    print(f"Pixel size Z: {meta['pixel_size_z']} µm")
    print(f"Image size: {meta.get('size_x')}x{meta.get('size_y')}x{meta.get('size_z')}")

    for i, ch in enumerate(meta["channels"]):
        print(f"\nChannel {i}:")
        print(f"  Excitation: {ch.get('excitation_wavelength')} nm")
        print(f"  Emission: {ch.get('emission_wavelength')} nm")
        print(f"  Pinhole: {ch.get('pinhole_size')} µm")
        print(f"  Acquisition: {ch.get('acquisition_mode')}")

    return data


def benchmark_psf(metadata):
    """Step 2: Generate PSFs for channel 0 using all available generators."""
    print("\n" + "=" * 70)
    print("STEP 2: Generate PSFs from metadata (ch0, all generators)")
    print("=" * 70)

    import shutil
    from deconvolve import _DW_BW_EXE

    # Collect (label, psf) pairs for ch0
    labelled_psfs: list[tuple[str, np.ndarray]] = []

    # 1. psf_generator (vectorial/scalar)
    psf_gen = generate_psf(metadata, channel_idx=0)
    labelled_psfs.append(("psf_generator", psf_gen))
    print(f"\n  psf_generator: shape={psf_gen.shape}, sum={psf_gen.sum():.6f}")

    # 2. deconwolf dw_bw (Born-Wolf)
    if _DW_BW_EXE:
        try:
            psf_dw = generate_psf_deconwolf(metadata, channel_idx=0)
            labelled_psfs.append(("deconwolf (dw_bw)", psf_dw))
            print(f"  deconwolf dw_bw: shape={psf_dw.shape}, sum={psf_dw.sum():.6f}")
        except Exception as exc:
            print(f"  deconwolf dw_bw: FAILED — {exc}")
    else:
        print("  deconwolf dw_bw: not available (skipped)")

    # Save montage images
    _save_psf_montage(labelled_psfs, metadata)

    return [psf for _, psf in labelled_psfs]


def _save_psf_montage(labelled_psfs, metadata):
    """Save XY and XZ grayscale montage of all PSFs with algorithm labels."""
    from PIL import Image, ImageDraw, ImageFont

    scale = 4  # upscale small PSFs for visibility

    # Try to load a font
    font = None
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 14)
        except OSError:
            font = ImageFont.load_default()

    for view in ("XY", "XZ"):
        panels = []
        labels = []
        for label, psf in labelled_psfs:
            if psf.ndim == 3:
                if view == "XY":
                    mip = psf.max(axis=0)  # max over Z → (Y, X)
                else:  # XZ
                    mip = psf.max(axis=1)  # max over Y → (Z, X)
            else:
                mip = psf  # 2D, same for both views

            # Normalise to 0-1
            lo, hi = mip.min(), mip.max()
            if hi > lo:
                mip = (mip - lo) / (hi - lo)
            else:
                mip = np.zeros_like(mip)

            # Apply gamma for better visibility of weak lobes
            mip = np.power(mip, 0.4)

            # Convert to grayscale uint8
            grey = np.clip(mip * 255, 0, 255).astype(np.uint8)

            # Upscale
            h, w = grey.shape
            img = Image.fromarray(grey, mode="L")
            img = img.resize((w * scale, h * scale), Image.NEAREST)
            panels.append(img)
            labels.append(label)

        if not panels:
            continue

        # Montage: all generators side by side with labels
        label_height = 24
        padding = 2
        max_w = max(p.size[0] for p in panels)
        max_h = max(p.size[1] for p in panels)
        cell_w = max_w + 2 * padding
        cell_h = max_h + label_height + 2 * padding

        n = len(panels)
        montage = Image.new("L", (n * cell_w, cell_h), color=0)
        draw = ImageDraw.Draw(montage)

        for i, panel in enumerate(panels):
            x0 = i * cell_w + padding + (max_w - panel.size[0]) // 2
            y0 = padding + (max_h - panel.size[1]) // 2
            montage.paste(panel, (x0, y0))

            bbox = draw.textbbox((0, 0), labels[i], font=font)
            tw = bbox[2] - bbox[0]
            tx = i * cell_w + padding + (max_w - tw) // 2
            ty = padding + max_h + 2
            draw.text((tx, ty), labels[i], fill=255, font=font)

        out = OUTPUT_DIR / f"psf_{PREFIX}_{view}.png"
        if not BENCHMARK_MODE or SAVE_PSF:
            montage.save(str(out))
            print(f"  Saved PSF {view} montage: {out}  ({montage.size[0]}x{montage.size[1]})")


def benchmark_deconvolution_2d(data):
    """Step 3: Quick 2D Richardson-Lucy deconvolution of a single Z-slice."""
    print("\n" + "=" * 70)
    print("STEP 3: 2D Richardson-Lucy deconvolution of single Z-slice")
    print("=" * 70)

    meta = data["metadata"]
    img_3d = data["images"][0]  # Channel 0

    # Take the middle Z-slice
    mid_z = img_3d.shape[0] // 2
    img_2d = img_3d[mid_z]
    print(f"\nUsing Z-slice {mid_z}, shape={img_2d.shape}")

    # Generate 2D PSF
    meta_2d = dict(meta)
    meta_2d["size_z"] = 1  # Force 2D PSF
    psf_2d = generate_psf(meta_2d, channel_idx=0)
    print(f"2D PSF shape: {psf_2d.shape}")

    info = METHODS["sdeconv_rl"]
    print(f"\n  sdeconv_rl: {info['description']}")
    print(f"  Memory factor: ~{info['memory_factor']}x image size")

    t0 = time.perf_counter()
    result = deconvolve(img_2d, psf_2d, method="sdeconv_rl", niter=30)
    elapsed = time.perf_counter() - t0
    print(f"  Result: shape={result.shape}, min={result.min():.2f}, max={result.max():.2f}")
    print(f"  Elapsed time: {elapsed:.1f} s")

    if not BENCHMARK_MODE:
        import tifffile
        out_path = OUTPUT_DIR / f"{PREFIX}_slice_z{mid_z}_ch0_richardson_lucy.tiff"
        tifffile.imwrite(str(out_path), result)
        print(f"  Saved to: {out_path}")


def benchmark_full_3d():
    """Step 4: Full 3D RL — compare all backends × three iteration counts."""
    print("\n" + "=" * 70)
    print("STEP 4: Full 3D deconvolution — multiple backends × iteration counts")
    print("=" * 70)

    image_path = DATA_DIR / IMAGE_FILE

    import shutil
    import torch
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("  CUDA not available — GPU backends will be skipped")

    from deconvolve import _DW_EXE, _DECONVLAB2_JAR, _IJ_JAR
    dw_available = bool(_DW_EXE)
    print(f"  deconwolf: {'available' if dw_available else 'NOT found (skipping)'}")
    dl2_available = _DECONVLAB2_JAR.exists() and _IJ_JAR.exists() and bool(shutil.which("java"))
    print(f"  DeconvolutionLab2: {'available' if dl2_available else 'NOT found (skipping)'}")

    # --- (a) sdeconv on CPU ---
    if not RUN_SDECONV_CPU_RL:
        print("\n--- (a) sdeconv CPU — SKIPPED (disabled) ---")
    for nit in NITERS if RUN_SDECONV_CPU_RL else []:
        print(f"\n--- (a) sdeconv Richardson-Lucy on CPU, {nit} iterations ---")
        t0 = time.perf_counter()
        result_cpu = deconvolve_image(
            image_path, method="sdeconv_rl", niter=nit, device="cpu",
            n_blocks=N_BLOCKS, **_META_OVERRIDES,
        )
        elapsed = time.perf_counter() - t0
        for i, ch in enumerate(result_cpu["channels"]):
            print(f"  Channel {i}: shape={ch.shape}, min={ch.min():.2f}, max={ch.max():.2f}")
        print(f"  Elapsed time: {elapsed:.1f} s")
        out = OUTPUT_DIR / f"{PREFIX}_RL_sdeconv_cpu_{nit}i.ome.tiff"
        save_result(result_cpu, out, mip_only=BENCHMARK_MODE)
        TIMINGS[out.stem] = elapsed
        print(f"  Saved to: {out}")

    # --- (b) sdeconv on CUDA ---
    if cuda_available and RUN_SDECONV_CUDA_RL:
        for nit in NITERS:
            print(f"\n--- (b) sdeconv Richardson-Lucy on CUDA, {nit} iterations ---")
            t0 = time.perf_counter()
            result_gpu = deconvolve_image(
                image_path, method="sdeconv_rl", niter=nit, device="cuda",
                n_blocks=N_BLOCKS, **_META_OVERRIDES,
            )
            elapsed = time.perf_counter() - t0
            for i, ch in enumerate(result_gpu["channels"]):
                print(f"  Channel {i}: shape={ch.shape}, min={ch.min():.2f}, max={ch.max():.2f}")
            print(f"  Elapsed time: {elapsed:.1f} s")
            out = OUTPUT_DIR / f"{PREFIX}_RL_sdeconv_cuda_{nit}i.ome.tiff"
            save_result(result_gpu, out, mip_only=BENCHMARK_MODE)
            TIMINGS[out.stem] = elapsed
            print(f"  Saved to: {out}")
        del result_gpu
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print("\n--- (b) sdeconv CUDA — SKIPPED (no GPU or disabled) ---")

    # --- (c) sdeconv plane-by-plane ---
    if RUN_SDECONV_PLANE_BY_PLANE_RL:
        for nit in NITERS:
            print(f"\n--- (c) sdeconv plane-by-plane, {nit} iterations ---")
            t0 = time.perf_counter()
            result_pbp = deconvolve_image(
                image_path,
                method="sdeconv_rl",
                niter=nit,
                plane_by_plane=True,
                n_blocks=N_BLOCKS, **_META_OVERRIDES,
            )
            elapsed = time.perf_counter() - t0
            for i, ch in enumerate(result_pbp["channels"]):
                print(f"  Channel {i}: shape={ch.shape}, min={ch.min():.2f}, max={ch.max():.2f}")
            print(f"  Elapsed time: {elapsed:.1f} s")
            out_path = OUTPUT_DIR / f"{PREFIX}_plane_by_plane_RL_{nit}i.ome.tiff"
            save_result(result_pbp, out_path, mip_only=BENCHMARK_MODE)
            TIMINGS[out_path.stem] = elapsed
            print(f"  Saved to: {out_path}")
    else:
        print("\n--- (c) sdeconv plane-by-plane — SKIPPED (disabled) ---")

    # --- (d) pycudadecon ---
    if cuda_available and RUN_PYCUDADECON_RL:
        for nit in NITERS:
            print(f"\n--- (d) pycudadecon Richardson-Lucy (CUDA), {nit} iterations ---")
            try:
                t0 = time.perf_counter()
                result_pcd = deconvolve_image(
                    image_path, method="pycudadecon_rl_cuda", niter=nit,
                    n_blocks=N_BLOCKS, **_META_OVERRIDES,
                )
                elapsed = time.perf_counter() - t0
                for i, ch in enumerate(result_pcd["channels"]):
                    print(f"  Channel {i}: shape={ch.shape}, min={ch.min():.2f}, max={ch.max():.2f}")
                print(f"  Elapsed time: {elapsed:.1f} s")
                out = OUTPUT_DIR / f"{PREFIX}_RL_pycudadecon_{nit}i.ome.tiff"
                save_result(result_pcd, out, mip_only=BENCHMARK_MODE)
                TIMINGS[out.stem] = elapsed
                print(f"  Saved to: {out}")
            except ImportError as exc:
                print(f"  SKIPPED — {exc}")
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print("\n--- (d) pycudadecon — SKIPPED (no GPU or disabled) ---")

    # --- (e) deconwolf RL ---
    if dw_available and RUN_DECONWOLF_RL:
        for nit in NITERS:
            print(f"\n--- (e) deconwolf RL, {nit} iterations ---")
            t0 = time.perf_counter()
            result_dw = deconvolve_image(
                image_path, method="deconwolf_rl", niter=nit,
                n_blocks=N_BLOCKS, **_META_OVERRIDES,
            )
            elapsed = time.perf_counter() - t0
            for i, ch in enumerate(result_dw["channels"]):
                print(f"  Channel {i}: shape={ch.shape}, min={ch.min():.2f}, max={ch.max():.2f}")
            print(f"  Elapsed time: {elapsed:.1f} s")
            out = OUTPUT_DIR / f"{PREFIX}_RL_deconwolf_{nit}i.ome.tiff"
            save_result(result_dw, out, mip_only=BENCHMARK_MODE)
            TIMINGS[out.stem] = elapsed
            print(f"  Saved to: {out}")
    else:
        print("\n--- (e) deconwolf RL — SKIPPED (dw not found or disabled) ---")

    # --- (f) deconwolf SHB ---
    if dw_available and RUN_DECONWOLF_SHB:
        for nit in NITERS:
            print(f"\n--- (f) deconwolf SHB, {nit} iterations ---")
            t0 = time.perf_counter()
            result_dw_shb = deconvolve_image(
                image_path, method="deconwolf_shb", niter=nit,
                n_blocks=N_BLOCKS, **_META_OVERRIDES,
            )
            elapsed = time.perf_counter() - t0
            for i, ch in enumerate(result_dw_shb["channels"]):
                print(f"  Channel {i}: shape={ch.shape}, min={ch.min():.2f}, max={ch.max():.2f}")
            print(f"  Elapsed time: {elapsed:.1f} s")
            out = OUTPUT_DIR / f"{PREFIX}_SHB_deconwolf_{nit}i.ome.tiff"
            save_result(result_dw_shb, out, mip_only=BENCHMARK_MODE)
            TIMINGS[out.stem] = elapsed
            print(f"  Saved to: {out}")
    else:
        print("\n--- (f) deconwolf SHB — SKIPPED (dw not found or disabled) ---")

    # --- (g) DeconvolutionLab2 RL ---
    if dl2_available and RUN_DECONVLAB2_RL:
        for nit in NITERS:
            print(f"\n--- (g) DeconvolutionLab2 RL, {nit} iterations ---")
            t0 = time.perf_counter()
            result_dl2 = deconvolve_image(
                image_path, method="deconvlab2_rl", niter=nit,
                n_blocks=N_BLOCKS, **_META_OVERRIDES,
            )
            elapsed = time.perf_counter() - t0
            for i, ch in enumerate(result_dl2["channels"]):
                print(f"  Channel {i}: shape={ch.shape}, min={ch.min():.2f}, max={ch.max():.2f}")
            print(f"  Elapsed time: {elapsed:.1f} s")
            out = OUTPUT_DIR / f"{PREFIX}_RL_deconvlab2_{nit}i.ome.tiff"
            save_result(result_dl2, out, mip_only=BENCHMARK_MODE)
            TIMINGS[out.stem] = elapsed
            print(f"  Saved to: {out}")
    else:
        print("\n--- (g) DeconvolutionLab2 RL — SKIPPED (JAR/java not found or disabled) ---")

    # --- (h) DeconvolutionLab2 RLTV ---
    if dl2_available and RUN_DECONVLAB2_RLTV:
        for nit in NITERS:
            print(f"\n--- (h) DeconvolutionLab2 RLTV, {nit} iterations ---")
            t0 = time.perf_counter()
            result_dl2tv = deconvolve_image(
                image_path, method="deconvlab2_rltv", niter=nit, tv_lambda=1e-4,
                n_blocks=N_BLOCKS, **_META_OVERRIDES,
            )
            elapsed = time.perf_counter() - t0
            for i, ch in enumerate(result_dl2tv["channels"]):
                print(f"  Channel {i}: shape={ch.shape}, min={ch.min():.2f}, max={ch.max():.2f}")
            print(f"  Elapsed time: {elapsed:.1f} s")
            out = OUTPUT_DIR / f"{PREFIX}_RLTV_deconvlab2_{nit}i.ome.tiff"
            save_result(result_dl2tv, out, mip_only=BENCHMARK_MODE)
            TIMINGS[out.stem] = elapsed
            print(f"  Saved to: {out}")
    else:
        print("\n--- (h) DeconvolutionLab2 RLTV — SKIPPED (JAR/java not found or disabled) ---")

    # --- (i) RedLionfish RL ---
    if RUN_REDLIONFISH_RL:
        try:
            import RedLionfishDeconv
            rlf_available = True
        except ImportError:
            rlf_available = False
        if rlf_available:
            for nit in NITERS:
                print(f"\n--- (i) RedLionfish RL, {nit} iterations ---")
                t0 = time.perf_counter()
                result_rlf = deconvolve_image(
                    image_path, method="redlionfish_rl", niter=nit,
                    n_blocks=N_BLOCKS, **_META_OVERRIDES,
                )
                elapsed = time.perf_counter() - t0
                for i, ch in enumerate(result_rlf["channels"]):
                    print(f"  Channel {i}: shape={ch.shape}, min={ch.min():.2f}, max={ch.max():.2f}")
                print(f"  Elapsed time: {elapsed:.1f} s")
                out = OUTPUT_DIR / f"{PREFIX}_RL_redlionfish_{nit}i.ome.tiff"
                save_result(result_rlf, out, mip_only=BENCHMARK_MODE)
                TIMINGS[out.stem] = elapsed
                print(f"  Saved to: {out}")
            gc.collect()
            torch.cuda.empty_cache()
        else:
            print("\n--- (i) RedLionfish RL — SKIPPED (not installed) ---")
    else:
        print("\n--- (i) RedLionfish RL — SKIPPED (disabled) ---")

    # --- (j) scikit-image RL ---
    if RUN_SKIMAGE_RL:
        for nit in NITERS:
            print(f"\n--- (j) scikit-image RL (CPU), {nit} iterations ---")
            t0 = time.perf_counter()
            result_ski = deconvolve_image(
                image_path, method="skimage_rl", niter=nit,
                n_blocks=N_BLOCKS, **_META_OVERRIDES,
            )
            elapsed = time.perf_counter() - t0
            for i, ch in enumerate(result_ski["channels"]):
                print(f"  Channel {i}: shape={ch.shape}, min={ch.min():.2f}, max={ch.max():.2f}")
            print(f"  Elapsed time: {elapsed:.1f} s")
            out = OUTPUT_DIR / f"{PREFIX}_RL_skimage_{nit}i.ome.tiff"
            save_result(result_ski, out, mip_only=BENCHMARK_MODE)
            TIMINGS[out.stem] = elapsed
            print(f"  Saved to: {out}")
    else:
        print("\n--- (j) scikit-image RL — SKIPPED (disabled) ---")

    # --- (k) cuCIM RL ---
    if cuda_available and RUN_SKIIMAGE_CUCIM_RL:
        try:
            import cupy  # noqa: F401
            import cucim  # noqa: F401
            cucim_available = True
        except ImportError:
            cucim_available = False
        if cucim_available:
            for nit in NITERS:
                print(f"\n--- (k) cuCIM RL (CUDA), {nit} iterations ---")
                t0 = time.perf_counter()
                result_cucim = deconvolve_image(
                    image_path, method="skimage_cucim_rl", niter=nit,
                    n_blocks=N_BLOCKS, **_META_OVERRIDES,
                )
                elapsed = time.perf_counter() - t0
                for i, ch in enumerate(result_cucim["channels"]):
                    print(f"  Channel {i}: shape={ch.shape}, min={ch.min():.2f}, max={ch.max():.2f}")
                print(f"  Elapsed time: {elapsed:.1f} s")
                out = OUTPUT_DIR / f"{PREFIX}_RL_cucim_{nit}i.ome.tiff"
                save_result(result_cucim, out, mip_only=BENCHMARK_MODE)
                TIMINGS[out.stem] = elapsed
                print(f"  Saved to: {out}")
            gc.collect()
            torch.cuda.empty_cache()
        else:
            print("\n--- (k) cuCIM RL — SKIPPED (cupy/cucim not installed) ---")
    else:
        print("\n--- (k) cuCIM RL — SKIPPED (no GPU or disabled) ---")


def benchmark_huygens_mip():
    """Step 5: Generate MIP from Huygens (SVI) deconvolution for comparison."""
    print("\n" + "=" * 70)
    print("STEP 5: MIP from Huygens deconvolution (for comparison)")
    print("=" * 70)

    huygens_path = ORIGINAL_DATA_DIR / HUYGENS_FILE
    if not huygens_path.exists():
        print(f"\n  Huygens file not found: {huygens_path}")
        print("  Skipping Huygens MIP generation.")
        return
    data = load_image(huygens_path)
    meta = data["metadata"]
    images = data["images"]

    import tifffile

    # Build multichannel MIP: stack all channels then take max over Z
    mip_channels = []
    for i, img in enumerate(images):
        print(f"  Channel {i}: shape={img.shape}, dtype={img.dtype}")
        mip_ch = img.max(axis=0) if img.ndim == 3 else img
        mip_channels.append(mip_ch)

    mip = np.stack(mip_channels, axis=0)  # (C, Y, X)
    print(f"  MIP shape: {mip.shape}")

    # In benchmark mode, centre-crop the Huygens MIP to match the
    # benchmark crop dimensions so it fits in the montage.
    if BENCHMARK_MODE:
        _, h, w = mip.shape
        ny, nx = min(h, MAX_TILE_XY), min(w, MAX_TILE_XY)
        y0, x0 = (h - ny) // 2, (w - nx) // 2
        mip = mip[:, y0:y0+ny, x0:x0+nx]
        print(f"  Cropped Huygens MIP to {mip.shape}")

    mip_tiff = OUTPUT_DIR / f"mip_Huygens_{PREFIX}.tiff"
    tifffile.imwrite(str(mip_tiff), mip)
    print(f"  Saved MIP TIFF: {mip_tiff}")

    mip_png = OUTPUT_DIR / f"mip_Huygens_{PREFIX}.png"
    save_mip_png(mip, mip_png, meta)
    print(f"  Saved MIP PNG:  {mip_png}")


def _timed_label(name, nit, stem):
    """Algorithm name on line 1, iteration count + timing on line 2."""
    t = TIMINGS.get(stem)
    if t is not None:
        return f"{name}\n{nit} iterations in {t:.1f} sec"
    return f"{name}\n{nit} iterations"


def _make_metadata_panel(meta, width, height, font):
    """Create an image panel showing PSF-relevant metadata."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (width, height), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)

    lines = [
        f"Microscope: {meta.get('microscope_type', '?')}",
        f"NA: {meta.get('na', '?')}",
        f"RI: {meta.get('refractive_index', '?')}",
        f"Pixel XY: {meta.get('pixel_size_x', '?')} \u00b5m",
        f"Pixel Z:  {meta.get('pixel_size_z', '?')} \u00b5m",
        f"Size: {meta.get('size_x', '?')}\u00d7{meta.get('size_y', '?')}\u00d7{meta.get('size_z', '?')}",
        "",
    ]
    for i, ch in enumerate(meta.get("channels", [])):
        em = ch.get("emission_wavelength") or "?"
        ex = ch.get("excitation_wavelength") or "?"
        lines.append(f"Ch{i}: Ex {ex} / Em {em} nm")

    text = "\n".join(lines)
    draw.text((8, 8), text, fill=(255, 255, 255), font=font)
    return img


def benchmark_montage():
    """Step 6: Create a montage of all MIP PNG images with algorithm labels."""
    print("\n" + "=" * 70)
    print("STEP 6: Create montage of all MIP images")
    print("=" * 70)

    from PIL import Image, ImageDraw, ImageFont

    # Rows of (png_path, label): row 0 = references, rows 1-3 = by iteration count
    rows = [
        # Row 0: Source and Huygens reference
        [
            (OUTPUT_DIR / "mip_source.ome.png", "Source"),
            (OUTPUT_DIR / f"mip_Huygens_{PREFIX}.png", "Huygens"),
        ],
    ]
    # Rows by iteration count, all backends
    for nit in NITERS:
        rows.append([
            (OUTPUT_DIR / f"mip_{PREFIX}_RL_sdeconv_cpu_{nit}i.ome.png", _timed_label("sdeconv_cpu_rl", nit, f"{PREFIX}_RL_sdeconv_cpu_{nit}i.ome")),
            (OUTPUT_DIR / f"mip_{PREFIX}_RL_sdeconv_cuda_{nit}i.ome.png", _timed_label("sdeconv_cuda_rl", nit, f"{PREFIX}_RL_sdeconv_cuda_{nit}i.ome")),
            (OUTPUT_DIR / f"mip_{PREFIX}_plane_by_plane_RL_{nit}i.ome.png", _timed_label("sdeconv_plane_by_plane_rl", nit, f"{PREFIX}_plane_by_plane_RL_{nit}i.ome")),
            (OUTPUT_DIR / f"mip_{PREFIX}_RL_pycudadecon_{nit}i.ome.png", _timed_label("pycudadecon_rl", nit, f"{PREFIX}_RL_pycudadecon_{nit}i.ome")),
            (OUTPUT_DIR / f"mip_{PREFIX}_RL_deconwolf_{nit}i.ome.png", _timed_label("deconwolf_rl", nit, f"{PREFIX}_RL_deconwolf_{nit}i.ome")),
            (OUTPUT_DIR / f"mip_{PREFIX}_SHB_deconwolf_{nit}i.ome.png", _timed_label("deconwolf_shb", nit, f"{PREFIX}_SHB_deconwolf_{nit}i.ome")),
            (OUTPUT_DIR / f"mip_{PREFIX}_RL_deconvlab2_{nit}i.ome.png", _timed_label("deconvlab2_rl", nit, f"{PREFIX}_RL_deconvlab2_{nit}i.ome")),
            (OUTPUT_DIR / f"mip_{PREFIX}_RLTV_deconvlab2_{nit}i.ome.png", _timed_label("deconvlab2_rltv", nit, f"{PREFIX}_RLTV_deconvlab2_{nit}i.ome")),
            (OUTPUT_DIR / f"mip_{PREFIX}_RL_redlionfish_{nit}i.ome.png", _timed_label("redlionfish_rl", nit, f"{PREFIX}_RL_redlionfish_{nit}i.ome")),
            (OUTPUT_DIR / f"mip_{PREFIX}_RL_skimage_{nit}i.ome.png", _timed_label("skimage_rl", nit, f"{PREFIX}_RL_skimage_{nit}i.ome")),
            (OUTPUT_DIR / f"mip_{PREFIX}_RL_cucim_{nit}i.ome.png", _timed_label("skiimage_cucim_rl", nit, f"{PREFIX}_RL_cucim_{nit}i.ome")),
        ])

    # Filter each row to existing files and load images
    loaded_rows = []
    total = 0
    for row_entries in rows:
        row_images = []
        for path, label in row_entries:
            if path.exists():
                img = Image.open(path).convert("RGB")
                row_images.append((img, label))
                print(f"    {label}: {img.size[0]}x{img.size[1]}")
                total += 1
        if row_images:
            loaded_rows.append(row_images)

    if total == 0:
        print("  No MIP PNG files found — skipping montage.")
        return

    print(f"  Found {total} MIP images")

    # Label bar height (extra space for optional timing line)
    label_height = 60
    padding = 4

    # All panels same size (use max dimensions across all images)
    all_imgs = [img for row in loaded_rows for img, _ in row]
    max_w = max(img.size[0] for img in all_imgs)
    max_h = max(img.size[1] for img in all_imgs)

    # Try to load a decent font, fall back to default
    font = None
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 18)
        except OSError:
            font = ImageFont.load_default()

    # Insert PSF metadata panel next to Source in row 0
    _meta_data = load_image(DATA_DIR / IMAGE_FILE)
    meta_panel = _make_metadata_panel(_meta_data["metadata"], max_w, max_h, font)
    if loaded_rows:
        loaded_rows[0].insert(1, (meta_panel, "PSF Parameters"))

    # Number of columns = max row width (after metadata panel insertion)
    n_cols = max(len(row) for row in loaded_rows)
    n_rows = len(loaded_rows)

    cell_w = max_w + 2 * padding
    cell_h = max_h + label_height + 2 * padding

    montage_w = n_cols * cell_w
    montage_h = n_rows * cell_h
    montage = Image.new("RGB", (montage_w, montage_h), color=(0, 0, 0))
    draw = ImageDraw.Draw(montage)

    for row_idx, row_images in enumerate(loaded_rows):
        for col_idx, (img, label) in enumerate(row_images):
            x0 = col_idx * cell_w + padding
            y0 = row_idx * cell_h + padding

            # Centre the image in the cell
            x_off = (max_w - img.size[0]) // 2
            y_off = (max_h - img.size[1]) // 2
            montage.paste(img, (x0 + x_off, y0 + y_off))

            # Draw label centred below the image
            bbox = draw.textbbox((0, 0), label, font=font)
            tw = bbox[2] - bbox[0]
            tx = x0 + (max_w - tw) // 2
            ty = y0 + max_h + padding
            draw.text((tx, ty), label, fill=(255, 255, 255), font=font)

    _clean = PREFIX.replace(".ome", "")
    out_path = OUTPUT_DIR / f"decon_benchmark_{_clean}.png"
    montage.save(str(out_path))
    print(f"  Saved montage: {out_path}  ({montage_w}x{montage_h})")

    # --- Per-channel montages ---
    _make_per_channel_montages(font)


def _make_per_channel_montages(font):
    """Create one montage per channel, comparing algorithms side by side (greyscale)."""
    from PIL import Image, ImageDraw, ImageFont
    import tifffile

    # Load metadata for channel info
    data = load_image(DATA_DIR / IMAGE_FILE)
    meta = data["metadata"]
    n_ch = meta["n_channels"]

    # MIP TIFF files to compare — organised by row
    # Row 0: references, Rows 1-3: by iteration count
    mip_rows = [
        [
            ("Source", OUTPUT_DIR / "mip_source.ome.tiff"),
            ("Huygens", OUTPUT_DIR / f"mip_Huygens_{PREFIX}.tiff"),
        ],
    ]
    for nit in NITERS:
        mip_rows.append([
            (_timed_label("sdeconv_cpu_rl", nit, f"{PREFIX}_RL_sdeconv_cpu_{nit}i.ome"), OUTPUT_DIR / f"mip_{PREFIX}_RL_sdeconv_cpu_{nit}i.ome.tiff"),
            (_timed_label("sdeconv_cuda_rl", nit, f"{PREFIX}_RL_sdeconv_cuda_{nit}i.ome"), OUTPUT_DIR / f"mip_{PREFIX}_RL_sdeconv_cuda_{nit}i.ome.tiff"),
            (_timed_label("sdeconv_plane_by_plane_rl", nit, f"{PREFIX}_plane_by_plane_RL_{nit}i.ome"), OUTPUT_DIR / f"mip_{PREFIX}_plane_by_plane_RL_{nit}i.ome.tiff"),
            (_timed_label("pycudadecon_rl", nit, f"{PREFIX}_RL_pycudadecon_{nit}i.ome"), OUTPUT_DIR / f"mip_{PREFIX}_RL_pycudadecon_{nit}i.ome.tiff"),
            (_timed_label("deconwolf_rl", nit, f"{PREFIX}_RL_deconwolf_{nit}i.ome"), OUTPUT_DIR / f"mip_{PREFIX}_RL_deconwolf_{nit}i.ome.tiff"),
            (_timed_label("deconwolf_shb", nit, f"{PREFIX}_SHB_deconwolf_{nit}i.ome"), OUTPUT_DIR / f"mip_{PREFIX}_SHB_deconwolf_{nit}i.ome.tiff"),
            (_timed_label("deconvlab2_rl", nit, f"{PREFIX}_RL_deconvlab2_{nit}i.ome"), OUTPUT_DIR / f"mip_{PREFIX}_RL_deconvlab2_{nit}i.ome.tiff"),
            (_timed_label("deconvlab2_rltv", nit, f"{PREFIX}_RLTV_deconvlab2_{nit}i.ome"), OUTPUT_DIR / f"mip_{PREFIX}_RLTV_deconvlab2_{nit}i.ome.tiff"),
            (_timed_label("redlionfish_rl", nit, f"{PREFIX}_RL_redlionfish_{nit}i.ome"), OUTPUT_DIR / f"mip_{PREFIX}_RL_redlionfish_{nit}i.ome.tiff"),
            (_timed_label("skimage_rl", nit, f"{PREFIX}_RL_skimage_{nit}i.ome"), OUTPUT_DIR / f"mip_{PREFIX}_RL_skimage_{nit}i.ome.tiff"),
            (_timed_label("skiimage_cucim_rl", nit, f"{PREFIX}_RL_cucim_{nit}i.ome"), OUTPUT_DIR / f"mip_{PREFIX}_RL_cucim_{nit}i.ome.tiff"),
        ])

    # Filter to existing files and load per row
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

    # Flatten for iteration
    mip_data = [item for row in loaded_rows for item in row]

    if not mip_data:
        return

    print(f"\n  Creating per-channel montages ({n_ch} channels)...")

    # Smaller font for channel overlay
    overlay_font = None
    try:
        overlay_font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        try:
            overlay_font = ImageFont.truetype("DejaVuSans.ttf", 18)
        except OSError:
            overlay_font = ImageFont.load_default()

    label_height = 60
    padding = 4

    for ch_idx in range(n_ch):
        ch_meta = meta["channels"][ch_idx]
        em = ch_meta.get("emission_wavelength") or 0

        panel_rows = []
        for row_data in loaded_rows:
            row_panels = []
            for label, arr in row_data:
                if ch_idx < arr.shape[0]:
                    ch_img = arr[ch_idx].astype(np.float64)
                else:
                    continue
                # Normalise to 0-255 greyscale
                lo, hi = ch_img.min(), ch_img.max()
                if hi > lo:
                    ch_img = (ch_img - lo) / (hi - lo) * 255.0
                else:
                    ch_img = np.zeros_like(ch_img)
                grey = ch_img.clip(0, 255).astype(np.uint8)
                img = Image.fromarray(grey, mode="L").convert("RGB")

                # Draw channel label on top-left of the image
                img_draw = ImageDraw.Draw(img)
                ch_label = f"Ch{ch_idx}"
                img_draw.text((4, 2), ch_label, fill=(255, 255, 255), font=overlay_font)

                row_panels.append((img, label))
            if row_panels:
                panel_rows.append(row_panels)

        if not panel_rows:
            continue

        all_imgs = [img for row in panel_rows for img, _ in row]
        max_w = max(img.size[0] for img in all_imgs)
        max_h = max(img.size[1] for img in all_imgs)

        # Insert PSF metadata panel next to Source in row 0
        ch_meta_panel = _make_metadata_panel(meta, max_w, max_h, font)
        panel_rows[0].insert(1, (ch_meta_panel, "PSF Parameters"))

        n_cols = max(len(row) for row in panel_rows)
        n_grid_rows = len(panel_rows)

        cell_w = max_w + 2 * padding
        cell_h = max_h + label_height + 2 * padding

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

        _clean = PREFIX.replace(".ome", "")
        out = OUTPUT_DIR / f"decon_benchmark_{_clean}_ch{ch_idx}.png"
        montage.save(str(out))
        print(f"    Ch{ch_idx}: {out}  ({montage_w}x{montage_h})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Microscopy Deconvolution Benchmark",
    )
    parser.add_argument(
        "image", nargs="?", default=IMAGE_FILE,
        help="Image filename in data/ folder (default: %(default)s)",
    )
    parser.add_argument(
        "blocks", nargs="?", default=None,
        help=(
            "Block tiling mode. 'auto' (default): calculate minimum blocks "
            "so each tile \u2264 1024 px per side. 0 or 1: no tiling. "
            ">1: explicit block count."
        ),
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help=(
            "Benchmark mode: centre-crop the image to MAX_TILE_XY × MAX_TILE_Z "
            "(if larger) and only save montage images."
        ),
    )
    parser.add_argument(
        "--psf", action="store_true",
        help="Save PSF montage PNG (XY and XZ MIP) in the output folder.",
    )
    args = parser.parse_args()

    IMAGE_FILE = args.image
    if args.blocks is not None:
        _blk = args.blocks.strip().lower()
        if _blk == "auto":
            N_BLOCKS = "auto"
        else:
            N_BLOCKS = max(int(_blk), 0)  # 0 = no tiling
    # If the user supplied a path (absolute or relative with directories),
    # set DATA_DIR to that folder so every DATA_DIR / IMAGE_FILE reference
    # resolves correctly for both the image and the Huygens _decon file.
    _image_path = Path(IMAGE_FILE).resolve()
    if _image_path.parent != Path(".").resolve():
        DATA_DIR = _image_path.parent
        IMAGE_FILE = _image_path.name
    ORIGINAL_DATA_DIR = DATA_DIR
    HUYGENS_FILE, PREFIX = _derive_names(IMAGE_FILE)
    if args.benchmark:
        BENCHMARK_MODE = True
    if args.psf:
        SAVE_PSF = True

    print(f"Microscopy Deconvolution Benchmark — {IMAGE_FILE}")
    print(f"  Data folder  : {DATA_DIR}")
    print(f"  Huygens file : {HUYGENS_FILE}")
    print(f"  Output prefix: {PREFIX}")
    print(f"  Blocks       : {N_BLOCKS}")
    if BENCHMARK_MODE:
        print(f"  Benchmark    : ON  (crop to {MAX_TILE_XY}×{MAX_TILE_XY}×{MAX_TILE_Z}, montage only)")
    if SAVE_PSF:
        print(f"  Save PSFs    : ON")
    print("Available methods:", list(METHODS.keys()))

    # Step 1: Load and inspect
    data = benchmark_metadata()

    # Benchmark mode: centre-crop all channels to tile-size limits
    if BENCHMARK_MODE:
        import tifffile as _tiff
        images = data["images"]
        cropped = []
        for img in images:
            if img.ndim == 3:
                Z, H, W = img.shape
                nz = min(Z, MAX_TILE_Z)
                ny = min(H, MAX_TILE_XY)
                nx = min(W, MAX_TILE_XY)
                z0 = (Z - nz) // 2
                y0 = (H - ny) // 2
                x0 = (W - nx) // 2
                img = img[z0:z0+nz, y0:y0+ny, x0:x0+nx]
            elif img.ndim == 2:
                H, W = img.shape
                ny = min(H, MAX_TILE_XY)
                nx = min(W, MAX_TILE_XY)
                y0 = (H - ny) // 2
                x0 = (W - nx) // 2
                img = img[y0:y0+ny, x0:x0+nx]
            cropped.append(img)
        data["images"] = cropped
        print(f"  Benchmark crop: {images[0].shape} -> {cropped[0].shape}")

        # Write cropped channels as OME-TIFF with full metadata so that
        # load_image() on the cropped file returns correct pixel sizes,
        # NA, emission wavelengths, etc.
        meta = data["metadata"]
        stack = np.stack(cropped, axis=0)  # (C, Z, Y, X) or (C, Y, X)
        cropped_path = OUTPUT_DIR / f"_benchmark_crop_{IMAGE_FILE}"
        axes = "CZYX" if cropped[0].ndim == 3 else "CYX"
        px_x = meta.get("pixel_size_x")
        px_y = meta.get("pixel_size_y")
        px_z = meta.get("pixel_size_z")
        resolution = (1.0 / px_x, 1.0 / px_y) if px_x and px_y else None
        _tiff.imwrite(
            str(cropped_path), stack,
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
                    "Name": [
                        ch.get("name", f"Ch{i}")
                        for i, ch in enumerate(meta.get("channels", []))
                    ],
                    "EmissionWavelength": [
                        ch.get("emission_wavelength")
                        for ch in meta.get("channels", [])
                    ],
                    "ExcitationWavelength": [
                        ch.get("excitation_wavelength")
                        for ch in meta.get("channels", [])
                    ],
                },
            },
        )

        # Store original metadata overrides so every deconvolve_image() call
        # on the cropped file gets the correct pixel sizes, NA, wavelengths.
        _META_OVERRIDES.update({
            "na": meta.get("na"),
            "refractive_index": meta.get("refractive_index"),
            "pixel_size_xy": meta.get("pixel_size_x"),
            "pixel_size_z": meta.get("pixel_size_z"),
            "emission_wavelengths": [
                ch.get("emission_wavelength")
                for ch in meta.get("channels", [])
            ] or None,
        })
        # Update metadata sizes for PSF generation
        if cropped[0].ndim == 3:
            data["metadata"]["size_z"] = cropped[0].shape[0]
            data["metadata"]["size_y"] = cropped[0].shape[1]
            data["metadata"]["size_x"] = cropped[0].shape[2]
        else:
            data["metadata"]["size_y"] = cropped[0].shape[0]
            data["metadata"]["size_x"] = cropped[0].shape[1]
        # Redirect globals so all subsequent benchmark steps use cropped file
        DATA_DIR = OUTPUT_DIR
        IMAGE_FILE = cropped_path.name
        print(f"  Cropped image saved: {cropped_path}")

    # Step 2: Generate PSFs
    psfs = benchmark_psf(data["metadata"])

    # Step 3: Quick 2D RL
    benchmark_deconvolution_2d(data)

    # Step 4: Full 3D RL deconvolution (includes plane-by-plane)
    benchmark_full_3d()

    # Step 5: Huygens MIP for comparison
    benchmark_huygens_mip()

    # Step 6: Montage of all MIP images
    benchmark_montage()

    # In benchmark mode, remove individual MIP files (keep only montages)
    if BENCHMARK_MODE:
        for pattern in ("mip_*.png", "mip_*.tiff"):
            for f in OUTPUT_DIR.glob(pattern):
                f.unlink(missing_ok=True)
                print(f"  Removed: {f.name}")

    # Clean up deconwolf VkFFT kernel cache and FFTW wisdom files
    cwd = Path(".")
    for pattern in ("VkFFT_kernelCache_*.binary", "fftw_wisdom_*.dat"):
        for f in cwd.glob(pattern):
            f.unlink(missing_ok=True)
            print(f"  Removed cache file: {f.name}")

    # Remove temporary benchmark-crop file
    if BENCHMARK_MODE:
        crop_file = OUTPUT_DIR / f"_benchmark_crop_{_derive_names(args.image)[1]}.tiff"
        # Find any _benchmark_crop_* files in output
        for f in OUTPUT_DIR.glob("_benchmark_crop_*"):
            f.unlink(missing_ok=True)
            print(f"  Removed temp crop file: {f.name}")

    print("\n" + "=" * 70)
    print("Benchmark complete! Check the output/ folder for results.")
    print("=" * 70)
