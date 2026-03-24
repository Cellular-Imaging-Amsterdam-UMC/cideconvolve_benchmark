"""
Demo: Richardson-Lucy deconvolution only — U2OS widefield data.

Loads the U2OS_R3D_WF_DV widefield image from the data/ folder, generates
channel-specific PSFs from OME metadata, and runs Richardson-Lucy
deconvolution (2D single-slice, full 3D, and plane-by-plane).
Also generates MIPs from the Huygens deconvolution for comparison.

Usage:
    python benchmark.py
"""

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

IMAGE_FILE = "U2OS_R3D_WF_DV.ome.tiff"
HUYGENS_FILE = "U2OS_R3D_WF_DV_decon.ome.tiff"
PREFIX = "U2OS"
NITERS = [15, 45]  # iteration counts to test for full 3D RL
TIMINGS = {}  # key: output filename stem → elapsed seconds

# --- Toggle individual algorithms on/off ---
RUN_SDECONV_CPU = True
RUN_SDECONV_CUDA = True
RUN_PYCUDADECON = True
RUN_DECONWOLF_RL = False
RUN_DECONWOLF_SHB = False
RUN_DECONVLAB2_RL = False
RUN_DECONVLAB2_RLTV = False
RUN_PLANE_BY_PLANE = True
RUN_REDLIONFISH = True
RUN_SKIMAGE_RL = True
RUN_CUCIM_RL = True

# Channels to skip in the filtered multicolour montage (0-based indices).
# For example, [0] skips the first channel (often DAPI).
MONTAGE_SKIP_CHANNELS = [0]


def demo_metadata():
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


def demo_psf(metadata):
    """Step 2: Generate PSFs for each channel."""
    print("\n" + "=" * 70)
    print("STEP 2: Generate PSFs from metadata")
    print("=" * 70)

    psfs = []
    for ch_idx in range(len(metadata["channels"])):
        psf = generate_psf(metadata, channel_idx=ch_idx)
        psfs.append(psf)
        print(f"\nChannel {ch_idx} PSF:")
        print(f"  Shape: {psf.shape}")
        print(f"  Sum: {psf.sum():.6f}")
        print(f"  Max: {psf.max():.6e}")
        print(f"  Center slice max at: {np.unravel_index(psf[psf.shape[0]//2].argmax(), psf[psf.shape[0]//2].shape)}")

    # Save PSF MIP images (XY and XZ) for visual inspection
    _save_psf_mips(psfs, metadata)

    return psfs


def _save_psf_mips(psfs, metadata):
    """Save XY and XZ maximum-intensity projections of PSFs as colour PNGs."""
    from PIL import Image, ImageDraw, ImageFont

    n_ch = len(psfs)
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
        for ch_idx, psf in enumerate(psfs):
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

            # Get channel colour
            ch_meta = metadata["channels"][ch_idx]
            em = ch_meta.get("emission_wavelength")
            if em:
                from deconvolve import _emission_to_rgb
                rgb = _emission_to_rgb(em)
            else:
                rgb = (255, 255, 255)

            # Build colour image
            h, w = mip.shape
            img_arr = np.zeros((h, w, 3), dtype=np.uint8)
            for c in range(3):
                img_arr[:, :, c] = np.clip(mip * rgb[c], 0, 255).astype(np.uint8)

            # Apply gamma for better visibility of weak lobes
            img_arr = np.clip(np.power(img_arr / 255.0, 0.4) * 255, 0, 255).astype(np.uint8)

            # Upscale
            img = Image.fromarray(img_arr, mode="RGB")
            img = img.resize((w * scale, h * scale), Image.NEAREST)
            panels.append(img)

        # Montage: all channels side by side with labels
        label_height = 24
        padding = 2
        max_w = max(p.size[0] for p in panels)
        max_h = max(p.size[1] for p in panels)
        cell_w = max_w + 2 * padding
        cell_h = max_h + label_height + 2 * padding

        montage = Image.new("RGB", (n_ch * cell_w, cell_h), color=(0, 0, 0))
        draw = ImageDraw.Draw(montage)

        for i, panel in enumerate(panels):
            x0 = i * cell_w + padding + (max_w - panel.size[0]) // 2
            y0 = padding + (max_h - panel.size[1]) // 2
            montage.paste(panel, (x0, y0))

            ch_meta = metadata["channels"][i]
            em = ch_meta.get("emission_wavelength", 0)
            label = f"Ch{i} ({em:.0f} nm)"
            bbox = draw.textbbox((0, 0), label, font=font)
            tw = bbox[2] - bbox[0]
            tx = i * cell_w + padding + (max_w - tw) // 2
            ty = padding + max_h + 2
            draw.text((tx, ty), label, fill=(255, 255, 255), font=font)

        out = OUTPUT_DIR / f"psf_{PREFIX}_{view}.png"
        montage.save(str(out))
        print(f"  Saved PSF {view} MIP: {out}  ({montage.size[0]}x{montage.size[1]})")


def demo_deconvolution_2d(data):
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

    info = METHODS["richardson_lucy"]
    print(f"\n  richardson_lucy: {info['description']}")
    print(f"  Memory factor: ~{info['memory_factor']}x image size")

    t0 = time.perf_counter()
    result = deconvolve(img_2d, psf_2d, method="richardson_lucy", niter=30)
    elapsed = time.perf_counter() - t0
    print(f"  Result: shape={result.shape}, min={result.min():.2f}, max={result.max():.2f}")
    print(f"  Elapsed time: {elapsed:.1f} s")

    import tifffile
    out_path = OUTPUT_DIR / f"{PREFIX}_slice_z{mid_z}_ch0_richardson_lucy.tiff"
    tifffile.imwrite(str(out_path), result)
    print(f"  Saved to: {out_path}")


def demo_full_3d():
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
    if not RUN_SDECONV_CPU:
        print("\n--- (a) sdeconv CPU — SKIPPED (disabled) ---")
    for nit in NITERS if RUN_SDECONV_CPU else []:
        print(f"\n--- (a) sdeconv Richardson-Lucy on CPU, {nit} iterations ---")
        t0 = time.perf_counter()
        result_cpu = deconvolve_image(
            image_path, method="richardson_lucy", niter=nit, device="cpu",
        )
        elapsed = time.perf_counter() - t0
        for i, ch in enumerate(result_cpu["channels"]):
            print(f"  Channel {i}: shape={ch.shape}, min={ch.min():.2f}, max={ch.max():.2f}")
        print(f"  Elapsed time: {elapsed:.1f} s")
        out = OUTPUT_DIR / f"{PREFIX}_RL_sdeconv_cpu_{nit}i.ome.tiff"
        save_result(result_cpu, out)
        TIMINGS[out.stem] = elapsed
        print(f"  Saved to: {out}")

    # --- (b) sdeconv on CUDA ---
    if cuda_available and RUN_SDECONV_CUDA:
        for nit in NITERS:
            print(f"\n--- (b) sdeconv Richardson-Lucy on CUDA, {nit} iterations ---")
            t0 = time.perf_counter()
            result_gpu = deconvolve_image(
                image_path, method="richardson_lucy", niter=nit, device="cuda",
            )
            elapsed = time.perf_counter() - t0
            for i, ch in enumerate(result_gpu["channels"]):
                print(f"  Channel {i}: shape={ch.shape}, min={ch.min():.2f}, max={ch.max():.2f}")
            print(f"  Elapsed time: {elapsed:.1f} s")
            out = OUTPUT_DIR / f"{PREFIX}_RL_sdeconv_cuda_{nit}i.ome.tiff"
            save_result(result_gpu, out)
            TIMINGS[out.stem] = elapsed
            print(f"  Saved to: {out}")
    else:
        print("\n--- (b) sdeconv CUDA — SKIPPED (no GPU or disabled) ---")

    # --- (c) pycudadecon ---
    if cuda_available and RUN_PYCUDADECON:
        for nit in NITERS:
            print(f"\n--- (c) pycudadecon Richardson-Lucy (CUDA), {nit} iterations ---")
            try:
                t0 = time.perf_counter()
                result_pcd = deconvolve_image(
                    image_path, method="richardson_lucy_cuda", niter=nit,
                )
                elapsed = time.perf_counter() - t0
                for i, ch in enumerate(result_pcd["channels"]):
                    print(f"  Channel {i}: shape={ch.shape}, min={ch.min():.2f}, max={ch.max():.2f}")
                print(f"  Elapsed time: {elapsed:.1f} s")
                out = OUTPUT_DIR / f"{PREFIX}_RL_pycudadecon_{nit}i.ome.tiff"
                save_result(result_pcd, out)
                TIMINGS[out.stem] = elapsed
                print(f"  Saved to: {out}")
            except ImportError as exc:
                print(f"  SKIPPED — {exc}")
    else:
        print("\n--- (c) pycudadecon — SKIPPED (no GPU or disabled) ---")

    # --- (d) deconwolf RL ---
    if dw_available and RUN_DECONWOLF_RL:
        for nit in NITERS:
            print(f"\n--- (d) deconwolf RL, {nit} iterations ---")
            t0 = time.perf_counter()
            result_dw = deconvolve_image(
                image_path, method="deconwolf_rl", niter=nit,
            )
            elapsed = time.perf_counter() - t0
            for i, ch in enumerate(result_dw["channels"]):
                print(f"  Channel {i}: shape={ch.shape}, min={ch.min():.2f}, max={ch.max():.2f}")
            print(f"  Elapsed time: {elapsed:.1f} s")
            out = OUTPUT_DIR / f"{PREFIX}_RL_deconwolf_{nit}i.ome.tiff"
            save_result(result_dw, out)
            TIMINGS[out.stem] = elapsed
            print(f"  Saved to: {out}")
    else:
        print("\n--- (d) deconwolf RL — SKIPPED (dw not found or disabled) ---")

    # --- (e) deconwolf SHB ---
    if dw_available and RUN_DECONWOLF_SHB:
        for nit in NITERS:
            print(f"\n--- (e) deconwolf SHB, {nit} iterations ---")
            t0 = time.perf_counter()
            result_dw_shb = deconvolve_image(
                image_path, method="deconwolf_shb", niter=nit,
            )
            elapsed = time.perf_counter() - t0
            for i, ch in enumerate(result_dw_shb["channels"]):
                print(f"  Channel {i}: shape={ch.shape}, min={ch.min():.2f}, max={ch.max():.2f}")
            print(f"  Elapsed time: {elapsed:.1f} s")
            out = OUTPUT_DIR / f"{PREFIX}_SHB_deconwolf_{nit}i.ome.tiff"
            save_result(result_dw_shb, out)
            TIMINGS[out.stem] = elapsed
            print(f"  Saved to: {out}")
    else:
        print("\n--- (e) deconwolf SHB — SKIPPED (dw not found or disabled) ---")

    # --- (f) DeconvolutionLab2 RL ---
    if dl2_available and RUN_DECONVLAB2_RL:
        for nit in NITERS:
            print(f"\n--- (f) DeconvolutionLab2 RL, {nit} iterations ---")
            t0 = time.perf_counter()
            result_dl2 = deconvolve_image(
                image_path, method="deconvlab2_rl", niter=nit,
            )
            elapsed = time.perf_counter() - t0
            for i, ch in enumerate(result_dl2["channels"]):
                print(f"  Channel {i}: shape={ch.shape}, min={ch.min():.2f}, max={ch.max():.2f}")
            print(f"  Elapsed time: {elapsed:.1f} s")
            out = OUTPUT_DIR / f"{PREFIX}_RL_deconvlab2_{nit}i.ome.tiff"
            save_result(result_dl2, out)
            TIMINGS[out.stem] = elapsed
            print(f"  Saved to: {out}")
    else:
        print("\n--- (f) DeconvolutionLab2 RL — SKIPPED (JAR/java not found or disabled) ---")

    # --- (g) DeconvolutionLab2 RLTV ---
    if dl2_available and RUN_DECONVLAB2_RLTV:
        for nit in NITERS:
            print(f"\n--- (g) DeconvolutionLab2 RLTV, {nit} iterations ---")
            t0 = time.perf_counter()
            result_dl2tv = deconvolve_image(
                image_path, method="deconvlab2_rltv", niter=nit, tv_lambda=1e-4,
            )
            elapsed = time.perf_counter() - t0
            for i, ch in enumerate(result_dl2tv["channels"]):
                print(f"  Channel {i}: shape={ch.shape}, min={ch.min():.2f}, max={ch.max():.2f}")
            print(f"  Elapsed time: {elapsed:.1f} s")
            out = OUTPUT_DIR / f"{PREFIX}_RLTV_deconvlab2_{nit}i.ome.tiff"
            save_result(result_dl2tv, out)
            TIMINGS[out.stem] = elapsed
            print(f"  Saved to: {out}")
    else:
        print("\n--- (g) DeconvolutionLab2 RLTV — SKIPPED (JAR/java not found or disabled) ---")

    # --- (h) RedLionfish RL ---
    if RUN_REDLIONFISH:
        try:
            import RedLionfishDeconv
            rlf_available = True
        except ImportError:
            rlf_available = False
        if rlf_available:
            for nit in NITERS:
                print(f"\n--- (h) RedLionfish RL, {nit} iterations ---")
                t0 = time.perf_counter()
                result_rlf = deconvolve_image(
                    image_path, method="redlionfish_rl", niter=nit,
                )
                elapsed = time.perf_counter() - t0
                for i, ch in enumerate(result_rlf["channels"]):
                    print(f"  Channel {i}: shape={ch.shape}, min={ch.min():.2f}, max={ch.max():.2f}")
                print(f"  Elapsed time: {elapsed:.1f} s")
                out = OUTPUT_DIR / f"{PREFIX}_RL_redlionfish_{nit}i.ome.tiff"
                save_result(result_rlf, out)
                TIMINGS[out.stem] = elapsed
                print(f"  Saved to: {out}")
        else:
            print("\n--- (h) RedLionfish RL — SKIPPED (not installed) ---")
    else:
        print("\n--- (h) RedLionfish RL — SKIPPED (disabled) ---")

    # --- (i) scikit-image RL ---
    if RUN_SKIMAGE_RL:
        for nit in NITERS:
            print(f"\n--- (i) scikit-image RL (CPU), {nit} iterations ---")
            t0 = time.perf_counter()
            result_ski = deconvolve_image(
                image_path, method="skimage_rl", niter=nit,
            )
            elapsed = time.perf_counter() - t0
            for i, ch in enumerate(result_ski["channels"]):
                print(f"  Channel {i}: shape={ch.shape}, min={ch.min():.2f}, max={ch.max():.2f}")
            print(f"  Elapsed time: {elapsed:.1f} s")
            out = OUTPUT_DIR / f"{PREFIX}_RL_skimage_{nit}i.ome.tiff"
            save_result(result_ski, out)
            TIMINGS[out.stem] = elapsed
            print(f"  Saved to: {out}")
    else:
        print("\n--- (i) scikit-image RL — SKIPPED (disabled) ---")

    # --- (j) cuCIM RL ---
    if cuda_available and RUN_CUCIM_RL:
        try:
            import cupy  # noqa: F401
            import cucim  # noqa: F401
            cucim_available = True
        except ImportError:
            cucim_available = False
        if cucim_available:
            for nit in NITERS:
                print(f"\n--- (j) cuCIM RL (CUDA), {nit} iterations ---")
                t0 = time.perf_counter()
                result_cucim = deconvolve_image(
                    image_path, method="cucim_rl", niter=nit,
                )
                elapsed = time.perf_counter() - t0
                for i, ch in enumerate(result_cucim["channels"]):
                    print(f"  Channel {i}: shape={ch.shape}, min={ch.min():.2f}, max={ch.max():.2f}")
                print(f"  Elapsed time: {elapsed:.1f} s")
                out = OUTPUT_DIR / f"{PREFIX}_RL_cucim_{nit}i.ome.tiff"
                save_result(result_cucim, out)
                TIMINGS[out.stem] = elapsed
                print(f"  Saved to: {out}")
        else:
            print("\n--- (j) cuCIM RL — SKIPPED (cupy/cucim not installed) ---")
    else:
        print("\n--- (j) cuCIM RL — SKIPPED (no GPU or disabled) ---")


def demo_plane_by_plane():
    """Step 5: Plane-by-plane 3D RL deconvolution (sdeconv, 30/60/90 iter)."""
    print("\n" + "=" * 70)
    print("STEP 5: Plane-by-plane Richardson-Lucy (memory-efficient, 30/60/90 iter)")
    print("=" * 70)

    image_path = DATA_DIR / IMAGE_FILE

    if not RUN_PLANE_BY_PLANE:
        print("\n--- plane-by-plane — SKIPPED (disabled) ---")
        return

    for nit in NITERS:
        print(f"\n--- sdeconv plane-by-plane, {nit} iterations ---")
        t0 = time.perf_counter()
        result = deconvolve_image(
            image_path,
            method="richardson_lucy",
            niter=nit,
            plane_by_plane=True,
        )
        elapsed = time.perf_counter() - t0
        for i, ch in enumerate(result["channels"]):
            print(f"  Channel {i}: shape={ch.shape}, min={ch.min():.2f}, max={ch.max():.2f}")
        print(f"  Elapsed time: {elapsed:.1f} s")
        out_path = OUTPUT_DIR / f"{PREFIX}_plane_by_plane_RL_{nit}i.ome.tiff"
        save_result(result, out_path)
        TIMINGS[out_path.stem] = elapsed
        print(f"  Saved to: {out_path}")


def demo_huygens_mip():
    """Step 6: Generate MIP from Huygens (SVI) deconvolution for comparison."""
    print("\n" + "=" * 70)
    print("STEP 6: MIP from Huygens deconvolution (for comparison)")
    print("=" * 70)

    huygens_path = DATA_DIR / HUYGENS_FILE
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

    mip_tiff = OUTPUT_DIR / f"mip_Huygens_{PREFIX}.tiff"
    tifffile.imwrite(str(mip_tiff), mip)
    print(f"  Saved MIP TIFF: {mip_tiff}")

    mip_png = OUTPUT_DIR / f"mip_Huygens_{PREFIX}.png"
    save_mip_png(mip, mip_png, meta)
    print(f"  Saved MIP PNG:  {mip_png}")


def _timed_label(label, stem):
    """Append timing info to a label if available."""
    t = TIMINGS.get(stem)
    if t is not None:
        return f"{label}\n{t:.1f}s"
    return label


def demo_montage():
    """Step 7: Create a montage of all MIP PNG images with algorithm labels."""
    print("\n" + "=" * 70)
    print("STEP 7: Create montage of all MIP images")
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
            (OUTPUT_DIR / f"mip_{PREFIX}_RL_sdeconv_cpu_{nit}i.ome.png", _timed_label(f"sdeconv CPU {nit}i", f"{PREFIX}_RL_sdeconv_cpu_{nit}i.ome")),
            (OUTPUT_DIR / f"mip_{PREFIX}_RL_sdeconv_cuda_{nit}i.ome.png", _timed_label(f"sdeconv CUDA {nit}i", f"{PREFIX}_RL_sdeconv_cuda_{nit}i.ome")),
            (OUTPUT_DIR / f"mip_{PREFIX}_RL_pycudadecon_{nit}i.ome.png", _timed_label(f"pycudadecon {nit}i", f"{PREFIX}_RL_pycudadecon_{nit}i.ome")),
            (OUTPUT_DIR / f"mip_{PREFIX}_plane_by_plane_RL_{nit}i.ome.png", _timed_label(f"PbP sdeconv {nit}i", f"{PREFIX}_plane_by_plane_RL_{nit}i.ome")),
            (OUTPUT_DIR / f"mip_{PREFIX}_RL_deconwolf_{nit}i.ome.png", _timed_label(f"deconwolf RL {nit}i", f"{PREFIX}_RL_deconwolf_{nit}i.ome")),
            (OUTPUT_DIR / f"mip_{PREFIX}_SHB_deconwolf_{nit}i.ome.png", _timed_label(f"deconwolf SHB {nit}i", f"{PREFIX}_SHB_deconwolf_{nit}i.ome")),
            (OUTPUT_DIR / f"mip_{PREFIX}_RL_deconvlab2_{nit}i.ome.png", _timed_label(f"DL2 RL {nit}i", f"{PREFIX}_RL_deconvlab2_{nit}i.ome")),
            (OUTPUT_DIR / f"mip_{PREFIX}_RLTV_deconvlab2_{nit}i.ome.png", _timed_label(f"DL2 RLTV {nit}i", f"{PREFIX}_RLTV_deconvlab2_{nit}i.ome")),
            (OUTPUT_DIR / f"mip_{PREFIX}_RL_redlionfish_{nit}i.ome.png", _timed_label(f"RedLionfish {nit}i", f"{PREFIX}_RL_redlionfish_{nit}i.ome")),
            (OUTPUT_DIR / f"mip_{PREFIX}_RL_skimage_{nit}i.ome.png", _timed_label(f"skimage RL {nit}i", f"{PREFIX}_RL_skimage_{nit}i.ome")),
            (OUTPUT_DIR / f"mip_{PREFIX}_RL_cucim_{nit}i.ome.png", _timed_label(f"cuCIM RL {nit}i", f"{PREFIX}_RL_cucim_{nit}i.ome")),
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

    # Number of columns = max row width
    n_cols = max(len(row) for row in loaded_rows)
    n_rows = len(loaded_rows)

    # Label bar height (extra space for optional timing line)
    label_height = 60
    padding = 4

    # All panels same size (use max dimensions across all images)
    all_imgs = [img for row in loaded_rows for img, _ in row]
    max_w = max(img.size[0] for img in all_imgs)
    max_h = max(img.size[1] for img in all_imgs)
    cell_w = max_w + 2 * padding
    cell_h = max_h + label_height + 2 * padding

    montage_w = n_cols * cell_w
    montage_h = n_rows * cell_h
    montage = Image.new("RGB", (montage_w, montage_h), color=(0, 0, 0))
    draw = ImageDraw.Draw(montage)

    # Try to load a decent font, fall back to default
    font = None
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 18)
        except OSError:
            font = ImageFont.load_default()

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

    out_path = OUTPUT_DIR / f"montage_{PREFIX}.png"
    montage.save(str(out_path))
    print(f"  Saved montage: {out_path}  ({montage_w}x{montage_h})")

    # --- Filtered multicolour montage (skip channels) ---
    if MONTAGE_SKIP_CHANNELS:
        _make_filtered_colour_montage(font)

    # --- Per-channel montages ---
    _make_per_channel_montages(font)


def _make_filtered_colour_montage(font):
    """Create a multicolour montage skipping the channels listed in MONTAGE_SKIP_CHANNELS.

    Re-composites from MIP TIFF files (C, Y, X) so that individual channels
    can be excluded — useful to e.g. drop the DAPI channel for clarity.
    """
    from PIL import Image, ImageDraw, ImageFont
    import tifffile

    data = load_image(DATA_DIR / IMAGE_FILE)
    meta = data["metadata"]
    n_ch = meta["n_channels"]

    keep = [i for i in range(n_ch) if i not in MONTAGE_SKIP_CHANNELS]
    if not keep:
        print("  All channels skipped — nothing to compose.")
        return

    skipped_str = ",".join(str(c) for c in sorted(MONTAGE_SKIP_CHANNELS))
    print(f"\n  Creating filtered colour montage (skipping ch {skipped_str})...")

    # Build the same row structure as demo_montage
    mip_rows = [
        [
            ("Source", OUTPUT_DIR / "mip_source.ome.tiff"),
            ("Huygens", OUTPUT_DIR / f"mip_Huygens_{PREFIX}.tiff"),
        ],
    ]
    for nit in NITERS:
        mip_rows.append([
            (_timed_label(f"sdeconv CPU {nit}i", f"{PREFIX}_RL_sdeconv_cpu_{nit}i.ome"), OUTPUT_DIR / f"mip_{PREFIX}_RL_sdeconv_cpu_{nit}i.ome.tiff"),
            (_timed_label(f"sdeconv CUDA {nit}i", f"{PREFIX}_RL_sdeconv_cuda_{nit}i.ome"), OUTPUT_DIR / f"mip_{PREFIX}_RL_sdeconv_cuda_{nit}i.ome.tiff"),
            (_timed_label(f"pycudadecon {nit}i", f"{PREFIX}_RL_pycudadecon_{nit}i.ome"), OUTPUT_DIR / f"mip_{PREFIX}_RL_pycudadecon_{nit}i.ome.tiff"),
            (_timed_label(f"PbP sdeconv {nit}i", f"{PREFIX}_plane_by_plane_RL_{nit}i.ome"), OUTPUT_DIR / f"mip_{PREFIX}_plane_by_plane_RL_{nit}i.ome.tiff"),
            (_timed_label(f"deconwolf RL {nit}i", f"{PREFIX}_RL_deconwolf_{nit}i.ome"), OUTPUT_DIR / f"mip_{PREFIX}_RL_deconwolf_{nit}i.ome.tiff"),
            (_timed_label(f"deconwolf SHB {nit}i", f"{PREFIX}_SHB_deconwolf_{nit}i.ome"), OUTPUT_DIR / f"mip_{PREFIX}_SHB_deconwolf_{nit}i.ome.tiff"),
            (_timed_label(f"DL2 RL {nit}i", f"{PREFIX}_RL_deconvlab2_{nit}i.ome"), OUTPUT_DIR / f"mip_{PREFIX}_RL_deconvlab2_{nit}i.ome.tiff"),
            (_timed_label(f"DL2 RLTV {nit}i", f"{PREFIX}_RLTV_deconvlab2_{nit}i.ome"), OUTPUT_DIR / f"mip_{PREFIX}_RLTV_deconvlab2_{nit}i.ome.tiff"),
            (_timed_label(f"RedLionfish {nit}i", f"{PREFIX}_RL_redlionfish_{nit}i.ome"), OUTPUT_DIR / f"mip_{PREFIX}_RL_redlionfish_{nit}i.ome.tiff"),
            (_timed_label(f"skimage RL {nit}i", f"{PREFIX}_RL_skimage_{nit}i.ome"), OUTPUT_DIR / f"mip_{PREFIX}_RL_skimage_{nit}i.ome.tiff"),
            (_timed_label(f"cuCIM RL {nit}i", f"{PREFIX}_RL_cucim_{nit}i.ome"), OUTPUT_DIR / f"mip_{PREFIX}_RL_cucim_{nit}i.ome.tiff"),
        ])

    # Load existing TIFF MIPs and composite only the kept channels
    loaded_rows = []
    total = 0
    for row_entries in mip_rows:
        row_images = []
        for label, path in row_entries:
            if not path.exists():
                continue
            arr = tifffile.imread(str(path))
            if arr.ndim == 2:
                arr = arr[np.newaxis]
            c, h, w = arr.shape
            # Additive false-colour composite of kept channels
            canvas = np.zeros((h, w, 3), dtype=np.float64)
            for ch_idx in keep:
                if ch_idx >= c:
                    continue
                ch_img = arr[ch_idx].astype(np.float64)
                lo, hi = ch_img.min(), ch_img.max()
                if hi > lo:
                    ch_img = (ch_img - lo) / (hi - lo)
                else:
                    ch_img = np.zeros_like(ch_img)
                rgb = _channel_color(meta, ch_idx)
                for ci in range(3):
                    canvas[:, :, ci] += ch_img * (rgb[ci] / 255.0)
            canvas = np.clip(canvas, 0, 1)
            canvas = (canvas * 255).astype(np.uint8)
            img = Image.fromarray(canvas, mode="RGB")
            row_images.append((img, label))
            total += 1
        if row_images:
            loaded_rows.append(row_images)

    if total == 0:
        print("  No MIP TIFF files found — skipping filtered montage.")
        return

    # Layout — same logic as the main montage
    n_cols = max(len(row) for row in loaded_rows)
    n_rows = len(loaded_rows)
    label_height = 60
    padding = 4

    all_imgs = [img for row in loaded_rows for img, _ in row]
    max_w = max(img.size[0] for img in all_imgs)
    max_h = max(img.size[1] for img in all_imgs)
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
            x_off = (max_w - img.size[0]) // 2
            y_off = (max_h - img.size[1]) // 2
            montage.paste(img, (x0 + x_off, y0 + y_off))

            bbox = draw.textbbox((0, 0), label, font=font)
            tw = bbox[2] - bbox[0]
            tx = x0 + (max_w - tw) // 2
            ty = y0 + max_h + padding
            draw.text((tx, ty), label, fill=(255, 255, 255), font=font)

    kept_str = "+".join(str(c) for c in keep)
    out_path = OUTPUT_DIR / f"montage_{PREFIX}_ch{kept_str}.png"
    montage.save(str(out_path))
    print(f"  Saved filtered montage: {out_path}  ({montage_w}x{montage_h})")


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
            (_timed_label(f"sdeconv CPU {nit}i", f"{PREFIX}_RL_sdeconv_cpu_{nit}i.ome"), OUTPUT_DIR / f"mip_{PREFIX}_RL_sdeconv_cpu_{nit}i.ome.tiff"),
            (_timed_label(f"sdeconv CUDA {nit}i", f"{PREFIX}_RL_sdeconv_cuda_{nit}i.ome"), OUTPUT_DIR / f"mip_{PREFIX}_RL_sdeconv_cuda_{nit}i.ome.tiff"),
            (_timed_label(f"pycudadecon {nit}i", f"{PREFIX}_RL_pycudadecon_{nit}i.ome"), OUTPUT_DIR / f"mip_{PREFIX}_RL_pycudadecon_{nit}i.ome.tiff"),
            (_timed_label(f"PbP sdeconv {nit}i", f"{PREFIX}_plane_by_plane_RL_{nit}i.ome"), OUTPUT_DIR / f"mip_{PREFIX}_plane_by_plane_RL_{nit}i.ome.tiff"),
            (_timed_label(f"deconwolf RL {nit}i", f"{PREFIX}_RL_deconwolf_{nit}i.ome"), OUTPUT_DIR / f"mip_{PREFIX}_RL_deconwolf_{nit}i.ome.tiff"),
            (_timed_label(f"deconwolf SHB {nit}i", f"{PREFIX}_SHB_deconwolf_{nit}i.ome"), OUTPUT_DIR / f"mip_{PREFIX}_SHB_deconwolf_{nit}i.ome.tiff"),
            (_timed_label(f"DL2 RL {nit}i", f"{PREFIX}_RL_deconvlab2_{nit}i.ome"), OUTPUT_DIR / f"mip_{PREFIX}_RL_deconvlab2_{nit}i.ome.tiff"),
            (_timed_label(f"DL2 RLTV {nit}i", f"{PREFIX}_RLTV_deconvlab2_{nit}i.ome"), OUTPUT_DIR / f"mip_{PREFIX}_RLTV_deconvlab2_{nit}i.ome.tiff"),
            (_timed_label(f"RedLionfish {nit}i", f"{PREFIX}_RL_redlionfish_{nit}i.ome"), OUTPUT_DIR / f"mip_{PREFIX}_RL_redlionfish_{nit}i.ome.tiff"),
            (_timed_label(f"skimage RL {nit}i", f"{PREFIX}_RL_skimage_{nit}i.ome"), OUTPUT_DIR / f"mip_{PREFIX}_RL_skimage_{nit}i.ome.tiff"),
            (_timed_label(f"cuCIM RL {nit}i", f"{PREFIX}_RL_cucim_{nit}i.ome"), OUTPUT_DIR / f"mip_{PREFIX}_RL_cucim_{nit}i.ome.tiff"),
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

    # Number of columns = max row width from loaded_rows
    n_cols = max(len(row) for row in loaded_rows)
    n_grid_rows = len(loaded_rows)

    for ch_idx in range(n_ch):
        ch_meta = meta["channels"][ch_idx]
        em = ch_meta.get("emission_wavelength", 520)

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
                ch_label = f"Ch{ch_idx} ({em:.0f} nm)"
                img_draw.text((4, 2), ch_label, fill=(255, 255, 255), font=overlay_font)

                row_panels.append((img, label))
            if row_panels:
                panel_rows.append(row_panels)

        if not panel_rows:
            continue

        all_imgs = [img for row in panel_rows for img, _ in row]
        max_w = max(img.size[0] for img in all_imgs)
        max_h = max(img.size[1] for img in all_imgs)
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

        out = OUTPUT_DIR / f"montage_{PREFIX}_ch{ch_idx}_{em:.0f}nm.png"
        montage.save(str(out))
        print(f"    Ch{ch_idx} ({em:.0f} nm): {out}  ({montage_w}x{montage_h})")


if __name__ == "__main__":
    print("Microscopy Deconvolution Demo — U2OS Widefield (Richardson-Lucy only)")
    print("Available methods:", list(METHODS.keys()))

    # Step 1: Load and inspect
    data = demo_metadata()

    # Step 2: Generate PSFs
    psfs = demo_psf(data["metadata"])

    # Step 3: Quick 2D RL
    demo_deconvolution_2d(data)

    # Step 4: Full 3D RL deconvolution
    demo_full_3d()

    # Step 5: Memory-efficient plane-by-plane RL
    demo_plane_by_plane()

    # Step 6: Huygens MIP for comparison
    demo_huygens_mip()

    # Step 7: Montage of all MIP images
    demo_montage()

    print("\n" + "=" * 70)
    print("Demo complete! Check the output/ folder for results.")
    print("=" * 70)
