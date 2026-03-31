"""
Demo: Deconvolve the example confocal dendrite dataset.

Loads the 2-channel 3D confocal image from the data/ folder, generates
channel-specific PSFs from OME metadata, and runs Richardson-Lucy
deconvolution. Also demonstrates Wiener and Spitfire on a single slice.

Usage:
    python demo_deconvolve.py
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
    deconvolve,
    deconvolve_image,
    generate_psf,
    load_image,
    save_mip_png,
    save_result,
)

import numpy as np

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def demo_metadata():
    """Step 1: Load the image and inspect metadata."""
    print("\n" + "=" * 70)
    print("STEP 1: Load image and extract metadata")
    print("=" * 70)

    image_path = DATA_DIR / "U2OS_R3D_WF_DV.ome.tiff"
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

    return psfs


def demo_deconvolution_2d(data):
    """Step 3: Quick 2D deconvolution of a single Z-slice."""
    print("\n" + "=" * 70)
    print("STEP 3: 2D deconvolution of single Z-slice (all algorithms)")
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

    # Run all sdeconv algorithms
    for method_name in ["sdeconv_rl", "sdeconv_wiener", "sdeconv_spitfire"]:
        info = METHODS[method_name]
        print(f"\n  {method_name}: {info['description']}")
        print(f"  Memory factor: ~{info['memory_factor']}x image size")

        t0 = time.perf_counter()
        result = deconvolve(img_2d, psf_2d, method=method_name, niter=10)
        elapsed = time.perf_counter() - t0
        print(f"  Result: shape={result.shape}, min={result.min():.2f}, max={result.max():.2f}")
        print(f"  Elapsed time: {elapsed:.1f} s")

        # Save
        import tifffile
        out_path = OUTPUT_DIR / f"slice_z{mid_z}_ch0_{method_name}.tiff"
        tifffile.imwrite(str(out_path), result)
        print(f"  Saved to: {out_path}")


def demo_full_3d():
    """Step 4: Full 3D deconvolution of all channels (all algorithms)."""
    print("\n" + "=" * 70)
    print("STEP 4: Full 3D deconvolution (all algorithms, all channels)")
    print("=" * 70)

    image_path = DATA_DIR / "U2OS_R3D_WF_DV.ome.tiff"

    method_params = {
        "sdeconv_rl": {"niter": 15},
        "sdeconv_wiener":          {"beta": 1e-5},
        "sdeconv_spitfire":        {"weight": 0.6, "reg": 0.995},
    }

    for method_name, params in method_params.items():
        info = METHODS[method_name]
        print(f"\n--- {method_name}: {info['description']} ---")
        print(f"  Memory factor: ~{info['memory_factor']}x image size")

        t0 = time.perf_counter()
        result = deconvolve_image(
            image_path,
            method=method_name,
            **params,
        )
        elapsed = time.perf_counter() - t0

        for i, ch in enumerate(result["channels"]):
            print(f"  Channel {i}: shape={ch.shape}, min={ch.min():.2f}, max={ch.max():.2f}")
        print(f"  Elapsed time: {elapsed:.1f} s")

        suffix = {"sdeconv_rl": "RL", "sdeconv_wiener": "Wiener", "sdeconv_spitfire": "Spitfire"}[method_name]
        out_path = OUTPUT_DIR / f"dendrites_deconvolved_{suffix}.ome.tiff"
        save_result(result, out_path)
        print(f"  Saved to: {out_path}")

    return result


def demo_plane_by_plane():
    """Step 5: Plane-by-plane 3D multichannel deconvolution (memory-efficient)."""
    print("\n" + "=" * 70)
    print("STEP 5: Plane-by-plane 3D deconvolution (all channels, memory-efficient)")
    print("=" * 70)

    image_path = DATA_DIR / "U2OS_R3D_WF_DV.ome.tiff"

    t0 = time.perf_counter()
    result = deconvolve_image(
        image_path,
        method="sdeconv_rl",
        niter=10,
        plane_by_plane=True,
    )
    elapsed = time.perf_counter() - t0

    for i, ch in enumerate(result["channels"]):
        print(f"  Channel {i}: shape={ch.shape}, min={ch.min():.2f}, max={ch.max():.2f}")
    print(f"  Elapsed time: {elapsed:.1f} s")

    out_path = OUTPUT_DIR / "dendrites_plane_by_plane_RL.ome.tiff"
    save_result(result, out_path)
    print(f"  Saved to: {out_path}")


def demo_huygens_mip():
    """Step 6: Generate MIP from Huygens (SVI) deconvolution for comparison."""
    print("\n" + "=" * 70)
    print("STEP 6: MIP from Huygens deconvolution (for comparison)")
    print("=" * 70)

    huygens_path = DATA_DIR / "U2OS_R3D_WF_DV_decon.ome.tiff"
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

    mip_tiff = OUTPUT_DIR / "mip_Huygens.tiff"
    tifffile.imwrite(str(mip_tiff), mip)
    print(f"  Saved MIP TIFF: {mip_tiff}")

    mip_png = OUTPUT_DIR / "mip_Huygens.png"
    save_mip_png(mip, mip_png, meta)
    print(f"  Saved MIP PNG:  {mip_png}")


if __name__ == "__main__":
    print("Microscopy Deconvolution Demo")
    print("Available methods:", list(METHODS.keys()))

    # Step 1: Load and inspect
    data = demo_metadata()

    # Step 2: Generate PSFs
    psfs = demo_psf(data["metadata"])

    # Step 3: Quick 2D comparison of algorithms
    # demo_deconvolution_2d(data)

    # Step 4: Full 3D deconvolution (may take a while on CPU)
    # Uncomment to run:
    demo_full_3d()

    # Step 5: Memory-efficient plane-by-plane
    # Uncomment to run:
    # demo_plane_by_plane()

    # Step 6: Huygens MIP for comparison
    demo_huygens_mip()

    print("\n" + "=" * 70)
    print("Demo complete! Check the output/ folder for results.")
    print("=" * 70)
