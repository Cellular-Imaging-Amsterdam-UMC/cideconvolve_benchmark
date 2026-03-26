"""
Microscopy image deconvolution module.

Reads OME-TIFF microscopy images, generates physically accurate PSFs from
image metadata, and performs deconvolution using multiple algorithms.

Backends:
    - sdeconv (primary): Pure PyTorch, 2D+3D, Richardson-Lucy / Wiener / Spitfire
    - pycudadecon (optional): CUDA-accelerated Richardson-Lucy for 3D

PSF generation via psf_generator: scalar/vectorial propagation models with
Gibson-Lanni aberration correction for high-NA objectives.

Considerations:
    1. Incomplete metadata: If OME-TIFF files lack microscope type or NA,
       sensible defaults are used (widefield, NA=1.4). All metadata can be
       overridden via function parameters.
    2. Memory for large volumes: Full 3D Spitfire on large volumes is
       memory-intensive. Richardson-Lucy is the default (most memory-efficient
       iterative method). Consider plane_by_plane=True for very large volumes.
       Approx memory: RL ~4x image size, Wiener ~3x, Spitfire ~8x.
    3. Edge artifacts: Padding (default pad=13) reduces boundary artifacts in
       all sdeconv algorithms. Adjustable via the pad parameter.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Optional, Sequence, Union

# ---------------------------------------------------------------------------
# Fix DLL conflicts between conda-MKL (numpy) and pip-installed PyTorch on
# Windows.  Torch must be imported *before* numpy so that its own CUDA/OpenMP
# DLLs are loaded first.  Then we add conda's Library\bin so MKL can be found.
# KMP_DUPLICATE_LIB_OK silences the duplicate-OpenMP warning that remains.
# ---------------------------------------------------------------------------
if sys.platform == "win32":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch  # must come before numpy

if sys.platform == "win32":
    _conda_prefix = os.environ.get("CONDA_PREFIX") or sys.prefix
    _lib_bin = os.path.join(_conda_prefix, "Library", "bin")
    if os.path.isdir(_lib_bin):
        os.add_dll_directory(_lib_bin)

import numpy as np
import tifffile

logger = logging.getLogger(__name__)

# OME XML namespace
_OME_NS = "http://www.openmicroscopy.org/Schemas/OME/2016-06"

# ---------------------------------------------------------------------------
# External tool paths
# ---------------------------------------------------------------------------

_DECONVLAB2_JAR = Path(__file__).parent / "bin" / "DeconvolutionLab_2.jar"
_IJ_JAR = Path(
    os.environ.get("USERPROFILE") or os.environ.get("HOME") or ""
) / ".m2" / "repository" / "net" / "imagej" / "ij" / "1.51h" / "ij-1.51h.jar"
_BIN_DIR = Path(__file__).parent / "bin"
_DW_EXE = str(_BIN_DIR / "dw.exe") if (_BIN_DIR / "dw.exe").is_file() else (shutil.which("dw") or "")
_DW_BW_EXE = str(_BIN_DIR / "dw_bw.exe") if (_BIN_DIR / "dw_bw.exe").is_file() else (shutil.which("dw_bw") or "")

# ---------------------------------------------------------------------------
# Helper: detect GPU availability
# ---------------------------------------------------------------------------

def _get_device() -> str:
    """Return 'cuda:0' if CUDA is available, else 'cpu'."""
    return "cuda:0" if torch.cuda.is_available() else "cpu"


# ===========================================================================
# Phase 2: OME-TIFF Reader & Metadata Extraction
# ===========================================================================

def _parse_ome_xml(xml_path: Union[str, Path]) -> dict[str, Any]:
    """Parse an OME companion XML file and extract microscopy metadata.

    Falls back gracefully when fields are missing.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns = {"ome": _OME_NS}
    meta: dict[str, Any] = {}

    # --- Objective ---
    obj = root.find(".//ome:Instrument/ome:Objective", ns)
    if obj is not None:
        na_str = obj.get("LensNA")
        meta["na"] = float(na_str) if na_str else None
        meta["magnification"] = (
            float(obj.get("NominalMagnification"))
            if obj.get("NominalMagnification")
            else None
        )
        meta["immersion"] = obj.get("Immersion")

    # --- ObjectiveSettings (refractive index) ---
    objset = root.find(".//ome:Image/ome:ObjectiveSettings", ns)
    if objset is not None:
        ri_str = objset.get("RefractiveIndex")
        meta["refractive_index"] = float(ri_str) if ri_str else None

    # --- Pixels ---
    pixels = root.find(".//ome:Image/ome:Pixels", ns)
    if pixels is not None:
        for dim in ("X", "Y", "Z"):
            key = f"PhysicalSize{dim}"
            val = pixels.get(key)
            meta[f"pixel_size_{dim.lower()}"] = float(val) if val else None
        meta["size_x"] = int(pixels.get("SizeX", 0))
        meta["size_y"] = int(pixels.get("SizeY", 0))
        meta["size_z"] = int(pixels.get("SizeZ", 0))
        meta["size_c"] = int(pixels.get("SizeC", 0))
        meta["size_t"] = int(pixels.get("SizeT", 0))

    # --- Channels ---
    channels = root.findall(".//ome:Image/ome:Pixels/ome:Channel", ns)
    ch_info: list[dict[str, Any]] = []
    microscope_type = None
    for ch in channels:
        info: dict[str, Any] = {}
        ex = ch.get("ExcitationWavelength")
        em = ch.get("EmissionWavelength")
        info["excitation_wavelength"] = float(ex) if ex else None
        info["emission_wavelength"] = float(em) if em else None
        info["pinhole_size"] = (
            float(ch.get("PinholeSize")) if ch.get("PinholeSize") else None
        )
        acq = ch.get("AcquisitionMode")
        info["acquisition_mode"] = acq
        if acq and "confocal" in acq.lower():
            microscope_type = "confocal"
        ch_info.append(info)

    meta["channels"] = ch_info
    meta["microscope_type"] = microscope_type or "widefield"
    return meta


def _extract_bioio_metadata(img) -> dict[str, Any]:
    """Extract metadata from a bioio BioImage object."""
    meta: dict[str, Any] = {}
    pps = img.physical_pixel_sizes
    meta["pixel_size_x"] = pps.X
    meta["pixel_size_y"] = pps.Y
    meta["pixel_size_z"] = pps.Z
    meta["size_x"] = img.dims.X
    meta["size_y"] = img.dims.Y
    meta["size_z"] = img.dims.Z
    meta["size_c"] = img.dims.C
    meta["size_t"] = img.dims.T
    meta["channel_names"] = img.channel_names

    # Try OME metadata for richer info
    try:
        ome = img.ome_metadata
        image = ome.images[0]
        pixels = image.pixels

        ch_info = []
        microscope_type = None
        for ch in pixels.channels:
            info: dict[str, Any] = {}
            info["excitation_wavelength"] = (
                float(ch.excitation_wavelength)
                if ch.excitation_wavelength is not None
                else None
            )
            info["emission_wavelength"] = (
                float(ch.emission_wavelength)
                if ch.emission_wavelength is not None
                else None
            )
            info["pinhole_size"] = (
                float(ch.pinhole_size) if ch.pinhole_size is not None else None
            )
            acq = getattr(ch, "acquisition_mode", None)
            info["acquisition_mode"] = str(acq) if acq else None
            if acq and "confocal" in str(acq).lower():
                microscope_type = "confocal"
            ch_info.append(info)
        meta["channels"] = ch_info
        meta["microscope_type"] = microscope_type or "widefield"

        # Objective info
        if image.objective_settings is not None:
            ri = getattr(image.objective_settings, "refractive_index", None)
            meta["refractive_index"] = float(ri) if ri is not None else None

        # Try to get NA from instrument
        if ome.instruments:
            for instr in ome.instruments:
                if instr.objectives:
                    obj = instr.objectives[0]
                    if obj.lens_na is not None:
                        meta["na"] = float(obj.lens_na)
                    if obj.nominal_magnification is not None:
                        meta["magnification"] = float(obj.nominal_magnification)
                    if obj.immersion is not None:
                        meta["immersion"] = str(obj.immersion)
                    break
    except Exception as e:
        logger.warning("Could not extract full OME metadata: %s", e)

    return meta


def load_image(
    path: Union[str, Path],
    *,
    # User overrides for missing/incorrect metadata
    na: Optional[float] = None,
    refractive_index: Optional[float] = None,
    microscope_type: Optional[str] = None,
    pixel_size_xy: Optional[float] = None,
    pixel_size_z: Optional[float] = None,
    emission_wavelengths: Optional[list[float]] = None,
    sample_refractive_index: float = 1.33,
) -> dict[str, Any]:
    """Load an OME-TIFF image and extract microscopy metadata.

    Parameters
    ----------
    path : str or Path
        Path to an OME-TIFF file or a companion .ome file. If a companion
        file is given, the associated TIFF files are read automatically.
    na : float, optional
        Override numerical aperture (default: from metadata or 1.4).
    refractive_index : float, optional
        Override immersion medium refractive index (default: from metadata
        or 1.515 for oil).
    microscope_type : str, optional
        Override microscope type: "confocal" or "widefield".
    pixel_size_xy : float, optional
        Override lateral pixel size in micrometers.
    pixel_size_z : float, optional
        Override axial pixel size in micrometers.
    emission_wavelengths : list[float], optional
        Override emission wavelengths in nm, one per channel.
    sample_refractive_index : float
        Refractive index of the sample medium (default 1.33 for aqueous).

    Returns
    -------
    dict with keys:
        'images': list of numpy arrays, one per channel, shape (Z,Y,X) or (Y,X)
        'metadata': dict with all microscopy parameters
    """
    path = Path(path)
    meta: dict[str, Any] = {}
    images: list[np.ndarray] = []

    # Check if we have a companion OME file
    companion_path = None
    if path.suffix == ".ome" and not path.name.endswith(".ome.tiff"):
        companion_path = path
    else:
        # Look for a companion file alongside the TIFF
        candidate = path.parent / path.name.replace(".ome.tiff", ".ome").replace(
            ".ome.tif", ".ome"
        )
        if candidate.exists() and candidate != path:
            companion_path = candidate
        # Also check common companion patterns (e.g., basename_ch.companion.ome)
        for f in path.parent.glob("*.companion.ome"):
            companion_path = f
            break

    # Parse companion OME XML if available
    if companion_path is not None:
        meta = _parse_ome_xml(companion_path)

    # Read image data via bioio or tifffile
    try:
        from bioio import BioImage

        if companion_path is not None and path.suffix == ".ome":
            # Find the associated TIFF files from the companion
            tiff_files = sorted(path.parent.glob("*.ome.tiff")) + sorted(
                path.parent.glob("*.ome.tif")
            )
            if tiff_files:
                # Read each channel file
                for tiff_path in tiff_files:
                    try:
                        img = BioImage(tiff_path)
                    except Exception:
                        # Binary OME-TIFFs (multi-file) are not supported
                        # by bioio-ome-tiff; fall back to tifffile.
                        logger.info("BioImage cannot read %s, using tifffile", tiff_path.name)
                        data = tifffile.imread(str(tiff_path))
                        images.append(np.asarray(data, dtype=np.float32))
                        continue
                    bioio_meta = _extract_bioio_metadata(img)
                    # Merge bioio metadata (companion XML takes priority)
                    for k, v in bioio_meta.items():
                        if k not in meta or meta[k] is None:
                            meta[k] = v
                    # Extract all channels from this file
                    for c in range(img.dims.C):
                        if img.dims.Z > 1:
                            channel_data = img.get_image_data("ZYX", T=0, C=c)
                        else:
                            channel_data = img.get_image_data("YX", T=0, C=c)
                        images.append(np.asarray(channel_data, dtype=np.float32))
            else:
                raise FileNotFoundError(
                    f"No .ome.tiff files found alongside companion {companion_path}"
                )
        else:
            img = BioImage(path)
            bioio_meta = _extract_bioio_metadata(img)
            for k, v in bioio_meta.items():
                if k not in meta or meta[k] is None:
                    meta[k] = v
            for c in range(img.dims.C):
                if img.dims.Z > 1:
                    channel_data = img.get_image_data("ZYX", T=0, C=c)
                else:
                    channel_data = img.get_image_data("YX", T=0, C=c)
                images.append(np.asarray(channel_data, dtype=np.float32))

    except ImportError:
        logger.warning("bioio not available, falling back to tifffile")
        if companion_path is not None and path.suffix == ".ome":
            tiff_files = sorted(path.parent.glob("*.ome.tiff")) + sorted(
                path.parent.glob("*.ome.tif")
            )
            for tiff_path in tiff_files:
                data = tifffile.imread(str(tiff_path))
                images.append(np.asarray(data, dtype=np.float32))
        else:
            data = tifffile.imread(str(path))
            if data.ndim == 4:  # Assume CZYX
                for c in range(data.shape[0]):
                    images.append(np.asarray(data[c], dtype=np.float32))
            elif data.ndim == 3:  # Single channel ZYX
                images.append(np.asarray(data, dtype=np.float32))
            elif data.ndim == 2:  # Single 2D image
                images.append(np.asarray(data, dtype=np.float32))

    # Apply user overrides (Consideration 1: incomplete metadata)
    if na is not None:
        meta["na"] = na
    if refractive_index is not None:
        meta["refractive_index"] = refractive_index
    if microscope_type is not None:
        meta["microscope_type"] = microscope_type
    if pixel_size_xy is not None:
        meta["pixel_size_x"] = pixel_size_xy
        meta["pixel_size_y"] = pixel_size_xy
    if pixel_size_z is not None:
        meta["pixel_size_z"] = pixel_size_z
    if emission_wavelengths is not None:
        if "channels" not in meta:
            meta["channels"] = [{} for _ in emission_wavelengths]
        for i, wl in enumerate(emission_wavelengths):
            if i < len(meta["channels"]):
                meta["channels"][i]["emission_wavelength"] = wl

    # Apply defaults for critical missing values (setdefault won't
    # replace an existing key whose value is None, so fix those too).
    _defaults = {
        "na": 1.4,
        "refractive_index": 1.515,
        "microscope_type": "widefield",
        "pixel_size_x": 0.1,
        "pixel_size_y": 0.1,
        "pixel_size_z": 0.3,
    }
    for _k, _v in _defaults.items():
        if meta.get(_k) is None:
            meta[_k] = _v
    meta["sample_refractive_index"] = sample_refractive_index
    meta["n_channels"] = len(images)

    # Ensure channels list
    if "channels" not in meta or not meta["channels"]:
        meta["channels"] = [
            {"emission_wavelength": 520.0} for _ in range(len(images))
        ]
    # Fill in missing emission wavelengths with a default
    for ch in meta["channels"]:
        ch.setdefault("emission_wavelength", 520.0)

    logger.info(
        "Loaded %d channel(s), shape=%s, microscope=%s, NA=%.2f",
        len(images),
        images[0].shape if images else "N/A",
        meta["microscope_type"],
        meta["na"],
    )

    return {"images": images, "metadata": meta}


# ===========================================================================
# Phase 3: PSF Generation from Metadata
# ===========================================================================

def generate_psf(
    metadata: dict[str, Any],
    channel_idx: int = 0,
    *,
    psf_size_xy: Optional[int] = None,
    n_pix_pupil: int = 129,
) -> np.ndarray:
    """Generate a physically accurate PSF from microscopy metadata.

    Uses psf_generator's propagation models. For high-NA objectives (>=0.9),
    a vectorial model is used; otherwise scalar. For confocal microscopes,
    the PSF intensity is squared (standard confocal approximation).

    Parameters
    ----------
    metadata : dict
        Microscopy metadata as returned by load_image()['metadata'].
    channel_idx : int
        Which channel's wavelength to use for PSF computation.
    psf_size_xy : int, optional
        Lateral size of PSF in pixels. If None, auto-calculated as ~4x the
        Airy disk radius to ensure adequate extent.
    n_pix_pupil : int
        Pupil plane discretization (higher = more accurate but slower).

    Returns
    -------
    numpy.ndarray
        Normalized PSF, shape (Z,Y,X) for 3D or (Y,X) for 2D. Sum = 1.
    """
    from vendor.psf_generator import (
        ScalarSphericalPropagator,
        VectorialSphericalPropagator,
    )

    na = metadata["na"]
    ri = metadata["refractive_index"]
    sample_ri = metadata.get("sample_refractive_index", 1.33)
    pix_xy_um = metadata["pixel_size_x"]
    pix_z_um = metadata.get("pixel_size_z", 0.3)
    n_z = metadata.get("size_z", 1)
    is_confocal = metadata.get("microscope_type", "widefield") == "confocal"

    # Channel wavelength
    ch = metadata["channels"][channel_idx]
    wavelength_nm = ch.get("emission_wavelength", 520.0)

    # Convert units: psf_generator uses nm
    pix_xy_nm = pix_xy_um * 1000.0
    pix_z_nm = pix_z_um * 1000.0

    # Auto-calculate PSF lateral size if not specified
    # Airy disk radius ~ 0.61 * lambda / NA (in nm), then convert to pixels
    if psf_size_xy is None:
        airy_radius_nm = 0.61 * wavelength_nm / na
        airy_radius_px = airy_radius_nm / pix_xy_nm
        psf_size_xy = int(max(64, 2 * int(4 * airy_radius_px) + 1))
        # Ensure odd for centering
        if psf_size_xy % 2 == 0:
            psf_size_xy += 1

    # 3D vs 2D
    is_3d = n_z > 1
    n_defocus = max(2 * n_z - 1, 1) if is_3d else 1
    defocus_step = pix_z_nm if is_3d else 0.0

    # psf_generator has device-placement bugs when using CUDA (tensors end
    # up on mixed devices), so always generate on CPU — PSFs are small.
    device = "cpu"

    # Choose propagator based on NA
    PropClass = (
        VectorialSphericalPropagator if na >= 0.9 else ScalarSphericalPropagator
    )

    propagator_kwargs: dict[str, Any] = {
        "n_pix_pupil": n_pix_pupil,
        "n_pix_psf": psf_size_xy,
        "na": na,
        "wavelength": wavelength_nm,
        "pix_size": pix_xy_nm,
        "defocus_step": defocus_step,
        "n_defocus": n_defocus,
        "device": device,
        "gibson_lanni": True,
        "n_i": ri,        # immersion medium RI
        "n_i0": ri,       # design immersion RI
        "n_s": sample_ri, # sample RI
        "n_g": 1.5,       # coverslip RI
        "n_g0": 1.5,      # design coverslip RI
        "t_g": 170e3,     # coverslip thickness (nm)
        "t_g0": 170e3,    # design coverslip thickness (nm)
    }

    logger.info(
        "Generating PSF: %s, NA=%.2f, λ=%d nm, size=%dx%dx%d, device=%s",
        PropClass.__name__,
        na,
        wavelength_nm,
        psf_size_xy,
        psf_size_xy,
        n_defocus,
        device,
    )

    propagator = PropClass(**propagator_kwargs)
    field = propagator.compute_focus_field()

    # Extract intensity |E|^2
    # Vectorial propagator returns shape (n_defocus, 3, ny, nx)
    # Scalar propagator returns shape (n_defocus, ny, nx)
    if field.dim() == 4:
        # Vectorial: sum |E|^2 over the 3 polarization components (axis 1)
        intensity = (torch.abs(field) ** 2).sum(dim=1)
    else:
        # Scalar: shape (n_defocus, ny, nx)
        intensity = torch.abs(field) ** 2

    # Confocal PSF approximation: square the intensity
    if is_confocal:
        intensity = intensity ** 2

    psf = intensity.cpu().numpy().astype(np.float32)

    # Squeeze to 2D if single plane
    if not is_3d:
        psf = psf.squeeze(axis=0)

    # Normalize to sum = 1
    psf_sum = psf.sum()
    if psf_sum > 0:
        psf = psf / psf_sum

    logger.info("PSF generated: shape=%s, sum=%.6f", psf.shape, psf.sum())
    return psf


# ===========================================================================
# Phase 4: Deconvolution Engine
# ===========================================================================

# Available methods and their approximate memory multiplier relative to image
# size (Consideration 2: memory for large volumes)
METHODS = {
    "sdeconv_rl": {"memory_factor": 4, "description": "Iterative RL (PyTorch)"},
    "sdeconv_wiener": {"memory_factor": 3, "description": "Wiener filter with Laplacian regularization"},
    "sdeconv_spitfire": {"memory_factor": 8, "description": "Sparse Hessian variational deconvolution"},
    "pycudadecon_rl_cuda": {"memory_factor": 4, "description": "CUDA-accelerated RL (pycudadecon)"},
    "deconwolf_rl": {"memory_factor": 4, "description": "deconwolf Richardson-Lucy (CLI)"},
    "deconwolf_shb": {"memory_factor": 4, "description": "deconwolf Scaled Heavy Ball (CLI)"},
    "deconvlab2_rl": {"memory_factor": 4, "description": "DeconvolutionLab2 Richardson-Lucy (CLI)"},
    "deconvlab2_rltv": {"memory_factor": 4, "description": "DeconvolutionLab2 RL-Total Variation (CLI)"},
    "redlionfish_rl": {"memory_factor": 4, "description": "RedLionfish RL (OpenCL GPU + CPU fallback)"},
    "skimage_rl": {"memory_factor": 4, "description": "scikit-image Richardson-Lucy (CPU)"},
    "skimage_cucim_rl": {"memory_factor": 4, "description": "cuCIM Richardson-Lucy (CUDA GPU)"},
}


# ---------------------------------------------------------------------------
# XY block-tiling helpers (large image support)
# ---------------------------------------------------------------------------

# Maximum tile dimensions for the "auto" tiling mode.  When the image
# exceeds these limits the auto-calculator splits it into the smallest
# number of blocks that keeps every tile within bounds.
# OpenCL device limits are [1024, 1024, 64].  We use smaller XY values
# because the extracted tile includes overlap margins on each side
# (core + 2*overlap must stay within 1024).
MAX_TILE_Z = 64
MAX_TILE_XY = 512
# Extra pixels extracted around each tile and discarded after
# deconvolution to eliminate FFT edge artifacts (bright stripes).
TILE_MARGIN = 16


def _auto_n_blocks(
    shape: tuple[int, ...],
    max_z: int = MAX_TILE_Z,
    max_xy: int = MAX_TILE_XY,
) -> int:
    """Return the minimum number of XY blocks so each tile fits within limits.

    Only XY is tiled; Z is never split.  If the image already fits
    within *max_z* and *max_xy* the function returns 1 (no tiling).
    If Z alone exceeds *max_z* tiling cannot help, so 1 is returned
    (the backend will have to cope or fail).
    """
    if len(shape) < 3:
        return 1  # 2-D image, no tiling needed
    Z, H, W = shape[:3]
    ny = max(1, -(-H // max_xy))  # ceil division
    nx = max(1, -(-W // max_xy))
    n_blocks = ny * nx
    if n_blocks <= 1:
        return 1
    return n_blocks


def _resolve_n_blocks(
    n_blocks: Union[int, str],
    shape: tuple[int, ...],
) -> int:
    """Resolve *n_blocks* to a concrete integer.

    Accepted values: ``"auto"`` (compute from image shape), or an int
    (0 and 1 both mean no tiling, >1 = explicit block count).
    """
    if isinstance(n_blocks, str):
        if n_blocks.lower() == "auto":
            return _auto_n_blocks(shape)
        raise ValueError(
            f"n_blocks must be 'auto' or an integer, got '{n_blocks}'"
        )
    return max(int(n_blocks), 1)  # 0 → 1 (no tiling)


def _compute_tile_grid(
    shape_yx: tuple[int, int], n_blocks: int,
) -> tuple[int, int]:
    """Return (ny, nx) tile counts that best cover *shape_yx*.

    Tries to keep tiles roughly square by minimising aspect-ratio skew.
    """
    if n_blocks <= 1:
        return (1, 1)
    best = (1, n_blocks)
    best_ratio = float("inf")
    for ny in range(1, n_blocks + 1):
        if n_blocks % ny != 0:
            continue
        nx = n_blocks // ny
        tile_h = shape_yx[0] / ny
        tile_w = shape_yx[1] / nx
        ratio = max(tile_h, tile_w) / max(min(tile_h, tile_w), 1)
        if ratio < best_ratio:
            best_ratio = ratio
            best = (ny, nx)
    return best


def _compute_tile_slices(
    shape_zyx: tuple[int, int, int],
    ny: int,
    nx: int,
    overlap: int,
) -> list[dict]:
    """Return a list of tile descriptors with overlap margins.

    Each descriptor is a dict with:
        'extract'  – (z, y, x) slices to cut from the source image
        'insert'   – (z, y, x) slices where the blended core goes in output
        'core'     – (z, y, x) slices within the *tile* that map to 'insert'
        'blend_y'  – (top, bottom) overlap widths inside the tile
        'blend_x'  – (left, right) overlap widths inside the tile
    """
    _, H, W = shape_zyx
    tile_h = H / ny
    tile_w = W / nx
    tiles = []
    for iy in range(ny):
        y0_core = round(iy * tile_h)
        y1_core = round((iy + 1) * tile_h)
        y0_ext = max(y0_core - overlap, 0)
        y1_ext = min(y1_core + overlap, H)
        ov_top = y0_core - y0_ext
        ov_bot = y1_ext - y1_core
        for ix in range(nx):
            x0_core = round(ix * tile_w)
            x1_core = round((ix + 1) * tile_w)
            x0_ext = max(x0_core - overlap, 0)
            x1_ext = min(x1_core + overlap, W)
            ov_left = x0_core - x0_ext
            ov_right = x1_ext - x1_core
            tiles.append({
                "extract": (slice(None), slice(y0_ext, y1_ext), slice(x0_ext, x1_ext)),
                "insert":  (slice(None), slice(y0_core, y1_core), slice(x0_core, x1_core)),
                "core":    (slice(None), slice(ov_top, ov_top + y1_core - y0_core),
                             slice(ov_left, ov_left + x1_core - x0_core)),
                "blend_y": (ov_top, ov_bot),
                "blend_x": (ov_left, ov_right),
            })
    return tiles


def _blend_tile(tile_result: np.ndarray, desc: dict) -> np.ndarray:
    """Apply linear ramp blending in overlap zones and return core region."""
    ov_top, ov_bot = desc["blend_y"]
    ov_left, ov_right = desc["blend_x"]
    _, core_y, core_x = desc["core"]
    core_h = core_y.stop - core_y.start
    core_w = core_x.stop - core_x.start

    # Build 2-D weight map for the full tile (Z is broadcast)
    tile_h = tile_result.shape[1]
    tile_w = tile_result.shape[2]
    weight = np.ones((tile_h, tile_w), dtype=np.float32)

    # Top overlap ramp
    if ov_top > 0:
        ramp = np.linspace(0, 1, ov_top + 1, dtype=np.float32)[1:]  # exclude 0
        weight[:ov_top, :] *= ramp[:, np.newaxis]
    # Bottom overlap ramp
    if ov_bot > 0:
        ramp = np.linspace(1, 0, ov_bot + 1, dtype=np.float32)[:-1]  # exclude 0
        weight[tile_h - ov_bot:, :] *= ramp[:, np.newaxis]
    # Left overlap ramp
    if ov_left > 0:
        ramp = np.linspace(0, 1, ov_left + 1, dtype=np.float32)[1:]
        weight[:, :ov_left] *= ramp[np.newaxis, :]
    # Right overlap ramp
    if ov_right > 0:
        ramp = np.linspace(1, 0, ov_right + 1, dtype=np.float32)[:-1]
        weight[:, tile_w - ov_right:] *= ramp[np.newaxis, :]

    # Apply weight and extract the region that maps back to the output
    weighted = tile_result * weight[np.newaxis, :, :]

    # The output slot is core-sized; we need the full overlap extent
    # We return (weighted_patch, weight_patch) for the *extended* region
    # so the caller can accumulate numerator/denominator.
    return weighted, weight


def _pad_to_shape(
    arr: np.ndarray, target_shape: tuple[int, ...],
) -> np.ndarray:
    """Centre-pad *arr* with zeros so it matches *target_shape*.

    Used when a backend returns a slightly smaller array than the input
    (e.g. pycudadecon trims to FFT-friendly dimensions).
    """
    if arr.shape == target_shape:
        return arr
    out = np.zeros(target_shape, dtype=arr.dtype)
    offsets = tuple((t - a) // 2 for t, a in zip(target_shape, arr.shape))
    slices = tuple(slice(o, o + min(a, t))
                   for o, a, t in zip(offsets, arr.shape, target_shape))
    src_slices = tuple(slice(0, min(a, t))
                       for a, t in zip(arr.shape, target_shape))
    out[slices] = arr[src_slices]
    return out


def _deconvolve_tiled(
    image: np.ndarray,
    psf: np.ndarray,
    n_blocks: int,
    method: str,
    **kwargs,
) -> np.ndarray:
    """Split *image* into XY tiles, deconvolve each, and blend back."""
    overlap = max(psf.shape[-1], psf.shape[-2]) // 2
    margin = TILE_MARGIN

    ny, nx = _compute_tile_grid(image.shape[1:], n_blocks)
    min_tile_yx = min(image.shape[1] / ny, image.shape[2] / nx)
    if min_tile_yx < max(psf.shape[-2:]):
        # Tiles too small for meaningful deconvolution — reduce n_blocks
        logger.warning(
            "n_blocks=%d produces tiles smaller than PSF; falling back to "
            "n_blocks=1 (no tiling).", n_blocks,
        )
        return deconvolve(image, psf, method=method, n_blocks=1, **kwargs)

    tiles = _compute_tile_slices(image.shape, ny, nx, overlap)
    logger.info(
        "Tiled deconvolution: %d blocks (%d×%d grid), overlap=%d, margin=%d px",
        n_blocks, ny, nx, overlap, margin,
    )

    Z, H, W = image.shape
    # Accumulate with weighted blending
    numerator = np.zeros_like(image, dtype=np.float64)
    denominator = np.zeros(image.shape, dtype=np.float64)

    for idx, desc in enumerate(tiles):
        # --- expand extraction by margin to capture edge artifacts ---
        _, ey, ex = desc["extract"]
        y0_m = max(ey.start - margin, 0)
        y1_m = min(ey.stop + margin, H)
        x0_m = max(ex.start - margin, 0)
        x1_m = min(ex.stop + margin, W)
        ext_expanded = (slice(None), slice(y0_m, y1_m), slice(x0_m, x1_m))
        tile_img = image[ext_expanded].copy()

        logger.info(
            "  Tile %d/%d  shape=%s", idx + 1, len(tiles), tile_img.shape,
        )
        tile_result = deconvolve(
            tile_img, psf, method=method, n_blocks=1, **kwargs,
        )

        # Fix shape if backend returned a different size (e.g. pycudadecon)
        if tile_result.shape != tile_img.shape:
            logger.debug(
                "  Tile result shape %s != input %s; padding to match.",
                tile_result.shape, tile_img.shape,
            )
            tile_result = _pad_to_shape(tile_result, tile_img.shape)

        # Crop margin back to the original extract region
        crop_y0 = ey.start - y0_m
        crop_y1 = crop_y0 + (ey.stop - ey.start)
        crop_x0 = ex.start - x0_m
        crop_x1 = crop_x0 + (ex.stop - ex.start)
        tile_cropped = tile_result[:, crop_y0:crop_y1, crop_x0:crop_x1]

        weighted, weight = _blend_tile(tile_cropped, desc)
        # Place the full extended region back (extract slices)
        ext = desc["extract"]
        numerator[ext] += weighted.astype(np.float64)
        denominator[ext] += weight[np.newaxis, :, :].astype(np.float64)

    # Normalise
    denominator = np.maximum(denominator, 1e-12)
    result = (numerator / denominator).astype(np.float32)
    return np.clip(result, 0, None)


def deconvolve(
    image: np.ndarray,
    psf: np.ndarray,
    method: str = "sdeconv_rl",
    *,
    # Richardson-Lucy parameters
    niter: int = 30,
    # Wiener parameters
    beta: float = 1e-5,
    # Spitfire parameters
    weight: float = 0.6,
    reg: float = 0.995,
    # General parameters (Consideration 3: edge artifacts)
    pad: Union[int, tuple] = 13,
    # Plane-by-plane processing for memory-constrained scenarios
    plane_by_plane: bool = False,
    # pycudadecon-specific parameters
    dzdata: Optional[float] = None,
    dxdata: Optional[float] = None,
    background: Union[int, str] = "auto",
    # Device override for sdeconv backend
    device: Optional[str] = None,
    # RLTV regularization (DeconvolutionLab2)
    tv_lambda: float = 1e-4,
    # XY block-tiling for large images
    n_blocks: Union[int, str] = "auto",
) -> np.ndarray:
    """Deconvolve an image using the specified method and PSF.

    Parameters
    ----------
    image : numpy.ndarray
        Input image, shape (Z,Y,X) for 3D or (Y,X) for 2D.
    psf : numpy.ndarray
        Point spread function, same dimensionality as image.
    method : str
        Deconvolution algorithm:
        - 'sdeconv_rl' (default): Iterative RL via sdeconv (PyTorch).
          Most memory-efficient iterative method. Good default.
        - 'sdeconv_wiener': Wiener filter. Fastest, single-step, but may
          amplify noise. Good for moderate SNR.
        - 'sdeconv_spitfire': Sparse Hessian variational. Best quality for
          sparse structures, but ~8x memory of image size. Avoid for very
          large 3D.
        - 'pycudadecon_rl_cuda': CUDA-accelerated RL via pycudadecon.
          Fastest for large 3D volumes. Requires NVIDIA GPU and pycudadecon.
    niter : int
        Number of iterations for Richardson-Lucy (default: 30).
    beta : float
        Regularization parameter for Wiener filter (default: 1e-5).
    weight : float
        Hessian/sparsity balance for Spitfire (default: 0.6).
    reg : float
        Regularization for Spitfire (default: 0.995).
    pad : int or tuple
        Padding to reduce edge artifacts (default: 13).
    plane_by_plane : bool
        If True, process 3D stacks as independent 2D slices. Reduces memory
        but loses axial deconvolution quality.
    dzdata : float, optional
        Z step size in microns (for pycudadecon). Auto-detected if None.
    dxdata : float, optional
        XY pixel size in microns (for pycudadecon). Auto-detected if None.
    background : int or str
        Background subtraction for pycudadecon (default: 'auto').
    device : str, optional
        Force device for sdeconv backend ('cpu' or 'cuda'). Auto-detected
        if None.
    n_blocks : int or str
        Number of XY tiles for deconvolution.  ``"auto"`` (default)
        calculates the minimum block count so each tile stays within
        MAX_TILE_XY (1024) pixels per side.  ``0`` or ``1`` disables
        tiling.  Values > 1 set an explicit block count.

    Returns
    -------
    numpy.ndarray
        Deconvolved image (float32, non-negative), same shape as input.
    """
    if method not in METHODS:
        raise ValueError(
            f"Unknown method '{method}'. Available: {list(METHODS.keys())}"
        )

    # --- XY block-tiling dispatch (before any backend) ---
    n_blocks = _resolve_n_blocks(n_blocks, image.shape)
    if n_blocks > 1 and image.ndim == 3:
        return _deconvolve_tiled(
            image, psf, n_blocks, method=method,
            niter=niter, beta=beta, weight=weight, reg=reg, pad=pad,
            plane_by_plane=plane_by_plane, dzdata=dzdata, dxdata=dxdata,
            background=background, device=device, tv_lambda=tv_lambda,
        )

    # Crop PSF to image size when it is larger (e.g. n_defocus = 2*nz-1).
    # The extra PSF extent beyond the image edges has no matching data, so
    # centre-cropping is safe and prevents size-mismatch errors in backends.
    if psf.ndim == image.ndim:
        slices = []
        for ax in range(psf.ndim):
            if psf.shape[ax] > image.shape[ax]:
                excess = psf.shape[ax] - image.shape[ax]
                lo = excess // 2
                slices.append(slice(lo, lo + image.shape[ax]))
            else:
                slices.append(slice(None))
        if any(s != slice(None) for s in slices):
            psf = psf[tuple(slices)].copy()
            logger.info("PSF cropped to image size: %s", psf.shape)

    if method == "pycudadecon_rl_cuda":
        return _deconvolve_pycudadecon(
            image, psf, niter=niter, dzdata=dzdata, dxdata=dxdata,
            background=background, plane_by_plane=plane_by_plane,
        )

    if method.startswith("deconwolf_"):
        dw_method = method.split("_", 1)[1]  # "rl" or "shb"
        use_gpu = (device or "").startswith("cuda") or (
            device is None and torch.cuda.is_available()
        )
        return _deconvolve_deconwolf(
            image, psf, niter=niter, method=dw_method, gpu=use_gpu,
        )

    if method.startswith("deconvlab2_"):
        dl2_algo = method.split("_", 1)[1].upper()  # "RL" or "RLTV"
        return _deconvolve_deconvlab2(
            image, psf, algorithm=dl2_algo, niter=niter, tv_lambda=tv_lambda,
        )

    if method == "redlionfish_rl":
        use_gpu = (device or "").startswith("cuda") or (
            device is None  # auto: let RedLionfish try GPU first
        )
        return _deconvolve_redlionfish(image, psf, niter=niter, gpu=use_gpu)

    if method == "skimage_rl":
        return _deconvolve_skimage_rl(image, psf, niter=niter)

    if method == "skimage_cucim_rl":
        return _deconvolve_cucim_rl(image, psf, niter=niter)

    return _deconvolve_sdeconv(
        image, psf, method=method, niter=niter, beta=beta, weight=weight,
        reg=reg, pad=pad, plane_by_plane=plane_by_plane, device=device,
    )


def _deconvolve_sdeconv(
    image: np.ndarray,
    psf: np.ndarray,
    method: str,
    niter: int,
    beta: float,
    weight: float,
    reg: float,
    pad: Union[int, tuple],
    plane_by_plane: bool,
    device: Optional[str] = None,
) -> np.ndarray:
    """Deconvolve using sdeconv (PyTorch backend)."""
    from vendor.sdeconv import SRichardsonLucy, SWiener, Spitfire

    device = torch.device(device if device else _get_device())
    psf_t = torch.from_numpy(psf).float().to(device)

    # Build the deconvolution filter
    if method == "sdeconv_rl":
        deconv_filter = SRichardsonLucy(psf_t, niter=niter, pad=pad)
    elif method == "sdeconv_wiener":
        deconv_filter = SWiener(psf_t, beta=beta, pad=pad)
    elif method == "sdeconv_spitfire":
        delta = 1.0  # Z/XY resolution ratio (could be computed from metadata)
        deconv_filter = Spitfire(psf_t, weight=weight, reg=reg, pad=pad, delta=delta)
    else:
        raise ValueError(f"Unknown sdeconv method: {method}")

    if plane_by_plane and image.ndim == 3:
        # Process each Z-slice independently (Consideration 2)
        # Use a 2D PSF: take the central Z-slice of the 3D PSF
        if psf.ndim == 3:
            center_z = psf.shape[0] // 2
            psf_2d = psf[center_z]
            psf_2d_t = torch.from_numpy(psf_2d).float().to(device)
            if method == "sdeconv_rl":
                deconv_filter = SRichardsonLucy(psf_2d_t, niter=niter, pad=pad)
            elif method == "sdeconv_wiener":
                deconv_filter = SWiener(psf_2d_t, beta=beta, pad=pad)
            elif method == "sdeconv_spitfire":
                deconv_filter = Spitfire(psf_2d_t, weight=weight, reg=reg, pad=pad)

        result_slices = []
        for z in range(image.shape[0]):
            slice_t = torch.from_numpy(image[z]).float().to(device)
            out = deconv_filter(slice_t)
            result_slices.append(out.detach().cpu().numpy())
        result = np.stack(result_slices, axis=0)
    else:
        image_t = torch.from_numpy(image).float().to(device)
        result = deconv_filter(image_t).detach().cpu().numpy()

    # Ensure non-negative
    result = np.clip(result, 0, None).astype(np.float32)
    return result


def _deconvolve_pycudadecon(
    image: np.ndarray,
    psf: np.ndarray,
    niter: int,
    dzdata: Optional[float],
    dxdata: Optional[float],
    background: Union[int, str],
    plane_by_plane: bool = False,
) -> np.ndarray:
    """Deconvolve using pycudadecon (CUDA-accelerated RL)."""
    try:
        from pycudadecon import decon
    except ImportError:
        raise ImportError(
            "pycudadecon is not installed. Install via: "
            "conda install -c conda-forge pycudadecon"
        )

    if plane_by_plane and image.ndim == 3:
        # Process each Z-slice independently using a 2D PSF
        if psf.ndim == 3:
            psf_2d = psf[psf.shape[0] // 2]  # central Z-slice
        else:
            psf_2d = psf
        psf_3d = psf_2d[np.newaxis, :, :]  # (1, Y, X)

        result_slices = []
        for z in range(image.shape[0]):
            slice_3d = image[z][np.newaxis, :, :]  # (1, Y, X)
            out = decon(
                slice_3d, psf_3d,
                dzdata=dzdata or 0.1, dxdata=dxdata or 0.1,
                n_iters=niter, background=background,
            )
            result_slices.append(np.asarray(out, dtype=np.float32)[0])
        return np.clip(np.stack(result_slices, axis=0), 0, None)

    if image.ndim != 3:
        raise ValueError(
            "pycudadecon only supports 3D images. Use a sdeconv method for 2D, "
            "or reshape your 2D image to (1, Y, X)."
        )
    if psf.ndim != 3:
        raise ValueError("PSF must be 3D for pycudadecon.")

    result = decon(
        image,
        psf,
        dzdata=dzdata or 0.1,
        dxdata=dxdata or 0.1,
        n_iters=niter,
        background=background,
    )
    return np.clip(np.asarray(result, dtype=np.float32), 0, None)


# ---------------------------------------------------------------------------
# External CLI backends: deconwolf & DeconvolutionLab2
# ---------------------------------------------------------------------------

def generate_psf_deconwolf(
    metadata: dict[str, Any],
    channel_idx: int = 0,
    *,
    output_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """Generate a PSF using deconwolf's *dw_bw* Born-Wolf model.

    Parameters
    ----------
    metadata : dict
        Microscopy metadata as returned by ``load_image()['metadata']``.
    channel_idx : int
        Channel index whose emission wavelength is used.
    output_path : path, optional
        Where to save the PSF TIFF.  A temp file is used if *None*.

    Returns
    -------
    numpy.ndarray
        Normalised PSF, shape ``(Z, Y, X)``, sum ≈ 1.
    """
    dw_bw = _DW_BW_EXE
    if not dw_bw:
        raise FileNotFoundError(
            "dw_bw not found on PATH. Install deconwolf: "
            "https://github.com/elgw/deconwolf/releases"
        )

    na = metadata["na"]
    ri = metadata["refractive_index"]
    pix_xy_nm = metadata["pixel_size_x"] * 1000.0
    pix_z_nm = metadata.get("pixel_size_z", 0.3) * 1000.0
    n_z = metadata.get("size_z", 1)
    ch = metadata["channels"][channel_idx]
    wavelength_nm = ch.get("emission_wavelength", 520.0)

    # Lateral size: ~4× Airy radius, odd, ≥ 65
    airy_px = 0.61 * wavelength_nm / na / pix_xy_nm
    size_xy = int(max(65, 2 * int(4 * airy_px) + 1))
    if size_xy % 2 == 0:
        size_xy += 1
    # Axial slices: match image depth (odd, ≥ 2*nz - 1)
    n_slice = max(2 * n_z - 1, 1)
    if n_slice % 2 == 0:
        n_slice += 1

    use_temp = output_path is None
    if use_temp:
        fd, output_path = tempfile.mkstemp(suffix=".tif", prefix="dw_psf_")
        os.close(fd)
    output_path = Path(output_path)

    cmd = [
        dw_bw,
        "--resxy", str(int(round(pix_xy_nm))),
        "--resz", str(int(round(pix_z_nm))),
        "--NA", f"{na:.4f}",
        "--ni", f"{ri:.4f}",
        "--lambda", str(int(round(wavelength_nm))),
        "--size", str(size_xy),
        "--nslice", str(n_slice),
        "--overwrite",
        str(output_path),
    ]
    logger.info("dw_bw command: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    psf = tifffile.imread(str(output_path)).astype(np.float32)
    if use_temp:
        output_path.unlink(missing_ok=True)
        Path(str(output_path) + ".log.txt").unlink(missing_ok=True)
    if psf.ndim == 2:
        psf = psf[np.newaxis]
    psf_sum = psf.sum()
    if psf_sum > 0:
        psf /= psf_sum
    return psf


def _deconvolve_deconwolf(
    image: np.ndarray,
    psf: np.ndarray,
    niter: int = 30,
    method: str = "rl",
    gpu: bool = False,
) -> np.ndarray:
    """Deconvolve a single 3-D volume using the *dw* CLI.

    *image* and *psf* are written to temp TIFFs, ``dw`` is invoked, and the
    result is read back as a numpy array.
    """
    dw = _DW_EXE
    if not dw:
        raise FileNotFoundError(
            "dw not found on PATH. Install deconwolf: "
            "https://github.com/elgw/deconwolf/releases"
        )

    if image.ndim != 3 or psf.ndim != 3:
        raise ValueError("deconwolf requires 3-D image and PSF arrays.")

    tmp = tempfile.mkdtemp(prefix="dw_")
    try:
        img_path = os.path.join(tmp, "image.tif")
        psf_path = os.path.join(tmp, "psf.tif")
        out_path = os.path.join(tmp, f"{method}_image.tif")

        tifffile.imwrite(img_path, image.astype(np.float32))
        tifffile.imwrite(psf_path, psf.astype(np.float32))

        cmd = [
            dw,
            "--iter", str(niter),
            "--method", method,
            "--overwrite",
        ]
        if gpu:
            cmd.append("--gpu")
        cmd += [img_path, psf_path]
        logger.info("dw command: %s", " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True)

        # Fallback: if --gpu failed due to OpenCL, retry without it
        if proc.returncode != 0 and gpu and (
            "cl_util.c" in proc.stderr or proc.returncode < 0
        ):
            logger.warning(
                "dw --gpu failed (OpenCL/signal error, rc=%d), "
                "retrying without --gpu",
                proc.returncode,
            )
            cmd = [c for c in cmd if c != "--gpu"]
            logger.info("dw command (cpu fallback): %s", " ".join(cmd))
            proc = subprocess.run(cmd, capture_output=True, text=True)

        if proc.returncode != 0:
            raise RuntimeError(f"dw failed (rc={proc.returncode}): {proc.stderr}")

        if not os.path.exists(out_path):
            # fallback: find any output that isn't our input files
            inputs = {"image.tif", "psf.tif"}
            candidates = [f for f in os.listdir(tmp)
                          if f.endswith(".tif") and f not in inputs]
            if candidates:
                out_path = os.path.join(tmp, candidates[0])
            else:
                raise FileNotFoundError(
                    f"deconwolf produced no output in {tmp}: {os.listdir(tmp)}"
                )

        result = tifffile.imread(out_path).astype(np.float32)
        return np.clip(result, 0, None)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _deconvolve_deconvlab2(
    image: np.ndarray,
    psf: np.ndarray,
    algorithm: str = "RL",
    niter: int = 30,
    tv_lambda: float = 1e-4,
) -> np.ndarray:
    """Deconvolve a single 3-D volume using DeconvolutionLab2 CLI.

    Writes temp TIFFs, invokes DL2 via ``java -cp ...``, reads the result.
    """
    dl2_jar = _DECONVLAB2_JAR
    ij_jar = _IJ_JAR
    if not dl2_jar.exists():
        raise FileNotFoundError(f"DeconvolutionLab2 JAR not found: {dl2_jar}")
    if not ij_jar.exists():
        raise FileNotFoundError(f"ImageJ JAR not found: {ij_jar}")
    java = shutil.which("java")
    if not java:
        raise FileNotFoundError("java not found on PATH.")

    if image.ndim != 3 or psf.ndim != 3:
        raise ValueError("DeconvolutionLab2 requires 3-D image and PSF arrays.")

    tmp = tempfile.mkdtemp(prefix="dl2_")
    try:
        img_path = os.path.join(tmp, "image.tif")
        psf_path = os.path.join(tmp, "psf.tif")
        result_name = "result"

        tifffile.imwrite(img_path, image.astype(np.float32))
        tifffile.imwrite(psf_path, psf.astype(np.float32))

        # Build algorithm spec
        if algorithm == "RLTV":
            algo_spec = f"RLTV {niter} {tv_lambda}"
        else:
            algo_spec = f"{algorithm} {niter}"

        sep = ";" if sys.platform == "win32" else ":"
        cp = f"{dl2_jar}{sep}{ij_jar}"
        cmd = [
            java, "-cp", cp, "DeconvolutionLab2", "Run",
            "-image", "file", img_path,
            "-psf", "file", psf_path,
            "-algorithm", *algo_spec.split(),
            "-out", "stack", "noshow", result_name,
            "-path", tmp,
            "-monitor", "console",
        ]
        logger.info("DL2 command: %s", " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"DeconvolutionLab2 failed (rc={proc.returncode}): {proc.stderr}"
            )

        # DL2 saves output as <result_name>.tif in -path directory
        out_path = os.path.join(tmp, f"{result_name}.tif")
        if not os.path.exists(out_path):
            # Try other common suffixes
            for ext in (".tiff", ".ome.tif"):
                alt = os.path.join(tmp, result_name + ext)
                if os.path.exists(alt):
                    out_path = alt
                    break
            else:
                tifs = [f for f in os.listdir(tmp) if f.endswith((".tif", ".tiff"))]
                raise FileNotFoundError(
                    f"DL2 output not found. Files in {tmp}: {tifs}\n"
                    f"stdout: {proc.stdout[-500:]}\nstderr: {proc.stderr[-500:]}"
                )

        result = tifffile.imread(out_path).astype(np.float32)
        return np.clip(result, 0, None)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _deconvolve_redlionfish(
    image: np.ndarray,
    psf: np.ndarray,
    niter: int = 30,
    gpu: bool = True,
) -> np.ndarray:
    """Deconvolve a single 3-D volume using RedLionfish.

    Uses OpenCL GPU acceleration with automatic CPU fallback.
    """
    try:
        import RedLionfishDeconv as rl
    except ImportError as exc:
        raise ImportError(
            "RedLionfish not installed.  pip install redlionfish"
        ) from exc

    if image.ndim != 3 or psf.ndim != 3:
        raise ValueError("RedLionfish requires 3-D image and PSF arrays.")

    method = "gpu" if gpu else "cpu"
    result = rl.doRLDeconvolutionFromNpArrays(
        image.astype(np.float32),
        psf.astype(np.float32),
        niter=niter,
        method=method,
        resAsUint8=False,
    )
    return np.clip(result.astype(np.float32), 0, None)


def _deconvolve_skimage_rl(
    image: np.ndarray,
    psf: np.ndarray,
    niter: int = 30,
) -> np.ndarray:
    """Deconvolve using scikit-image Richardson-Lucy (CPU).

    Handles 2-D and 3-D images natively.
    Pads with reflect boundary to suppress edge artifacts from FFT zero-padding.
    """
    from skimage.restoration import richardson_lucy

    # Pad each spatial axis by half the PSF extent (reflect) to avoid
    # bright-edge artifacts caused by the zero-padding inside fftconvolve.
    pad_widths = tuple((s // 2, s // 2) for s in psf.shape)
    image_padded = np.pad(image.astype(np.float32), pad_widths, mode="reflect")

    result_padded = richardson_lucy(
        image_padded, psf.astype(np.float32),
        num_iter=niter, clip=False, filter_epsilon=None,
    )
    # Crop back to original size
    slices = tuple(slice(p[0], p[0] + s) for p, s in zip(pad_widths, image.shape))
    result = result_padded[slices]
    return np.clip(result.astype(np.float32), 0, None)


def _deconvolve_cucim_rl(
    image: np.ndarray,
    psf: np.ndarray,
    niter: int = 30,
) -> np.ndarray:
    """Deconvolve using cuCIM Richardson-Lucy (CUDA GPU).

    Drop-in GPU replacement for scikit-image RL; requires cupy + cucim.
    """
    try:
        import cupy as cp
        from cucim.skimage.restoration import richardson_lucy
    except ImportError as exc:
        raise ImportError(
            "cuCIM not installed.  pip install cucim-cu12 cupy-cuda12x"
        ) from exc

    image_gpu = cp.asarray(image.astype(np.float32))
    psf_gpu = cp.asarray(psf.astype(np.float32))
    result_gpu = richardson_lucy(
        image_gpu, psf_gpu, num_iter=niter, clip=False, filter_epsilon=None,
    )
    result = cp.asnumpy(result_gpu).astype(np.float32)
    return np.clip(result, 0, None)


# ===========================================================================
# Phase 5: High-level convenience function
# ===========================================================================

def deconvolve_image(
    path: Union[str, Path],
    method: str = "sdeconv_rl",
    channels: Optional[Sequence[int]] = None,
    *,
    # Metadata overrides (Consideration 1)
    na: Optional[float] = None,
    refractive_index: Optional[float] = None,
    microscope_type: Optional[str] = None,
    pixel_size_xy: Optional[float] = None,
    pixel_size_z: Optional[float] = None,
    emission_wavelengths: Optional[list[float]] = None,
    sample_refractive_index: float = 1.33,
    # PSF options
    psf_size_xy: Optional[int] = None,
    n_pix_pupil: int = 129,
    # Deconvolution options
    niter: int = 30,
    beta: float = 1e-5,
    weight: float = 0.6,
    reg: float = 0.995,
    pad: Union[int, tuple] = 13,
    plane_by_plane: bool = False,
    background: Union[int, str] = "auto",
    device: Optional[str] = None,
    tv_lambda: float = 1e-4,
    n_blocks: Union[int, str] = "auto",
) -> dict[str, Any]:
    """Load, generate PSFs, and deconvolve all channels of an OME-TIFF image.

    This is the main entry point for end-to-end deconvolution.

    Parameters
    ----------
    path : str or Path
        Path to OME-TIFF file or companion .ome file.
    method : str
        Deconvolution algorithm (see deconvolve() for options).
    channels : sequence of int, optional
        Which channels to process. None = all channels.
    na, refractive_index, microscope_type, pixel_size_xy, pixel_size_z,
    emission_wavelengths, sample_refractive_index
        Metadata overrides (see load_image() for details).
    psf_size_xy, n_pix_pupil
        PSF generation options (see generate_psf() for details).
    niter, beta, weight, reg, pad, plane_by_plane, background
        Deconvolution options (see deconvolve() for details).

    Returns
    -------
    dict with keys:
        'channels': list of deconvolved numpy arrays
        'psfs': list of PSF numpy arrays used
        'metadata': microscopy metadata dict
        'source_channels': list of original (unprocessed) numpy arrays
    """
    data = load_image(
        path,
        na=na,
        refractive_index=refractive_index,
        microscope_type=microscope_type,
        pixel_size_xy=pixel_size_xy,
        pixel_size_z=pixel_size_z,
        emission_wavelengths=emission_wavelengths,
        sample_refractive_index=sample_refractive_index,
    )

    images = data["images"]
    metadata = data["metadata"]

    if channels is None:
        channels = list(range(len(images)))

    # Prepare pycudadecon pixel sizes if needed
    dzdata = metadata.get("pixel_size_z")
    dxdata = metadata.get("pixel_size_x")

    results: list[np.ndarray] = []
    psfs: list[np.ndarray] = []

    for ch_idx in channels:
        if ch_idx >= len(images):
            raise IndexError(
                f"Channel {ch_idx} requested but only {len(images)} available"
            )

        logger.info("Processing channel %d / %d ...", ch_idx + 1, len(channels))

        # Generate PSF for this channel's wavelength
        psf = generate_psf(
            metadata, channel_idx=ch_idx,
            psf_size_xy=psf_size_xy, n_pix_pupil=n_pix_pupil,
        )
        psfs.append(psf)

        # Match PSF dimensionality to image
        img = images[ch_idx]
        if img.ndim == 2 and psf.ndim == 3:
            psf = psf[psf.shape[0] // 2]  # Take central slice
        elif img.ndim == 3 and psf.ndim == 2:
            # Expand 2D PSF into 3D (single-plane)
            psf = psf[np.newaxis, :, :]

        # Deconvolve
        result = deconvolve(
            img, psf, method=method,
            niter=niter, beta=beta, weight=weight, reg=reg, pad=pad,
            plane_by_plane=plane_by_plane, dzdata=dzdata, dxdata=dxdata,
            background=background, device=device, tv_lambda=tv_lambda,
            n_blocks=n_blocks,
        )
        results.append(result)

    return {
        "channels": results,
        "psfs": psfs,
        "metadata": metadata,
        "source_channels": [images[i] for i in channels],
    }


# ===========================================================================
# Phase 6: Save results
# ===========================================================================

# -- Emission‑wavelength → RGB colour mapping for MIP PNGs ----------------

# Fallback palette when no emission wavelength is available
_FALLBACK_COLORS = [
    (0, 255, 0),      # Green
    (255, 0, 255),     # Magenta
    (0, 255, 255),     # Cyan
    (255, 0, 0),       # Red
    (0, 0, 255),       # Blue
    (255, 255, 0),     # Yellow
]


def _emission_to_rgb(wavelength_nm: Optional[float]) -> tuple[int, int, int]:
    """Map an emission wavelength (nm) to an approximate RGB colour.

    Uses a piecewise‑linear visible‑spectrum approximation.  Returns a
    fallback green if the wavelength is *None* or outside 380‑780 nm.
    """
    if wavelength_nm is None:
        return (255, 255, 255)  # white → caller should use fallback palette
    wl = wavelength_nm
    r = g = b = 0.0
    if 380 <= wl < 440:
        r = -(wl - 440) / (440 - 380)
        b = 1.0
    elif 440 <= wl < 490:
        g = (wl - 440) / (490 - 440)
        b = 1.0
    elif 490 <= wl < 510:
        g = 1.0
        b = -(wl - 510) / (510 - 490)
    elif 510 <= wl < 580:
        r = (wl - 510) / (580 - 510)
        g = 1.0
    elif 580 <= wl < 645:
        r = 1.0
        g = -(wl - 645) / (645 - 580)
    elif 645 <= wl <= 780:
        r = 1.0
    else:
        return (255, 255, 255)  # outside visible → white
    return (int(r * 255), int(g * 255), int(b * 255))


def _channel_color(metadata: dict[str, Any], ch_idx: int) -> tuple[int, int, int]:
    """Determine the display colour for a channel.

    Priority:
      1. Emission wavelength → spectral RGB
      2. Fallback palette (Green, Magenta, Cyan, Red, Blue, Yellow, …)
    """
    channels = metadata.get("channels", [])
    em = None
    if ch_idx < len(channels):
        em = channels[ch_idx].get("emission_wavelength")
    rgb = _emission_to_rgb(em)
    if rgb == (255, 255, 255):
        # Use fallback palette
        rgb = _FALLBACK_COLORS[ch_idx % len(_FALLBACK_COLORS)]
    return rgb


def save_mip_png(
    mip_data: np.ndarray,
    png_path: Union[str, Path],
    metadata: dict[str, Any],
    *,
    channel_indices: Optional[Sequence[int]] = None,
) -> Path:
    """Save a MIP array as a false‑colour PNG image.

    Parameters
    ----------
    mip_data : np.ndarray
        MIP image, shape ``(C, Y, X)`` or ``(Y, X)`` for single channel.
    png_path : str or Path
        Output path for the PNG file.
    metadata : dict
        Microscopy metadata (must contain ``channels`` with emission info).
    channel_indices : sequence of int, optional
        Which metadata channel indices map to each slice of *mip_data*.
        Defaults to ``[0, 1, …, C-1]``.

    Returns
    -------
    Path to saved PNG.
    """
    from PIL import Image

    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)

    # Normalise to (C, Y, X)
    if mip_data.ndim == 2:
        mip_data = mip_data[np.newaxis]
    n_ch, h, w = mip_data.shape

    if channel_indices is None:
        channel_indices = list(range(n_ch))

    # Build RGB canvas by additive blending of coloured channels
    canvas = np.zeros((h, w, 3), dtype=np.float64)
    for i in range(n_ch):
        ch_img = mip_data[i].astype(np.float64)
        lo, hi = ch_img.min(), ch_img.max()
        if hi > lo:
            ch_img = (ch_img - lo) / (hi - lo)
        else:
            ch_img = np.zeros_like(ch_img)
        rgb = _channel_color(metadata, channel_indices[i])
        for c_idx in range(3):
            canvas[:, :, c_idx] += ch_img * (rgb[c_idx] / 255.0)

    # Clip and convert to uint8
    canvas = np.clip(canvas, 0, 1)
    canvas = (canvas * 255).astype(np.uint8)

    img = Image.fromarray(canvas, mode="RGB")
    img.save(str(png_path))
    logger.info("Saved colour MIP PNG to %s", png_path)
    return png_path


def save_result(
    result: dict[str, Any],
    output_path: Union[str, Path],
    *,
    compress: bool = True,
    mip_only: bool = False,
) -> Path:
    """Save deconvolved images as OME-TIFF, preserving metadata.

    Parameters
    ----------
    result : dict
        Output from deconvolve_image().
    output_path : str or Path
        Output file path (.ome.tiff).
    compress : bool
        Whether to apply zlib compression (default: True).
    mip_only : bool
        If True, skip writing OME-TIFF files and only save MIP PNGs.
        Useful for benchmark mode where TIFFs are not needed.

    Returns
    -------
    Path to the saved file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = result["metadata"]
    channels_data = result["channels"]

    # Stack channels into CZYX or CYX
    if channels_data[0].ndim == 3:
        # 3D: stack to (C, Z, Y, X)
        stack = np.stack(channels_data, axis=0)
        axes = "CZYX"
    else:
        # 2D: stack to (C, Y, X)
        stack = np.stack(channels_data, axis=0)
        axes = "CYX"

    px_x = metadata.get("pixel_size_x")
    px_y = metadata.get("pixel_size_y")
    px_z = metadata.get("pixel_size_z")

    resolution = None
    if px_x and px_y:
        # tifffile resolution is in pixels per unit (inverse of pixel size)
        resolution = (1.0 / px_x, 1.0 / px_y)
    resolution_unit = 1  # No standard unit in basic TIFF; OME handles it

    if not mip_only:
        tifffile.imwrite(
            str(output_path),
            stack.astype(np.float32),
            ome=True,
            photometric="minisblack",
            compression="zlib" if compress else None,
            resolution=resolution,
            resolutionunit=resolution_unit,
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
                        f"Ch{i}" for i in range(len(channels_data))
                    ]
                },
            },
        )
        logger.info("Saved deconvolved result to %s", output_path)

    # Save maximum intensity projection for 3D results
    if axes == "CZYX":
        mip = stack.max(axis=1)  # Project along Z → (C, Y, X)
        mip_path = output_path.parent / ("mip_" + output_path.name)
        tifffile.imwrite(
            str(mip_path),
            mip.astype(np.float32),
            ome=True,
            photometric="minisblack",
            compression="zlib" if compress else None,
            resolution=resolution,
            resolutionunit=resolution_unit,
            metadata={
                "axes": "CYX",
                "PhysicalSizeX": px_x,
                "PhysicalSizeY": px_y,
                "PhysicalSizeXUnit": "µm",
                "PhysicalSizeYUnit": "µm",
                "Channel": {
                    "Name": [
                        f"Ch{i}" for i in range(len(channels_data))
                    ]
                },
            },
        )
        logger.info("Saved MIP to %s", mip_path)

        # Save colour PNG of MIP (always — needed for montage)
        mip_png = mip_path.with_suffix(".png")
        save_mip_png(mip, mip_png, metadata)

    # Save maximum intensity projection of the source image for 3D data
    source_channels = result.get("source_channels")
    if source_channels and axes == "CZYX":
        src_mip_path = output_path.parent / "mip_source.ome.tiff"
        src_stack = np.stack(source_channels, axis=0)  # (C, Z, Y, X)
        src_mip = src_stack.max(axis=1)  # Project along Z → (C, Y, X)
        tifffile.imwrite(
            str(src_mip_path),
            src_mip.astype(np.float32),
            ome=True,
            photometric="minisblack",
            compression="zlib" if compress else None,
            resolution=resolution,
            resolutionunit=resolution_unit,
            metadata={
                "axes": "CYX",
                "PhysicalSizeX": px_x,
                "PhysicalSizeY": px_y,
                "PhysicalSizeXUnit": "µm",
                "PhysicalSizeYUnit": "µm",
                "Channel": {
                    "Name": [
                        f"Ch{i}" for i in range(len(source_channels))
                    ]
                },
            },
        )
        logger.info("Saved source MIP to %s", src_mip_path)

        # Save colour PNG of source MIP (always — needed for montage)
        src_mip_png = src_mip_path.with_suffix(".png")
        save_mip_png(src_mip, src_mip_png, metadata)

    return output_path
