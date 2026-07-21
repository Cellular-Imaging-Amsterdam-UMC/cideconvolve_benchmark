"""CI Deconvolve — GPU-accelerated Richardson-Lucy with SHB momentum & PSF generation.

This module provides three public functions:

* ``ci_rl_deconvolve``  – Scaled Heavy Ball (SHB) accelerated Richardson-Lucy
  deconvolution with optional Total Variation regularisation, Bertero boundary
  weights, and I-divergence convergence monitoring.  All heavy lifting runs on
  GPU via PyTorch when a CUDA device is available; CPU fallback is automatic.

* ``ci_sparse_hessian_deconvolve`` – Variational deconvolution with a
  sparse-Hessian / SPITFIRE-style regulariser, using the same preprocessing,
  FFT setup, and GPU/CPU execution model as the RL-family methods.

* ``ci_generate_psf``  – Physically accurate PSF generation using the vectorial
  Richards-Wolf model (high NA) or scalar Kirchhoff model (lower NA), with
  Gibson-Lanni refractive-index mismatch correction and optional sub-pixel
  integration.

References
----------
[1] Wang & Miller 2014, IEEE TIP 23(2):848-854  (SHB acceleration)
[2] Bertero & Boccacci 2005, A&A 437:369-374     (boundary weights)
[3] Dey et al. 2006, Microsc. Res. Tech. 69:260  (RLTV)
[4] Richards & Wolf 1959, Proc. R. Soc. A 253    (vectorial PSF)
[5] Gibson & Lanni 1991, JOSA A 8(10):1601       (RI mismatch)
"""

from __future__ import annotations

import logging
import math
import os
import hashlib
import threading
from collections import OrderedDict
from functools import wraps
from typing import Any, Callable, Optional, Union

import numpy as np
import torch
from torch.special import bessel_j0, bessel_j1

log = logging.getLogger(__name__)

CONCRETE_START_MODES = (
    "flat",
    "percentile_flat",
    "observed",
    "observed_bgsub",
    "lowpass",
    "lowpass_bgsub",
    "hybrid",
)
START_MODES = ("auto",) + CONCRETE_START_MODES

SNR_AUTO = "auto"

BACKEND_AUTO = "auto"
BACKEND_OPTIMIZED_CUDA = "optimized_cuda"
BACKEND_PYTORCH_CUDA = "pytorch_cuda"
BACKEND_CPU = "cpu"
BACKENDS = (
    BACKEND_AUTO,
    BACKEND_OPTIMIZED_CUDA,
    BACKEND_PYTORCH_CUDA,
    BACKEND_CPU,
)

# ---------------------------------------------------------------------------
# Helpers — device / dtype
# ---------------------------------------------------------------------------

def _pick_device(device: Optional[str]) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _release_cuda_cache() -> None:
    """Release PyTorch's cached CUDA blocks after large independent tiles."""
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception:
        pass


def _release_cuda_cache_after(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        finally:
            _release_cuda_cache()
    return wrapper


def _pick_dtype(dev: torch.device) -> torch.dtype:
    if dev.type == "cpu" and os.environ.get("CIDE_CONVOLVE_CPU_FLOAT64") == "1":
        return torch.float64
    return torch.float32


def _resolve_backend(
    backend: Optional[str], device: Optional[str],
) -> tuple[torch.device, Any | None, str]:
    """Resolve user-facing backend selection to device and optional CUDA ops."""
    requested = str(backend or BACKEND_AUTO).strip().lower().replace(" ", "_")
    aliases = {
        "optimized": BACKEND_OPTIMIZED_CUDA,
        "cuda": BACKEND_PYTORCH_CUDA,
        "pytorch": BACKEND_PYTORCH_CUDA,
    }
    requested = aliases.get(requested, requested)
    if requested not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    if requested == BACKEND_CPU:
        return torch.device("cpu"), None, BACKEND_CPU
    if requested in (BACKEND_OPTIMIZED_CUDA, BACKEND_PYTORCH_CUDA):
        if not torch.cuda.is_available():
            raise RuntimeError(f"{requested.replace('_', ' ')} requires a CUDA-capable PyTorch runtime")
        dev = torch.device(device or "cuda")
        if dev.type != "cuda":
            raise ValueError(f"backend {requested!r} cannot use device {device!r}")
        if requested == BACKEND_PYTORCH_CUDA:
            return dev, None, requested
        from optimized_cuda import load_optimized_extension

        return dev, load_optimized_extension(required=True), requested

    dev = _pick_device(device)
    if dev.type != "cuda":
        return dev, None, BACKEND_CPU
    from optimized_cuda import load_optimized_extension

    ops = load_optimized_extension(required=False)
    return dev, ops, BACKEND_OPTIMIZED_CUDA if ops is not None else BACKEND_PYTORCH_CUDA


def _next_smooth(value: int) -> int:
    candidate = max(1, int(value))
    while True:
        remainder = candidate
        for prime in (2, 3, 5, 7):
            while remainder % prime == 0:
                remainder //= prime
        if remainder == 1:
            return candidate
        candidate += 1


_OPTIMIZED_CONTEXTS: "OrderedDict[tuple[Any, ...], dict[str, Any]]" = OrderedDict()
_OPTIMIZED_CONTEXTS_LOCK = threading.Lock()
# Tiled execution is sequential. Grouping equal geometries and retaining one
# direct-cuFFT arena matches the winning benchmark and avoids keeping four
# multi-gigabyte edge/centre arenas resident simultaneously.
_MAX_OPTIMIZED_CONTEXTS = 1


def _psf_cache_digest(psf: torch.Tensor) -> str:
    array = psf.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    return hashlib.blake2b(array.view(np.uint8), digest_size=12).hexdigest()


def _optimized_context(
    ops: Any,
    psf: torch.Tensor,
    image_shape: tuple[int, ...],
    minimum_shape: tuple[int, ...],
) -> dict[str, Any]:
    """Return cached FP32 OTF, weights, in-place arena, plan and workspace."""
    if len(minimum_shape) != 3:
        raise ValueError("the optimized direct-cuFFT backend currently requires a 3D work volume")
    work_shape = tuple(_next_smooth(v) for v in minimum_shape)
    device_index = psf.device.index if psf.device.index is not None else torch.cuda.current_device()
    stream_id = int(torch.cuda.current_stream(device_index).cuda_stream)
    key = (
        device_index,
        threading.get_ident(),
        stream_id,
        tuple(image_shape),
        work_shape,
        _psf_cache_digest(psf),
    )
    evicted = False
    with _OPTIMIZED_CONTEXTS_LOCK:
        context = _OPTIMIZED_CONTEXTS.get(key)
        if context is not None:
            _OPTIMIZED_CONTEXTS.move_to_end(key)
            return context
        # Drop the previous geometry before allocating the next large arena.
        # Local references keep an in-flight context alive in the unlikely
        # event of concurrent callers.
        if len(_OPTIMIZED_CONTEXTS) >= _MAX_OPTIMIZED_CONTEXTS:
            _OPTIMIZED_CONTEXTS.clear()
            evicted = True

    if evicted:
        _release_cuda_cache()

    otf, otf_conj = _prepare_otf(psf, work_shape)
    weights = _bertero_weights(otf, otf_conj, image_shape, work_shape).contiguous()
    with torch.cuda.device(device_index):
        plan = ops.DirectFFTPlan(*work_shape)
        pitch = int(plan.physical_x)
        storage = torch.empty(
            (work_shape[0], work_shape[1], pitch), dtype=torch.float32, device=psf.device,
        )
        frequency = storage.view(torch.complex64).view(
            work_shape[0], work_shape[1], work_shape[2] // 2 + 1,
        )
        workspace = torch.empty(
            max(int(plan.workspace_bytes), 1), dtype=torch.uint8, device=psf.device,
        )
    context = {
        "otf": otf.resolve_conj().contiguous(),
        "weights": weights,
        "plan": plan,
        "pitch": pitch,
        "storage": storage,
        "frequency": frequency,
        "workspace": workspace,
        "work_shape": work_shape,
        "normalization": float(math.prod(work_shape)),
        "workspace_bytes": int(plan.workspace_bytes),
    }
    with _OPTIMIZED_CONTEXTS_LOCK:
        _OPTIMIZED_CONTEXTS[key] = context
        _OPTIMIZED_CONTEXTS.move_to_end(key)
        while len(_OPTIMIZED_CONTEXTS) > _MAX_OPTIMIZED_CONTEXTS:
            _OPTIMIZED_CONTEXTS.popitem(last=False)
    return context


def clear_optimized_context_cache() -> None:
    with _OPTIMIZED_CONTEXTS_LOCK:
        _OPTIMIZED_CONTEXTS.clear()
    _release_cuda_cache()


def _numpy_dtype_for_torch(dtype: torch.dtype) -> np.dtype:
    return np.dtype(np.float64 if dtype == torch.float64 else np.float32)


def _as_torch_input_array(arr: np.ndarray, dtype: torch.dtype) -> np.ndarray:
    """Return a contiguous NumPy array matching the torch compute dtype."""
    target_dtype = _numpy_dtype_for_torch(dtype)
    arr_np = np.asarray(arr)
    if arr_np.dtype != target_dtype or not arr_np.flags.c_contiguous:
        return np.ascontiguousarray(arr_np, dtype=target_dtype)
    return arr_np


def _to_tensor(arr: np.ndarray, dev: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(_as_torch_input_array(arr, dtype), dtype=dtype, device=dev)


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()

# ===================================================================
#  PART 1 — Richardson-Lucy deconvolution engine
# ===================================================================

# ---------------------------------------------------------------------------
# FFT helpers (rfftn for real-valued data — halves memory vs fftn)
# ---------------------------------------------------------------------------

def _rfft(x: torch.Tensor) -> torch.Tensor:
    return torch.fft.rfftn(x)


def _irfft(X: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    return torch.fft.irfftn(X, s=shape)

# ---------------------------------------------------------------------------
# PSF → OTF preparation
# ---------------------------------------------------------------------------

def _prepare_otf(
    psf: torch.Tensor,
    work_shape: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad & circularly-shift PSF, return (OTF, conj(OTF))."""
    # Normalise
    psf = psf / psf.sum()

    # Zero-pad PSF into the work volume
    padded = torch.zeros(work_shape, dtype=psf.dtype, device=psf.device)
    slices = tuple(slice(0, s) for s in psf.shape)
    padded[slices] = psf

    # Circular shift so PSF centre sits at (0, 0, ..., 0)
    shifts = [-(s // 2) for s in psf.shape]
    padded = torch.roll(padded, shifts=shifts, dims=list(range(psf.ndim)))

    otf = _rfft(padded)
    otf_conj = torch.conj(otf)
    return otf, otf_conj


def _origin_support_slices(shape: tuple[int, ...]) -> tuple[slice, ...]:
    """Return slices that place a support region at the origin of a work volume."""
    return tuple(slice(0, s) for s in shape)

# ---------------------------------------------------------------------------
# Bertero boundary weights  (Bertero & Boccacci 2005)
# ---------------------------------------------------------------------------

def _bertero_weights(
    otf: torch.Tensor,
    otf_conj: torch.Tensor,
    image_shape: tuple[int, ...],
    work_shape: tuple[int, ...],
    sigma: float = 0.01,
) -> torch.Tensor:
    """Compute boundary correction weights W = 1 / H^T(𝟏_M).

    A flat image of ones (image-sized, the data support 𝟏_M) is correlated
    with the PSF (H^T) in the work domain. Where the result exceeds *sigma*
    it is inverted to give W; elsewhere W is zero.  This corrects for
    partial PSF overlap near the data boundary (Bertero & Boccacci 2005).
    """
    return _bertero_weights_for_support(
        otf,
        otf_conj,
        _origin_support_slices(image_shape),
        work_shape,
        sigma=sigma,
    )


def _bertero_weights_for_support(
    otf: torch.Tensor,
    otf_conj: torch.Tensor,
    support_slices: tuple[slice, ...],
    work_shape: tuple[int, ...],
    sigma: float = 0.01,
) -> torch.Tensor:
    """Compute boundary weights for an arbitrary observed support region."""
    ones = torch.zeros(work_shape, dtype=otf.real.dtype, device=otf.device)
    ones[support_slices] = 1.0

    ones_fft = _rfft(ones)
    # H^T(1_M) = IFFT(conj(H) * FFT(1_M))  — Bertero & Boccacci 2005
    denom_fft = ones_fft * otf_conj
    denom = _irfft(denom_fft, work_shape)

    W = torch.zeros_like(denom)
    mask = denom > sigma
    W[mask] = 1.0 / denom[mask]
    return W

# ---------------------------------------------------------------------------
# I-divergence  (Csiszár 1991)
# ---------------------------------------------------------------------------

def _i_divergence(observed: torch.Tensor, estimated: torch.Tensor, eps: float = 1e-12) -> float:
    """Compute mean I-divergence (generalised KL) for Poisson data."""
    est_safe = estimated.clamp(min=eps)
    obs_safe = observed.clamp(min=eps)
    div = obs_safe * torch.log(obs_safe / est_safe) - obs_safe + est_safe
    return float(div.mean())

# ---------------------------------------------------------------------------
# Total-Variation multiplicative penalty  (Dey et al. 2006 / DL2 RLTV)
# ---------------------------------------------------------------------------

def _axis_scales(
    ndim: int,
    pixel_size_xy: Optional[float],
    pixel_size_z: Optional[float],
) -> tuple[float, ...]:
    """Return per-axis relative derivative scaling for anisotropic voxels."""
    xy = max(float(pixel_size_xy), 1e-12) if pixel_size_xy is not None else 1.0
    if ndim == 2:
        return (1.0, 1.0)
    z = max(float(pixel_size_z), 1e-12) if pixel_size_z is not None else xy
    z_scale = xy / z
    return (z_scale, 1.0, 1.0)


def _tv_penalty(
    x: torch.Tensor,
    tv_lambda: float,
    axis_scales: tuple[float, ...],
) -> torch.Tensor:
    """Multiplicative TV correction factor.

    Returns a tensor of the same shape as *x* such that
    ``x_new = x * _tv_penalty(x, λ)`` applies one TV step.
    """
    ndim = x.ndim
    eps = 1e-8

    # Forward differences with zero-padded boundary
    grads = []
    for d in range(ndim):
        g = torch.zeros_like(x)
        slc_src = [slice(None)] * ndim
        slc_dst = [slice(None)] * ndim
        slc_src[d] = slice(0, -1)
        slc_dst[d] = slice(1, None)
        g[tuple(slc_dst)] = (
            x[tuple(slc_dst)] - x[tuple(slc_src)]
        ) * axis_scales[d]
        grads.append(g)

    # Gradient magnitude
    mag = torch.zeros_like(x)
    for g in grads:
        mag = mag + g * g
    mag = torch.sqrt(mag + eps)

    # Normalised gradients
    normed = [g / mag for g in grads]

    # Backward divergence
    div = torch.zeros_like(x)
    for d, gn in enumerate(normed):
        # Backward difference of normalised gradient
        bg = torch.zeros_like(x)
        slc_src = [slice(None)] * ndim
        slc_dst = [slice(None)] * ndim
        slc_src[d] = slice(1, None)
        slc_dst[d] = slice(0, -1)
        bg[tuple(slc_dst)] = (
            gn[tuple(slc_dst)] - gn[tuple(slc_src)]
        ) * axis_scales[d]
        div = div + bg

    # Multiplicative factor
    factor = 1.0 / (1.0 - tv_lambda * div)
    return factor.clamp(min=0.1, max=10.0)

# ---------------------------------------------------------------------------
# Background estimation
# ---------------------------------------------------------------------------

def _estimate_background(image: torch.Tensor) -> float:
    """Estimate background from the median of the lowest 10% of voxels."""
    flat = image.flatten()
    n = max(int(flat.numel() * 0.1), 1)
    lowest, _ = torch.topk(flat, n, largest=False)
    # Approximate mode via median of lowest decile (robust)
    return float(lowest.median())


def _estimate_background_local_plane(
    image: torch.Tensor,
    pixel_size_xy: Optional[float],
    radius_um: float = 0.5,
) -> float:
    """Estimate 2D background from the darkest local neighborhoods."""
    if pixel_size_xy is None:
        px_nm = 100.0
    else:
        px_val = max(float(pixel_size_xy), 1e-6)
        # GUI callers pass nm, while metadata-driven paths historically pass
        # um. Normalize small metadata-style values to nm before sizing the
        # local neighborhood.
        px_nm = px_val * 1000.0 if px_val < 10.0 else px_val
    radius_px = max(int(round((radius_um * 1000.0) / px_nm)), 1)
    radius_px = min(radius_px, max(min(int(image.shape[-2]), int(image.shape[-1])) - 1, 0))
    kernel = 2 * radius_px + 1
    work = image[None, None]
    if radius_px > 0:
        padded = torch.nn.functional.pad(
            work, (radius_px, radius_px, radius_px, radius_px), mode="reflect",
        )
    else:
        padded = work
    local_mean = torch.nn.functional.avg_pool2d(
        padded, kernel_size=kernel, stride=1,
    ).squeeze(0).squeeze(0)
    flat = local_mean.flatten()
    n = max(int(flat.numel() * 0.01), 1)
    lowest, _ = torch.topk(flat, n, largest=False)
    return float(lowest.median())


def _collapse_widefield_psf_to_2d(
    psf: torch.Tensor,
    aggressiveness: str,
) -> torch.Tensor:
    """Collapse a 3D widefield PSF to 2D with center weighting by aggressiveness."""
    if psf.ndim != 3 or psf.shape[0] <= 1:
        return psf.squeeze() if psf.ndim == 3 else psf

    mode = str(aggressiveness).strip().lower()
    nz = psf.shape[0]
    if mode == "very strong":
        weights = torch.ones(nz, dtype=psf.dtype, device=psf.device)
    elif mode == "strong":
        z = torch.arange(nz, dtype=psf.dtype, device=psf.device)
        center = (nz - 1) / 2.0
        sigma = max(nz / 2.5, 1.0)
        weights = torch.exp(-0.5 * ((z - center) / sigma) ** 2)
    elif mode == "very conservative":
        z = torch.arange(nz, dtype=psf.dtype, device=psf.device)
        center = (nz - 1) / 2.0
        sigma = max(nz / 12.0, 1.0)
        weights = torch.exp(-0.5 * ((z - center) / sigma) ** 2)
    elif mode == "conservative":
        z = torch.arange(nz, dtype=psf.dtype, device=psf.device)
        center = (nz - 1) / 2.0
        sigma = max(nz / 8.0, 1.0)
        weights = torch.exp(-0.5 * ((z - center) / sigma) ** 2)
    else:
        z = torch.arange(nz, dtype=psf.dtype, device=psf.device)
        center = (nz - 1) / 2.0
        sigma = max(nz / 4.0, 1.0)
        weights = torch.exp(-0.5 * ((z - center) / sigma) ** 2)

    weights = weights / weights.sum().clamp(min=1e-12)
    plane = (psf * weights[:, None, None]).sum(dim=0)
    return plane / plane.sum().clamp(min=1e-12)


def _estimate_noise_sigma(image: torch.Tensor) -> float:
    """Robust noise estimate via MAD of the lowest quartile."""
    flat = image.flatten()
    n = max(int(flat.numel() * 0.25), 1)
    lowest, _ = torch.topk(flat, n, largest=False)
    med = lowest.median()
    mad = (lowest - med).abs().median()
    sigma = float(mad) * 1.4826
    return max(sigma, 1e-12)


def estimate_image_snr(image: np.ndarray) -> dict[str, Any]:
    """Estimate image SNR without interpreting empty pixels as zero noise.

    Sparse, quantised data is treated as photon-count-like when most values
    lie close to integer multiples of a recurring positive intensity step.
    Other data uses a robust adjacent-difference noise estimate.  The result
    is intentionally deterministic so it can be frozen across a time series.
    """
    values = np.asarray(image, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"snr": 1.0, "reliability": "low", "mode": "empty", "intensity_step": None,
                "noise_sigma": 0.0}
    if values.size > 1_000_000:
        values = values[::max(1, int(math.ceil(values.size / 1_000_000)))]

    background = float(np.percentile(values, 10.0))
    signal = np.clip(values - background, 0.0, None)
    positive = signal[signal > 0.0]
    if positive.size:
        low = positive[positive <= np.percentile(positive, 25.0)]
        step = float(np.median(low)) if low.size else float(np.min(positive))
        if step > 0.0 and np.isfinite(step):
            multiples = signal / step
            quantised_fraction = float(np.mean(np.abs(multiples - np.rint(multiples)) <= 0.05))
            sparse_fraction = float(np.mean(signal <= 0.0))
            if quantised_fraction >= 0.90 and sparse_fraction >= 0.20:
                bright = float(np.percentile(signal, 99.99))
                photons = max(bright / step, 1.0)
                return {
                    "snr": float(np.clip(math.sqrt(photons), 0.1, 1e4)),
                    "reliability": "high" if quantised_fraction >= 0.98 else "medium",
                    "mode": "photon-count",
                    "intensity_step": step,
                    "noise_sigma": math.sqrt(photons) * step,
                }

    shaped = np.asarray(image, dtype=np.float64)
    differences = []
    for axis, size in enumerate(shaped.shape):
        if size > 1:
            differences.append(np.diff(shaped, axis=axis).reshape(-1))
    diff = np.concatenate(differences) if differences else values - np.median(values)
    diff = diff[np.isfinite(diff)]
    if diff.size > 1_000_000:
        diff = diff[::max(1, int(math.ceil(diff.size / 1_000_000)))]
    diff_med = float(np.median(diff)) if diff.size else 0.0
    noise_sigma = 1.4826 * float(np.median(np.abs(diff - diff_med))) / math.sqrt(2.0)
    dynamic = max(float(np.percentile(values, 99.9)) - float(np.median(values)), 0.0)
    scale = max(float(np.max(values) - np.min(values)), 1e-12)
    reliable = noise_sigma > scale * 1e-9
    snr = dynamic / max(noise_sigma, scale * 1e-9)
    return {
        "snr": float(np.clip(snr, 0.1, 1e4)),
        "reliability": "medium" if reliable else "low",
        "mode": "continuous",
        "intensity_step": None,
        "noise_sigma": float(max(noise_sigma, 0.0)),
    }


def _snr_prefilter_sigma(snr: float) -> float:
    """Piecewise-linear SNR-to-prefilter mapping from the public UI contract."""
    snr = max(float(snr), 0.1)
    if snr <= 4.0:
        return 0.8
    if snr <= 8.0:
        return 0.8 + (snr - 4.0) * (0.5 - 0.8) / 4.0
    if snr <= 15.0:
        return 0.5 + (snr - 8.0) * (0.25 - 0.5) / 7.0
    if snr <= 30.0:
        return 0.25 + (snr - 15.0) * (0.0 - 0.25) / 15.0
    return 0.0


def _resolve_snr_settings(
    image: np.ndarray,
    snr: Union[str, float, None],
    acuity: float,
    prefilter_sigma: float,
    rel_threshold: float,
) -> dict[str, Any]:
    """Resolve SNR-aware preprocessing while leaving the RL update unchanged."""
    if snr is None or (isinstance(snr, str) and snr.strip().lower() in {"", "off", "none"}):
        return {
            "enabled": False, "requested_snr": None, "effective_snr": None,
            "acuity": 0.0, "prefilter_sigma": max(float(prefilter_sigma), 0.0),
            "rel_threshold": float(rel_threshold), "estimate": None,
        }
    acuity_value = float(np.clip(float(acuity), -100.0, 100.0))
    if isinstance(snr, str) and snr.strip().lower() == SNR_AUTO:
        estimate = estimate_image_snr(image)
        effective_snr = float(estimate["snr"])
        requested: Union[str, float] = SNR_AUTO
    else:
        effective_snr = float(snr)
        if not np.isfinite(effective_snr) or effective_snr <= 0.0:
            raise ValueError("snr must be 'auto', None/off, or a finite value > 0")
        estimate = {"snr": effective_snr, "reliability": "manual", "mode": "manual",
                    "intensity_step": None, "noise_sigma": 0.0}
        requested = effective_snr
    factor = 2.0 ** (-acuity_value / 50.0)
    auto_prefilter = _snr_prefilter_sigma(effective_snr) * factor
    effective_prefilter = max(float(prefilter_sigma), 0.0) if float(prefilter_sigma) > 0.0 else auto_prefilter
    effective_threshold = float(np.clip(0.02 / effective_snr, 0.0005, 0.005)) * factor
    return {
        "enabled": True, "requested_snr": requested, "effective_snr": effective_snr,
        "acuity": acuity_value, "prefilter_sigma": effective_prefilter,
        "rel_threshold": effective_threshold, "estimate": estimate,
    }


def _scale_aware_auto_offset(image: np.ndarray, estimate: Optional[dict[str, Any]]) -> float:
    """Return a small positive offset expressed in the image's own intensity scale."""
    if estimate and estimate.get("intensity_step"):
        return float(np.clip(0.05 * float(estimate["intensity_step"]), 1e-6, 5.0))
    # Full float64 percentile calculation dominated setup for large tiled
    # volumes (about seven seconds for nine DNA tiles). A deterministic strided
    # sample is already the established policy for responsive image statistics
    # and SNR estimation, and is far more precise than this tiny stabilizing
    # offset requires.
    values = np.asarray(image).reshape(-1)
    if values.size > 1_000_000:
        values = values[::max(1, int(math.ceil(values.size / 1_000_000)))]
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1e-6
    dynamic = max(float(np.percentile(values, 99.9)) - float(np.percentile(values, 1.0)), 0.0)
    noise = float((estimate or {}).get("noise_sigma", 0.0) or 0.0)
    return float(np.clip(max(0.05 * noise, 1e-6 * dynamic, 1e-6), 1e-6, 5.0))


def _gaussian_smooth(image: torch.Tensor, sigma: float) -> torch.Tensor:
    """Apply separable Gaussian smoothing along each axis."""
    if sigma <= 0.0:
        return image.clone()

    radius = int(math.ceil(3.0 * sigma))
    radius = max(radius, 1)
    x = torch.arange(-radius, radius + 1, dtype=image.dtype, device=image.device)
    kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()

    smoothed = image
    from torch.nn.functional import conv1d

    for d in range(image.ndim):
        # reflect padding requires pad < dim_size; fall back to replicate for
        # small dimensions (e.g. thin Z stacks) so output shape is preserved
        pad_mode = "reflect" if smoothed.shape[d] > radius else "replicate"
        pad_widths = [0] * (2 * image.ndim)
        idx = 2 * (image.ndim - 1 - d)
        pad_widths[idx] = radius
        pad_widths[idx + 1] = radius
        padded = torch.nn.functional.pad(
            smoothed.unsqueeze(0).unsqueeze(0),
            pad_widths,
            mode=pad_mode,
        ).squeeze(0).squeeze(0)

        perm = list(range(image.ndim))
        perm.remove(d)
        perm.append(d)
        t = padded.permute(*perm).contiguous()
        batch_shape = t.shape[:-1]
        t = t.reshape(-1, 1, t.shape[-1])
        k1 = kernel_1d.reshape(1, 1, -1)
        t = conv1d(t, k1)
        t = t.reshape(*batch_shape, t.shape[-1])
        inv_perm = [0] * image.ndim
        for i, p in enumerate(perm):
            inv_perm[p] = i
        smoothed = t.permute(*inv_perm).contiguous()

    return smoothed


def _positive_floor(dtype: torch.dtype, bg: float) -> float:
    base = 1e-4 if dtype in (torch.float16, torch.bfloat16) else 1e-7
    return max(base, float(bg) * 1e-6)


def _resolve_start_mode(start: str, img_t: torch.Tensor, bg: float, microscope_type: str = "") -> str:
    """Choose a concrete start mode for ``start='auto'`` from robust image stats."""
    start = str(start or "flat").strip().lower()
    if start not in START_MODES:
        return "flat"
    if start != "auto":
        return start

    flat = img_t.detach().float().reshape(-1)
    flat = flat[torch.isfinite(flat)]
    if flat.numel() == 0:
        return "flat"
    if flat.numel() > 1_000_000:
        step = max(1, int(math.ceil(flat.numel() / 1_000_000)))
        flat = flat[::step]

    quantiles = torch.quantile(
        flat,
        torch.tensor([0.50, 0.90, 0.99], dtype=torch.float32, device=flat.device),
    )
    p50, _, p99 = (float(v) for v in quantiles)
    bg_f = float(bg)
    dynamic = max(p99 - p50, 1e-6)
    if p99 <= max(bg_f * 1.05, bg_f + 1e-6):
        return "percentile_flat"

    signal = (flat - bg_f).clamp(min=0.0)
    signal_cut = max(0.25 * max(p99 - bg_f, 1e-6), 1e-6)
    bright_fraction = float((signal > signal_cut).float().mean())
    bg_ratio = max((p50 - bg_f) / dynamic, 0.0)
    noise_sigma = max(_estimate_noise_sigma(flat), 1e-6)
    snr = dynamic / noise_sigma

    microscope = str(microscope_type or "").strip().lower()
    if microscope == "widefield":
        if snr < 8.0 or bg_ratio > 0.25:
            return "lowpass_bgsub"
        return "hybrid"

    if bright_fraction < 0.04 and snr > 12.0:
        return "observed_bgsub"
    if snr < 8.0:
        return "lowpass_bgsub"
    return "hybrid"


def _resolve_snr_start_mode(start: str, snr_settings: dict[str, Any]) -> str:
    """Resolve automatic initialisation from an explicitly enabled SNR."""
    if str(start).strip().lower() != "auto" or not snr_settings.get("enabled"):
        return start
    snr = float(snr_settings["effective_snr"])
    if snr < 8.0:
        return "lowpass_bgsub"
    if snr <= 20.0:
        return "hybrid"
    return "observed_bgsub"


def _initial_estimate(
    start: str,
    img_t: torch.Tensor,
    d_work: torch.Tensor,
    work_shape: tuple[int, ...],
    slices: tuple[slice, ...],
    bg: float,
    dtype: torch.dtype,
    dev: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build initial estimates for iterative solvers."""
    eps = _positive_floor(dtype, bg)
    img_clamped = img_t.clamp(min=0.0)
    img_bgsub_signal = (img_t - float(bg)).clamp(min=0.0)

    if start == "flat":
        mean_val = max(float(img_t.mean()), bg, eps)
        x_prev = torch.full(work_shape, mean_val, dtype=dtype, device=dev)
        x_cur = x_prev.clone()
        return x_prev, x_cur

    if start == "percentile_flat":
        percentile = torch.quantile(img_clamped.float().reshape(-1), 0.50)
        value = max(float(percentile), bg, eps)
        x_prev = torch.full(work_shape, value, dtype=dtype, device=dev)
        x_cur = x_prev.clone()
        return x_prev, x_cur

    if start == "observed":
        x_prev = d_work.clone().clamp(min=eps)
        x_cur = x_prev.clone()
        return x_prev, x_cur

    if start == "observed_bgsub":
        x_prev = torch.full(work_shape, bg, dtype=dtype, device=dev)
        x_prev[slices] = (img_bgsub_signal + float(bg)).clamp(min=bg).to(dtype=dtype, device=dev)
        x_cur = x_prev.clone()
        return x_prev, x_cur

    if start == "lowpass":
        x_prev = torch.full(work_shape, bg, dtype=dtype, device=dev)
        lowpass = _gaussian_smooth(img_t, sigma=8.0).clamp(min=bg)
        x_prev[slices] = lowpass
        x_cur = x_prev.clone()
        return x_prev, x_cur

    if start == "lowpass_bgsub":
        x_prev = torch.full(work_shape, bg, dtype=dtype, device=dev)
        lowpass = _gaussian_smooth(img_bgsub_signal, sigma=4.0)
        x_prev[slices] = (lowpass + float(bg)).clamp(min=bg).to(dtype=dtype, device=dev)
        x_cur = x_prev.clone()
        return x_prev, x_cur

    observed = img_bgsub_signal
    lowpass = _gaussian_smooth(img_bgsub_signal, sigma=3.0)
    hybrid = (0.65 * observed + 0.35 * lowpass + float(bg)).clamp(min=bg)
    x_prev = torch.full(work_shape, bg, dtype=dtype, device=dev)
    x_prev[slices] = hybrid.to(dtype=dtype, device=dev)
    x_cur = x_prev.clone()
    return x_prev, x_cur


def _initial_estimate_center_plane(
    start: str,
    img_t: torch.Tensor,
    latent_shape: tuple[int, int, int],
    work_shape: tuple[int, int, int],
    obs_slice: tuple[slice, slice, slice],
    bg: float,
    dtype: torch.dtype,
    dev: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Initial estimate for hidden-volume 2D widefield RL."""
    center = latent_shape[0] // 2
    eps = _positive_floor(dtype, bg)
    img_clamped = img_t.clamp(min=0.0)
    img_bgsub_signal = (img_t - float(bg)).clamp(min=0.0)

    if start == "flat":
        mean_val = max(float(img_t.mean()), bg, eps)
        x_prev = torch.full(work_shape, bg, dtype=dtype, device=dev)
        x_prev[center, :latent_shape[1], :latent_shape[2]] = mean_val
        x_cur = x_prev.clone()
        return x_prev, x_cur

    if start == "percentile_flat":
        percentile = torch.quantile(img_clamped.float().reshape(-1), 0.50)
        value = max(float(percentile), bg, eps)
        x_prev = torch.full(work_shape, bg, dtype=dtype, device=dev)
        x_prev[center, :latent_shape[1], :latent_shape[2]] = value
        x_cur = x_prev.clone()
        return x_prev, x_cur

    x_prev = torch.full(work_shape, bg, dtype=dtype, device=dev)
    if start == "lowpass":
        plane = _gaussian_smooth(img_t, sigma=8.0).clamp(min=bg)
    elif start == "observed_bgsub":
        plane = (img_bgsub_signal + float(bg)).clamp(min=bg)
    elif start == "lowpass_bgsub":
        plane = (_gaussian_smooth(img_bgsub_signal, sigma=4.0) + float(bg)).clamp(min=bg)
    elif start == "hybrid":
        lowpass = _gaussian_smooth(img_bgsub_signal, sigma=3.0)
        plane = (0.65 * img_bgsub_signal + 0.35 * lowpass + float(bg)).clamp(min=bg)
    else:
        plane = img_t.clamp(min=max(bg, eps))
    x_prev[obs_slice] = plane.unsqueeze(0)
    x_prev[center, :latent_shape[1], :latent_shape[2]] = plane
    x_cur = x_prev.clone()
    return x_prev, x_cur


def _estimate_widefield_2d_pixel_size_z_nm(
    wavelength_nm: float,
    na: float,
    ri_sample: float,
    pixel_size_xy_nm: Optional[float],
) -> float:
    """Return a practical axial sampling step for hidden-volume 2D widefield RL."""
    na_safe = max(float(na), 1e-6)
    sample_ri = max(float(ri_sample), na_safe + 1e-6)
    xy_nm = max(float(pixel_size_xy_nm), 1.0) if pixel_size_xy_nm is not None else 100.0
    denom = max(sample_ri - math.sqrt(max(sample_ri ** 2 - na_safe ** 2, 1e-9)), 1e-6)
    nyquist_nm = wavelength_nm / (2.0 * denom)
    return max(min(nyquist_nm, 1000.0), xy_nm / 2.0)


def _crop_psf_axial_support(
    psf: torch.Tensor,
    energy_fraction: float = 0.995,
    min_planes: int = 17,
    max_planes: int = 65,
) -> torch.Tensor:
    """Crop a 3D PSF to the smallest odd axial extent covering most energy."""
    if psf.ndim != 3 or psf.shape[0] <= 1:
        return psf
    nz = psf.shape[0]
    center = nz // 2
    axial = psf.sum(dim=(1, 2))
    total = float(axial.sum().detach())
    if total <= 0.0:
        return psf

    target = energy_fraction * total
    cum = float(axial[center].detach())
    radius = 0
    max_radius = center
    while cum < target and radius < max_radius:
        radius += 1
        cum += float(axial[center - radius].detach()) + float(axial[center + radius].detach())

    planes = 2 * radius + 1
    planes = max(min_planes, planes)
    planes = min(max_planes, planes, nz)
    if planes % 2 == 0:
        planes = max(1, planes - 1)
    half = planes // 2
    lo = max(center - half, 0)
    hi = min(center + half + 1, nz)
    if hi - lo < planes:
        lo = max(hi - planes, 0)
        hi = min(lo + planes, nz)
    return psf[lo:hi]


def _forward_project(
    estimate: torch.Tensor,
    otf: torch.Tensor,
    work_shape: tuple[int, ...],
) -> torch.Tensor:
    """Forward convolution using the prepared OTF."""
    return _irfft(_rfft(estimate) * otf, work_shape)


def _embed_in_work(
    image: torch.Tensor,
    work_shape: tuple[int, ...],
    slices: tuple[slice, ...],
    background: float,
) -> torch.Tensor:
    """Embed an image-domain estimate in the full linear-convolution domain."""
    work = torch.full(work_shape, background, dtype=image.dtype, device=image.device)
    work[slices] = image
    return work


def _poisson_nll(
    observed: torch.Tensor,
    estimated: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Differentiable Poisson negative log-likelihood up to constants."""
    est_safe = estimated.clamp(min=eps)
    return torch.mean(est_safe - observed * torch.log(est_safe))


def _sparse_hessian_penalty(
    x: torch.Tensor,
    weighting: float,
    z_scale: float = 1.0,
) -> torch.Tensor:
    """Sparse-Hessian / SPITFIRE-style regularisation penalty."""
    eps = 1e-8
    weighting = float(np.clip(weighting, 0.0, 1.0))

    # Treat singleton axes as lower-dimensional data so 2-D inputs stored
    # as (1, Y, X) still receive XY regularisation instead of a zero penalty.
    if x.ndim == 3 and 1 in x.shape:
        squeezed = x.squeeze()
        if squeezed.ndim in (2, 3):
            eff_z_scale = z_scale if squeezed.ndim == 3 else 1.0
            return _sparse_hessian_penalty(
                squeezed, weighting, z_scale=eff_z_scale
            )

    if x.ndim == 2:
        if min(x.shape) < 3:
            return x.sum() * 0.0
        core = x[1:-1, 1:-1]
        dxx = -x[2:, 1:-1] + 2.0 * core - x[:-2, 1:-1]
        dyy = -x[1:-1, 2:] + 2.0 * core - x[1:-1, :-2]
        dxy = x[2:, 2:] - x[2:, 1:-1] - x[1:-1, 2:] + core
        hv = (
            (weighting * dxx) ** 2
            + (weighting * dyy) ** 2
            + 2.0 * (weighting * dxy) ** 2
            + ((1.0 - weighting) * core) ** 2
        )
        return torch.mean(torch.sqrt(hv + eps))

    if x.ndim == 3:
        if min(x.shape) < 3:
            return x.sum() * 0.0
        core = x[1:-1, 1:-1, 1:-1]
        dxx = -x[1:-1, 1:-1, 2:] + 2.0 * core - x[1:-1, 1:-1, :-2]
        dyy = -x[1:-1, 2:, 1:-1] + 2.0 * core - x[1:-1, :-2, 1:-1]
        dzz = z_scale * z_scale * (
            -x[2:, 1:-1, 1:-1] + 2.0 * core - x[:-2, 1:-1, 1:-1]
        )
        dxy = x[1:-1, 2:, 2:] - x[1:-1, 1:-1, 2:] - x[1:-1, 2:, 1:-1] + core
        dxz = z_scale * (
            x[2:, 1:-1, 2:] - x[1:-1, 1:-1, 2:] - x[2:, 1:-1, 1:-1] + core
        )
        dyz = z_scale * (
            x[2:, 2:, 1:-1] - x[1:-1, 2:, 1:-1] - x[2:, 1:-1, 1:-1] + core
        )
        hv = (
            (weighting * dxx) ** 2
            + (weighting * dyy) ** 2
            + (weighting * dzz) ** 2
            + 2.0 * (weighting * dxy) ** 2
            + 2.0 * (weighting * dxz) ** 2
            + 2.0 * (weighting * dyz) ** 2
            + ((1.0 - weighting) * core) ** 2
        )
        return torch.mean(torch.sqrt(hv + eps))

    raise ValueError("Sparse-Hessian regularisation supports only 2D or 3D tensors")


def _anscombe_prefilter(
    image: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Variance-stabilising Anscombe pre-filter for Poisson data.

    Applies the generalised Anscombe transform  ``2·√(x + 3/8)`` to
    convert Poisson noise to approximately unit-variance Gaussian,
    smooths with a Gaussian of width *sigma* pixels, then applies the
    exact unbiased inverse  ``(y/2)² − 3/8``.

    References: Anscombe 1948; Makitalo & Foi 2011.
    """
    # Forward Anscombe transform
    stabilised = 2.0 * torch.sqrt(image.clamp(min=0.0) + 3.0 / 8.0)

    # Separable Gaussian smoothing in the stabilised domain
    smoothed = _gaussian_smooth(stabilised, sigma)

    # Inverse Anscombe transform (exact unbiased)
    result = (smoothed / 2.0) ** 2 - 3.0 / 8.0
    return result.clamp(min=0.0)


# ---------------------------------------------------------------------------
# XY tiling helpers (large image support)
# ---------------------------------------------------------------------------

TILE_MARGIN = 16


def _get_memory_budget_bytes(device: Optional[str] = None) -> int:
    """Return an estimated safe memory budget in bytes for tiling decisions.

    Uses 55% of total GPU VRAM when CUDA is available, capped by current
    free VRAM so other processes on the same card force smaller tiles.  CPU
    execution uses 50% of available system RAM (capped at 16 GB).
    """
    dev = _pick_device(device)
    if dev.type == "cuda":
        try:
            idx = dev.index if dev.index is not None else torch.cuda.current_device()
            total = torch.cuda.get_device_properties(idx).total_memory
            try:
                free, _ = torch.cuda.mem_get_info(idx)
                return int(max(512 * 1024 ** 2, min(total * 0.55, free * 0.80)))
            except Exception:
                return int(total * 0.55)
        except Exception:
            return int(8e9 * 0.55)
    else:
        try:
            import psutil
            available = psutil.virtual_memory().available
            return int(min(available * 0.50, 16e9))
        except ImportError:
            return int(8e9 * 0.50)


def _suggest_max_tile_xy(
    n_z: int,
    psf_xy_est: int = 65,
    device: Optional[str] = None,
) -> int:
    """Return the largest safe XY tile dimension for the given Z depth and device.

    Memory model: ``budget ≈ 64 × padded_z × (tile_xy + psf_xy)²`` bytes,
    where ``padded_z ≈ 4 × n_z`` (FFT zero-padding for a 3D PSF of size
    ``2*n_z - 1``).
    """
    budget = _get_memory_budget_bytes(device)
    dev = _pick_device(device)
    if _pick_dtype(dev) == torch.float64:
        budget //= 2  # float64 uses twice the memory
    padded_z = max(1, 4 * n_z) if n_z > 1 else 1
    inner_sq = budget / max(64 * padded_z, 1)
    tile_xy = int(math.sqrt(max(inner_sq, 0.0))) - psf_xy_est
    return max(256, min(tile_xy, 4096))


def _auto_n_tiles(
    shape: tuple[int, ...],
    device: Optional[str] = None,
    psf_xy_est: int = 65,
) -> int:
    """Return the minimum number of XY tiles so each tile fits in device memory.

    Tile size is computed automatically from the available GPU/CPU memory
    budget via :func:`_suggest_max_tile_xy`.  Only XY is tiled; Z is always
    processed in full.
    """
    if len(shape) < 3:
        return 1
    n_z, H, W = shape[:3]
    max_xy = _suggest_max_tile_xy(n_z, psf_xy_est=psf_xy_est, device=device)
    ny = max(1, -(-H // max_xy))
    nx = max(1, -(-W // max_xy))
    n_tiles = ny * nx
    if n_tiles > 1:
        log.info(
            "Auto-tiling: image z=%d h=%d w=%d, budget max_tile_xy=%d → %d×%d grid",
            n_z, H, W, max_xy, ny, nx,
        )
    return 1 if n_tiles <= 1 else n_tiles


def _resolve_tiling(
    tiling: str,
    shape: tuple[int, ...],
    device: Optional[str] = None,
    psf_xy_est: int = 65,
) -> int:
    """Resolve *tiling* mode to a concrete tile count.

    Accepted values: ``"none"`` (disable tiling) or ``"auto"`` / ``"custom"``
    (compute from image shape using the device memory budget).
    """
    if isinstance(tiling, str):
        mode = tiling.strip().lower()
        if mode == "none":
            return 1
        if mode in ("custom", "auto"):
            return _auto_n_tiles(shape, device=device, psf_xy_est=psf_xy_est)
        raise ValueError(f"tiling must be 'none' or 'auto', got '{tiling}'")
    return max(int(tiling), 1)


def _compute_tile_grid(
    shape_yx: tuple[int, int], n_tiles: int,
) -> tuple[int, int]:
    """Return (ny, nx) tile counts that best cover *shape_yx*."""
    if n_tiles <= 1:
        return (1, 1)
    best = (1, n_tiles)
    best_ratio = float("inf")
    for ny in range(1, n_tiles + 1):
        if n_tiles % ny != 0:
            continue
        nx = n_tiles // ny
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
    """Return a list of tile descriptors with overlap margins."""
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


def _blend_tile(tile_result: np.ndarray, desc: dict) -> tuple[np.ndarray, np.ndarray]:
    """Apply linear ramp blending in overlap zones; return (weighted, weight)."""
    ov_top, ov_bot = desc["blend_y"]
    ov_left, ov_right = desc["blend_x"]

    tile_h = tile_result.shape[1]
    tile_w = tile_result.shape[2]
    weight = np.ones((tile_h, tile_w), dtype=np.float32)

    if ov_top > 0:
        ramp = np.linspace(0, 1, ov_top + 1, dtype=np.float32)[1:]
        weight[:ov_top, :] *= ramp[:, np.newaxis]
    if ov_bot > 0:
        ramp = np.linspace(1, 0, ov_bot + 1, dtype=np.float32)[:-1]
        weight[tile_h - ov_bot:, :] *= ramp[:, np.newaxis]
    if ov_left > 0:
        ramp = np.linspace(0, 1, ov_left + 1, dtype=np.float32)[1:]
        weight[:, :ov_left] *= ramp[np.newaxis, :]
    if ov_right > 0:
        ramp = np.linspace(1, 0, ov_right + 1, dtype=np.float32)[:-1]
        weight[:, tile_w - ov_right:] *= ramp[np.newaxis, :]

    weighted = tile_result * weight[np.newaxis, :, :]
    return weighted, weight


def _ci_deconvolve_tiled(
    image: np.ndarray,
    psf: np.ndarray,
    n_tiles: int,
    solver,
    **kwargs,
) -> dict[str, Any]:
    """Split *image* into XY tiles, deconvolve each, and blend back."""
    overlap = max(psf.shape[-1], psf.shape[-2]) // 2
    margin = TILE_MARGIN

    ny, nx = _compute_tile_grid(image.shape[1:], n_tiles)
    min_tile_yx = min(image.shape[1] / ny, image.shape[2] / nx)
    if min_tile_yx < max(psf.shape[-2:]):
        log.warning(
            "n_tiles=%d produces tiles smaller than PSF; falling back to "
            "no tiling.", n_tiles,
        )
        return solver(image, psf, tiling="none", **kwargs)

    tiles = _compute_tile_slices(image.shape, ny, nx, overlap)
    log.info(
        "Tiled deconvolution: %d tiles (%d×%d grid), overlap=%d, margin=%d px",
        n_tiles, ny, nx, overlap, margin,
    )

    Z, H, W = image.shape
    numerator = np.zeros_like(image, dtype=np.float32)
    # Feather weights vary in XY only. Storing the same denominator for every Z
    # plane wastes a complete image-sized host allocation and repeats identical
    # additions along Z.
    denominator = np.zeros((H, W), dtype=np.float32)

    total_iterations = 0
    all_convergence: list[float] = []
    effective_parameters: dict[str, Any] = {}
    backend_used: Optional[str] = None
    representative_work_shape: Optional[tuple[int, ...]] = None
    optimized_workspace_bytes = 0

    jobs: list[tuple[dict[str, Any], int, int, int, int]] = []
    for desc in tiles:
        _, ey, ex = desc["extract"]
        y0_m = max(ey.start - margin, 0)
        y1_m = min(ey.stop + margin, H)
        x0_m = max(ex.start - margin, 0)
        x1_m = min(ex.stop + margin, W)
        jobs.append((desc, y0_m, y1_m, x0_m, x1_m))

    # Reordering does not affect blending. It lets equal tile geometries reuse
    # one OTF/weights/plan/arena before moving to the next geometry.
    jobs.sort(key=lambda job: (Z, job[2] - job[1], job[4] - job[3]))

    for idx, (desc, y0_m, y1_m, x0_m, x1_m) in enumerate(jobs):
        _, ey, ex = desc["extract"]
        tile_img = image[:, y0_m:y1_m, x0_m:x1_m].copy()

        log.info("  Tile %d/%d  shape=%s", idx + 1, len(tiles), tile_img.shape)
        tile_out = solver(tile_img, psf, tiling="none", **kwargs)
        tile_result = np.asarray(tile_out["result"], dtype=np.float32)

        total_iterations = max(total_iterations, tile_out["iterations_used"])
        if tile_out["convergence"]:
            all_convergence = tile_out["convergence"]
        if tile_out.get("effective_parameters"):
            effective_parameters = dict(tile_out["effective_parameters"])
        backend_used = tile_out.get("backend", backend_used)
        if tile_out.get("work_shape"):
            representative_work_shape = tuple(tile_out["work_shape"])
        optimized_workspace_bytes = max(
            optimized_workspace_bytes, int(tile_out.get("optimized_workspace_bytes", 0)),
        )

        # Crop margin back to the original extract region
        crop_y0 = ey.start - y0_m
        crop_y1 = crop_y0 + (ey.stop - ey.start)
        crop_x0 = ex.start - x0_m
        crop_x1 = crop_x0 + (ex.stop - ex.start)
        tile_cropped = tile_result[:, crop_y0:crop_y1, crop_x0:crop_x1]

        weighted, weight = _blend_tile(tile_cropped, desc)
        ext = desc["extract"]
        numerator[ext] += weighted.astype(np.float32, copy=False)
        denominator[ext[-2:]] += weight.astype(np.float32, copy=False)
        del tile_img, tile_out, tile_result, tile_cropped, weighted, weight
        _release_cuda_cache()

    np.maximum(denominator, np.float32(1e-12), out=denominator)
    np.divide(numerator, denominator[np.newaxis, :, :], out=numerator)
    np.clip(numerator, 0, None, out=numerator)
    result = numerator
    del denominator
    _release_cuda_cache()
    return {
        "result": result,
        "convergence": all_convergence,
        "iterations_used": total_iterations,
        "effective_parameters": effective_parameters,
        "backend": backend_used,
        "work_shape": representative_work_shape,
        "optimized_workspace_bytes": optimized_workspace_bytes,
        "tile_count": len(jobs),
    }


@_release_cuda_cache_after
def _ci_rl_deconvolve_2d_widefield(
    image: np.ndarray,
    psf: np.ndarray,
    *,
    niter: int,
    tv_lambda: float,
    offset: Union[str, float],
    prefilter_sigma: float,
    snr: Union[str, float, None],
    acuity: float,
    start: str,
    background: Union[str, float],
    convergence: str,
    rel_threshold: float,
    check_every: int,
    pixel_size_xy: Optional[float],
    pixel_size_z: Optional[float],
    two_d_wf_aggressiveness: str,
    two_d_wf_bg_radius_um: float,
    two_d_wf_bg_scale: float,
    device: Optional[str],
    backend: str = BACKEND_AUTO,
    iteration_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    channel_index: int = 0,
) -> dict[str, Any]:
    """Conservative widefield-aware 2D RL using a collapsed 3D PSF."""
    if start not in START_MODES:
        start = "flat"

    dev, _, _ = _resolve_backend(backend, device)
    dtype = _pick_dtype(dev)
    img_t = _to_tensor(image, dev, dtype)

    if psf.ndim == 3:
        psf_t = _crop_psf_axial_support(_to_tensor(psf, dev, dtype))
        psf_2d = _to_numpy(
            _collapse_widefield_psf_to_2d(psf_t, two_d_wf_aggressiveness).detach()
        ).astype(np.float32, copy=False)
    elif psf.ndim == 2:
        psf_2d = np.asarray(psf, dtype=np.float32)
        psf_2d = psf_2d / max(float(psf_2d.sum()), 1e-12)
    else:
        raise ValueError(f"Unsupported PSF dimensionality for 2D widefield auto mode: {psf.shape}")

    mode = str(two_d_wf_aggressiveness).strip().lower()
    if mode == "very conservative":
        auto_offset_default = 3.0
        auto_bg_preset_scale = 1.2
    elif mode == "conservative":
        auto_offset_default = 2.5
        auto_bg_preset_scale = 1.1
    elif mode == "very strong":
        auto_offset_default = 1.0
        auto_bg_preset_scale = 0.8
    elif mode == "strong":
        auto_offset_default = 1.5
        auto_bg_preset_scale = 0.9
    else:
        auto_offset_default = 2.0
        auto_bg_preset_scale = 1.0

    bg_radius_um = max(float(two_d_wf_bg_radius_um), 0.05)
    bg_scale = max(float(two_d_wf_bg_scale), 0.1)

    if background == "auto":
        local_bg = _estimate_background_local_plane(
            img_t, pixel_size_xy, radius_um=bg_radius_um,
        )
        global_bg = _estimate_background(img_t)
        base_bg = min(local_bg, global_bg)
        effective_background: Union[str, float] = max(
            base_bg * auto_bg_preset_scale * bg_scale, 1e-6,
        )
    else:
        effective_background = max(float(background), 1e-6)

    if offset == "auto":
        effective_offset = "auto" if snr is not None else auto_offset_default
    else:
        effective_offset = offset

    effective_prefilter = max(float(prefilter_sigma), 0.0)

    log.info(
        "  2D WF auto -> collapsed-PSF RL  mode=%s  radius_um=%.3g  bg_scale=%.3g  "
        "bg=%.4g  offset=%s  prefilter=%.4g",
        two_d_wf_aggressiveness,
        bg_radius_um,
        bg_scale,
        float(effective_background),
        effective_offset,
        effective_prefilter,
    )

    try:
        return ci_rl_deconvolve(
            image,
            psf_2d,
            niter=niter,
            tv_lambda=tv_lambda,
            offset=effective_offset,
            prefilter_sigma=effective_prefilter,
            snr=snr,
            acuity=acuity,
            start=start,
            background=effective_background,
            convergence=convergence,
            rel_threshold=rel_threshold,
            check_every=check_every,
            pixel_size_xy=pixel_size_xy,
            pixel_size_z=pixel_size_z,
            microscope_type="widefield",
            two_d_mode="legacy_2d",
            device=device,
            backend=backend,
            tiling="none",
            iteration_callback=iteration_callback,
            channel_index=channel_index,
        )
    finally:
        try:
            del img_t
        except UnboundLocalError:
            pass
        try:
            del psf_t
        except UnboundLocalError:
            pass
        try:
            del psf_2d
        except UnboundLocalError:
            pass
        _release_cuda_cache()


# ---------------------------------------------------------------------------
# Top-level RL deconvolution
# ---------------------------------------------------------------------------

@_release_cuda_cache_after
def ci_rl_deconvolve(
    image: np.ndarray,
    psf: np.ndarray,
    *,
    niter: int = 50,
    tv_lambda: float = 0.0,
    offset: Union[str, float] = "auto",
    prefilter_sigma: float = 0.0,
    snr: Union[str, float, None] = None,
    acuity: float = 0.0,
    start: str = "auto",
    background: Union[str, float] = "auto",
    convergence: str = "auto",
    rel_threshold: float = 0.005,
    check_every: int = 5,
    pixel_size_xy: Optional[float] = None,
    pixel_size_z: Optional[float] = None,
    microscope_type: str = "widefield",
    two_d_mode: str = "auto",
    two_d_wf_aggressiveness: str = "balanced",
    two_d_wf_bg_radius_um: float = 0.5,
    two_d_wf_bg_scale: float = 1.0,
    device: Optional[str] = None,
    backend: str = BACKEND_AUTO,
    tiling: str = "auto",
    iteration_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    channel_index: int = 0,
) -> dict[str, Any]:
    """SHB-accelerated Richardson-Lucy deconvolution (GPU / CPU).

    Parameters
    ----------
    image : ndarray
        Observed (noisy) image, 2-D or 3-D.
    psf : ndarray
        Point Spread Function (same dimensionality as image).
    niter : int
        Maximum number of iterations.
    tv_lambda : float
        Total-Variation regularisation strength (0 = disabled).
    offset : ``"auto"``, or float >= 0
        Constant added to the image before RL iterations and subtracted
        afterwards.  Shifts all pixels away from zero, preventing the
        extreme ratio amplification that causes background noise.
        With SNR off, ``"auto"`` uses the legacy value 5.0. With SNR
        enabled it is scaled to the image intensity step. ``0`` disables it.
    prefilter_sigma : float
        Gaussian sigma (in pixels) for Anscombe variance-stabilising
        pre-filter.  ``0`` = disabled.  Typical values: 0.5–1.0.
    snr : ``None``, ``"auto"``, or float
        Enable noise-aware initializer, prefilter, convergence, and offset
        selection. ``None`` preserves legacy behavior.
    acuity : float
        Smooth/sharp balance from -100 to +100 when SNR is enabled.
    start : str
        Initial estimate.  Supported modes are ``"auto"``, ``"flat"``,
        ``"percentile_flat"``, ``"observed"``, ``"observed_bgsub"``,
        ``"lowpass"``, ``"lowpass_bgsub"``, and ``"hybrid"``.
    background : ``"auto"`` or float
        Background level used as positivity floor and safe-division epsilon.
    convergence : ``"fixed"`` or ``"auto"``
        ``"fixed"`` runs exactly *niter* iterations; ``"auto"`` stops when
        the relative I-divergence change drops below *rel_threshold*.
    rel_threshold : float
        Relative change threshold for auto-convergence.
    check_every : int
        Evaluate I-divergence every *check_every* iterations.
    microscope_type : str
        Microscope mode. ``"widefield"`` activates the hidden-volume 2-D
        model when *image* is 2-D and *two_d_mode* is ``"auto"``.
    two_d_mode : str
        ``"auto"`` enables the enhanced 2-D widefield model; ``"legacy_2d"``
        preserves the historical pure-2-D RL behavior.
    two_d_wf_aggressiveness : str
        Expert tuning for 2-D widefield auto mode: ``"conservative"``,
        ``"balanced"``, or ``"strong"``.
    two_d_wf_bg_radius_um : float
        Expert background-estimator neighborhood radius in micrometers for
        2-D widefield auto mode.
    two_d_wf_bg_scale : float
        Expert multiplier applied to the auto-estimated 2-D widefield
        background.
    device : str or None
        PyTorch device (``"cuda"``, ``"cpu"``).  ``None`` = auto.

    Returns
    -------
    dict
        ``"result"`` — deconvolved image (ndarray, same shape as input).
        ``"convergence"`` — list of I-divergence values at check-points.
        ``"iterations_used"`` — number of iterations actually performed.
    """
    dev, optimized_ops, backend_used = _resolve_backend(backend, device)
    device = str(dev)

    # --- Tiling dispatch ---
    psf_xy_est = max(psf.shape[-1], psf.shape[-2]) if psf.ndim >= 2 else 65
    n_tiles = _resolve_tiling(tiling, image.shape, device=device, psf_xy_est=psf_xy_est)
    if n_tiles > 1 and iteration_callback is not None:
        log.info(
            "Per-iteration image previews disabled for %d-tile execution; "
            "this does not disable the selected compute backend.", n_tiles,
        )
        iteration_callback = None
    if n_tiles > 1:
        dispatch_snr = snr
        if isinstance(snr, str) and snr.strip().lower() == SNR_AUTO:
            dispatch_snr = float(estimate_image_snr(image)["snr"])
        return _ci_deconvolve_tiled(
            image, psf, n_tiles,
            solver=ci_rl_deconvolve,
            niter=niter, tv_lambda=tv_lambda, offset=offset,
            prefilter_sigma=prefilter_sigma, snr=dispatch_snr, acuity=acuity, start=start,
            background=background,
            convergence=convergence, rel_threshold=rel_threshold,
            check_every=check_every,
            pixel_size_xy=pixel_size_xy, pixel_size_z=pixel_size_z,
            microscope_type=microscope_type, two_d_mode=two_d_mode,
            two_d_wf_aggressiveness=two_d_wf_aggressiveness,
            two_d_wf_bg_radius_um=two_d_wf_bg_radius_um,
            two_d_wf_bg_scale=two_d_wf_bg_scale,
            device=device,
            backend=backend,
            iteration_callback=iteration_callback,
            channel_index=channel_index,
        )

    two_d_mode = str(two_d_mode).strip().lower()
    microscope_type = str(microscope_type).strip().lower()
    if image.ndim == 2 and microscope_type == "widefield" and two_d_mode == "auto":
        return _ci_rl_deconvolve_2d_widefield(
            image,
            psf,
            niter=niter,
            tv_lambda=tv_lambda,
            offset=offset,
            prefilter_sigma=prefilter_sigma,
            snr=snr,
            acuity=acuity,
            start=start,
            background=background,
            convergence=convergence,
            rel_threshold=rel_threshold,
            check_every=check_every,
            pixel_size_xy=pixel_size_xy,
            pixel_size_z=pixel_size_z,
            two_d_wf_aggressiveness=two_d_wf_aggressiveness,
            two_d_wf_bg_radius_um=two_d_wf_bg_radius_um,
            two_d_wf_bg_scale=two_d_wf_bg_scale,
            device=device,
            backend=backend,
            iteration_callback=iteration_callback,
            channel_index=channel_index,
        )

    snr_settings = _resolve_snr_settings(image, snr, acuity, prefilter_sigma, rel_threshold)
    effective_snr = snr_settings["effective_snr"]
    prefilter_sigma = float(snr_settings["prefilter_sigma"])
    if snr_settings["enabled"] and convergence == "auto":
        rel_threshold = float(snr_settings["rel_threshold"])
    start = _resolve_snr_start_mode(start, snr_settings)
    if offset == "auto" and snr_settings["enabled"]:
        offset = _scale_aware_auto_offset(image, snr_settings.get("estimate"))
    if snr_settings["enabled"]:
        estimate = snr_settings["estimate"] or {}
        log.info(
            "  SNR requested=%s effective=%.4g reliability=%s mode=%s acuity=%.1f "
            "effective_prefilter=%.4g effective_rel_threshold=%.4g",
            snr_settings["requested_snr"], effective_snr, estimate.get("reliability", "manual"),
            estimate.get("mode", "manual"), snr_settings["acuity"], prefilter_sigma, rel_threshold,
        )

    dtype = _pick_dtype(dev)

    # Resolve offset
    if offset == "auto":
        offset_val = 5.0
    else:
        offset_val = max(float(offset), 0.0)

    if start not in START_MODES:
        start = "flat"

    log.info("ci_rl_deconvolve  backend=%s  device=%s  dtype=%s  shape=%s  niter=%d  "
             "tv_lambda=%.4g  offset=%.4g  prefilter_sigma=%.4g  "
             "start=%s  convergence=%s  microscope=%s  two_d_mode=%s",
             backend_used, dev, dtype, image.shape, niter, tv_lambda, offset_val,
             prefilter_sigma, start, convergence, microscope_type, two_d_mode)

    # Move data to device
    img_t = _to_tensor(image, dev, dtype)
    psf_t = _to_tensor(psf, dev, dtype)
    optimized_squeeze_2d = optimized_ops is not None and img_t.ndim == 2
    if optimized_squeeze_2d:
        img_t = img_t.unsqueeze(0)
        psf_t = psf_t.unsqueeze(0)

    # Background
    if background == "auto":
        bg = max(_estimate_background(img_t), 1e-6)
    else:
        bg = max(float(background), 1e-6)
    log.info("  background=%.4g", bg)

    # Apply offset — shift all intensities away from zero
    if offset_val > 0.0:
        img_t = img_t + offset_val
        bg = bg + offset_val

    # Anscombe pre-filter (variance stabilisation)
    if prefilter_sigma > 0.0:
        img_t = _anscombe_prefilter(img_t, prefilter_sigma)
        log.info("  prefilter_sigma=%.4g applied", prefilter_sigma)

    requested_start = start
    start = _resolve_start_mode(start, img_t, bg, microscope_type)
    if requested_start != start:
        log.info("  start=%s resolved to %s", requested_start, start)

    # Work shape = image + psf - 1  (full linear convolution)
    minimum_work_shape = tuple(si + sp - 1 for si, sp in zip(img_t.shape, psf_t.shape))
    axis_scales = _axis_scales(img_t.ndim, pixel_size_xy, pixel_size_z)

    # Prepare OTF & weights
    optimized_context = None
    if optimized_ops is not None:
        try:
            optimized_context = _optimized_context(
                optimized_ops, psf_t, tuple(img_t.shape), minimum_work_shape,
            )
        except Exception as exc:
            if str(backend or BACKEND_AUTO).strip().lower() != BACKEND_AUTO:
                raise
            log.warning("Optimized CUDA setup failed; falling back to PyTorch CUDA: %s", exc)
            clear_optimized_context_cache()
            optimized_ops = None
            backend_used = BACKEND_PYTORCH_CUDA
            if optimized_squeeze_2d:
                img_t = img_t[0]
                psf_t = psf_t[0]
                optimized_squeeze_2d = False
                minimum_work_shape = tuple(
                    si + sp - 1 for si, sp in zip(img_t.shape, psf_t.shape)
                )
                axis_scales = _axis_scales(img_t.ndim, pixel_size_xy, pixel_size_z)
    if optimized_context is not None:
        work_shape = optimized_context["work_shape"]
        otf = optimized_context["otf"]
        otf_conj = otf.conj()
        W = optimized_context["weights"]
    else:
        work_shape = minimum_work_shape
        otf, otf_conj = _prepare_otf(psf_t, work_shape)
        W = _bertero_weights(otf, otf_conj, img_t.shape, work_shape)

    # Zero-pad observed image into work domain
    d_work = torch.full(work_shape, bg, dtype=dtype, device=dev)
    slices = tuple(slice(0, s) for s in img_t.shape)
    d_work[slices] = img_t

    # Initialise estimate
    x_prev, x_cur = _initial_estimate(
        start, img_t, d_work, work_shape, slices, bg, dtype, dev,
    )
    # Padding is only needed for initialization; do not retain a complete
    # FFT-sized tensor throughout every iteration of a large tile.
    del d_work

    convergence_history: list[float] = []
    use_tv = tv_lambda > 0.0
    early_stop_min_iter = max(10, niter // 4)
    iterations_used = niter

    for k in range(1, niter + 1):
        # --- SHB momentum (Wang & Miller 2014) ---
        if k >= 3:
            alpha_max = 1.0 - 2.0 / math.sqrt(k + 3.0)
            alpha = min((k - 1.0) / (k + 2.0), alpha_max)
        else:
            alpha = 0.0
        if optimized_context is not None:
            storage = optimized_context["storage"]
            plan = optimized_context["plan"]
            workspace = optimized_context["workspace"]
            pitch = optimized_context["pitch"]
            normalization = optimized_context["normalization"]
            next_buffer = x_prev
            optimized_ops.momentum_pack(
                x_cur, next_buffer, alpha, bg, storage, *work_shape, pitch,
            )
            plan.forward(storage, workspace)
            optimized_ops.multiply_otf(optimized_context["frequency"], otf, False)
            plan.inverse(storage, workspace)
            optimized_ops.ratio_pitched(
                storage, img_t, *work_shape, pitch, bg, normalization,
            )
            plan.forward(storage, workspace)
            optimized_ops.multiply_otf(optimized_context["frequency"], otf, True)
            plan.inverse(storage, workspace)
            optimized_ops.update_pitched(
                next_buffer, storage, W, bg, normalization, *work_shape, pitch,
            )
            x_prev, x_cur = x_cur, next_buffer
            if use_tv:
                optimized_ops.tv_update(x_cur, storage, tv_lambda, *axis_scales, bg)
            y = None
        else:
            p = (x_cur + alpha * (x_cur - x_prev)).clamp(min=bg)
            P_fft = _rfft(p)
            Y_fft = P_fft * otf
            y = _irfft(Y_fft, work_shape)
            del P_fft, Y_fft
            r = torch.zeros(work_shape, dtype=dtype, device=dev)
            r[slices] = img_t / y[slices].clamp(min=bg)
            R_fft = _rfft(r)
            corr = _irfft(R_fft * otf_conj, work_shape)
            del R_fft, r
            x_new = p * corr * W
            if use_tv:
                x_new = x_new * _tv_penalty(x_new, tv_lambda, axis_scales)
            x_new = x_new.clamp(min=bg)
            x_prev, x_cur = x_cur, x_new

        stop_after_callback = False
        convergence_value: Optional[float] = None
        check_iteration = k % check_every == 0 or k == niter
        # --- Convergence check ---
        if check_iteration:
            # Recompute forward for I-divergence (reuse y from last iter if
            # it's a check iteration — here y is still valid for p, not x_new,
            # but the difference is small; for exactness re-project)
            if optimized_context is not None:
                storage = optimized_context["storage"]
                storage.zero_()
                storage[..., :work_shape[-1]].copy_(x_cur)
                optimized_context["plan"].forward(storage, optimized_context["workspace"])
                optimized_ops.multiply_otf(optimized_context["frequency"], otf, False)
                optimized_context["plan"].inverse(storage, optimized_context["workspace"])
                fwd = storage[..., :work_shape[-1]] / optimized_context["normalization"]
            else:
                fwd_fft = _rfft(x_cur) * otf
                fwd = _irfft(fwd_fft, work_shape)
                del fwd_fft
            idiv = _i_divergence(img_t, fwd[slices].clamp(min=bg))
            convergence_history.append(idiv)
            convergence_value = idiv
            log.info("  iter %4d/%d  I-div=%.6g", k, niter, idiv)

            if convergence == "auto" and len(convergence_history) >= 2 and k > early_stop_min_iter:
                prev_idiv = convergence_history[-2]
                if prev_idiv > 0:
                    rel_change = (prev_idiv - idiv) / prev_idiv
                    if rel_change < rel_threshold:
                        log.info("  converged at iter %d (rel_change=%.4g)", k, rel_change)
                        iterations_used = k
                        stop_after_callback = True

        if iteration_callback is not None:
            if not check_iteration:
                if optimized_context is not None:
                    storage = optimized_context["storage"]
                    storage.zero_()
                    storage[..., :work_shape[-1]].copy_(x_cur)
                    optimized_context["plan"].forward(storage, optimized_context["workspace"])
                    optimized_ops.multiply_otf(optimized_context["frequency"], otf, False)
                    optimized_context["plan"].inverse(storage, optimized_context["workspace"])
                    fwd = storage[..., :work_shape[-1]] / optimized_context["normalization"]
                else:
                    fwd_fft = _rfft(x_cur) * otf
                    fwd = _irfft(fwd_fft, work_shape)
                    del fwd_fft
                convergence_value = _i_divergence(img_t, fwd[slices].clamp(min=bg))
            frame = x_cur[slices]
            estimated = fwd[slices]
            if optimized_squeeze_2d:
                frame = frame[0]
                estimated = estimated[0]
            if offset_val > 0.0:
                frame = (frame - offset_val).clamp(min=0.0)
                estimated = (estimated - offset_val).clamp(min=0.0)
            iteration_callback({
                "channel_index": int(channel_index),
                "iteration": k,
                "total_iterations": niter,
                "convergence": convergence_value,
                "image": _to_numpy(frame),
                "estimated": _to_numpy(estimated),
                "background": max(float(bg) - offset_val, 0.0),
                "is_final": bool(stop_after_callback or k == niter),
            })
            del frame, estimated
        if "fwd" in locals():
            del fwd
        if stop_after_callback:
            break

    # Extract the image-sized region
    result = x_cur[slices]
    if optimized_squeeze_2d:
        result = result[0]

    # Remove offset
    if offset_val > 0.0:
        result = (result - offset_val).clamp(min=0.0)

    result_np = _to_numpy(result).astype(np.float32, copy=False)
    del result, img_t, psf_t, otf, otf_conj, W, x_prev, x_cur
    _release_cuda_cache()
    return {
        "result": result_np,
        "convergence": convergence_history,
        "iterations_used": iterations_used,
        "backend": backend_used,
        "work_shape": tuple(work_shape),
        "optimized_workspace_bytes": (
            optimized_context["workspace_bytes"] if optimized_context is not None else 0
        ),
        "effective_parameters": {
            "snr_requested": snr_settings["requested_snr"],
            "snr": effective_snr,
            "snr_reliability": (snr_settings.get("estimate") or {}).get("reliability"),
            "snr_mode": (snr_settings.get("estimate") or {}).get("mode"),
            "acuity": snr_settings["acuity"],
            "start": start,
            "prefilter_sigma": prefilter_sigma,
            "rel_threshold": rel_threshold,
            "offset": offset_val,
        },
    }


@_release_cuda_cache_after
def ci_sparse_hessian_deconvolve(
    image: np.ndarray,
    psf: np.ndarray,
    *,
    niter: int = 50,
    sparse_hessian_weight: float = 0.6,
    sparse_hessian_reg: float = 0.98,
    offset: Union[str, float] = "auto",
    prefilter_sigma: float = 0.0,
    start: str = "auto",
    background: Union[str, float] = "auto",
    convergence: str = "auto",
    rel_threshold: float = 0.005,
    check_every: int = 5,
    pixel_size_xy: Optional[float] = None,
    pixel_size_z: Optional[float] = None,
    device: Optional[str] = None,
    backend: str = BACKEND_AUTO,
    tiling: str = "auto",
    iteration_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    channel_index: int = 0,
) -> dict[str, Any]:
    """Sparse-Hessian / SPITFIRE-style deconvolution with alternating updates."""
    dev, optimized_ops, backend_used = _resolve_backend(backend, device)
    device = str(dev)
    psf_xy_est = max(psf.shape[-1], psf.shape[-2]) if psf.ndim >= 2 else 65
    n_tiles = _resolve_tiling(tiling, image.shape, device=device, psf_xy_est=psf_xy_est)
    if n_tiles > 1 and iteration_callback is not None:
        log.info(
            "Per-iteration image previews disabled for %d-tile execution; "
            "this does not disable the selected compute backend.", n_tiles,
        )
        iteration_callback = None
    if n_tiles > 1:
        return _ci_deconvolve_tiled(
            image, psf, n_tiles,
            solver=ci_sparse_hessian_deconvolve,
            niter=niter,
            sparse_hessian_weight=sparse_hessian_weight,
            sparse_hessian_reg=sparse_hessian_reg,
            offset=offset,
            prefilter_sigma=prefilter_sigma,
            start=start,
            background=background,
            convergence=convergence,
            rel_threshold=rel_threshold,
            check_every=check_every,
            pixel_size_xy=pixel_size_xy,
            pixel_size_z=pixel_size_z,
            device=device,
            backend=backend,
            iteration_callback=iteration_callback,
            channel_index=channel_index,
        )

    dtype = _pick_dtype(dev)

    if start not in START_MODES:
        start = "flat"

    if offset == "auto":
        offset_val = 5.0
    else:
        offset_val = max(float(offset), 0.0)

    sparse_hessian_weight = float(np.clip(sparse_hessian_weight, 0.0, 1.0))
    sparse_hessian_reg = float(np.clip(sparse_hessian_reg, 0.0, 1.0))

    log.info(
        "ci_sparse_hessian_deconvolve  backend=%s  device=%s  dtype=%s  shape=%s  niter=%d  "
        "weight=%.4g  reg=%.4g  offset=%.4g  prefilter_sigma=%.4g  start=%s  convergence=%s",
        backend_used, dev, dtype, image.shape, niter,
        sparse_hessian_weight, sparse_hessian_reg,
        offset_val, prefilter_sigma, start, convergence,
    )

    img_t = _to_tensor(image, dev, dtype)
    psf_t = _to_tensor(psf, dev, dtype)
    optimized_squeeze_2d = optimized_ops is not None and img_t.ndim == 2
    if optimized_squeeze_2d:
        img_t = img_t.unsqueeze(0)
        psf_t = psf_t.unsqueeze(0)

    if background == "auto":
        bg = max(_estimate_background(img_t), 1e-6)
    else:
        bg = max(float(background), 1e-6)
    log.info("  background=%.4g", bg)

    if offset_val > 0.0:
        img_t = img_t + offset_val
        bg = bg + offset_val

    if prefilter_sigma > 0.0:
        img_t = _anscombe_prefilter(img_t, prefilter_sigma)
        log.info("  prefilter_sigma=%.4g applied", prefilter_sigma)

    requested_start = start
    start = _resolve_start_mode(start, img_t, bg)
    if requested_start != start:
        log.info("  start=%s resolved to %s", requested_start, start)

    minimum_work_shape = tuple(si + sp - 1 for si, sp in zip(img_t.shape, psf_t.shape))
    axis_scales = _axis_scales(img_t.ndim, pixel_size_xy, pixel_size_z)
    z_scale = axis_scales[0] if img_t.ndim == 3 else 1.0

    optimized_context = None
    if optimized_ops is not None:
        try:
            optimized_context = _optimized_context(
                optimized_ops, psf_t, tuple(img_t.shape), minimum_work_shape,
            )
        except Exception as exc:
            if str(backend or BACKEND_AUTO).strip().lower() != BACKEND_AUTO:
                raise
            log.warning("Optimized CUDA setup failed; falling back to PyTorch CUDA: %s", exc)
            clear_optimized_context_cache()
            optimized_ops = None
            backend_used = BACKEND_PYTORCH_CUDA
            if optimized_squeeze_2d:
                img_t = img_t[0]
                psf_t = psf_t[0]
                optimized_squeeze_2d = False
                minimum_work_shape = tuple(
                    si + sp - 1 for si, sp in zip(img_t.shape, psf_t.shape)
                )
                axis_scales = _axis_scales(img_t.ndim, pixel_size_xy, pixel_size_z)
                z_scale = 1.0
    if optimized_context is not None:
        work_shape = optimized_context["work_shape"]
        otf = optimized_context["otf"]
        otf_conj = otf.conj()
        W = optimized_context["weights"]
    else:
        work_shape = minimum_work_shape
        otf, otf_conj = _prepare_otf(psf_t, work_shape)
        W = _bertero_weights(otf, otf_conj, img_t.shape, work_shape)

    d_work = torch.full(work_shape, bg, dtype=dtype, device=dev)
    slices = tuple(slice(0, s) for s in img_t.shape)
    d_work[slices] = img_t

    x_prev, x_cur = _initial_estimate(
        start,
        img_t,
        d_work,
        work_shape,
        slices,
        bg,
        dtype,
        dev,
    )
    del d_work
    x_prev = x_prev.clamp(min=bg)
    x_cur = x_cur.clamp(min=bg)

    with torch.no_grad():
        fwd0 = _forward_project(x_cur, otf, work_shape)[slices]
        data_scale = max(float(_poisson_nll(img_t, fwd0).detach()), 1e-6)
        prior_scale = max(
            float(_sparse_hessian_penalty(x_cur[slices], sparse_hessian_weight, z_scale=z_scale).detach()),
            1e-6,
        )
        del fwd0

    convergence_history: list[float] = []
    iterations_used = niter
    early_stop_min_iter = max(10, niter // 4)
    sparse_gradient = (
        torch.empty(tuple(img_t.shape), dtype=torch.float32, device=dev)
        if optimized_context is not None else None
    )

    for k in range(1, niter + 1):
        if k >= 3:
            alpha_max = 1.0 - 2.0 / math.sqrt(k + 3.0)
            alpha = min((k - 1.0) / (k + 2.0), alpha_max)
        else:
            alpha = 0.0
        if optimized_context is not None:
            storage = optimized_context["storage"]
            plan = optimized_context["plan"]
            workspace = optimized_context["workspace"]
            pitch = optimized_context["pitch"]
            normalization = optimized_context["normalization"]
            x_data = x_prev
            optimized_ops.momentum_pack(
                x_cur, x_data, alpha, bg, storage, *work_shape, pitch,
            )
            plan.forward(storage, workspace)
            optimized_ops.multiply_otf(optimized_context["frequency"], otf, False)
            plan.inverse(storage, workspace)
            optimized_ops.ratio_pitched(
                storage, img_t, *work_shape, pitch, bg, normalization,
            )
            plan.forward(storage, workspace)
            optimized_ops.multiply_otf(optimized_context["frequency"], otf, True)
            plan.inverse(storage, workspace)
            optimized_ops.update_pitched(
                x_data, storage, W, bg, normalization, *work_shape, pitch,
            )
            optimized_ops.sparse_hessian_gradient(
                x_data, sparse_gradient, *img_t.shape, sparse_hessian_weight, z_scale,
            )
            prior_grad = sparse_gradient
            grad_scale = prior_grad.abs().mean().detach().clamp(min=1e-12)
        else:
            p = (x_cur + alpha * (x_cur - x_prev)).clamp(min=bg)
            y = _forward_project(p, otf, work_shape)
            r = torch.zeros(work_shape, dtype=dtype, device=dev)
            r[slices] = img_t / y[slices].clamp(min=bg)
            corr = _irfft(_rfft(r) * otf_conj, work_shape)
            del y, r
            x_data = (p * corr * W).clamp(min=bg)
            del corr, p
            prior_probe = x_data[slices].detach().requires_grad_(True)
            prior_loss_probe = _sparse_hessian_penalty(
                prior_probe, sparse_hessian_weight, z_scale=z_scale,
            )
            prior_grad = torch.autograd.grad(prior_loss_probe, prior_probe)[0]
            grad_scale = prior_grad.abs().mean().detach().clamp(min=1e-12)
        signal_scale = max(float((x_data[slices].mean() - bg).detach()), 1.0)
        reg_step = 0.1 * max(1.0 - sparse_hessian_reg, 0.0) * signal_scale
        if optimized_context is not None:
            optimized_ops.sparse_hessian_update(
                x_data, prior_grad, *img_t.shape, reg_step / float(grad_scale), bg,
            )
            x_prev, x_cur = x_cur, x_data
        else:
            x_new = x_data.clone()
            x_new[slices] = (
                x_data[slices] - reg_step * prior_grad / grad_scale
            ).clamp(min=bg)
            del prior_probe, prior_loss_probe, prior_grad, x_data
            x_prev, x_cur = x_cur, x_new

        stop_after_callback = False
        convergence_value: Optional[float] = None
        check_iteration = k % check_every == 0 or k == niter
        if check_iteration:
            fwd = _forward_project(x_cur, otf, work_shape)[slices]
            data_loss = _poisson_nll(img_t, fwd)
            prior_loss = _sparse_hessian_penalty(
                x_cur[slices], sparse_hessian_weight, z_scale=z_scale,
            )
            total_loss = (
                sparse_hessian_reg * (data_loss / data_scale)
                + (1.0 - sparse_hessian_reg) * (prior_loss / prior_scale)
            )
            obj = float(total_loss.detach())
            convergence_history.append(obj)
            convergence_value = obj
            log.info(
                "  iter %4d/%d  objective=%.6g  data=%.6g  prior=%.6g",
                k, niter, obj, float(data_loss.detach()), float(prior_loss.detach()),
            )

            if convergence == "auto" and len(convergence_history) >= 2 and k > early_stop_min_iter:
                prev_obj = convergence_history[-2]
                if prev_obj > 0:
                    rel_change = (prev_obj - obj) / prev_obj
                    if rel_change < rel_threshold:
                        log.info("  converged at iter %d (rel_change=%.4g)", k, rel_change)
                        iterations_used = k
                        stop_after_callback = True

        if iteration_callback is not None:
            if not check_iteration:
                fwd = _forward_project(x_cur, otf, work_shape)[slices]
                data_loss = _poisson_nll(img_t, fwd)
                prior_loss = _sparse_hessian_penalty(
                    x_cur[slices], sparse_hessian_weight, z_scale=z_scale,
                )
                total_loss = (
                    sparse_hessian_reg * (data_loss / data_scale)
                    + (1.0 - sparse_hessian_reg) * (prior_loss / prior_scale)
                )
                convergence_value = float(total_loss.detach())
            frame = x_cur[slices]
            estimated = fwd
            if optimized_squeeze_2d:
                frame = frame[0]
                estimated = estimated[0]
            if offset_val > 0.0:
                frame = (frame - offset_val).clamp(min=0.0)
                estimated = (estimated - offset_val).clamp(min=0.0)
            iteration_callback({
                "channel_index": int(channel_index),
                "iteration": k,
                "total_iterations": niter,
                "convergence": convergence_value,
                "data_loss": float(data_loss.detach()),
                "prior_loss": float(prior_loss.detach()),
                "image": _to_numpy(frame),
                "estimated": _to_numpy(estimated),
                "background": max(float(bg) - offset_val, 0.0),
                "is_final": bool(stop_after_callback or k == niter),
            })
            del frame, estimated
        if "fwd" in locals():
            del fwd
        if "data_loss" in locals():
            del data_loss
        if "prior_loss" in locals():
            del prior_loss
        if "total_loss" in locals():
            del total_loss
        if stop_after_callback:
            break

    result = x_cur[slices]
    if optimized_squeeze_2d:
        result = result[0]
    if offset_val > 0.0:
        result = (result - offset_val).clamp(min=0.0)

    result_np = _to_numpy(result).astype(np.float32, copy=False)
    del result, img_t, psf_t, otf, otf_conj, W, x_prev, x_cur
    _release_cuda_cache()
    return {
        "result": result_np,
        "convergence": convergence_history,
        "iterations_used": iterations_used,
        "backend": backend_used,
        "work_shape": tuple(work_shape),
        "optimized_workspace_bytes": (
            optimized_context["workspace_bytes"] if optimized_context is not None else 0
        ),
    }


# ===================================================================
#  PART 2 — PSF generation
# ===================================================================

# ---------------------------------------------------------------------------
# Simpson's rule (1-D, torch)
# ---------------------------------------------------------------------------

def _simpsons(fs: torch.Tensor, dx: float) -> torch.Tensor:
    """Simpson's rule along dim 0.  *fs* must have odd size along dim 0."""
    return (fs[0] + 4.0 * torch.sum(fs[1:-1:2], dim=0)
            + 2.0 * torch.sum(fs[2:-1:2], dim=0) + fs[-1]) * dx / 3.0

# ---------------------------------------------------------------------------
# Gibson-Lanni OPD
# ---------------------------------------------------------------------------

def _gibson_lanni_opd(
    sin_t: torch.Tensor,
    *,
    z_p: float,
    n_s: float,
    n_i: float,
    n_i0: float,
    n_g: float,
    n_g0: float,
    t_g: float,
    t_g0: float,
    t_i0: float,
) -> torch.Tensor:
    """Optical Path Difference from Gibson-Lanni model.

    Parameters in **nanometres** (consistent with psf_generator convention).
    Returns the OPD tensor of the same shape as *sin_t*.
    """
    ni2_sin2 = n_i ** 2 * sin_t ** 2
    t_i = n_i * (t_g0 / n_g0 + t_i0 / n_i0 - t_g / n_g - z_p / n_s)

    opd = (z_p   * torch.sqrt((n_s  ** 2 - ni2_sin2).clamp(min=0))
         + t_i   * torch.sqrt((n_i  ** 2 - ni2_sin2).clamp(min=0))
         - t_i0  * torch.sqrt((n_i0 ** 2 - ni2_sin2).clamp(min=0))
         + t_g   * torch.sqrt((n_g  ** 2 - ni2_sin2).clamp(min=0))
         - t_g0  * torch.sqrt((n_g0 ** 2 - ni2_sin2).clamp(min=0)))
    return opd

# ---------------------------------------------------------------------------
# Scalar PSF slice (single z-plane)
# ---------------------------------------------------------------------------

def _scalar_psf_slice(
    k: float,
    thetas: torch.Tensor,
    dtheta: float,
    rs: torch.Tensor,
    pupil: torch.Tensor,
    defocus_phase: torch.Tensor,
) -> torch.Tensor:
    """Compute scalar PSF for unique radii at one z-plane.

    Returns complex field values for each unique radius.
    """
    sin_t = torch.sin(thetas)
    bessel_arg = k * rs[None, :] * sin_t[:, None]
    J0 = bessel_j0(bessel_arg)

    integrand = J0 * (pupil * defocus_phase * sin_t)[:, None]
    field = _simpsons(integrand, dtheta)
    return field

# ---------------------------------------------------------------------------
# Vectorial PSF slice (single z-plane)
# ---------------------------------------------------------------------------

def _vectorial_psf_slice(
    k: float,
    thetas: torch.Tensor,
    dtheta: float,
    rs: torch.Tensor,
    pupil: torch.Tensor,
    defocus_phase: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute I0, I1, I2 integrals for one z-plane (unpolarised)."""
    sin_t = torch.sin(thetas)
    cos_t = torch.cos(thetas)

    bessel_arg = k * rs[None, :] * sin_t[:, None]
    J0 = bessel_j0(bessel_arg)
    J1 = bessel_j1(bessel_arg)
    # J2 via recurrence: J2(x) = 2*J1(x)/x - J0(x)
    J2 = 2.0 * torch.where(
        bessel_arg.abs() > 1e-6,
        J1 / bessel_arg,
        torch.tensor(0.5, dtype=bessel_arg.dtype, device=bessel_arg.device),
    ) - J0

    base = pupil * defocus_phase * sin_t

    integrand_0 = J0 * (base * (1.0 + cos_t))[:, None]
    integrand_1 = J1 * (base * sin_t)[:, None]
    integrand_2 = J2 * (base * (1.0 - cos_t))[:, None]

    I0 = _simpsons(integrand_0, dtheta)
    I1 = _simpsons(integrand_1, dtheta)
    I2 = _simpsons(integrand_2, dtheta)
    return I0, I1, I2

# ---------------------------------------------------------------------------
# Sub-pixel integration wrapper
# ---------------------------------------------------------------------------

def _pixel_integrate_psf(
    psf_func,
    pixel_size_xy: float,
    n_xy: int,
    n_subpixels: int,
    **kwargs,
) -> torch.Tensor:
    """Compute PSF by averaging over sub-pixel grid positions.

    *psf_func(fov, n_xy, **kwargs)* returns a 3-D PSF for a given field-of-
    view and lateral pixel count. We shrink the effective pixel, evaluate on a
    finer grid, and block-average.
    """
    if n_subpixels <= 1:
        fov = pixel_size_xy * n_xy
        return psf_func(fov=fov, n_xy=n_xy, **kwargs)

    n_fine = n_xy * n_subpixels
    fov = pixel_size_xy * n_xy  # total field of view unchanged
    fine_psf = psf_func(fov=fov, n_xy=n_fine, **kwargs)  # (Z, fineY, fineX)

    # Block-average back to n_xy × n_xy
    nz = fine_psf.shape[0]
    fine_psf = fine_psf.reshape(nz, n_xy, n_subpixels, n_xy, n_subpixels)
    return fine_psf.mean(dim=(2, 4))


def _make_circular_pinhole_kernel(
    *,
    pinhole_airy_units: float,
    wavelength_nm: float,
    na: float,
    pixel_size_xy_nm: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a normalized circular pinhole aperture kernel."""
    airy_diameter_nm = 1.22 * wavelength_nm / max(float(na), 1e-12)
    diameter_px = pinhole_airy_units * airy_diameter_nm / pixel_size_xy_nm
    radius_px = max(diameter_px / 2.0, 0.0)
    half_size = max(1, int(math.ceil(radius_px)))
    coords = torch.arange(-half_size, half_size + 1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    kernel = ((xx ** 2 + yy ** 2) <= radius_px ** 2).to(dtype)
    if torch.count_nonzero(kernel) == 0:
        kernel[half_size, half_size] = 1.0
    kernel = kernel / kernel.sum()
    return kernel


def _convolve_lateral_with_kernel(psf: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Convolve each Z plane of a centered PSF with a lateral aperture kernel."""
    import torch.nn.functional as F

    if kernel.shape == (1, 1):
        return psf
    nz = psf.shape[0]
    image = psf.reshape(nz, 1, psf.shape[-2], psf.shape[-1])
    weight = kernel.reshape(1, 1, kernel.shape[0], kernel.shape[1])
    pad_y = kernel.shape[0] // 2
    pad_x = kernel.shape[1] // 2
    return F.conv2d(image, weight, padding=(pad_y, pad_x)).reshape_as(psf)

# ---------------------------------------------------------------------------
# Core PSF builder
# ---------------------------------------------------------------------------

def _build_psf_stack(
    *,
    fov: float,
    n_xy: int,
    n_z: int,
    wavelength_nm: float,
    na: float,
    ri_immersion: float,
    ri_sample: float,
    ri_coverslip: float,
    ri_coverslip_design: float,
    ri_immersion_design: float,
    t_g: float,
    t_g0: float,
    t_i0: float,
    z_p: float,
    pixel_size_z_nm: float,
    n_pupil: int,
    use_vectorial: bool,
    gibson_lanni: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build a 3-D PSF stack.  Returns tensor of shape (n_z, n_xy, n_xy)."""
    k = 2.0 * math.pi / wavelength_nm
    ri = ri_sample if gibson_lanni else ri_immersion

    # Theta grid (pupil samples)
    s_max = na / ri_immersion_design
    if s_max > 1.0:
        s_max = 1.0
    theta_max = math.asin(s_max)
    thetas = torch.linspace(0, theta_max, n_pupil, device=device, dtype=dtype)
    dtheta = theta_max / (n_pupil - 1)

    # PSF spatial coordinates — unique radii
    x = torch.linspace(-fov / 2.0, fov / 2.0, n_xy, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(x, x, indexing="ij")
    rr = torch.sqrt(xx ** 2 + yy ** 2)
    r_unique, rr_inv = torch.unique(rr, return_inverse=True)
    rs = r_unique  # (n_unique,)

    # Bessel argument scaling is handled inside slice functions via k*ri

    # Correction / pupil
    sin_t = torch.sin(thetas)
    cos_t = torch.cos(thetas)
    pupil = torch.sqrt(cos_t).to(torch.complex128 if dtype == torch.float64 else torch.complex64)

    if gibson_lanni:
        clamp_val = min(ri_sample / ri_immersion, ri_coverslip / ri_immersion)
        sin_t_gl = sin_t.clamp(max=clamp_val)
        opd = _gibson_lanni_opd(
            sin_t_gl,
            z_p=z_p, n_s=ri_sample, n_i=ri_immersion, n_i0=ri_immersion_design,
            n_g=ri_coverslip, n_g0=ri_coverslip_design,
            t_g=t_g, t_g0=t_g0, t_i0=t_i0,
        )
        pupil = pupil * torch.exp(1j * k * opd.to(pupil.dtype))

    # Scale rs for Bessel arg — handled inside slice functions

    # Z planes
    z_min = -pixel_size_z_nm * (n_z // 2)
    zs = torch.linspace(z_min, -z_min, n_z, device=device, dtype=dtype)

    cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64

    slices_out = []
    for zi in range(n_z):
        z = zs[zi]
        defocus = torch.exp(1j * k * z * cos_t * ri).to(cdtype)

        if use_vectorial:
            I0, I1, I2 = _vectorial_psf_slice(

                k * ri, thetas, dtheta, rs, pupil.to(cdtype), defocus,
            )
            # Intensity = |I0|^2 + 2|I1|^2 + |I2|^2  (unpolarised)
            intensity = (I0.abs() ** 2 + 2.0 * I1.abs() ** 2 + I2.abs() ** 2)
        else:
            field = _scalar_psf_slice(
                k * ri, thetas, dtheta, rs, pupil.to(cdtype), defocus,
            )
            intensity = field.abs() ** 2

        # Scatter from unique radii to 2-D grid
        plane = intensity[rr_inv.flatten()].reshape(n_xy, n_xy)
        slices_out.append(plane)

    psf_stack = torch.stack(slices_out, dim=0).to(dtype)  # (Z, Y, X)
    del slices_out, rr, r_unique, rr_inv, rs, thetas, zs, pupil
    return psf_stack

# ---------------------------------------------------------------------------
# Top-level PSF function
# ---------------------------------------------------------------------------

def ci_generate_psf(
    na: float,
    wavelength_nm: float,
    pixel_size_xy_nm: float,
    pixel_size_z_nm: float,
    n_xy: int,
    n_z: int,
    *,
    ri_immersion: float = 1.515,
    ri_sample: float = 1.33,
    ri_coverslip: float = 1.5,
    ri_coverslip_design: float = 1.5,
    ri_immersion_design: float = 1.515,
    t_g: float = 170e3,
    t_g0: float = 170e3,
    t_i0: float = 100e3,
    z_p: float = 0.0,
    microscope_type: str = "widefield",
    excitation_nm: Optional[float] = None,
    pinhole_airy_units: float = 1.0,
    integrate_pixels: bool = True,
    n_subpixels: int = 3,
    n_pupil: int = 129,
    device: Optional[str] = None,
) -> np.ndarray:
    """Generate a physically accurate 3-D PSF.

    Parameters
    ----------
    na : float
        Numerical aperture.
    wavelength_nm : float
        Emission wavelength in nm.
    pixel_size_xy_nm, pixel_size_z_nm : float
        Pixel sizes in nm.
    n_xy, n_z : int
        Lateral / axial pixel counts (should be odd).
    ri_immersion, ri_sample, ri_coverslip : float
        Refractive indices for immersion medium, sample, and coverslip.
    ri_coverslip_design, ri_immersion_design : float
        Design (nominal) RI values for coverslip and immersion.
    t_g, t_g0 : float
        Actual and design coverslip thickness in nm.
    t_i0 : float
        Design immersion thickness in nm.
    z_p : float
        Depth of the particle below coverslip in nm.
    microscope_type : str
        ``"widefield"`` or ``"confocal"``.
    excitation_nm : float or None
        Excitation wavelength for confocal.
    pinhole_airy_units : float
        Confocal pinhole diameter in Airy disk units. ``0`` keeps the legacy
        point-detector model; values > 0 convolve the detection PSF laterally
        with a circular pinhole aperture before multiplying by excitation.
    integrate_pixels : bool
        Integrate over pixel area (more accurate, slower).
    n_subpixels : int
        Sub-pixel grid size per axis for pixel integration.
    n_pupil : int
        Number of pupil integration samples (should be odd).
    device : str or None
        PyTorch device.

    Returns
    -------
    ndarray
        Normalised PSF (sum = 1), shape ``(n_z, n_xy, n_xy)``.
    """
    dev = _pick_device(device)
    dtype = _pick_dtype(dev)

    use_vectorial = na >= 0.9
    gibson_lanni = (abs(ri_sample - ri_immersion) > 0.001
                    or abs(ri_coverslip - ri_coverslip_design) > 0.001
                    or z_p > 0)

    log.info("ci_generate_psf  NA=%.2f  λ=%gnm  pixel_xy=%gnm  pixel_z=%gnm  "
             "size=%dx%dx%d  vectorial=%s  GL=%s  device=%s",
             na, wavelength_nm, pixel_size_xy_nm, pixel_size_z_nm,
             n_xy, n_xy, n_z, use_vectorial, gibson_lanni, dev)

    common = dict(
        n_z=n_z,
        wavelength_nm=wavelength_nm,
        na=na,
        ri_immersion=ri_immersion,
        ri_sample=ri_sample,
        ri_coverslip=ri_coverslip,
        ri_coverslip_design=ri_coverslip_design,
        ri_immersion_design=ri_immersion_design,
        t_g=t_g, t_g0=t_g0, t_i0=t_i0, z_p=z_p,
        pixel_size_z_nm=pixel_size_z_nm,
        n_pupil=n_pupil,
        use_vectorial=use_vectorial,
        gibson_lanni=gibson_lanni,
        device=dev,
        dtype=dtype,
    )

    def _psf_func(*, fov: float, n_xy: int, **kw) -> torch.Tensor:
        return _build_psf_stack(fov=fov, n_xy=n_xy, **{**common, **kw})

    try:
        if integrate_pixels and n_subpixels > 1:
            psf = _pixel_integrate_psf(
                _psf_func,
                pixel_size_xy=pixel_size_xy_nm,
                n_xy=n_xy,
                n_subpixels=n_subpixels,
            )
        else:
            fov = pixel_size_xy_nm * n_xy
            psf = _psf_func(fov=fov, n_xy=n_xy)

        # Confocal: detection PSF × excitation PSF. Finite pinholes are modelled
        # by laterally integrating the detection/emission PSF over a circular
        # object-space aperture measured in Airy disk units.
        if microscope_type == "confocal":
            detector_psf = psf
            pinhole_airy_units = float(pinhole_airy_units)
            if pinhole_airy_units > 0.0:
                kernel = _make_circular_pinhole_kernel(
                    pinhole_airy_units=pinhole_airy_units,
                    wavelength_nm=wavelength_nm,
                    na=na,
                    pixel_size_xy_nm=pixel_size_xy_nm,
                    device=dev,
                    dtype=dtype,
                )
                detector_psf = _convolve_lateral_with_kernel(detector_psf, kernel)
                del kernel

            if excitation_nm is not None and excitation_nm != wavelength_nm:
                common_ex = {**common, "wavelength_nm": excitation_nm}

                def _psf_ex(*, fov, n_xy, **kw):
                    return _build_psf_stack(fov=fov, n_xy=n_xy, **{**common_ex, **kw})

                if integrate_pixels and n_subpixels > 1:
                    psf_ex = _pixel_integrate_psf(
                        _psf_ex,
                        pixel_size_xy=pixel_size_xy_nm,
                        n_xy=n_xy,
                        n_subpixels=n_subpixels,
                    )
                else:
                    fov = pixel_size_xy_nm * n_xy
                    psf_ex = _psf_ex(fov=fov, n_xy=n_xy)
                psf = detector_psf * psf_ex
                del psf_ex
            else:
                psf = detector_psf * psf
            del detector_psf

        # Normalise
        psf = psf / psf.sum()

        result = _to_numpy(psf).astype(np.float32, copy=False)
        log.info("  PSF range [%.3g, %.3g], sum=%.6f", result.min(), result.max(),
                 result.sum())
        return result
    finally:
        if "psf" in locals():
            del psf
        _release_cuda_cache()
