"""Experimental ci_rl + 2.5D residual U-Net refinement.

This module wraps the existing physics-based ``ci_rl_deconvolve``
implementation instead of changing it. Internal volume layout is ``Z, Y, X``;
2-D inputs are promoted to a singleton ``Z`` axis during refinement and
squeezed back to the original shape at the end.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from deconvolve_ci import (
    _ci_deconvolve_tiled,
    _forward_project,
    _get_memory_budget_bytes,
    _pick_device,
    _prepare_otf,
    _resolve_tiling,
    _rfft,
    _irfft,
    _to_tensor,
    ci_rl_deconvolve,
)

log = logging.getLogger(__name__)


def _as_zyx(image: np.ndarray) -> tuple[np.ndarray, bool]:
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 2:
        return arr[np.newaxis, ...], True
    if arr.ndim != 3:
        raise ValueError(f"ci_rl_dl supports 2-D or 3-D single-channel arrays, got {arr.shape}")
    return arr, False


def _restore_shape(volume: np.ndarray, was_2d: bool) -> np.ndarray:
    return volume[0] if was_2d else volume


def resolve_torch_device(device: str | torch.device | None = "auto") -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device is None or str(device).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested = torch.device(str(device))
    if requested.type == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA requested for ci_rl_dl, but CUDA is unavailable; falling back to CPU")
        return torch.device("cpu")
    return requested


def _module_device(module: nn.Module) -> torch.device | None:
    for tensor in module.parameters():
        return tensor.device
    for tensor in module.buffers():
        return tensor.device
    return None


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int = 8) -> None:
        super().__init__()
        group_count = min(groups, out_channels)
        while out_channels % group_count != 0 and group_count > 1:
            group_count -= 1
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(group_count, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(group_count, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualUNet25D(nn.Module):
    """Small 2-D U-Net for 2.5D residual prediction.

    Input shape is ``N, C, Y, X`` and output shape is ``N, 1, Y, X``.
    """

    def __init__(
        self,
        input_channels: int,
        base_channels: int = 16,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.base_channels = int(base_channels)
        self.residual_scale = float(residual_scale)

        c = self.base_channels
        self.enc1 = ConvBlock(self.input_channels, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4)
        self.bottleneck = ConvBlock(c * 4, c * 8)
        self.up3 = nn.Conv2d(c * 8, c * 4, kernel_size=1)
        self.dec3 = ConvBlock(c * 8, c * 4)
        self.up2 = nn.Conv2d(c * 4, c * 2, kernel_size=1)
        self.dec2 = ConvBlock(c * 4, c * 2)
        self.up1 = nn.Conv2d(c * 2, c, kernel_size=1)
        self.dec1 = ConvBlock(c * 2, c)
        self.out = nn.Conv2d(c, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        b = self.bottleneck(F.avg_pool2d(e3, 2))

        u3 = F.interpolate(b, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([self.up3(u3), e3], dim=1))
        u2 = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([self.up2(u2), e2], dim=1))
        u1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([self.up1(u1), e1], dim=1))
        return self.out(d1) * self.residual_scale


CONDITIONING_CHANNELS = [
    "rl_iterations_norm",
    "psf_sigma_xy_norm",
    "psf_sigma_z_norm",
    "pixel_size_xy_norm",
    "pixel_size_z_norm",
    "na_norm",
    "emission_wavelength_norm",
    "microscope_widefield",
    "microscope_confocal",
]


class GatedResidualUNet25D(ResidualUNet25D):
    """2.5D U-Net with a gated, bounded residual output."""

    def __init__(
        self,
        input_channels: int,
        base_channels: int = 16,
        residual_scale: float = 1.0,
        z_radius: int = 2,
        use_residual_channel: bool = True,
        residual_bound_fraction: float = 0.35,
        residual_bound_scale: float = 0.05,
    ) -> None:
        super().__init__(input_channels, base_channels=base_channels, residual_scale=residual_scale)
        self.z_radius = int(z_radius)
        self.use_residual_channel = bool(use_residual_channel)
        self.residual_bound_fraction = float(residual_bound_fraction)
        self.residual_bound_scale = float(residual_bound_scale)
        self.out = nn.Conv2d(self.base_channels, 2, kernel_size=1)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        b = self.bottleneck(F.avg_pool2d(e3, 2))

        u3 = F.interpolate(b, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([self.up3(u3), e3], dim=1))
        u2 = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([self.up2(u2), e2], dim=1))
        u1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        return self.dec1(torch.cat([self.up1(u1), e1], dim=1))

    def _central_ci(self, x: torch.Tensor) -> torch.Tensor:
        n_context = 2 * self.z_radius + 1
        ci_index = n_context + self.z_radius
        if ci_index >= x.shape[1]:
            return torch.zeros((x.shape[0], 1, *x.shape[-2:]), dtype=x.dtype, device=x.device)
        return x[:, ci_index:ci_index + 1].clamp(min=0)

    def forward_details(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        raw_out = self.out(self._features(x))
        proposal = torch.tanh(raw_out[:, 0:1])
        gate = torch.sigmoid(raw_out[:, 1:2])
        ci = self._central_ci(x)
        bound = self.residual_bound_fraction * ci + self.residual_bound_scale
        bounded_residual = proposal * bound
        residual = gate * bounded_residual * self.residual_scale
        return {
            "residual": residual,
            "gate": gate,
            "proposal": proposal,
            "bound": bound,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_details(x)["residual"]


def input_channel_count(
    z_radius: int,
    use_residual_channel: bool = True,
    conditioning_channels: int | Sequence[str] = 0,
) -> int:
    n_conditioning = (
        len(conditioning_channels)
        if isinstance(conditioning_channels, Sequence) and not isinstance(conditioning_channels, (str, bytes))
        else int(conditioning_channels)
    )
    return 2 * (2 * int(z_radius) + 1) + (1 if use_residual_channel else 0) + n_conditioning


def _robust_scale(*arrays: np.ndarray) -> float:
    samples = [np.asarray(a, dtype=np.float32).reshape(-1) for a in arrays if a.size]
    if not samples:
        return 1.0
    values = np.concatenate(samples)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    return max(float(np.percentile(np.clip(finite, 0, None), 99.5)), 1e-6)


def make_25d_input(
    raw_zyx: np.ndarray,
    deconv_zyx: np.ndarray,
    z_index: int,
    *,
    z_radius: int = 2,
    use_residual_channel: bool = True,
    conditioning_values: Optional[Sequence[float]] = None,
    scale: Optional[float] = None,
) -> tuple[np.ndarray, float]:
    """Build one normalized 2.5D input tensor in ``C, Y, X`` layout."""
    raw = np.asarray(raw_zyx, dtype=np.float32)
    deconv = np.asarray(deconv_zyx, dtype=np.float32)
    if raw.shape != deconv.shape:
        raise ValueError(f"raw and deconv shapes differ: {raw.shape} vs {deconv.shape}")
    if raw.ndim != 3:
        raise ValueError("make_25d_input expects Z, Y, X arrays")

    scale_f = float(scale) if scale is not None else _robust_scale(raw, deconv)
    z_indices = np.clip(
        np.arange(int(z_index) - z_radius, int(z_index) + z_radius + 1),
        0,
        raw.shape[0] - 1,
    )
    channels = [raw[z] / scale_f for z in z_indices]
    channels.extend(deconv[z] / scale_f for z in z_indices)
    if use_residual_channel:
        channels.append((raw[int(z_index)] - deconv[int(z_index)]) / scale_f)
    if conditioning_values is not None:
        for value in conditioning_values:
            channels.append(np.full(raw.shape[1:], float(value), dtype=np.float32))
    return np.stack(channels, axis=0).astype(np.float32), scale_f


def _reflect_pad_xy(volume: np.ndarray, padding: int) -> np.ndarray:
    pad = max(int(padding), 0)
    if pad <= 0:
        return np.asarray(volume, dtype=np.float32)
    arr = np.asarray(volume, dtype=np.float32)
    mode = "reflect" if arr.shape[-2] > 1 and arr.shape[-1] > 1 else "edge"
    return np.pad(arr, ((0, 0), (pad, pad), (pad, pad)), mode=mode).astype(np.float32)


def _crop_xy_padding(volume: np.ndarray, padding: int) -> np.ndarray:
    pad = max(int(padding), 0)
    if pad <= 0:
        return np.asarray(volume, dtype=np.float32)
    return np.asarray(volume, dtype=np.float32)[:, pad:-pad, pad:-pad]


def psf_sigma_summary(psf: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(psf, dtype=np.float32)
    if arr.size == 0 or float(arr.sum()) <= 0:
        return 0.0, 0.0
    arr = np.clip(arr, 0, None)
    arr = arr / max(float(arr.sum()), 1e-12)
    coords = np.indices(arr.shape, dtype=np.float32)
    center = [float((coords[axis] * arr).sum()) for axis in range(arr.ndim)]
    variances = [float((((coords[axis] - center[axis]) ** 2) * arr).sum()) for axis in range(arr.ndim)]
    if arr.ndim == 2:
        sigma_yx = float(np.sqrt(max((variances[0] + variances[1]) / 2.0, 0.0)))
        return sigma_yx, 0.0
    sigma_z = float(np.sqrt(max(variances[0], 0.0)))
    sigma_xy = float(np.sqrt(max((variances[-2] + variances[-1]) / 2.0, 0.0)))
    return sigma_xy, sigma_z


def conditioning_vector(
    *,
    psf: Optional[np.ndarray] = None,
    metadata: Optional[dict[str, Any]] = None,
    rl_iterations: Optional[int] = None,
    microscope_type: Optional[str] = None,
) -> list[float]:
    metadata = dict(metadata or {})
    psf_params = dict(metadata.get("psf_params") or metadata.get("deconv_psf_params") or {})
    sigma_xy, sigma_z = psf_sigma_summary(psf) if psf is not None else (0.0, 0.0)
    micro = str(microscope_type or psf_params.get("microscope_type") or metadata.get("microscope_type") or "").lower()
    pixel_xy = float(psf_params.get("pixel_size_xy_nm") or metadata.get("pixel_size_xy_nm") or 0.0)
    pixel_z = float(psf_params.get("pixel_size_z_nm") or metadata.get("pixel_size_z_nm") or 0.0)
    if 0.0 < pixel_xy < 10.0:
        pixel_xy *= 1000.0
    if 0.0 < pixel_z < 10.0:
        pixel_z *= 1000.0
    return [
        float(rl_iterations or metadata.get("rl_requested_iterations") or 0) / 100.0,
        sigma_xy / 20.0,
        sigma_z / 20.0,
        pixel_xy / 200.0,
        pixel_z / 600.0,
        float(psf_params.get("na") or metadata.get("na") or 0.0) / 1.5,
        float(psf_params.get("wavelength_nm") or metadata.get("emission_wavelength_nm") or 0.0) / 700.0,
        1.0 if micro == "widefield" else 0.0,
        1.0 if micro == "confocal" else 0.0,
    ]


def reconvolve_same(
    volume: np.ndarray,
    psf: np.ndarray,
    *,
    device: str | torch.device | None = "auto",
) -> np.ndarray:
    """Convolve ``volume`` with ``psf`` and crop to the input image support."""
    vol, was_2d = _as_zyx(np.asarray(volume, dtype=np.float32))
    psf_arr = np.asarray(psf, dtype=np.float32)
    if was_2d and psf_arr.ndim == 3:
        psf_arr = psf_arr[psf_arr.shape[0] // 2]
    if psf_arr.ndim == 2:
        work_image = vol[0]
    else:
        work_image = vol
    dev = resolve_torch_device(device)
    dtype = torch.float32
    image_t = _to_tensor(work_image.astype(np.float32), dev, dtype)
    psf_t = _to_tensor(psf_arr.astype(np.float32), dev, dtype)
    work_shape = tuple(si + sp - 1 for si, sp in zip(image_t.shape, psf_t.shape))
    otf, _ = _prepare_otf(psf_t, work_shape)
    work = torch.zeros(work_shape, dtype=dtype, device=dev)
    slices = tuple(slice(0, s) for s in image_t.shape)
    work[slices] = image_t
    result = _forward_project(work, otf, work_shape)[slices]
    out = result.detach().cpu().numpy().astype(np.float32)
    if was_2d:
        return out
    return out


def _ci_rl_dl_auto_n_tiles(
    shape: tuple[int, ...],
    *,
    device: str | torch.device | None,
    psf_xy_est: int,
    batch_size: int,
) -> int:
    """Choose a more conservative XY tile count for RL+DL refinement."""
    if len(shape) < 3:
        return 1
    n_z, height, width = (int(shape[0]), int(shape[1]), int(shape[2]))
    dev = resolve_torch_device(device)
    budget = _get_memory_budget_bytes(str(dev))
    if dev.type == "cuda":
        batch_factor = max(1.0, math.sqrt(max(int(batch_size), 1) / 8.0))
        budget = int(budget * 0.90 / batch_factor)
    else:
        budget = int(budget * 0.75)
    padded_z = max(1, 4 * n_z) if n_z > 1 else 1
    inner_sq = budget / max(64 * padded_z, 1)
    max_xy = int(math.sqrt(max(inner_sq, 0.0))) - int(psf_xy_est)
    max_xy = max(256, min(max_xy, 4096))
    ny = max(1, -(-height // max_xy))
    nx = max(1, -(-width // max_xy))
    n_tiles = ny * nx
    if n_tiles > 1:
        log.info(
            "ci_rl_dl auto-tiling: image z=%d h=%d w=%d, max_tile_xy=%d -> %dx%d grid",
            n_z, height, width, max_xy, ny, nx,
        )
    return 1 if n_tiles <= 1 else n_tiles


def load_residual_unet_checkpoint(
    model_path: str | Path,
    *,
    device: str | torch.device | None = "auto",
) -> tuple[nn.Module, dict[str, Any]]:
    path = Path(model_path)
    dev = resolve_torch_device(device)
    checkpoint = torch.load(path, map_location=dev, weights_only=False)
    sidecar = path.with_suffix(".json")
    if sidecar.exists() and isinstance(checkpoint, dict):
        try:
            checkpoint["sidecar_metadata"] = json.loads(sidecar.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("Could not read ci_rl_dl sidecar metadata %s: %s", sidecar, exc)
    sidecar_metadata = checkpoint.get("sidecar_metadata") if isinstance(checkpoint, dict) else {}
    model_kwargs = dict(checkpoint.get("model_kwargs") or (sidecar_metadata or {}).get("model_kwargs") or {})
    if "input_channels" not in model_kwargs:
        model_kwargs["input_channels"] = int(checkpoint.get("input_channels", 11))
    model_type = str(
        checkpoint.get("model_type")
        or (sidecar_metadata or {}).get("model_type")
        or model_kwargs.pop("model_type", "")
        or "ResidualUNet25D"
    )
    if model_type == "GatedResidualUNet25D":
        model = GatedResidualUNet25D(**model_kwargs).to(dev)
    else:
        model = ResidualUNet25D(**model_kwargs).to(dev)
    state = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state)
    model.eval()
    return model, checkpoint if isinstance(checkpoint, dict) else {}


def load_checkpoint_sidecar(model_path: str | Path | None) -> dict[str, Any]:
    if model_path is None:
        return {}
    sidecar = Path(model_path).with_suffix(".json")
    if not sidecar.exists():
        return {}
    try:
        return json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Could not read ci_rl_dl sidecar metadata %s: %s", sidecar, exc)
        return {}


@torch.no_grad()
def apply_dl_refinement_25d(
    raw: np.ndarray,
    deconv: np.ndarray,
    model: nn.Module,
    psf: Optional[np.ndarray] = None,
    *,
    device: str | torch.device | None = "auto",
    z_radius: int = 2,
    batch_size: int = 8,
    use_residual_channel: bool = True,
    use_forward_projection_channel: bool = False,
    clamp_nonnegative: bool = True,
    mixed_precision: bool = True,
    residual_strength: float = 1.0,
    xy_padding: int = 0,
    conditioning_values: Optional[Sequence[float]] = None,
    compute_reconvolution_diagnostics: bool = False,
) -> dict[str, Any]:
    """Apply a trained 2.5D residual model slice by slice."""
    if use_forward_projection_channel:
        raise NotImplementedError("forward-projection input channels are reserved for a later model version")
    raw_zyx, was_2d = _as_zyx(np.asarray(raw, dtype=np.float32))
    deconv_zyx, _ = _as_zyx(np.asarray(deconv, dtype=np.float32))
    if raw_zyx.shape != deconv_zyx.shape:
        raise ValueError(f"raw and deconv shapes differ: {raw_zyx.shape} vs {deconv_zyx.shape}")

    dev = resolve_torch_device(device)
    if _module_device(model) != dev:
        model = model.to(dev)
    model.eval()
    scale = _robust_scale(raw_zyx, deconv_zyx)
    xy_padding_i = max(int(xy_padding), 0)
    raw_work = _reflect_pad_xy(raw_zyx, xy_padding_i)
    deconv_work = _reflect_pad_xy(deconv_zyx, xy_padding_i)
    residual_work = np.zeros_like(deconv_work, dtype=np.float32)
    z_values = list(range(raw_work.shape[0]))
    autocast_enabled = bool(mixed_precision and dev.type == "cuda")

    for start in range(0, len(z_values), max(int(batch_size), 1)):
        batch_z = z_values[start:start + max(int(batch_size), 1)]
        batch = [
            make_25d_input(
                raw_work,
                deconv_work,
                z,
                z_radius=z_radius,
                use_residual_channel=use_residual_channel,
                conditioning_values=conditioning_values,
                scale=scale,
            )[0]
            for z in batch_z
        ]
        x = torch.from_numpy(np.stack(batch, axis=0)).to(dev)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=autocast_enabled):
            pred = model(x)
        pred_np = pred.detach().float().cpu().numpy()[:, 0] * scale
        for local_idx, z in enumerate(batch_z):
            residual_work[z] = pred_np[local_idx]

    residual_strength_f = float(residual_strength)
    residual = _crop_xy_padding(residual_work, xy_padding_i)
    refined = deconv_zyx + residual_strength_f * residual
    if clamp_nonnegative:
        refined = np.clip(refined, 0, None)

    diagnostics: dict[str, Any] = {
        "dl_input_channels": input_channel_count(z_radius, use_residual_channel, len(conditioning_values or [])),
        "dl_z_radius": int(z_radius),
        "dl_residual_strength": residual_strength_f,
        "dl_xy_padding": int(xy_padding_i),
        "dl_conditioning_channels": len(conditioning_values or []),
        "intensity_sum_before_dl": float(np.sum(deconv_zyx)),
        "intensity_sum_after_dl": float(np.sum(refined)),
    }
    if psf is not None and bool(compute_reconvolution_diagnostics):
        try:
            recon_before = reconvolve_same(deconv_zyx, psf, device=dev)
            recon_after = reconvolve_same(refined, psf, device=dev)
            diagnostics["reconvolution_error_before"] = float(np.mean(np.abs(recon_before - raw_zyx)))
            diagnostics["reconvolution_error_after"] = float(np.mean(np.abs(recon_after - raw_zyx)))
        except Exception as exc:
            diagnostics["reconvolution_error"] = f"unavailable: {exc}"

    return {
        "result": _restore_shape(refined.astype(np.float32), was_2d),
        "residual": _restore_shape((residual_strength_f * residual).astype(np.float32), was_2d),
        "raw_residual": _restore_shape(residual.astype(np.float32), was_2d),
        "diagnostics": diagnostics,
    }


def deconvolve_ci_rl_dl(
    image: np.ndarray,
    psf: np.ndarray,
    *,
    model_path: str | Path | None = None,
    optical_params: Optional[dict[str, Any]] = None,
    device: str | None = "auto",
    rl_kwargs: Optional[dict[str, Any]] = None,
    dl_kwargs: Optional[dict[str, Any]] = None,
    return_diagnostics: bool = False,
) -> dict[str, Any] | np.ndarray:
    """Run existing ci_rl followed by optional 2.5D DL residual refinement."""
    image_arr = np.asarray(image)

    if image_arr.ndim == 4:
        channels = []
        channel_diagnostics = []
        psf_seq: Sequence[Any]
        if isinstance(psf, Sequence) and not isinstance(psf, np.ndarray):
            psf_seq = psf
        else:
            psf_arr = np.asarray(psf)
            if psf_arr.ndim == 4 and psf_arr.shape[0] == image_arr.shape[0]:
                psf_seq = [psf_arr[i] for i in range(psf_arr.shape[0])]
            else:
                psf_seq = [psf_arr] * image_arr.shape[0]
        model_paths: Sequence[Any]
        if isinstance(model_path, Sequence) and not isinstance(model_path, (str, bytes, Path)):
            model_paths = model_path
        else:
            model_paths = [model_path] * image_arr.shape[0]

        for ch in range(image_arr.shape[0]):
            ch_out = deconvolve_ci_rl_dl(
                image_arr[ch],
                psf_seq[ch] if ch < len(psf_seq) else psf_seq[-1],
                model_path=model_paths[ch] if ch < len(model_paths) else model_paths[-1],
                optical_params=optical_params,
                device=device,
                rl_kwargs=rl_kwargs,
                dl_kwargs=dl_kwargs,
                return_diagnostics=return_diagnostics,
            )
            if return_diagnostics:
                channels.append(ch_out["result"])
                channel_diagnostics.append(ch_out)
            else:
                channels.append(ch_out)
        stacked = np.stack(channels, axis=0).astype(np.float32)
        if return_diagnostics:
            return {
                "result": stacked,
                "channels": channel_diagnostics,
                "diagnostics": {"n_channels": int(image_arr.shape[0]), "layout": "CZYX"},
            }
        return stacked

    metadata = load_checkpoint_sidecar(model_path)
    inference_defaults = dict(metadata.get("recommended_inference") or {})

    rl_options = dict(inference_defaults.get("rl_kwargs") or {})
    rl_options.update(dict(rl_kwargs or {}))
    if device is not None and str(device).lower() != "auto":
        rl_options.setdefault("device", device)
    elif device == "auto":
        rl_options.setdefault("device", None)

    psf_arr = np.asarray(psf)
    psf_xy_est = max(psf_arr.shape[-1], psf_arr.shape[-2]) if psf_arr.ndim >= 2 else 65
    tiling_mode = rl_options.get("tiling", "auto")
    dl_options_for_tiling = {**dict(inference_defaults.get("dl_kwargs") or {}), **dict(dl_kwargs or {})}
    dl_batch_size_for_tiling = int(dl_options_for_tiling.get("batch_size", 8) or 8)
    if model_path is not None and image_arr.ndim == 3 and str(tiling_mode).strip().lower() in {"auto", "custom"}:
        n_tiles = _ci_rl_dl_auto_n_tiles(
            image_arr.shape,
            device=rl_options.get("device", device),
            psf_xy_est=psf_xy_est,
            batch_size=dl_batch_size_for_tiling,
        )
    else:
        n_tiles = _resolve_tiling(
            tiling_mode,
            image_arr.shape,
            device=rl_options.get("device", device),
            psf_xy_est=psf_xy_est,
        )
        if (
            model_path is not None
            and image_arr.ndim == 3
            and not (isinstance(tiling_mode, str) and str(tiling_mode).strip().lower() == "none")
        ):
            n_tiles = max(
                n_tiles,
                _ci_rl_dl_auto_n_tiles(
                    image_arr.shape,
                    device=rl_options.get("device", device),
                    psf_xy_est=psf_xy_est,
                    batch_size=dl_batch_size_for_tiling,
                ),
            )
    if model_path is not None and image_arr.ndim == 3 and n_tiles > 1:
        log.info("ci_rl_dl tiled refinement: %d XY tiles", n_tiles)
        dl_options = dict(dl_kwargs or {})
        model, checkpoint = load_residual_unet_checkpoint(model_path, device=device)
        checkpoint_metadata = checkpoint.get("sidecar_metadata") or checkpoint
        dl_defaults = dict(checkpoint_metadata.get("recommended_inference") or {})
        dl_options = {**dict(dl_defaults.get("dl_kwargs") or {}), **dl_options}
        dl_options.pop("compute_reconvolution_diagnostics", None)
        z_radius = int(dl_options.pop("z_radius", checkpoint.get("z_radius", 2)))
        use_residual_channel = bool(
            dl_options.pop("use_residual_channel", checkpoint.get("use_residual_channel", True))
        )
        conditioning_channels = list(checkpoint_metadata.get("conditioning_channels") or [])
        conditioning_values = None
        if conditioning_channels:
            conditioning_values = conditioning_vector(
                psf=np.asarray(psf_arr, dtype=np.float32),
                metadata=optical_params or {},
                rl_iterations=int(rl_options.get("niter", 0) or 0),
                microscope_type=(optical_params or {}).get("microscope_type") if optical_params else None,
            )

        def _tile_solver(tile_img: np.ndarray, tile_psf: np.ndarray, **_unused: Any) -> dict[str, Any]:
            tile_rl_options = dict(rl_options)
            tile_rl_options["tiling"] = "none"
            tile_rl_out = ci_rl_deconvolve(tile_img, np.asarray(tile_psf), **tile_rl_options)
            tile_ci_rl = np.asarray(tile_rl_out["result"], dtype=np.float32)
            refined_out = apply_dl_refinement_25d(
                np.asarray(tile_img, dtype=np.float32),
                tile_ci_rl,
                model,
                psf=np.asarray(tile_psf, dtype=np.float32),
                device=device,
                z_radius=z_radius,
                use_residual_channel=use_residual_channel,
                conditioning_values=conditioning_values,
                compute_reconvolution_diagnostics=False,
                **dl_options,
            )
            return {
                "result": np.clip(refined_out["result"], 0, None).astype(np.float32),
                "convergence": tile_rl_out.get("convergence", []),
                "iterations_used": int(tile_rl_out.get("iterations_used", 0)),
            }

        tiled_out = _ci_deconvolve_tiled(
            np.asarray(image_arr, dtype=np.float32),
            np.asarray(psf_arr, dtype=np.float32),
            n_tiles,
            solver=_tile_solver,
        )
        if return_diagnostics:
            return {
                "result": tiled_out["result"],
                "raw": np.asarray(image),
                "ci_rl": None,
                "dl_refined": tiled_out["result"],
                "residual": None,
                "reconvolved_prediction": None,
                "iterations_used": int(tiled_out.get("iterations_used", 0)),
                "convergence": tiled_out.get("convergence", []),
                "diagnostics": {
                    "rl_iterations": int(tiled_out.get("iterations_used", 0)),
                    "rl_convergence": tiled_out.get("convergence", []),
                    "dl_model_path": str(model_path),
                    "dl_tiling": int(n_tiles),
                    "dl_refinement": "tiled",
                },
            }
        return np.asarray(tiled_out["result"], dtype=np.float32)

    rl_out = ci_rl_deconvolve(image_arr, psf_arr, **rl_options)
    ci_rl = np.asarray(rl_out["result"], dtype=np.float32)
    diagnostics: dict[str, Any] = {
        "rl_iterations": int(rl_out.get("iterations_used", 0)),
        "rl_convergence": rl_out.get("convergence", []),
        "dl_model_path": str(model_path) if model_path else None,
    }

    if model_path is None:
        log.info("ci_rl_dl called without model_path; returning ci_rl result unchanged")
        result = np.clip(ci_rl, 0, None).astype(np.float32)
        diagnostics["dl_refinement"] = "skipped_no_model_path"
        if return_diagnostics:
            return {
                "result": result,
                "raw": np.asarray(image),
                "ci_rl": ci_rl,
                "dl_refined": result,
                "residual": np.zeros_like(result, dtype=np.float32),
                "diagnostics": diagnostics,
            }
        return result

    dl_options = dict(dl_kwargs or {})
    model, checkpoint = load_residual_unet_checkpoint(model_path, device=device)
    metadata = checkpoint.get("sidecar_metadata") or checkpoint
    inference_defaults = dict(metadata.get("recommended_inference") or {})
    dl_options = {**dict(inference_defaults.get("dl_kwargs") or {}), **dl_options}
    compute_reconvolution_diagnostics = bool(dl_options.pop("compute_reconvolution_diagnostics", False))
    z_radius = int(dl_options.pop("z_radius", checkpoint.get("z_radius", 2)))
    use_residual_channel = bool(
        dl_options.pop("use_residual_channel", checkpoint.get("use_residual_channel", True))
    )
    conditioning_channels = list(metadata.get("conditioning_channels") or [])
    conditioning_values = None
    if conditioning_channels:
        conditioning_values = conditioning_vector(
            psf=np.asarray(psf, dtype=np.float32),
            metadata=optical_params or {},
            rl_iterations=int(rl_options.get("niter", diagnostics.get("rl_iterations", 0)) or 0),
            microscope_type=(optical_params or {}).get("microscope_type") if optical_params else None,
        )
    refined_out = apply_dl_refinement_25d(
        np.asarray(image, dtype=np.float32),
        ci_rl,
        model,
        psf=np.asarray(psf, dtype=np.float32),
        device=device,
        z_radius=z_radius,
        use_residual_channel=use_residual_channel,
        conditioning_values=conditioning_values,
        compute_reconvolution_diagnostics=compute_reconvolution_diagnostics,
        **dl_options,
    )
    diagnostics.update(refined_out["diagnostics"])
    result = np.clip(refined_out["result"], 0, None).astype(np.float32)
    if return_diagnostics:
        reconvolved = None
        if compute_reconvolution_diagnostics:
            try:
                reconvolved = reconvolve_same(result, np.asarray(psf, dtype=np.float32), device=device)
            except Exception:
                pass
        return {
            "result": result,
            "raw": np.asarray(image),
            "ci_rl": ci_rl,
            "dl_refined": result,
            "residual": refined_out["residual"],
            "reconvolved_prediction": reconvolved,
            "diagnostics": diagnostics,
        }
    return result


ci_rl_dl = deconvolve_ci_rl_dl


__all__ = [
    "CONDITIONING_CHANNELS",
    "GatedResidualUNet25D",
    "ResidualUNet25D",
    "apply_dl_refinement_25d",
    "ci_rl_dl",
    "conditioning_vector",
    "deconvolve_ci_rl_dl",
    "input_channel_count",
    "load_residual_unet_checkpoint",
    "make_25d_input",
    "reconvolve_same",
    "resolve_torch_device",
]
