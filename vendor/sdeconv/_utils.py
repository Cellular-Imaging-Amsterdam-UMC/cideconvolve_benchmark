"""Shared utilities for the deconvolution algorithms.

Bug fix: pad_3d now uses resize_psf_3d to match the PSF to the padded image
size (consistent with pad_2d which uses resize_psf_2d). The original sdeconv
used torch.nn.functional.pad which kept the PSF at its original size, causing
FFT size mismatches in 3D.
"""
import numpy as np
import torch
from ._core import SSettings


def np_to_torch(image: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(image, np.ndarray):
        return torch.tensor(image).to(SSettings.instance().device)
    return image


def resize_psf_2d(image: torch.Tensor, psf: torch.Tensor) -> torch.Tensor:
    kernel = torch.zeros(image.shape).to(SSettings.instance().device)
    x_start = int(image.shape[0] / 2 - psf.shape[0] / 2) + 1
    y_start = int(image.shape[1] / 2 - psf.shape[1] / 2) + 1
    kernel[x_start:x_start + psf.shape[0], y_start:y_start + psf.shape[1]] = psf
    return kernel


def resize_psf_3d(image: torch.Tensor, psf: torch.Tensor) -> torch.Tensor:
    kernel = torch.zeros(image.shape).to(SSettings.instance().device)
    x_start = int(image.shape[0] / 2 - psf.shape[0] / 2) + 1
    y_start = int(image.shape[1] / 2 - psf.shape[1] / 2) + 1
    z_start = int(image.shape[2] / 2 - psf.shape[2] / 2) + 1
    kernel[x_start:x_start + psf.shape[0], y_start:y_start + psf.shape[1],
           z_start:z_start + psf.shape[2]] = psf
    return kernel


def pad_2d(image: torch.Tensor,
           psf: torch.Tensor,
           pad: int | tuple[int, int]
           ) -> tuple[torch.Tensor, torch.Tensor, int | tuple[int, int]]:
    padding = pad
    if isinstance(pad, tuple) and len(pad) != image.ndim:
        raise ValueError("Padding must be the same dimension as image")
    if isinstance(pad, int):
        if pad == 0:
            return image, psf, (0, 0)
        padding = (pad, pad)

    if padding[0] > 0 and padding[1] > 0:
        pad_fn = torch.nn.ReflectionPad2d((padding[0], padding[0], padding[1], padding[1]))
        image_pad = pad_fn(image.detach().clone().to(
            SSettings.instance().device).view(1, 1, image.shape[0], image.shape[1])).view(
            (image.shape[0] + 2 * padding[0], image.shape[1] + 2 * padding[0]))
    else:
        image_pad = image.detach().clone().to(SSettings.instance().device)
    psf_pad = resize_psf_2d(image_pad, psf)
    return image_pad, psf_pad, padding


def pad_3d(image: torch.Tensor,
           psf: torch.Tensor,
           pad: int | tuple[int, int, int]
           ) -> tuple[torch.Tensor, torch.Tensor, int | tuple[int, int, int]]:
    padding = pad
    if isinstance(pad, tuple) and len(pad) != image.ndim:
        raise ValueError("Padding must be the same dimension as image")
    if isinstance(pad, int):
        if pad == 0:
            return image, psf, (0, 0, 0)
        padding = (pad, pad, pad)

    if padding[0] > 0 and padding[1] > 0 and padding[2] > 0:
        p3d = (padding[2], padding[2], padding[1], padding[1], padding[0], padding[0])
        pad_fn = torch.nn.ReflectionPad3d(p3d)
        image_pad = pad_fn(
            image.detach().clone().to(SSettings.instance().device).view(1, 1, image.shape[0],
                                                                        image.shape[1],
                                                                        image.shape[2])).view(
            (image.shape[0] + 2 * padding[0], image.shape[1] + 2 * padding[1],
             image.shape[2] + 2 * padding[2]))
        # BUG FIX: use resize_psf_3d (like pad_2d uses resize_psf_2d)
        psf_pad = resize_psf_3d(image_pad, psf)
    else:
        image_pad = image
        psf_pad = psf
    return image_pad, psf_pad, padding


def unpad_3d(image: torch.Tensor, padding: tuple[int, int, int]) -> torch.Tensor:
    return image[padding[0]:-padding[0],
                 padding[1]:-padding[1],
                 padding[2]:-padding[2]]
