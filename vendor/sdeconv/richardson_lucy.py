"""Richardson-Lucy deconvolution for 2D and 3D images."""
import torch
import numpy as np
from ._interface import SDeconvFilter
from ._utils import pad_2d, pad_3d, np_to_torch


class SRichardsonLucy(SDeconvFilter):
    """Richardson-Lucy deconvolution.

    Parameters
    ----------
    psf : torch.Tensor
        Point spread function.
    niter : int
        Number of iterations.
    pad : int or tuple
        Padding size.
    """
    def __init__(self,
                 psf: torch.Tensor,
                 niter: int = 30,
                 pad: int | tuple[int, int] | tuple[int, int, int] = 0):
        super().__init__()
        self.psf = psf
        self.niter = niter
        self.pad = pad

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim == 2:
            return self._deconv_2d(image)
        if image.ndim == 3:
            return self._deconv_3d(image)
        raise ValueError('Richardson-Lucy can only deblur 2D or 3D tensors')

    def _deconv_2d(self, image: torch.Tensor) -> torch.Tensor:
        image_pad, psf_pad, padding = pad_2d(image, self.psf / torch.sum(self.psf), self.pad)

        psf_roll = torch.roll(psf_pad, [int(-psf_pad.shape[0] / 2),
                                        int(-psf_pad.shape[1] / 2)], dims=(0, 1))
        fft_psf = torch.fft.fft2(psf_roll)
        fft_psf_mirror = torch.fft.fft2(torch.flip(psf_roll, dims=[0, 1]))

        out_image = image_pad.detach().clone()
        for _ in range(self.niter):
            fft_out = torch.fft.fft2(out_image)
            fft_tmp = fft_out * fft_psf
            tmp = torch.real(torch.fft.ifft2(fft_tmp))
            tmp = image_pad / tmp
            fft_tmp = torch.fft.fft2(tmp)
            fft_tmp = fft_tmp * fft_psf_mirror
            tmp = torch.real(torch.fft.ifft2(fft_tmp))
            out_image = out_image * tmp

        if image_pad.shape != image.shape:
            return out_image[padding[0]:-padding[0], padding[1]:-padding[1]]
        return out_image

    def _deconv_3d(self, image: torch.Tensor) -> torch.Tensor:
        image_pad, psf_pad, padding = pad_3d(image, self.psf / torch.sum(self.psf), self.pad)

        psf_roll = torch.roll(psf_pad, int(-psf_pad.shape[0] / 2), dims=0)
        psf_roll = torch.roll(psf_roll, int(-psf_pad.shape[1] / 2), dims=1)
        psf_roll = torch.roll(psf_roll, int(-psf_pad.shape[2] / 2), dims=2)

        fft_psf = torch.fft.fftn(psf_roll)
        fft_psf_mirror = torch.fft.fftn(torch.flip(psf_roll, dims=[0, 1]))

        out_image = image_pad.detach().clone()
        for _ in range(self.niter):
            fft_out = torch.fft.fftn(out_image)
            fft_tmp = fft_out * fft_psf
            tmp = torch.real(torch.fft.ifftn(fft_tmp))
            tmp = image_pad / tmp
            fft_tmp = torch.fft.fftn(tmp)
            fft_tmp = fft_tmp * fft_psf_mirror
            tmp = torch.real(torch.fft.ifftn(fft_tmp))
            out_image = out_image * tmp

        if image_pad.shape != image.shape:
            return out_image[padding[0]:-padding[0],
                             padding[1]:-padding[1],
                             padding[2]:-padding[2]]
        return out_image
