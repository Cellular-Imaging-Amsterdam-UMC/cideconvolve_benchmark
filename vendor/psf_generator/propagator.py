"""Abstract propagator base class."""
import json
import os
from abc import ABC, abstractmethod

import torch

from .misc import convert_tensor_to_array


class Propagator(ABC):
    r"""
    Base class propagator.

    Parameters
    ----------
    n_pix_pupil : int
        Number of pixels of the pupil. Default ``128``.
    n_pix_psf : int
        Number of pixels of the PSF. Default ``128``.
    device : str
        ``'cpu'`` or ``'gpu'``. Default ``'cpu'``.
    zernike_coefficients : array-like or None
        Zernike coefficients. Default ``None``.
    wavelength : float
        Wavelength in nm. Default ``632``.
    na : float
        Numerical aperture. Default ``1.3``.
    pix_size : float
        Camera pixel size in nm. Default ``20``.
    defocus_step : float
        Step size of defocus in nm. Default ``0.0``.
    n_defocus : int
        Number of z-stack. Default ``1``.
    apod_factor : bool
        Apply apodization factor. Default ``False``.
    envelope : float or None
        Gaussian envelope size. Default ``None``.
    gibson_lanni : bool
        Apply Gibson-Lanni correction. Default ``False``.
    """

    def __init__(self,
                 n_pix_pupil: int = 128,
                 n_pix_psf: int = 128,
                 device: str = 'cpu',
                 zernike_coefficients=None,
                 wavelength: float = 632,
                 na: float = 1.3,
                 pix_size: float = 20,
                 defocus_step: float = 0.0,
                 n_defocus: int = 1,
                 apod_factor: bool = False,
                 envelope=None,
                 gibson_lanni: bool = False,
                 z_p: float = 1e3,
                 n_s: float = 1.3,
                 n_g: float = 1.5,
                 n_g0: float = 1.5,
                 t_g: float = 170e3,
                 t_g0: float = 170e3,
                 n_i: float = 1.5,
                 n_i0: float = 1.5,
                 t_i0: float = 100e3):
        self.n_pix_pupil = n_pix_pupil
        self.n_pix_psf = n_pix_psf
        self.device = device
        if zernike_coefficients is None:
            zernike_coefficients = [0]
        if not isinstance(zernike_coefficients, torch.Tensor):
            zernike_coefficients = torch.tensor(zernike_coefficients)
        self.zernike_coefficients = zernike_coefficients
        self.wavelength = wavelength
        self.na = na
        self.pix_size = pix_size
        self.fov = pix_size * n_pix_psf
        self.defocus_step = defocus_step
        self.n_defocus = n_defocus
        self.defocus_min = -defocus_step * n_defocus // 2
        self.defocus_max = defocus_step * n_defocus // 2
        self.apod_factor = apod_factor
        self.envelope = envelope
        self.gibson_lanni = gibson_lanni
        self.z_p = z_p
        self.n_s = n_s
        self.n_g = n_g
        self.n_g0 = n_g0
        self.t_g = t_g
        self.t_g0 = t_g0
        self.n_i = n_i
        self.n_i0 = n_i0
        self.t_i0 = t_i0
        self.t_i = n_i * (t_g0 / n_g0 + t_i0 / self.n_i0 - t_g / n_g - z_p / n_s)
        if gibson_lanni:
            self.refractive_index = n_s
        else:
            self.refractive_index = n_i

    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        raise NotImplementedError

    @abstractmethod
    def initialize_input_field(self) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_pupil(self) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def compute_focus_field(self) -> torch.Tensor:
        raise NotImplementedError

    def update_zernike_coefficients(self, zernike_coefficients):
        if not isinstance(zernike_coefficients, torch.Tensor):
            zernike_coefficients = torch.tensor(zernike_coefficients)
        self.zernike_coefficients = zernike_coefficients
        if hasattr(self, '_compute_zernike_aberrations'):
            self._compute_zernike_aberrations()

    def compute_optical_path(self, sin_t: torch.Tensor) -> torch.Tensor:
        path = (self.z_p * torch.sqrt(self.n_s ** 2 - self.n_i ** 2 * sin_t ** 2)
                + self.t_i * torch.sqrt(self.n_i ** 2 - self.n_i ** 2 * sin_t ** 2)
                - self.t_i0 * torch.sqrt(self.n_i0 ** 2 - self.n_i ** 2 * sin_t ** 2)
                + self.t_g * torch.sqrt(self.n_g ** 2 - self.n_i ** 2 * sin_t ** 2)
                - self.t_g0 * torch.sqrt(self.n_g0 ** 2 - self.n_i ** 2 * sin_t ** 2))
        return path

    def _get_args(self) -> dict:
        args = {
            'n_pix_pupil': self.n_pix_pupil,
            'n_pix_psf': self.n_pix_psf,
            'device': self.device,
            'zernike_coefficients': convert_tensor_to_array(self.zernike_coefficients).tolist(),
            'wavelength': self.wavelength,
            'na': self.na,
            'pix_size': self.pix_size,
            'refractive_index': self.refractive_index,
            'defocus_step': self.defocus_step,
            'n_defocus': self.n_defocus,
            'apod_factor': self.apod_factor,
            'envelope': self.envelope,
            'gibson_lanni': self.gibson_lanni,
            'z_p': self.z_p,
            'n_s': self.n_s,
            'n_g': self.n_g,
            'n_g0': self.n_g0,
            't_g': self.t_g,
            't_g0': self.t_g0,
            'n_i': self.n_i,
            't_i0': self.t_i0,
            't_i': self.t_i,
        }
        return args

    def save_parameters(self, json_filepath: str):
        args = self._get_args()
        os.makedirs(os.path.dirname(json_filepath), exist_ok=True)
        with open(json_filepath, 'w') as file:
            json.dump(args, file, indent=2)
