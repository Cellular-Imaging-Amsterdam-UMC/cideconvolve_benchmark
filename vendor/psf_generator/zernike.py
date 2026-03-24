"""Zernike polynomial utilities."""
import warnings

import numpy as np
import torch
from scipy.special import binom
from zernikepy import zernike_polynomials


def create_pupil_mesh(n_pixels: int) -> tuple[torch.Tensor, ...]:
    x = torch.linspace(-1, 1, n_pixels)
    y = torch.linspace(-1, 1, n_pixels)
    kx, ky = torch.meshgrid(x, y, indexing='xy')
    return kx, ky


def zernike_nl(n: int, l: int, rho: torch.float, phi: float, radius: float = 1) -> torch.Tensor:
    m = abs(l)
    R = 0
    for k in np.arange(0, (n - m) / 2 + 1):
        R = R + (-1) ** k * binom(n - k, k) * binom(n - 2 * k, (n - m) / 2 - k) * (rho / radius) ** (n - 2 * k)
    Z = torch.where(rho <= radius, R, 0)
    Z *= np.cos(m * phi) if l >= 0 else np.sin(m * phi)
    return Z


def index_to_nl(index: int) -> tuple[int, int]:
    n = 0
    while True:
        for l in range(n + 1):
            if n * (n + 1) / 2 + l == index:
                return n, - n + 2 * l
            elif n * (n + 1) / 2 + l > index:
                raise ValueError('Index out of bounds.')
        n += 1


def create_zernike_aberrations(zernike_coefficients: torch.Tensor, n_pix_pupil: int,
                               mesh_type: str) -> torch.Tensor:
    n_zernike = len(zernike_coefficients)
    if mesh_type == 'cartesian':
        zernike_basis = zernike_polynomials(mode=n_zernike - 1, size=n_pix_pupil, select='all')
        zernike_coefficients = zernike_coefficients.reshape(1, 1, n_zernike)
        zernike_phase = torch.sum(zernike_coefficients * torch.from_numpy(zernike_basis), dim=2)
    elif mesh_type == 'spherical':
        rho = torch.linspace(0, 1, n_pix_pupil)
        phi = 0
        zernike_phase = torch.zeros(n_pix_pupil)
        for i in range(n_zernike):
            n, l = index_to_nl(index=i)
            curr_coefficient = zernike_coefficients[i]
            if l != 0 and curr_coefficient != 0:
                warnings.warn("Warning: Zernike polynomials that are not axis-symmetric "
                              "are not supported in spherical coordinates!")
            elif l == 0:
                zernike_phase += curr_coefficient * zernike_nl(n=n, l=l, rho=rho, phi=phi)
    else:
        raise ValueError(f"Invalid mesh type {mesh_type}, choose 'spherical' or 'cartesian'.")

    return torch.exp(1j * zernike_phase).to(torch.complex64)
