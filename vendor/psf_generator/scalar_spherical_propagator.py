"""Scalar spherical propagator."""
import math

import torch
from torch import vmap
from torch.special import bessel_j0

from .spherical_propagator import SphericalPropagator


class ScalarSphericalPropagator(SphericalPropagator):
    r"""
    Propagator for the scalar approximation of the Richard's-Wolf integral
    in spherical parameterization.
    """

    @classmethod
    def get_name(cls) -> str:
        return 'scalar_spherical'

    def initialize_input_field(self) -> torch.Tensor:
        input_field = torch.ones(self.n_pix_pupil).to(torch.complex64).to(self.device)
        return input_field

    def compute_focus_field(self) -> torch.Tensor:
        sin_t = torch.sin(self.thetas)
        bessel_arg = self.k * self.rs[None, :] * sin_t[:, None] * self.refractive_index
        J0 = bessel_j0(bessel_arg)

        batched = vmap(self._compute_psf_at_defocus, in_dims=(0, None, None, None))
        return batched(self.defocus_filters, J0, self.get_pupil(), sin_t)

    def _compute_psf_at_defocus(self, defocus_term, J0, pupil, sin_t):
        integrand = J0 * (pupil * defocus_term * sin_t)[:, None]
        field = self.integrator(fs=integrand, dx=self.dtheta)
        field = field[self.rr_indices].unsqueeze(0)
        return field / math.sqrt(self.refractive_index)
