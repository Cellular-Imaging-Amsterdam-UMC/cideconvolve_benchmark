"""Vendored subset of psf_generator (MIT License, Biomedical Imaging Group, EPFL 2024-2025).

Only the spherical propagators needed by deconvolve.py are included.
"""
from .vectorial_spherical_propagator import VectorialSphericalPropagator
from .scalar_spherical_propagator import ScalarSphericalPropagator

__all__ = ["VectorialSphericalPropagator", "ScalarSphericalPropagator"]
