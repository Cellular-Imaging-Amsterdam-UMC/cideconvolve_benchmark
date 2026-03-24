"""Numerical integration rules in 1D (PyTorch)."""
import warnings

import torch

__all__ = ['riemann_rule', 'simpsons_rule']


def is_power_of_two(k: int) -> bool:
    k = int(k)
    return (k & (k - 1) == 0) and k != 0


def riemann_rule(fs: torch.Tensor, dx: float) -> torch.Tensor:
    return torch.sum(fs, dim=0) * dx


def trapezoid_rule(fs: torch.Tensor, dx: float) -> torch.Tensor:
    return 0.5 * (fs[0] + 2.0 * torch.sum(fs[1:-1], dim=0) + fs[-1]) * dx


def simpsons_rule(fs: torch.Tensor, dx: float) -> torch.Tensor:
    if fs.shape[0] % 2 == 0:
        warnings.warn("Pupil size is not an odd number! The computed "
                       "integral will not have high-order accuracy.")
    return (fs[0] + 4 * torch.sum(fs[1:-1:2], dim=0) +
            2 * torch.sum(fs[2:-1:2], dim=0) + fs[-1]) * dx / 3.0
