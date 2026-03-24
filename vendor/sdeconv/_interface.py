"""Interface for a deconvolution filter."""
import torch
from ._core import SObservable


class SDeconvFilter(SObservable):
    """Interface for a deconvolution filter.

    All algorithm settings must be set in __init__ (PSF included) and
    __call__ is used to do the calculation.
    """
    def __init__(self):
        super().__init__()
        self.type = 'SDeconvFilter'

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('SDeconvFilter is an interface.')
