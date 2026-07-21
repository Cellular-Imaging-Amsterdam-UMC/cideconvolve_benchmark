"""Optional compiled CUDA backend for CI deconvolution.

The public status helpers are deliberately lightweight: importing this package
does not compile the extension. Compilation/loading happens only when an
optimized CUDA run is requested.
"""

from .loader import backend_status, load_optimized_extension, reset_backend_state

__all__ = ["backend_status", "load_optimized_extension", "reset_backend_state"]
