"""
utils
=====

Shared utility functions and helpers for psyphy.

This subpackage provides:
- candidates : functions for generating candidate stimulus pools.
- math : mathematical utilities (basis functions, distances, kernels).
- rng : random number handling for reproducibility.

MVP implementation
------------------
- candidates: grid, Sobol, custom pools.
- math: Chebyshev basis, Mahalanobis distance, RBF kernel.
- rng: seed() and split() for JAX PRNG keys.


Full WPPM mode
--------------
- candidates: adaptive refinement around posterior uncertainty.
- math: richer kernels and basis expansions for Wishart processes.
- rng: experiment-wide RNG registry.

"""

from .candidates import custom_candidates, grid_candidates, sobol_candidates
from .math import chebyshev_basis, mahalanobis_distance, rbf_kernel
from .rng import seed, split

__all__ = [
    # candidates
    "grid_candidates",
    "sobol_candidates",
    "custom_candidates",
    # math
    "chebyshev_basis",
    "mahalanobis_distance",
    "rbf_kernel",
    # rng
    "seed",
    "split",
]
