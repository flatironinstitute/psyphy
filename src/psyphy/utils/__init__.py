"""
utils
=====

Shared utility functions and helpers for psyphy.

This subpackage provides:
- bootstrap : frequentist confidence intervals via resampling.
- candidates : functions for generating candidate stimulus pools.
- math : mathematical utilities (basis functions, distances, kernels).
- rng : random number handling for reproducibility.

MVP implementation
------------------
- bootstrap: prediction CIs, model comparison, arbitrary statistics.
- candidates: grid, Sobol, custom pools.
- math: Chebyshev basis, Mahalanobis distance, RBF kernel.
- rng: seed() and split() for JAX PRNG keys.


Full WPPM mode
--------------
- candidates: adaptive refinement around posterior uncertainty.
- math: richer kernels and basis expansions for Wishart processes.
- rng: experiment-wide RNG registry.

"""

from .bootstrap import (
    bootstrap_compare_models,
    bootstrap_predictions,
    bootstrap_statistic,
)
from .candidates import custom_candidates, grid_candidates, sobol_candidates
from .math import chebyshev_basis, mahalanobis_distance, rbf_kernel
from .rng import seed, split

__all__ = [
    # bootstrap
    "bootstrap_predictions",
    "bootstrap_statistic",
    "bootstrap_compare_models",
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
