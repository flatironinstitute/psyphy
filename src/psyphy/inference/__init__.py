"""
inference
=========

Inference engines for WPPM.

This subpackage provides different strategies for fitting model parameters
to data and returning posterior objects.

Implementations
---------------
- MAPOptimizer : maximum a posteriori fit with Optax optimizers.
- NUTSSampler : full posterior sampling via NUTS (requires blackjax).
- MALASampler : full posterior sampling via MALA, fixed step size (requires blackjax).
- LaplaceApproximation : approximate posterior covariance around MAP (stub).
- LangevinSampler : skeleton for Langevin-based sampling (stub).

Optional dependencies
---------------------
- NUTSSampler and MALASampler require blackjax: pip install 'psyphy[sampling]'
"""

from .base import InferenceEngine
from .langevin import LangevinSampler
from .laplace import LaplaceApproximation
from .mala import MALASampler
from .map_optimizer import MAPOptimizer

# NUTSSampler and MALASampler import blackjax inside fit, so both classes import
# without it. A missing blackjax raises at .fit() time, with a message naming the extra.
from .nuts import NUTSSampler

__all__ = [
    "InferenceEngine",
    "MAPOptimizer",
    "NUTSSampler",
    "MALASampler",
    "LangevinSampler",
    "LaplaceApproximation",
]
