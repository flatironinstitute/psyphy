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
- LaplaceApproximation : approximate posterior covariance around MAP (stub).
- LangevinSampler : skeleton for Langevin-based sampling (stub).

Optional dependencies
---------------------
- NUTSSampler requires blackjax: pip install 'psyphy[sampling]'

Future extensions
-----------------
- adjusted MC samplers, e.g., MALA (for Bayesian posterior inference).
"""

from .base import InferenceEngine
from .langevin import LangevinSampler
from .laplace import LaplaceApproximation
from .map_optimizer import MAPOptimizer

# NUTSSampler soft-imports blackjax at call time, so the class itself is always
# importable -- a missing blackjax only raises at .fit() time, with a message
# naming the extra.
from .nuts import NUTSSampler

__all__ = [
    "InferenceEngine",
    "MAPOptimizer",
    "NUTSSampler",
    "LangevinSampler",
    "LaplaceApproximation",
]
