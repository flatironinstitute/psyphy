"""
inference
=========

Inference engines for WPPM.

This subpackage provides different strategies for fitting model parameters
to data and returning posterior objects.

MVP implementations
-------------------
- MAPOptimizer : maximum a posteriori fit with Optax optimizers.
- LaplaceApproximation : approximate posterior covariance around MAP.
- LangevinSampler : skeleton for sampling-based inference.

Future extensions
-----------------
- adjusted MC samplers, e.g., MALA (for Bayesian posterior inference).
"""
from .base import InferenceEngine
from .langevin import LangevinSampler
from .laplace import LaplaceApproximation
from .map_optimizer import MAPOptimizer

__all__ = [
    "InferenceEngine",
    "MAPOptimizer",
    "LangevinSampler",
    "LaplaceApproximation",
]




