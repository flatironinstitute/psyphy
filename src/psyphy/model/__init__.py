"""
psyphy.model
============

Model-layer API: everything model-related in one place.

Includes
--------
- WPPM (core model)
- Priors (Prior)
- Tasks (TaskLikelihood base, OddityTask)
- Noise models (GaussianNoise, StudentTNoise)

All functions/classes use JAX arrays (jax.numpy as jnp) for autodiff
and optimization with Optax.

Typical usage
-------------
    from psyphy.model import WPPM, Prior, OddityTask, GaussianNoise
"""

from .base import Model
from .covariance_field import CovarianceField, WPPMCovarianceField
from .likelihood import (
    NeuralSurrogateOddityTask,
    NeuralSurrogateTask,
    OddityTask,
    OddityTaskConfig,
    TaskLikelihood,
)
from .noise import GaussianNoise, StudentTNoise
from .prior import Prior
from .wppm import WPPM

__all__ = [
    # Base
    "Model",
    # Covariance fields
    "CovarianceField",
    "WPPMCovarianceField",
    # Models
    "WPPM",
    # Priors
    "Prior",
    # Tasks
    "TaskLikelihood",
    "OddityTask",
    "OddityTaskConfig",
    "NeuralSurrogateTask",
    "NeuralSurrogateOddityTask",
    # Noise models
    "GaussianNoise",
    "StudentTNoise",
]
