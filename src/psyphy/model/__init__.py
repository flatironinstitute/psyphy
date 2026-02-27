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

from .base import Model, OnlineConfig
from .covariance_field import CovarianceField, WPPMCovarianceField
from .noise import GaussianNoise, StudentTNoise
from .prior import Prior
from .task import OddityTask, OddityTaskConfig, TaskLikelihood
from .wppm import WPPM

__all__ = [
    # Base
    "Model",
    "OnlineConfig",
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
    # Noise models
    "GaussianNoise",
    "StudentTNoise",
]
