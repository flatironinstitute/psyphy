"""
psyphy.model
============

Model-layer API: everything model-related in one place.

Includes
--------
- WPPM (core model)
- Priors (Prior)
- Tasks (TaskLikelihood base, OddityTask, TwoAFC)
- Noise models (GaussianNoise, StudentTNoise)

All functions/classes use JAX arrays (jax.numpy as jnp) for autodiff
and optimization with Optax.

Typical usage
-------------
    from psyphy.model import WPPM, Prior, OddityTask, GaussianNoise
"""

from .base import Model, OnlineConfig, auto_online_config
from .noise import GaussianNoise, StudentTNoise
from .prior import Prior
from .task import OddityTask, TaskLikelihood, TwoAFC
from .wppm import WPPM

__all__ = [
    # Base
    "Model",
    "OnlineConfig",
    "auto_online_config",
    # Models
    "WPPM",
    # Priors
    "Prior",
    # Tasks
    "TaskLikelihood",
    "OddityTask",
    "TwoAFC",
    # Noise models
    "GaussianNoise",
    "StudentTNoise",
]
