"""
psyphy.model
==========

Core model components:
- TaskLikelihood interfaces and example tasks
- prior distributions over model parameters enabling warm starts/transfer learning
- WPPM (Wishart Process Psychophysical Model) 

All code in this submodule uses JAX arrays (jax.numpy as jnp), such that it works
nicely with autodiff (jax.grad), and Optax optimizers.

typical usage:

    from psyphy.model import WPPM, Prior, OddityTask
"""

from .prior import Prior
from .task import OddityTask, TwoAFC
from .wppm import WPPM

__all__ = [ "OddityTask", "TwoAFC", "Prior", "WPPM"]


