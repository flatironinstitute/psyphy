"""
rng.py
------

Random number utilities for psyphy.

This module standardizes RNG handling across the package,
especially important when mixing NumPy and JAX.

MVP implementation:
- Wrappers around JAX PRNG keys.
- Helpers for reproducibility.

Future extensions:
- Experiment-wide RNG registry.
- Splitting strategies for parallel adaptive placement.

Examples
--------
>>> import jax
>>> from psyphy.utils.rng import seed, split
>>> key = seed(0)
>>> k1, k2 = split(key)
"""

from __future__ import annotations

import jax
import jax.random as jr


def seed(seed_value: int) -> jax.random.KeyArray:
    """
    Create a new PRNG key from an integer seed.

    Parameters
    ----------
    seed_value : int
        Seed for random number generation.

    Returns
    -------
    jax.random.KeyArray
        New PRNG key.
    """
    return jr.PRNGKey(seed_value)


def split(key: jax.random.KeyArray, num: int = 2):
    """
    Split a PRNG key into multiple independent keys.

    Parameters
    ----------
    key : jax.random.KeyArray
        RNG key to split.
    num : int, default=2
        Number of new keys to return.

    Returns
    -------
    tuple of jax.random.KeyArray
        Independent new PRNG keys.
    """
    return jr.split(key, num=num)
