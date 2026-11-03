"""
posterior.py
------------

Concrete ParameterPosterior implementations.

This module provides:
- MAPPosterior: delta distribution at θ_MAP (point estimate)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


class MAPPosterior:
    """
    MAP (Maximum A Posteriori) posterior - delta distribution at θ_MAP.

    Represents a point estimate with no uncertainty.

    Parameters
    ----------
    params : dict
        MAP parameter dictionary (θ_MAP)
    model : WPPM
        Model instance used for predictions

    Notes
    -----
    This implements the ParameterPosterior protocol.
    """

    def __init__(self, params, model):
        self._params = params
        self._model = model

    # ------------------------------------------------------------------
    # ParameterPosterior protocol implementation
    # ------------------------------------------------------------------
    @property
    def params(self):
        """Return the MAP parameters (θ_MAP)."""
        return self._params

    @property
    def model(self):
        """Return the associated model."""
        return self._model

    def sample(self, n: int, *, key):
        """
        Sample from delta distribution (returns repeated θ_MAP).

        Parameters
        ----------
        n : int
            Number of samples
        key : jax.Array
            PRNG key. A delta distribution has no randomness, so the key goes
            unread.

        Returns
        -------
        dict
            Parameter PyTree with leading dimension n.
            Each array has shape (n, ...) with identical values.

        Notes
        -----
        Every row is the same parameter set, so a mean over the draws is θ_MAP and
        their spread is zero.
        """
        return jax.tree.map(
            lambda x: jnp.tile(x[None, ...], (n,) + (1,) * x.ndim), self._params
        )
