"""
posterior.py
------------

Concrete ParameterPosterior implementations.

This module provides:
- MAPPosterior: delta distribution at θ_MAP (point estimate)
- Posterior: backwards compatibility alias (deprecated)

Future additions:
- LaplacePosterior: Gaussian approximation
- LangevinPosterior: MCMC samples
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from psyphy.posterior.base_posterior import BasePosterior


class MAPPosterior(BasePosterior):
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
    For uncertainty quantification, use LaplacePosterior or LangevinPosterior.
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

    def MAP_params(self):
        """
        Return the MAP parameters.

        Returns
        -------
        dict
            Parameter dictionary.

        Notes
        -----
        Kept for backwards compatibility with BasePosterior.
        Use .params property instead.
        """
        return self._params

    def sample(self, n: int = 1, *, key=None):
        """
        Sample from delta distribution (returns repeated θ_MAP).

        Parameters
        ----------
        n : int, default=1
            Number of samples
        key : jax.random.KeyArray, optional
            PRNG key (unused for delta distribution)

        Returns
        -------
        dict
            Parameter PyTree with leading dimension n.
            Each array has shape (n, ...) with identical values.

        Notes
        -----
        Delta distribution has no randomness - returns repeated MAP estimate.
        """
        # Stack params n times along new leading dimension
        return jax.tree.map(
            lambda x: jnp.tile(x[None, ...], (n,) + (1,) * x.ndim), self._params
        )

    def log_prob(self, params: dict) -> jnp.ndarray:
        """
        Evaluate log p(θ | data) under delta distribution.

        Parameters
        ----------
        params : dict
            Parameters to evaluate

        Returns
        -------
        jnp.ndarray
            0.0 if params == θ_MAP, -∞ otherwise

        Notes
        -----
        Delta distribution: all mass at θ_MAP.
        In practice, returns 0.0 at MAP and -inf elsewhere.
        """
        # Check if params match MAP (element-wise)
        matches = jax.tree.map(lambda x, y: jnp.allclose(x, y), params, self._params)
        all_match = jax.tree.reduce(lambda a, b: a & b, matches)

        return jnp.where(all_match, 0.0, -jnp.inf)

    def diagnostics(self) -> dict:
        """
        Return diagnostic information.

        Returns
        -------
        dict
            Empty dict (no diagnostics for delta distribution)

        Notes
        -----
        Override this in subclasses to add optimizer convergence info.
        """
        return {}

    # ------------------------------------------------------------------
    # PREDICTIONS: delegates to model
    # ------------------------------------------------------------------
    def predict_prob(self, stimulus):
        """
        Predict probability of correct response for a stimulus.

        Parameters
        ----------
        stimulus : tuple
            (reference, probe).

        Returns
        -------
        jnp.ndarray
            Probability of correct response.

        Notes
        -----
        Delegates to WPPM.predict_prob().
        This is not recursion: Posterior calls WPPM’s method with stored params.
        """
        return self.model.predict_prob(self.params, stimulus)

    def predict_thresholds(
        self, reference, criterion: float = 0.667, directions: int = 16
    ):
        """
        Predict discrimination threshold contour around a reference stimulus.

        Parameters
        ----------
        reference : jnp.ndarray
            Reference point in model space.
        criterion : float, default=0.667
            Target performance (e.g., 2/3 for oddity).
        directions : int, default=16
            Number of directions to probe.

        Returns
        -------
        jnp.ndarray
            Contour points (MVP: unit circle).

        MVP
        ---
        Returns a placeholder unit circle.

        Future
        ------
        - Search outward in each direction until performance crosses criterion.
        - Average over posterior samples (Laplace, MCMC) to get credible intervals.
        """
        angles = jnp.linspace(0, 2 * jnp.pi, directions, endpoint=False)
        contour = jnp.stack(
            [reference + jnp.array([jnp.cos(a), jnp.sin(a)]) for a in angles]
        )
        return contour


# ============================================================================
# Backwards compatibility
# ============================================================================

# TODO: Remove in v1.0.0
Posterior = MAPPosterior
