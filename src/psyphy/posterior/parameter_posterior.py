"""
parameter_posterior.py
---------------------

Protocol and implementations for posterior distributions over model parameters.

This module defines the ParameterPosterior interface representing p(θ | data).

Design
------
Different inference engines produce different posterior representations:
- MAP: delta distribution at θ_MAP
- MCMC: collection of samples

All implement a common protocol for polymorphic use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import jax.random as jr


@runtime_checkable
class ParameterPosterior(Protocol):
    """
    Protocol for posterior distributions over model parameters p(θ | data).

    Returned by InferenceEngine.fit(model, data).
    Used for diagnostics, parameter sampling, uncertainty.
    """

    @property
    def params(self) -> dict:
        """
        Point estimate or posterior mean parameters.

        Returns
        -------
        dict
            Parameter PyTree (e.g., {"log_diag": jnp.ndarray, ...})

        Notes
        -----
        - MAP: θ_MAP
        - MCMC: posterior mean of samples
        """
        ...

    @property
    def model(self):
        """
        Associated generative model.

        Returns
        -------
        Model
            The WPPM or (other model) instance used for predictions.
        """
        ...

    def sample(self, n: int, *, key: jr.KeyArray) -> dict:
        """
        Sample parameter vectors from p(θ | data).

        Parameters
        ----------
        n : int
            Number of samples
        key : jax.random.KeyArray
            PRNG key for randomness

        Returns
        -------
        dict
            Parameter PyTree with leading dimension n.


        Notes
        -----
        - MAP: returns repeated θ_MAP
        - Numpyro/Blackjax: returns stored samples (may subsample if n differs)
        """
        ...
