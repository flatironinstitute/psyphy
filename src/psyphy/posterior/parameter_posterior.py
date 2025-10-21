"""
parameter_posterior.py
---------------------

Protocol and implementations for posterior distributions over model parameters.

This module defines the ParameterPosterior interface representing p(θ | data),
used for research workflows: diagnostics, parameter uncertainty, sampling.

Design
------
Different inference engines produce different posterior representations:
- MAP: delta distribution at θ_MAP
- Laplace: Gaussian N(θ_MAP, Σ)
- MCMC: collection of samples

All implement a common protocol for polymorphic use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import jax.numpy as jnp

if TYPE_CHECKING:
    import jax.random as jr


@runtime_checkable
class ParameterPosterior(Protocol):
    """
    Protocol for posterior distributions over model parameters p(θ | data).

    Returned by InferenceEngine.fit(model, data).
    Used for research workflows: diagnostics, parameter sampling, uncertainty.
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
        - Laplace: E[θ] (Gaussian mean, equals θ_MAP)
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
            The WPPM or other model instance used for predictions.
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
            Example: {"log_diag": jnp.ndarray with shape (n, input_dim)}

        Notes
        -----
        - MAP: returns repeated θ_MAP
        - Laplace: samples from N(θ_MAP, Σ)
        - MCMC: returns stored samples (may subsample if n differs)
        """
        ...

    def log_prob(self, params: dict) -> jnp.ndarray:
        """
        Evaluate log p(θ | data) at given parameters.

        Parameters
        ----------
        params : dict
            Parameter PyTree to evaluate

        Returns
        -------
        jnp.ndarray
            Log probability (scalar)

        Raises
        ------
        NotImplementedError
            For MCMC posteriors (no tractable density)

        Notes
        -----
        - MAP: -∞ for θ ≠ θ_MAP, 0 at θ_MAP (delta distribution)
        - Laplace: Gaussian log density
        - MCMC: raises NotImplementedError
        """
        ...

    def diagnostics(self) -> dict:
        """
        Return inference-specific diagnostic information.

        Returns
        -------
        dict
            Diagnostic metrics. Contents vary by inference method:

            MAP:
                - convergence_info: optimizer status
                - final_loss: negative log posterior at MAP

            Laplace:
                - log_marginal_likelihood: evidence approximation
                - hessian_condition_number: numerical stability
                - marginal_variances: parameter uncertainties

            MCMC:
                - ess: effective sample size per parameter
                - rhat: Gelman-Rubin statistic
                - acceptance_rate: step acceptance (if applicable)
        """
        ...

    def predict_prob(self, stimulus) -> jnp.ndarray:
        """
        Predict probability of correct response for a stimulus.

        Parameters
        ----------
        stimulus : tuple
            (reference, probe) in model space

        Returns
        -------
        jnp.ndarray
            Probability of correct response

        Notes
        -----
        Uses point estimate (self.params) for prediction.
        For uncertainty quantification, use PredictivePosterior instead.
        """
        ...

    def predict_thresholds(
        self,
        reference,
        criterion: float = 0.667,
        directions: int = 16,
    ) -> jnp.ndarray:
        """
        Predict discrimination threshold contour around a reference.

        Parameters
        ----------
        reference : jnp.ndarray
            Reference point in model space
        criterion : float, default=0.667
            Target performance level
        directions : int, default=16
            Number of directions to probe

        Returns
        -------
        jnp.ndarray
            Threshold contour points

        Notes
        -----
        Uses point estimate. For credible regions, use PredictivePosterior
        with multiple samples.
        """
        ...


# Backwards compatibility alias
# TODO: Remove in v1.0.0
Posterior = ParameterPosterior
