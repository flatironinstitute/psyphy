"""
predictive_posterior.py
----------------------

Predictive posterior distributions p(f(X*) | data) at test stimuli.

This module defines posteriors over **predictions** (not parameters),
used by acquisition functions for Bayesian optimization.

Design
------
PredictivePosterior wraps a ParameterPosterior and computes predictions via:
    E[f(X*) | data] ≈ (1/N) Σ_i f(X*; θ_i) where θ_i ~ p(θ | data)

This separates concerns:
- ParameterPosterior: represents uncertainty over θ (research)
- PredictivePosterior: represents uncertainty over f(X*) (decision-making)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import jax.random as jr

if TYPE_CHECKING:
    from psyphy.posterior.parameter_posterior import ParameterPosterior


@runtime_checkable
class PredictivePosterior(Protocol):
    """
    Protocol for predictive distributions p(f(X*) | data) at test stimuli.

    Returned by Model.posterior(X) for use in acquisition functions.
    """

    @property
    def mean(self) -> jnp.ndarray:
        """
        Posterior predictive mean E[f(X*) | data].

        Returns
        -------
        jnp.ndarray
            Shape (n_test,) for scalar outputs
            Shape (n_test, output_dim) for vector outputs (future)

        Notes
        -----
        Computed via Monte Carlo integration over parameter posterior.
        """
        ...

    @property
    def variance(self) -> jnp.ndarray:
        """
        Posterior predictive marginal variances Var[f(X*) | data].

        Returns
        -------
        jnp.ndarray
            Shape (n_test,) for scalar outputs
            Shape (n_test, output_dim) for vector outputs (future)

        Notes
        -----
        Captures both aleatoric (model) and epistemic (parameter) uncertainty.
        """
        ...

    def rsample(self, sample_shape: tuple = (), *, key: jr.KeyArray) -> jnp.ndarray:
        """
        Reparameterized samples from p(f(X*) | data).

        Parameters
        ----------
        sample_shape : tuple, default=()
            Shape of sample batch
        key : jax.random.KeyArray
            PRNG key

        Returns
        -------
        jnp.ndarray
            Shape (*sample_shape, n_test) for scalar outputs
            Shape (*sample_shape, n_test, output_dim) for vector outputs

        Notes
        -----
        Enables gradient-based acquisition optimization via reparameterization trick.
        """
        ...

    def cov_field(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Posterior over perceptual covariance field Σ(X).

        Parameters
        ----------
        X : jnp.ndarray
            Test stimuli, shape (n_test, input_dim)

        Returns
        -------
        jnp.ndarray
            Posterior mean covariance E[Σ(X) | data],
            shape (n_test, input_dim, input_dim)

        Notes
        -----
        WPPM-specific method for visualizing perceptual noise structure.
        This is NOT the predictive covariance - it's the model's
        internal representation of perceptual uncertainty.
        """
        ...


class WPPMPredictivePosterior:
    """
    Predictive posterior for WPPM models.

    Computes p(f(X*) | data) via Monte Carlo integration over
    parameter posterior p(θ | data).

    Parameters
    ----------
    param_posterior : ParameterPosterior
        Posterior over model parameters
    X : jnp.ndarray, shape (n_test, input_dim)
        Test reference stimuli
    probes : jnp.ndarray, shape (n_test, input_dim), optional
        Test probe stimuli. If None, predictions are over thresholds.
    n_samples : int, default=100
        Number of posterior samples for MC integration

    Attributes
    ----------
    param_posterior : ParameterPosterior
        Wrapped parameter posterior
    X : jnp.ndarray
        Test stimuli
    probes : jnp.ndarray | None
        Test probes
    n_samples : int
        MC sample count

    Notes
    -----
    Uses lazy evaluation: moments computed on first access.
    """

    def __init__(
        self,
        param_posterior: ParameterPosterior,
        X: jnp.ndarray,
        probes: jnp.ndarray | None = None,
        n_samples: int = 100,
    ):
        self.param_posterior = param_posterior
        self.X = X
        self.probes = probes
        self.n_samples = n_samples

        # Lazy evaluation cache
        self._mean = None
        self._variance = None
        self._computed = False

    def _ensure_computed(self):
        """Compute moments via MC integration (lazy)."""
        if self._computed:
            return

        # Sample parameters from posterior
        key = jr.PRNGKey(0)  # TODO: Make configurable via init
        param_samples = self.param_posterior.sample(self.n_samples, key=key)

        model = self.param_posterior.model

        if self.probes is None:
            # TODO: Implement threshold prediction
            raise NotImplementedError(
                "Threshold prediction not yet implemented. "
                "Pass probes argument for pairwise discrimination."
            )

        # Vectorized prediction over parameter samples
        def predict_batch(params):
            """Predict p(correct) for all (ref, probe) pairs given params."""
            return jax.vmap(lambda r, p: model.predict_prob(params, (r, p)))(
                self.X, self.probes
            )

        # predictions: shape (n_samples, n_test)
        predictions = jax.vmap(predict_batch)(param_samples)

        # Compute moments
        self._mean = jnp.mean(predictions, axis=0)
        self._variance = jnp.var(predictions, axis=0)
        self._computed = True

    @property
    def mean(self) -> jnp.ndarray:
        """E[f(X*) | data], shape (n_test,)."""
        self._ensure_computed()
        return self._mean

    @property
    def variance(self) -> jnp.ndarray:
        """Var[f(X*) | data], shape (n_test,)."""
        self._ensure_computed()
        return self._variance

    def rsample(self, sample_shape: tuple = (), *, key: jr.KeyArray) -> jnp.ndarray:
        """
        Sample predictions from p(f(X*) | data).

        Parameters
        ----------
        sample_shape : tuple
            Batch shape
        key : jax.random.KeyArray
            PRNG key

        Returns
        -------
        jnp.ndarray
            Shape (*sample_shape, n_test)
        """
        n = int(jnp.prod(jnp.array(sample_shape))) if sample_shape else 1
        param_samples = self.param_posterior.sample(n, key=key)

        model = self.param_posterior.model

        if self.probes is None:
            raise NotImplementedError("Threshold sampling not yet implemented")

        def predict_one(params):
            """Predict for all test points with given params."""
            return jax.vmap(lambda r, p: model.predict_prob(params, (r, p)))(
                self.X, self.probes
            )

        samples = jax.vmap(predict_one)(param_samples)

        if sample_shape:
            return samples.reshape(*sample_shape, -1)
        return samples

    def cov_field(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Posterior mean covariance field E[Σ(X) | data].

        Parameters
        ----------
        X : jnp.ndarray
            Test stimuli, shape (n_test, input_dim)

        Returns
        -------
        jnp.ndarray
            Covariance matrices, shape (n_test, input_dim, input_dim)

        Notes
        -----
        Averages local_covariance(x) over parameter posterior samples.
        """
        key = jr.PRNGKey(0)
        param_samples = self.param_posterior.sample(self.n_samples, key=key)

        model = self.param_posterior.model

        def cov_at_x(params, x):
            """Evaluate Σ(x) with given parameters."""
            return model.local_covariance(params, x)

        # Vectorized evaluation: (n_samples, n_test, input_dim, input_dim)
        cov_samples = jax.vmap(
            lambda params: jax.vmap(lambda x: cov_at_x(params, x))(X)
        )(param_samples)

        # Return posterior mean
        return jnp.mean(cov_samples, axis=0)
