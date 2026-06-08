"""
predictive_posterior.py
----------------------

Predictive posterior distributions p(f(X*) | data) at test stimuli.

This module defines posteriors over **predictions** (not parameters),
used by acquisition functions for Bayesian optimization.

Design
------
PredictivePosterior wraps a ParameterPosterior and computes predictions via:
    E[f(X*) | data] \approx (1/N) Σ_i f(X*; θ_i) where θ_i ~ p(θ | data)

This separates concerns:
- ParameterPosterior: represents uncertainty over θ
- PredictivePosterior: represents uncertainty over f(X*) (decision-making)
- effectively decoupling how we FIT the model from how we USE the fitted model
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import jax.random as jr

if TYPE_CHECKING:
    from psyphy.posterior.parameter_posterior import ParameterPosterior

from ..model.likelihood import BernoulliTaskLikelihood, GaussianTaskLikelihood


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
    X : jnp.ndarray, shape (n_test, k_stimuli, input_dim)
        Test stimuli
    n_samples : int, default=100
        Number of posterior samples for MC integration
    threshold_pred: bool, default = False
        Whether the given X should be used for threshold prediction (not yet implemented)

    Attributes
    ----------
    param_posterior : ParameterPosterior
        Wrapped parameter posterior
    X : jnp.ndarray
        Test stimuli
    n_samples : int
        MC sample count
    threshold_pred: bool
        Whether threshold prediction (not yet implemented)

    Notes
    -----
    Uses lazy evaluation: moments computed on first access.
    """

    def __init__(
        self,
        param_posterior: ParameterPosterior,
        X: jnp.ndarray,
        n_samples: int = 100,
        threshold_pred: bool = False,
    ):
        self.param_posterior = param_posterior
        self.X = X
        self.n_samples = n_samples
        self.threshold_pred = threshold_pred

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

        if self.threshold_pred:
            # TODO: Implement threshold prediction
            raise NotImplementedError(
                "Threshold prediction not yet implemented. "
                "Current version requires X to include all stimuli in relevant task."
            )

        # Vectorized prediction over parameter samples
        def predict_batch(params):
            """Predict probability parameters for given params
            For OddityTask, this is p(correct) for all (ref, probe) pairs given params."""

            return jax.vmap(lambda x: model.predict_prob(params, x))(self.X)

        if isinstance(model.likelihood, BernoulliTaskLikelihood):
            # p_correct: shape (n_samples, n_test)
            p_correct = jax.vmap(predict_batch)(param_samples)[0]

            # Compute moments
            self._mean = jnp.mean(p_correct, axis=0)
            self._variance = jnp.var(p_correct, axis=0)
            self._computed = True
        elif isinstance(model.likelihood, GaussianTaskLikelihood):
            # mu: shape (n_samples, n_test, r_dim)
            mu = jax.vmap(predict_batch)(param_samples)[0]

            # Compute moments
            self._mean = jnp.mean(mu, axis=0)
            self._variance = jax.vmap(lambda m: jnp.cov(m, rowvar=False), in_axes=1)(mu)
            self._computed = True
        else:
            raise NotImplementedError(
                "WPPMPredictivePosterior Currently only supports Bernoulli and"
                "Gaussian TaskLikelihoods."
                "Support for more distributions requires updating to handle"
                "more prob_parameter returns."
            )

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
            Shape (*sample_shape, n_test, r_dim) or (*sample_shape, n_test) if r_dim == 1
        """
        n = int(jnp.prod(jnp.array(sample_shape))) if sample_shape else 1
        param_samples = self.param_posterior.sample(n, key=key)

        model = self.param_posterior.model

        if self.threshold_pred:
            raise NotImplementedError("Threshold sampling not yet implemented")

        def predict_one(params):
            """Predict for all test points with given params."""
            return jax.vmap(lambda x: model.predict_prob(params, x))(self.X)

        if isinstance(model.likelihood, BernoulliTaskLikelihood) | isinstance(
            model.likelihood, GaussianTaskLikelihood
        ):
            samples = jax.vmap(predict_one)(param_samples)[0]
        else:
            raise NotImplementedError(
                "WPPMPredictivePosterior Currently only supports Bernoulli and"
                "Gaussian TaskLikelihoods."
                "Support for more distributions requires updating to handle"
                "more prob_parameter returns."
            )

        if sample_shape:
            r_dim = model.likelihood.resp_dim
            return jnp.squeeze(samples.reshape(*sample_shape, -1, r_dim))
        return samples

    def cov_field(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Posterior mean covariance field E[Σ(X) | data].

        Parameters
        ----------
        X : jnp.ndarray
            Test stimuli, shape (n_test, input_dim) OR (n_test, k_stim, input_dim)
            Note that standard use for OddityTask is 2D with X as test REFS, not all stimuli.

        Returns
        -------
        jnp.ndarray
            Covariance matrices, shape (n_test, input_dim, input_dim) OR (n_test, k_stim, input_dim, input_dim)

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

        # Vectorized evaluation: (n_samples, n_test, k_stim, input_dim, input_dim)
        if jnp.ndim(X) == 3:  # more than 1 stimulus
            cov_samples = jax.vmap(
                lambda params: jax.vmap(jax.vmap(lambda s: cov_at_x(params, s)))(X)
            )(param_samples)
        elif jnp.ndim(X) == 2:  # only 1 stimulus
            cov_samples = jax.vmap(
                lambda params: jax.vmap(lambda s: cov_at_x(params, s))(X)
            )(param_samples)
        else:
            raise ValueError(
                "Incorrect input dimensionality"
                "Expected 2D (n_test, input_dim) or 3D (n_test, k_stim, input_dim)"
                f"Received {jnp.ndim(X)}D."
            )

        # Return posterior mean
        return jnp.mean(cov_samples, axis=0)
