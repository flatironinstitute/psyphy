r"""psyphy.model.likelihood.base

The task-likelihood hierarchy, independent of any particular task.

``TaskLikelihood`` is the abstract interface. The two intermediate classes below
it are organized by the **assumed distribution of the responses**, not by the
task: each one implements ``loglik`` and ``simulate`` once, so a concrete task
only has to implement ``predict``.

    TaskLikelihood (ABC)
      |-- BernoulliTaskLikelihood   binary responses; predict -> (p_correct,)
      \-- GaussianTaskLikelihood    continuous responses; predict -> (mu, sigma)

Concrete tasks live beside this module, one file per task; see ``oddity.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.stats as jsp


class TaskLikelihood(ABC):
    """
    Abstract base class for task likelihoods.

    Subclasses must implement:
    - ``predict(params, stimuli, model, *, key)`` -> probability parameters for one trial

    The base class provides concrete implementations of:
    - ``loglik(params, data, model, *, key)`` -> Bernoulli log-likelihood over a batch
    - ``simulate(params, stimuli, model, *, key)`` -> simulated responses & probability parameters

    The Bernoulli log-likelihood step is identical for all binary-response tasks,
    so it lives here rather than being re-implemented in every subclass.
    """

    @abstractmethod
    def predict(
        self,
        params: Any,
        stimuli: jnp.ndarray,
        model: Any,
        *,
        key: Any = None,
    ) -> tuple[jnp.ndarray, ...]:
        """Return parameters sufficient to determine probability of any given
        response on a single trial. The parameters will be specific to the
        distribution (specified by abstract subclass)

        BernoulliTaskLikelihood: returns (p(correct))
        GaussianTaskLikelihood: returns (mu, sigma), the parameters defining
        a Gaussian distribution.

        Parameters
        ----------
        params : Any
            Model parameters.
        stimuli : jnp.ndarray, shape (stim_dim, K)
            stimuli for a given trial (stimulus dimensions, number of stimuli)
        model : Any
            Model instance (provides covariance structure and ``model.noise``).
        key : jax.Array, optional
            PRNG key for stochastic tasks. When None, the task falls back to
            its ``config.default_key_seed``.

        Returns
        -------
        parameter(s): jnp.ndarray
            any number of parameters necessary to specify probability for likelihood calculation
            - for Bernoulli: p(correct) (scalar)
            - for Gaussian: mu (r_dim vector), sigma (r_dim x r_dim covariance matrix)
        """
        ...

    @abstractmethod
    def loglik(
        self,
        params: Any,
        data: jnp.ndarray,
        model: Any,
        *,
        key: Any = None,
    ) -> jnp.ndarray:
        """
        Compute and return log-likelihood over a batch of trials.
        Each abstract subclass will concretely implement this for all tasks with
        the assumption of a given distribution.

        Parameters
        ----------
        params : Any
            Model parameters.
        data : Any
            Object with ``.stimuli``, ``.responses`` array attributes.
        model : Any
            Model instance (provides covariance structure and ``model.noise``).
        key : jax.Array, optional
            PRNG key for stochastic tasks. When None, the task falls back to
            its ``config.default_key_seed``.

        Returns
        -------
        jnp.ndarray, shape ()
            Scalar sum of log-likelihoods over all trials.
        """
        ...

    @abstractmethod
    def simulate(
        self,
        params: Any,
        stimuli: jnp.ndarray,
        model: Any,
        *,
        key: Any,
    ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, ...]]:
        """Simulate responses for a batch of trials.

        Parameters
        ----------
        params : Any
            Model parameters.
        stimuli : jnp.ndarray, shape (n_trials, k_stimuli, input_dim)
            Stimuli.
        model : Any
            Model instance.
        key : jax.Array
            PRNG key (required; split internally for prediction and sampling).

        Returns
        -------
        responses : jnp.ndarray, shape (n_trials, r_dim)
            Simulated responses.
        probability parameters : tuple[jnp.ndarray, ...], each array has shape[0] = n_trials
            Bernoulli: Estimated (P(correct)) per trial used to draw the responses. shape = (n_trials)
            Gaussian: Estimated (mu, sigma) per trial used to draw the responses. shape for mu = (n_trials, r_dims); shape for sigma = (n_trials, r_dims, r_dims).

        Note: not yet implemented for Gaussians
        """
        ...


class BernoulliTaskLikelihood(TaskLikelihood):
    """
    Intermediate abstract subclass for task likelihoods.

    Subclasses must implement:
    - ``predict(params, stimuli, model, *, key)`` -> p(correct) for one trial

    This abstract subclass provides concrete implementations of:
    - ``loglik(params, data, model, *, key)`` -> Bernoulli log-likelihood over a batch
    - ``simulate(params, inputs, model, *, key)`` -> simulated responses

    The Bernoulli log-likelihood step is identical for all binary-response tasks,
    so it lives here rather than being re-implemented in every subclass.
    """

    def loglik(
        self,
        params: Any,
        data: Any,
        model: Any,
        *,
        key: Any = None,
    ) -> jnp.ndarray:
        """Compute Bernoulli log-likelihood over a batch of trials.

        This is a concrete base-class method: it vmaps ``predict`` over trials
        then applies the Bernoulli log-likelihood formula. Subclasses only need
        to implement ``predict``.

        Parameters
        ----------
        params : Any
            Model parameters.
        data : Any
            Object with ``.stimuli``, ``.responses`` array attributes.
        model : Any
            Model instance.
        key : jax.Array, optional
            PRNG key. Passed as independent per-trial subkeys to ``predict``.
            When None, falls back to ``key=jr.PRNGKey(0)`` (deterministic).

        Returns
        -------
        jnp.ndarray, shape ()
            Scalar sum of Bernoulli log-likelihoods over all trials.
        """
        stimuli = jnp.asarray(data.stimuli)
        # TrialData stores responses as (N, R): R=1 for binary tasks.
        # Extract the single binary channel to get shape (N,).
        # Without this, jnp.where broadcasts (N, 1) against probs (N,),
        # treating (N,) as (1, N) and producing an (N, N) result whose
        # sum scrambles the gradient: every p[j] receives signal from all
        # N responses instead of only response[j].

        responses = jnp.asarray(data.responses)
        if responses.ndim == 2:  # shape (N, R)
            responses = responses[:, 0]
        responses = responses.astype(int)
        n_trials = int(stimuli.shape[0])

        base_key = key if key is not None else jr.PRNGKey(0)
        trial_keys = jr.split(base_key, n_trials)

        prob_params = jax.vmap(
            lambda stim, k: self.predict(params, stim, model, key=k)
        )(stimuli, trial_keys)

        probs = prob_params[0]

        log_likelihoods = jnp.where(
            responses == 1,
            jnp.log(probs),
            jnp.log(1.0 - probs),
        )

        return jnp.sum(log_likelihoods)

    def simulate(
        self,
        params: Any,
        stimuli: jnp.ndarray,
        model: Any,
        *,
        key: Any,
    ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, ...]]:
        """Simulate observed binary responses for a batch of trials.

        Parameters
        ----------
        params : Any
            Model parameters.
        stimuli : jnp.ndarray, shape (n_trials, k_stim, input_dim)
            Stimuli.
        model : Any
            Model instance.
        key : jax.Array
            PRNG key (required; split internally for prediction and sampling).

        Returns
        -------
        responses : jnp.ndarray, shape (n_trials,), dtype int32
            Simulated binary responses (1 = correct, 0 = incorrect).
        probability parameters : tuple[jnp.ndarray]
            1-tuple ``(p_correct,)``. ``p_correct`` has shape (n_trials,) and holds
            the estimated P(correct) per trial used to draw the responses.
        """
        stimuli = jnp.asarray(stimuli)
        n_trials = int(stimuli.shape[0])

        k_pred, k_bernoulli = jr.split(key)
        trial_keys = jr.split(k_pred, n_trials)

        p_correct = jax.vmap(lambda stim, k: self.predict(params, stim, model, key=k))(
            stimuli, trial_keys
        )

        responses = jr.bernoulli(k_bernoulli, p_correct[0]).astype(jnp.int32)
        return (responses, p_correct)


class GaussianTaskLikelihood(TaskLikelihood):
    """
    Intermediate abstract subclass for task likelihoods.

    Subclasses must implement:
    - ``predict(params, stimuli, model, *, key)`` -> (mu, sigma) of distribution for one trial

    This abstract subclass provides concrete implementations of:
    - ``loglik(params, data, model, *, key)`` -> Gaussian log-likelihood over a batch
    - ``simulate(params, inputs, model, *, key)`` -> simulated responses

    The Gaussian log-likelihood step is identical for all tasks with the assumption
    of Gaussian response distributions, so it lives here rather than being
    re-implemented in every subclass.
    """

    def loglik(
        self,
        params: Any,
        data: Any,
        model: Any,
        *,
        key: Any = None,
    ) -> jnp.ndarray:
        """Compute Gaussian log-likelihood over a batch of trials.

        This is a concrete base-class method: it vmaps ``predict`` over trials
        then applies the Gaussian log-likelihood formula. Subclasses only need
        to implement ``predict``.

        Parameters
        ----------
        params : Any
            Model parameters.
        data : Any
            Object with ``.stimuli``, ``.responses`` array attributes.
        model : Any
            Model instance.
        key : jax.Array, optional
            PRNG key. Passed as independent per-trial subkeys to ``predict``.
            When None, falls back to ``key=jr.PRNGKey(0)`` (deterministic).

        Returns
        -------
        jnp.ndarray, shape ()
            Scalar sum of Gaussian log-likelihoods over all trials.
        """
        stimuli = jnp.asarray(data.stimuli)
        responses = jnp.asarray(data.responses)
        n_trials = int(stimuli.shape[0])

        base_key = key if key is not None else jr.PRNGKey(0)
        trial_keys = jr.split(base_key, n_trials)

        mu, sigma = jax.vmap(lambda stim, k: self.predict(params, stim, model, key=k))(
            stimuli, trial_keys
        )

        # ensure sigma is correct shape:
        sigma = jnp.squeeze(sigma)
        if n_trials == 1:
            sigma = jnp.expand_dims(sigma, axis=0)

        log_likelihoods = jax.vmap(
            lambda resp, m, s: jsp.multivariate_normal.logpdf(x=resp, mean=m, cov=s)
        )(responses, mu, sigma)

        def nan_loglik(logliks):
            if any(jnp.isnan(logliks)):
                import warnings

                warnings.warn(
                    "GaussianTaskLikelihood: NaN log-likelihoods detected. "
                    "This is likely caused by a non positive-definite covariance matrix",
                    stacklevel=1,
                )

        jax.debug.callback(nan_loglik, log_likelihoods)

        return jnp.sum(log_likelihoods)

    def simulate(
        self,
        params: Any,
        stimuli: jnp.ndarray,
        model: Any,
        *,
        key: Any,
    ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, ...]]:
        """Simulate observed responses for a batch of trials.

        Parameters
        ----------
        params : Any
            Model parameters.
        stimuli : jnp.ndarray, shape (n_trials, k_stim, input_dim)
            Stimuli.
        model : Any
            Model instance.
        key : jax.Array
            PRNG key (required; split internally for prediction and sampling).

        Returns
        -------
        responses : jnp.ndarray, shape (n_trials, r_dims),
            Simulated responses.
        prob_params : jnp.ndarray, shape (n_trials, )
            Estimated (mu, sigma) per trial used to draw the responses.
        """
        raise NotImplementedError("Gaussian Task simulations not yet implemented")
