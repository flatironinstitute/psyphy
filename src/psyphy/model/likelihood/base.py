"""psyphy.model.likelihood.base
------------------------------

Abstract base class for all task likelihoods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr


class TaskLikelihood(ABC):
    """
    Abstract base class for task likelihoods.

    Subclasses must implement:
    - ``predict(params, ref, comparison, model, *, key)`` → p(correct) for one trial

    The base class provides concrete implementations of:
    - ``loglik(params, data, model, *, key)`` → Bernoulli log-likelihood over a batch
    - ``simulate(params, refs, comparisons, model, *, key)`` → simulated responses

    The Bernoulli log-likelihood step is identical for all binary-response tasks,
    so it lives here rather than being re-implemented in every subclass.
    """

    @abstractmethod
    def predict(
        self,
        params: Any,
        ref: jnp.ndarray,
        comparison: jnp.ndarray,
        model: Any,
        *,
        key: Any = None,
    ) -> jnp.ndarray:
        """Return p(correct) for a single (ref, comparison) trial.

        Parameters
        ----------
        params : Any
            Model parameters.
        ref : jnp.ndarray, shape (input_dim,)
            Reference stimulus.
        comparison : jnp.ndarray, shape (input_dim,)
            Comparison stimulus.
        model : Any
            Model instance (provides covariance structure and ``model.noise``).
        key : jax.random.KeyArray, optional
            PRNG key for stochastic tasks. When None, the task falls back to
            its ``config.default_key_seed``.

        Returns
        -------
        jnp.ndarray
            Scalar p(correct) in (0, 1).
        """
        ...

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
            Object with ``.refs``, ``.comparisons``, ``.responses`` array attributes.
        model : Any
            Model instance.
        key : jax.random.KeyArray, optional
            PRNG key. Passed as independent per-trial subkeys to ``predict``.
            When None, falls back to ``key=jr.PRNGKey(0)`` (deterministic).

        Returns
        -------
        jnp.ndarray
            Scalar sum of Bernoulli log-likelihoods over all trials.
        """
        refs = jnp.asarray(data.refs)
        comparisons = jnp.asarray(data.comparisons)
        responses = jnp.asarray(data.responses)
        n_trials = int(refs.shape[0])

        base_key = key if key is not None else jr.PRNGKey(0)
        trial_keys = jr.split(base_key, n_trials)

        probs = jax.vmap(
            lambda ref, comparison, k: self.predict(
                params, ref, comparison, model, key=k
            )
        )(refs, comparisons, trial_keys)

        log_likelihoods = jnp.where(
            responses == 1,
            jnp.log(probs),
            jnp.log(1.0 - probs),
        )
        return jnp.sum(log_likelihoods)

    def simulate(
        self,
        params: Any,
        refs: jnp.ndarray,
        comparisons: jnp.ndarray,
        model: Any,
        *,
        key: Any,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Simulate observed binary responses for a batch of trials.

        Parameters
        ----------
        params : Any
            Model parameters.
        refs : jnp.ndarray, shape (n_trials, input_dim)
            Reference stimuli.
        comparisons : jnp.ndarray, shape (n_trials, input_dim)
            Comparison stimuli.
        model : Any
            Model instance.
        key : jax.random.KeyArray
            PRNG key (required; split internally for prediction and sampling).

        Returns
        -------
        responses : jnp.ndarray, shape (n_trials,), dtype int32
            Simulated binary responses (1 = correct, 0 = incorrect).
        p_correct : jnp.ndarray, shape (n_trials,)
            Estimated P(correct) per trial used to draw the responses.
        """
        refs = jnp.asarray(refs)
        comparisons = jnp.asarray(comparisons)
        n_trials = int(refs.shape[0])

        k_pred, k_bernoulli = jr.split(key)
        trial_keys = jr.split(k_pred, n_trials)

        p_correct = jax.vmap(
            lambda ref, comparison, k: self.predict(
                params, ref, comparison, model, key=k
            )
        )(refs, comparisons, trial_keys)

        responses = jr.bernoulli(k_bernoulli, p_correct).astype(jnp.int32)
        return responses, p_correct
