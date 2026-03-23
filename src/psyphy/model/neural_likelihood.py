"""
neural_likelihood.py
--------------------

Demonstration of how to implement a neural network-based likelihood model.
This is an alternative to the computationally expensive Monte Carlo simulation in OddityTask
with a fast, differentiable neural network approximation at prediction time.

This skeleton demonstrates the minimum requirements to integrate a new
likelihood model into the library.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp

from psyphy.model.likelihood import Stimulus, TaskLikelihood

# Type alias for neural network parameter PyTree
NNParams = Any


class NeuralSurrogateOddityTask(TaskLikelihood):
    """
    Likelihood estimated by a Neural Network surrogate.

    This class skeleton shows how to plug in a pre-trained neural network
    to estimate P(correct) instead of running MC simulations.
    """

    def __init__(
        self,
        nn_params: NNParams,
        forward: Callable[[NNParams, jnp.ndarray], jnp.ndarray],
    ) -> None:
        """
        Initialize with a trained neural network.

        Parameters
        ----------
        nn_params : NNParams
            The frozen parameters (weights/biases) of the pre-trained neural network.
        forward : Callable[[NNParams, jnp.ndarray], jnp.ndarray]
            The JAX function that runs the forward pass:
            `p_correct = forward(params, features)`.
        """
        # If you're coming from PyTorch: JAX vs PyTorch difference:
        # in JAX, the "model" function is typically pure and stateless.
        # We must store the parameters (state) separately and pass them explicitly.
        # PyTorch: self.model = Model() (contains both state and logic)
        self.nn_params = nn_params  # The frozen WEIGHTS (state)
        self.forward = forward  # The pure FUNCTION (stateless logic)

    def predict(
        self, params: Any, stimuli: Stimulus, model: Any, noise: Any
    ) -> jnp.ndarray:
        """
        Predict p(correct) for a single stimulus pair using the Neural Net.

        Implementation steps:
        1. Extract relevant geometry (e.g. covariance matrices, x_0, x_1, z_0, z_0', z_1) from `model`
           at the stimulus locations.
        2. Format these as features for the neural network.
        3. Run `self.forward`.
        """
        # ref, comparison = stimuli

        # Placeholder: Extract features from the model based on ref/comparison
        # e.g., features = custom_feature_extractor(model, ref, comparison, ...)
        features = jnp.zeros(10)  # Dummy features

        # Forward pass through the surrogate
        # Unlike PyTorch `model(x)`, JAX requires explicit parameter passing: `f(params, x)`
        p_correct = self.forward(self.nn_params, features)

        return p_correct

    def loglik(
        self, params: Any, data: Any, model: Any, noise: Any, **kwargs: Any
    ) -> jnp.ndarray:
        """
        Compute total log-likelihood for a dataset using the Neural Net.

        This allows you to vectorize the neural network application across all trials.
        """
        # 1. Access data arrays
        # The data object is expected to be a TrialData instance or similar PyTree
        # compatible with JIT compilation.
        refs = data.refs
        comparisons = data.comparisons
        responses = data.responses

        # 2. Define single-trial logic (same as predict, but efficient for vmap)
        def single_trial_prob(r, c):
            # Same feature extraction logic as predict
            features = jnp.zeros(10)  # Dummy features
            return self.forward(self.nn_params, features)

        # 3. Vectorize prediction across all trials
        probs = jax.vmap(single_trial_prob)(refs, comparisons)

        # 4. Compute standard Bernoulli log-likelihood
        # LL = Σ [y * log(p) + (1-y) * log(1-p)]
        # Add clipping for numerical stability
        probs = jnp.clip(probs, 1e-6, 1.0 - 1e-6)
        log_likelihoods = jnp.where(
            responses == 1,
            jnp.log(probs),
            jnp.log(1.0 - probs),
        )

        return jnp.sum(log_likelihoods)
