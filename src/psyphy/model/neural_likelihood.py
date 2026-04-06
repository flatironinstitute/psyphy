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

import jax.numpy as jnp

from psyphy.model.likelihood import TaskLikelihood

# Type alias for neural network parameter PyTree
NNParams = Any


class NeuralSurrogateOddityTask(TaskLikelihood):
    """
    Likelihood estimated by a Neural Network surrogate.

    This class skeleton shows how to plug in a pre-trained neural network
    to estimate P(correct) instead of running MC simulations.

    Only ``predict`` needs to be implemented — ``loglik`` and ``simulate``
    are inherited from ``TaskLikelihood`` for free.
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
            ``p_correct = forward(nn_params, features)``.
        """
        # If you're coming from PyTorch: JAX vs PyTorch difference:
        # in JAX, the "model" function is typically pure and stateless.
        # We must store the parameters (state) separately and pass them explicitly.
        # PyTorch: self.model = Model() (contains both state and logic)
        self.nn_params = nn_params  # The frozen WEIGHTS (state)
        self.forward = forward  # The pure FUNCTION (stateless logic)

    def predict(
        self,
        params: Any,
        ref: jnp.ndarray,
        comparison: jnp.ndarray,
        model: Any,
        *,
        key: Any = None,
    ) -> jnp.ndarray:
        """
        Predict p(correct) for a single stimulus pair using the neural network.

        Implementation steps:
        1. Extract relevant geometry (e.g. covariance matrices at ref/comparison)
           from ``model`` using ``model._compute_sqrt(params, ref)`` etc.
        2. Format these as a feature vector for the network.
        3. Run ``self.forward(self.nn_params, features)``.

        ``loglik`` and ``simulate`` are inherited from ``TaskLikelihood`` and will
        vmap this method automatically — no further implementation needed.
        """
        # Placeholder: extract features from the model at ref/comparison.
        # e.g., features = custom_feature_extractor(model, params, ref, comparison)
        features = jnp.zeros(10)  # Dummy features

        # Forward pass through the surrogate.
        # Unlike PyTorch `model(x)`, JAX requires explicit parameter passing.
        p_correct = self.forward(self.nn_params, features)

        return p_correct
