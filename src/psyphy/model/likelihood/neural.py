"""psyphy.model.likelihood.neural
---------------------------------

Neural network surrogate likelihoods.

Provides a fast, differentiable alternative to MC simulation by replacing
the likelihood computation with a pre-trained neural network.

Class hierarchy
---------------
    TaskLikelihood (ABC)                  base.py
         |
    NeuralSurrogateTask (ABC)             this file
         |
    NeuralSurrogateOddityTask             this file
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Callable

import jax.numpy as jnp

from .base import TaskLikelihood

# Type alias for neural network parameter PyTree
NNParams = Any


class NeuralSurrogateTask(TaskLikelihood):
    """
    Abstract base class for neural network surrogate likelihoods.

    Owns the shared NN infrastructure contract: ``nn_params`` and ``forward``.
    Subclasses implement ``predict`` for their specific task (oddity, 2AFC, etc.).

    ``loglik`` and ``simulate`` are inherited from ``TaskLikelihood`` for free —
    they vmap ``predict`` automatically.

    Parameters
    ----------
    nn_params : NNParams
        Frozen parameters (weights/biases) of the pre-trained neural network.
    forward : Callable[[NNParams, jnp.ndarray], jnp.ndarray]
        Pure JAX function for the forward pass: ``p_correct = forward(nn_params, features)``.
        Must be JIT-compatible and return a scalar in (0, 1).
    """

    def __init__(
        self,
        nn_params: NNParams,
        forward: Callable[[NNParams, jnp.ndarray], jnp.ndarray],
    ) -> None:
        # JAX convention: state (nn_params) and logic (forward) are kept separate.
        # This is unlike PyTorch where both live inside the model object.
        self.nn_params = nn_params  # frozen weights
        self.forward = forward  # pure stateless function

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
        """Return p(correct) for a single trial using the neural surrogate.

        Subclasses must implement feature extraction from ``model`` at
        ``ref`` / ``comparison`` and pass those features to ``self.forward``.
        """
        ...


class NeuralSurrogateOddityTask(NeuralSurrogateTask):
    """
    Neural network surrogate for the 3-AFC oddity task.

    Replaces the MC simulation in :class:`~psyphy.model.likelihood.OddityTask`
    with a fast neural network forward pass.

    Only ``predict`` is implemented here — ``loglik`` and ``simulate`` are
    inherited from ``TaskLikelihood`` through ``NeuralSurrogateTask``.

    Notes
    -----
    Training data for the surrogate should be generated with
    :class:`~psyphy.model.likelihood.OddityTask` (the MC ground truth).
    """

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
        Predict p(correct) for a single oddity trial using the neural surrogate.

        Implementation steps:
        1. Extract geometry at ref and comparison using ``model._compute_sqrt(params, x)``.
        2. Format the covariance information as a fixed-size feature vector.
        3. Call ``self.forward(self.nn_params, features)`` to get p(correct).

        ``loglik`` and ``simulate`` are inherited from ``TaskLikelihood`` and will
        vmap this method automatically — no further implementation needed.
        """
        # Placeholder: extract features from model at ref/comparison.
        # e.g.:
        #   U_ref = model._compute_sqrt(params, ref)
        #   U_comp = model._compute_sqrt(params, comparison)
        #   features = jnp.concatenate([U_ref.ravel(), U_comp.ravel(), ref, comparison])
        features = jnp.zeros(10)  # replace with real feature extraction

        return self.forward(self.nn_params, features)
