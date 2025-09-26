"""
map_optimizer.py
----------------

MAP (Maximum A Posteriori) optimizer using Optax.

MVP implementation:
- Uses gradient ascent on log posterior.
- Defaults to SGD with momentum, but any Optax optimizer can be passed in.

Connections
-----------
- Calls WPPM.log_posterior_from_data(params, data) as the objective.
- Returns a Posterior object wrapping the MAP estimate.
"""

from __future__ import annotations

import jax
import optax

from psyphy.inference.base import InferenceEngine
from psyphy.posterior.posterior import Posterior


class MAPOptimizer(InferenceEngine):
    """
    MAP (Maximum A Posteriori) optimizer.

    Parameters
    ----------
    steps : int, default=500
        Number of optimization steps.
    optimizer : optax.GradientTransformation, optional
        Optax optimizer to use. Default: SGD with momentum.

    Notes
    -----
    - Loss function = negative log posterior.
    - Gradients computed with jax.grad.
    """


    def __init__(self, steps: int = 500, optimizer: optax.GradientTransformation | None = None):
        self.steps = steps
        self.optimizer = optimizer or optax.sgd(learning_rate=5e-5, momentum=0.9)

    def fit(self, model, data) -> Posterior:
        """
        Fit model parameters with MAP optimization.

        Parameters
        ----------
        model : WPPM
            Model instance.
        data : ResponseData
            Observed trials.

        Returns
        -------
        Posterior
            Posterior wrapper around MAP params and model.
        """

        def loss_fn(params):
            return -model.log_posterior_from_data(params, data)

        params = model.init_params(jax.random.PRNGKey(0))
        opt_state = self.optimizer.init(params)

        @jax.jit
        def step(params, opt_state):
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = self.optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        for _ in range(self.steps):
            params, opt_state, loss = step(params, opt_state)

        return Posterior(params=params, model=model)
