"""
task.py
-------

Task likelihoods for different psychophysical experimetns.

Each TaskLikelihood defines:
- predict(params, stimuli, model, noise)
    Map discriminability (computed by model) to probability of correct response.

- loglik(params, data, model, noise)
    Compute log-likelihood of observed responses under this task.

MVP implementation:
- OddityTask (3AFC) and TwoAFC.
- Both use simple sigmoid-like mappings of discriminability -> performance
- loglik implemented as Bernoulli log-prob with these predictions

Forward compatibility (Full WPPM mode):
- Tasks will call into WPPM for discriminability computed via Monte Carlo
  observer simulations, not closed forms.
- Noise models will be used explicitly to generate internal noisy reps.
- This ensures the same API supports both MVP and Full WPPM mode.

Connections
-----------
- WPPM delegates to task.predict and task.loglik (never re-implements likelihood)
-  noise model is passed through from WPPM so tasks can simulate responses.
- we can define new tasks by subclassing TaskLikelihood and implementing
  predict() and loglik().
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import jax.numpy as jnp

Stimulus = tuple[jnp.ndarray, jnp.ndarray]


class TaskLikelihood(ABC):
    """
    Abstract base class for task likelihoods
    """

    @abstractmethod
    def predict(
        self, params: Any, stimuli: Stimulus, model: Any, noise: Any
    ) -> jnp.ndarray:
        """Predict probability of correct response for a stimulus."""
        ...

    @abstractmethod
    def loglik(self, params: Any, data: Any, model: Any, noise: Any) -> jnp.ndarray:
        """Compute log-likelihood of observed responses under this task"""
        ...


class OddityTask(TaskLikelihood):
    """Three-alternative forced-choice oddity task (MVP placeholder) ("pick the odd-one out)."""

    def __init__(self, slope: float = 1.5) -> None:
        self.slope = float(slope)
        self.chance_level: float = 1.0 / 3.0
        self.performance_range: float = 1.0 - self.chance_level

    def predict(
        self, params: Any, stimuli: Stimulus, model: Any, noise: Any
    ) -> jnp.ndarray:
        d = model.discriminability(params, stimuli)
        g = 0.5 * (jnp.tanh(self.slope * d) + 1.0)
        return self.chance_level + self.performance_range * g

    def loglik(self, params: Any, data: Any, model: Any, noise: Any) -> jnp.ndarray:
        refs, probes, responses = data.to_numpy()
        ps = jnp.array(
            [self.predict(params, (r, p), model, noise) for r, p in zip(refs, probes)]
        )
        eps = 1e-9
        return jnp.sum(
            jnp.where(responses == 1, jnp.log(ps + eps), jnp.log(1.0 - ps + eps))
        )


class TwoAFC(TaskLikelihood):
    """2-alternative forced-choice task (MVP placeholder)."""

    def __init__(self, slope: float = 2.0) -> None:
        self.slope = float(slope)
        self.chance_level: float = 0.5
        self.performance_range: float = 1.0 - self.chance_level

    def predict(
        self, params: Any, stimuli: Stimulus, model: Any, noise: Any
    ) -> jnp.ndarray:
        d = model.discriminability(params, stimuli)
        return self.chance_level + self.performance_range * jnp.tanh(self.slope * d)

    def loglik(self, params: Any, data: Any, model: Any, noise: Any) -> jnp.ndarray:
        refs, probes, responses = data.to_numpy()
        ps = jnp.array(
            [self.predict(params, (r, p), model, noise) for r, p in zip(refs, probes)]
        )
        eps = 1e-9
        return jnp.sum(
            jnp.where(responses == 1, jnp.log(ps + eps), jnp.log(1.0 - ps + eps))
        )
