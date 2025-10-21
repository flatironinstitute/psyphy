"""
acquisition
===========

Acquisition functions for adaptive experimental design.

This module provides:
- AcquisitionFunction: Protocol for acquisition functions
- optimize_acqf(): Functional interface for optimization
- Common acquisition functions (Expected Improvement (EI),
    Upper Confidence Bound (UCB), Information Gain (IG))

Design
------
Unlike BoTorch's class-based approach, we use a functional style:
    acq_fn = lambda X: expected_improvement(model.posterior(X), best_f)
    X_next = optimize_acqf(acq_fn, bounds, q=1)

This is simpler and more composable than inheritance hierarchies.

Available Functions
-------------------
- expected_improvement: Maximize expected improvement over best observation
- upper_confidence_bound: Balance exploration vs exploitation
- probability_of_improvement: Maximize probability of improvement
- mutual_information: Maximize information gain
    (Bayesian Active Learning by Disagreement (BALD))

Optimization Methods
--------------------
- optimize_acqf_discrete: Exhaustive search over candidate set
- optimize_acqf: Gradient-based optimization (Optax)
- optimize_acqf_random: Random search baseline

Examples
--------
>>> # Discrete optimization
>>> candidates = jnp.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
>>> acq_values = expected_improvement(model.posterior(candidates), best_f=0.5)
>>> X_next = candidates[jnp.argmax(acq_values)]

>>> # Continuous optimization with gradient descent
>>> def acq_fn(X):
...     return expected_improvement(model.posterior(X), best_f=0.5)
>>> X_next = optimize_acqf(
...     acq_fn,
...     bounds=jnp.array([[0.0, 1.0], [0.0, 1.0]]),
...     q=1,
...     method="gradient",
... )
"""

from psyphy.acquisition.base import AcquisitionFunction
from psyphy.acquisition.expected_improvement import (
    expected_improvement,
    log_expected_improvement,
)
from psyphy.acquisition.mutual_information import mutual_information
from psyphy.acquisition.optimize import (
    optimize_acqf,
    optimize_acqf_discrete,
    optimize_acqf_random,
)
from psyphy.acquisition.upper_confidence_bound import upper_confidence_bound

__all__ = [
    "AcquisitionFunction",
    "expected_improvement",
    "log_expected_improvement",
    "upper_confidence_bound",
    "mutual_information",
    "optimize_acqf",
    "optimize_acqf_discrete",
    "optimize_acqf_random",
]
