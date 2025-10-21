"""
base.py
-------

Base protocol for acquisition functions.

Design
------
We use Protocol (not ABC) for maximum flexibility. An acquisition function
is just any callable that:
    1. Takes test points X
    2. Returns scalar scores (higher = better)

This enables functional composition without inheritance.
"""

from __future__ import annotations

from typing import Callable, Protocol

import jax.numpy as jnp


class AcquisitionFunction(Protocol):
    """
    Protocol for acquisition functions.

    An acquisition function scores candidate points for selection in
    adaptive experimental design. Higher scores indicate more valuable points.

    Examples
    --------
    >>> # Function-based acquisition
    >>> def my_acquisition(X):
    ...     posterior = model.posterior(X)
    ...     return posterior.mean + 2.0 * jnp.sqrt(posterior.variance)
    >>>
    >>> X_next = optimize_acqf(my_acquisition, bounds, q=1)

    >>> # Lambda-based acquisition
    >>> acq_fn = lambda X: -posterior.variance  # Minimize uncertainty
    >>> X_next = optimize_acqf(acq_fn, bounds, q=1)

    Notes
    -----
    We deliberately do NOT use a class hierarchy (unlike BoTorch's
    AcquisitionFunction base class). Functional composition is simpler
    and more flexible for research code.

    If you need stateful acquisition (e.g., caching), use a callable class:
        class CachedAcquisition:
            def __init__(self, model):
                self.model = model
                self._cache = {}

            def __call__(self, X):
                # Use cache...
                return scores
    """

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate acquisition function at candidate points.

        Parameters
        ----------
        X : jnp.ndarray, shape (n_candidates, input_dim)
            Candidate test points

        Returns
        -------
        jnp.ndarray, shape (n_candidates,)
            Acquisition scores (higher = better)
        """
        ...


# Type alias for convenience
AcqFn = Callable[[jnp.ndarray], jnp.ndarray]
