"""
upper_confidence_bound.py
-------------------------

Upper Confidence Bound (UCB) acquisition function.

Balances exploration and exploitation via a tunable parameter β.

References
----------
Srinivas, N., Krause, A., Kakade, S. M., & Seeger, M. (2009).
Gaussian process optimization in the bandit setting: No regret and
experimental design. arXiv preprint arXiv:0912.3995.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

if TYPE_CHECKING:
    from psyphy.posterior import PredictivePosterior


def upper_confidence_bound(
    posterior: PredictivePosterior,
    beta: float = 2.0,
    maximize: bool = True,
) -> jnp.ndarray:
    r"""
    Upper confidence bound acquisition function.

    Computes μ(x) + β * \sigma(x) for maximization.
    Computes μ(x) - β * \sigma(x) for minimization.

    Parameters
    ----------
    posterior : PredictivePosterior
        Predictive posterior p(f(X*) | data)
    beta : float, default=2.0
        Exploration-exploitation trade-off parameter.
        - β = 0: Pure exploitation (greedy selection)
        - β = 1: Balanced
        - β > 2: Aggressive exploration
    maximize : bool, default=True
        If True, maximize (higher is better).
        If False, minimize (lower is better).

    Returns
    -------
    jnp.ndarray, shape (n_candidates,)
        UCB values for each candidate

    Examples
    --------
    >>> # Balanced exploration-exploitation
    >>> posterior = model.posterior(X_candidates, probes=probes)
    >>> ucb = upper_confidence_bound(posterior, beta=2.0)
    >>> X_next = X_candidates[jnp.argmax(ucb)]

    >>> # Aggressive exploration (high β)
    >>> ucb = upper_confidence_bound(posterior, beta=5.0)

    >>> # Pure exploitation (β = 0)
    >>> ucb = upper_confidence_bound(posterior, beta=0.0)  # Just selects max mean

    >>> # With optimization
    >>> def acq_fn(X):
    ...     return upper_confidence_bound(model.posterior(X), beta=2.0, maximize=True)
    >>> X_next, ucb_val = optimize_acqf(acq_fn, bounds, q=1)

    Notes
    -----
    For psychophysics:
    - Use β ∈ [1, 3] for typical experiments
    - Larger β explores uncertain regions more
    - Smaller β focuses on high-accuracy regions

    UCB is often faster to compute than EI (no need for CDF/PDF),
    but theoretically less well-motivated for finite-sample regret.

    Adaptive β
    ----------
    Theoretically, β should grow with number of trials:
        β_t = sqrt(2 * log(input_dim * t^2 * π^2 / 6δ))

    where t is the trial number and δ is a confidence parameter.
    In practice, a fixed β ∈ [1, 3] works well.
    """
    mean = posterior.mean
    std = jnp.sqrt(posterior.variance)

    ucb = mean + beta * std if maximize else mean - beta * std

    return ucb


def lower_confidence_bound(
    posterior: PredictivePosterior,
    beta: float = 2.0,
) -> jnp.ndarray:
    """
    Lower confidence bound (LCB) for minimization.

    Alias for upper_confidence_bound(..., maximize=False).

    Parameters
    ----------
    posterior : PredictivePosterior
        Predictive posterior
    beta : float, default=2.0
        Exploration parameter

    Returns
    -------
    jnp.ndarray
        LCB values
    """
    return upper_confidence_bound(posterior, beta=beta, maximize=False)
