"""
expected_improvement.py
-----------------------

Expected Improvement (EI) acquisition function.

The most popular acquisition function for Bayesian optimization.
Balances exploration (high uncertainty) and exploitation (high mean).

References
----------
Mockus, J., Tiesis, V., & Zilinskas, A. (1978). The application of Bayesian
methods for seeking the extremum. Towards Global Optimization, 2, 117-129.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax.scipy import stats

if TYPE_CHECKING:
    from psyphy.posterior import PredictivePosterior


def expected_improvement(
    posterior: PredictivePosterior,
    best_f: float,
    maximize: bool = True,
) -> jnp.ndarray:
    """
    Expected improvement acquisition function.

    Computes E[max(0, f(x) - best_f)] for each candidate point.

    Parameters
    ----------
    posterior : PredictivePosterior
        Predictive posterior p(f(X*) | data)
    best_f : float
        Best observed value so far
    maximize : bool, default=True
        If True, maximize (higher is better).
        If False, minimize (lower is better).

    Returns
    -------
    jnp.ndarray, shape (n_candidates,)
        EI values for each candidate

    Examples
    --------
    >>> # Basic usage
    >>> posterior = model.posterior(X_candidates, probes=probes)
    >>> best_f = jnp.max(y_observed)
    >>> ei = expected_improvement(posterior, best_f)
    >>> X_next = X_candidates[jnp.argmax(ei)]

    >>> # Minimization
    >>> best_f = jnp.min(y_observed)
    >>> ei = expected_improvement(posterior, best_f, maximize=False)

    >>> # With optimization
    >>> def acq_fn(X):
    ...     return expected_improvement(model.posterior(X), best_f=0.8, maximize=True)
    >>> X_next, ei_val = optimize_acqf(acq_fn, bounds, q=1)

    Notes
    -----
    For psychophysics:
    - `best_f` is typically the highest accuracy observed
    - Use `maximize=True` to find points that maximize accuracy
    - EI naturally balances exploration (high variance) and
      exploitation (high mean)

    Mathematical Details
    --------------------
    Let μ(x), σ(x) be the posterior mean and std at x.
    Let u = (μ(x) - best_f) / σ(x) (standardized improvement).

    Then:
        EI(x) = σ(x) * [u * Φ(u) + φ(u)]

    where Φ is the standard normal CDF and φ is the PDF.

    When σ(x) = 0 (no uncertainty), EI(x) = max(0, μ(x) - best_f).
    """
    mean = posterior.mean
    std = jnp.sqrt(posterior.variance)

    if not maximize:
        # For minimization, flip the improvement
        u = (best_f - mean) / (std + 1e-9)  # Numerical stability
    else:
        u = (mean - best_f) / (std + 1e-9)

    # EI formula: σ * [u * Φ(u) + φ(u)]
    normal_cdf = stats.norm.cdf(u)
    normal_pdf = stats.norm.pdf(u)

    ei = std * (u * normal_cdf + normal_pdf)

    # Handle numerical issues: EI should be non-negative
    ei = jnp.maximum(ei, 0.0)

    return ei


def log_expected_improvement(
    posterior: PredictivePosterior,
    best_f: float,
    maximize: bool = True,
) -> jnp.ndarray:
    """
    Log expected improvement for numerical stability.

    Useful when EI values span many orders of magnitude.

    Parameters
    ----------
    posterior : PredictivePosterior
        Predictive posterior
    best_f : float
        Best observed value
    maximize : bool, default=True
        If True, maximize. If False, minimize.

    Returns
    -------
    jnp.ndarray, shape (n_candidates,)
        log(EI) values

    Examples
    --------
    >>> # When EI values are very small, log(EI) is more stable
    >>> log_ei = log_expected_improvement(posterior, best_f)
    >>> X_next = X_candidates[jnp.argmax(log_ei)]

    Notes
    -----
    Since we only care about ranking, log(EI) preserves the order:
        argmax EI(x) = argmax log(EI(x))

    This is numerically more stable when EI values are near zero.
    """
    ei = expected_improvement(posterior, best_f, maximize=maximize)

    # Add small constant for log stability
    log_ei = jnp.log(ei + 1e-25)

    return log_ei
