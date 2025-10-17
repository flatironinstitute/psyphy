"""
diagnostics.py
--------------

Posterior diagnostics.

Provides functions to check quality of posterior inference.

MVP implementation:
- Stubs for effective sample size and R-hat.


Full WPPM mode:
- Implement real diagnostics from posterior chains.
- Include posterior predictive checks.
"""

from __future__ import annotations

import jax.numpy as jnp


def effective_sample_size(samples: jnp.ndarray) -> float:
    """
    Estimate effective sample size (ESS) to calculate the number of independent
    samples that a correlated MCMC chain is equivalent to.

    Parameters
    ----------
    samples : jnp.ndarray
        Posterior samples, shape (n_samples, ...).

    Returns
    -------
    float
        Effective sample size (stub).

    Notes
    -----
    MVP:
        Returns the number of samples.
    Full WPPM mode:
        Compute ESS using autocorrelation structure.
    """
    return samples.shape[0]


def rhat(chains: jnp.ndarray) -> float:
    """
    Compute R-hat convergence diagnostic.

    Parameters
    ----------
    chains : jnp.ndarray
        Posterior samples across chains, shape (n_chains, n_samples, ...).

    Returns
    -------
    float
        R-hat statistic (stub).

    Notes
    -----
    MVP:
        Always returns 1.0.
    Full WPPM mode:
        Implement Gelman-Rubin diagnostic [1]

    References:
    ----------
        [1] https://bookdown.org/rdpeng/advstatcomp/monitoring-convergence.html

    """
    return 1.0
