"""
posterior
=========

Posterior representations and diagnostics.

This subpackage provides:
- Posterior : wrapper around model parameters, samples, and prediction APIs.
- diagnostics : tools for checking posterior quality (ESS, R-hat, etc.).

MVP implementation
------------------
- Posterior: stores MAP parameters and delegates predictions to WPPM.
- Diagnostics: stubs and simple summaries.

Future extensions
-----------------
- Posterior: store MCMC samples, support predictive intervals.
- Diagnostics: effective sample size, R-hat, posterior predictive checks.
"""

from .diagnostics import effective_sample_size, rhat
from .posterior import Posterior

__all__ = [
	"Posterior",
	# diagnostics
	"effective_sample_size",
	"rhat",
]
    