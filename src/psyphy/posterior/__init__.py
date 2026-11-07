"""
posterior
=========

Posterior representations.

This subpackage provides:
- ParameterSource: protocol for anything we can draw parameters from, including
  a Prior
- ParameterPosterior: a source that also carries p(θ | data) and its model
- MAPPosterior: point estimate at θ_MAP (a delta distribution)

Two-tier design
---------------
- ParameterPosterior: represents p(θ | data)
- PredictivePosterior: represents p(f(X*) | data)

Future extensions
-----------------
- LaplacePosterior: Gaussian approximation N(θ_MAP, Σ)
"""

from .mcmc_posterior import MCMCPosterior
from .parameter_posterior import ParameterPosterior, ParameterSource
from .posterior import MAPPosterior
from .predictive_posterior import (
    PredictivePosterior,
    ThresholdConfig,
    WPPMPredictivePosterior,
)

__all__ = [
    # Core protocols
    "ParameterSource",
    "ParameterPosterior",
    "PredictivePosterior",
    # Parameter posterior implementations
    "MAPPosterior",
    "MCMCPosterior",
    # Predictive posterior implementations
    "WPPMPredictivePosterior",
    "ThresholdConfig",
]
