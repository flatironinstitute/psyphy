"""
posterior
=========

Posterior representations and diagnostics.

This subpackage provides:
- ParameterPosterior: protocol for posteriors over model parameters p(θ | data)
- MAPPosterior: delta distribution at θ_MAP (point estimate)
- Posterior: backwards compatibility alias (deprecated, use MAPPosterior)
- diagnostics: tools for checking posterior quality (ESS, R-hat, etc.)

Two-tier design
---------------
- ParameterPosterior: represents p(θ | data) for research/diagnostics
- PredictivePosterior: represents p(f(X*) | data) for acquisition functions
  (to be added in next phase)

Future extensions
-----------------
- LaplacePosterior: Gaussian approximation N(θ_MAP, Σ)
- LangevinPosterior: MCMC samples
- PredictivePosterior: predictions at test points
"""

from .diagnostics import effective_sample_size, rhat
from .parameter_posterior import ParameterPosterior
from .posterior import MAPPosterior, Posterior
from .predictive_posterior import PredictivePosterior, WPPMPredictivePosterior

__all__ = [
    # Core protocols
    "ParameterPosterior",
    "PredictivePosterior",
    # Parameter posterior implementations
    "MAPPosterior",
    # Predictive posterior implementations
    "WPPMPredictivePosterior",
    # Backwards compatibility (deprecated)
    "Posterior",
    # Diagnostics
    "effective_sample_size",
    "rhat",
]
