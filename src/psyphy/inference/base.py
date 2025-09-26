"""
base.py
-------

Abstract base class for inference engines.

All inference engines must implement a `fit(model, data)` method
that returns a Posterior object.

All inference engines (MAPOptimizer, LangevinSampler, LaplaceApproximation)
subclass from this base.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class InferenceEngine(ABC):
    """
    Abstract interface for inference engines.

    Methods
    -------
    fit(model, data) -> Posterior
        Fit model parameters to data and return a Posterior object.
    """

    @abstractmethod
    def fit(self, model: Any, data: Any) -> Any:
        """
        Fit model parameters to data.

        Parameters
        ----------
        model : WPPM
            Psychophysical model to fit.
        data : ResponseData
            Observed trials.

        Returns
        -------
        Posterior
            Posterior object wrapping fitted params and model reference.
        """
        ...
