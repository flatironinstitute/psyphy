"""
base.py
-------

Abstract base class for inference engines.

Every psyphy inference engine (MAPOptimizer, NUTSSampler, LangevinSampler,
LaplaceApproximation) subclasses this base and implements
`fit(model, data, init_params=None, seed=None)`, which returns a posterior.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class InferenceEngine(ABC):
    """
    Abstract interface for psyphy inference engines.

    The constructor of an engine holds the settings of the method, such as the
    number of steps. Everything that changes from one run to the next goes to
    `fit`: the data, the starting point and the seed.

    Methods
    -------
    fit(model, data, init_params=None, seed=None) -> Posterior
        Fit model parameters to data and return a posterior.
    """

    @abstractmethod
    def fit(
        self,
        model: Any,
        data: Any,
        init_params: dict | None = None,
        seed: int | None = None,
    ) -> Any:
        """
        Fit model parameters to data.

        Parameters
        ----------
        model : WPPM
            Psychophysical model to fit.
        data : TrialData
            Observed trials.
        init_params : dict | None, optional
            Starting parameter PyTree, for example `map_posterior.params` to
            warm-start a sampler from a MAP fit. If None, the engine draws one
            from the model's prior with the seed.
        seed : int | None, optional
            Seed for every random step of this fit. None means 0, so two fits
            with no seed give the same result.

        Returns
        -------
        Posterior
            Posterior object wrapping fitted params and model reference.
        """
        ...
