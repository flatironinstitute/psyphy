"""
langevin.py
-----------

Langevin samplers for posterior inference.

Will lkely implement:
- Overdamped (unadjusted) Langevin Algorithm (ULA) (exists in BlackJax)
- Underdamped Langevin (with BAOAB splitting scheme?)

Used for posterior-aware trial placement (InfoGain).


Currently a stub: `fit` returns an initial posterior and runs no Langevin dynamics.
"""

from psyphy.inference.base import InferenceEngine
from psyphy.posterior.posterior import MAPPosterior


class LangevinSampler(InferenceEngine):
    """
    Langevin sampler (stub).

    Parameters
    ----------
    steps : int, default=1000
        Number of Langevin steps.
    step_size : float, default=1e-3
        Integration step size.
    temperature : float, default=1.0
        Noise scale (temperature).
    """

    def __init__(
        self, steps: int = 1000, step_size: float = 1e-3, temperature: float = 1.0
    ):
        self.steps = steps
        self.step_size = step_size
        self.temperature = temperature

    def fit(self, model, data) -> MAPPosterior:
        """
        Fit model parameters with Langevin dynamics (stub).

        Parameters
        ----------
        model : WPPM
            Model instance.
        data : ResponseData
            Observed trials.

        """

        raise NotImplementedError("Langevin Sampler not implemented yet.")


class NumpyroSampler(InferenceEngine):
    """
    Numpyro-based sampler (stub).

    Parameters
    ----------
    steps : int, default=1000
        Number of sampling steps.
    """

    def __init__(self, steps: int = 1000):
        self.steps = steps

    def fit(self, model, data) -> MAPPosterior:
        """
        Fit model parameters using Numpyro (stub).

        Parameters
        ----------
        model : WPPM
            Model instance.
        data : ResponseData
            Observed trials.

        Returns
        -------
        MAPPosterior
            Posterior distribution containing sample history.
        """
        raise NotImplementedError("NumpyroSampler.fit not implemented yet.")
