"""
sobol.py
--------

Sobol quasi-random placement.

MVP:
- Uses a Sobol engine to generate low-discrepancy points.
- Ignores the posterior (pure exploration).

Full WPPM mode:
- Could combine Sobol exploration (early) with posterior-aware exploitation (later).
"""

import numpy as np
from scipy.stats.qmc import Sobol

from psyphy.data.dataset import TrialBatch
from psyphy.trial_placement.base import TrialPlacement


class SobolPlacement(TrialPlacement):
    """
    Sobol quasi-random placement.

    Parameters
    ----------
    dim : int
        Dimensionality of stimulus space.
    bounds : list of (low, high)
        Bounds per dimension.
    seed : int, optional
        RNG seed.
    """

    def __init__(self, dim: int, bounds, seed: int = 0):
        self.engine = Sobol(d=dim, scramble=True, seed=seed)
        self.bounds = bounds

    def propose(self, posterior, batch_size: int) -> TrialBatch:
        """
        Propose Sobol points (ignores posterior).

        Parameters
        ----------
        posterior : Posterior
            Ignored in MVP.
        batch_size : int
            Number of trials to return.

        Returns
        -------
        TrialBatch
            Candidate trials from Sobol sequence.

        Notes
        -----
        MVP:
            Pure exploration of space.
        Full WPPM mode:
            Use Sobol as initialization, then switch to InfoGain.
        """
        raw = self.engine.random(batch_size)
        scaled = [low + (high - low) * raw[:, i] for i, (low, high) in enumerate(self.bounds)]
        stimuli = list(zip(*scaled))
        return TrialBatch.from_stimuli(stimuli)
        