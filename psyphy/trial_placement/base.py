"""
base.py
-------

Abstract base class for trial placement strategies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class TrialPlacement(ABC):
    """
    Abstract interface for trial placement strategies.

    Methods
    -------
    propose(posterior, batch_size) -> TrialBatch
        Propose the next batch of trials.

    All trial placement strategies (grid, staircase, info gain) subclass this.

    """

    @abstractmethod
    def propose(self, posterior: Any, batch_size: int):
        """
        Propose the next batch of trials.

        Parameters
        ----------
        posterior : Posterior
            Posterior distribution (MAP, Laplace, or MCMC).
        batch_size : int
            Number of trials to propose.

        Returns
        -------
        TrialBatch
            Proposed batch of (reference, probe) stimuli.
        """
        return NotImplementedError()
