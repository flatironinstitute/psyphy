"""
grid.py
-------

Grid-based placement strategy.

MVP:
- Iterates through a fixed list of grid points.
- Ignores the posterior (non-adaptive).

Full WPPM mode:
- Could refine the grid adaptively around regions of high posterior uncertainty.
"""

from psyphy.data.dataset import TrialBatch


class GridPlacement:
    """
    Fixed grid placement.

    Parameters
    ----------
    grid_points : list of (ref, probe)
        Predefined set of trial stimuli.

    Notes
    -----
    - grid = your set of allowable trials.
    - this class simply walks through that set.
    """

    def __init__(self, grid_points):
        self.grid_points = list(grid_points)
        self._index = 0

    def propose(self, posterior, batch_size: int) -> TrialBatch:
        """
        Return the next batch of trials from the grid.

        Parameters
        ----------
        posterior : Posterior
            Ignored in MVP (grid is non-adaptive).
        batch_size : int
            Number of trials to return.

        Returns
        -------
        TrialBatch
            Fixed batch of (ref, probe).
        """
        start, end = self._index, self._index + batch_size
        batch = self.grid_points[start:end]
        self._index = end
        return TrialBatch.from_stimuli(batch)
