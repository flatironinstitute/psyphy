"""
grid.py
-------

Grid-based placement strategy.


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
    - grid = your set of allowable trials; this class simply walks through that set.
    - Not yet tested. Pending the same ``TrialBatch`` redesign as ``SobolPlacement``:
      each trial's stimuli should be a single ``np.ndarray`` of shape ``(K, d)``
      rather than a two-element tuple, to align with the generalised ``TrialData``
      layout introduced in the data-object refactor.
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
