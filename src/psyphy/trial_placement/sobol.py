"""
sobol.py
--------

Sobol quasi-random placement.

What it does:
- Uses a Sobol engine to generate low-discrepancy points.
- Ignores the posterior (pure exploration).

Levaraging WPPM's posterior:
- Could combine Sobol exploration (early) with posterior-aware exploitation (later).
"""

from scipy.stats.qmc import Sobol

from psyphy.data.dataset import TrialBatch


class SobolPlacement:
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

    Notes
    -----
    ``TrialBatch`` stores stimuli as ``list[tuple[Any, Any]]``, and the reference
    vector is fixed at zero.
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
            Ignored.
        batch_size : int
            Number of trials to return.

        Returns
        -------
        TrialBatch
            Candidate trials from Sobol sequence.

        """
        raw = self.engine.random(batch_size)
        scaled = [
            low + (high - low) * raw[:, i] for i, (low, high) in enumerate(self.bounds)
        ]
        # Convert column-wise scaled arrays into list of probe vectors
        comparisons = [tuple(vals) for vals in zip(*scaled)]
        # reference is always a zero vector of matching dimension
        dim = len(self.bounds)
        zero_ref = 0.0 if dim == 1 else tuple(0.0 for _ in range(dim))
        trials = [(zero_ref, p) for p in comparisons]
        return TrialBatch.from_stimuli(trials)
