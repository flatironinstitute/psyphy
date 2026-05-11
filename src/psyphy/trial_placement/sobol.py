"""
sobol.py
--------

Sobol quasi-random placement.

MVP:
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
    Not yet tested. Pending two design changes tracked in the trial-placement
    follow-up issue:

    1. ``TrialBatch`` currently stores stimuli as ``list[tuple[Any, Any]]``
       (two-stimulus tuples). It should be updated to ``list[np.ndarray]`` each
       of shape ``(K, d)`` to align with ``TrialData.stimuli`` and allow
       ``ResponseData.add_batch`` to consume it without conversion.

    2. The zero reference vector is hardcoded here. It should be an explicit
       parameter so the caller controls which point in stimulus space acts as
       the reference, rather than always using the origin.
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

        """
        raw = self.engine.random(batch_size)
        scaled = [
            low + (high - low) * raw[:, i] for i, (low, high) in enumerate(self.bounds)
        ]
        # Convert column-wise scaled arrays into list of probe vectors
        comparisons = [tuple(vals) for vals in zip(*scaled)]
        # MVP: use a zero reference vector of matching dimension
        dim = len(self.bounds)
        zero_ref = 0.0 if dim == 1 else tuple(0.0 for _ in range(dim))
        trials = [(zero_ref, p) for p in comparisons]
        return TrialBatch.from_stimuli(trials)
