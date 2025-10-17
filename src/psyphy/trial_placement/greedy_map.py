"""
greedy_map.py
-------------

Greedy placement using MAP estimate.

MVP:
- Returns first N candidates (stub).
- Ignores posterior discriminability.

Full WPPM mode:
- Score each candidate using posterior.MAP_params().
- Rank candidates by informativeness (e.g., discriminability).
"""

from psyphy.data.dataset import TrialBatch
from psyphy.trial_placement.base import TrialPlacement


class GreedyMAPPlacement(TrialPlacement):
    """
    Greedy adaptive placement (MAP-based).

    Parameters
    ----------
    candidate_pool : list of (ref, probe)
        Candidate stimuli.
    """

    def __init__(self, candidate_pool):
        self.pool = candidate_pool

    def _score_candidate(self, posterior, candidate) -> float:
        """
        Score a single candidate stimulus.

        Parameters
        ----------
        posterior : Posterior
            Posterior object containing MAP parameters.
        candidate : (ref, probe)
            Candidate trial.

        Returns
        -------
        float
            Candidate score (higher = more informative).

        Notes
        -----
        MVP:
            Returns a constant score (stub).
        Full WPPM mode:
            - Get MAP params: params = posterior.MAP_params()
            - Compute discriminability: d = model.discriminability(params, (ref, probe))
            - Score = d, larger discriminability = more informative.
        """
        # TODO: implement MAP-based discriminability scoring
        return 0.0  # stub

    def propose(self, posterior, batch_size: int) -> TrialBatch:
        """
        Select trials based on MAP discriminability.

        Parameters
        ----------
        posterior : Posterior
            Posterior object. Provides MAP params.
        batch_size : int
            Number of trials to propose.

        Returns
        -------
        TrialBatch
            Selected trials.

        Notes
        -----
        MVP:
            Returns first N candidates.
        Full WPPM mode:
            - Score all candidates with _score_candidate().
            - Rank candidates by score.
            - Select top-N.
        """
        # MVP stub
        selected = self.pool[:batch_size]

        # TODO: Uncomment when scoring is implemented
        # scores = [self._score_candidate(posterior, cand) for cand in self.pool]
        # idx = jnp.argsort(jnp.array(scores))[::-1]  # sort descending
        # selected = [self.pool[i] for i in idx[:batch_size]]

        return TrialBatch.from_stimuli(selected)
