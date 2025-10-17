"""
info_gain.py
------------

Information-gain placement (e.g., entropy, expected Absolute Volume Change (EAVC)).

MVP:
- Returns first N candidates (stub)

Full WPPM mode:
- Requires posterior.sample() (Laplace/MCMC).
- For each candidate:
    * Draw posterior samples.
    * Compute predictive distribution of responses.
    * Compute expected entropy or EAVC.
    * Rank candidates by information gain.
"""

from psyphy.data.dataset import TrialBatch
from psyphy.trial_placement.base import TrialPlacement


class InfoGainPlacement(TrialPlacement):
    """
    Information-gain adaptive placement.

    Parameters
    ----------
    candidate_pool : list of (ref, probe)
        Candidate stimuli.
    """

    def __init__(self, candidate_pool):
        self.pool = candidate_pool

    def _score_candidate(self, posterior, candidate) -> float:
        """
        Score a single candidate stimulus by expected information gain.

        Parameters
        ----------
        posterior : Posterior
            Posterior object. Must support .sample().
        candidate : (ref, probe)
            Candidate trial.

        Returns
        -------
        float
            Expected information gain for candidate.

        Notes
        -----
        MVP:
            Returns a constant score (stub).
        Full WPPM mode:
            - Draw parameter samples: samples = posterior.sample(n)
            - For each sample, compute p(correct | params, candidate)
            - Approximate predictive distribution of responses,
            - Compute predictive entropy:
                H[p] = - p log p - (1-p) log (1-p)
            - InfoGain = expected reduction in entropy
        Extensions:
            - look up other acquisition functions
              (as a starting point, APsych's acquisition module might be helpful [1])


        References
        ----------
            [1] https://aepsych.org/api/acquisition.html
        """
        # TODO: implement information gain scoring with posterior.sample()
        return 0.0  # stub

    def propose(self, posterior, batch_size: int) -> TrialBatch:
        """
        Propose trials that maximize information gain.

        Parameters
        ----------
        posterior : Posterior
            Posterior object. Must support sample().
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
        # idx = jnp.argsort(jnp.array(scores))[::-1]
        # selected = [self.pool[i] for i in idx[:batch_size]]

        return TrialBatch.from_stimuli(selected)
