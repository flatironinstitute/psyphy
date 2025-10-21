"""
staircase.py
------------

Classical staircase placement (1-up, 2-down).

MVP:
- Purely response-driven, 1D only.
- Ignores posterior.

Full WPPM mode:
- Extend to multi-D tasks, integrate  with WPPM-based discriminability thresholds.
"""

from psyphy.data.dataset import TrialBatch


class StaircasePlacement:
    """
    Staircase procedure.

    Parameters
    ----------
    start_level : float
        Starting stimulus intensity.
    step_size : float
        Step increment.
    rule : str, default="1up-2down"
        Adaptive rule.
    """

    def __init__(self, start_level: float, step_size: float, rule: str = "1up-2down"):
        self.current_level = start_level
        self.step_size = step_size
        self.rule = rule
        self.correct_counter = 0

    def propose(self, posterior, batch_size: int) -> TrialBatch:
        """
        Return next trial(s) based on staircase rule.

        Parameters
        ----------
        posterior : Posterior
            Ignored in MVP (not posterior-aware).
        batch_size : int
            Number of trials to propose.

        Returns
        -------
        TrialBatch
            Batch of trials with current staircase level.
        """
        trials = [(0.0, self.current_level)] * batch_size  # Stub: (ref=0, probe=level)
        return TrialBatch.from_stimuli(trials)

    def update(self, response: int):
        """
        Update staircase level given last response.

        Parameters
        ----------
        response : int
            1 = correct, 0 = incorrect.
        """
        if response == 1:
            self.correct_counter += 1
            if self.rule == "1up-2down" and self.correct_counter >= 2:
                self.current_level -= self.step_size
                self.correct_counter = 0
        else:
            self.current_level += self.step_size
            self.correct_counter = 0
