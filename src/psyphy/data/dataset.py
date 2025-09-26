"""
dataset.py
-----------

Core data containers for psyphy.

defines:
- ResponseData: container for psychophysical trial data
- TrialBatch: container for a proposed batch of trials

Notes
-----
- Data is stored in standard NumPy (mutable!) arrays or Python lists.
- Use numpy for I/O and analysis.
- Convert to jax.numpy (jnp) (immutable!) arrays only when passing into WPPM
  or inference engines that require JAX/Optax.
"""

from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np


class ResponseData:
    """
    Container for psychophysical trial data.

    Attributes
    ----------
    refs : List[Any]
        List of reference stimuli.
    probes : List[Any]
        List of probe stimuli.
    responses : List[int]
        List of subject responses (e.g., 0/1 or categorical).
    """

    def __init__(self) -> None:
        self.refs: List[Any] = []
        self.probes: List[Any] = []
        self.responses: List[int] = []

    def add_trial(self, ref: Any, probe: Any, resp: int) -> None:
        """
        append a single trial.

        Parameters
        ----------
        ref : Any
            Reference stimulus (numpy array, list, etc.)
        probe : Any
            Probe stimulus
        resp : int
            Subject response (binary or categorical)
        """
        self.refs.append(ref)
        self.probes.append(probe)
        self.responses.append(resp)

    def add_batch(self, responses: List[int], trial_batch: TrialBatch) -> None:
        """
        Append responses for a batch of trials.

        Parameters
        ----------
        responses : List[int]
            Responses corresponding to each (ref, probe) in the trial batch.
        trial_batch : TrialBatch
            The batch of proposed trials.
        """
        for (ref, probe), resp in zip(trial_batch.stimuli, responses):
            self.add_trial(ref, probe, resp)

    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return refs, probes, responses as numpy arrays.

        Returns
        -------
        refs : np.ndarray
        probes : np.ndarray
        responses : np.ndarray
        """
        return (
            np.array(self.refs),
            np.array(self.probes),
            np.array(self.responses),
        )


class TrialBatch:
    """
    Container for a proposed batch of trials

    Attributes
    ----------
    stimuli : List[Tuple[Any, Any]]
        Each trial is a (reference, probe) tuple.
    """

    def __init__(self, stimuli: List[Tuple[Any, Any]]) -> None:
        self.stimuli = list(stimuli)

    @classmethod
    def from_stimuli(cls, pairs: List[Tuple[Any, Any]]) -> TrialBatch:
        """
        Construct a TrialBatch from a list of stimuli (ref, probe) pairs.
        """
        return cls(pairs)
