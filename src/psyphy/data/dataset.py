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

from typing import Any

import jax.numpy as jnp
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
        self.refs: list[Any] = []
        self.probes: list[Any] = []
        self.responses: list[int] = []

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

    def add_batch(self, responses: list[int], trial_batch: TrialBatch) -> None:
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

    def to_numpy(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    @property
    def trials(self) -> list[tuple[Any, Any, int]]:
        """
        Return list of (ref, probe, response) tuples.

        Returns
        -------
        list[tuple]
            Each element is (ref, probe, resp)
        """
        return list(zip(self.refs, self.probes, self.responses))

    def __len__(self) -> int:
        """Return number of trials."""
        return len(self.refs)

    @classmethod
    def from_arrays(
        cls,
        X: jnp.ndarray | np.ndarray,
        y: jnp.ndarray | np.ndarray,
        *,
        probes: jnp.ndarray | np.ndarray | None = None,
    ) -> ResponseData:
        """
        Construct ResponseData from arrays.

        Parameters
        ----------
        X : array, shape (n_trials, 2, input_dim) or (n_trials, input_dim)
            Stimuli. If 3D, second axis is [reference, probe].
            If 2D, probes must be provided separately.
        y : array, shape (n_trials,)
            Responses
        probes : array, shape (n_trials, input_dim), optional
            Probe stimuli. Only needed if X is 2D.

        Returns
        -------
        ResponseData
            Data container

        Examples
        --------
        >>> # From paired stimuli
        >>> X = jnp.array([[[0, 0], [1, 0]], [[1, 1], [2, 1]]])
        >>> y = jnp.array([1, 0])
        >>> data = ResponseData.from_arrays(X, y)

        >>> # From separate refs and probes
        >>> refs = jnp.array([[0, 0], [1, 1]])
        >>> probes = jnp.array([[1, 0], [2, 1]])
        >>> data = ResponseData.from_arrays(refs, y, probes=probes)
        """
        data = cls()

        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim == 3:
            # X is (n_trials, 2, input_dim)
            refs = X[:, 0, :]
            probes_arr = X[:, 1, :]
        elif X.ndim == 2 and probes is not None:
            refs = X
            probes_arr = np.asarray(probes)
        else:
            raise ValueError(
                "X must be shape (n_trials, 2, input_dim) or "
                "(n_trials, input_dim) with probes argument"
            )

        for ref, probe, response in zip(refs, probes_arr, y):
            data.add_trial(ref, probe, int(response))

        return data

    def merge(self, other: ResponseData) -> None:
        """
        Merge another dataset into this one (in-place).

        Parameters
        ----------
        other : ResponseData
            Dataset to merge
        """
        self.refs.extend(other.refs)
        self.probes.extend(other.probes)
        self.responses.extend(other.responses)

    def tail(self, n: int) -> ResponseData:
        """
        Return last n trials as a new ResponseData.

        Parameters
        ----------
        n : int
            Number of trials to keep

        Returns
        -------
        ResponseData
            New dataset with last n trials
        """
        new_data = ResponseData()
        new_data.refs = self.refs[-n:]
        new_data.probes = self.probes[-n:]
        new_data.responses = self.responses[-n:]
        return new_data

    def copy(self) -> ResponseData:
        """
        Create a deep copy of this dataset.

        Returns
        -------
        ResponseData
            New dataset with copied data
        """
        new_data = ResponseData()
        new_data.refs = list(self.refs)
        new_data.probes = list(self.probes)
        new_data.responses = list(self.responses)
        return new_data


class TrialBatch:
    """
    Container for a proposed batch of trials

    Attributes
    ----------
    stimuli : List[Tuple[Any, Any]]
        Each trial is a (reference, probe) tuple.
    """

    def __init__(self, stimuli: list[tuple[Any, Any]]) -> None:
        self.stimuli = list(stimuli)

    @classmethod
    def from_stimuli(cls, pairs: list[tuple[Any, Any]]) -> TrialBatch:
        """
        Construct a TrialBatch from a list of stimuli (ref, probe) pairs.
        """
        return cls(pairs)
