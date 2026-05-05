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

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True, slots=True)
class TrialData:
    """Batched trial data for compute.

    This is the canonical, compute-efficient representation of observed trials.

    Shapes
    ------
    inputs : (N, s, d)
    responses : (N, d)

    Notes
    -----
    - You can also think of this as a more generic ML-style dataset
      ``X`` with shape (N, s, d) plus ``y`` with shape (N, d). N corresponds to
      number of trials, s to number of stimuli, and d to the feature dimensions
      of a given stimulus or response.
    - This is intended to be JAX-friendly (PyTree of arrays) so likelihood and
      inference code can be JIT-compiled without touching Python containers.
    """

    inputs: jnp.ndarray
    responses: jnp.ndarray

    def __post_init__(self) -> None:
        # Basic shape validation (keep lightweight; raise early for common mistakes).
        if self.inputs.ndim > 3:
            raise ValueError(
                f"inputs must be < 3D (N, s, d), got shape {self.inputs.shape}"
            )
        if self.responses.ndim > 2:
            raise ValueError(
                f"responses must be < 2D (N, d), got shape {self.responses.shape}"
            )
        if self.inputs.shape[0] != self.responses.shape[0]:
            raise ValueError(
                "inputs and responses must have same first dimension; "
                f"got {self.inputs.shape[0]} vs {self.responses.shape[0]}"
            )

    def __len__(self) -> int:
        """Number of trials (N)."""
        return int(self.responses.shape[0])

    @property
    def num_trials(self) -> int:
        """Number of trials (N)."""
        return len(self)


class ResponseData:
    """Python-friendly incremental trial log.

    This container is convenient for adaptive trial placement and I/O (e.g., CSV),
    but it is not a compute-efficient representation for JAX.

    Use :class:`TrialData` for model fitting and likelihood evaluation.
    """

    def __init__(self) -> None:
        self.inputs: list[tuple[Any, ...]] = []
        self.responses: list[Any] = []
        if self.inputs:
            self.num_stim = len(self.inputs[0])

    def add_trial(self, input: tuple[Any, ...], resp: Any) -> None:
        """
        append a single trial.

        Parameters
        ----------
        input : tuple(Any, ...)
            Group of presented stimuli represented in any format (numpy array,
            list, etc.)
            Tuple must contain appropriate number of stimuli
        comparison : Any
            Probe stimulus
        resp : Any
            Subject response (binary or categorical)
        """
        if self.inputs and self.num_stim != len(input):
            raise ValueError(
                f"inputs must always contain the same number of stimuli. \
                Expected {self.num_stim}, but received {len(input)}"
            )
        else:
            self.num_stim = len(input)

        self.inputs.append(input)
        self.responses.append(resp)

    def add_batch(self, responses: list[Any], trial_batch: TrialBatch) -> None:
        """
        Append responses for a batch of trials.

        Parameters
        ----------
        responses : List[Any]
            Responses corresponding to each stimulus group in the trial batch.
        trial_batch : TrialBatch
            The batch of proposed trials.
        """
        for input, resp in zip(trial_batch.stimuli, responses):
            self.add_trial(input, resp)

    def to_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        """Return inputs and responses as NumPy arrays."""
        return (
            np.asarray(self.inputs),  # shape = (N, s, d)
            np.asarray(self.responses),
        )

    def to_trial_data(self) -> TrialData:
        """Convert this log into the canonical JAX batch (:class:`TrialData`)."""
        inputs, responses = self.to_numpy()
        return TrialData(
            inputs=jnp.asarray(inputs),
            responses=jnp.asarray(responses),
        )

    @property
    def trials(self) -> list[tuple[Any, ...]]:
        """
        Return list of (stim1, stim2, ... , response) tuples.

        Returns
        -------
        list[tuple]
            Each element is tuple representing all stimuli and the associated
            response for a given trial.
        """
        return [i + (r,) for i, r in zip(self.inputs, self.responses)]

    def __len__(self) -> int:
        """Return number of trials."""
        return len(self.inputs)

    @classmethod
    def from_arrays(
        cls,
        X: jnp.ndarray | np.ndarray,
        y: jnp.ndarray | np.ndarray,
    ) -> ResponseData:
        """
        Construct ResponseData from arrays.

        Parameters
        ----------
        X : array, shape (n_trials, n_stimuli, input_dim) or (n_trials, input_dim)
            Stimuli. If 3D, second axis is input stumili. For OddityTask, this is
            (ref, comparison)
        y : array, shape (n_trials, response_dim)
            Responses

        Returns
        -------
        ResponseData
            Data container

        OddityTask Example
        --------
        >>> # From paired stimuli
        >>> X = jnp.array([[[0, 0], [1, 0]], [[1, 1], [2, 1]]])
        >>> # X is formed from refs = [[0, 0], [1, 1]], comparisons = [[1, 0], [2, 1]]
        >>> y = jnp.array([1, 0])
        >>> data = ResponseData.from_arrays(X, y)
        """
        data = cls()

        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim == 2:
            # reshape to ensure appropriate conversion to stimuli groups
            dims = X.shape()
            new_dims = (dims[0], 1, dims[1])
            X = np.reshape(X, new_dims)
        elif X.ndim != 3:
            raise ValueError(
                "X must be shape (n_trials, n_stimuli, input_dim) or "
                "(n_trials, input_dim)."
            )
        if y.shape[0] != X.shape[0]:
            raise ValueError("X and y must contain the same n_trials")

        # X is (n_trials, n_stim, input_dim)
        inputs = []
        for plane in X:
            inputs.append(tuple(plane))
        # --> X is(n_trials,) where each entry is tuple of stimuli

        for input, response in zip(inputs, y):
            data.add_trial(input, response)

        return data

    @classmethod
    def from_trial_data(cls, data: TrialData) -> ResponseData:
        """Build a ResponseData log from a :class:`TrialData` batch."""
        inputs = np.asarray(data.inputs)
        ys = np.asarray(data.responses)
        out = cls()
        for i, y in zip(inputs, ys):
            out.add_trial(i, y)
        return out

    def merge(self, other: ResponseData) -> None:
        """
        Merge another dataset into this one (in-place).

        Parameters
        ----------
        other : ResponseData
            Dataset to merge
        """
        self.inputs.extend(other.inputs)
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
        new_data.inputs = self.inputs[-n:]
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
        new_data.inputs = list(self.inputs)
        new_data.responses = list(self.responses)
        return new_data


class TrialBatch:
    """
    Container for a proposed batch of trials

    Attributes
    ----------
    stimuli : List[Tuple[Any, ...]]
        Each trial is a tuple of all presented stimuli (stim1, stim2, ...).
        For OddityTask this is (reference, comparison)
    """

    def __init__(self, stimuli: list[tuple[Any, ...]]) -> None:
        self.stimuli = list(stimuli)

    @classmethod
    def from_stimuli(cls, groups: list[tuple[Any, ...]]) -> TrialBatch:
        """
        Construct a TrialBatch from a list of stimuli (stim1, stim2, ...) groups.
        """
        return cls(groups)
