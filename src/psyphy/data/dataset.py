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
    responses : (N, r)
    context : optional (N, c)

    Notes
    -----
    - You can also think of this as a more generic ML-style dataset
      ``X`` with shape (N, s, d) plus ``y`` with shape (N, r). N corresponds to
      number of trials, s to number of stimuli, d to the feature dimensions
      of a given stimulus, and r to dimensions of a given response.
    - This is intended to be JAX-friendly (PyTree of arrays) so likelihood and
      inference code can be JIT-compiled without touching Python containers.
    - Context is optional with N trials and c context dimensions. No current
        inbuilt uses.
    """

    inputs: jnp.ndarray
    responses: jnp.ndarray
    context: jnp.ndarray | None = None

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
                f"got {self.inputs.shape[0]} vs {self.responses.shape[0]}."
            )
        if self.context is not None and self.context.shape[0] != self.inputs.shape[0]:
            raise ValueError(
                "if context is provided, it must share the same first dimension;"
                f"got {self.context.shape[0]} vs {self.inputs.shape[0]}."
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
        self.inputs: list[np.array] = []
        self.responses: list[np.array] = []
        if self.inputs:
            self.stim_shape = self.inputs[0].shape
        self.contexts: list[np.array] = []

    def add_trial(self, input: tuple[Any, ...], resp: Any, context: Any = None) -> None:
        """
        append a single trial.

        Parameters
        ----------
        input : tuple(Any, ...)
            Group of presented stimuli each represented in any format (numpy array,
            list, etc.)
            Input must contain appropriate number of stimuli of appropriate dimension.
        resp : Any
            Subject response
        """
        input_arr = np.asarray(input)
        resp_arr = np.asarray(resp)
        if self.inputs:
            if self.stim_shape != input_arr.shape:
                raise ValueError(
                    f"inputs must contain a consistent number of stimuli and number of \
                    stimulus dimensions. Expected {self.stim_shape}, but received {input_arr.shape}"
                )
        else:
            self.stim_shape = input_arr.shape

        if context is None:
            if self.contexts:
                raise ValueError(
                    "Context cannot be omitted if it was included in previous trials."
                    "This ResponseData instance expected context but received none."
                )
        else:
            if self.contexts or self.inputs == []:
                self.contexts.append(np.asarray(context))
            else:
                raise ValueError(
                    "Context cannot be accepted if it was excluded from prior trials."
                    f"This ResponseData instance expected no context, but received {context}"
                )
        self.inputs.append(input_arr)
        self.responses.append(resp_arr)

    def add_batch(
        self,
        responses: list[Any],
        trial_batch: TrialBatch,
        contexts: list[Any] | None = None,
    ) -> None:
        """
        Append responses for a batch of trials.

        Parameters
        ----------
        responses : List[Any]
            Responses corresponding to each stimulus group in the trial batch.
        trial_batch : TrialBatch
            The batch of proposed trials.
        """
        if contexts is None:
            for input, resp in zip(trial_batch.stimuli, responses):
                self.add_trial(input, resp)
        else:
            for input, resp, context in zip(trial_batch.stimuli, responses, contexts):
                self.add_trial(input, resp, context)

    def to_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        """Return inputs and responses as NumPy arrays.
        Will NOT include contexts by default. Output always fixed length of 2.
        """
        return (
            np.asarray(self.inputs),  # shape = (N, s, d)
            np.asarray(self.responses),
        )

    def to_trial_data(self) -> TrialData:
        """Convert this log into the canonical JAX batch (:class:`TrialData`)."""
        inputs, responses = self.to_numpy()
        if self.contexts:
            context = np.asarray(self.contexts)
            return TrialData(
                inputs=jnp.asarray(inputs),
                responses=jnp.asarray(responses),
                context=jnp.asarray(context),
            )
        else:
            return TrialData(
                inputs=jnp.asarray(inputs), responses=jnp.asarray(responses)
            )

    @property
    def trials(self) -> list[tuple[Any, ...]]:
        """
        Return list of (stim1, stim2, ... , response) tuples.
        Does NOT include context information.

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
        c: jnp.ndarray | np.ndarray | None = None,
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
        c : optional array, shape (n_trials, context_dim)
            Context

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
                "X must be shape (n_trials, n_stimuli, input_dim) or \
                (n_trials, input_dim)."
            )
        if y.shape[0] != X.shape[0]:
            raise ValueError("X and y must contain the same n_trials.")
        if c is not None and c.shape[0] != X.shape[0]:
            raise ValueError("c must contain same n_trials as X.")

        # X is (n_trials, n_stim, input_dim)
        inputs = []
        for plane in X:
            inputs.append(tuple(plane))
        # --> X is(n_trials,) where each entry is tuple of stimuli

        if c is not None:
            for input, response, context in zip(inputs, y, c):
                data.add_trial(input, response, context)
        else:
            for input, response in zip(inputs, y):
                data.add_trial(input, response)

        return data

    @classmethod
    def from_trial_data(cls, data: TrialData) -> ResponseData:
        """Build a ResponseData log from a :class:`TrialData` batch."""
        inputs = np.asarray(data.inputs)
        ys = np.asarray(data.responses)
        out = cls()
        if data.context is not None:
            cs = np.asarray(data.context)
            for i, y, c in zip(inputs, ys, cs):
                out.add_trial(i, y, c)
        else:
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
        no_empty = self.inputs and other.inputs

        if no_empty and self.inputs[0].shape != other.inputs[0].shape:
            raise ValueError(
                "Cannot merge ResponseData instances with inconsistent input shapes."
                f"Received input shapes of {self.inputs[0].shape} and {other.inputs[0].shape}"
            )
        if no_empty and self.responses[0].shape != other.responses[0].shape:
            raise ValueError(
                "Cannot merge ResponseData instances with inconsistent response shapes."
                f"Received response shapes of {self.responses[0].shape} and {other.responses[0].shape}"
            )

        self.inputs.extend(other.inputs)
        self.responses.extend(other.responses)
        both_contexts = self.contexts and other.contexts

        if self.contexts == [] and other.contexts == []:
            pass
        elif both_contexts and self.contexts[0].shape[1] == other.contexts[0].shape[1]:
            self.contexts.extend(other.contexts)
        else:
            raise ValueError(
                "Cannot merge ResponseData instances with inconsistent context."
            )

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
        if self.contexts is not None:
            new_data.contexts = self.contexts[-n:]
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
        if self.contexts is not None:
            new_data.contexts = list(self.contexts)
        return new_data


class TrialBatch:
    """
    Container for a proposed batch of trials.
    Does NOT include context or responses.

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
