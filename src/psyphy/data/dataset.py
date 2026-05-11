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
    stimuli : (N, K, d)
    responses : (N, R)
    context : optional (N, C)

    Dimension key
    -------------
    N : number of trials (batch dimension)
    K : number of stimuli per trial (e.g. K=2 for a two-alternative task;
        K=2 for the oddity task, where the reference is presented twice but
        only the unique mean is stored — the duplication is encoded in the
        task likelihood, not here)
    d : dimensionality of each stimulus coordinate
    R : number of response channels (R=1 for binary; R=2 for e.g. (choice, RT))
    C : number of context channels (observer-state covariates that condition the
        likelihood but are not part of the stimulus space, e.g. fatigue level)

    Notes
    -----
    - You can also think of this as a more generic ML-style dataset
      ``X`` with shape (N, K, d) plus ``y`` with shape (N, R).
    - This is intended to be JAX-friendly (PyTree of arrays) so likelihood and
      inference code can be JIT-compiled without touching Python containers.
    - Context is optional. No current inbuilt uses.
    """

    stimuli: jnp.ndarray
    responses: jnp.ndarray
    context: jnp.ndarray | None = None
    stimulus_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # Callers that construct TrialData directly (bypassing ResponseData.add_trial)
        # may pass responses as a plain 1D array (N,). Expand to
        # (N, 1) here so strict shape validation below does not break those call sites.
        # object.__setattr__ is required because the dataclass is frozen; __post_init__
        # is the only window to mutate fields during construction.
        if self.responses.ndim == 1:
            object.__setattr__(self, "responses", self.responses[:, None])
        # Basic shape validation (keep lightweight; raise early for common mistakes).
        if self.stimuli.ndim != 3:
            raise ValueError(
                f"stimuli must be 3D (N, K, d), got shape {self.stimuli.shape}"
            )
        if self.responses.ndim != 2:
            raise ValueError(
                f"responses must be 2D (N, R), got shape {self.responses.shape}"
            )
        if self.stimuli.shape[0] != self.responses.shape[0]:
            raise ValueError(
                "stimuli and responses must have same first dimension; "
                f"got {self.stimuli.shape[0]} vs {self.responses.shape[0]}."
            )
        if self.context is not None and self.context.shape[0] != self.stimuli.shape[0]:
            raise ValueError(
                "if context is provided, it must share the same first dimension;"
                f"got {self.context.shape[0]} vs {self.stimuli.shape[0]}."
            )

    def stimulus(self, name: str) -> jnp.ndarray:
        """Return stimuli[:, k, :] for the slot named `name`.

        Parameters
        ----------
        name : str
            Must match one of the entries in ``stimulus_names``.

        Returns
        -------
        jnp.ndarray, shape (N, d)
            Stimulus coordinates for all trials at the named slot.
        """
        if not self.stimulus_names:
            raise ValueError(
                "stimulus_names is empty — set it at construction time to use "
                "named access, e.g. stimulus_names=('ref', 'comp')."
            )
        if name not in self.stimulus_names:
            raise ValueError(
                f"unknown stimulus name '{name}'. "
                f"Available names: {self.stimulus_names}."
            )
        idx = self.stimulus_names.index(name)  # resolved in Python, not JAX-traced
        return self.stimuli[:, idx, :]

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
        self.stimuli: list[np.array] = []
        self.responses: list[np.array] = []
        self.stim_shape: tuple | None = None  # set on first add_trial call
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
        input_arr = np.atleast_2d(np.asarray(input))  # (K, d) — 1D input treated as K=1
        resp_arr = np.atleast_1d(np.asarray(resp))  # (R,)   — scalar treated as R=1
        if self.stimuli:
            if self.stim_shape != input_arr.shape:
                raise ValueError(
                    f"stimuli must have consistent shape (K, d). Expected {self.stim_shape}, but received {input_arr.shape}"
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
            if self.contexts or self.stimuli == []:
                self.contexts.append(np.asarray(context))
            else:
                raise ValueError(
                    "Context cannot be accepted if it was excluded from prior trials."
                    f"This ResponseData instance expected no context, but received {context}"
                )
        self.stimuli.append(input_arr)
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
        """Return stimuli and responses as NumPy arrays.
        Will NOT include contexts by default. Output always fixed length of 2.
        """
        return (
            np.asarray(self.stimuli),  # shape = (N, K, d)
            np.asarray(self.responses),
        )

    def to_trial_data(self) -> TrialData:
        """Convert this log into the canonical JAX batch (:class:`TrialData`)."""
        stimuli, responses = self.to_numpy()
        if self.contexts:
            context = np.asarray(self.contexts)
            return TrialData(
                stimuli=jnp.asarray(stimuli),
                responses=jnp.asarray(responses),
                context=jnp.asarray(context),
            )
        else:
            return TrialData(
                stimuli=jnp.asarray(stimuli), responses=jnp.asarray(responses)
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
        return [i + (r,) for i, r in zip(self.stimuli, self.responses)]

    def __len__(self) -> int:
        """Return number of trials."""
        return len(self.stimuli)

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
            dims = X.shape
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

        # X is (n_trials, K, d) — split into per-trial tuples of K stimulus rows
        stimuli = []
        for plane in X:
            stimuli.append(tuple(plane))

        if c is not None:
            for stim, response, context in zip(stimuli, y, c):
                data.add_trial(stim, response, context)
        else:
            for stim, response in zip(stimuli, y):
                data.add_trial(stim, response)

        return data

    @classmethod
    def from_trial_data(cls, data: TrialData) -> ResponseData:
        """Build a ResponseData log from a :class:`TrialData` batch."""
        stimuli = np.asarray(data.stimuli)
        ys = np.asarray(data.responses)
        out = cls()
        if data.context is not None:
            cs = np.asarray(data.context)
            for s, y, c in zip(stimuli, ys, cs):
                out.add_trial(s, y, c)
        else:
            for s, y in zip(stimuli, ys):
                out.add_trial(s, y)
        return out

    def merge(self, other: ResponseData) -> None:
        """
        Merge another dataset into this one (in-place).

        Parameters
        ----------
        other : ResponseData
            Dataset to merge
        """
        no_empty = self.stimuli and other.stimuli

        if no_empty and self.stimuli[0].shape != other.stimuli[0].shape:
            raise ValueError(
                "Cannot merge ResponseData instances with inconsistent input shapes."
                f"Received input shapes of {self.stimuli[0].shape} and {other.stimuli[0].shape}"
            )
        if no_empty and self.responses[0].shape != other.responses[0].shape:
            raise ValueError(
                "Cannot merge ResponseData instances with inconsistent response shapes."
                f"Received response shapes of {self.responses[0].shape} and {other.responses[0].shape}"
            )

        self.stimuli.extend(other.stimuli)
        self.responses.extend(other.responses)
        both_contexts = self.contexts and other.contexts

        if self.contexts == [] and other.contexts == []:
            pass
        elif both_contexts and self.contexts[0].shape == other.contexts[0].shape:
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
        new_data.stimuli = self.stimuli[-n:]
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
        new_data.stimuli = list(self.stimuli)
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
