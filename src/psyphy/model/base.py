"""
base.py
-------

Base class for psychophysical models with BoTorch-style API.

Provides:
- Model.fit(X, y) --> fit model to data
- Model.posterior(X) --> get predictive posterior
- Model.condition_on_observations(X, y) --> online learning
- OnlineConfig --> memory management strategies

Design
------
This façade delegates inference to specialized engines (MAP, Laplace, MCMC)
while maintaining a simple, composable API for users.

Inspired by BoTorch but adapted for psychophysics:
- Explicit parameter posteriors for research transparency
- Online learning with bounded memory (sliding window, reservoir sampling)
- Immutable updates (returns new instances)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import jax.numpy as jnp
import jax.random as jr

if TYPE_CHECKING:
    from psyphy.data import ResponseData
    from psyphy.inference.base import InferenceEngine
    from psyphy.posterior import ParameterPosterior, PredictivePosterior


@dataclass
class OnlineConfig:
    """
    Configuration for online learning and memory management.

    Attributes
    ----------
    strategy : {"full", "sliding_window", "reservoir", "none"}
        Data retention strategy:
        - "full": Keep all data (unbounded memory)
        - "sliding_window": Keep only last N trials (FIFO)
        - "reservoir": Reservoir sampling for uniform coverage
        - "none": No caching, refit from scratch each time

    window_size : int | None
        Maximum number of trials to retain (for sliding_window/reservoir).
        Required for sliding_window and reservoir strategies.

    refit_interval : int
        Refit model every N updates (1=always, 10=batch every 10 trials).
        Trades off accuracy vs. computational cost.

    warm_start : bool
        If True, initialize refitting from cached parameters.
        Speeds up convergence for small updates.

    Examples
    --------
    >>> # Unbounded memory (default)
    >>> config = OnlineConfig(strategy="full")

    >>> # Sliding window: keep last 10K trials
    >>> config = OnlineConfig(
    ...     strategy="sliding_window",
    ...     window_size=10_000,
    ...     refit_interval=10,  # Batch every 10 trials
    ... )

    >>> # Reservoir sampling: uniform coverage with 5K trials
    >>> config = OnlineConfig(
    ...     strategy="reservoir",
    ...     window_size=5_000,
    ... )
    """

    strategy: Literal["full", "sliding_window", "reservoir", "none"] = "full"
    window_size: int | None = None
    refit_interval: int = 1
    warm_start: bool = True

    def __post_init__(self):
        """Validate configuration."""
        if self.strategy in ["sliding_window", "reservoir"]:
            if self.window_size is None:
                raise ValueError(f"window_size required for strategy='{self.strategy}'")
            if self.window_size <= 0:
                raise ValueError(
                    f"window_size must be positive, got {self.window_size}"
                )

        if self.refit_interval <= 0:
            raise ValueError(
                f"refit_interval must be positive, got {self.refit_interval}"
            )


class Model(ABC):
    """
    Abstract base class for psychophysical models.

    Provides API that mimics BoTorch style:
    - fit(X, y) --> train model
    - posterior(X) --> get predictions
    - condition_on_observations(X, y) --> online updates

    Subclasses must implement:
    - init_params(key) --> sample initial parameters
    - log_likelihood_from_data(params, data) --> compute likelihood

    Parameters
    ----------
    online_config : OnlineConfig | None
        Configuration for online learning. If None, uses default (unbounded memory).

    Attributes
    ----------
    _posterior : ParameterPosterior | None
        Cached parameter posterior from last fit
    _inference_engine : InferenceEngine | None
        Cached inference engine for warm-start refitting
    _data_buffer : ResponseData | None
        Data buffer managed according to online_config
    _n_updates : int
        Number of condition_on_observations calls
    online_config : OnlineConfig
        Online learning configuration
    """

    def __init__(self, *, online_config: OnlineConfig | None = None):
        """
        Initialize model.

        Parameters
        ----------
        online_config : OnlineConfig | None
            Online learning configuration. If None, uses default settings.
        """
        self._posterior: ParameterPosterior | None = None
        self._inference_engine: InferenceEngine | None = None
        self._data_buffer: ResponseData | None = None
        self._n_updates: int = 0
        self.online_config = online_config or OnlineConfig()

    # ------------------------------------------------------------------
    # Abstract methods (must be implemented by subclasses)
    # ------------------------------------------------------------------

    @abstractmethod
    def init_params(self, key: Any) -> dict:  # jax.random.KeyArray
        """
        Sample initial parameters from prior.

        Parameters
        ----------
        key : jax.random.KeyArray
            PRNG key

        Returns
        -------
        dict
            Parameter PyTree
        """
        ...

    @abstractmethod
    def log_likelihood_from_data(self, params: dict, data: ResponseData) -> jnp.ndarray:
        """
        Compute log p(data | params).

        Parameters
        ----------
        params : dict
            Model parameters
        data : ResponseData
            Observed trials

        Returns
        -------
        jnp.ndarray
            Log-likelihood (scalar)
        """
        ...

    # ------------------------------------------------------------------
    # BoTorch-style API: fit, posterior, condition_on_observations
    # ------------------------------------------------------------------

    def fit(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        *,
        inference: InferenceEngine | str = "laplace",
        inference_config: dict | None = None,
    ) -> Model:
        """
        Fit model to data.

        Parameters
        ----------
        X : jnp.ndarray
            Stimuli, shape (n_trials, 2, input_dim) for (ref, probe) pairs
            or (n_trials, input_dim) for references only
        y : jnp.ndarray
            Responses, shape (n_trials,)
        inference : InferenceEngine | str, default="laplace"
            Inference engine or string key ("map", "laplace", "langevin")
        inference_config : dict | None
            Hyperparameters for string-based inference.
            Examples: {"steps": 500, "lr": 1e-3} for MAP

        Returns
        -------
        Model
            Self for method chaining

        Examples
        --------
        >>> # Simple: use defaults
        >>> model.fit(X, y)

        >>> # Explicit optimizer
        >>> from psyphy.inference import MAPOptimizer
        >>> model.fit(X, y, inference=MAPOptimizer(steps=500))

        >>> # String + config (for experiment tracking)
        >>> model.fit(X, y, inference="map", inference_config={"steps": 500})
        """
        from psyphy.data import ResponseData
        from psyphy.inference import INFERENCE_ENGINES, InferenceEngine

        # Resolve inference engine
        is_string_inference = isinstance(inference, str)

        if is_string_inference:
            config = inference_config or {}
            inference_key: str = inference  # type: ignore[assignment]
            if inference_key not in INFERENCE_ENGINES:
                available = ", ".join(INFERENCE_ENGINES.keys())
                raise ValueError(
                    f"Unknown inference: '{inference}'. Available: {available}"
                )
            inference_engine: InferenceEngine = INFERENCE_ENGINES[inference_key](
                **config
            )
        elif isinstance(inference, InferenceEngine):
            inference_engine = inference
        else:
            raise TypeError(
                f"inference must be InferenceEngine or str, got {type(inference)}"
            )

        if inference_config is not None and not is_string_inference:
            raise ValueError(
                "Cannot pass inference_config with InferenceEngine instance"
            )

        # Convert data
        data = ResponseData.from_arrays(X, y)

        # Fit
        self._posterior = inference_engine.fit(self, data)
        self._inference_engine = inference_engine
        self._data_buffer = data  # Initialize buffer
        return self

    def posterior(
        self,
        X: jnp.ndarray | None = None,
        *,
        probes: jnp.ndarray | None = None,
        kind: str = "predictive",
    ) -> PredictivePosterior | ParameterPosterior:
        """
        Return posterior distribution.

        Parameters
        ----------
        X : jnp.ndarray | None
            Test stimuli (references), shape (n_test, input_dim).
            Required for predictive posteriors, optional for parameter posteriors.
        probes : jnp.ndarray | None
            Test probes, shape (n_test, input_dim).
            Required for predictive posteriors.
        kind : {"predictive", "parameter"}
            Type of posterior to return:
            - "predictive": PredictivePosterior over f(X*) [for acquisitions]
            - "parameter": ParameterPosterior over θ [for diagnostics]

        Returns
        -------
        PredictivePosterior | ParameterPosterior
            Posterior distribution

        Raises
        ------
        RuntimeError
            If model has not been fit yet

        Examples
        --------
        >>> # For acquisition functions
        >>> pred_post = model.posterior(X_candidates, probes=X_probes)
        >>> mean = pred_post.mean
        >>> var = pred_post.variance

        >>> # For diagnostics
        >>> param_post = model.posterior(kind="parameter")
        >>> samples = param_post.sample(100, key=jr.PRNGKey(42))
        """
        if self._posterior is None:
            raise RuntimeError("Must call fit() before posterior()")

        if kind == "parameter":
            return self._posterior
        elif kind == "predictive":
            if X is None:
                raise ValueError("X is required for predictive posteriors")
            from psyphy.posterior import WPPMPredictivePosterior

            return WPPMPredictivePosterior(self._posterior, X, probes=probes)
        else:
            raise ValueError(
                f"Unknown kind: '{kind}'. Use 'predictive' or 'parameter'."
            )

    def condition_on_observations(self, X: jnp.ndarray, y: jnp.ndarray) -> Model:
        """
        Update model with new observations (online learning).

        Behavior depends on self.online_config.strategy:
        - "full": Accumulate all data, refit periodically
        - "sliding_window": Keep only recent window_size trials
        - "reservoir": Random sampling of window_size trials
        - "none": Refit from scratch (no caching)

        Returns a NEW model instance (immutable update).

        Parameters
        ----------
        X : jnp.ndarray
            New stimuli
        y : jnp.ndarray
            New responses

        Returns
        -------
        Model
            Updated model (new instance)

        Examples
        --------
        >>> # Online learning loop
        >>> model = WPPM(...).fit(X_init, y_init)
        >>> for X_new, y_new in stream:
        ...     model = model.condition_on_observations(X_new, y_new)
        ...     # Model automatically manages memory and refitting
        """
        from psyphy.data import ResponseData

        # Convert new data
        new_data = ResponseData.from_arrays(X, y)

        # Update data buffer according to strategy
        if self.online_config.strategy == "none":
            data_to_fit = new_data

        elif self.online_config.strategy == "full":
            if self._data_buffer is None:
                self._data_buffer = ResponseData()
            self._data_buffer.merge(new_data)
            data_to_fit = self._data_buffer

        elif self.online_config.strategy == "sliding_window":
            if self._data_buffer is None:
                self._data_buffer = ResponseData()
            self._data_buffer.merge(new_data)

            # Keep only last N trials
            window_size = self.online_config.window_size
            assert window_size is not None, "window_size must be set for sliding_window"
            if len(self._data_buffer) > window_size:
                self._data_buffer = self._data_buffer.tail(window_size)
            data_to_fit = self._data_buffer

        elif self.online_config.strategy == "reservoir":
            if self._data_buffer is None:
                self._data_buffer = ResponseData()

            # Reservoir sampling
            window_size = self.online_config.window_size
            assert window_size is not None, "window_size must be set for reservoir"
            self._data_buffer = self._reservoir_update(
                self._data_buffer,
                new_data,
                window_size,
            )
            data_to_fit = self._data_buffer
        else:
            raise ValueError(f"Unknown strategy: {self.online_config.strategy}")

        # Decide whether to refit
        self._n_updates += 1
        should_refit = self._n_updates % self.online_config.refit_interval == 0

        if not should_refit:
            # Return clone with updated buffer but old posterior
            new_model = self._clone()
            new_model._data_buffer = data_to_fit
            return new_model

        # Refit with optional warm start
        inference = self._inference_engine
        assert inference is not None, (
            "Model must be fit before condition_on_observations"
        )

        if self.online_config.warm_start and self._posterior is not None:
            # TODO: Add warm-start support to inference engines
            # For now, just refit from cached params
            pass

        new_model = self._clone()
        new_model._data_buffer = data_to_fit
        new_model._posterior = inference.fit(new_model, data_to_fit)
        new_model._inference_engine = inference

        return new_model

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _clone(self) -> Model:
        """Create shallow copy of model for immutable updates."""
        new_model = self.__class__.__new__(self.__class__)
        new_model.__dict__.update(self.__dict__)
        return new_model

    @staticmethod
    def _reservoir_update(
        buffer: ResponseData,
        new_data: ResponseData,
        capacity: int,
        *,
        key: Any = None,  # type: ignore[type-arg]
    ) -> ResponseData:
        """
        Update buffer using reservoir sampling (Vitter 1985).

        Maintains uniform sample of all seen data with fixed memory.

        Algorithm:
        For each new item with index n:
            If buffer not full: add
            Else: with probability capacity/n, replace random item

        Parameters
        ----------
        buffer : ResponseData
            Current data buffer
        new_data : ResponseData
            New trials to incorporate
        capacity : int
            Maximum buffer size
        key : jax.random.KeyArray | None
            PRNG key for randomness. If None, uses default seed.

        Returns
        -------
        ResponseData
            Updated buffer
        """
        if key is None:
            key = jr.PRNGKey(0)

        combined = buffer.copy()
        n_existing = len(buffer)

        for i, trial in enumerate(new_data.trials):
            n_total = n_existing + i + 1

            if len(combined) < capacity:
                # Buffer not full: just add
                combined.add_trial(*trial)
            else:
                # Accept with probability capacity / n_total
                key, subkey = jr.split(key)
                if jr.uniform(subkey) < capacity / n_total:
                    # Replace random existing trial
                    key, subkey = jr.split(key)
                    idx = int(jr.randint(subkey, (), 0, capacity))
                    combined.trials[idx] = trial

        return combined


def auto_online_config(memory_budget_mb: float) -> OnlineConfig:
    """
    Automatically choose online learning strategy based on memory budget.

    Parameters
    ----------
    memory_budget_mb : float
        Available memory in megabytes

    Returns
    -------
    OnlineConfig
        Recommended configuration

    Examples
    --------
    >>> # 100 MB budget → sliding window of ~10K trials (2D stimuli)
    >>> config = auto_online_config(100.0)
    >>> model = WPPM(..., online_config=config)
    """
    # Estimate: float64 × (2 refs + 2 probes + 1 response) × input_dim
    # Assume 2D stimuli: 8 bytes × 5 × 2 = 80 bytes/trial
    bytes_per_trial = 80
    max_trials = int(memory_budget_mb * 1e6 / bytes_per_trial)

    if max_trials > 1_000_000:
        # Very large budget -> keep everything
        return OnlineConfig(strategy="full")
    elif max_trials > 100_000:
        # Medium budget -> sliding window (100K-1M trials ≈ 8-80 MB)
        return OnlineConfig(
            strategy="sliding_window",
            window_size=max_trials,
            refit_interval=10,
        )
    else:
        # Small budget -> reservoir sampling (< 100K trials ≈ < 8 MB)
        return OnlineConfig(
            strategy="reservoir",
            window_size=max_trials,
            refit_interval=5,
        )
