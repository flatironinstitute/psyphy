"""
test_model_api.py
-----------------

Tests for the Model façade API (Issue #1):
- Model.fit() with hybrid inference configuration
- Model.posterior() for predictive and parameter posteriors
- Model.condition_on_observations() for online learning
- OnlineConfig strategies (full, sliding_window, reservoir)
"""

import jax.numpy as jnp
import jax.random as jr
import pytest

from psyphy.inference import MAPOptimizer
from psyphy.model import WPPM, OnlineConfig, Prior, auto_online_config
from psyphy.model.noise import GaussianNoise
from psyphy.model.task import OddityTask
from psyphy.posterior import ParameterPosterior, PredictivePosterior


class TestModelFit:
    """Test Model.fit() with different inference configurations."""

    @pytest.fixture
    def model(self):
        """Create a WPPM model."""
        return WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, scale=0.5),
            task=OddityTask(),
            noise=GaussianNoise(),
        )

    @pytest.fixture
    def data_arrays(self):
        """Generate synthetic data arrays."""
        n = 20
        key = jr.PRNGKey(42)
        key, subkey = jr.split(key)
        refs = jr.normal(subkey, (n, 2))
        key, subkey = jr.split(key)
        probes = refs + jr.normal(subkey, (n, 2)) * 0.3
        y = jnp.ones(n, dtype=int)  # All correct

        # Stack into (n, 2, input_dim) format
        X = jnp.stack([refs, probes], axis=1)
        return X, y

    def test_fit_returns_self(self, model, data_arrays):
        """fit() returns self for method chaining."""
        X, y = data_arrays
        result = model.fit(X, y, inference="map", inference_config={"steps": 10})
        assert result is model

    def test_fit_with_string_default(self, model, data_arrays):
        """Can fit with string inference key and default config."""
        X, y = data_arrays
        model.fit(X, y, inference="map")
        assert model._posterior is not None
        assert isinstance(model._posterior, ParameterPosterior)

    def test_fit_with_string_and_config(self, model, data_arrays):
        """Can fit with string and config dict."""
        X, y = data_arrays
        model.fit(X, y, inference="map", inference_config={"steps": 50})
        assert model._posterior is not None

    def test_fit_with_optimizer_instance(self, model, data_arrays):
        """Can fit with optimizer instance."""
        X, y = data_arrays
        optimizer = MAPOptimizer(steps=10)
        model.fit(X, y, inference=optimizer)
        assert model._posterior is not None

    def test_fit_caches_inference_engine(self, model, data_arrays):
        """fit() caches the inference engine."""
        X, y = data_arrays
        optimizer = MAPOptimizer(steps=10)
        model.fit(X, y, inference=optimizer)
        assert model._inference_engine is optimizer

    def test_fit_invalid_string_raises(self, model, data_arrays):
        """Invalid inference string raises ValueError."""
        X, y = data_arrays
        with pytest.raises(ValueError, match="Unknown inference"):
            model.fit(X, y, inference="invalid_method")

    def test_fit_config_with_instance_raises(self, model, data_arrays):
        """Passing config with instance raises ValueError."""
        X, y = data_arrays
        optimizer = MAPOptimizer(steps=10)
        with pytest.raises(ValueError, match="Cannot pass inference_config"):
            model.fit(X, y, inference=optimizer, inference_config={"steps": 20})


class TestModelPosterior:
    """Test Model.posterior() for predictive and parameter posteriors."""

    @pytest.fixture
    def fitted_model(self):
        """Create and fit a model."""
        model = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, scale=0.5),
            task=OddityTask(),
            noise=GaussianNoise(),
        )

        # Generate data
        n = 20
        key = jr.PRNGKey(42)
        key, subkey = jr.split(key)
        refs = jr.normal(subkey, (n, 2))
        key, subkey = jr.split(key)
        probes = refs + jr.normal(subkey, (n, 2)) * 0.3
        y = jnp.ones(n, dtype=int)
        X = jnp.stack([refs, probes], axis=1)

        model.fit(X, y, inference="map", inference_config={"steps": 20})
        return model

    def test_posterior_before_fit_raises(self):
        """Calling posterior() before fit() raises RuntimeError."""
        model = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2),
            task=OddityTask(),
            noise=GaussianNoise(),
        )
        with pytest.raises(RuntimeError, match="Must call fit"):
            model.posterior(jnp.array([[0.0, 0.0]]))

    def test_posterior_predictive_default(self, fitted_model):
        """posterior() returns PredictivePosterior by default."""
        X_test = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        probes = jnp.array([[0.5, 0.0], [1.5, 1.0]])
        post = fitted_model.posterior(X_test, probes=probes)
        assert isinstance(post, PredictivePosterior)

    def test_posterior_parameter_kind(self, fitted_model):
        """posterior(kind='parameter') returns ParameterPosterior."""
        post = fitted_model.posterior(kind="parameter")
        assert isinstance(post, ParameterPosterior)

    def test_posterior_invalid_kind_raises(self, fitted_model):
        """Invalid kind raises ValueError."""
        with pytest.raises(ValueError, match="Unknown kind"):
            fitted_model.posterior(kind="invalid")

    def test_posterior_predictive_has_correct_shape(self, fitted_model):
        """Predictive posterior has correct shape."""
        X_test = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        probes = jnp.array([[0.5, 0.0], [1.5, 1.0], [2.5, 2.0]])
        post = fitted_model.posterior(X_test, probes=probes)

        mean = post.mean
        var = post.variance

        assert mean.shape == (3,)
        assert var.shape == (3,)


class TestConditionOnObservations:
    """Test online learning with condition_on_observations()."""

    @pytest.fixture
    def model(self):
        """Create a model with online config."""
        return WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, scale=0.5),
            task=OddityTask(),
            noise=GaussianNoise(),
            online_config=OnlineConfig(strategy="full", refit_interval=1),
        )

    @pytest.fixture
    def initial_data(self):
        """Generate initial training data."""
        n = 10
        key = jr.PRNGKey(42)
        key, subkey = jr.split(key)
        refs = jr.normal(subkey, (n, 2))
        key, subkey = jr.split(key)
        probes = refs + jr.normal(subkey, (n, 2)) * 0.3
        y = jnp.ones(n, dtype=int)
        X = jnp.stack([refs, probes], axis=1)
        return X, y

    def test_condition_returns_new_instance(self, model, initial_data):
        """condition_on_observations returns new model instance."""
        X, y = initial_data
        model.fit(X, y, inference="map", inference_config={"steps": 10})

        X_new = jnp.array([[[0.0, 0.0], [0.5, 0.0]]])
        y_new = jnp.array([1])

        model2 = model.condition_on_observations(X_new, y_new)

        assert model2 is not model
        assert model2._posterior is not model._posterior

    def test_condition_updates_buffer(self, model, initial_data):
        """condition_on_observations updates data buffer."""
        X, y = initial_data
        model.fit(X, y, inference="map", inference_config={"steps": 10})

        initial_buffer_len = len(model._data_buffer)

        X_new = jnp.array([[[0.0, 0.0], [0.5, 0.0]]])
        y_new = jnp.array([1])

        model2 = model.condition_on_observations(X_new, y_new)

        assert len(model2._data_buffer) == initial_buffer_len + 1

    def test_condition_increments_update_counter(self, model, initial_data):
        """condition_on_observations increments update counter."""
        X, y = initial_data
        model.fit(X, y, inference="map", inference_config={"steps": 10})

        assert model._n_updates == 0

        X_new = jnp.array([[[0.0, 0.0], [0.5, 0.0]]])
        y_new = jnp.array([1])

        model2 = model.condition_on_observations(X_new, y_new)
        assert model2._n_updates == 1


class TestOnlineConfig:
    """Test OnlineConfig strategies."""

    def test_online_config_full_strategy(self):
        """Full strategy keeps all data."""
        config = OnlineConfig(strategy="full")
        assert config.strategy == "full"
        assert config.window_size is None

    def test_online_config_sliding_window(self):
        """Sliding window requires window_size."""
        config = OnlineConfig(strategy="sliding_window", window_size=100)
        assert config.window_size == 100

    def test_online_config_sliding_window_without_size_raises(self):
        """Sliding window without window_size raises ValueError."""
        with pytest.raises(ValueError, match="window_size required"):
            OnlineConfig(strategy="sliding_window")

    def test_online_config_reservoir(self):
        """Reservoir sampling requires window_size."""
        config = OnlineConfig(strategy="reservoir", window_size=50)
        assert config.window_size == 50

    def test_online_config_invalid_window_size_raises(self):
        """Negative window_size raises ValueError."""
        with pytest.raises(ValueError, match="window_size must be positive"):
            OnlineConfig(strategy="sliding_window", window_size=-10)

    def test_online_config_invalid_refit_interval_raises(self):
        """Invalid refit_interval raises ValueError."""
        with pytest.raises(ValueError, match="refit_interval must be positive"):
            OnlineConfig(refit_interval=0)


class TestSlidingWindowStrategy:
    """Test sliding window memory management."""

    def test_sliding_window_limits_buffer_size(self):
        """Sliding window keeps only last N trials."""
        model = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2),
            task=OddityTask(),
            noise=GaussianNoise(),
            online_config=OnlineConfig(
                strategy="sliding_window",
                window_size=5,
                refit_interval=1,
            ),
        )

        # Initial fit with 10 trials
        n = 10
        key = jr.PRNGKey(42)
        key, subkey = jr.split(key)
        refs = jr.normal(subkey, (n, 2))
        key, subkey = jr.split(key)
        probes = refs + jr.normal(subkey, (n, 2)) * 0.3
        y = jnp.ones(n, dtype=int)
        X = jnp.stack([refs, probes], axis=1)

        model.fit(X, y, inference="map", inference_config={"steps": 10})

        # Add more data
        X_new = jnp.array([[[0.0, 0.0], [0.5, 0.0]]] * 3)
        y_new = jnp.array([1, 1, 1])

        model2 = model.condition_on_observations(X_new, y_new)

        # Should keep only last 5 trials
        assert len(model2._data_buffer) == 5


class TestAutoOnlineConfig:
    """Test auto_online_config helper."""

    def test_large_budget_uses_full(self):
        """Large memory budget uses full strategy."""
        config = auto_online_config(1000.0)  # 1 GB
        assert config.strategy == "full"

    def test_medium_budget_uses_sliding_window(self):
        """Medium budget uses sliding window."""
        config = auto_online_config(50.0)  # 50 MB
        assert config.strategy == "sliding_window"
        assert config.window_size is not None
        assert config.window_size > 10_000

    def test_small_budget_uses_reservoir(self):
        """Small budget uses reservoir sampling."""
        config = auto_online_config(5.0)  # 5 MB
        assert config.strategy == "reservoir"
        assert config.window_size is not None


class TestIntegrationWorkflow:
    """Integration tests for complete workflows."""

    def test_full_new_api_workflow(self):
        """Test complete workflow with new API."""
        # 1. Create model
        model = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2),
            task=OddityTask(),
            noise=GaussianNoise(),
        )

        # 2. Generate data
        n = 30
        key = jr.PRNGKey(123)
        key, subkey = jr.split(key)
        refs = jr.normal(subkey, (n, 2))
        key, subkey = jr.split(key)
        probes = refs + jr.normal(subkey, (n, 2)) * 0.4
        y = jnp.ones(n, dtype=int)
        X = jnp.stack([refs, probes], axis=1)

        # 3. Fit model (new API)
        model.fit(X, y, inference="map", inference_config={"steps": 50})

        # 4. Get predictive posterior
        X_test = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        probes_test = jnp.array([[0.3, 0.0], [1.3, 1.0]])
        pred_post = model.posterior(X_test, probes=probes_test)

        # 5. Make predictions
        mean = pred_post.mean
        var = pred_post.variance

        assert mean.shape == (2,)
        assert var.shape == (2,)
        assert jnp.all((mean >= 0) & (mean <= 1))

        # 6. Get parameter posterior for diagnostics
        param_post = model.posterior(kind="parameter")
        samples = param_post.sample(10, key=jr.PRNGKey(42))
        assert samples["log_diag"].shape == (10, 2)

    def test_online_learning_workflow(self):
        """Test online learning with bounded memory."""
        # 1. Create model with online config
        model = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2),
            task=OddityTask(),
            noise=GaussianNoise(),
            online_config=OnlineConfig(
                strategy="sliding_window",
                window_size=20,
                refit_interval=5,
            ),
        )

        # 2. Initial training
        n = 15
        key = jr.PRNGKey(456)
        key, subkey = jr.split(key)
        refs = jr.normal(subkey, (n, 2))
        key, subkey = jr.split(key)
        probes = refs + jr.normal(subkey, (n, 2)) * 0.3
        y = jnp.ones(n, dtype=int)
        X = jnp.stack([refs, probes], axis=1)

        model.fit(X, y, inference="map", inference_config={"steps": 20})

        # 3. Online updates
        for _ in range(10):
            key, subkey = jr.split(key)
            ref_new = jr.normal(subkey, (1, 2))
            key, subkey = jr.split(key)
            probe_new = ref_new + jr.normal(subkey, (1, 2)) * 0.3
            X_new = jnp.stack([ref_new, probe_new], axis=1)
            y_new = jnp.array([1])

            model = model.condition_on_observations(X_new, y_new)

        # 4. Verify memory bounded
        assert len(model._data_buffer) <= 20

        # 5. Can still make predictions
        X_test = jnp.array([[0.0, 0.0]])
        probes_test = jnp.array([[0.3, 0.0]])
        pred_post = model.posterior(X_test, probes=probes_test)
        mean = pred_post.mean
        assert mean.shape == (1,)
