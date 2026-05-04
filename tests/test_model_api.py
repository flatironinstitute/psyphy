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

from psyphy.data import ResponseData
from psyphy.inference import MAPOptimizer
from psyphy.model import WPPM, Prior
from psyphy.model.likelihood import OddityTask
from psyphy.model.noise import GaussianNoise
from psyphy.posterior import ParameterPosterior


class TestModelFit:
    """Test Model.fit() with different inference configurations."""

    @pytest.fixture
    def model(self):
        """Create a WPPM model."""
        return WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=3),
            likelihood=OddityTask(),
            noise=GaussianNoise(),
        )

    @pytest.fixture
    def data_arrays(self):
        """Generate synthetic data arrays in ResponseData format."""
        n = 20
        key = jr.PRNGKey(42)
        key, subkey = jr.split(key)
        refs = jr.normal(subkey, (n, 2))
        key, subkey = jr.split(key)
        comparisons = refs + jr.normal(subkey, (n, 2)) * 0.3
        y = jnp.ones(n, dtype=int)  # All correct

        # Create ResponseData object
        data = ResponseData()
        for i in range(n):
            data.add_trial((refs[i], comparisons[i]), int(y[i]))
        return data

    def test_optimizer_fit(self, model, data_arrays):
        """Optimizer.fit() returns a posterior."""
        optimizer = MAPOptimizer(steps=10)
        posterior = optimizer.fit(model, data_arrays)
        assert posterior is not None
        assert isinstance(posterior, ParameterPosterior)

    def test_fit_with_different_steps(self, model, data_arrays):
        """Optimizer is configurable."""
        optimizer = MAPOptimizer(steps=50)
        posterior = optimizer.fit(model, data_arrays)
        assert posterior.params is not None


class TestModelPosterior:
    """Test Posterior predictions."""

    @pytest.fixture
    def posterior(self):
        """Create a fitted posterior."""
        model = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=3),
            likelihood=OddityTask(),
            noise=GaussianNoise(),
        )

        # Generate data
        n = 20
        key = jr.PRNGKey(42)
        key, subkey = jr.split(key)
        refs = jr.normal(subkey, (n, 2))
        key, subkey = jr.split(key)
        comparisons = refs + jr.normal(subkey, (n, 2)) * 0.3
        y = jnp.ones(n, dtype=int)

        data = ResponseData()
        for i in range(n):
            data.add_trial((refs[i], comparisons[i]), int(y[i]))

        optimizer = MAPOptimizer(steps=20)
        return optimizer.fit(model, data)

    def test_posterior_predictive_default(self, posterior):
        """posterior.predict() returns PredictivePosterior by default."""
        X_test = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        comparisons = jnp.array([[0.5, 0.0], [1.5, 1.0]])

        # In the new API (see WPPMPredictivePosterior), prediction is creating
        # a new predictive object around the parameter posterior.
        from psyphy.posterior.predictive_posterior import WPPMPredictivePosterior

        # It's not a method on posterior, but a wrapper class construction
        pred = WPPMPredictivePosterior(posterior, X_test, comparisons=comparisons)

        # Check duck typing
        assert hasattr(pred, "mean")
        assert hasattr(pred, "variance")

    def test_posterior_parameter_access(self, posterior):
        """posterior.params access."""
        params = posterior.params
        assert isinstance(params, dict)

    def test_posterior_predictive_has_correct_shape(self, posterior):
        """Predictive posterior has correct shape."""
        X_test = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        comparisons = jnp.array([[0.5, 0.0], [1.5, 1.0], [2.5, 2.0]])

        from psyphy.posterior.predictive_posterior import WPPMPredictivePosterior

        pred = WPPMPredictivePosterior(posterior, X_test, comparisons=comparisons)

        # Triggers lazy computation
        mean = pred.mean
        var = pred.variance

        assert mean.shape == (3,)
        assert var.shape == (3,)


class TestConditionOnObservations:
    """Test online learning via creating new posteriors."""

    @pytest.fixture
    def model(self):
        """Create a model."""
        return WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=3),
            likelihood=OddityTask(),
            noise=GaussianNoise(),
        )

    @pytest.fixture
    def initial_data(self):
        """Generate initial training data."""
        n = 10
        key = jr.PRNGKey(42)
        key, subkey = jr.split(key)
        refs = jr.normal(subkey, (n, 2))
        key, subkey = jr.split(key)
        comparisons = refs + jr.normal(subkey, (n, 2)) * 0.3
        y = jnp.ones(n, dtype=int)

        data = ResponseData()
        for i in range(n):
            data.add_trial((refs[i], comparisons[i]), int(y[i]))
        return data

    def test_initial_fit(self, model, initial_data):
        optimizer = MAPOptimizer(steps=10)
        posterior = optimizer.fit(model, initial_data)
        assert posterior is not None


class TestIntegrationWorkflow:
    """Integration tests for complete workflows."""

    def test_full_new_api_workflow(self):
        """Test complete workflow with new API."""
        # 1. Create model
        model = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=3),
            likelihood=OddityTask(),
            noise=GaussianNoise(),
        )

        # 2. Generate data
        n = 30
        key = jr.PRNGKey(123)
        key, subkey = jr.split(key)
        refs = jr.normal(subkey, (n, 2))
        key, subkey = jr.split(key)
        comparisons = refs + jr.normal(subkey, (n, 2)) * 0.4
        y = jnp.ones(n, dtype=int)

        data = ResponseData()
        for i in range(n):
            data.add_trial((refs[i], comparisons[i]), int(y[i]))

        # 3. Fit model (new API)
        optimizer = MAPOptimizer(steps=50)
        posterior = optimizer.fit(model, data)

        # 4. Get predictive posterior
        X_test = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        probes_test = jnp.array([[0.3, 0.0], [1.3, 1.0]])

        from psyphy.posterior.predictive_posterior import WPPMPredictivePosterior

        pred = WPPMPredictivePosterior(posterior, X_test, comparisons=probes_test)

        # 5. Make predictions
        mean = pred.mean
        var = pred.variance

        assert mean.shape == (2,)
        assert var.shape == (2,)
        assert jnp.all((mean >= 0) & (mean <= 1))

    def test_manual_online_loop(self):
        """Test online learning manually."""
        # 1. Create model
        model = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=3),
            likelihood=OddityTask(),
            noise=GaussianNoise(),
        )
        optimizer = MAPOptimizer(steps=10)
        data = ResponseData()

        key = jr.PRNGKey(456)

        # Online loop
        for i in range(5):
            key, subkey = jr.split(key)
            ref = jr.normal(subkey, (2,))
            key, subkey = jr.split(key)
            comp = ref + jr.normal(subkey, (2,)) * 0.1

            data.add_trial((ref, comp), 1)

            # Re-fit every step (naive online learning)
            posterior = optimizer.fit(model, data)
            assert posterior is not None
            assert len(data.responses) == i + 1
