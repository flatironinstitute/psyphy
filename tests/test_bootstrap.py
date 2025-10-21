"""
test_bootstrap.py
-----------------

Tests for bootstrap resampling utilities.

Coverage:
- bootstrap_predictions: confidence intervals for psychometric functions
- bootstrap_statistic: CIs for arbitrary model statistics
- bootstrap_compare_models: statistical model comparison
"""

import jax.numpy as jnp
import jax.random as jr
import pytest

from psyphy.model import WPPM, Prior
from psyphy.model.noise import GaussianNoise
from psyphy.model.task import OddityTask
from psyphy.utils.bootstrap import (
    bootstrap_compare_models,
    bootstrap_predictions,
    bootstrap_statistic,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def simple_data():
    """Create simple training data for bootstrap tests."""
    n = 30
    key = jr.PRNGKey(42)
    key, subkey = jr.split(key)
    refs = jr.normal(subkey, (n, 2))
    key, subkey = jr.split(key)
    probes = refs + jr.normal(subkey, (n, 2)) * 0.3
    y = jnp.ones(n, dtype=int)
    X = jnp.stack([refs, probes], axis=1)
    return X, y


@pytest.fixture
def unfitted_model():
    """Create an unfitted WPPM model for bootstrap."""
    return WPPM(
        input_dim=2,
        prior=Prior(input_dim=2, scale=0.5),
        task=OddityTask(),
        noise=GaussianNoise(),
    )


# ============================================================================
# Test bootstrap_predictions
# ============================================================================


class TestBootstrapPredictions:
    """Test bootstrap confidence intervals for predictions."""

    def test_basic_functionality(self, unfitted_model, simple_data):
        """Bootstrap CIs run and return expected shapes."""
        X_train, y_train = simple_data
        X_test = jnp.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        probes_test = X_test + 0.1

        mean, lower, upper = bootstrap_predictions(
            unfitted_model,
            X_train,
            y_train,
            X_test,
            probes=probes_test,
            n_bootstrap=5,  # Small for speed
            inference_config={"steps": 10},
            key=jr.PRNGKey(0),
        )

        # Check shapes
        assert mean.shape == (3,)
        assert lower.shape == (3,)
        assert upper.shape == (3,)

    def test_ci_ordering(self, unfitted_model, simple_data):
        """CI bounds are properly ordered."""
        X_train, y_train = simple_data
        X_test = jnp.array([[0.0, 0.0], [0.5, 0.5]])
        probes_test = X_test + 0.1

        mean, lower, upper = bootstrap_predictions(
            unfitted_model,
            X_train,
            y_train,
            X_test,
            probes=probes_test,
            n_bootstrap=5,
            inference_config={"steps": 10},
            key=jr.PRNGKey(1),
        )

        # Lower < mean < upper
        assert jnp.all(lower <= mean)
        assert jnp.all(mean <= upper)

    def test_deterministic_with_seed(self, unfitted_model, simple_data):
        """Same seed gives same results."""
        X_train, y_train = simple_data
        X_test = jnp.array([[0.0, 0.0]])
        probes_test = X_test + 0.1

        mean1, lower1, upper1 = bootstrap_predictions(
            unfitted_model,
            X_train,
            y_train,
            X_test,
            probes=probes_test,
            n_bootstrap=5,
            inference_config={"steps": 10},
            key=jr.PRNGKey(42),
        )

        mean2, lower2, upper2 = bootstrap_predictions(
            unfitted_model,
            X_train,
            y_train,
            X_test,
            probes=probes_test,
            n_bootstrap=5,
            inference_config={"steps": 10},
            key=jr.PRNGKey(42),
        )

        assert jnp.allclose(mean1, mean2)
        assert jnp.allclose(lower1, lower2)
        assert jnp.allclose(upper1, upper2)

    def test_confidence_level(self, unfitted_model, simple_data):
        """Higher confidence level gives wider intervals."""
        X_train, y_train = simple_data
        X_test = jnp.array([[0.0, 0.0]])
        probes_test = X_test + 0.1

        _, lower_95, upper_95 = bootstrap_predictions(
            unfitted_model,
            X_train,
            y_train,
            X_test,
            probes=probes_test,
            confidence_level=0.95,
            n_bootstrap=10,
            inference_config={"steps": 10},
            key=jr.PRNGKey(0),
        )

        _, lower_99, upper_99 = bootstrap_predictions(
            unfitted_model,
            X_train,
            y_train,
            X_test,
            probes=probes_test,
            confidence_level=0.99,
            n_bootstrap=10,
            inference_config={"steps": 10},
            key=jr.PRNGKey(0),
        )

        # 99% CI should be wider than 95% CI
        width_95 = upper_95[0] - lower_95[0]
        width_99 = upper_99[0] - lower_99[0]
        assert width_99 >= width_95


# ============================================================================
# Test bootstrap_statistic
# ============================================================================


class TestBootstrapStatistic:
    """Test bootstrap CIs for arbitrary statistics."""

    def test_scalar_statistic(self, unfitted_model, simple_data):
        """Bootstrap works with scalar-valued statistics."""
        X, y = simple_data

        # Example statistic: sum of parameter norms
        def get_param_norm(fitted_model):
            params = fitted_model._posterior.params
            return float(
                jnp.sum(jnp.array([jnp.linalg.norm(p) for p in params.values()]))
            )

        estimate, lower, upper = bootstrap_statistic(
            unfitted_model,
            X,
            y,
            statistic_fn=get_param_norm,
            n_bootstrap=5,
            inference_config={"steps": 10},
            key=jr.PRNGKey(0),
        )

        assert isinstance(estimate, (float, jnp.ndarray))
        assert isinstance(lower, (float, jnp.ndarray))
        assert isinstance(upper, (float, jnp.ndarray))
        assert lower <= estimate <= upper

    def test_vector_statistic(self, unfitted_model, simple_data):
        """Bootstrap works with vector-valued statistics."""
        X, y = simple_data

        # Example: predictions at multiple test points
        X_test = jnp.array([[0.0, 0.0], [0.5, 0.5]])
        probes_test = X_test + 0.1

        def get_predictions(fitted_model):
            posterior = fitted_model.posterior(X_test, probes=probes_test)
            return posterior.mean  # type: ignore[attr-defined]

        estimate, lower, upper = bootstrap_statistic(
            unfitted_model,
            X,
            y,
            statistic_fn=get_predictions,
            n_bootstrap=5,
            inference_config={"steps": 10},
            key=jr.PRNGKey(1),
        )

        # Should be vectors (type: ignore because return type is float | ndarray)
        assert jnp.asarray(estimate).shape == (2,)
        assert jnp.asarray(lower).shape == (2,)
        assert jnp.asarray(upper).shape == (2,)

    def test_deterministic_statistic(self, unfitted_model, simple_data):
        """Deterministic statistics give same results with same seed."""
        X, y = simple_data

        def get_param_sum(fitted_model):
            # Sum all parameters as a simple statistic
            params = fitted_model._posterior.params
            return float(jnp.sum(jnp.array([jnp.sum(p) for p in params.values()])))

        est1, l1, u1 = bootstrap_statistic(
            unfitted_model,
            X,
            y,
            statistic_fn=get_param_sum,
            n_bootstrap=5,
            inference_config={"steps": 10},
            key=jr.PRNGKey(42),
        )

        est2, l2, u2 = bootstrap_statistic(
            unfitted_model,
            X,
            y,
            statistic_fn=get_param_sum,
            n_bootstrap=5,
            inference_config={"steps": 10},
            key=jr.PRNGKey(42),
        )

        assert jnp.allclose(est1, est2)
        assert jnp.allclose(l1, l2)
        assert jnp.allclose(u1, u2)


# ============================================================================
# Test bootstrap_compare_models
# ============================================================================


class TestBootstrapCompareModels:
    """Test statistical model comparison."""

    def test_basic_comparison(self, simple_data):
        """Model comparison runs and returns expected output."""
        X_train, y_train = simple_data

        # Split into train/test
        n_train = 20
        X_tr, X_te = X_train[:n_train], X_train[n_train:]
        y_tr, y_te = y_train[:n_train], y_train[n_train:]

        # Create two models with different priors
        model1 = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, scale=0.3),
            task=OddityTask(),
            noise=GaussianNoise(),
        )
        model2 = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, scale=0.7),
            task=OddityTask(),
            noise=GaussianNoise(),
        )

        probes_te = X_te[:, 0, :] + 0.1  # Extract references and add offset

        diff, lower, upper, significant = bootstrap_compare_models(
            model1,
            model2,
            X_tr,
            y_tr,
            X_te[:, 0, :],  # Just use references for simplicity
            y_te,
            probes=probes_te,
            n_bootstrap=5,
            inference_config={"steps": 10},
            key=jr.PRNGKey(0),
        )

        # Check types
        assert isinstance(diff, float)
        assert isinstance(lower, float)
        assert isinstance(upper, float)
        assert isinstance(significant, bool)

        # Check ordering
        assert lower <= diff <= upper

    def test_same_model_no_difference(self, simple_data):
        """Comparing identical models shows no significant difference."""
        X_train, y_train = simple_data

        # Split into train/test
        n_train = 20
        X_tr, X_te = X_train[:n_train], X_train[n_train:]
        y_tr, y_te = y_train[:n_train], y_train[n_train:]

        # Same model configuration
        model1 = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, scale=0.5),
            task=OddityTask(),
            noise=GaussianNoise(),
        )
        model2 = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, scale=0.5),
            task=OddityTask(),
            noise=GaussianNoise(),
        )

        probes_te = X_te[:, 0, :] + 0.1

        diff, lower, upper, significant = bootstrap_compare_models(
            model1,
            model2,
            X_tr,
            y_tr,
            X_te[:, 0, :],
            y_te,
            probes=probes_te,
            n_bootstrap=5,
            inference_config={"steps": 10},
            key=jr.PRNGKey(1),
        )

        # Difference should be close to zero
        assert jnp.abs(diff) < 0.3  # Allow some variance from randomness

        # CI should include zero (not significant)
        assert lower <= 0 <= upper

    def test_custom_metric(self, simple_data):
        """Custom metrics work correctly."""
        X_train, y_train = simple_data

        n_train = 20
        X_tr, X_te = X_train[:n_train], X_train[n_train:]
        y_tr, y_te = y_train[:n_train], y_train[n_train:]

        # Custom metric: count correct predictions
        def count_correct(y_true, y_pred):
            return float(jnp.sum(y_pred == y_true))

        model1 = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2),
            task=OddityTask(),
            noise=GaussianNoise(),
        )
        model2 = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2),
            task=OddityTask(),
            noise=GaussianNoise(),
        )

        probes_te = X_te[:, 0, :] + 0.1

        diff, lower, upper, significant = bootstrap_compare_models(
            model1,
            model2,
            X_tr,
            y_tr,
            X_te[:, 0, :],
            y_te,
            probes=probes_te,
            metric_fn=count_correct,
            n_bootstrap=5,
            inference_config={"steps": 10},
            key=jr.PRNGKey(2),
        )

        # With count_correct, values should be in range [0, n_test]
        n_test = len(y_te)
        assert -n_test <= diff <= n_test
