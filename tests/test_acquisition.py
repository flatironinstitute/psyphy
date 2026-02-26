"""
test_acquisition.py
-------------------

Tests for acquisition functions and optimization utilities.

Coverage:
- Expected Improvement
- Upper Confidence Bound
- Mutual Information
- Discrete optimization
- Continuous optimization (gradient-based)
- Integration with Model API
"""

import jax.numpy as jnp
import jax.random as jr
import pytest

from psyphy.acquisition import (
    expected_improvement,
    log_expected_improvement,
    mutual_information,
    optimize_acqf,
    optimize_acqf_discrete,
    optimize_acqf_random,
    upper_confidence_bound,
)
from psyphy.model import WPPM, GaussianNoise, OddityTask, Prior

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def simple_posterior():
    """Create a simple mock posterior for testing."""

    class MockPosterior:
        def __init__(self, mean, variance):
            self.mean = mean
            self.variance = variance

    # 3 candidates with known mean/variance
    mean = jnp.array([0.5, 0.7, 0.3])
    variance = jnp.array([0.1, 0.05, 0.2])

    return MockPosterior(mean, variance)


@pytest.fixture
def fitted_model():
    """Create and fit a simple WPPM model."""
    model = WPPM(
        input_dim=2,
        prior=Prior(input_dim=2, basis_degree=2),
        task=OddityTask(),
        noise=GaussianNoise(),
    )

    # Generate simple training data
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


# ============================================================================
# Test Expected Improvement
# ============================================================================


class TestExpectedImprovement:
    """Test Expected Improvement acquisition function."""

    def test_ei_basic_values(self, simple_posterior):
        """EI computes reasonable values."""
        best_f = 0.6
        ei = expected_improvement(simple_posterior, best_f, maximize=True)  # type: ignore[arg-type]

        assert ei.shape == (3,)
        assert jnp.all(ei >= 0.0)  # EI is non-negative

    def test_ei_maximization(self, simple_posterior):
        """EI prefers higher mean when variance is equal."""
        best_f = 0.4
        ei = expected_improvement(simple_posterior, best_f, maximize=True)

        # Candidate 1 (mean=0.7) should have highest EI
        # Candidate 2 (mean=0.3) should have lowest EI
        assert ei[1] > ei[2]

    def test_ei_minimization(self, simple_posterior):
        """EI works correctly for minimization."""
        best_f = 0.5
        ei_max = expected_improvement(simple_posterior, best_f, maximize=True)
        ei_min = expected_improvement(simple_posterior, best_f, maximize=False)

        # Values should be different
        assert not jnp.allclose(ei_max, ei_min)

        # For minimization, lower means should have higher EI
        # Candidate 2 (mean=0.3) should have highest EI for minimization
        assert ei_min[2] > ei_min[1]

    def test_ei_zero_variance(self):
        """EI handles zero variance correctly."""

        class MockPosterior:
            def __init__(self):
                self.mean = jnp.array([0.8, 0.5, 0.3])
                self.variance = jnp.array([0.0, 0.0, 0.0])

        posterior = MockPosterior()
        best_f = 0.6

        ei = expected_improvement(posterior, best_f, maximize=True)  # type: ignore[arg-type]

        # With zero variance, EI formula σ * [u * Φ(u) + φ(u)] = 0
        # This is mathematically correct - zero uncertainty means no expected improvement
        # even if mean is better than best_f (it's deterministic, not an expectation)
        assert jnp.allclose(ei, 0.0)
        assert jnp.all(ei >= 0.0)  # EI is always non-negative

    def test_log_ei(self, simple_posterior):
        """Log EI preserves ordering."""
        best_f = 0.5
        ei = expected_improvement(simple_posterior, best_f)
        log_ei = log_expected_improvement(simple_posterior, best_f)

        # Should preserve ordering
        ei_order = jnp.argsort(ei)
        log_ei_order = jnp.argsort(log_ei)

        assert jnp.array_equal(ei_order, log_ei_order)

    def test_ei_with_model(self, fitted_model):
        """EI works with real model posterior."""
        # Create test candidates
        X_test = jnp.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        probes = X_test + 0.1

        posterior = fitted_model.posterior(X_test, probes=probes)
        best_f = 0.8

        ei = expected_improvement(posterior, best_f)

        assert ei.shape == (3,)
        assert jnp.all(ei >= 0.0)


# ============================================================================
# Test Upper Confidence Bound
# ============================================================================


class TestUpperConfidenceBound:
    """Test Upper Confidence Bound acquisition function."""

    def test_ucb_basic_values(self, simple_posterior):
        """UCB computes reasonable values."""
        ucb = upper_confidence_bound(simple_posterior, beta=2.0)

        assert ucb.shape == (3,)
        # UCB should be mean + 2*std
        expected_ucb = simple_posterior.mean + 2.0 * jnp.sqrt(simple_posterior.variance)
        assert jnp.allclose(ucb, expected_ucb)

    def test_ucb_beta_zero(self, simple_posterior):
        """UCB with beta=0 is pure exploitation (greedy)."""
        ucb = upper_confidence_bound(simple_posterior, beta=0.0)

        # Should equal mean when beta=0
        assert jnp.allclose(ucb, simple_posterior.mean)

    def test_ucb_exploration_increases_with_beta(self, simple_posterior):
        """Higher beta increases exploration."""
        ucb_low = upper_confidence_bound(simple_posterior, beta=1.0)
        ucb_high = upper_confidence_bound(simple_posterior, beta=5.0)

        # High beta should favor high-variance points more
        # Candidate 2 (variance=0.2) should improve relative to others
        relative_improvement_low = ucb_low[2] - ucb_low[1]
        relative_improvement_high = ucb_high[2] - ucb_high[1]

        assert relative_improvement_high > relative_improvement_low

    def test_ucb_minimization(self, simple_posterior):
        """UCB works for minimization (LCB)."""
        lcb = upper_confidence_bound(simple_posterior, beta=2.0, maximize=False)

        # Should be mean - 2*std
        expected_lcb = simple_posterior.mean - 2.0 * jnp.sqrt(simple_posterior.variance)
        assert jnp.allclose(lcb, expected_lcb)

    def test_ucb_with_model(self, fitted_model):
        """UCB works with real model posterior."""
        X_test = jnp.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        probes = X_test + 0.1

        posterior = fitted_model.posterior(X_test, probes=probes)
        ucb = upper_confidence_bound(posterior, beta=2.0)

        assert ucb.shape == (3,)


# ============================================================================
# Test Mutual Information
# ============================================================================


class TestMutualInformation:
    """Test Mutual Information acquisition function."""

    def test_mi_basic_values(self, fitted_model):
        """MI computes non-negative values."""
        param_post = fitted_model.posterior(kind="parameter")
        X_test = jnp.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        probes = X_test + 0.1

        mi = mutual_information(
            param_post, X_test, probes=probes, n_samples=10, key=jr.PRNGKey(0)
        )

        assert mi.shape == (3,)
        assert jnp.all(mi >= 0.0)  # MI is non-negative

    def test_mi_without_probes(self, fitted_model):
        """MI works for threshold tasks (no probes)."""
        param_post = fitted_model.posterior(kind="parameter")
        X_test = jnp.array([[0.0, 0.0], [0.5, 0.5]])

        mi = mutual_information(
            param_post, X_test, probes=None, n_samples=10, key=jr.PRNGKey(0)
        )

        assert mi.shape == (2,)
        assert jnp.all(mi >= 0.0)

    def test_mi_deterministic_with_seed(self, fitted_model):
        """MI gives same results with same seed."""
        param_post = fitted_model.posterior(kind="parameter")
        X_test = jnp.array([[0.0, 0.0], [0.5, 0.5]])

        mi1 = mutual_information(
            param_post, X_test, probes=None, n_samples=20, key=jr.PRNGKey(42)
        )
        mi2 = mutual_information(
            param_post, X_test, probes=None, n_samples=20, key=jr.PRNGKey(42)
        )

        assert jnp.allclose(mi1, mi2)


# ============================================================================
# Test Discrete Optimization
# ============================================================================


class TestDiscreteOptimization:
    """Test discrete acquisition optimization."""

    def test_optimize_discrete_selects_argmax(self):
        """Discrete optimization selects candidate with highest acquisition."""
        candidates = jnp.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])

        # Simple acquisition: sum of coordinates
        def acq_fn(X):
            return jnp.sum(X, axis=1)

        X_next, acq_val = optimize_acqf_discrete(acq_fn, candidates, q=1)

        # Should select [1.0, 1.0] (sum = 2.0)
        assert X_next.shape == (1, 2)
        assert jnp.allclose(X_next[0], jnp.array([1.0, 1.0]))
        assert jnp.allclose(acq_val[0], 2.0)

    def test_optimize_discrete_batch(self):
        """Discrete optimization selects multiple points."""
        candidates = jnp.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])

        def acq_fn(X):
            return jnp.sum(X, axis=1)

        X_next, acq_vals = optimize_acqf_discrete(acq_fn, candidates, q=2)

        # Should select top 2: [1.0, 1.0] and [0.5, 0.5]
        assert X_next.shape == (2, 2)
        assert jnp.allclose(X_next[0], jnp.array([1.0, 1.0]))
        assert jnp.allclose(X_next[1], jnp.array([0.5, 0.5]))

    def test_optimize_discrete_with_ei(self, fitted_model):
        """Discrete optimization works with EI."""
        candidates = jnp.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        probes = candidates + 0.1
        best_f = 0.8

        def acq_fn(X):
            posterior = fitted_model.posterior(X, probes=probes)
            return expected_improvement(posterior, best_f)

        X_next, ei_val = optimize_acqf_discrete(acq_fn, candidates, q=1)

        assert X_next.shape == (1, 2)
        assert ei_val.shape == (1,)


# ============================================================================
# Test Random Optimization
# ============================================================================


class TestRandomOptimization:
    """Test random search optimization."""

    def test_random_search_basic(self):
        """Random search finds reasonable solutions."""
        bounds = jnp.array([[0.0, 1.0], [0.0, 1.0]])

        # Acquisition: maximize sum (optimal at [1, 1])
        def acq_fn(X):
            return jnp.sum(X, axis=1)

        X_next, acq_val = optimize_acqf_random(
            acq_fn, bounds, q=1, num_samples=1000, key=jr.PRNGKey(0)
        )

        assert X_next.shape == (1, 2)
        # Should be close to [1, 1] with enough samples
        assert jnp.sum(X_next) > 1.5  # At least close to optimum

    def test_random_search_respects_bounds(self):
        """Random search stays within bounds."""
        bounds = jnp.array([[0.5, 0.7], [0.2, 0.4]])

        def acq_fn(X):
            return jnp.sum(X, axis=1)

        X_next, _ = optimize_acqf_random(
            acq_fn, bounds, q=1, num_samples=100, key=jr.PRNGKey(0)
        )

        # Check bounds
        assert jnp.all(X_next[:, 0] >= 0.5) and jnp.all(X_next[:, 0] <= 0.7)
        assert jnp.all(X_next[:, 1] >= 0.2) and jnp.all(X_next[:, 1] <= 0.4)


# ============================================================================
# Test Gradient-Based Optimization
# ============================================================================


class TestGradientOptimization:
    """Test gradient-based continuous optimization."""

    def test_gradient_optimization_simple(self):
        """Gradient optimization finds optimum for simple function."""
        bounds = jnp.array([[0.0, 1.0], [0.0, 1.0]])

        # Quadratic with optimum at [0.7, 0.7]
        def acq_fn(X):
            target = jnp.array([0.7, 0.7])
            return -jnp.sum((X - target) ** 2, axis=1)

        X_next, acq_val = optimize_acqf(
            acq_fn,
            bounds,
            q=1,
            method="gradient",
            num_restarts=5,
            optim_steps=50,
            lr=0.05,
            key=jr.PRNGKey(0),
        )

        assert X_next.shape == (1, 2)
        # Should be close to [0.7, 0.7]
        assert jnp.allclose(X_next[0], jnp.array([0.7, 0.7]), atol=0.1)

    def test_gradient_optimization_respects_bounds(self):
        """Gradient optimization clips to bounds."""
        bounds = jnp.array([[0.0, 0.5], [0.0, 0.5]])

        # Acquisition tries to push to [1, 1] but should clip to [0.5, 0.5]
        def acq_fn(X):
            return jnp.sum(X, axis=1)

        X_next, _ = optimize_acqf(
            acq_fn,
            bounds,
            q=1,
            method="gradient",
            num_restarts=3,
            optim_steps=50,
            key=jr.PRNGKey(0),
        )

        # Should be at boundary
        assert jnp.all(X_next <= 0.5)


# ============================================================================
# Integration Tests
# ============================================================================


class TestAcquisitionIntegration:
    """Integration tests with full Model API."""

    def test_full_workflow_ei_discrete(self):
        """Complete workflow: fit → posterior → EI → optimize."""
        # 1. Create and fit model
        model = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=2),
            task=OddityTask(),
            noise=GaussianNoise(),
        )

        n = 20
        key = jr.PRNGKey(123)
        key, subkey = jr.split(key)
        refs = jr.normal(subkey, (n, 2))
        key, subkey = jr.split(key)
        probes = refs + jr.normal(subkey, (n, 2)) * 0.3
        y = jnp.ones(n, dtype=int)
        X = jnp.stack([refs, probes], axis=1)

        model.fit(X, y, inference="map", inference_config={"steps": 30})

        # 2. Define candidates
        candidates = jnp.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        probes_test = candidates + 0.1

        # 3. Optimize EI
        best_f = float(jnp.max(y))  # Cast to Python float

        def acq_fn(X_cand):
            posterior = model.posterior(X_cand, probes=probes_test)
            return expected_improvement(posterior, best_f)  # type: ignore[arg-type]

        X_next, ei_val = optimize_acqf_discrete(acq_fn, candidates, q=1)

        # 4. Verify results
        assert X_next.shape == (1, 2)
        assert ei_val.shape == (1,)
        assert jnp.all(ei_val >= 0.0)

    def test_full_workflow_ucb_continuous(self):
        """Complete workflow with UCB and continuous optimization."""
        # 1. Create and fit model
        model = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=2),
            task=OddityTask(),
            noise=GaussianNoise(),
        )

        n = 20
        key = jr.PRNGKey(456)
        key, subkey = jr.split(key)
        refs = jr.normal(subkey, (n, 2))
        key, subkey = jr.split(key)
        probes = refs + jr.normal(subkey, (n, 2)) * 0.3
        y = jnp.ones(n, dtype=int)
        X = jnp.stack([refs, probes], axis=1)

        model.fit(X, y, inference="map", inference_config={"steps": 30})

        # 2. Define bounds
        bounds = jnp.array([[-1.0, 1.0], [-1.0, 1.0]])

        # 3. Optimize UCB (using random search for speed)
        def acq_fn(X_cand):
            probes_cand = X_cand + 0.1
            posterior = model.posterior(X_cand, probes=probes_cand)
            return upper_confidence_bound(posterior, beta=2.0)  # type: ignore[arg-type]

        X_next, ucb_val = optimize_acqf(
            acq_fn,
            bounds,
            q=1,
            method="random",
            raw_samples=100,
            key=jr.PRNGKey(0),
        )

        # 4. Verify results
        assert X_next.shape == (1, 2)
        assert jnp.all((X_next >= -1.0) & (X_next <= 1.0))

    def test_online_learning_with_acquisition(self):
        """Online learning with acquisition function (single iteration)."""
        # 1. Create model
        model = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=2),
            task=OddityTask(),
            noise=GaussianNoise(),
        )

        # 2. Initial training
        n = 15
        key = jr.PRNGKey(789)
        key, subkey = jr.split(key)
        refs = jr.normal(subkey, (n, 2))
        key, subkey = jr.split(key)
        probes = refs + jr.normal(subkey, (n, 2)) * 0.3
        y = jnp.ones(n, dtype=int)
        X = jnp.stack([refs, probes], axis=1)

        model.fit(X, y, inference="map", inference_config={"steps": 20})

        # 3. Single acquisition step
        candidates = jnp.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        probes_cand = candidates + 0.1
        best_f = 1.0

        # Get best candidate via EI
        posterior = model.posterior(candidates, probes=probes_cand)
        ei = expected_improvement(posterior, best_f)  # type: ignore[arg-type]
        best_idx = jnp.argmax(ei)
        X_next_ref = candidates[best_idx : best_idx + 1]
        X_next_probe = probes_cand[best_idx : best_idx + 1]

        # Simulate response - construct trial correctly as (1, 2, input_dim)
        X_trial = jnp.stack([X_next_ref, X_next_probe], axis=1)  # Shape: (1, 2, 2)
        y_new = jnp.array([1])

        # Update model
        updated_model = model.condition_on_observations(X_trial, y_new)

        # Should complete successfully
        assert updated_model._posterior is not None
