"""
test_mc_likelihood.py
---------------------

Tests for Monte Carlo likelihood computation.

This tests the MC likelihood method in the OddityTask model, which estimates
the log-likelihood of observed responses by simulating internal noisy:
- Sample internal noisy representations from Gaussian with covariance Σ(x)
- Compute Mahalanobis distances for ref vs comparison
- apply logistic smoothing with bandwidth parameter
- Average over MC samples to estimate P(correct)
- we also test for numerical stability and convergence


"""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from psyphy.data.dataset import ResponseData, TrialData
from psyphy.model import WPPM, Prior
from psyphy.model.noise import GaussianNoise
from psyphy.model.task import OddityTask, OddityTaskConfig


class TestMCLikelihood:
    """Test Monte Carlo likelihood computation for oddity task."""

    @pytest.fixture
    def model(self):
        """Create a simple WPPM model for testing.

        MC likelihood works in both modes (MVP and full WPPM).
        """
        return WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=3),
            task=OddityTask(),
            noise=GaussianNoise(sigma=0.03),
        )

    @pytest.fixture
    def simple_params(self, model):
        """Generate simple test parameters."""
        # Initialize with a known seed for reproducibility
        return model.init_params(jr.PRNGKey(42))

    def test_mc_likelihood_method_exists(self, model):
        """Test that OddityTask has a loglik method."""
        assert hasattr(model.task, "loglik"), "OddityTask should have loglik method"

    def test_mc_likelihood_shape_and_dtype(self, model, simple_params):
        """Test MC likelihood returns scalar with correct dtype."""
        data = TrialData(
            refs=jnp.array([[0.0, 0.0]]),
            comparisons=jnp.array([[0.1, 0.1]]),
            responses=jnp.array([1], dtype=jnp.int32),
        )

        # Compute MC likelihood
        ll_mc = model.task.loglik(
            params=simple_params,
            data=data,
            model=model,
            noise=model.noise,
            key=jr.PRNGKey(0),
        )

        # Should be scalar
        assert ll_mc.shape == (), f"Expected scalar, got shape {ll_mc.shape}"
        assert jnp.isfinite(ll_mc), "Log-likelihood should be finite"
        assert ll_mc <= 0.0, "Log-likelihood should be ≤ 0"

    def test_mc_likelihood_convergence_to_analytical(self, model, simple_params):
        """MC likelihood should be in reasonable range compared to analytical.

        Note: MC and analytical may differ significantly because they compute
        different things (MC simulates the full process, analytical uses
        discriminability approximation). This test just verifies both are
        finite and in reasonable ranges.
        """
        # Create trial with nearby comparison (easy discrimination)
        data = ResponseData()
        ref = jnp.array([0.0, 0.0])
        comparison = jnp.array([0.5, 0.5])  # Far enough for clear discrimination
        data = TrialData(
            refs=jnp.array([ref]),
            comparisons=jnp.array([comparison]),
            responses=jnp.array([1], dtype=jnp.int32),
        )

        # Analytical likelihood (current implementation)
        ll_analytical = model.task.loglik(
            params=simple_params, data=data, model=model, noise=model.noise
        )

        # MC likelihood with many samples: create a separate model instance
        # whose task config sets MC fidelity and smoothing.
        mc_model = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=3),
            task=OddityTask(config=OddityTaskConfig(num_samples=5000, bandwidth=1e-3)),
            noise=GaussianNoise(sigma=0.03),
        )
        mc_params = mc_model.init_params(jr.PRNGKey(42))
        ll_mc = mc_model.task.loglik(
            params=mc_params,
            data=data,
            model=mc_model,
            noise=mc_model.noise,
            key=jr.PRNGKey(42),
        )

        # Both should be finite and negative (log probabilities)
        assert jnp.isfinite(ll_analytical), (
            f"Analytical LL should be finite, got {ll_analytical}"
        )
        assert jnp.isfinite(ll_mc), f"MC LL should be finite, got {ll_mc}"
        assert ll_analytical <= 0.0, f"Analytical LL should be ≤ 0, got {ll_analytical}"
        assert ll_mc <= 0.0, f"MC LL should be ≤ 0, got {ll_mc}"

        # Both should be in reasonable range (not too negative)
        assert ll_analytical > -100, f"Analytical LL too negative: {ll_analytical}"
        assert ll_mc > -100, f"MC LL too negative: {ll_mc}"

    def test_mc_likelihood_increases_num_samples(self, model, simple_params):
        """Increasing num_samples should reduce MC variance."""
        model = WPPM(
            input_dim=model.input_dim,
            prior=model.prior,
            task=OddityTask(config=OddityTaskConfig(num_samples=100, bandwidth=1e-2)),
            noise=model.noise,
        )
        data = ResponseData()
        data = TrialData(
            refs=jnp.array([[0.0, 0.0]]),
            comparisons=jnp.array([[0.2, 0.1]]),
            responses=jnp.array([1], dtype=jnp.int32),
        )

        # Compute with different sample sizes
        ll_100 = model.task.loglik(
            params=simple_params,
            data=data,
            model=model,
            noise=model.noise,
            key=jr.PRNGKey(0),
        )

        model_1000 = WPPM(
            input_dim=model.input_dim,
            prior=model.prior,
            task=OddityTask(config=OddityTaskConfig(num_samples=1000, bandwidth=1e-2)),
            noise=model.noise,
        )
        ll_1000 = model_1000.task.loglik(
            params=simple_params,
            data=data,
            model=model_1000,
            noise=model_1000.noise,
            key=jr.PRNGKey(0),
        )

        # both should be finite and negative
        assert jnp.isfinite(ll_100) and jnp.isfinite(ll_1000)
        assert ll_100 <= 0.0 and ll_1000 <= 0.0

        # With same seed, larger sample size should give more stable estimate
        # (This is a weak test - just checking both are reasonable)
        assert jnp.abs(ll_1000 - ll_100) < 2.0, "Estimates should be in same ballpark"

    def test_mc_likelihood_batch_correctness(self, model, simple_params):
        """MC likelihood should handle multiple trials correctly."""
        data = TrialData(
            refs=jnp.array([[0.0, 0.0], [0.5, 0.5], [-0.3, 0.2]]),
            comparisons=jnp.array([[0.1, 0.1], [0.6, 0.4], [-0.2, 0.3]]),
            responses=jnp.array([1, 0, 1], dtype=jnp.int32),
        )

        ll_mc = model.task.loglik(
            params=simple_params,
            data=data,
            model=model,
            noise=model.noise,
            key=jr.PRNGKey(123),
        )

        # Should be sum of log-likelihoods (scalar)
        assert ll_mc.shape == ()
        assert jnp.isfinite(ll_mc)
        assert ll_mc <= 0.0

    @pytest.mark.parametrize("bandwidth", [1e-3, 1e-2, 5e-2])
    def test_mc_likelihood_bandwidth_sensitivity(self, model, simple_params, bandwidth):
        """MC likelihood should be sensitive to bandwidth parameter.

        Bandwidth controls smoothness of logistic CDF approximation.
        Smaller bandwidth -> sharper transitions (closer to step function).
        """
        data = ResponseData()
        data = TrialData(
            refs=jnp.array([[0.0, 0.0]]),
            comparisons=jnp.array([[0.1, 0.1]]),
            responses=jnp.array([1], dtype=jnp.int32),
        )

        model = WPPM(
            input_dim=model.input_dim,
            prior=model.prior,
            task=OddityTask(
                config=OddityTaskConfig(num_samples=1000, bandwidth=bandwidth)
            ),
            noise=model.noise,
        )
        ll_mc = model.task.loglik(
            params=simple_params,
            data=data,
            model=model,
            noise=model.noise,
            key=jr.PRNGKey(0),
        )

        assert jnp.isfinite(ll_mc)
        assert ll_mc <= 0.0

    def test_mc_likelihood_reproducibility(self, model, simple_params):
        """Same seed should give same MC likelihood."""
        model = WPPM(
            input_dim=model.input_dim,
            prior=model.prior,
            task=OddityTask(config=OddityTaskConfig(num_samples=500, bandwidth=1e-2)),
            noise=model.noise,
        )
        data = ResponseData()
        data.add_trial(
            ref=jnp.array([0.0, 0.0]), comparison=jnp.array([0.1, 0.1]), resp=1
        )

        ll_1 = model.task.loglik(
            params=simple_params,
            data=data,
            model=model,
            noise=model.noise,
            key=jr.PRNGKey(42),
        )

        ll_2 = model.task.loglik(
            params=simple_params,
            data=data,
            model=model,
            noise=model.noise,
            key=jr.PRNGKey(42),
        )

        assert jnp.allclose(ll_1, ll_2, rtol=1e-6), "Same seed should give same result"

    def test_mc_likelihood_different_seeds_vary(self, model, simple_params):
        """Different seeds should give slightly different MC estimates."""
        model = WPPM(
            input_dim=model.input_dim,
            prior=model.prior,
            task=OddityTask(config=OddityTaskConfig(num_samples=100, bandwidth=1e-2)),
            noise=model.noise,
        )
        data = ResponseData()
        data.add_trial(
            ref=jnp.array([0.0, 0.0]), comparison=jnp.array([0.1, 0.1]), resp=1
        )

        ll_1 = model.task.loglik(
            params=simple_params,
            data=data,
            model=model,
            noise=model.noise,
            key=jr.PRNGKey(0),
        )

        ll_2 = model.task.loglik(
            params=simple_params,
            data=data,
            model=model,
            noise=model.noise,
            key=jr.PRNGKey(999),
        )

        # Should differ slightly due to MC variance
        assert not jnp.allclose(ll_1, ll_2, rtol=1e-4), "Different seeds should vary"
        # But should be in similar range
        assert jnp.abs(ll_1 - ll_2) < 2.0, "Variance should be reasonable"


class TestMCLikelihoodEdgeCases:
    """Test edge cases and numerical stability."""

    @pytest.fixture
    def model(self):
        return WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=3),
            task=OddityTask(),
            noise=GaussianNoise(sigma=0.03),
        )

    def test_mc_likelihood_wishart_mode(self):
        """Test that MC likelihood works in Wishart mode (basis expansion)."""
        # Create model in Wishart mode
        model = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=3, extra_embedding_dims=0),
            task=OddityTask(config=OddityTaskConfig(num_samples=500, bandwidth=1e-2)),
            noise=GaussianNoise(sigma=0.03),
        )

        # Verify it's actually in Wishart mode
        assert model.basis_degree is not None
        assert model.basis_degree == 3

        # Init params
        params = model.init_params(jr.PRNGKey(42))
        assert "W" in params

        # Create oddity trial data
        data = ResponseData()
        data.add_trial(
            ref=jnp.array([0.0, 0.0]),
            comparison=jnp.array([0.5, 0.0]),
            resp=1,
        )

        # Compute MC likelihood
        loglik = model.task.loglik(
            params=params,
            data=data,
            model=model,
            noise=model.noise,
            key=jr.PRNGKey(123),
        )

        # Basic checks
        assert jnp.isfinite(loglik)
        assert loglik.shape == ()  # scalar shape

        # Should be negative (log probability)
        assert loglik < 0

        # Sanity: should be reasonable log probability
        assert loglik > -10  # not absurdly unlikely

    def test_mc_likelihood_identical_stimuli(self, model):
        """Test MC likelihood when ref == comparison (chance performance expected)."""
        params = model.init_params(jr.PRNGKey(0))
        data = ResponseData()
        # Identical stimuli
        stim = jnp.array([0.5, 0.5])
        data.add_trial(ref=stim, comparison=stim, resp=1)

        model = WPPM(
            input_dim=model.input_dim,
            prior=model.prior,
            task=OddityTask(config=OddityTaskConfig(num_samples=1000, bandwidth=1e-2)),
            noise=model.noise,
        )
        ll_mc = model.task.loglik(
            params=params,
            data=data,
            model=model,
            noise=model.noise,
            key=jr.PRNGKey(42),
        )

        # Should be finite and correspond to near-chance performance
        assert jnp.isfinite(ll_mc)
        # P(correct) ≈ 1/3 for oddity -> log(1/3) ≈ -1.1
        assert ll_mc < -0.5, "Identical stimuli should give low likelihood"

    def test_mc_likelihood_extreme_discriminability(self, model):
        """Test MC likelihood with very far apart stimuli (high discriminability)."""
        params = model.init_params(jr.PRNGKey(0))
        data = ResponseData()
        # Very far apart
        data.add_trial(
            ref=jnp.array([-0.9, -0.9]), comparison=jnp.array([0.9, 0.9]), resp=1
        )

        model = WPPM(
            input_dim=model.input_dim,
            prior=model.prior,
            task=OddityTask(config=OddityTaskConfig(num_samples=1000, bandwidth=1e-2)),
            noise=model.noise,
        )
        ll_mc = model.task.loglik(
            params=params,
            data=data,
            model=model,
            noise=model.noise,
            key=jr.PRNGKey(42),
        )

        # Should be finite and higher than chance performance
        # (not necessarily close to 0, depends on bandwidth and other factors)
        assert jnp.isfinite(ll_mc)
        assert ll_mc > -20, "High discriminability should give reasonable likelihood"

    def test_mc_likelihood_zero_samples_fails(self, model):
        """Test that num_samples=0 is rejected (strict task-owned config)."""
        with pytest.raises(ValueError, match="num_samples must be > 0"):
            _ = OddityTask(config=OddityTaskConfig(num_samples=0, bandwidth=1e-2))


#########
"""
Robustness tests for MC likelihood
The following tests focus on three things that can silently break optimization:
1. Gradient computation (NaN gradients -> optimization fails)
2. Probability clipping (log(0) -> -inf -> NaN gradients)
3. Numerical stability (ill-conditioned matrices -> NaN distances)

"""


class TestGradientCompatibility:
    """
    Test that MC likelihood is compatible with gradient-based optimization.

    --> If gradients are NaN or inf, optimization breaks immediately.
    Once NaN enters the parameter values, it spreads and becomes unrecoverable.
    """

    @pytest.fixture
    def model(self):
        return WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=3),
            task=OddityTask(config=OddityTaskConfig(num_samples=500, bandwidth=1e-2)),
            noise=GaussianNoise(sigma=0.03),
        )

    def test_gradients_are_finite_normal_case(self, model):
        """
        Test that gradients are finite (not NaN, not inf) for typical inputs.

        This is the most basic check: can we compute gradients at all?
        If this fails, nothing else matters - optimization is impossible.
        """
        params = model.init_params(jr.PRNGKey(0))

        # Create simple dataset: one trial with moderate discriminability
        data = ResponseData()
        data.add_trial(
            ref=jnp.array([0.0, 0.0]), comparison=jnp.array([0.5, 0.5]), resp=1
        )

        # Define loss function (negative log-likelihood)
        def loss_fn(p):
            return -model.task.loglik(
                params=p,
                data=data,
                model=model,
                noise=model.noise,
                key=jr.PRNGKey(0),
            )

        # Compute gradient using JAX autodiff
        grad = jax.grad(loss_fn)(params)

        # Check that ALL gradient values are finite (not NaN, not inf)
        # We use tree_map because params is a nested dict structure
        all_finite = jax.tree_util.tree_all(
            jax.tree.map(lambda x: jnp.all(jnp.isfinite(x)), grad)
        )

        assert all_finite, (
            "Gradients contain NaN or inf values - this breaks optimization!"
        )

        # Also check that gradients aren't all zero (vanishing gradient)
        # If all gradients are zero, the model can't learn anything
        grad_norm = jnp.sqrt(
            sum(jnp.sum(x**2) for x in jax.tree_util.tree_leaves(grad))
        )
        assert grad_norm > 1e-6, (
            f"Gradient norm too small ({grad_norm}) - model can't learn"
        )

    def test_gradients_finite_with_identical_stimuli(self, model):
        """
        Test gradients when ref == comparison (chance performance).

        This is an important edge case: when stimuli are identical, P(correct) ≈ 1/3.
        The log-likelihood should be around log(1/3) ≈ -1.1, and gradients should
        still be finite even though the model can't really learn from this trial.
        """
        params = model.init_params(jr.PRNGKey(0))

        data = ResponseData()
        # Identical stimuli - observer is just guessing
        data.add_trial(
            ref=jnp.array([0.0, 0.0]), comparison=jnp.array([0.0, 0.0]), resp=1
        )

        def loss_fn(p):
            return -model.task.loglik(
                params=p,
                data=data,
                model=model,
                noise=model.noise,
                key=jr.PRNGKey(0),
            )

        grad = jax.grad(loss_fn)(params)

        # Even with no discriminability, gradients should be finite
        all_finite = jax.tree_util.tree_all(
            jax.tree.map(lambda x: jnp.all(jnp.isfinite(x)), grad)
        )

        assert all_finite, "Gradients are NaN/inf with identical stimuli!"

    def test_gradients_finite_with_extreme_discriminability(self, model):
        """
        Test gradients when stimuli are very far apart (near-perfect performance).

        When ref and comparison are very different, P(correct) -> 1.0.
        Without proper clipping, this would give log(1.0) = 0, but with clipping
        we get log(1 - eps) ≈ -1e-8. Gradients should still be finite but very small.
        """
        params = model.init_params(jr.PRNGKey(0))

        data = ResponseData()
        # Very far apart stimuli - near perfect discrimination
        data.add_trial(
            ref=jnp.array([0.0, 0.0]), comparison=jnp.array([5.0, 5.0]), resp=1
        )

        def loss_fn(p):
            return -model.task.loglik(
                params=p,
                data=data,
                model=model,
                noise=model.noise,
                key=jr.PRNGKey(0),
            )

        grad = jax.grad(loss_fn)(params)

        # With extreme discriminability, gradients should be finite (though possibly small)
        all_finite = jax.tree_util.tree_all(
            jax.tree.map(lambda x: jnp.all(jnp.isfinite(x)), grad)
        )

        assert all_finite, "Gradients are NaN/inf with extreme discriminability!"

    def test_gradients_finite_with_small_bandwidth(self, model):
        """
        Test gradients with very small bandwidth (sharp decision boundary).

        Small bandwidth makes the decision more deterministic (step function-like).
        This can cause numerical issues because the sigmoid becomes very steep.
        We need to verify gradients stay finite even with bandwidth = 1e-4.
        """
        model = WPPM(
            input_dim=model.input_dim,
            prior=model.prior,
            task=OddityTask(config=OddityTaskConfig(num_samples=500, bandwidth=1e-4)),
            noise=model.noise,
        )
        params = model.init_params(jr.PRNGKey(0))

        data = ResponseData()
        data.add_trial(
            ref=jnp.array([0.0, 0.0]), comparison=jnp.array([0.3, 0.3]), resp=1
        )

        def loss_fn(p):
            return -model.task.loglik(
                params=p,
                data=data,
                model=model,
                noise=model.noise,
                key=jr.PRNGKey(0),
            )

        grad = jax.grad(loss_fn)(params)

        all_finite = jax.tree_util.tree_all(
            jax.tree.map(lambda x: jnp.all(jnp.isfinite(x)), grad)
        )

        assert all_finite, "Gradients are NaN/inf with small bandwidth!"


class TestProbabilityClipping:
    """
    Test that probabilities are properly clipped to avoid log(0) catastrophes.

    Why this matters: The MC likelihood computes log(P(correct)). If P is exactly
    0 or 1, we get log(0) = -inf or log(1) = 0. When we take gradients of -inf,
    we get NaN. The solution is to clip: P ∈ [eps, 1-eps] where eps = 1e-9.
    """

    @pytest.fixture
    def model(self):
        """Model with very low noise for testing extreme cases."""
        return WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=3),
            task=OddityTask(config=OddityTaskConfig(num_samples=1000, bandwidth=1e-3)),
            noise=GaussianNoise(sigma=0.001),  # Very low noise -> sharper decisions
        )

    def test_log_likelihood_finite_at_chance(self, model):
        """
        Test that log-likelihood is finite when performance is at chance.

        When ref == comparison, P(correct) should be around 1/3.
        Without clipping, if MC estimate happens to be exactly 0, log(0) = -inf.
        We need log-likelihood to stay finite even at chance.
        """
        params = model.init_params(jr.PRNGKey(0))

        data = ResponseData()
        # Identical stimuli -> chance performance
        data.add_trial(
            ref=jnp.array([0.0, 0.0]), comparison=jnp.array([0.0, 0.0]), resp=1
        )

        ll = model.task.loglik(
            params=params,
            data=data,
            model=model,
            noise=model.noise,
            key=jr.PRNGKey(0),
        )

        # Log-likelihood should be finite (not -inf, not NaN)
        assert jnp.isfinite(ll), f"Log-likelihood is {ll} (should be finite!)"

        # For chance performance, expect ll ≈ log(1/3) ≈ -1.1
        # Allow some MC variance though
        assert -2.0 < ll < -0.5, (
            f"Log-likelihood {ll} seems wrong for chance performance"
        )

    def test_log_likelihood_finite_at_perfect(self, model):
        """
        Test that log-likelihood is finite when performance is near perfect.

        When ref and comparison are very different, P(correct) -> 1.0.
        Without clipping to (1 - eps), we'd get log(1.0) = 0 (seems fine) but
        actually the raw MC probability can be exactly 1.0, and then we get
        issues with gradients. Clipping ensures log(1 - eps) ≈ -1e-8.
        """
        params = model.init_params(jr.PRNGKey(0))

        data = ResponseData()
        # Very far apart -> near perfect discrimination
        data.add_trial(
            ref=jnp.array([0.0, 0.0]), comparison=jnp.array([10.0, 10.0]), resp=1
        )

        ll = model.task.loglik(
            params=params,
            data=data,
            model=model,
            noise=model.noise,
            key=jr.PRNGKey(0),
        )

        # Should be finite (not -inf, not NaN)
        assert jnp.isfinite(ll), f"Log-likelihood is {ll} (should be finite!)"

        # For near-perfect performance, expect ll ≈ log(1 - eps) ≈ 0 (but slightly negative)
        # Should be very close to 0 but not exactly 0
        assert -1.0 < ll < 0.0, (
            f"Log-likelihood {ll} seems wrong for perfect performance"
        )

    def test_probability_never_exactly_zero_or_one(self, model):
        """
        Test that the implied probability stays in valid range [eps, 1-eps].

        This test checks the actual probability values (not just log-likelihood).
        For a single trial with resp=1, we can recover P(correct) from the
        log-likelihood: P = exp(ll). This should never be exactly 0 or 1.
        """
        params = model.init_params(jr.PRNGKey(0))

        # Test both extreme cases
        test_cases = [
            (jnp.array([0.0, 0.0]), jnp.array([0.0, 0.0]), "identical"),
            (jnp.array([0.0, 0.0]), jnp.array([10.0, 10.0]), "far apart"),
        ]

        eps = 1e-9  # Expected clipping threshold

        for ref, comp, description in test_cases:
            data = ResponseData()
            data.add_trial(ref=ref, comparison=comp, resp=1)

            ll = model.task.loglik(
                params=params,
                data=data,
                model=model,
                noise=model.noise,
                key=jr.PRNGKey(0),
            )

            # For single trial with resp=1: P(correct) = exp(ll)
            p_correct = jnp.exp(ll)

            # Check that probability is in valid range
            assert eps <= p_correct <= (1 - eps), (
                f"P(correct) = {p_correct} outside [{eps}, {1 - eps}] for {description} stimuli"
            )


class TestNumericalStability:
    """
    Test numerical stability of Mahalanobis distance computation.

    Why this matters: We compute distances as d^2 = diff @ Σ^{-1} @ diff. If the
    covariance matrix Σ is nearly singular (very small eigenvalues), the inverse
    becomes unstable. We use solve() instead of inv() to improve stability, but
    we still need to test edge cases with ill-conditioned matrices.
    """

    def test_stability_with_very_small_noise(self):
        """
        Test with extremely small noise (nearly singular covariance).

        When sigma is very small (like 1e-6), the covariance matrix becomes
        nearly singular. This is a stress test for the Mahalanobis distance
        computation using jnp.linalg.solve(). Should not return NaN or inf.
        """
        model = WPPM(
            input_dim=3,
            prior=Prior(input_dim=3, basis_degree=3),
            task=OddityTask(config=OddityTaskConfig(num_samples=500, bandwidth=1e-2)),
            noise=GaussianNoise(sigma=1e-6),  # Tiny noise -> nearly singular Σ
        )

        params = model.init_params(jr.PRNGKey(0))

        data = ResponseData()
        data.add_trial(
            ref=jnp.array([0.0, 0.0, 0.0]),
            comparison=jnp.array([0.1, 0.1, 0.1]),
            resp=1,
        )

        # This should not crash or return NaN
        ll = model.task.loglik(
            params=params,
            data=data,
            model=model,
            noise=model.noise,
            key=jr.PRNGKey(0),
        )

        assert jnp.isfinite(ll), (
            f"Log-likelihood is {ll} (NaN or inf) with tiny noise! Mahalanobis computation unstable."
        )

    def test_stability_with_wishart_mode(self):
        """
        Test numerical stability in Wishart mode (low-rank + diagonal).

        Wishart mode uses manual sampling: z = n @ U.T + mean + sqrt(σ^2) * n_diag.
        This is slightly different from multivariate_normal and could have
        different numerical properties. Test that it's also stable.
        """
        model = WPPM(
            input_dim=3,
            prior=Prior(input_dim=3, extra_embedding_dims=2, basis_degree=3),
            task=OddityTask(config=OddityTaskConfig(num_samples=500, bandwidth=1e-2)),
            noise=GaussianNoise(sigma=0.001),  # Small noise
        )

        params = model.init_params(jr.PRNGKey(0))

        data = ResponseData()
        data.add_trial(
            ref=jnp.array([0.0, 0.0, 0.0]),
            comparison=jnp.array([0.5, 0.5, 0.5]),
            resp=1,
        )

        ll = model.task.loglik(
            params=params,
            data=data,
            model=model,
            noise=model.noise,
            key=jr.PRNGKey(0),
        )

        assert jnp.isfinite(ll), (
            f"Log-likelihood is {ll} (NaN or inf) in Wishart mode with small noise!"
        )


class TestConvergenceRate:
    """
    Test that MC variance decreases at the correct rate (Central Limit Theorem).

    Why this matters: We need to know how many samples to use. If variance
    decreases as 1/√N (as it should by CLT), we know that 4x more samples
    gives 2x more precision. This helps us choose num_samples wisely.
    """

    @pytest.fixture
    def model(self):
        """Simple model for convergence testing."""
        return WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=3),
            task=OddityTask(config=OddityTaskConfig(num_samples=100, bandwidth=1e-2)),
            noise=GaussianNoise(sigma=0.03),
        )

    def test_variance_decreases_with_more_samples(self, model):
        """
        Test that MC variance decreases as we increase num_samples.

        By the Central Limit Theorem, variance should decrease as 1/N.
        We don't need exact 1/√N scaling, but variance with N=1600 samples
        should be noticeably smaller than with N=100 samples.
        """
        params = model.init_params(jr.PRNGKey(0))

        data = ResponseData()
        # Use intermediate discriminability for variance (not too easy, not too hard)
        data.add_trial(
            ref=jnp.array([0.0, 0.0]), comparison=jnp.array([0.3, 0.3]), resp=1
        )

        # Test with small and large sample sizes
        sample_sizes = [100, 1600]  # 16x difference
        variances = []

        for n_samples in sample_sizes:
            # Configure MC fidelity via task config (strict API).
            model_n = WPPM(
                input_dim=model.input_dim,
                prior=model.prior,
                task=OddityTask(
                    config=OddityTaskConfig(num_samples=n_samples, bandwidth=1e-2)
                ),
                noise=model.noise,
            )

            # Compute MC likelihood multiple times with different seeds
            lls = []
            for seed in range(30):  # 30 independent estimates
                ll = model_n.task.loglik(
                    params=params,
                    data=data,
                    model=model_n,
                    noise=model_n.noise,
                    key=jr.PRNGKey(seed),
                )
                lls.append(ll)

            variances.append(jnp.var(jnp.array(lls)))

        # Variance with more samples should be smaller
        var_100, var_1600 = variances

        # With 16x more samples, variance should decrease significantly
        # (by theory, should be ~4x smaller, but we'll be lenient)
        assert var_1600 < var_100 * 0.5, (
            f"Variance didn't decrease enough: {var_100:.6f} -> {var_1600:.6f} (expected >2x reduction)"
        )


if __name__ == "__main__":
    # Can run this file directly for quick testing
    pytest.main([__file__, "-v"])
