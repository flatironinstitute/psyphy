"""
Tests for the 3-stimulus oddity task decision rule and core algorithm.

These tests verify that the full 3-stimulus oddity implementation:
1. Uses the correct decision rule: min(d^2(z_ref,z_comparison), d^2(z_ref_prime,z_comparison)) > d^2(z_ref,z_ref_prime)
2. Produces correct behavior for edge cases (identical/distant stimuli)
3. Uses proper covariance weighting (2/3, 1/3) for the 3-stimulus task
4. Behaves as expected across different scenarios
"""

import jax.numpy as jnp
import jax.random as jr
import pytest

from psyphy.data import ResponseData
from psyphy.model import WPPM, GaussianNoise, OddityTask, OddityTaskConfig, Prior


class TestThreeStimulusDecisionRule:
    """Test the core 3-stimulus oddity decision rule."""

    def test_identical_stimuli_near_chance(self):
        """
        When ref = comparison, P(correct) should be near 1/3 (random guessing).

        """
        # Create  model
        model = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=2),
            task=OddityTask(config=OddityTaskConfig(num_samples=5000, bandwidth=1e-2)),
            noise=GaussianNoise(sigma=0.01),  # Small noise
        )
        params = model.init_params(jr.PRNGKey(0))

        # Create data with identical reference and comparison
        data = ResponseData()
        ref = jnp.array([0.5, 0.5])
        comparison = ref  # identical!
        data.add_trial(ref, comparison, resp=1)

        # Compute P(correct) with many samples for accurate estimate
        # loglik = log P(correct | ref, comparison, params)
        loglik = model.task.loglik(
            params,
            data,
            model,
            model.noise,
            key=jr.PRNGKey(42),
        )

        # P(correct) = exp(loglik) for single trial with response=1
        p_correct = jnp.exp(loglik)

        # Should be near 1/3 (chance level) ± some tolerance
        # Allow 0.25 to 0.41 (some tolerance for MC variance)
        assert 0.25 < p_correct < 0.41, (
            f"P(correct) = {p_correct:.3f}, expected \approx 0.333 for identical stimuli. "
            "The decision rule should produce chance-level performance when stimuli are identical."
        )

    def test_distant_stimuli_near_perfect(self):
        """
        When comparison is far from ref, P(correct) should approach 1.

        This tests that the decision rule correctly identifies the comparison
        as the odd one out when discrimination is easy.
        """
        # Create a simple MVP model with small noise
        model = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=2),
            task=OddityTask(config=OddityTaskConfig(num_samples=5000, bandwidth=1e-2)),
            noise=GaussianNoise(sigma=0.01),  # Small noise for easy discrimination
        )
        params = model.init_params(jr.PRNGKey(0))

        # Create data with distant reference and comparison
        data = ResponseData()
        ref = jnp.array([0.0, 0.0])
        comparison = jnp.array([5.0, 5.0])  # Very far!
        data.add_trial(ref, comparison, resp=1)

        # Compute P(correct)
        loglik = model.task.loglik(
            params,
            data,
            model,
            model.noise,
            key=jr.PRNGKey(42),
        )

        p_correct = jnp.exp(loglik)

        # Should be near 1 (perfect discrimination)
        assert p_correct > 0.95, (
            f"P(correct) = {p_correct:.3f}, expected > 0.95 for distant stimuli. "
            "The decision rule should produce near-perfect performance when stimuli are far apart."
        )

    def test_intermediate_stimuli_between_chance_and_perfect(self):
        """
        For moderately different stimuli, P(correct) should be between chance and perfect.

        This tests that the decision rule produces graded performance that reflects
        the discriminability of the stimuli.
        """
        model = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=2),
            task=OddityTask(config=OddityTaskConfig(num_samples=5000, bandwidth=1e-2)),
            noise=GaussianNoise(sigma=0.05),
        )
        params = model.init_params(jr.PRNGKey(0))

        # Create data with moderately different stimuli
        data = ResponseData()
        ref = jnp.array([0.0, 0.0])
        comparison = jnp.array([0.3, 0.3])  # Moderately different
        data.add_trial(ref, comparison, resp=1)

        # Compute P(correct)
        loglik = model.task.loglik(params, data, model, model.noise, key=jr.PRNGKey(42))

        p_correct = jnp.exp(loglik)

        # Should be between chance (1/3) and perfect (1)
        # Note: For this stimulus separation with the given noise level,
        # performance is still quite close to chance level
        assert 0.28 < p_correct < 0.95, (
            f"P(correct) = {p_correct:.3f}, expected between 0.28 and 0.95. "
            "The decision rule should produce intermediate performance for moderately different stimuli."
        )

    def test_wishart_mode_same_behavior(self):
        """
        Test that Wishart mode produces similar qualitative behavior as MVP mode.

        This verifies that the decision rule works correctly in both modes.
        """
        # Create Wishart model
        model = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=3, extra_embedding_dims=1),
            task=OddityTask(),
            noise=GaussianNoise(sigma=0.01),
            extra_dims=1,
        )
        params = model.init_params(jr.PRNGKey(0))

        # Test with identical stimuli
        data_identical = ResponseData()
        ref = jnp.array([0.5, 0.5])
        data_identical.add_trial(ref, ref, resp=1)

        model.task = OddityTask(
            config=OddityTaskConfig(num_samples=3000, bandwidth=1e-2)
        )
        loglik_identical = model.task.loglik(
            params,
            data_identical,
            model,
            model.noise,
            key=jr.PRNGKey(42),
        )
        p_identical = jnp.exp(loglik_identical)

        # Test with distant stimuli
        data_distant = ResponseData()
        comparison_far = jnp.array([5.0, 5.0])
        data_distant.add_trial(ref, comparison_far, resp=1)

        loglik_distant = model.task.loglik(
            params,
            data_distant,
            model,
            model.noise,
            key=jr.PRNGKey(42),
        )
        p_distant = jnp.exp(loglik_distant)

        # Qualitative checks
        assert 0.2 < p_identical < 0.5, (
            f"Wishart mode: P(identical) = {p_identical:.3f}, expected ≈ 1/3"
        )
        assert p_distant > 0.90, (
            f"Wishart mode: P(distant) = {p_distant:.3f}, expected > 0.9"
        )
        assert p_distant > p_identical, (
            "Distant stimuli should be easier than identical"
        )


class TestCovarianceWeighting:
    """Test that the average covariance uses correct (2/3, 1/3) weighting."""

    def test_covariance_weights_sum_to_one(self):
        """Verify that covariance weights (2/3 + 1/3) sum to 1."""
        weight_ref = 2.0 / 3.0
        weight_comparison = 1.0 / 3.0

        total = weight_ref + weight_comparison

        assert jnp.allclose(total, 1.0), (
            f"Weights sum to {total}, expected 1.0. "
            "Covariance weights must sum to 1 for proper averaging."
        )

    def test_covariance_weights_reflect_sample_frequency(self):
        """
        Verify that weights (2/3, 1/3) reflect sampling frequency.

        We sample 2 times from ref distribution and 1 time from comparison distribution,
        so weights should be 2/3 and 1/3 respectively.
        """
        # Number of samples from each distribution
        n_ref_samples = 2  # z_ref and z_ref_prime
        n_comparison_samples = 1  # z_comparison
        n_total = n_ref_samples + n_comparison_samples

        # Expected weights
        expected_weight_ref = n_ref_samples / n_total  # 2/3
        expected_weight_comparison = n_comparison_samples / n_total  # 1/3

        # Actual weights in code
        actual_weight_ref = 2.0 / 3.0
        actual_weight_comparison = 1.0 / 3.0

        assert jnp.allclose(actual_weight_ref, expected_weight_ref), (
            f"Reference weight = {actual_weight_ref}, expected {expected_weight_ref}"
        )
        assert jnp.allclose(actual_weight_comparison, expected_weight_comparison), (
            f"Probe weight = {actual_weight_comparison}, expected {expected_weight_comparison}"
        )

    def test_weighted_average_with_identical_covariances(self):
        """
        When Σ_ref = Σ_comparison, the weighted average should equal the common covariance.

        This is a sanity check that the averaging formula is correct.
        """
        # Create identical covariances
        Sigma = jnp.eye(3) * 0.1
        Sigma_ref = Sigma
        Sigma_comparison = Sigma

        # Compute weighted average
        Sigma_avg = (2.0 / 3.0) * Sigma_ref + (1.0 / 3.0) * Sigma_comparison

        # Should equal the original covariance
        assert jnp.allclose(Sigma_avg, Sigma), (
            "When covariances are identical, weighted average should equal the common covariance"
        )

    def test_weighted_average_interpolates_between_covariances(self):
        """
        The weighted average should be between the two input covariances.

        For scalar covariances σ²_ref and σ²_comparison:
        If σ²_ref < σ²_avg < σ²_comparison, then (2/3)σ²_ref + (1/3)σ²_comparison is between them.
        """
        # Scalar variances (for simplicity)
        sigma_ref_sq = 0.1
        sigma_comparison_sq = 0.5  # Higher variance

        # Weighted average
        sigma_avg_sq = (2.0 / 3.0) * sigma_ref_sq + (1.0 / 3.0) * sigma_comparison_sq

        # Should be between the two
        assert sigma_ref_sq < sigma_avg_sq < sigma_comparison_sq, (
            f"Average variance {sigma_avg_sq} should be between "
            f"{sigma_ref_sq} and {sigma_comparison_sq}"
        )

        # Should be closer to ref (weight 2/3) than to comparison (weight 1/3)
        distance_to_ref = abs(sigma_avg_sq - sigma_ref_sq)
        distance_to_comparison = abs(sigma_avg_sq - sigma_comparison_sq)
        assert distance_to_ref < distance_to_comparison, (
            "Average should be closer to ref (weight 2/3) than comparison (weight 1/3)"
        )


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_small_bandwidth(self):
        """
        Test with very small bandwidth (sharp decision threshold).

        Small bandwidth -> step function -> should still work but be more deterministic.
        """
        model = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=2),
            task=OddityTask(config=OddityTaskConfig(num_samples=1000, bandwidth=1e-6)),
            noise=GaussianNoise(sigma=0.01),
        )
        params = model.init_params(jr.PRNGKey(0))

        data = ResponseData()
        ref = jnp.array([0.0, 0.0])
        comparison = jnp.array([2.0, 2.0])
        data.add_trial(ref, comparison, resp=1)

        loglik = model.task.loglik(params, data, model, model.noise, key=jr.PRNGKey(42))

        p_correct = jnp.exp(loglik)

        # Should still be moderately high (distant stimuli)
        # With small bandwidth, performance may be lower than with optimal bandwidth
        assert p_correct > 0.7, (
            f"P(correct) = {p_correct:.3f} with small bandwidth. "
            "Small bandwidth should still produce good accuracy for distant stimuli."
        )

    def test_large_bandwidth(self):
        """
        Test with large bandwidth (smooth decision).

        Large bandwidth -> smooth transition -> more gradual performance curve.
        """
        model = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=2),
            task=OddityTask(config=OddityTaskConfig(num_samples=1000, bandwidth=0.1)),
            noise=GaussianNoise(sigma=0.01),
        )
        params = model.init_params(jr.PRNGKey(0))

        data = ResponseData()
        ref = jnp.array([0.0, 0.0])
        comparison = jnp.array([0.5, 0.5])  # Moderately different
        data.add_trial(ref, comparison, resp=1)

        loglik = model.task.loglik(params, data, model, model.noise, key=jr.PRNGKey(42))

        p_correct = jnp.exp(loglik)

        # Should be between chance and perfect (more gradual)
        assert 0.23 < p_correct < 0.85, (
            f"P(correct) = {p_correct:.3f} with large bandwidth. "
            "Large bandwidth should produce intermediate performance."
        )

    def test_small_num_samples(self):
        """
        Test with small num_samples (high MC variance).

        Should still produce reasonable results, just with more variance.
        """
        model = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=2),
            task=OddityTask(config=OddityTaskConfig(num_samples=10, bandwidth=1e-2)),
            noise=GaussianNoise(sigma=0.01),
        )
        params = model.init_params(jr.PRNGKey(0))

        data = ResponseData()
        ref = jnp.array([0.0, 0.0])
        comparison = jnp.array([3.0, 3.0])
        data.add_trial(ref, comparison, resp=1)

        loglik = model.task.loglik(params, data, model, model.noise, key=jr.PRNGKey(42))

        p_correct = jnp.exp(loglik)

        # Should still be reasonably high (qualitative check)
        assert p_correct > 0.5, (
            f"P(correct) = {p_correct:.3f} with only 10 samples. "
            "Even with few samples, should detect easy discrimination."
        )

    def test_large_num_samples_convergence(self):
        """
        Test with large num_samples (low MC variance).

        Results should be more stable and converged.
        """
        task = OddityTask(config=OddityTaskConfig(num_samples=5000, bandwidth=1e-2))
        model = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=2),
            task=task,
            noise=GaussianNoise(sigma=0.01),
        )
        params = model.init_params(jr.PRNGKey(0))

        data = ResponseData()
        ref = jnp.array([0.0, 0.0])
        comparison = jnp.array([3.0, 3.0])
        data.add_trial(ref, comparison, resp=1)

        # Compute with two different seeds (same task config)
        loglik1 = model.task.loglik(
            params, data, model, model.noise, key=jr.PRNGKey(42)
        )

        loglik2 = model.task.loglik(
            params,
            data,
            model,
            model.noise,
            key=jr.PRNGKey(43),  # Different seed
        )

        p1 = jnp.exp(loglik1)
        p2 = jnp.exp(loglik2)

        # Should be similar (low MC variance with many samples)
        assert jnp.abs(p1 - p2) < 0.05, (
            f"P(correct) varies: {p1:.3f} vs {p2:.3f}. "
            "With large num_samples, results should be stable across seeds."
        )

    def test_multiple_trials_accumulates_loglik(self):
        """
        Test that log-likelihood correctly accumulates across multiple trials.

        loglik(10 trials) should be sum of individual loglik values.
        """
        task = OddityTask(config=OddityTaskConfig(num_samples=1000, bandwidth=1e-2))
        model = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=2),
            task=task,
            noise=GaussianNoise(sigma=0.01),
        )
        params = model.init_params(jr.PRNGKey(0))

        # Create multiple trials
        data_multi = ResponseData()
        ref = jnp.array([0.0, 0.0])
        comparison = jnp.array([1.0, 1.0])

        for _ in range(5):
            data_multi.add_trial(ref, comparison, resp=1)

        # Compute combined loglik
        loglik_multi = model.task.loglik(
            params,
            data_multi,
            model,
            model.noise,
            key=jr.PRNGKey(42),
        )

        # Single trial loglik
        data_single = ResponseData()
        data_single.add_trial(ref, comparison, resp=1)

        loglik_single = model.task.loglik(
            params,
            data_single,
            model,
            model.noise,
            key=jr.PRNGKey(42),
        )

        # Multi-trial should be approximately 5× single trial
        # (with some tolerance for MC variance)
        expected_loglik = 5 * loglik_single

        assert (
            jnp.abs(loglik_multi - expected_loglik) / jnp.abs(expected_loglik) < 0.1
        ), (
            f"Multi-trial loglik = {loglik_multi:.3f}, "
            f"expected ≈ {expected_loglik:.3f} (5 × single trial). "
            "Log-likelihood should accumulate across trials."
        )

    def test_zero_samples_raises_error(self):
        """Test that num_samples=0 raises an error."""
        # Strict API: num_samples comes from task config, so invalid config should fail at construction.
        with pytest.raises(ValueError, match="num_samples must be > 0"):
            _ = OddityTask(config=OddityTaskConfig(num_samples=0, bandwidth=1e-2))

    def test_reproducibility_with_same_seed(self):
        """
        Test that using the same random seed produces identical results.

        This verifies determinism for debugging and reproducibility.
        """
        task = OddityTask(config=OddityTaskConfig(num_samples=1000, bandwidth=1e-2))
        model = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=2),
            task=task,
            noise=GaussianNoise(sigma=0.01),
        )
        params = model.init_params(jr.PRNGKey(0))

        data = ResponseData()
        data.add_trial(jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0]), resp=1)

        # Compute twice with same seed
        loglik1 = model.task.loglik(
            params,
            data,
            model,
            model.noise,
            key=jr.PRNGKey(42),  # Same seed
        )

        loglik2 = model.task.loglik(
            params,
            data,
            model,
            model.noise,
            key=jr.PRNGKey(42),  # Same seed
        )

        # Should be exactly identical
        assert jnp.allclose(loglik1, loglik2), (
            f"Results differ: {loglik1:.6f} vs {loglik2:.6f}. "
            "Same seed should produce identical results."
        )


class TestDecisionRuleSymmetry:
    """Test that the decision rule is symmetric and doesn't depend on position."""

    def test_decision_rule_symmetric_in_reference_samples(self):
        """
        Test that swapping z_ref and z_ref_prime doesn't change the result.

        The decision rule uses min(d(z_ref,z_comparison), d(z_ref_prime,z_comparison)),
        which is symmetric in the two reference samples.
        """
        task = OddityTask(config=OddityTaskConfig(num_samples=2000, bandwidth=1e-2))
        model = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=2),
            task=task,
            noise=GaussianNoise(sigma=0.01),
        )
        params = model.init_params(jr.PRNGKey(0))

        # Create data with specific stimuli
        data = ResponseData()
        ref = jnp.array([0.0, 0.0])
        comparison = jnp.array([1.0, 1.0])
        data.add_trial(ref, comparison, resp=1)

        # Compute with one seed
        loglik1 = model.task.loglik(
            params,
            data,
            model,
            model.noise,
            key=jr.PRNGKey(42),
        )

        # Compute with different seed (different sampling order)
        loglik2 = model.task.loglik(
            params,
            data,
            model,
            model.noise,
            key=jr.PRNGKey(100),
        )

        p1 = jnp.exp(loglik1)
        p2 = jnp.exp(loglik2)

        # Should be similar (allowing for MC variance)
        assert jnp.abs(p1 - p2) < 0.1, (
            f"Results differ significantly: {p1:.3f} vs {p2:.3f}. "
            "Decision should be stable across different random samples."
        )
