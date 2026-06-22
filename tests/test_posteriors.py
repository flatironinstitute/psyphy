"""
test_posteriors.py
-----------------

Tests for the two-tier posterior design:
- ParameterPosterior protocol and implementations
- PredictivePosterior protocol and implementations
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from psyphy.data import TrialData
from psyphy.inference import MAPOptimizer
from psyphy.model import WPPM, ContinuousTouchTask, GaussianNoise, OddityTask, Prior
from psyphy.posterior import (
    MAPPosterior,
    ParameterPosterior,
    PredictivePosterior,
    WPPMPredictivePosterior,
)


class TestParameterPosterior:
    """Test ParameterPosterior protocol and MAPPosterior implementation."""

    @pytest.fixture
    def oddity_model(self):
        """Create a simple WPPM model with an oddity likelihood."""
        return WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=3),
            likelihood=OddityTask(),
            noise=GaussianNoise(),
        )

    @pytest.fixture
    def touch_model(self):
        """Create a simple WPPM model with a touch task likelihood."""
        return WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=3),
            likelihood=ContinuousTouchTask(),
            noise=GaussianNoise(),
        )

    @pytest.fixture
    def oddity_data(self):
        """Create dummy response data."""

        refs = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        comparisons = jnp.array([[0.5, 0.5], [1.5, 1.0]])
        responses = jnp.array([1, 0], dtype=jnp.int32)
        return TrialData(
            stimuli=jnp.stack([refs, comparisons], axis=1), responses=responses
        )

    @pytest.fixture
    def touch_data(self):
        """Create dummy response data."""

        stims = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        responses = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        return TrialData(stimuli=jnp.expand_dims(stims, axis=1), responses=responses)

    @pytest.fixture
    def oddity_param_posterior(self, oddity_model, oddity_data):
        """Fit model and return ParameterPosterior."""
        optimizer = MAPOptimizer(steps=10)  # Few steps for speed
        return optimizer.fit(oddity_model, oddity_data)

    @pytest.fixture
    def touch_param_posterior(self, touch_model, touch_data):
        """Fit model and return ParameterPosterior."""
        optimizer = MAPOptimizer(steps=10)  # Few steps for speed
        return optimizer.fit(touch_model, touch_data)

    def test_map_posterior_is_parameter_posterior(
        self, oddity_param_posterior, touch_param_posterior
    ):
        """MAPPosterior implements ParameterPosterior protocol."""
        assert isinstance(oddity_param_posterior, ParameterPosterior)
        assert isinstance(oddity_param_posterior, MAPPosterior)

        assert isinstance(touch_param_posterior, ParameterPosterior)
        assert isinstance(touch_param_posterior, MAPPosterior)

    def test_params_property(self, oddity_param_posterior, touch_param_posterior):
        """params property returns parameter dict."""
        oddity_params = oddity_param_posterior.params
        assert isinstance(oddity_params, dict)

        touch_params = touch_param_posterior.params
        assert isinstance(touch_params, dict)

    def test_model_property(self, oddity_param_posterior, touch_param_posterior):
        """model property returns associated model."""
        oddity_model = oddity_param_posterior.model
        touch_model = touch_param_posterior.model
        assert isinstance(oddity_model, WPPM)
        assert isinstance(touch_model, WPPM)
        assert oddity_model.input_dim == 2
        assert touch_model.input_dim == 2

    def test_sample_with_key(self, oddity_param_posterior, touch_param_posterior):
        """Sample from MAP posterior returns identical replicates."""
        n_samples = 3
        key = jr.PRNGKey(0)
        oddity_samples = oddity_param_posterior.sample(n=n_samples, key=key)
        touch_samples = touch_param_posterior.sample(n=n_samples, key=key)

        assert isinstance(oddity_samples, dict)
        assert "W" in oddity_samples
        assert oddity_samples["W"].shape[0] == n_samples

        assert isinstance(touch_samples, dict)
        assert "W" in touch_samples
        assert touch_samples["W"].shape[0] == n_samples

        # All samples should be identical to MAP estimate
        oddity_map_params = oddity_param_posterior.params
        for i in range(n_samples):
            assert jnp.allclose(oddity_samples["W"][i], oddity_map_params["W"])

        touch_map_params = touch_param_posterior.params
        for i in range(n_samples):
            assert jnp.allclose(touch_samples["W"][i], touch_map_params["W"])


class TestPredictivePosterior:
    """Test PredictivePosterior protocol and WPPMPredictivePosterior implementation."""

    @pytest.fixture
    def oddity_model(self):
        """Create a simple WPPM model."""
        return WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=3),
            likelihood=OddityTask(),
            noise=GaussianNoise(),
        )

    @pytest.fixture
    def touch_model(self):
        """Create a simple WPPM model."""
        return WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=3),
            likelihood=ContinuousTouchTask(),
            noise=GaussianNoise(),
        )

    @pytest.fixture
    def oddity_data(self):
        """Create dummy response data for oddity task."""
        # Build a small batched dataset.
        # Note: this fixture intentionally reuses fixed keys; it's a test.
        refs = jr.normal(jr.PRNGKey(0), (10, 2))
        comparisons = refs + jr.normal(jr.PRNGKey(1), (10, 2)) * 0.3
        responses = jnp.ones((10,), dtype=jnp.int32)
        return TrialData(
            stimuli=jnp.stack([refs, comparisons], axis=1), responses=responses
        )

    @pytest.fixture
    def touch_data(self):
        """Create dummy response data for continuous touch task."""
        # Build a small batched dataset.
        # Note: this fixture intentionally reuses fixed keys; it's a test.
        stims = jr.normal(jr.PRNGKey(0), (10, 2))
        responses = stims
        return TrialData(stimuli=jnp.expand_dims(stims, axis=1), responses=responses)

    @pytest.fixture
    def oddity_param_posterior(self, oddity_model, oddity_data):
        """Fit model and return ParameterPosterior."""
        optimizer = MAPOptimizer(steps=20)
        return optimizer.fit(oddity_model, oddity_data)

    @pytest.fixture
    def touch_param_posterior(self, touch_model, touch_data):
        """Fit model and return ParameterPosterior."""
        optimizer = MAPOptimizer(steps=20)
        return optimizer.fit(touch_model, touch_data)

    @pytest.fixture
    def oddity_predictive_posterior(self, oddity_param_posterior):
        """Create predictive posterior."""
        refs_test = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        comparisons = jnp.array([[0.5, 0.0], [1.5, 1.0], [2.5, 2.0]])
        X_test = jnp.stack([refs_test, comparisons], axis=1)
        return WPPMPredictivePosterior(oddity_param_posterior, X_test, n_samples=10)

    @pytest.fixture
    def touch_predictive_posterior(self, touch_param_posterior):
        """Create predictive posterior."""
        stims_test = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        X_test = jnp.expand_dims(stims_test, axis=1)
        return WPPMPredictivePosterior(touch_param_posterior, X_test, n_samples=10)

    def test_is_predictive_posterior(
        self, oddity_predictive_posterior, touch_predictive_posterior
    ):
        """WPPMPredictivePosterior implements PredictivePosterior protocol."""
        assert isinstance(oddity_predictive_posterior, PredictivePosterior)
        assert isinstance(touch_predictive_posterior, PredictivePosterior)

    def test_mean_shape(self, oddity_predictive_posterior, touch_predictive_posterior):
        """mean property has correct shape."""
        oddity_mean = oddity_predictive_posterior.mean
        touch_mean = touch_predictive_posterior.mean

        assert oddity_mean.shape == (3,)  # n_test
        assert jnp.all((oddity_mean >= 0) & (oddity_mean <= 1))  # Probabilities

        assert touch_mean.shape == (3, 2)  # n_test, r_dim

    def test_variance_shape(
        self, oddity_predictive_posterior, touch_predictive_posterior
    ):
        """variance property has correct shape."""
        oddity_var = oddity_predictive_posterior.variance
        touch_var = touch_predictive_posterior.variance

        assert oddity_var.shape == (3,)  # n_test
        assert jnp.all(oddity_var >= 0)  # Variances non-negative

        assert touch_var.shape == (3, 2, 2)  # r_dim
        assert jnp.all(touch_var >= 0)  # Variances non-negative

    def test_lazy_evaluation(
        self, oddity_predictive_posterior, touch_predictive_posterior
    ):
        """Moments computed lazily on first access."""
        assert not oddity_predictive_posterior._computed
        _ = oddity_predictive_posterior.mean
        assert oddity_predictive_posterior._computed
        # Second access should use cache
        mean2 = oddity_predictive_posterior.mean
        assert jnp.array_equal(oddity_predictive_posterior.mean, mean2)

        assert not touch_predictive_posterior._computed
        _ = touch_predictive_posterior.mean
        assert touch_predictive_posterior._computed
        # Second access should use cache
        mean2 = touch_predictive_posterior.mean
        assert jnp.array_equal(touch_predictive_posterior.mean, mean2)

    def test_rsample_shape(
        self, oddity_predictive_posterior, touch_predictive_posterior
    ):
        """rsample returns correct shape."""
        key = jr.PRNGKey(42)
        oddity_samples = oddity_predictive_posterior.rsample(sample_shape=(5,), key=key)
        touch_samples = touch_predictive_posterior.rsample(sample_shape=(5,), key=key)

        assert oddity_samples.shape == (5, 3)  # (n_samples, n_test)
        assert touch_samples.shape == (5, 3, 2)  # (n_samples, n_test, r_dim)

    def test_rsample_statistics(
        self, oddity_predictive_posterior, touch_predictive_posterior
    ):
        """rsample mean/std match moment properties."""
        key = jr.PRNGKey(42)
        oddity_samples = oddity_predictive_posterior.rsample(
            sample_shape=(1000,), key=key
        )
        oddity_sample_mean = jnp.mean(oddity_samples, axis=0)
        oddity_sample_var = jnp.var(oddity_samples, axis=0)

        touch_samples = touch_predictive_posterior.rsample(
            sample_shape=(1000,), key=key
        )
        touch_sample_mean = jnp.mean(touch_samples, axis=0)
        touch_sample_var = jax.vmap(lambda m: jnp.cov(m, rowvar=False), in_axes=1)(
            touch_samples
        )

        # Should be close (MC convergence)
        assert jnp.allclose(
            oddity_sample_mean, oddity_predictive_posterior.mean, atol=0.1
        )
        assert jnp.allclose(
            oddity_sample_var, oddity_predictive_posterior.variance, atol=0.1
        )

        assert jnp.allclose(
            touch_sample_mean, touch_predictive_posterior.mean, atol=0.1
        )
        assert jnp.allclose(
            touch_sample_var, touch_predictive_posterior.variance, atol=0.1
        )

    def test_cov_field_shape(
        self, oddity_predictive_posterior, touch_predictive_posterior
    ):
        """cov_field returns covariance matrices."""
        refs_test = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        oddity_Sigma = oddity_predictive_posterior.cov_field(refs_test)
        assert oddity_Sigma.shape == (2, 2, 2)  # (n_test, input_dim, input_dim)

        stims_test = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        touch_Sigma = touch_predictive_posterior.cov_field(stims_test)
        assert touch_Sigma.shape == (2, 2, 2)  # (n_test, input_dim, input_dim)

    def test_cov_field_psd(
        self, oddity_predictive_posterior, touch_predictive_posterior
    ):
        """Covariance matrices are positive semi-definite."""
        X_test = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        oddity_Sigma = oddity_predictive_posterior.cov_field(X_test)
        touch_Sigma = touch_predictive_posterior.cov_field(X_test)

        for i in range(len(X_test)):
            oddity_eigvals = jnp.linalg.eigvalsh(oddity_Sigma[i])
            touch_eigvals = jnp.linalg.eigvalsh(touch_Sigma[i])
            assert jnp.all(oddity_eigvals >= -1e-6)  # Numerically PSD
            assert jnp.all(touch_eigvals >= -1e-6)  # Numerically PSD

    def test_no_probes_raises(self, oddity_param_posterior):
        """Creating predictive posterior with threshold_pred set to True raises NotImplementedError."""
        X_test = jnp.array([[0.0, 0.0]])
        pred_post = WPPMPredictivePosterior(
            oddity_param_posterior, X_test, threshold_pred=True
        )

        with pytest.raises(NotImplementedError, match="Threshold prediction"):
            _ = pred_post.mean


class TestIntegration:
    """Integration tests for the two-tier design."""

    def test_full_workflow(self):
        """Test complete workflow: fit → parameter posterior → predictive posterior."""
        # 1. Create models
        oddity_model = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=3),
            likelihood=OddityTask(),
            noise=GaussianNoise(),
        )
        touch_model = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=3),
            likelihood=ContinuousTouchTask(),
            noise=GaussianNoise(),
        )

        # 2. Create data
        key = jr.PRNGKey(123)
        key, k_ref, k_eps = jr.split(key, 3)
        refs = jr.normal(k_ref, (20, 2))
        comparisons = refs + jr.normal(k_eps, (20, 2)) * 0.5
        responses = jnp.ones((20,), dtype=jnp.int32)
        oddity_data = TrialData(
            stimuli=jnp.stack([refs, comparisons], axis=1), responses=responses
        )
        touch_data = TrialData(
            stimuli=jnp.expand_dims(refs, axis=1), responses=comparisons
        )

        # 3. Fit models -> ParameterPosterior
        optimizer = MAPOptimizer(steps=50)
        oddity_param_post = optimizer.fit(oddity_model, oddity_data)
        touch_param_post = optimizer.fit(touch_model, touch_data)
        assert isinstance(oddity_param_post, ParameterPosterior)
        assert isinstance(touch_param_post, ParameterPosterior)

        # 5. Create PredictivePosterior
        refs_test = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        comparisons = jnp.array([[0.3, 0.0], [1.3, 1.0]])
        oddity_X_test = jnp.stack([refs_test, comparisons], axis=1)
        touch_X_test = jnp.expand_dims(refs_test, axis=1)
        oddity_pred_post = WPPMPredictivePosterior(
            oddity_param_post, oddity_X_test, n_samples=20
        )
        touch_pred_post = WPPMPredictivePosterior(
            touch_param_post, touch_X_test, n_samples=20
        )

        # 6. Get predictions
        oddity_mean = oddity_pred_post.mean
        oddity_var = oddity_pred_post.variance
        touch_mean = touch_pred_post.mean
        touch_var = touch_pred_post.variance
        assert oddity_mean.shape == (2,)
        assert oddity_var.shape == (2,)
        assert touch_mean.shape == (2, 2)
        assert touch_var.shape == (2, 2, 2)

        # 7. Sample predictions
        key, subkey = jr.split(key)
        oddity_pred_samples = oddity_pred_post.rsample((5,), key=subkey)
        assert oddity_pred_samples.shape == (5, 2)
        touch_pred_samples = touch_pred_post.rsample((5,), key=subkey)
        assert touch_pred_samples.shape == (5, 2, 2)

        # 8. Get covariance field (just refs for oddity)
        oddity_Sigma = oddity_pred_post.cov_field(refs_test)
        assert oddity_Sigma.shape == (2, 2, 2)

        touch_Sigma = touch_pred_post.cov_field(refs_test)
        assert touch_Sigma.shape == (2, 2, 2)

        # 9. Get covariance field (all stimuli)
        oddity_Sigma = oddity_pred_post.cov_field(oddity_X_test)
        assert oddity_Sigma.shape == (2, 2, 2, 2)
        touch_Sigma = touch_pred_post.cov_field(touch_X_test)
        assert touch_Sigma.shape == (2, 1, 2, 2)
