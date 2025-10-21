"""
test_posteriors.py
-----------------

Tests for the two-tier posterior design:
- ParameterPosterior protocol and implementations
- PredictivePosterior protocol and implementations
"""

import jax.numpy as jnp
import jax.random as jr
import pytest

from psyphy.data import ResponseData
from psyphy.inference import MAPOptimizer
from psyphy.model import WPPM
from psyphy.model.noise import GaussianNoise
from psyphy.model.prior import Prior
from psyphy.model.task import OddityTask
from psyphy.posterior import (
    MAPPosterior,
    ParameterPosterior,
    Posterior,
    PredictivePosterior,
    WPPMPredictivePosterior,
)


class TestParameterPosterior:
    """Test ParameterPosterior protocol and MAPPosterior implementation."""

    @pytest.fixture
    def model(self):
        """Create a simple WPPM model."""
        return WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, scale=0.5),
            task=OddityTask(),
            noise=GaussianNoise(),
        )

    @pytest.fixture
    def data(self):
        """Create dummy response data."""
        data = ResponseData()
        data.add_trial(
            ref=jnp.array([0.0, 0.0]),
            probe=jnp.array([0.5, 0.5]),
            resp=1,
        )
        data.add_trial(
            ref=jnp.array([1.0, 1.0]),
            probe=jnp.array([1.5, 1.0]),
            resp=0,
        )
        return data

    @pytest.fixture
    def param_posterior(self, model, data):
        """Fit model and return ParameterPosterior."""
        optimizer = MAPOptimizer(steps=10)  # Few steps for speed
        return optimizer.fit(model, data)

    def test_map_posterior_is_parameter_posterior(self, param_posterior):
        """MAPPosterior implements ParameterPosterior protocol."""
        assert isinstance(param_posterior, ParameterPosterior)
        assert isinstance(param_posterior, MAPPosterior)

    def test_params_property(self, param_posterior):
        """params property returns parameter dict."""
        params = param_posterior.params
        assert isinstance(params, dict)
        assert "log_diag" in params
        assert isinstance(params["log_diag"], jnp.ndarray)

    def test_model_property(self, param_posterior):
        """model property returns associated model."""
        model = param_posterior.model
        assert isinstance(model, WPPM)
        assert model.input_dim == 2

    def test_sample_with_key(self, param_posterior):
        """sample() returns parameter samples with correct shape."""
        key = jr.PRNGKey(42)
        samples = param_posterior.sample(5, key=key)

        assert isinstance(samples, dict)
        assert "log_diag" in samples
        # Should have leading dimension 5
        assert samples["log_diag"].shape[0] == 5
        assert samples["log_diag"].shape[1] == 2  # input_dim

    def test_sample_delta_distribution(self, param_posterior):
        """MAP posterior returns repeated MAP estimate."""
        key = jr.PRNGKey(42)
        samples = param_posterior.sample(3, key=key)

        # All samples should be identical (delta distribution)
        for i in range(1, 3):
            assert jnp.allclose(samples["log_diag"][i], samples["log_diag"][0])

    def test_log_prob_at_map(self, param_posterior):
        """log_prob at MAP should be 0."""
        log_p = param_posterior.log_prob(param_posterior.params)
        assert jnp.isfinite(log_p)
        # Delta distribution: 0 at MAP (or close to it due to numerics)
        assert jnp.allclose(log_p, 0.0, atol=1e-5)

    def test_diagnostics(self, param_posterior):
        """diagnostics() returns dict."""
        diag = param_posterior.diagnostics()
        assert isinstance(diag, dict)

    def test_predict_prob(self, param_posterior):
        """predict_prob delegates to model."""
        stimulus = (jnp.array([0.0, 0.0]), jnp.array([0.5, 0.5]))
        prob = param_posterior.predict_prob(stimulus)
        assert isinstance(prob, jnp.ndarray)
        assert 0.0 <= prob <= 1.0

    def test_predict_thresholds(self, param_posterior):
        """predict_thresholds returns contour."""
        reference = jnp.array([0.0, 0.0])
        contour = param_posterior.predict_thresholds(reference, directions=8)
        assert contour.shape == (8, 2)

    def test_backwards_compatibility_alias(self):
        """Posterior is alias for MAPPosterior."""
        assert Posterior is MAPPosterior


class TestPredictivePosterior:
    """Test PredictivePosterior protocol and WPPMPredictivePosterior implementation."""

    @pytest.fixture
    def model(self):
        """Create a simple WPPM model."""
        return WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, scale=0.5),
            task=OddityTask(),
            noise=GaussianNoise(),
        )

    @pytest.fixture
    def data(self):
        """Create dummy response data."""
        data = ResponseData()
        for _ in range(10):  # More data for better posterior
            ref = jr.normal(jr.PRNGKey(0), (2,))
            probe = ref + jr.normal(jr.PRNGKey(1), (2,)) * 0.3
            data.add_trial(ref, probe, resp=1)
        return data

    @pytest.fixture
    def param_posterior(self, model, data):
        """Fit model and return ParameterPosterior."""
        optimizer = MAPOptimizer(steps=20)
        return optimizer.fit(model, data)

    @pytest.fixture
    def predictive_posterior(self, param_posterior):
        """Create predictive posterior."""
        X_test = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        probes = jnp.array([[0.5, 0.0], [1.5, 1.0], [2.5, 2.0]])
        return WPPMPredictivePosterior(
            param_posterior, X_test, probes=probes, n_samples=10
        )

    def test_is_predictive_posterior(self, predictive_posterior):
        """WPPMPredictivePosterior implements PredictivePosterior protocol."""
        assert isinstance(predictive_posterior, PredictivePosterior)

    def test_mean_shape(self, predictive_posterior):
        """mean property has correct shape."""
        mean = predictive_posterior.mean
        assert mean.shape == (3,)  # n_test
        assert jnp.all((mean >= 0) & (mean <= 1))  # Probabilities

    def test_variance_shape(self, predictive_posterior):
        """variance property has correct shape."""
        var = predictive_posterior.variance
        assert var.shape == (3,)  # n_test
        assert jnp.all(var >= 0)  # Variances non-negative

    def test_lazy_evaluation(self, predictive_posterior):
        """Moments computed lazily on first access."""
        assert not predictive_posterior._computed
        _ = predictive_posterior.mean
        assert predictive_posterior._computed
        # Second access should use cache
        mean2 = predictive_posterior.mean
        assert jnp.array_equal(predictive_posterior.mean, mean2)

    def test_rsample_shape(self, predictive_posterior):
        """rsample returns correct shape."""
        key = jr.PRNGKey(42)
        samples = predictive_posterior.rsample(sample_shape=(5,), key=key)
        assert samples.shape == (5, 3)  # (n_samples, n_test)

    def test_rsample_statistics(self, predictive_posterior):
        """rsample mean/std match moment properties."""
        key = jr.PRNGKey(42)
        samples = predictive_posterior.rsample(sample_shape=(1000,), key=key)

        sample_mean = jnp.mean(samples, axis=0)
        sample_var = jnp.var(samples, axis=0)

        # Should be close (MC convergence)
        assert jnp.allclose(sample_mean, predictive_posterior.mean, atol=0.1)
        assert jnp.allclose(sample_var, predictive_posterior.variance, atol=0.1)

    def test_cov_field_shape(self, predictive_posterior):
        """cov_field returns covariance matrices."""
        X_test = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        Sigma = predictive_posterior.cov_field(X_test)
        assert Sigma.shape == (2, 2, 2)  # (n_test, input_dim, input_dim)

    def test_cov_field_psd(self, predictive_posterior):
        """Covariance matrices are positive semi-definite."""
        X_test = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        Sigma = predictive_posterior.cov_field(X_test)

        for i in range(len(X_test)):
            eigvals = jnp.linalg.eigvalsh(Sigma[i])
            assert jnp.all(eigvals >= -1e-6)  # Numerically PSD

    def test_no_probes_raises(self, param_posterior):
        """Creating predictive posterior without probes raises NotImplementedError."""
        X_test = jnp.array([[0.0, 0.0]])
        pred_post = WPPMPredictivePosterior(param_posterior, X_test, probes=None)

        with pytest.raises(NotImplementedError, match="Threshold prediction"):
            _ = pred_post.mean


class TestIntegration:
    """Integration tests for the two-tier design."""

    def test_full_workflow(self):
        """Test complete workflow: fit → parameter posterior → predictive posterior."""
        # 1. Create model
        model = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, scale=0.5),
            task=OddityTask(),
            noise=GaussianNoise(),
        )

        # 2. Create data
        data = ResponseData()
        key = jr.PRNGKey(123)
        for _ in range(20):
            key, subkey = jr.split(key)
            ref = jr.normal(subkey, (2,))
            key, subkey = jr.split(key)
            probe = ref + jr.normal(subkey, (2,)) * 0.5
            data.add_trial(ref, probe, resp=1)

        # 3. Fit model → ParameterPosterior
        optimizer = MAPOptimizer(steps=50)
        param_post = optimizer.fit(model, data)
        assert isinstance(param_post, ParameterPosterior)

        # 4. Sample parameters
        key, subkey = jr.split(key)
        param_samples = param_post.sample(10, key=subkey)
        assert param_samples["log_diag"].shape == (10, 2)

        # 5. Create PredictivePosterior
        X_test = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        probes = jnp.array([[0.3, 0.0], [1.3, 1.0]])
        pred_post = WPPMPredictivePosterior(param_post, X_test, probes, n_samples=20)

        # 6. Get predictions
        mean = pred_post.mean
        var = pred_post.variance
        assert mean.shape == (2,)
        assert var.shape == (2,)

        # 7. Sample predictions
        key, subkey = jr.split(key)
        pred_samples = pred_post.rsample((5,), key=subkey)
        assert pred_samples.shape == (5, 2)

        # 8. Get covariance field
        Sigma = pred_post.cov_field(X_test)
        assert Sigma.shape == (2, 2, 2)
