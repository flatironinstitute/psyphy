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

from psyphy.data import TrialData
from psyphy.inference import MAPOptimizer
from psyphy.model import WPPM, GaussianNoise, OddityTask, OddityTaskConfig, Prior
from psyphy.posterior import (
    MAPPosterior,
    ParameterPosterior,
    PredictivePosterior,
    ThresholdConfig,
    WPPMPredictivePosterior,
)


class TestParameterPosterior:
    """Test ParameterPosterior protocol and MAPPosterior implementation."""

    @pytest.fixture
    def model(self):
        """Create a simple WPPM model."""
        return WPPM(
            prior=Prior(input_dim=2, basis_degree=3),
            likelihood=OddityTask(),
            noise=GaussianNoise(),
        )

    @pytest.fixture
    def data(self):
        """Create dummy response data."""

        refs = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        comparisons = jnp.array([[0.5, 0.5], [1.5, 1.0]])
        responses = jnp.array([1, 0], dtype=jnp.int32)
        return TrialData(
            stimuli=jnp.stack([refs, comparisons], axis=1), responses=responses
        )

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

    def test_model_property(self, param_posterior):
        """model property returns associated model."""
        model = param_posterior.model
        assert isinstance(model, WPPM)
        assert model.input_dim == 2

    def test_sample_with_key(self, param_posterior):
        """Sample from MAP posterior returns identical replicates."""
        n_samples = 3
        key = jr.PRNGKey(0)
        samples = param_posterior.sample(n=n_samples, key=key)

        assert isinstance(samples, dict)
        assert "W" in samples
        assert samples["W"].shape[0] == n_samples

        # All samples should be identical to MAP estimate
        map_params = param_posterior.params
        for i in range(n_samples):
            assert jnp.allclose(samples["W"][i], map_params["W"])


class TestPredictivePosterior:
    """Test PredictivePosterior protocol and WPPMPredictivePosterior implementation."""

    @pytest.fixture
    def model(self):
        """Create a simple WPPM model."""
        return WPPM(
            prior=Prior(input_dim=2, basis_degree=3),
            likelihood=OddityTask(),
            noise=GaussianNoise(),
        )

    @pytest.fixture
    def data(self):
        """Create dummy response data."""
        # Build a small batched dataset.
        # Note: this fixture reuses fixed keys; it's a test.
        refs = jr.normal(jr.PRNGKey(0), (10, 2))
        comparisons = refs + jr.normal(jr.PRNGKey(1), (10, 2)) * 0.3
        responses = jnp.ones((10,), dtype=jnp.int32)
        return TrialData(
            stimuli=jnp.stack([refs, comparisons], axis=1), responses=responses
        )

    @pytest.fixture
    def param_posterior(self, model, data):
        """Fit model and return ParameterPosterior."""
        optimizer = MAPOptimizer(steps=20)
        return optimizer.fit(model, data)

    @pytest.fixture
    def predictive_posterior(self, param_posterior):
        """Create predictive posterior."""
        refs_test = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        comparisons = jnp.array([[0.5, 0.0], [1.5, 1.0], [2.5, 2.0]])
        X_test = jnp.stack([refs_test, comparisons], axis=1)
        return WPPMPredictivePosterior(
            param_posterior, X_test, n_samples=10, key=jr.PRNGKey(0)
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

    def test_threshold_pred_rejects_paired_stimuli_shape(self, param_posterior):
        """threshold_pred=True expects bare reference points, not (ref, comp) pairs."""
        X_test = jnp.zeros((2, 2, 2))  # (n_test, k_stimuli, input_dim) -- wrong shape
        pred_post = WPPMPredictivePosterior(
            param_posterior, X_test, threshold_pred=True, key=jr.PRNGKey(0)
        )
        with pytest.raises(ValueError, match="bare reference points"):
            _ = pred_post.mean

    @pytest.fixture
    def threshold_param_posterior(self):
        """A MAPPosterior with a fast OddityTaskConfig, for threshold-mode tests.

        No fitting needed -- threshold inversion only consumes params/model,
        so a prior sample via MAPPosterior is enough and much faster than
        running MAPOptimizer.
        """
        model = WPPM(
            prior=Prior(input_dim=2, basis_degree=3),
            likelihood=OddityTask(config=OddityTaskConfig(num_samples=20)),
            noise=GaussianNoise(),
        )
        params = model.init_params(jr.PRNGKey(0))
        return MAPPosterior(params, model)

    @pytest.fixture
    def threshold_predictive_posterior(self, threshold_param_posterior):
        """Predictive posterior in threshold_pred mode, tiny settings for speed."""
        X_test = jnp.array([[0.0, 0.0], [0.3, 0.3]])  # bare reference points
        cfg = ThresholdConfig(n_theta=6, n_length=25, chunk=1000)
        return WPPMPredictivePosterior(
            threshold_param_posterior,
            X_test,
            n_samples=2,
            threshold_pred=True,
            threshold_config=cfg,
            key=jr.PRNGKey(0),
        )

    def test_threshold_mean_shape(self, threshold_predictive_posterior):
        """threshold mean has shape (n_test, input_dim, input_dim)."""
        mean = threshold_predictive_posterior.mean
        assert mean.shape == (2, 2, 2)

    def test_threshold_mean_is_psd(self):
        """Recovered threshold covariances are positive semi-definite.

        Not using the tiny ``threshold_predictive_posterior``
        fixture, and building its own model with a realistic ``num_samples``.

        The inversion fits Sigma^-1 by least squares through one boundary point
        per probed direction. When those radii are dominated by Monte Carlo
        noise the fit is ill-conditioned and can come back indefinite -- the
        Monte Carlo sample count is the dominant term, not the direction count.
        At ``num_samples=20`` this fails readily and *inconsistently*: observed
        eigenvalues of (-0.17, +0.02) under jax 0.4.28 where jax 0.11 passed,
        then the reverse after raising only n_theta/n_length. Those outcomes
        differ solely because the PRNG stream differs between versions.

        A PSD assertion under settings too coarse to support one tests the seed,
        not the implementation, so this uses settings near the tutorial's.
        """
        model = WPPM(
            prior=Prior(input_dim=2, basis_degree=3),
            likelihood=OddityTask(config=OddityTaskConfig(num_samples=500)),
            noise=GaussianNoise(),
        )
        posterior = MAPPosterior(model.init_params(jr.PRNGKey(0)), model)
        X_test = jnp.array([[0.0, 0.0], [0.3, 0.3]])
        predictive = WPPMPredictivePosterior(
            posterior,
            X_test,
            n_samples=1,
            threshold_pred=True,
            threshold_config=ThresholdConfig(n_theta=16, n_length=200, chunk=2000),
            key=jr.PRNGKey(0),
        )
        mean = predictive.mean
        for i in range(mean.shape[0]):
            eigvals = jnp.linalg.eigvalsh(mean[i])
            assert jnp.all(eigvals >= -1e-6), f"non-PSD at test point {i}: {eigvals}"

    def test_threshold_variance_shape_and_nonneg(self, threshold_predictive_posterior):
        """threshold variance has the same shape as mean and is non-negative."""
        var = threshold_predictive_posterior.variance
        assert var.shape == (2, 2, 2)
        assert jnp.all(var >= 0)

    def test_threshold_rsample_shape(self, threshold_predictive_posterior):
        """threshold rsample draws one covariance per sample, per test point."""
        key = jr.PRNGKey(42)
        samples = threshold_predictive_posterior.rsample(sample_shape=(3,), key=key)
        assert samples.shape == (3, 2, 2, 2)  # (*sample_shape, n_test, d, d)

    def test_threshold_n_theta_too_small_raises(self, threshold_param_posterior):
        """n_theta below d*(d+1)/2 (=3 for 2D) makes the ellipsoid fit underdetermined."""
        X_test = jnp.array([[0.0, 0.0]])
        cfg = ThresholdConfig(n_theta=2, n_length=10)
        pred_post = WPPMPredictivePosterior(
            threshold_param_posterior,
            X_test,
            threshold_pred=True,
            threshold_config=cfg,
            key=jr.PRNGKey(0),
        )
        with pytest.raises(ValueError, match="too small"):
            _ = pred_post.mean


class TestThresholdPredictionExternalValidity:
    """Threshold inversion recovers Hong et al. (2025)'s published thresholds.

    Requires the real OSF download; skips otherwise. This is the first check
    that exercises the oddity likelihood itself, rather than only the
    deterministic covariance-field construction (see
    tests/test_data_published_hong2025.py for that sibling test).

    Runtime: a handful of grid points at reduced-from-paper MC settings,
    budgeted to stay under a minute; not the full 49-point, paper-precision
    sweep, which takes about 11 CPU minutes.
    """

    @staticmethod
    def _data_paths():
        from psyphy.data.published import hong2025

        sub1 = hong2025.default_data_dir() / "sub1"
        return sub1 / "Bestfit_W_sub1.csv", sub1 / "Thres_ellipses_sub1.csv"

    def test_recovers_published_threshold_ellipses(self):
        from psyphy.data.published import hong2025

        weights_path, thres_path = self._data_paths()
        if not (weights_path.exists() and thres_path.exists()):
            pytest.skip(
                "Published data not downloaded. Run: python -c "
                "'from psyphy.data.published import hong2025; "
                "hong2025.fetch(1)'"
            )

        W_org = hong2025.load_reference_W(weights_path)
        coords, published = hong2025.load_sigma_table(thres_path)
        model = hong2025.build_paper_model(mc_samples=500)
        posterior = MAPPosterior({"W": W_org}, model)

        # A handful of grid points, not all 49 -- keeps this test fast.
        idx = jnp.linspace(0, len(coords) - 1, 4).astype(int)
        X_test = jnp.asarray(coords)[idx]

        pred_post = WPPMPredictivePosterior(
            posterior,
            X_test,
            n_samples=1,
            threshold_pred=True,
            threshold_config=ThresholdConfig(n_theta=16, n_length=300),
            key=jr.PRNGKey(0),
        )
        recovered = pred_post.mean  # (4, 2, 2)

        got = jnp.sqrt(jnp.linalg.eigvalsh(recovered))
        want = jnp.sqrt(jnp.linalg.eigvalsh(jnp.asarray(published)[idx]))
        rel_err = jnp.abs(got - want) / want
        # Loose tolerance: this is a stochastic MC inversion (16 directions,
        # finite MC samples), not the bit-exact rung-1 comparison.
        assert float(jnp.median(rel_err)) < 0.10, (
            f"median semi-axis relative error {float(jnp.median(rel_err)):.3f} "
            "exceeds 10%"
        )


class TestIntegration:
    """Integration tests for the two-tier design."""

    def test_full_workflow(self):
        """Test complete workflow: fit -> parameter posterior -> predictive posterior."""
        # 1. Create model
        model = WPPM(
            prior=Prior(input_dim=2, basis_degree=3),
            likelihood=OddityTask(),
            noise=GaussianNoise(),
        )

        # 2. Create data
        key = jr.PRNGKey(123)
        key, k_ref, k_eps = jr.split(key, 3)
        refs = jr.normal(k_ref, (20, 2))
        comparisons = refs + jr.normal(k_eps, (20, 2)) * 0.5
        responses = jnp.ones((20,), dtype=jnp.int32)
        data = TrialData(
            stimuli=jnp.stack([refs, comparisons], axis=1), responses=responses
        )

        # 3. Fit model -> ParameterPosterior
        optimizer = MAPOptimizer(steps=50)
        param_post = optimizer.fit(model, data)
        assert isinstance(param_post, ParameterPosterior)

        # 5. Create PredictivePosterior
        refs_test = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        comparisons = jnp.array([[0.3, 0.0], [1.3, 1.0]])
        X_test = jnp.stack([refs_test, comparisons], axis=1)
        pred_post = WPPMPredictivePosterior(
            param_post, X_test, n_samples=20, key=jr.PRNGKey(0)
        )

        # 6. Get predictions
        mean = pred_post.mean
        var = pred_post.variance
        assert mean.shape == (2,)
        assert var.shape == (2,)

        # 7. Sample predictions
        key, subkey = jr.split(key)
        pred_samples = pred_post.rsample((5,), key=subkey)
        assert pred_samples.shape == (5, 2)
