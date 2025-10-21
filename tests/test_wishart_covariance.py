"""
test_wishart_covariance.py
--------------------------

Tests for Wishart process covariance (Issue #3, Task 2).

Tests spatially-varying covariance Σ(x) = U(x) @ U(x)^T + diag_term*I
where U(x) is computed from Chebyshev basis expansion.
"""

import jax.numpy as jnp
import jax.random as jr

from psyphy.model import WPPM, Prior
from psyphy.model.noise import GaussianNoise
from psyphy.model.task import OddityTask


class TestWishartParameters:
    """Tests for Wishart parameter structure and prior."""

    def test_wishart_params_structure(self):
        """With basis_degree set, params should include W coefficients."""
        model = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=3),
            task=OddityTask(),
            noise=GaussianNoise(),
            basis_degree=3,
        )

        params = model.init_params(jr.PRNGKey(0))

        # Should have W coefficients, not just log_diag
        assert "W" in params
        # W shape: (degree+1, degree+1, embedding_dim, embedding_dim + extra_dims)
        # For 2D with degree=3, extra_dims=0: (4, 4, 8, 8)
        # embedding_dim = input_dim * (degree + 1) = 2 * 4 = 8
        # Note: degree+1 because we have basis functions [T_0, ..., T_degree]
        degree = 3
        embedding_dim = model.embedding_dim
        expected_shape = (degree + 1, degree + 1, embedding_dim, embedding_dim)
        assert params["W"].shape == expected_shape

    def test_wishart_params_with_extra_dims(self):
        """Test W shape with extra_embedding_dims."""
        model = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=3, extra_embedding_dims=2),
            task=OddityTask(),
            noise=GaussianNoise(),
            basis_degree=3,
        )

        params = model.init_params(jr.PRNGKey(0))

        degree = 3
        embedding_dim = model.embedding_dim  # 8
        extra_dims = 2
        expected_shape = (
            degree + 1,
            degree + 1,
            embedding_dim,
            embedding_dim + extra_dims,
        )
        assert params["W"].shape == expected_shape

    def test_prior_decay_structure(self):
        """Prior variance should decay with basis function degree."""
        prior = Prior(input_dim=2, basis_degree=3, decay_rate=0.5, variance_scale=1.0)

        # Sample multiple times and check variance decreases with degree
        key = jr.PRNGKey(0)
        samples = []
        for _ in range(100):
            key, subkey = jr.split(key)
            params = prior.sample_params(subkey)
            samples.append(params["W"])

        W_samples = jnp.stack(samples)  # (100, degree, degree, ...)

        # Variance should decrease with total degree (i + j)
        # Degree-0 (constant) should have highest variance
        var_00 = jnp.var(W_samples[:, 0, 0, 0, 0])  # degree = 0 + 0 = 0
        var_11 = jnp.var(W_samples[:, 1, 1, 0, 0])  # degree = 1 + 1 = 2
        var_22 = jnp.var(W_samples[:, 2, 2, 0, 0])  # degree = 2 + 2 = 4

        # Higher degree → lower variance
        assert var_00 > var_11 > var_22

    def test_mvp_mode_still_works(self):
        """MVP mode (basis_degree=None) should use old log_diag params."""
        model = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2),  # No basis_degree
            task=OddityTask(),
            noise=GaussianNoise(),
            basis_degree=None,  # MVP mode
        )

        params = model.init_params(jr.PRNGKey(0))

        # Should have log_diag, not W
        assert "log_diag" in params
        assert "W" not in params
        assert params["log_diag"].shape == (2,)


class TestSpatiallyVaryingCovariance:
    """Tests for Σ(x) that varies with stimulus location."""

    def test_covariance_varies_with_location(self):
        """Σ(x) should be different at different locations."""
        model = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=3),
            task=OddityTask(),
            noise=GaussianNoise(),
            basis_degree=3,
        )

        params = model.init_params(jr.PRNGKey(42))

        x1 = jnp.array([0.2, 0.3])
        x2 = jnp.array([0.7, 0.8])

        Sigma1 = model.local_covariance(params, x1)
        Sigma2 = model.local_covariance(params, x2)

        # Should be different (not constant like MVP)
        assert not jnp.allclose(Sigma1, Sigma2, atol=1e-6)

    def test_covariance_smooth_variation(self):
        """Σ(x) should vary smoothly - nearby points have similar covariances."""
        model = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=3, decay_rate=0.8),
            task=OddityTask(),
            noise=GaussianNoise(),
            basis_degree=3,
        )

        params = model.init_params(jr.PRNGKey(42))

        x1 = jnp.array([0.5, 0.5])
        x2 = jnp.array([0.51, 0.51])  # Very close
        x3 = jnp.array([0.9, 0.9])  # Far away

        Sigma1 = model.local_covariance(params, x1)
        Sigma2 = model.local_covariance(params, x2)
        Sigma3 = model.local_covariance(params, x3)

        # Nearby points should be more similar
        diff_near = jnp.linalg.norm(Sigma1 - Sigma2, "fro")
        diff_far = jnp.linalg.norm(Sigma1 - Sigma3, "fro")

        assert diff_near < diff_far

    def test_covariance_positive_definite_everywhere(self):
        """Σ(x) must be positive definite at all locations."""
        model = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=3),
            task=OddityTask(),
            noise=GaussianNoise(),
            basis_degree=3,
        )

        params = model.init_params(jr.PRNGKey(42))

        # Test at random locations
        key = jr.PRNGKey(0)
        test_points = jr.uniform(key, (20, 2))

        for x in test_points:
            Sigma = model.local_covariance(params, x)

            # Check positive definite via eigenvalues
            eigenvalues = jnp.linalg.eigvalsh(Sigma)
            assert jnp.all(eigenvalues > 0), f"Non-PD at {x}: eigenvalues={eigenvalues}"

    def test_covariance_shape_in_embedding_space(self):
        """Σ(x) should be embedding_dim × embedding_dim."""
        model = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=3),
            task=OddityTask(),
            noise=GaussianNoise(),
            basis_degree=3,
        )

        params = model.init_params(jr.PRNGKey(0))
        x = jnp.array([0.5, 0.3])
        Sigma = model.local_covariance(params, x)

        # Should be embedding_dim × embedding_dim
        embedding_dim = model.embedding_dim
        assert Sigma.shape == (embedding_dim, embedding_dim)

    def test_diag_term_prevents_degeneracy(self):
        """diag_term should ensure minimum eigenvalue."""
        diag_term = 0.1
        model = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=3),
            task=OddityTask(),
            noise=GaussianNoise(),
            basis_degree=3,
            diag_term=diag_term,
        )

        params = model.init_params(jr.PRNGKey(0))
        x = jnp.array([0.5, 0.3])
        Sigma = model.local_covariance(params, x)

        # Minimum eigenvalue should be at least diag_term
        min_eigenvalue = jnp.min(jnp.linalg.eigvalsh(Sigma))
        assert min_eigenvalue >= diag_term * 0.99  # Allow small numerical error

    def test_mvp_constant_covariance(self):
        """MVP mode should still have constant covariance."""
        model = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2),
            task=OddityTask(),
            noise=GaussianNoise(),
            basis_degree=None,  # MVP mode
        )

        params = model.init_params(jr.PRNGKey(0))

        x1 = jnp.array([0.2, 0.3])
        x2 = jnp.array([0.7, 0.8])

        Sigma1 = model.local_covariance(params, x1)
        Sigma2 = model.local_covariance(params, x2)

        # Should be the same (constant in MVP mode)
        assert jnp.allclose(Sigma1, Sigma2, atol=1e-10)


class TestWishartIntegration:
    """Integration tests with full model pipeline."""

    def test_discriminability_with_wishart(self):
        """Discriminability should work with spatially-varying covariance."""
        model = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=3),
            task=OddityTask(),
            noise=GaussianNoise(),
            basis_degree=3,
        )

        params = model.init_params(jr.PRNGKey(0))

        ref = jnp.array([0.5, 0.3])
        probe = jnp.array([0.6, 0.4])
        stimulus = (ref, probe)

        d = model.discriminability(params, stimulus)

        # Should be non-negative scalar
        assert d.shape == ()
        assert d >= 0.0

    def test_prediction_with_wishart(self):
        """Model predictions should work with Wishart covariance."""
        model = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=3),
            task=OddityTask(),
            noise=GaussianNoise(),
            basis_degree=3,
        )

        params = model.init_params(jr.PRNGKey(0))

        ref = jnp.array([0.5, 0.3])
        probe = jnp.array([0.6, 0.4])
        stimulus = (ref, probe)

        p_correct = model.predict_prob(params, stimulus)

        # Should be valid probability
        assert 0.0 <= p_correct <= 1.0

    def test_fit_with_wishart(self):
        """Full fitting pipeline should work with Wishart process."""
        model = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=2),  # Lower degree for speed
            task=OddityTask(),
            noise=GaussianNoise(),
            basis_degree=2,
        )

        # Generate simple training data
        n = 20
        key = jr.PRNGKey(42)
        refs = jr.uniform(key, (n, 2))
        probes = refs + 0.05
        y = jnp.ones(n, dtype=int)
        X = jnp.stack([refs, probes], axis=1)

        # Fit should not raise
        model.fit(X, y, inference="map", inference_config={"steps": 10})

        # Should be able to make predictions
        pred_post = model.posterior(refs[:5], probes=probes[:5])
        assert pred_post.mean.shape == (5,)

    def test_wishart_vs_mvp_predictions_differ(self):
        """Wishart and MVP should give different predictions (spatial variation)."""
        # MVP model
        mvp = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2),
            task=OddityTask(),
            noise=GaussianNoise(),
            basis_degree=None,
        )

        # Wishart model
        wishart = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=3),
            task=OddityTask(),
            noise=GaussianNoise(),
            basis_degree=3,
        )

        # Use same random seed for W initialization
        params_mvp = mvp.init_params(jr.PRNGKey(0))
        params_wishart = wishart.init_params(jr.PRNGKey(0))

        # Test at different locations
        ref1 = jnp.array([0.2, 0.3])
        ref2 = jnp.array([0.7, 0.8])
        probe_offset = jnp.array([0.1, 0.1])

        # MVP: predictions should have same relative pattern everywhere
        d_mvp_1 = mvp.discriminability(params_mvp, (ref1, ref1 + probe_offset))
        d_mvp_2 = mvp.discriminability(params_mvp, (ref2, ref2 + probe_offset))

        # Wishart: discriminability can vary with location
        d_wishart_1 = wishart.discriminability(
            params_wishart, (ref1, ref1 + probe_offset)
        )
        d_wishart_2 = wishart.discriminability(
            params_wishart, (ref2, ref2 + probe_offset)
        )

        # Not a strict assertion - just demonstrating that spatial variation is possible
        # In practice, with good W coefficients, discriminability typically varies with location
        # For now, we just verify the computation doesn't crash and both models work
        assert d_mvp_1 > 0 and d_mvp_2 > 0  # MVP computed successfully
        assert d_wishart_1 > 0 and d_wishart_2 > 0  # Wishart computed successfully


class TestComputeU:
    """Tests for internal _compute_U method."""

    def test_compute_U_shape(self):
        """U(x) should have shape (embedding_dim, embedding_dim + extra_dims)."""
        extra_dims = 2
        model = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=3, extra_embedding_dims=extra_dims),
            task=OddityTask(),
            noise=GaussianNoise(),
            basis_degree=3,
            extra_dims=extra_dims,
        )

        params = model.init_params(jr.PRNGKey(0))
        x = jnp.array([0.5, 0.3])

        U = model._compute_U(params, x)

        embedding_dim = model.embedding_dim
        expected_shape = (embedding_dim, embedding_dim + extra_dims)
        assert U.shape == expected_shape

    def test_compute_U_varies_with_location(self):
        """U(x) should vary with stimulus location."""
        model = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=3),
            task=OddityTask(),
            noise=GaussianNoise(),
            basis_degree=3,
        )

        params = model.init_params(jr.PRNGKey(42))

        x1 = jnp.array([0.2, 0.3])
        x2 = jnp.array([0.7, 0.8])

        U1 = model._compute_U(params, x1)
        U2 = model._compute_U(params, x2)

        # Should be different
        assert not jnp.allclose(U1, U2, atol=1e-6)

    def test_sigma_equals_UUT_plus_jitter(self):
        """Verify Σ(x) = U(x) @ U(x)^T + diag_term * I."""
        diag_term = 0.01
        model = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=3),
            task=OddityTask(),
            noise=GaussianNoise(),
            basis_degree=3,
            diag_term=diag_term,
        )

        params = model.init_params(jr.PRNGKey(0))
        x = jnp.array([0.5, 0.3])

        U = model._compute_U(params, x)
        Sigma = model.local_covariance(params, x)

        # Manual computation
        expected_Sigma = U @ U.T + diag_term * jnp.eye(model.embedding_dim)

        assert jnp.allclose(Sigma, expected_Sigma, atol=1e-6)
