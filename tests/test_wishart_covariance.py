"""
test_wishart_covariance.py
--------------------------

Tests for Wishart process covariance (Issue #3, Task 2).

Tests spatially-varying covariance Σ(x) = U(x) @ U(x)^T + diag_term*I
where U(x) is computed from Chebyshev basis expansion.
"""

import jax.numpy as jnp
import jax.random as jr

from psyphy.data import TrialData
from psyphy.inference import MAPOptimizer
from psyphy.model import WPPM, GaussianNoise, OddityTask, Prior
from psyphy.posterior import WPPMPredictivePosterior


class TestWishartParameters:
    """Tests for Wishart parameter structure and prior."""

    def test_wishart_params_structure(self):
        """With basis_degree set, params should include W coefficients."""
        model = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=3),
            likelihood=OddityTask(),
            noise=GaussianNoise(),
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
        expected_shape = (degree + 1, degree + 1, model.input_dim, embedding_dim)
        assert params["W"].shape == expected_shape

    def test_wishart_params_with_extra_dims(self):
        """W parameters should be rectangular with extra_dims (Hong et al. design).

        Rectangular design: W shape is (degree+1, degree+1, input_dim, embedding_dim)
        where embedding_dim = input_dim + extra_dims
        """
        extra_dims = 2
        model = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=3, extra_embedding_dims=extra_dims),
            likelihood=OddityTask(),
            noise=GaussianNoise(),
            extra_dims=extra_dims,
        )

        params = model.init_params(jr.PRNGKey(0))

        degree = 3
        # Rectangular design: embedding_dim = input_dim + extra_dims = 2 + 2 = 4
        embedding_dim = model.embedding_dim
        assert embedding_dim == 4  # Verify new convention

        # W is rectangular: (input_dim, embedding_dim)
        expected_shape = (
            degree + 1,
            degree + 1,
            model.input_dim,  # 2
            embedding_dim,  # 4 (rectangular!)
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


class TestSpatiallyVaryingCovariance:
    """Tests for Σ(x) that varies with stimulus location."""

    def test_covariance_varies_with_location(self):
        """Σ(x) should be different at different locations."""
        model = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=3),
            likelihood=OddityTask(),
            noise=GaussianNoise(),
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
            likelihood=OddityTask(),
            noise=GaussianNoise(),
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
            likelihood=OddityTask(),
            noise=GaussianNoise(),
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

    def test_covariance_shape_in_stimulus_space(self):
        """Σ(x) should be input_dim x input_dim."""
        model = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=3),
            likelihood=OddityTask(),
            noise=GaussianNoise(),
        )

        params = model.init_params(jr.PRNGKey(0))
        x = jnp.array([0.5, 0.3])
        Sigma = model.local_covariance(params, x)

        assert Sigma.shape == (model.input_dim, model.input_dim)

    def test_sqrt_U_shape(self):
        """U(x) should be input_dim x embedding_dim."""
        model = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=3),
            likelihood=OddityTask(),
            noise=GaussianNoise(),
        )

        params = model.init_params(jr.PRNGKey(0))
        x = jnp.array([0.5, 0.3])
        # Should be input_dim × embedding_dim
        embedding_dim = model.embedding_dim
        U = model._compute_sqrt(params, x)
        assert U.shape == (model.input_dim, embedding_dim)

    def test_diag_term_prevents_degeneracy(self):
        """diag_term should ensure minimum eigenvalue."""
        diag_term = 0.1
        model = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=3),
            likelihood=OddityTask(),
            noise=GaussianNoise(),
            diag_term=diag_term,
        )

        params = model.init_params(jr.PRNGKey(0))
        x = jnp.array([0.5, 0.3])
        Sigma = model.local_covariance(params, x)

        # Minimum eigenvalue should be at least diag_term
        min_eigenvalue = jnp.min(jnp.linalg.eigvalsh(Sigma))
        assert min_eigenvalue >= diag_term * 0.99  # Allow small numerical error


class TestWishartIntegration:
    """Integration tests with model pipeline."""

    def test_prediction_with_wishart(self):
        """Model predictions should work with Wishart covariance."""
        model = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=3),
            likelihood=OddityTask(),
            noise=GaussianNoise(),
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
            prior=Prior(input_dim=2, basis_degree=3),  # Lower degree for speed
            likelihood=OddityTask(),
            noise=GaussianNoise(),
        )

        # Generate simple training data
        n = 20
        key = jr.PRNGKey(42)
        refs = jr.uniform(key, (n, 2))
        comparisons = refs + 0.05
        # y = jnp.ones(n, dtype=int)
        # X = jnp.stack([refs, comparisons], axis=1)
        responses = jnp.ones((n,), dtype=jnp.int32)

        inputs = jnp.stack([refs, comparisons], axis=1)
        data = TrialData(inputs=inputs, responses=responses)

        # fit
        optimizer = MAPOptimizer(steps=10)
        param_post = optimizer.fit(model, data)

        # Should be able to make predictions
        # pred_post = model.posterior(refs[:5], comparisons=comparisons[:5])
        pred_post = WPPMPredictivePosterior(
            param_post, refs[:5], comparisons=comparisons[:5]
        )
        assert pred_post.mean.shape == (5,)


class TestComputeU:
    """Tests for internal _compute_sqrt method."""

    def test_compute_sqrt_shape(self):
        """
        U(x) should have correct shape (rectangular).

        Rectangular design: U is (input_dim, embedding_dim)
        where embedding_dim = input_dim + extra_dims
        """
        extra_dims = 2
        model = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=3, extra_embedding_dims=extra_dims),
            likelihood=OddityTask(),
            noise=GaussianNoise(),
            extra_dims=extra_dims,
        )

        params = model.init_params(jr.PRNGKey(0))
        x = jnp.array([0.5, 0.3])

        U = model._compute_sqrt(params, x)

        # Rectangular design: U is (input_dim, embedding_dim)
        embedding_dim = model.embedding_dim  # input_dim + extra_dims = 4
        expected_shape = (model.input_dim, embedding_dim)  # (2, 4)
        assert U.shape == expected_shape

    def test_compute_sqrt_varies_with_location(self):
        """U(x) should vary with stimulus location."""
        model = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=3),
            likelihood=OddityTask(),
            noise=GaussianNoise(),
        )

        params = model.init_params(jr.PRNGKey(42))

        x1 = jnp.array([0.2, 0.3])
        x2 = jnp.array([0.7, 0.8])

        U1 = model._compute_sqrt(params, x1)
        U2 = model._compute_sqrt(params, x2)

        # Should be different
        assert not jnp.allclose(U1, U2, atol=1e-6)

    def test_sigma_equals_UUT_plus_jitter(self):
        """Verify Σ(x) = U(x) @ U(x)^T + diag_term * I."""
        diag_term = 0.01
        model = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=3),
            likelihood=OddityTask(),
            noise=GaussianNoise(),
            diag_term=diag_term,
        )

        params = model.init_params(jr.PRNGKey(0))
        x = jnp.array([0.5, 0.3])

        U = model._compute_sqrt(params, x)
        Sigma = model.local_covariance(params, x)

        # Manual computation
        expected_Sigma = U @ U.T + diag_term * jnp.eye(model.input_dim)

        assert jnp.allclose(Sigma, expected_Sigma, atol=1e-6)
