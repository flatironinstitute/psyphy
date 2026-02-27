"""
Tests for rectangular U matrix design (Hong et al.)

We test that U(x) has shape (input_dim, embedding_dim) (rectangular),
as opposed to being square (embedding_dim x embedding_dim), where we then would
need to extract blocks for stimulus covariance. The square design could
be useful in some contexts, for example when modeling another cognitive
process, eg attantion.

Note:
- W coefficients: (degree+1, degree+1, input_dim, embedding_dim)
- U(x) matrices: (input_dim, embedding_dim)
- resulting Σ(x) = U @ U^T: (input_dim, input_dim) directly
"""

import jax.numpy as jnp
import jax.random as jr
import pytest

from psyphy.model import WPPM, OddityTask, Prior


class TestRectangularUShape:
    """Test that U matrices have correct rectangular shape."""

    @pytest.mark.parametrize(
        "input_dim,extra_dims,basis_degree",
        [
            (2, 0, 3),  # 2D, no extra dims -> U is (2, 2) square
            (2, 1, 3),  # 2D, 1 extra -> U is (2, 3) rectangular
            (2, 2, 5),  # 2D, 2 extra -> U is (2, 4) rectangular
            (3, 0, 3),  # 3D, no extra -> U is (3, 3) square
            (3, 1, 3),  # 3D, 1 extra -> U is (3, 4) rectangular
        ],
    )
    def test_W_shape_is_rectangular(self, input_dim, extra_dims, basis_degree):
        """W coefficients should have shape (degree+1, ..., input_dim, input_dim+extra_dims)."""
        prior = Prior(
            input_dim=input_dim,
            basis_degree=basis_degree,
            extra_embedding_dims=extra_dims,
        )

        params = prior.sample_params(jr.PRNGKey(42))
        W = params["W"]

        embedding_dim = input_dim + extra_dims

        if input_dim == 2:
            # Expected: (degree+1, degree+1, input_dim, embedding_dim)
            expected_shape = (
                basis_degree + 1,
                basis_degree + 1,
                input_dim,
                embedding_dim,
            )
        elif input_dim == 3:
            # Expected: (degree+1, degree+1, degree+1, input_dim, embedding_dim)
            expected_shape = (
                basis_degree + 1,
                basis_degree + 1,
                basis_degree + 1,
                input_dim,
                embedding_dim,
            )
        else:
            raise ValueError(f"Unsupported input_dim: {input_dim}")

        assert W.shape == expected_shape, (
            f"W shape mismatch for input_dim={input_dim}, extra_dims={extra_dims}. "
            f"Expected {expected_shape}, got {W.shape}"
        )

    @pytest.mark.parametrize(
        "input_dim,extra_dims",
        [
            (2, 0),
            (2, 1),
            (2, 2),
            (3, 0),
            (3, 1),
        ],
    )
    def test_U_matrix_shape(self, input_dim, extra_dims):
        """U(x) should be (input_dim, input_dim + extra_dims)."""
        prior = Prior(
            input_dim=input_dim, basis_degree=5, extra_embedding_dims=extra_dims
        )
        task = OddityTask()
        model = WPPM(input_dim=input_dim, extra_dims=extra_dims, prior=prior, task=task)

        params = prior.sample_params(jr.PRNGKey(42))
        x = jnp.ones(input_dim) * 0.5

        U = model._compute_sqrt(params, x)

        embedding_dim = input_dim + extra_dims
        expected_shape = (input_dim, embedding_dim)

        assert U.shape == expected_shape, (
            f"U shape mismatch for input_dim={input_dim}, extra_dims={extra_dims}. "
            f"Expected {expected_shape}, got {U.shape}"
        )


class TestLocalCovarianceShape:
    """Test that local_covariance returns (input_dim, input_dim) directly."""

    @pytest.mark.parametrize(
        "input_dim,extra_dims",
        [
            (2, 0),
            (2, 1),
            (2, 3),
            (3, 0),
            (3, 1),
            (3, 2),
        ],
    )
    def test_local_covariance_is_stimulus_size(self, input_dim, extra_dims):
        """Σ(x) should be (input_dim, input_dim), not (embedding_dim, embedding_dim)."""
        prior = Prior(
            input_dim=input_dim, basis_degree=5, extra_embedding_dims=extra_dims
        )
        task = OddityTask()
        model = WPPM(input_dim=input_dim, extra_dims=extra_dims, prior=prior, task=task)

        params = prior.sample_params(jr.PRNGKey(42))
        x = jnp.ones(input_dim) * 0.5

        Sigma = model.local_covariance(params, x)

        # Should be (input_dim, input_dim) regardless of extra_dims
        expected_shape = (input_dim, input_dim)

        assert Sigma.shape == expected_shape, (
            f"Σ shape mismatch for input_dim={input_dim}, extra_dims={extra_dims}. "
            f"Expected {expected_shape}, got {Sigma.shape}"
        )

    def test_local_covariance_positive_definite(self):
        """Σ(x) = U @ U^T + diag_term*I should be positive definite."""
        prior = Prior(input_dim=2, basis_degree=5, extra_embedding_dims=1)
        task = OddityTask()
        model = WPPM(input_dim=2, extra_dims=1, prior=prior, task=task)

        params = prior.sample_params(jr.PRNGKey(42))
        x = jnp.array([0.3, 0.7])

        Sigma = model.local_covariance(params, x)

        # Check positive definite via eigenvalues
        eigvals = jnp.linalg.eigvalsh(Sigma)
        assert jnp.all(eigvals > 0), f"Σ not positive definite. Eigenvalues: {eigvals}"


class TestDiscriminabilityNoExtraction:
    """Test that discriminability works directly without block extraction."""

    def test_discriminability_no_extraction_needed(self):
        """discriminability should use Σ directly, no [:input_dim, :input_dim] needed."""
        prior = Prior(input_dim=2, basis_degree=5, extra_embedding_dims=1)
        task = OddityTask()
        model = WPPM(input_dim=2, extra_dims=1, prior=prior, task=task)

        params = prior.sample_params(jr.PRNGKey(42))
        x_ref = jnp.array([0.5, 0.5])
        x_odd = jnp.array([0.6, 0.5])
        stimulus = (x_ref, x_odd)

        # Should compute without errors
        d = model.discriminability(params, stimulus)

        assert d.shape == (), "Discriminability should be scalar"
        assert d >= 0, "Discriminability should be non-negative"

    @pytest.mark.parametrize("input_dim", [2, 3])
    def test_discriminability_matches_mahalanobis(self, input_dim):
        """Discriminability should equal Mahalanobis distance."""
        prior = Prior(input_dim=input_dim, basis_degree=5, extra_embedding_dims=1)
        task = OddityTask()
        model = WPPM(
            input_dim=input_dim, extra_dims=1, prior=prior, task=task, diag_term=1e-6
        )

        params = prior.sample_params(jr.PRNGKey(42))
        x_ref = jnp.ones(input_dim) * 0.5
        x_odd = x_ref.at[0].set(0.6)
        stimulus = (x_ref, x_odd)

        d = model.discriminability(params, stimulus)

        # Manual computation (discriminability returns sqrt of quadratic form)
        Sigma = model.local_covariance(params, x_ref)
        delta = x_odd - x_ref
        d2_manual = delta @ jnp.linalg.solve(Sigma, delta)
        d_manual = jnp.sqrt(d2_manual)

        assert jnp.allclose(d, d_manual, atol=1e-4), (
            f"Discriminability mismatch. Model: {d}, Manual: {d_manual}"
        )


class TestParameterCount:
    """Test that rectangular design uses fewer parameters."""

    def test_parameter_count_2d(self):
        """For 2D with extra_dims=1, rectangular uses fewer params than old square design."""
        prior = Prior(input_dim=2, basis_degree=5, extra_embedding_dims=1)
        params = prior.sample_params(jr.PRNGKey(42))
        W = params["W"]

        # Rectangular: (6, 6, 2, 3) = 216 params
        assert W.shape == (6, 6, 2, 3)
        assert W.size == 216

        # Old square design would be: (6, 6, 3, 3) = 324 params
        # We save 108 parameters (33%)!

    def test_parameter_count_3d(self):
        """For 3D with extra_dims=1, rectangular uses fewer params."""
        prior = Prior(input_dim=3, basis_degree=5, extra_embedding_dims=1)
        params = prior.sample_params(jr.PRNGKey(42))
        W = params["W"]

        # Rectangular: (6, 6, 6, 3, 4) = 2592 params
        assert W.shape == (6, 6, 6, 3, 4)
        assert W.size == 2592

        # Old square design would be: (6, 6, 6, 4, 4) = 3456 params
        # We save 864 parameters (25%)!


class TestCovarianceFieldProtocol:
    """Test that covariance field returns correct shapes."""

    def test_covariance_field_cov_shape(self):
        """CovarianceField.cov() should return (input_dim, input_dim)."""
        from psyphy.model.covariance_field import WPPMCovarianceField

        prior = Prior(input_dim=2, basis_degree=5, extra_embedding_dims=1)
        task = OddityTask()
        model = WPPM(input_dim=2, extra_dims=1, prior=prior, task=task)

        params = prior.sample_params(jr.PRNGKey(42))
        field = WPPMCovarianceField(model, params)

        x = jnp.array([0.5, 0.5])
        Sigma = field.cov(x)

        assert Sigma.shape == (2, 2), f"Expected (2, 2), got {Sigma.shape}"

    def test_sqrt_cov_shape(self):
        """CovarianceField.sqrt_cov() should return (input_dim, embedding_dim)."""
        from psyphy.model.covariance_field import WPPMCovarianceField

        prior = Prior(input_dim=2, basis_degree=5, extra_embedding_dims=1)
        task = OddityTask()
        model = WPPM(input_dim=2, extra_dims=1, prior=prior, task=task)

        params = prior.sample_params(jr.PRNGKey(42))
        field = WPPMCovarianceField(model, params)

        x = jnp.array([0.5, 0.5])
        U = field.sqrt_cov(x)

        # U should be rectangular: (input_dim, embedding_dim) = (2, 3)
        assert U.shape == (2, 3), f"Expected (2, 3), got {U.shape}"

    def test_no_cov_stimulus_method_needed(self):
        """cov_stimulus() method should not exist - cov() already returns stimulus size."""
        from psyphy.model.covariance_field import WPPMCovarianceField

        prior = Prior(input_dim=2, basis_degree=5, extra_embedding_dims=1)
        task = OddityTask()
        model = WPPM(input_dim=2, extra_dims=1, prior=prior, task=task)

        params = prior.sample_params(jr.PRNGKey(42))
        field = WPPMCovarianceField(model, params)

        # cov_stimulus should not exist in new design
        # (or if it exists, should just return cov() unchanged)
        assert not hasattr(field, "cov_stimulus") or (
            jnp.array_equal(
                field.cov(jnp.array([0.5, 0.5])),
                field.cov_stimulus(jnp.array([0.5, 0.5])),
            )
        )
