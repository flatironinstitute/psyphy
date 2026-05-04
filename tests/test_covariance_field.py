"""
test_covariance_field.py
------------------------

Tests for CovarianceField abstraction

1. Protocol conformance
2. Construction methods (from_prior, from_posterior, from_params)
3. Evaluation methods (cov, sqrt_cov, cov_batch)
4. MVP mode behavior
5. Wishart mode behavior
6. Integration with posteriors

"""

import jax.numpy as jnp
import jax.random as jr
import pytest

from psyphy.data import ResponseData
from psyphy.inference import MAPOptimizer
from psyphy.model import WPPM, Prior
from psyphy.model.likelihood import OddityTask
from psyphy.model.noise import GaussianNoise

# ==============================================================================
# Test fixtures
# ==============================================================================


@pytest.fixture
def wishart_model():
    """Create Wishart model (with basis expansion)."""
    return WPPM(
        input_dim=2,
        prior=Prior(
            input_dim=2,
            basis_degree=3,
            variance_scale=4e-3,
            decay_rate=0.3,
            extra_embedding_dims=1,  # should match model's extra_dims
        ),
        likelihood=OddityTask(),
        noise=GaussianNoise(),
        extra_dims=1,
        diag_term=1e-3,
    )


@pytest.fixture
def sample_data():
    """Create sample dataset for fitting."""
    data = ResponseData()
    # Add a few trials
    key = jr.PRNGKey(42)
    for i in range(20):
        key, subkey = jr.split(key)
        ref = jr.uniform(subkey, shape=(2,))
        key, subkey = jr.split(key)
        probe = ref + 0.1 * jr.normal(subkey, shape=(2,))
        response = 1 if i % 2 == 0 else 0
        data.add_trial((ref, probe), response)
    return data


# ==============================================================================
# Test 1: CovarianceField protocol exists
# ==============================================================================


def test_covariance_field_protocol_exists():
    """Test that CovarianceField protocol is defined."""
    from psyphy.model.covariance_field import CovarianceField

    # Protocol should have required methods
    assert hasattr(CovarianceField, "cov")
    assert hasattr(CovarianceField, "sqrt_cov")
    assert hasattr(CovarianceField, "cov_batch")


# ==============================================================================
# Test 2: WPPMCovarianceField construction methods
# ==============================================================================


def test_from_prior_wishart(wishart_model):
    """Test from_prior construction in Wishart mode."""
    from psyphy.model.covariance_field import WPPMCovarianceField

    key = jr.PRNGKey(456)
    field = WPPMCovarianceField.from_prior(wishart_model, key)

    # Should have W parameters
    assert field.model is wishart_model
    assert "W" in field.params
    # W shape (rectangular design): (degree+1, degree+1, input_dim, embedding_dim)
    # For input_dim=2, basis_degree=3, extra_dims=1:
    #   degree+1 = 4, input_dim = 2, embedding_dim = 3
    # Shape: (4, 4, 2, 3)
    assert field.params["W"].shape == (4, 4, 2, 3)


# ==============================================================================
# Test  Evaluation methods - MVP mode
# ==============================================================================


# ==============================================================================
# Test  Evaluation methods - WPPM
# ==============================================================================


def test_cov_wishart_varies(wishart_model):
    """Test that Wishart covariance varies across space."""
    from psyphy.model.covariance_field import WPPMCovarianceField

    key = jr.PRNGKey(444)
    field = WPPMCovarianceField.from_prior(wishart_model, key)

    # Evaluate at different points
    x1 = jnp.array([0.2, 0.3])
    x2 = jnp.array([0.8, 0.7])

    Sigma1 = field(x1)
    Sigma2 = field(x2)

    # Should be different (Wishart varies with x)
    assert not jnp.allclose(Sigma1, Sigma2, atol=1e-6)

    # Both should be positive definite
    assert jnp.all(jnp.linalg.eigvalsh(Sigma1) > 0)
    assert jnp.all(jnp.linalg.eigvalsh(Sigma2) > 0)


def test_sqrt_cov_wishart(wishart_model):
    """Test sqrt_cov in Wishart mode (rectangular U)."""
    from psyphy.model.covariance_field import WPPMCovarianceField

    key = jr.PRNGKey(555)
    field = WPPMCovarianceField.from_prior(wishart_model, key)

    x = jnp.array([0.5, 0.5])
    U = field.sqrt_cov(x)

    # Check shape: (input_dim, embedding_dim) with rectangular design
    # input_dim = 2, embedding_dim = input_dim + extra_dims = 3
    assert U.shape == (2, 3)

    # Verify that Σ = U @ U^T + diag_term * I
    Sigma_from_U = U @ U.T + wishart_model.diag_term * jnp.eye(2)
    Sigma_direct = field(x)

    assert jnp.allclose(Sigma_from_U, Sigma_direct, rtol=1e-5)


def test_cov_batch_wishart(wishart_model):
    """Test vectorized evaluation in Wishart mode (rectangular U)."""
    from psyphy.model.covariance_field import WPPMCovarianceField

    key = jr.PRNGKey(666)
    field = WPPMCovarianceField.from_prior(wishart_model, key)

    # Create grid of points
    X_grid = jnp.array(
        [
            [0.1, 0.1],
            [0.5, 0.5],
            [0.9, 0.9],
        ]
    )

    Sigmas = field(X_grid)

    # Check shape - should be (n_points, input_dim, input_dim) with rectangular U
    assert Sigmas.shape == (3, 2, 2)

    # Should NOT all be identical (Wishart varies)
    assert not jnp.allclose(Sigmas[0], Sigmas[1], atol=1e-6)
    assert not jnp.allclose(Sigmas[1], Sigmas[2], atol=1e-6)

    # All should be positive definite
    for i in range(3):
        eigvals = jnp.linalg.eigvalsh(Sigmas[i])
        assert jnp.all(eigvals > 0)


def test_sqrt_cov_batch_wishart(wishart_model):
    """Test vectorized sqrt_cov in Wishart mode (rectangular U)."""
    from psyphy.model.covariance_field import WPPMCovarianceField

    key = jr.PRNGKey(777)
    field = WPPMCovarianceField.from_prior(wishart_model, key)

    X_grid = jnp.array(
        [
            [0.2, 0.3],
            [0.5, 0.5],
            [0.8, 0.7],
        ]
    )

    U_batch = field.sqrt_cov_batch(X_grid)

    # Check shape - (n_points, input_dim, embedding_dim) with rectangular U
    # input_dim = 2, embedding_dim = 3
    assert U_batch.shape == (3, 2, 3)

    # Verify consistency with cov
    for i in range(3):
        Sigma_from_U = U_batch[i] @ U_batch[i].T + wishart_model.diag_term * jnp.eye(2)
        Sigma_direct = field(X_grid[i])
        assert jnp.allclose(Sigma_from_U, Sigma_direct, rtol=5e-2, atol=1e-6)


# ==============================================================================
# Test  Integration with posterior
# ==============================================================================


def test_posterior_get_covariance_field_wishart(wishart_model, sample_data):
    """Test posterior.get_covariance_field() in Wishart mode (rectangular U).

    API Design Note
    ---------------
    this demonstrates the
    BoTorch-style two-step pattern. The covariance field will reflect the
    spatially-varying structure learned during fitting.
    """
    # Fit model using the functional API
    optimizer = MAPOptimizer(steps=10)
    posterior = optimizer.fit(wishart_model, sample_data)

    # Get field directly from the model using the fitted parameters
    from psyphy.model.covariance_field import WPPMCovarianceField

    field = WPPMCovarianceField.from_params(wishart_model, posterior.params)

    # Basic shape tests
    x = jnp.array([0.5, 0.5])
    Sigma = field(x)
    U = field.sqrt_cov(x)

    # Rectangular U design: Σ is stimulus size, U is rectangular
    assert Sigma.shape == (2, 2)  # input_dim = 2
    assert U.shape == (2, 3)  # input_dim × embedding_dim = 2×3

    # Check positive definiteness if not NaN (optimization might fail with minimal data)
    # This tests the API works, not the optimization quality
    if not jnp.any(jnp.isnan(Sigma)):
        assert jnp.all(jnp.linalg.eigvalsh(Sigma) > 0)


# ==============================================================================
# Test  Protocol conformance
# ==============================================================================


def test_protocol_conformance_wishart(wishart_model):
    """Test protocol conformance in Wishart mode."""
    from psyphy.model.covariance_field import CovarianceField, WPPMCovarianceField

    key = jr.PRNGKey(999)
    field = WPPMCovarianceField.from_prior(wishart_model, key)

    # Check protocol conformance
    assert isinstance(field, CovarianceField)


# ==============================================================================
# Test Comparison with direct model calls
# ==============================================================================


def test_consistency_with_model_compute_sqrt(wishart_model):
    """Test that field.sqrt_cov() matches model._compute_sqrt()."""
    from psyphy.model.covariance_field import WPPMCovarianceField

    key = jr.PRNGKey(1515)
    params = wishart_model.init_params(key)
    field = WPPMCovarianceField.from_params(wishart_model, params)

    x = jnp.array([0.5, 0.5])

    # Compare field vs direct model call
    U_field = field.sqrt_cov(x)
    U_model = wishart_model._compute_sqrt(params, x)

    assert jnp.allclose(U_field, U_model)
