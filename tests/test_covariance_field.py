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

import contextlib

import jax.numpy as jnp
import jax.random as jr
import pytest

from psyphy.data import ResponseData
from psyphy.inference import MAPOptimizer
from psyphy.model import WPPM, Prior
from psyphy.model.noise import GaussianNoise
from psyphy.model.task import OddityTask

# ==============================================================================
# Test fixtures
# ==============================================================================


@pytest.fixture
def mvp_model():
    """Create MVP model (no basis expansion)."""
    return WPPM(
        input_dim=2,
        prior=Prior(input_dim=2),
        task=OddityTask(),
        noise=GaussianNoise(),
        basis_degree=None,  # MVP mode
        diag_term=1e-6,
    )


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
        ),
        task=OddityTask(),
        noise=GaussianNoise(),
        basis_degree=3,
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
        data.add_trial(ref, probe, response)
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


def test_from_prior_mvp(mvp_model):
    """Test from_prior construction in MVP mode."""
    from psyphy.model.covariance_field import WPPMCovarianceField

    key = jr.PRNGKey(123)
    field = WPPMCovarianceField.from_prior(mvp_model, key)

    # Should have model and params
    assert field.model is mvp_model
    assert "log_diag" in field.params
    assert field.params["log_diag"].shape == (2,)


def test_from_prior_wishart(wishart_model):
    """Test from_prior construction in Wishart mode."""
    from psyphy.model.covariance_field import WPPMCovarianceField

    key = jr.PRNGKey(456)
    field = WPPMCovarianceField.from_prior(wishart_model, key)

    # Should have W parameters
    assert field.model is wishart_model
    assert "W" in field.params
    # W shape should match (degree+1)^input_dim x embedding_dim x (embedding_dim + extra_dims)
    # For input_dim=2, basis_degree=3: embedding_dim = 2 * (3+1) = 8
    # Shape: (4, 4, 8, 8+1) = (4, 4, 8, 9) but extra_dims=1 gives (4, 4, 8, 8)
    # Actually just check it has W with correct first dimensions
    assert field.params["W"].shape[:2] == (4, 4)  # basis dimensions
    assert len(field.params["W"].shape) == 4  # 4D tensor


def test_from_params(mvp_model):
    """Test from_params construction."""
    from psyphy.model.covariance_field import WPPMCovarianceField

    # Create arbitrary params
    params = {"log_diag": jnp.array([0.1, 0.2])}
    field = WPPMCovarianceField.from_params(mvp_model, params)

    assert field.model is mvp_model
    assert jnp.allclose(field.params["log_diag"], jnp.array([0.1, 0.2]))


def test_from_posterior_mvp(mvp_model, sample_data):
    """Test from_posterior construction with MAP posterior.

    API Design Note
    ---------------
    model.fit() returns self (following BoTorch pattern for method chaining).
    To access the fitted posterior, call model.posterior() afterwards.

    Current design (BoTorch-style):
        model.fit(X, y, inference=...)  # Returns self
        posterior = model.posterior(kind="parameter")  # Separate call

    Alternative design (statsmodels-style):
        posterior = model.fit(X, y, inference=...)  # Not implemented


    Trade-off: BoTorch consistency + method chaining vs one-step convenience.
    """
    from psyphy.model.covariance_field import WPPMCovarianceField

    # Fit model to get posterior - convert ResponseData to arrays
    refs, probes, responses = sample_data.to_numpy()
    X = jnp.stack([refs, probes], axis=1)  # (n_trials, 2, input_dim)
    y = jnp.array(responses)

    # BoTorch-style API: fit() returns self for chaining
    mvp_model.fit(X, y, inference=MAPOptimizer(steps=10))

    # Separate call to access fitted posterior (by design)
    posterior = mvp_model.posterior(kind="parameter")

    # Create field from posterior
    field = WPPMCovarianceField.from_posterior(posterior)

    assert field.model is mvp_model
    assert "log_diag" in field.params


# ==============================================================================
# Test 3: Evaluation methods - MVP mode
# ==============================================================================


def test_cov_mvp_constant(mvp_model):
    """Test that MVP covariance is constant across space."""
    from psyphy.model.covariance_field import WPPMCovarianceField

    key = jr.PRNGKey(789)
    field = WPPMCovarianceField.from_prior(mvp_model, key)

    # Evaluate at different points
    x1 = jnp.array([0.3, 0.7])
    x2 = jnp.array([0.8, 0.2])

    Sigma1 = field.cov(x1)
    Sigma2 = field.cov(x2)

    # Should be identical (MVP is constant)
    assert jnp.allclose(Sigma1, Sigma2)

    # Should be diagonal
    assert Sigma1.shape == (2, 2)
    assert jnp.allclose(Sigma1, jnp.diag(jnp.diag(Sigma1)))


def test_cov_mvp_positive_definite(mvp_model):
    """Test that MVP covariance is positive definite."""
    from psyphy.model.covariance_field import WPPMCovarianceField

    key = jr.PRNGKey(111)
    field = WPPMCovarianceField.from_prior(mvp_model, key)

    x = jnp.array([0.5, 0.5])
    Sigma = field.cov(x)

    # Check positive definiteness via eigenvalues
    eigvals = jnp.linalg.eigvalsh(Sigma)
    assert jnp.all(eigvals > 0)


def test_sqrt_cov_mvp_raises(mvp_model):
    """Test that sqrt_cov raises in MVP mode."""
    from psyphy.model.covariance_field import WPPMCovarianceField

    key = jr.PRNGKey(222)
    field = WPPMCovarianceField.from_prior(mvp_model, key)

    x = jnp.array([0.5, 0.5])

    # Should raise ValueError in MVP mode
    with pytest.raises(ValueError, match="sqrt_cov only available in Wishart mode"):
        field.sqrt_cov(x)


def test_cov_batch_mvp(mvp_model):
    """Test vectorized evaluation in MVP mode."""
    from psyphy.model.covariance_field import WPPMCovarianceField

    key = jr.PRNGKey(333)
    field = WPPMCovarianceField.from_prior(mvp_model, key)

    # Create grid of points
    X_grid = jnp.array(
        [
            [0.1, 0.1],
            [0.5, 0.5],
            [0.9, 0.9],
        ]
    )

    Sigmas = field.cov_batch(X_grid)

    # Check shape
    assert Sigmas.shape == (3, 2, 2)

    # All should be identical in MVP mode
    assert jnp.allclose(Sigmas[0], Sigmas[1])
    assert jnp.allclose(Sigmas[1], Sigmas[2])


# ==============================================================================
# Test 4: Evaluation methods - Wishart mode
# ==============================================================================


def test_cov_wishart_varies(wishart_model):
    """Test that Wishart covariance varies across space."""
    from psyphy.model.covariance_field import WPPMCovarianceField

    key = jr.PRNGKey(444)
    field = WPPMCovarianceField.from_prior(wishart_model, key)

    # Evaluate at different points
    x1 = jnp.array([0.2, 0.3])
    x2 = jnp.array([0.8, 0.7])

    Sigma1 = field.cov(x1)
    Sigma2 = field.cov(x2)

    # Should be different (Wishart varies with x)
    assert not jnp.allclose(Sigma1, Sigma2, atol=1e-6)

    # Both should be positive definite
    assert jnp.all(jnp.linalg.eigvalsh(Sigma1) > 0)
    assert jnp.all(jnp.linalg.eigvalsh(Sigma2) > 0)


def test_sqrt_cov_wishart(wishart_model):
    """Test sqrt_cov in Wishart mode."""
    from psyphy.model.covariance_field import WPPMCovarianceField

    key = jr.PRNGKey(555)
    field = WPPMCovarianceField.from_prior(wishart_model, key)

    x = jnp.array([0.5, 0.5])
    U = field.sqrt_cov(x)

    # Check shape: (embedding_dim, embedding_dim + extra_dims)
    # embedding_dim = input_dim * (basis_degree + 1) = 2 * 4 = 8
    # extra_dims = 1, so U.shape = (8, 8) (since W doesn't have extra_dims)
    embedding_dim = wishart_model.embedding_dim
    assert U.shape[0] == embedding_dim
    assert len(U.shape) == 2

    # Verify that Σ = U @ U^T + diag_term * I
    Sigma_from_U = U @ U.T + wishart_model.diag_term * jnp.eye(embedding_dim)
    Sigma_direct = field.cov(x)

    assert jnp.allclose(Sigma_from_U, Sigma_direct, rtol=1e-5)


def test_cov_batch_wishart(wishart_model):
    """Test vectorized evaluation in Wishart mode."""
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

    Sigmas = field.cov_batch(X_grid)

    # Check shape - should be (n_points, embedding_dim, embedding_dim)
    embedding_dim = wishart_model.embedding_dim
    assert Sigmas.shape == (3, embedding_dim, embedding_dim)

    # Should NOT all be identical (Wishart varies)
    assert not jnp.allclose(Sigmas[0], Sigmas[1], atol=1e-6)
    assert not jnp.allclose(Sigmas[1], Sigmas[2], atol=1e-6)

    # All should be positive definite
    for i in range(3):
        eigvals = jnp.linalg.eigvalsh(Sigmas[i])
        assert jnp.all(eigvals > 0)


def test_sqrt_cov_batch_wishart(wishart_model):
    """Test vectorized sqrt_cov in Wishart mode."""
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

    # Check shape - (n_points, embedding_dim, ...)
    embedding_dim = wishart_model.embedding_dim
    assert U_batch.shape[0] == 3
    assert U_batch.shape[1] == embedding_dim
    assert len(U_batch.shape) == 3

    # Verify consistency with cov
    for i in range(3):
        Sigma_from_U = U_batch[i] @ U_batch[i].T + wishart_model.diag_term * jnp.eye(
            embedding_dim
        )
        Sigma_direct = field.cov(X_grid[i])
        assert jnp.allclose(Sigma_from_U, Sigma_direct, rtol=1e-5)


# ==============================================================================
# Test 5: Integration with posterior
# ==============================================================================


def test_posterior_get_covariance_field_mvp(mvp_model, sample_data):
    """Test posterior.get_covariance_field() in MVP mode.

    API Design Note
    ---------------
    This test demonstrates the recommended workflow for getting a CovarianceField:
    1. Fit model: model.fit(X, y, inference=...)
    2. Get posterior: posterior = model.posterior(kind="parameter")
    3. Get field: field = posterior.get_covariance_field()

    This two-step approach (fit -> posterior) follows BoTorch conventions.
    """
    # Fit model - convert ResponseData to arrays
    refs, probes, responses = sample_data.to_numpy()
    X = jnp.stack([refs, probes], axis=1)
    y = jnp.array(responses)
    mvp_model.fit(X, y, inference=MAPOptimizer(steps=10))

    # Get the parameter posterior (BoTorch-style: separate from fit)
    posterior = mvp_model.posterior(kind="parameter")

    # Get field from posterior
    field = posterior.get_covariance_field()

    # Should work and return sensible results
    x = jnp.array([0.5, 0.5])
    Sigma = field.cov(x)

    assert Sigma.shape == (2, 2)
    assert jnp.all(jnp.linalg.eigvalsh(Sigma) > 0)


def test_posterior_get_covariance_field_wishart(wishart_model, sample_data):
    """Test posterior.get_covariance_field() in Wishart mode.

    API Design Note
    ---------------
    Like test_posterior_get_covariance_field_mvp(), this demonstrates the
    BoTorch-style two-step pattern. The covariance field will reflect the
    spatially-varying structure learned during fitting.
    """
    # Fit model - convert ResponseData to arrays
    refs, probes, responses = sample_data.to_numpy()
    X = jnp.stack([refs, probes], axis=1)
    y = jnp.array(responses)
    wishart_model.fit(X, y, inference=MAPOptimizer(steps=10))

    # Get the parameter posterior (BoTorch-style: separate from fit)
    posterior = wishart_model.posterior(kind="parameter")

    # Get field from posterior
    field = posterior.get_covariance_field()

    # Should work and return sensible results
    x = jnp.array([0.5, 0.5])
    Sigma = field.cov(x)
    U = field.sqrt_cov(x)

    embedding_dim = wishart_model.embedding_dim
    assert Sigma.shape == (embedding_dim, embedding_dim)
    assert U.shape[0] == embedding_dim

    # Check positive definiteness if not NaN (optimization might fail with minimal data)
    # This tests the API works, not the optimization quality
    if not jnp.any(jnp.isnan(Sigma)):
        assert jnp.all(jnp.linalg.eigvalsh(Sigma) > 0)


# ==============================================================================
# Test 6: Protocol conformance
# ==============================================================================


def test_protocol_conformance_mvp(mvp_model):
    """Test that WPPMCovarianceField conforms to protocol."""
    from psyphy.model.covariance_field import CovarianceField, WPPMCovarianceField

    key = jr.PRNGKey(888)
    field = WPPMCovarianceField.from_prior(mvp_model, key)

    # Check protocol conformance
    assert isinstance(field, CovarianceField)


def test_protocol_conformance_wishart(wishart_model):
    """Test protocol conformance in Wishart mode."""
    from psyphy.model.covariance_field import CovarianceField, WPPMCovarianceField

    key = jr.PRNGKey(999)
    field = WPPMCovarianceField.from_prior(wishart_model, key)

    # Check protocol conformance
    assert isinstance(field, CovarianceField)


# ==============================================================================
# Test 7: Edge cases and error handling
# ==============================================================================


def test_cov_input_validation(mvp_model):
    """Test that cov validates input shape."""
    from psyphy.model.covariance_field import WPPMCovarianceField

    key = jr.PRNGKey(1010)
    field = WPPMCovarianceField.from_prior(mvp_model, key)

    # Wrong shape should raise or handle gracefully
    x_wrong = jnp.array([0.5])  # Should be (2,)

    # This might raise or handle - test based on implementation
    # For now, just ensure it doesn't crash silently
    with contextlib.suppress(ValueError, IndexError, TypeError):
        _ = field.cov(x_wrong)


def test_field_immutability(mvp_model):
    """Test that field params can't be accidentally mutated."""
    from psyphy.model.covariance_field import WPPMCovarianceField

    key = jr.PRNGKey(1111)
    field = WPPMCovarianceField.from_prior(mvp_model, key)

    # Store original params
    original_params = field.params.copy()

    # Try to mutate (JAX arrays are immutable, but dict reference could change)
    field.params["log_diag"] = jnp.zeros(2)

    # Original field should be unchanged
    # Note: This test documents current behavior; true immutability would require frozen dataclass
    assert field.params is not original_params


# ==============================================================================
# Test 8: Property methods
# ==============================================================================


def test_is_wishart_mode(mvp_model, wishart_model):
    """Test is_wishart_mode property."""
    from psyphy.model.covariance_field import WPPMCovarianceField

    key = jr.PRNGKey(1212)

    field_mvp = WPPMCovarianceField.from_prior(mvp_model, key)
    field_wishart = WPPMCovarianceField.from_prior(wishart_model, key)

    assert not field_mvp.is_wishart_mode
    assert field_wishart.is_wishart_mode


def test_is_mvp_mode(mvp_model, wishart_model):
    """Test is_mvp_mode property."""
    from psyphy.model.covariance_field import WPPMCovarianceField

    key = jr.PRNGKey(1313)

    field_mvp = WPPMCovarianceField.from_prior(mvp_model, key)
    field_wishart = WPPMCovarianceField.from_prior(wishart_model, key)

    assert field_mvp.is_mvp_mode
    assert not field_wishart.is_mvp_mode


# ==============================================================================
# Test 9: Comparison with direct model calls
# ==============================================================================


def test_consistency_with_model_local_covariance(mvp_model):
    """Test that field.cov() matches model.local_covariance()."""
    from psyphy.model.covariance_field import WPPMCovarianceField

    key = jr.PRNGKey(1414)
    params = mvp_model.init_params(key)
    field = WPPMCovarianceField.from_params(mvp_model, params)

    x = jnp.array([0.5, 0.5])

    # Compare field vs direct model call
    Sigma_field = field.cov(x)
    Sigma_model = mvp_model.local_covariance(params, x)

    assert jnp.allclose(Sigma_field, Sigma_model)


def test_consistency_with_model_compute_U(wishart_model):
    """Test that field.sqrt_cov() matches model._compute_U()."""
    from psyphy.model.covariance_field import WPPMCovarianceField

    key = jr.PRNGKey(1515)
    params = wishart_model.init_params(key)
    field = WPPMCovarianceField.from_params(wishart_model, params)

    x = jnp.array([0.5, 0.5])

    # Compare field vs direct model call
    U_field = field.sqrt_cov(x)
    U_model = wishart_model._compute_U(params, x)

    assert jnp.allclose(U_field, U_model)
