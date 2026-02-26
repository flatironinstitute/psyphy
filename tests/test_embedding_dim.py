"""
test_embedding_dim.py
-------------------------------

Tests for embedding_dim:  making it a computed property = input_dim + extra_dims.
"""

import jax.numpy as jnp
import jax.random as jr
import pytest

from psyphy.model import WPPM, GaussianNoise, OddityTask, Prior
from psyphy.model.covariance_field import WPPMCovarianceField

# ==============================================================================
# Test WPPM constructor should NOT accept embedding_dim parameter
# ==============================================================================


def test_wppm_no_embedding_dim_parameter():
    """WPPM should not have embedding_dim as constructor parameter."""
    # This should work WITHOUT embedding_dim
    model = WPPM(
        input_dim=2,
        prior=Prior(input_dim=2, basis_degree=3),  # Need Wishart mode for extra_dims
        task=OddityTask(),
        noise=GaussianNoise(),
        extra_dims=1,
    )

    # embedding_dim should be computed property
    assert hasattr(model, "embedding_dim")
    assert model.embedding_dim == 2 + 1  # input_dim + extra_dims

    # Should NOT be settable in constructor
    with pytest.raises(TypeError):
        WPPM(
            input_dim=2,
            embedding_dim=5,  # This should fail
            prior=Prior(input_dim=2),
            task=OddityTask(),
            noise=GaussianNoise(),
        )


# ==============================================================================
# Test embedding_dim should be computed from input_dim + extra_dims
# ==============================================================================


@pytest.mark.parametrize(
    "input_dim,extra_dims,expected_embedding_dim",
    [
        (2, 0, 2),  # No extra dims
        (2, 1, 3),  # 1 extra dim
        (2, 3, 5),  # 3 extra dims
        (3, 0, 3),  # 3D input, no extra
        (3, 2, 5),  # 3D input, 2 extra
    ],
)
def test_embedding_dim_computed_correctly(
    input_dim, extra_dims, expected_embedding_dim
):
    """Test that embedding_dim = input_dim + extra_dims."""
    model = WPPM(
        input_dim=input_dim,
        prior=Prior(
            input_dim=input_dim, basis_degree=3
        ),  # Need Wishart mode for extra_dims
        task=OddityTask(),
        noise=GaussianNoise(),
        extra_dims=extra_dims,
    )

    assert model.embedding_dim == expected_embedding_dim


# ==============================================================================
# Test Wishart model W parameter shape should use full embedding space
# ==============================================================================


def test_wishart_W_shape_uses_full_embedding_dim():
    """W should have shape (degree+1, degree+1, input_dim, embedding_dim) (rectangular)."""
    model = WPPM(
        input_dim=2,
        prior=Prior(input_dim=2, basis_degree=3, extra_embedding_dims=1),
        task=OddityTask(),
        noise=GaussianNoise(),
        extra_dims=1,
    )

    key = jr.PRNGKey(42)
    params = model.init_params(key)

    W = params["W"]

    # W shape for 2D (rectangular design): (degree+1, degree+1, input_dim, embedding_dim)
    # where embedding_dim = input_dim + extra_dims = 2 + 1 = 3
    expected_shape = (4, 4, 2, 3)  # degree+1=4, input_dim=2, embedding_dim=3

    assert W.shape == expected_shape, f"Expected {expected_shape}, got {W.shape}"


# ==============================================================================
# Test local_covariance should return (input_dim, input_dim)
# ==============================================================================


def test_local_covariance_shape_wishart():
    """local_covariance should return (input_dim, input_dim) with rectangular U."""
    model = WPPM(
        input_dim=2,
        prior=Prior(input_dim=2, basis_degree=3, extra_embedding_dims=1),
        task=OddityTask(),
        noise=GaussianNoise(),
        extra_dims=1,
    )

    key = jr.PRNGKey(42)
    params = model.init_params(key)

    x = jnp.array([0.5, 0.5])
    Sigma = model.local_covariance(params, x)

    # Should be (input_dim, input_dim) = (2, 2) with rectangular U design
    assert Sigma.shape == (2, 2)


# ==============================================================================
# Test _compute_sqrt should return (embedding_dim, embedding_dim)
# ==============================================================================


def test_compute_sqrt_shape():
    """_compute_sqrt should return (input_dim, embedding_dim) with rectangular design."""
    model = WPPM(
        input_dim=2,
        prior=Prior(input_dim=2, basis_degree=3, extra_embedding_dims=1),
        task=OddityTask(),
        noise=GaussianNoise(),
        extra_dims=1,
    )

    key = jr.PRNGKey(42)
    params = model.init_params(key)

    x = jnp.array([0.5, 0.5])
    U = model._compute_sqrt(params, x)

    # Rectangular design: U should be (input_dim, embedding_dim)
    assert U.shape == (2, 3)  # input_dim=2, embedding_dim=3


# ==============================================================================
# Test  Verify positive definiteness in stimulus space
# ==============================================================================


def test_full_embedding_covariance_positive_definite():
    """Covariance in stimulus space should be positive definite with rectangular U."""
    model = WPPM(
        input_dim=2,
        prior=Prior(input_dim=2, basis_degree=3, extra_embedding_dims=2),
        task=OddityTask(),
        noise=GaussianNoise(),
        extra_dims=2,
    )

    key = jr.PRNGKey(42)
    params = model.init_params(key)

    x = jnp.array([0.5, 0.5])
    Sigma = model.local_covariance(params, x)

    # Check shape (rectangular U returns stimulus covariance)
    assert Sigma.shape == (2, 2)  # input_dim = 2

    # Check positive definiteness
    eigvals = jnp.linalg.eigvalsh(Sigma)
    assert jnp.all(eigvals > 0), f"Got eigenvalues: {eigvals}"


# ==============================================================================
# Test  Stimulus subspace should also be positive definite
# ==============================================================================


def test_stimulus_subspace_positive_definite():
    """Stimulus subspace covariance should be positive definite."""
    model = WPPM(
        input_dim=2,
        prior=Prior(input_dim=2, basis_degree=3, extra_embedding_dims=1),
        task=OddityTask(),
        noise=GaussianNoise(),
        extra_dims=1,
    )

    key = jr.PRNGKey(42)
    field = WPPMCovarianceField.from_prior(model, key)

    x = jnp.array([0.5, 0.5])
    Sigma_stim = field.cov(x)

    # Check positive definiteness
    eigvals = jnp.linalg.eigvalsh(Sigma_stim)
    assert jnp.all(eigvals > 0)
