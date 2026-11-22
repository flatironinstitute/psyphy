"""
test_covariance_field.py
------------------------

Tests for WPPMCovarianceField built from 1 parameter set: from the prior with
n_samples=1, from_params, and a MAP fit. Every result keeps the sample axis of 1.

1. Construction (from_prior, from_params, a MAP posterior)
2. Evaluation (field(x), sqrt_cov)
3. Covariance varies with x and is positive definite
4. Rectangular U (extra embedding dims)
5. Read-only attributes

"""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from psyphy.data import ResponseData
from psyphy.inference import MAPOptimizer
from psyphy.model import WPPM, Prior, WPPMCovarianceField
from psyphy.model.likelihood import OddityTask
from psyphy.model.noise import GaussianNoise

# ==============================================================================
# Test fixtures
# ==============================================================================


@pytest.fixture
def wishart_model():
    """Create a model with basis_degree=3."""
    return WPPM(
        prior=Prior(
            input_dim=2,
            basis_degree=3,
            variance_scale=4e-3,
            decay_rate=0.3,
            extra_embedding_dims=1,
        ),
        likelihood=OddityTask(),
        noise=GaussianNoise(),
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


def _sigma(model, params, X):
    """Σ at every row of X with model.local_covariance, (n_points, d, d)."""
    return jax.vmap(lambda x: model.local_covariance(params, x))(X)


# ==============================================================================
# Construction
# ==============================================================================


def test_from_prior_with_one_sample(wishart_model):
    """from_prior with n_samples=1 holds 1 parameter set, with the sample axis kept."""
    field = WPPMCovarianceField.from_prior(
        wishart_model, n_samples=1, key=jr.PRNGKey(456)
    )
    assert field.model is wishart_model
    assert field.n_samples == 1
    # (n_samples=1, degree+1=4, degree+1=4, d=2, m=3), m = input_dim + extra_dims
    assert field.samples["W"].shape == (1, 4, 4, 2, 3)


def test_from_params_matches_the_model(wishart_model):
    """from_params(model, p)(X)[0] is the model's own Σ at every point."""
    params = wishart_model.init_params(jr.PRNGKey(0))
    X = jnp.array([[0.1, 0.2], [0.5, 0.4], [0.9, 0.8]])  # (n_points=3, d=2)
    field = WPPMCovarianceField.from_params(wishart_model, params)
    Sigmas = field(X)  # (1, n_points, d, d): 1 parameter set
    assert Sigmas.shape == (1, 3, 2, 2)
    # the 2 paths differ by up to 4e-9 (float32), Σ entries run to 0.04
    assert jnp.allclose(
        Sigmas[0], _sigma(wishart_model, params, X), rtol=0.0, atol=1e-7
    )


def test_map_fit_holds_theta_map(wishart_model, sample_data):
    """A MAP posterior gives 1 sample, θ_MAP, with no key."""
    posterior = MAPOptimizer(steps=10).fit(wishart_model, sample_data)
    field = WPPMCovarianceField(posterior)
    assert field.n_samples == 1

    x = jnp.array([0.5, 0.5])
    Sigma = field(x)  # (1, d, d)
    U = field.sqrt_cov(x)  # (1, d, m)
    assert Sigma.shape == (1, 2, 2)
    assert U.shape == (1, 2, 3)
    expected = wishart_model.local_covariance(posterior.params, x)  # (d, d)
    assert jnp.allclose(Sigma[0], expected, rtol=0.0, atol=1e-7)
    # the mean over 1 sample is that sample
    assert jnp.allclose(field.mean(x), Sigma[0], rtol=0.0, atol=1e-7)


# ==============================================================================
# Evaluation
# ==============================================================================


def test_cov_wishart_varies(wishart_model):
    """Σ differs between 2 points and is positive definite at both."""
    field = WPPMCovarianceField.from_prior(
        wishart_model, n_samples=1, key=jr.PRNGKey(444)
    )
    Sigma1 = field(jnp.array([0.2, 0.3]))[0]  # (d, d)
    Sigma2 = field(jnp.array([0.8, 0.7]))[0]

    assert not jnp.allclose(Sigma1, Sigma2, atol=1e-6)
    assert jnp.all(jnp.linalg.eigvalsh(Sigma1) > 0)
    assert jnp.all(jnp.linalg.eigvalsh(Sigma2) > 0)


def test_sqrt_cov_wishart(wishart_model):
    """sqrt_cov gives the rectangular U, and U U^T + diag_term * I is the field."""
    field = WPPMCovarianceField.from_prior(
        wishart_model, n_samples=1, key=jr.PRNGKey(555)
    )
    x = jnp.array([0.5, 0.5])
    U = field.sqrt_cov(x)[0]  # (d=2, m=3)
    assert U.shape == (2, 3)

    Sigma_from_U = U @ U.T + wishart_model.diag_term * jnp.eye(2)
    assert jnp.allclose(Sigma_from_U, field(x)[0], rtol=1e-5)


def test_batch_eval_wishart(wishart_model):
    """A batch of points gives 1 matrix per point after the sample axis."""
    field = WPPMCovarianceField.from_prior(
        wishart_model, n_samples=1, key=jr.PRNGKey(666)
    )
    X_grid = jnp.array([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]])  # (n_points=3, d=2)
    Sigmas = field(X_grid)  # (1, n_points, d, d)
    assert Sigmas.shape == (1, 3, 2, 2)

    assert not jnp.allclose(Sigmas[0, 0], Sigmas[0, 1], atol=1e-6)
    assert not jnp.allclose(Sigmas[0, 1], Sigmas[0, 2], atol=1e-6)
    assert jnp.all(jnp.linalg.eigvalsh(Sigmas[0]) > 0)


def test_vmap_over_points_moves_the_sample_axis(wishart_model):
    """jax.vmap(field)(X) puts the points first; field(X) puts the samples first."""
    field = WPPMCovarianceField.from_prior(
        wishart_model, n_samples=1, key=jr.PRNGKey(0)
    )
    X = jnp.array([[0.1, 0.2], [0.5, 0.4]])  # (n_points=2, d=2)
    assert field(X).shape == (1, 2, 2, 2)  # (n_samples, n_points, d, d)
    assert jax.vmap(field)(X).shape == (2, 1, 2, 2)  # (n_points, n_samples, d, d)


def test_consistency_with_model_compute_sqrt(wishart_model):
    """field.sqrt_cov(x)[0] is model._compute_sqrt(params, x)."""
    params = wishart_model.init_params(jr.PRNGKey(1515))
    field = WPPMCovarianceField.from_params(wishart_model, params)
    x = jnp.array([0.5, 0.5])
    assert jnp.allclose(field.sqrt_cov(x)[0], wishart_model._compute_sqrt(params, x))


# ==============================================================================
# Read-only attributes (backlog C24, decision 0012)
# ==============================================================================


def test_field_ignores_changes_to_the_dict_it_was_built_from(wishart_model):
    """Replacing an item of the dict passed to from_params changes no later result."""
    params = wishart_model.init_params(jr.PRNGKey(0))
    field = WPPMCovarianceField.from_params(wishart_model, params)
    X = jnp.array([[0.1, 0.2], [0.5, 0.4]])  # (n_points=2, d=2)
    before = field(X)  # (1, n_points, d, d)
    params["W"] = jnp.zeros_like(params["W"])
    assert jnp.array_equal(field(X), before)
    # compiled and uncompiled paths differ by about 1e-8 here, so atol=1e-6 has 100x headroom
    assert jnp.allclose(field(X[0]), before[:, 0], rtol=0.0, atol=1e-6)
