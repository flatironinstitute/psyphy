"""
tests/test_map_one_sample.py
----------------------------

Tests for 1 parameter set in WPPMCovarianceField: a MAP fit, a ground truth or
a published W, held as a field with a sample axis of length 1.

- MAPPosterior stores its 1 sample the way an MCMCPosterior stores its chains,
  so WPPMCovarianceField(map_post) doesn't need n_samples or a key
- WPPMCovarianceField.from_params(model, params) takes 1 parameter set with no
  leading axis
- credible_band raises for 1 sample, since a band of zero width reads as certainty

Every check compares 2 routes to the same numbers, with W built from
arithmetic, so no value here depends on jax.random.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from psyphy.model import WPPM, GaussianNoise, OddityTask, Prior
from psyphy.model.covariance_field import WPPMCovarianceField
from psyphy.model.likelihood import OddityTaskConfig
from psyphy.posterior.mcmc_posterior import MCMCPosterior
from psyphy.posterior.posterior import MAPPosterior

W_SHAPE = (2, 2, 2, 3)  # degree 0 to 1 in x1, the same in x2, d=2, m=3


@pytest.fixture(autouse=True)
def _full_matmul_precision():
    """Full float32 matmuls, so 2 routes through the same arithmetic agree exactly."""
    with jax.default_matmul_precision("float32"):
        yield


@pytest.fixture
def model():
    """Minimal WPPM: degree=1, 2D, 1 extra dim -> W shape (2, 2, 2, 3)."""
    return WPPM(
        prior=Prior(input_dim=2, basis_degree=1, extra_embedding_dims=1),
        likelihood=OddityTask(config=OddityTaskConfig(num_samples=5)),
        noise=GaussianNoise(),
    )


def _W(offset=0.0):
    """A W of shape (2, 2, 2, 3) from arithmetic, entries between -0.5 and 0.5."""
    n = 2 * 2 * 2 * 3
    return jnp.sin(jnp.arange(n, dtype=jnp.float32) + offset).reshape(W_SHAPE) / 2


@pytest.fixture
def params():
    return {"W": _W()}  # {"W": (2, 2, 2, 3)}


@pytest.fixture
def map_post(model, params):
    return MAPPosterior(model, params)


@pytest.fixture
def X():
    return jnp.array([[-0.5, -0.5], [0.0, 0.3], [0.5, 0.5]])  # (n_points=3, d=2)


def _sigma_one_set(model, params, X):
    """Σ at every point from model.local_covariance alone: (n_points, d, d)."""
    return jax.vmap(lambda x: model.local_covariance(params, x))(X)


# ==============================================================================
# MAPPosterior stores its 1 sample
# ==============================================================================


def test_map_posterior_stores_one_sample(map_post, params):
    """samples has a chain axis and a sample axis of length 1, holding θ_MAP."""
    assert map_post.n_chains == 1
    assert map_post.n_draws == 1
    assert map_post.samples["W"].shape == (1, 1, *W_SHAPE)
    assert jnp.array_equal(map_post.samples["W"][0, 0], params["W"])


def test_map_posterior_samples_is_a_fresh_dict(map_post):
    """Editing the returned dict leaves the posterior alone, as for MCMCPosterior."""
    map_post.samples["W"] = jnp.zeros((1, 1, *W_SHAPE))
    assert map_post.samples["W"].any()


def test_map_posterior_samples_match_mcmc_layout(model, params, map_post):
    """A 1-chain, 1-sample MCMCPosterior holding θ_MAP stores the same arrays."""
    mcmc = MCMCPosterior(model, {"W": params["W"][None, None]})
    assert map_post.samples["W"].shape == mcmc.samples["W"].shape
    assert jnp.array_equal(map_post.samples["W"], mcmc.samples["W"])


# ==============================================================================
# WPPMCovarianceField(map_post) doesn't need a key
# ==============================================================================


def test_samples_from_map_posterior_needs_no_key(map_post):
    fields = WPPMCovarianceField(map_post)
    assert fields.n_samples == 1
    assert fields.samples["W"].shape == (1, *W_SHAPE)


def test_samples_from_map_posterior_mean_is_sigma_of_theta_map(model, map_post, X):
    """The mean of 1 sample is Σ(θ_MAP), as model.local_covariance gives it."""
    fields = WPPMCovarianceField(map_post)
    expected = _sigma_one_set(model, map_post.params, X)  # (n_points, d, d)
    assert fields(X).shape == (1, *expected.shape)
    assert jnp.allclose(fields.mean(X), expected, rtol=0, atol=1e-7)
    assert jnp.allclose(fields(X)[0], expected, rtol=0, atol=1e-7)


def test_posterior_storing_nothing_still_needs_n_samples(model, params):
    """Only a posterior with samples, n_chains and n_draws skips the key."""

    class SampleOnly:
        def __init__(self, model, params):
            self.model = model
            self.params = params

        def sample(self, n, *, key):
            return jax.tree.map(lambda x: jnp.broadcast_to(x, (n, *x.shape)), params)

    with pytest.raises(ValueError, match="n_samples"):
        WPPMCovarianceField(SampleOnly(model, params))


# ==============================================================================
# from_params
# ==============================================================================


def test_from_params_matches_local_covariance(model, params, X):
    """fields(X)[0] is Σ from model.local_covariance at every point."""
    fields = WPPMCovarianceField.from_params(model, params)
    expected = _sigma_one_set(model, params, X)  # (n_points, d, d)
    assert fields.n_samples == 1
    assert fields(X).shape == (1, 3, 2, 2)  # (n_samples, n_points, d, d)
    assert jnp.allclose(fields(X)[0], expected, rtol=0, atol=1e-7)


def test_from_params_holds_the_set_with_a_leading_axis(model, params):
    fields = WPPMCovarianceField.from_params(model, params)
    assert fields.samples["W"].shape == (1, *W_SHAPE)
    assert jnp.array_equal(fields.samples["W"][0], params["W"])


def test_from_params_matches_map_posterior(model, params, X):
    """Both routes to 1 parameter set give the same field."""
    a = WPPMCovarianceField.from_params(model, params)(X)
    b = WPPMCovarianceField(MAPPosterior(model, params))(X)
    assert jnp.array_equal(a, b)


def test_from_params_swapped_raises(model, params):
    with pytest.raises(TypeError, match=r"from_params\(model, params\)"):
        WPPMCovarianceField.from_params(params, model)


def test_from_params_model_is_kept(model, params):
    assert WPPMCovarianceField.from_params(model, params).model is model


# ==============================================================================
# credible_band with 1 sample
# ==============================================================================


def test_credible_band_raises_for_one_sample(model, params, X):
    fields = WPPMCovarianceField.from_params(model, params)
    with pytest.raises(ValueError, match="MCMC"):
        fields.credible_band(X)


def test_credible_band_raises_for_map_posterior(map_post, X):
    with pytest.raises(ValueError, match="1 parameter set"):
        WPPMCovarianceField(map_post).credible_band(X)


def test_credible_band_runs_for_two_samples(model, X):
    """2 distinct sets have a spread, so the band is defined."""
    W = jnp.stack([_W(), _W(offset=1.0)])  # (2, 2, 2, 2, 3)
    fields = WPPMCovarianceField(MCMCPosterior(model, {"W": W[None]}))
    band = fields.credible_band(X, n_directions=19)
    assert band.lower.shape == (3, 19)  # (n_points, n_directions)
