"""The model comes first in the posterior and covariance field constructors."""

import jax.numpy as jnp
import pytest

from psyphy.model import WPPM, OddityTask, Prior, WPPMCovarianceField
from psyphy.posterior import MAPPosterior, MCMCPosterior

W_SHAPE = (3, 3, 2, 2)  # (degree+1, degree+1, input_dim, embedding_dim)


@pytest.fixture
def model():
    return WPPM(Prior(input_dim=2, basis_degree=2), OddityTask())


@pytest.fixture
def params():
    # arithmetic fixture, not jax.random
    return {"W": 0.01 * jnp.arange(36.0).reshape(W_SHAPE)}


@pytest.fixture
def samples(params):
    # (n_chains=2, n_draws=3, *W_SHAPE)
    offsets = jnp.arange(6.0).reshape(2, 3, 1, 1, 1, 1)
    return {"W": params["W"][None, None] + 0.001 * offsets}


def test_map_posterior_model_first(model, params):
    new = MAPPosterior(model, params)
    kw = MAPPosterior(params=params, model=model)
    assert new.model is model
    assert new.params is params
    assert jnp.array_equal(new.params["W"], kw.params["W"])


def test_mcmc_posterior_model_first(model, samples):
    new = MCMCPosterior(model, samples)
    kw = MCMCPosterior(samples=samples, model=model)
    assert new.model is model
    assert jnp.array_equal(new.params["W"], kw.params["W"])
    assert jnp.array_equal(new.params["W"], samples["W"].mean(axis=(0, 1)))


def test_mcmc_posterior_sampler_stats_stays_keyword_only(model, samples):
    stats = {"diverging": jnp.zeros((2, 3))}
    assert MCMCPosterior(model, samples, sampler_stats=stats).sampler_stats is stats
    with pytest.raises(TypeError):
        MCMCPosterior(model, samples, stats)


def test_field_model_first_unchanged(model, params):
    field = WPPMCovarianceField(model, params)
    assert field.model is model
    assert field(jnp.array([0.5, 0.3])).shape == (2, 2)


def test_old_order_map_posterior_raises(model, params):
    with pytest.raises(TypeError, match=r"MAPPosterior\(model, params\)"):
        MAPPosterior(params, model)


def test_old_order_mcmc_posterior_raises(model, samples):
    with pytest.raises(TypeError, match=r"MCMCPosterior\(model, samples\)"):
        MCMCPosterior(samples, model)


def test_old_order_field_raises(model, params):
    with pytest.raises(TypeError, match=r"WPPMCovarianceField\(model, params\)"):
        WPPMCovarianceField(params, model)


def test_non_mapping_params_raises(model):
    with pytest.raises(TypeError, match=r"MAPPosterior\(model, params\)"):
        MAPPosterior(model, jnp.zeros(3))
    with pytest.raises(TypeError, match=r"MCMCPosterior\(model, samples\)"):
        MCMCPosterior(model, jnp.zeros(3))
    with pytest.raises(TypeError, match=r"WPPMCovarianceField\(model, params\)"):
        WPPMCovarianceField(model, jnp.zeros(3))
