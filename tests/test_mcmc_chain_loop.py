"""
tests/test_mcmc_chain_loop.py
-----------------------------

Tests for ``_fit_chains``, the part of a fit that every MCMC engine shares.

A stub replaces the BlackJAX sampler, so these tests don't need blackjax and run
no chains. The stub records what the shared loop hands an engine and returns the
starting point as every draw. That pins the prior preconditioning on both sides:
the engine sees z = W / prior_std, and the posterior holds W again.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from psyphy.inference.mcmc import _fit_chains
from psyphy.model import WPPM, GaussianNoise, OddityTask, Prior
from psyphy.posterior.mcmc_posterior import MCMCPosterior

NUM_CHAINS = 3
NUM_SAMPLES = 4


@pytest.fixture
def model():
    """Degree-1 2D WPPM: W shape (2, 2, 2, 3)."""
    return WPPM(
        prior=Prior(input_dim=2, basis_degree=1, extra_embedding_dims=1),
        likelihood=OddityTask(),
        noise=GaussianNoise(),
    )


@pytest.fixture
def init_params(model):
    """A W with every entry 0.5, so W / prior_std is easy to predict."""
    shape = model.prior.param_scales()["W"].shape
    return {"W": jnp.full(shape, 0.5)}


def _stub(seen):
    """Stand-in for an engine: record the inputs, return the start as every draw."""

    def run_chains(logdensity_fn, init_position, chain_keys):
        seen["init_position"] = init_position
        seen["chain_keys"] = chain_keys
        seen["logdensity"] = logdensity_fn(init_position)
        positions = jax.tree.map(
            lambda p: jnp.broadcast_to(p, (NUM_CHAINS, NUM_SAMPLES, *p.shape)),
            init_position,
        )  # {"W": (NUM_CHAINS, NUM_SAMPLES, 2, 2, 2, 3)}
        stats = {"acceptance_rate": jnp.ones((NUM_CHAINS, NUM_SAMPLES))}
        return positions, stats

    return run_chains


def _fit(model, init_params, *, precondition, seen):
    return _fit_chains(
        model,
        data=None,
        init_params=init_params,
        seed=0,
        num_chains=NUM_CHAINS,
        logdensity_key_seed=0,
        precondition=precondition,
        run_chains=_stub(seen),
    )


@pytest.fixture(autouse=True)
def _no_likelihood(monkeypatch):
    """Skip the likelihood: the stub evaluates the log density once, on no data."""
    monkeypatch.setattr(
        WPPM,
        "log_posterior_from_data",
        lambda self, params, data, key: -jnp.sum(params["W"] ** 2),
    )


def test_engine_samples_z_and_posterior_holds_w(model, init_params):
    """With preconditioning the engine starts at W / prior_std, draws come back as W."""
    seen = {}
    post = _fit(model, init_params, precondition=True, seen=seen)
    scales = model.prior.param_scales()["W"]

    np.testing.assert_allclose(seen["init_position"]["W"], 0.5 / scales, rtol=1e-6)
    assert isinstance(post, MCMCPosterior)
    assert post.samples["W"].shape == (NUM_CHAINS, NUM_SAMPLES, *scales.shape)
    np.testing.assert_allclose(post.samples["W"][1, 2], init_params["W"], rtol=1e-6)


def test_preconditioned_log_density_is_the_w_log_density(model, init_params):
    """The engine's log density at z equals the model's at W = prior_std * z."""
    seen = {}
    _fit(model, init_params, precondition=True, seen=seen)
    expected = -jnp.sum(init_params["W"] ** 2)
    np.testing.assert_allclose(seen["logdensity"], expected, rtol=1e-6)


def test_without_preconditioning_engine_samples_w(model, init_params):
    seen = {}
    post = _fit(model, init_params, precondition=False, seen=seen)
    np.testing.assert_array_equal(seen["init_position"]["W"], init_params["W"])
    np.testing.assert_array_equal(post.samples["W"][0, 0], init_params["W"])


def test_one_key_per_chain(model, init_params):
    seen = {}
    _fit(model, init_params, precondition=False, seen=seen)
    assert seen["chain_keys"].shape == (NUM_CHAINS, 2)


def test_sampler_stats_pass_through(model, init_params):
    """The shared loop keeps whatever stats the engine returns, keys included."""
    post = _fit(model, init_params, precondition=True, seen={})
    assert set(post.sampler_stats) == {"acceptance_rate"}
