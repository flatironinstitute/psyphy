"""
tests/test_nuts_sampler.py
---------------------------

Integration tests for NUTSSampler.

Requires blackjax to be installed::

    pip install 'psyphy[sampling]'

Tests are skipped gracefully when blackjax is not available.
Slow tests (those that actually run NUTS) are marked with @pytest.mark.slow
and use minimal settings (num_warmup=20, num_samples=10) for speed.
"""

from __future__ import annotations

import sys

import jax.numpy as jnp
import jax.random as jr
import pytest

blackjax = pytest.importorskip("blackjax", reason="blackjax not installed")

from psyphy.data.dataset import TrialData
from psyphy.inference.nuts import NUTSSampler
from psyphy.model import WPPM, GaussianNoise, OddityTask, Prior
from psyphy.model.likelihood import OddityTaskConfig
from psyphy.posterior.mcmc_posterior import MCMCPosterior
from psyphy.posterior.parameter_posterior import ParameterPosterior
from psyphy.posterior.predictive_posterior import WPPMPredictivePosterior

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_model():
    """Minimal WPPM: degree=1, 2D, 1 extra dim -> W shape (2,2,2,3)."""
    return WPPM(
        input_dim=2,
        extra_dims=1,
        prior=Prior(input_dim=2, basis_degree=1, extra_embedding_dims=1),
        likelihood=OddityTask(config=OddityTaskConfig(num_samples=5)),
        noise=GaussianNoise(),
    )


@pytest.fixture(scope="module")
def tiny_data():
    """5 trials with refs and comparisons inside [-0.5, 0.5]."""
    key = jr.PRNGKey(0)
    refs = jr.uniform(key, (5, 2), minval=-0.5, maxval=0.5)
    comparisons = jnp.clip(refs + jr.normal(jr.PRNGKey(1), (5, 2)) * 0.05, -1.0, 1.0)
    responses = jnp.ones(5, dtype=jnp.int32)
    return TrialData(
        stimuli=jnp.stack([refs, comparisons], axis=1), responses=responses
    )


@pytest.fixture(scope="module")
def sampler_adaptive():
    """Minimal adaptive NUTSSampler (Mode A)."""
    return NUTSSampler(
        num_warmup=20,
        num_samples=10,
        num_chains=2,
        show_progress=False,
    )


@pytest.fixture(scope="module")
def sampler_fixed():
    """Minimal fixed-stepsize NUTSSampler (Mode B)."""
    return NUTSSampler(
        num_warmup=20,
        num_samples=10,
        num_chains=2,
        step_size=0.01,
        show_progress=False,
    )


@pytest.fixture(scope="module")
def posterior_adaptive(sampler_adaptive, tiny_model, tiny_data):
    """Run adaptive NUTS once for the whole module."""
    return sampler_adaptive.fit(tiny_model, tiny_data)


@pytest.fixture(scope="module")
def posterior_fixed(sampler_fixed, tiny_model, tiny_data):
    """Run fixed-stepsize NUTS once for the whole module."""
    return sampler_fixed.fit(tiny_model, tiny_data)


# Tests: import guard
# ---------------------------------------------------------------------------


class TestImportGuard:
    def test_import_error_without_blackjax(self, tiny_model, tiny_data, monkeypatch):
        monkeypatch.setitem(sys.modules, "blackjax", None)
        sampler = NUTSSampler(
            num_warmup=5, num_samples=5, num_chains=1, show_progress=False
        )
        with pytest.raises(ImportError, match="psyphy\\[sampling\\]"):
            sampler.fit(tiny_model, tiny_data)


# -------------------------------------------------------------------
# Tests: Mode A (adaptive)
# ---------------------------------------------------------------------------


class TestNUTSSamplerAdaptive:
    def test_fit_returns_mcmc_posterior(self, posterior_adaptive):
        assert isinstance(posterior_adaptive, MCMCPosterior)

    def test_satisfies_parameter_posterior_protocol(self, posterior_adaptive):
        assert isinstance(posterior_adaptive, ParameterPosterior)

    def test_n_chains(self, posterior_adaptive):
        assert posterior_adaptive.n_chains == 2

    def test_n_draws(self, posterior_adaptive):
        assert posterior_adaptive.n_draws == 10

    def test_W_shape(self, posterior_adaptive):
        W = posterior_adaptive._samples["W"]
        # (n_chains, n_draws, degree+1, degree+1, input_dim, emb_dim)
        assert W.shape == (2, 10, 2, 2, 2, 3)

    def test_params_shape(self, posterior_adaptive):
        params = posterior_adaptive.params
        assert "W" in params
        assert params["W"].shape == (2, 2, 2, 3)

    def test_sample_method_shape(self, posterior_adaptive):
        key = jr.PRNGKey(42)
        s = posterior_adaptive.sample(5, key=key)
        assert s["W"].shape == (5, 2, 2, 2, 3)

    def test_sampler_stats_present(self, posterior_adaptive):
        assert "acceptance_rate" in posterior_adaptive.sampler_stats
        acc = posterior_adaptive.sampler_stats["acceptance_rate"]
        assert acc.shape == (2, 10)

    def test_model_property(self, posterior_adaptive, tiny_model):
        assert posterior_adaptive.model is tiny_model

    def test_acceptance_rates_in_range(self, posterior_adaptive):
        acc = posterior_adaptive.sampler_stats["acceptance_rate"]
        assert jnp.all(acc >= 0.0)
        assert jnp.all(acc <= 1.0)


# --------------------------------------------------------------
# Tests: Mode B (fixed step_size, vmapped)
# ---------------------------------------------------------------------------


class TestNUTSSamplerFixed:
    def test_fit_returns_mcmc_posterior(self, posterior_fixed):
        assert isinstance(posterior_fixed, MCMCPosterior)

    def test_satisfies_parameter_posterior_protocol(self, posterior_fixed):
        assert isinstance(posterior_fixed, ParameterPosterior)

    def test_n_chains(self, posterior_fixed):
        assert posterior_fixed.n_chains == 2

    def test_n_draws(self, posterior_fixed):
        assert posterior_fixed.n_draws == 10

    def test_W_shape(self, posterior_fixed):
        W = posterior_fixed._samples["W"]
        assert W.shape == (2, 10, 2, 2, 2, 3)

    def test_params_shape(self, posterior_fixed):
        params = posterior_fixed.params
        assert params["W"].shape == (2, 2, 2, 3)


# ---------------------------------------------------------------------------
# Tests: downstream integration
# ---------------------------------------------------------------------------


class TestNUTSDownstreamIntegration:
    """
    Verify that MCMCPosterior plugs into WPPMPredictivePosterior exactly as
    MAPPosterior does — the predictive posterior is agnostic about which
    inference engine produced the parameter posterior it wraps.

    context:
    -----------
    WPPMPredictivePosterior answers: "Given everything we've observed so far
    (captured in the posterior over W), how likely is an observer to correctly
    identify the comparison stimulus at a candidate test location?"

    It approximates the posterior predictive:
        p(correct | x_ref, x_comp, D)
            ≈ (1/N) Σ_i  p(correct | x_ref, x_comp, θ_i),   θ_i ~ posterior
    where p(correct | ..., θ_i) is evaluated by OddityTask.predict() and
    averaged over N parameter samples drawn from MCMCPosterior.sample().
    """

    def test_predictive_posterior_mean_shape(self, posterior_adaptive, tiny_model):
        # refs: reference stimulus locations (x_ref) — the "anchor" stimulus
        # shown to the observer in each trial (matches TrialData.refs).
        # comparisons: the paired probe stimuli (x_comp) placed nearby.
        # The model predicts p(correct | x_ref, x_comp, θ) for each pair.
        refs = jnp.array([[0.0, 0.0], [0.2, 0.2]])  # reference stimuli, ie X_test
        comparisons = jnp.array(
            [[0.1, 0.0], [0.3, 0.2]]
        )  # comparison stimuli, one per reference
        pred = WPPMPredictivePosterior(
            posterior_adaptive, jnp.stack([refs, comparisons], axis=1), n_samples=5
        )
        # pred.mean[i] ≈ E[p(correct | refs[i], comparisons[i], θ) | D]
        mean = pred.mean
        assert mean.shape == (2,)
        assert jnp.all((mean > 0) & (mean < 1))

    def test_predictive_posterior_variance_shape(self, posterior_adaptive, tiny_model):
        # Variance over the posterior samples quantifies uncertainty about
        # p(correct) at each test location — high variance means the model
        # parameters disagree about discriminability there.
        refs = jnp.array([[0.0, 0.0], [0.2, 0.2]])  # reference stimuli
        comparisons = jnp.array(
            [[0.1, 0.0], [0.3, 0.2]]
        )  # comparison stimuli, one per reference
        pred = WPPMPredictivePosterior(
            posterior_adaptive, jnp.stack([refs, comparisons], axis=1), n_samples=5
        )
        var = pred.variance
        assert var.shape == (2,)
        assert jnp.all(var >= 0)

    def test_map_and_nuts_interchangeable(self, posterior_adaptive, tiny_model):
        # WPPMPredictivePosterior only depends on the ParameterPosterior
        # protocol (.sample(), .params, .model) — it never inspects whether
        # the posterior came from MAP or MCMC. This test confirms both work
        # identically at the call site, which is the key design invariant.
        from psyphy.posterior.posterior import MAPPosterior

        map_params = tiny_model.init_params(jr.PRNGKey(0))
        map_posterior = MAPPosterior(params=map_params, model=tiny_model)

        refs = jnp.array([[0.0, 0.0]])  # reference stimulus
        comparisons = jnp.array(
            [[0.1, 0.0]]
        )  # comparison stimulus paired with the reference

        for post in [map_posterior, posterior_adaptive]:
            pred = WPPMPredictivePosterior(
                post, jnp.stack([refs, comparisons], axis=1), n_samples=3
            )
            assert pred.mean.shape == (1,)


# ---------------------------------------------------------------------------
# Tests: warmup-cost controls (tree-depth cap, prior preconditioning)
# ---------------------------------------------------------------------------


class TestWarmupCostControls:
    def test_param_scales_match_param_shapes(self, tiny_model):
        # NUTSSampler multiplies z by these elementwise, so they must line up
        # leaf-for-leaf with the parameter PyTree.
        scales = tiny_model.prior.param_scales()
        params = tiny_model.init_params(jr.PRNGKey(0))
        assert scales.keys() == params.keys()
        assert scales["W"].shape == params["W"].shape
        assert jnp.all(scales["W"] > 0)

    def test_max_num_doublings_caps_tree_size(self, tiny_model, tiny_data):
        # With a cap of 2 doublings no draw may use more than 2**2 - 1 = 3
        # leapfrog steps, in either mode.
        for step_size in (None, 0.01):
            post = NUTSSampler(
                num_warmup=10,
                num_samples=5,
                num_chains=2,
                step_size=step_size,
                max_num_doublings=2,
                show_progress=False,
            ).fit(tiny_model, tiny_data)
            assert jnp.all(post.sampler_stats["n_steps"] <= 3)

    def test_invalid_max_num_doublings_raises(self):
        with pytest.raises(ValueError, match="max_num_doublings"):
            NUTSSampler(max_num_doublings=0)

    def test_preconditioning_off_still_samples(self, tiny_model, tiny_data):
        post = NUTSSampler(
            num_warmup=10,
            num_samples=5,
            num_chains=2,
            prior_preconditioning=False,
            show_progress=False,
        ).fit(tiny_model, tiny_data)
        assert post._samples["W"].shape == (2, 5, 2, 2, 2, 3)

    def test_preconditioned_draws_are_in_W_space(self, tiny_model, tiny_data):
        # Starting exactly at init_params, a preconditioned run must report
        # draws on the W scale (~prior std), not the O(1) z scale it sampled
        # in. With variance_scale=4e-3, W draws of magnitude ~1 would mean
        # the z -> W map-back was skipped.
        init = tiny_model.init_params(jr.PRNGKey(5))
        post = NUTSSampler(
            num_warmup=10, num_samples=5, num_chains=2, show_progress=False
        ).fit(tiny_model, tiny_data, init_params=init)
        assert float(jnp.max(jnp.abs(post._samples["W"]))) < 0.5
