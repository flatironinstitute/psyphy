"""
tests/test_mcmc_posterior.py
-----------------------------

Unit tests for MCMCPosterior.

the tests are independent of BlackJAX: fake samples are constructed manually
so this suite runs without any sampling backend installed.
Integration with ArViz for sampler diagnostics is also tested.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import pytest

from psyphy.model import WPPM, GaussianNoise, OddityTask, Prior
from psyphy.model.likelihood import OddityTaskConfig
from psyphy.posterior.mcmc_posterior import MCMCPosterior
from psyphy.posterior.parameter_posterior import ParameterPosterior

# ---------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------


def _has_arviz() -> bool:
    try:
        import arviz  # noqa: F401

        return True
    except ImportError:
        return False


# -----------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_model():
    """Minimal WPPM: degree=1, 2D, 1 extra dim -> W shape (2,2,2,3)."""
    return WPPM(
        prior=Prior(input_dim=2, basis_degree=1, extra_embedding_dims=1),
        likelihood=OddityTask(config=OddityTaskConfig(num_samples=5)),
        noise=GaussianNoise(),
    )


@pytest.fixture
def fake_samples():
    """
    manually constructed samples dict
    W shape: (n_chains=2, n_draws=10, 2, 2, 2, 3)
    """
    key = jr.PRNGKey(0)
    W = jr.normal(key, (2, 10, 2, 2, 2, 3))
    return {"W": W}


@pytest.fixture
def posterior(tiny_model, fake_samples):
    stats = {"acceptance_rate": jnp.ones((2, 10)) * 0.8}
    return MCMCPosterior(fake_samples, tiny_model, sampler_stats=stats)


# ---------------------------------------------------------------------------
# Tests


class TestMCMCPosteriorProtocol:
    def test_satisfies_parameter_posterior_protocol(self, posterior):
        assert isinstance(posterior, ParameterPosterior)

    def test_params_is_posterior_mean(self, posterior, fake_samples):
        expected = jnp.mean(fake_samples["W"], axis=(0, 1))
        assert jnp.allclose(posterior.params["W"], expected)

    def test_model_property(self, posterior, tiny_model):
        assert posterior.model is tiny_model

    def test_n_chains(self, posterior):
        assert posterior.n_chains == 2

    def test_n_draws(self, posterior):
        assert posterior.n_draws == 10


class TestMCMCPosteriorSample:
    def test_sample_shape(self, posterior):
        key = jr.PRNGKey(1)
        s = posterior.sample(5, key=key)
        assert "W" in s
        assert s["W"].shape == (5, 2, 2, 2, 3)

    def test_sample_n_equals_one(self, posterior):
        key = jr.PRNGKey(2)
        s = posterior.sample(1, key=key)
        assert s["W"].shape == (1, 2, 2, 2, 3)

    def test_sample_without_replacement(self, posterior):
        """n <= total (2*10=20): should not error, returns n distinct rows."""
        key = jr.PRNGKey(3)
        s = posterior.sample(15, key=key)
        assert s["W"].shape[0] == 15

    def test_sample_with_replacement(self, posterior):
        """n > total (2*10=20): falls back to sampling with replacement."""
        key = jr.PRNGKey(4)
        s = posterior.sample(50, key=key)
        assert s["W"].shape[0] == 50

    def test_sample_draws_from_pool(self, posterior, fake_samples):
        """Every returned sample should match some row in the pooled chain."""
        key = jr.PRNGKey(5)
        s = posterior.sample(3, key=key)
        pool = fake_samples["W"].reshape(20, 2, 2, 2, 3)
        for i in range(3):
            found = any(jnp.allclose(s["W"][i], pool[j]) for j in range(20))
            assert found, f"Sample {i} not found in chain pool"


class TestMCMCPosteriorArviZ:
    def test_to_arviz_raises_without_arviz(self, posterior, monkeypatch):
        import sys

        monkeypatch.setitem(sys.modules, "arviz", None)
        with pytest.raises(ImportError, match="psyphy\\[diagnostics\\]"):
            posterior.to_arviz()

    @pytest.mark.skipif(not _has_arviz(), reason="arviz not installed")
    def test_to_arviz_returns_datatree(self, posterior):
        # ArviZ 1.0 returns xarray.DataTree (not InferenceData as in 0.x)
        from xarray import DataTree

        idata = posterior.to_arviz()
        assert isinstance(idata, DataTree)

    @pytest.mark.skipif(not _has_arviz(), reason="arviz not installed")
    def test_to_arviz_posterior_group(self, posterior):
        idata = posterior.to_arviz()
        # ArviZ 1.0: groups are DataTree children
        assert "posterior" in list(idata.children)

    @pytest.mark.skipif(not _has_arviz(), reason="arviz not installed")
    def test_to_arviz_W_variable_shape(self, posterior):
        idata = posterior.to_arviz()
        W = idata["posterior"]["W"].values
        # (n_chains, n_draws, 2, 2, 2, 3)
        assert W.shape == (2, 10, 2, 2, 2, 3)

    @pytest.mark.skipif(not _has_arviz(), reason="arviz not installed")
    def test_to_arviz_sample_stats(self, posterior):
        idata = posterior.to_arviz()
        assert "sample_stats" in list(idata.children)
        assert "acceptance_rate" in idata["sample_stats"]

    @pytest.mark.skipif(not _has_arviz(), reason="arviz not installed")
    def test_to_arviz_rhat_computable(self, posterior):
        """R-hat should be computable without errors (values may be noisy with 10 draws)."""
        import arviz as az
        from xarray import DataTree

        idata = posterior.to_arviz()
        rhat = az.rhat(idata)
        assert isinstance(rhat, DataTree)
        assert "W" in rhat.ds.data_vars
