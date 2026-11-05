"""
Tests that WPPMPredictivePosterior takes its randomness from one explicit key.

What these tests pin:

1. The constructor requires a key, and `mean`, `variance` and the threshold sweep
   all read it. The same key gives the same answer; different keys do not.
2. `cov_field` is gone from the class and from the PredictivePosterior protocol,
   so there is only one path to E[Sigma(x)] (as opposed to 2 paths in
   the previous version).
3. The threshold sweep's result depends on the key by several percent even at the
   settings the Hong et al. reproduction uses. A figure built from it is only
   reproducible when the key travels with it.
"""

import jax.numpy as jnp
import jax.random as jr
import pytest

from psyphy.model import (
    WPPM,
    GaussianNoise,
    OddityTask,
    OddityTaskConfig,
    Prior,
)
from psyphy.posterior import (
    MAPPosterior,
    MCMCPosterior,
    PredictivePosterior,
    ThresholdConfig,
    WPPMPredictivePosterior,
)


@pytest.fixture
def model():
    return WPPM(
        prior=Prior(input_dim=2, basis_degree=2),
        likelihood=OddityTask(OddityTaskConfig(num_samples=100)),
        noise=GaussianNoise(),
    )


@pytest.fixture
def posterior(model):
    """Prior draws reshaped as 4 chains of 10 draws, standing in for a fit."""
    keys = jr.split(jr.PRNGKey(1), 40)
    W = jnp.stack([model.init_params(k)["W"] for k in keys])  # (40, 3, 3, 2, 3)
    return MCMCPosterior({"W": W.reshape(4, 10, *W.shape[1:])}, model)


@pytest.fixture
def X_paired():
    refs = jnp.array([[0.0, 0.0], [0.3, 0.3]])
    comps = jnp.array([[0.05, 0.0], [0.35, 0.3]])
    return jnp.stack([refs, comps], axis=1)  # (n_test=2, K=2, d=2)


# ==============================================================================
# The key
# ==============================================================================


class TestConstructorRequiresAKey:
    def test_key_is_required(self, posterior, X_paired):
        with pytest.raises(TypeError):
            WPPMPredictivePosterior(posterior, X_paired, n_samples=8)

    def test_key_is_keyword_only(self, posterior, X_paired):
        """Positional would make it ambiguous with threshold_config."""
        with pytest.raises(TypeError):
            WPPMPredictivePosterior(posterior, X_paired, 8, False, None, jr.PRNGKey(0))


class TestTheKeyIsActuallyRead:
    """mean and variance are properties, so the constructor key is their only
    source of randomness."""

    def test_same_key_gives_the_same_mean(self, posterior, X_paired):
        a = WPPMPredictivePosterior(
            posterior, X_paired, n_samples=8, key=jr.PRNGKey(0)
        ).mean
        b = WPPMPredictivePosterior(
            posterior, X_paired, n_samples=8, key=jr.PRNGKey(0)
        ).mean

        assert jnp.array_equal(a, b)

    def test_different_keys_give_a_different_mean(self, posterior, X_paired):
        a = WPPMPredictivePosterior(
            posterior, X_paired, n_samples=8, key=jr.PRNGKey(0)
        ).mean
        b = WPPMPredictivePosterior(
            posterior, X_paired, n_samples=8, key=jr.PRNGKey(12345)
        ).mean

        # the measured spread of the mean across keys at n_samples=40 was 1.6e-2,
        # so 1e-4 is well inside what a real key change moves
        assert not jnp.allclose(a, b, atol=1e-4)

    def test_variance_follows_the_same_key(self, posterior, X_paired):
        a = WPPMPredictivePosterior(
            posterior, X_paired, n_samples=8, key=jr.PRNGKey(0)
        ).variance
        b = WPPMPredictivePosterior(
            posterior, X_paired, n_samples=8, key=jr.PRNGKey(0)
        ).variance

        assert jnp.array_equal(a, b)

    def test_mean_and_variance_share_one_draw(self, posterior, X_paired):
        """Both read the same cached draw, so asking for one then the other does
        not resample."""
        p = WPPMPredictivePosterior(posterior, X_paired, n_samples=8, key=jr.PRNGKey(0))

        first_mean = p.mean
        _ = p.variance

        assert jnp.array_equal(first_mean, p.mean)


class TestThresholdSweepUsesTheKey:
    def test_same_key_reproduces_the_sweep(self, model):
        post = MAPPosterior(model.init_params(jr.PRNGKey(0)), model)
        coords = jnp.array([[0.0, 0.0]])
        cfg = ThresholdConfig(n_theta=8, n_length=20, chunk=500)

        a = WPPMPredictivePosterior(
            post,
            coords,
            n_samples=1,
            threshold_pred=True,
            threshold_config=cfg,
            key=jr.PRNGKey(0),
        ).mean
        b = WPPMPredictivePosterior(
            post,
            coords,
            n_samples=1,
            threshold_pred=True,
            threshold_config=cfg,
            key=jr.PRNGKey(0),
        ).mean

        assert jnp.array_equal(a, b)

    def test_the_sweep_key_moves_the_recovered_threshold(self, model):
        """Measured at the reproduction's own settings (n_length=1000, mc=2000),
        4 keys spread the recovered trace by 4.3%. A figure built from the sweep
        is only reproducible when its key is recorded with it."""
        post = MAPPosterior(model.init_params(jr.PRNGKey(0)), model)
        coords = jnp.array([[0.0, 0.0]])
        cfg = ThresholdConfig(n_theta=8, n_length=20, chunk=500)

        traces = [
            float(
                jnp.trace(
                    WPPMPredictivePosterior(
                        post,
                        coords,
                        n_samples=1,
                        threshold_pred=True,
                        threshold_config=cfg,
                        key=jr.PRNGKey(seed),
                    ).mean[0]
                )
            )
            for seed in (0, 1, 2)
        ]

        assert max(traces) != min(traces)


# ==============================================================================
# cov_field is gone
# ==============================================================================


class TestCovFieldRemoved:
    """One path to E[Sigma(x)]: WPPMCovarianceFieldSamples.mean."""

    def test_the_method_is_gone(self, posterior, X_paired):
        p = WPPMPredictivePosterior(posterior, X_paired, n_samples=8, key=jr.PRNGKey(0))

        assert not hasattr(p, "cov_field")

    def test_the_protocol_no_longer_requires_it(self):
        """A model with no covariance field can satisfy PredictivePosterior."""
        assert "cov_field" not in PredictivePosterior.__dict__

    def test_the_class_still_satisfies_the_protocol(self, posterior, X_paired):
        p = WPPMPredictivePosterior(posterior, X_paired, n_samples=8, key=jr.PRNGKey(0))

        assert isinstance(p, PredictivePosterior)
