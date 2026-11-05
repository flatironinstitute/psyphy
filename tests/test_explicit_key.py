"""
Tests that a parameter draw names its key, and that the all-draws path is keyless.

What these pin:

1. `MCMCPosterior.sample` and `MAPPosterior.sample` require both `n` and `key`.
   Omitting either is a TypeError, so no result carries a seed the call site does
   not show.
2. `WPPMCovarianceFieldSamples.from_posterior(posterior)` returns the same numbers
   its reductions returned when they routed through `posterior.sample`. Pooling the
   stored draws in order replaces a permutation of the same draws.
3. That entry point needs no key for every stored draw, and still accepts one when
   `n_samples` subsamples.
"""

import jax.numpy as jnp
import jax.random as jr
import pytest

from psyphy.model import WPPM, OddityTask, Prior, WPPMCovarianceFieldSamples
from psyphy.posterior import MAPPosterior, MCMCPosterior


@pytest.fixture
def model():
    return WPPM(Prior(input_dim=2, basis_degree=2), OddityTask())


@pytest.fixture
def posterior(model):
    """Prior draws reshaped as 4 chains of 50 draws, standing in for a fit.

    200 stored draws rather than 100: at exactly 100 the subsample path and the
    pooled path agree by coincidence, because jr.choice over the whole pool is a
    permutation under the same default key. 200 makes the two distinguishable.
    """
    keys = jr.split(jr.PRNGKey(1), 200)
    W = jnp.stack([model.init_params(k)["W"] for k in keys])  # (200, 3, 3, 2, 3)
    return MCMCPosterior({"W": W.reshape(4, 50, *W.shape[1:])}, model)


@pytest.fixture
def ref_points():
    return jnp.array([[-0.5, -0.5], [0.0, 0.0], [0.5, 0.5]])  # (n_points=3, d=2)


# ==============================================================================
# The signatures
# ==============================================================================


class TestSampleRequiresBoth:
    """Both arguments are required on every source, matching ParameterSource."""

    def test_mcmc_sample_needs_a_key(self, posterior):
        with pytest.raises(TypeError):
            posterior.sample(5)

    def test_mcmc_sample_with_a_key_works(self, posterior):
        assert posterior.sample(5, key=jr.PRNGKey(0))["W"].shape[0] == 5

    def test_map_sample_needs_a_key(self, model):
        map_posterior = MAPPosterior(
            params=model.init_params(jr.PRNGKey(0)), model=model
        )

        with pytest.raises(TypeError):
            map_posterior.sample(3)

    def test_map_sample_needs_n(self, model):
        """n: int = 1 is gone, so a MAP fit has the same signature as the others."""
        map_posterior = MAPPosterior(
            params=model.init_params(jr.PRNGKey(0)), model=model
        )

        with pytest.raises(TypeError):
            map_posterior.sample(key=jr.PRNGKey(0))

    def test_prior_sample_needs_a_key(self):
        with pytest.raises(TypeError):
            Prior(input_dim=2, basis_degree=2).sample(5)


# ==============================================================================
# The all-draws path: same numbers, no key
# ==============================================================================


class TestAllDrawsPathIsUnchanged:
    """Pooling in stored order gives what the permutation gave."""

    def test_mean_field_matches_the_subsample_path(self, model, posterior, ref_points):
        """The pooled mean equals the mean over a full resample of the same pool.

        A full resample is a permutation, and a mean does not depend on order, so
        these agree. Checked with a tolerance rather than exactly: float32
        summation over 200 matrices depends on the order the terms arrive in, and
        the residue is about 4e-9 against entries of order 1e-2.
        """
        pooled = WPPMCovarianceFieldSamples.from_posterior(posterior).mean(ref_points)

        resampled = posterior.sample(200, key=jr.PRNGKey(0))["W"]
        by_hand = jnp.mean(
            jnp.stack(
                [
                    jnp.stack(
                        [
                            model.local_covariance({"W": resampled[i]}, x)
                            for x in ref_points
                        ]
                    )
                    for i in range(200)
                ]
            ),
            axis=0,
        )

        assert jnp.allclose(pooled, by_hand, atol=1e-7)

    def test_band_matches_the_subsample_path(self, posterior, ref_points):
        """Quantiles sort their input, so the band is exactly order-invariant."""
        pooled = WPPMCovarianceFieldSamples.from_posterior(posterior).band(
            ref_points, ci_prob=0.89
        )
        subsampled = WPPMCovarianceFieldSamples.from_posterior(
            posterior, n_samples=200, key=jr.PRNGKey(0)
        ).band(ref_points, ci_prob=0.89)

        assert jnp.array_equal(pooled.lower, subsampled.lower)
        assert jnp.array_equal(pooled.median, subsampled.median)
        assert jnp.array_equal(pooled.upper, subsampled.upper)

    def test_fields_use_every_stored_draw(self, posterior, ref_points):
        fields = WPPMCovarianceFieldSamples.from_posterior(posterior)

        assert fields.n_samples == 200  # n_chains * n_draws

    def test_mean_needs_no_key(self, posterior, ref_points):
        """The common call keeps working: no key, no n_samples."""
        fields = WPPMCovarianceFieldSamples.from_posterior(posterior)

        assert fields.mean(ref_points).shape == (3, 2, 2)

    def test_band_needs_no_key(self, posterior, ref_points):
        fields = WPPMCovarianceFieldSamples.from_posterior(posterior)
        band = fields.band(ref_points, ci_prob=0.89)

        assert band.lower.shape == (3, 181)

    def test_repeated_evaluation_agrees(self, posterior, ref_points):
        """Two fields over the same posterior give the same answer, with no key
        passed to either."""
        a = WPPMCovarianceFieldSamples.from_posterior(posterior).mean(ref_points)
        b = WPPMCovarianceFieldSamples.from_posterior(posterior).mean(ref_points)

        assert jnp.array_equal(a, b)


class TestSubsamplingStillTakesAKey:
    """n_samples smaller than the pool is a real draw, so the key matters.

    These raise ValueError rather than TypeError: an argument cannot be required
    only when another one is set, so the check is in the body and the message says
    both what to pass and how to avoid needing it.
    """

    def test_subsample_needs_a_key(self, posterior, ref_points):
        """One check at construction now covers the mean and the band both."""
        with pytest.raises(ValueError, match="pass key="):
            WPPMCovarianceFieldSamples.from_posterior(posterior, n_samples=50)

    def test_different_keys_give_different_subsamples(self, posterior, ref_points):
        a = WPPMCovarianceFieldSamples.from_posterior(
            posterior, n_samples=50, key=jr.PRNGKey(0)
        ).mean(ref_points)
        b = WPPMCovarianceFieldSamples.from_posterior(
            posterior, n_samples=50, key=jr.PRNGKey(12345)
        ).mean(ref_points)

        assert not jnp.allclose(a, b, atol=1e-6)

    def test_point_mass_posterior_still_needs_n_samples(self, model, ref_points):
        """A MAPPosterior exposes no n_chains, so it cannot infer the count."""
        map_posterior = MAPPosterior(
            params=model.init_params(jr.PRNGKey(0)), model=model
        )

        with pytest.raises(ValueError, match="n_samples"):
            WPPMCovarianceFieldSamples.from_posterior(map_posterior)
