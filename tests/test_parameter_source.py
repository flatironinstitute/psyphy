"""
Tests for the ParameterSource protocol and Prior.sample.

What these pin:

1. Prior.sample(n, key=...) is sample_params over split keys, draw for draw. If
   that stops holding, a prior ensemble no longer matches the loop it replaces.
2. Prior, MAPPosterior and MCMCPosterior all satisfy ParameterSource, and all
   three return the same leading axis. A function written against the protocol
   breaks if any one of them drifts.
3. The three draw semantics behind the one method name: independent draws from a
   prior, one repeated parameter set from a MAP fit, a subsample of stored draws
   from an MCMC fit.
4. Every consumer of ParameterPosterior accepts a class with only the protocol's
   members, and gives the same numbers as for a MAPPosterior holding the same
   parameters.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from psyphy.model import (
    WPPM,
    GaussianNoise,
    OddityTask,
    Prior,
    WPPMCovarianceField,
    WPPMCovarianceFieldSamples,
)
from psyphy.posterior import (
    MAPPosterior,
    MCMCPosterior,
    ParameterPosterior,
    ParameterSource,
    WPPMPredictivePosterior,
)


@pytest.fixture
def model():
    return WPPM(Prior(input_dim=2, basis_degree=2), OddityTask())


@pytest.fixture
def prior():
    return Prior(input_dim=2, basis_degree=2)


@pytest.fixture
def mcmc_posterior(model):
    """Prior draws reshaped as 4 chains of 5 draws, standing in for a fit."""
    keys = jr.split(jr.PRNGKey(1), 20)
    W = jnp.stack([model.init_params(k)["W"] for k in keys])  # (20, 3, 3, 2, 3)
    return MCMCPosterior({"W": W.reshape(4, 5, *W.shape[1:])}, model)


# ==============================================================================
# Prior.sample: the vectorized form of sample_params
# ==============================================================================


class TestPriorSample:
    """Prior gains the n-draw signature the posteriors already have."""

    def test_leading_axis_is_n(self, prior):
        draws = prior.sample(7, key=jr.PRNGKey(0))

        # one draw is (degree+1, degree+1, input_dim, embedding_dim) = (3, 3, 2, 3)
        assert draws["W"].shape == (7, 3, 3, 2, 3)

    def test_agrees_with_sample_params_draw_by_draw(self, prior):
        """sample(n) is sample_params over split keys, not a different generator.

        This is the pin on the whole method: if it ever stops holding, sample()
        has grown its own randomness and a prior ensemble no longer matches the
        loop it replaces.
        """
        key = jr.PRNGKey(0)
        n = 5
        draws = prior.sample(n, key=key)

        keys = jr.split(key, n)
        for i in range(n):
            one = prior.sample_params(keys[i])
            assert jnp.array_equal(draws["W"][i], one["W"])

    def test_draws_differ_from_each_other(self, prior):
        """A prior generates i.i.d. draws, so no two rows are the same."""
        draws = prior.sample(4, key=jr.PRNGKey(0))["W"]

        for i in range(4):
            for j in range(i + 1, 4):
                assert not jnp.array_equal(draws[i], draws[j])

    def test_same_key_gives_same_draws(self, prior):
        a = prior.sample(3, key=jr.PRNGKey(42))["W"]
        b = prior.sample(3, key=jr.PRNGKey(42))["W"]

        assert jnp.array_equal(a, b)

    def test_three_dimensional_prior(self):
        """3-D priors carry a third basis axis, and vmap has to survive it."""
        prior_3d = Prior(input_dim=3, basis_degree=2)

        draws = prior_3d.sample(4, key=jr.PRNGKey(0))

        # (degree+1, degree+1, degree+1, input_dim, embedding_dim) = (3, 3, 3, 3, 4)
        assert draws["W"].shape == (4, 3, 3, 3, 3, 4)

    def test_sample_params_still_returns_one_draw(self, prior):
        """sample_params stays: model.init_params is on the inference path and
        wants one unbatched draw, not a batch of 1 to index."""
        one = prior.sample_params(jr.PRNGKey(0))

        assert one["W"].shape == (3, 3, 2, 3)


# ==============================================================================
# The protocol, and the three things that satisfy it
# ==============================================================================


class TestParameterSourceConformance:
    """Each source satisfies the protocol and returns the documented shape."""

    def test_prior_conforms(self, prior):
        assert isinstance(prior, ParameterSource)

    def test_map_posterior_conforms(self, model):
        posterior = MAPPosterior(params=model.init_params(jr.PRNGKey(0)), model=model)

        assert isinstance(posterior, ParameterSource)

    def test_mcmc_posterior_conforms(self, mcmc_posterior):
        assert isinstance(mcmc_posterior, ParameterSource)

    def test_parameter_posterior_extends_parameter_source(self):
        """A posterior is a source plus params and model, so the waist is the
        narrower of the two protocols."""
        from psyphy.posterior import ParameterPosterior

        assert issubclass(ParameterPosterior, ParameterSource)

    @pytest.mark.parametrize("n", [1, 3, 10])
    def test_every_source_gives_the_same_leading_axis(
        self, prior, model, mcmc_posterior, n
    ):
        """One signature, one shape contract, whatever the parameters came from."""
        map_posterior = MAPPosterior(
            params=model.init_params(jr.PRNGKey(0)), model=model
        )
        sources = [prior, map_posterior, mcmc_posterior]

        for source in sources:
            draws = source.sample(n, key=jr.PRNGKey(0))
            assert draws["W"].shape[0] == n, (
                f"{type(source).__name__} gave the wrong axis"
            )


class TestDrawSemantics:
    """The three meanings behind the one method name."""

    def test_map_repeats_one_parameter_set(self, model):
        """A MAP fit is a point mass, so every draw is the same parameter set.

        Checked as exact row equality rather than zero variance: jnp.var over
        identical float32 rows leaves about 3e-18 of summation residue, so a
        variance test would pin a number that has nothing to do with the claim.
        """
        params = model.init_params(jr.PRNGKey(0))
        posterior = MAPPosterior(params=params, model=model)

        draws = posterior.sample(5, key=jr.PRNGKey(0))["W"]

        for i in range(5):
            assert jnp.array_equal(draws[i], params["W"])

    def test_mcmc_draws_come_from_the_stored_pool(self, mcmc_posterior):
        """Every returned draw matches some stored draw: MCMC subsamples, it does
        not generate."""
        pool = mcmc_posterior.samples["W"].reshape(20, 3, 3, 2, 3)

        draws = mcmc_posterior.sample(6, key=jr.PRNGKey(0))["W"]

        for i in range(6):
            assert any(jnp.array_equal(draws[i], pool[j]) for j in range(20))

    def test_a_source_is_enough_to_build_an_ensemble(self, model, prior):
        """What the protocol is for: one function over a prior and a posterior
        alike, which needed a loop and a fake posterior before."""

        def covariances(source, points, n, key):
            draws = source.sample(n, key=key)
            return jax.vmap(
                lambda W: jax.vmap(lambda x: model.local_covariance({"W": W}, x))(
                    points
                )
            )(draws["W"])

        points = jnp.array([[0.0, 0.0], [0.5, 0.5]])  # (n_points=2, d=2)

        from_prior = covariances(prior, points, 3, jr.PRNGKey(0))
        map_posterior = MAPPosterior(
            params=model.init_params(jr.PRNGKey(0)), model=model
        )
        from_map = covariances(map_posterior, points, 3, jr.PRNGKey(0))

        assert from_prior.shape == (3, 2, 2, 2)  # (n_draws, n_points, d, d)
        assert from_map.shape == (3, 2, 2, 2)


# ==============================================================================
# Consumers read only the protocol's members
# ==============================================================================


class MinimalPosterior:
    """params, model and sample, and nothing else."""

    def __init__(self, params, model):
        self.params = params
        self.model = model

    def sample(self, n, *, key):
        return jax.tree.map(lambda x: jnp.broadcast_to(x, (n, *x.shape)), self.params)


class TestMinimalPosteriorThroughConsumers:
    """A posterior from outside psyphy works wherever a MAPPosterior does.

    Each test compares against a MAPPosterior with the same parameters, so a
    consumer that reaches for a private attribute fails with AttributeError.
    """

    @pytest.fixture
    def pair(self):
        # predict_prob needs a noise model, which the module fixture leaves as None
        model = WPPM(
            Prior(input_dim=2, basis_degree=2),
            OddityTask(),
            GaussianNoise(),
        )
        params = model.init_params(jr.PRNGKey(0))
        return MinimalPosterior(params, model), MAPPosterior(params, model)

    @pytest.fixture
    def points(self):
        return jnp.array([[0.0, 0.0], [0.5, -0.3]])  # (n_points=2, d=2)

    def test_conforms(self, pair):
        minimal, _ = pair
        assert isinstance(minimal, ParameterPosterior)

    def test_covariance_field_from_posterior(self, pair, points):
        minimal, map_posterior = pair
        a = WPPMCovarianceField.from_posterior(minimal)(points)
        b = WPPMCovarianceField.from_posterior(map_posterior)(points)
        assert jnp.array_equal(a, b)

    def test_covariance_field_samples_constructor(self, pair, points):
        minimal, map_posterior = pair
        key = jr.PRNGKey(0)
        a = WPPMCovarianceFieldSamples(minimal, n_samples=3, key=key)
        b = WPPMCovarianceFieldSamples(map_posterior, n_samples=3, key=key)
        assert jnp.array_equal(a.mean(points), b.mean(points))

    def test_predictive_posterior(self, pair, points):
        minimal, map_posterior = pair
        X = jnp.stack(
            [points, points + 0.1], axis=1
        )  # (n_test=2, 2: ref, comparison, d=2)
        key = jr.PRNGKey(0)
        a = WPPMPredictivePosterior(minimal, X, n_samples=2, key=key)
        b = WPPMPredictivePosterior(map_posterior, X, n_samples=2, key=key)
        assert jnp.array_equal(a.mean, b.mean)
        assert jnp.array_equal(a.variance, b.variance)
