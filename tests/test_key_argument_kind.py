"""
Tests that `key` is positional by default, and keyword-only where a positional call
has 2 readings.

What these pin:

1. The 14 signatures below take `key` positionally or by name.
2. The 5 below take it only by name.
3. A positional key and a named key give the same draw.
"""

import inspect

import jax.random as jr
import pytest

from psyphy.model import WPPM, OddityTask, Prior, WPPMCovarianceFieldSamples
from psyphy.model.likelihood import (
    BernoulliTaskLikelihood,
    GaussianTaskLikelihood,
    TaskLikelihood,
)
from psyphy.posterior import (
    MAPPosterior,
    MCMCPosterior,
    ParameterSource,
    PredictivePosterior,
    WPPMPredictivePosterior,
)

POSITIONAL = [
    Prior.sample,
    ParameterSource.sample,
    MAPPosterior.sample,
    MCMCPosterior.sample,
    TaskLikelihood.simulate,
    TaskLikelihood.loglik,
    TaskLikelihood.predict,
    BernoulliTaskLikelihood.simulate,
    BernoulliTaskLikelihood.loglik,
    GaussianTaskLikelihood.simulate,
    GaussianTaskLikelihood.loglik,
    OddityTask.predict,
    WPPM.log_likelihood_from_data,
    WPPM.log_posterior_from_data,
]

KEYWORD_ONLY = [
    WPPMCovarianceFieldSamples.from_prior,  # same kind as __init__
    WPPMCovarianceFieldSamples.__init__,  # (post, k) would bind k to n_samples
    WPPMPredictivePosterior.__init__,  # required key after optional arguments
    PredictivePosterior.rsample,
    WPPMPredictivePosterior.rsample,
]


@pytest.mark.parametrize("fn", POSITIONAL, ids=lambda f: f.__qualname__)
def test_key_is_positional(fn):
    kind = inspect.signature(fn).parameters["key"].kind
    assert kind is inspect.Parameter.POSITIONAL_OR_KEYWORD


@pytest.mark.parametrize("fn", KEYWORD_ONLY, ids=lambda f: f.__qualname__)
def test_key_is_keyword_only(fn):
    kind = inspect.signature(fn).parameters["key"].kind
    assert kind is inspect.Parameter.KEYWORD_ONLY


def test_positional_key_gives_the_same_draw():
    prior = Prior(input_dim=2, basis_degree=2)
    key = jr.PRNGKey(0)
    by_position = prior.sample(3, key)
    by_name = prior.sample(3, key=key)
    assert (by_position["W"] == by_name["W"]).all()
