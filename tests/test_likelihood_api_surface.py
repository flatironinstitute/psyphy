"""Characterization test for the public surface of psyphy.model.likelihood.

Written before the module was split into a package (issue #01) and run green
against the pre-split code, so that "the split changed no behaviour" is a
checked claim rather than an intention. It pins three things:

1. every public name, at each of the three import paths it is reachable from;
2. the class hierarchy, because two `isinstance` gates elsewhere depend on it
   (`WPPM.predict_prob` and `WPPMPredictivePosterior` both require a
   `BernoulliTaskLikelihood`, so a task that quietly stops being one keeps
   fitting while silently breaking every prediction path);
3. the `predict` return contract: a Bernoulli task returns the tuple
   `(log_p, log_1mp)`, and every consumer of `predict` accepts a minimal task
   that returns it (section 4).

This file should keep passing unchanged through the package split.
"""

from __future__ import annotations

import importlib

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from psyphy.model.likelihood import BernoulliTaskLikelihood, log_probs_from_logit

# --------------------------------------------------------------------------
# 1. public names, at every path they are importable from
# --------------------------------------------------------------------------

LIKELIHOOD_NAMES = [
    "TaskLikelihood",
    "BernoulliTaskLikelihood",
    "GaussianTaskLikelihood",
    "OddityTask",
    "OddityTaskConfig",
    "log_probs_from_logit",
]

# Names re-exported one level up, from psyphy.model. A subset:
# the two intermediate ABCs are not currently exported there, and this test
# records that rather than asserting it ought to change.
MODEL_NAMES = ["TaskLikelihood", "OddityTask", "OddityTaskConfig"]


@pytest.mark.parametrize("name", LIKELIHOOD_NAMES)
def test_importable_from_likelihood_module(name):
    mod = importlib.import_module("psyphy.model.likelihood")
    assert hasattr(mod, name), f"psyphy.model.likelihood lost {name}"


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_importable_from_model_package(name):
    mod = importlib.import_module("psyphy.model")
    assert hasattr(mod, name), f"psyphy.model lost {name}"
    assert name in mod.__all__, f"{name} missing from psyphy.model.__all__"


@pytest.mark.parametrize("name", ["OddityTask", "OddityTaskConfig"])
def test_importable_from_top_level(name):
    psyphy = importlib.import_module("psyphy")
    assert hasattr(psyphy, name), f"psyphy lost {name}"


def test_likelihood_is_importable_as_a_module_path():
    """`from psyphy.model.likelihood import X` must keep working.

    This is the import style used across the test suite and the docs, and it is
    what a package split is most likely to break.
    """
    from psyphy.model.likelihood import (  # noqa: F401
        BernoulliTaskLikelihood,
        GaussianTaskLikelihood,
        OddityTask,
        OddityTaskConfig,
        TaskLikelihood,
    )


# --------------------------------------------------------------------------
# 2. class hierarchy -- load-bearing for two isinstance gates
# --------------------------------------------------------------------------


def test_hierarchy():
    from psyphy.model.likelihood import (
        BernoulliTaskLikelihood,
        GaussianTaskLikelihood,
        OddityTask,
        TaskLikelihood,
    )

    assert issubclass(BernoulliTaskLikelihood, TaskLikelihood)
    assert issubclass(GaussianTaskLikelihood, TaskLikelihood)
    assert issubclass(OddityTask, BernoulliTaskLikelihood)
    # ... and therefore, transitively:
    assert issubclass(OddityTask, TaskLikelihood)
    # The two ABCs are siblings, not ancestors of one another.
    assert not issubclass(BernoulliTaskLikelihood, GaussianTaskLikelihood)
    assert not issubclass(GaussianTaskLikelihood, BernoulliTaskLikelihood)


def test_oddity_task_instance_passes_the_bernoulli_gate():
    """The gate `WPPM.predict_prob` and `WPPMPredictivePosterior` apply."""
    from psyphy.model.likelihood import BernoulliTaskLikelihood, OddityTask

    assert isinstance(OddityTask(), BernoulliTaskLikelihood)


# --------------------------------------------------------------------------
# 3. the predict() return contract
# --------------------------------------------------------------------------


def _toy_model():
    from psyphy.model import WPPM, GaussianNoise, OddityTask, OddityTaskConfig, Prior

    return WPPM(
        prior=Prior(input_dim=2, basis_degree=1),
        likelihood=OddityTask(OddityTaskConfig(num_samples=20)),
        noise=GaussianNoise(),
    )


def test_predict_returns_a_tuple_not_a_bare_array():
    model = _toy_model()
    params = model.init_params(jr.PRNGKey(0))
    stimuli = jnp.stack([jnp.array([0.1, 0.0]), jnp.array([0.3, 0.2])], axis=0)

    out = model.likelihood.predict(params, stimuli, model, key=jr.PRNGKey(1))

    assert isinstance(out, tuple), "predict must return a tuple"
    assert len(out) == 2, "Bernoulli tasks return exactly (log_p, log_1mp)"
    log_p, log_1mp = out
    assert jnp.ndim(log_p) == 0 and jnp.ndim(log_1mp) == 0, "one scalar each per trial"
    assert float(log_p) < 0.0 and float(log_1mp) < 0.0
    # complementary outcomes; float32 rounding is about 1e-7, so 1e-5 is 100x headroom
    assert float(jnp.exp(log_p) + jnp.exp(log_1mp)) == pytest.approx(1.0, abs=1e-5)


def test_loglik_uses_each_trial_separately():
    """Guards the failure a constant-probability mock cannot detect.

    Two trials with different p: the result must be the sum of two
    different log terms, not twice the first one.
    """
    import types

    from psyphy.model.likelihood import BernoulliTaskLikelihood

    class VaryingTask(BernoulliTaskLikelihood):
        def predict(self, params, stimuli, model, *, key=None):
            p = jnp.where(stimuli[0, 0] > 0.5, 0.9, 0.1)
            return jnp.log(p), jnp.log1p(-p)

    data = types.SimpleNamespace(
        stimuli=jnp.array(
            [[[1.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]
        ),  # -> p = 0.9, then 0.1
        responses=jnp.array([[1], [1]]),
    )
    got = float(VaryingTask().loglik("unused", data, "unused"))
    expected = float(jnp.log(0.9) + jnp.log(0.1))
    collapsed = float(2 * jnp.log(0.9))  # what a trial-0-only bug would give

    assert got == pytest.approx(expected, abs=1e-5), (
        f"loglik={got}, expected {expected}; {collapsed} would mean every trial "
        "is using trial 0's probability"
    )


# --------------------------------------------------------------------------
# 4. a minimal conforming Bernoulli task, through every consumer of predict
# --------------------------------------------------------------------------

SLOPE = 50.0  # logit units per unit of distance
RADIUS = 0.1  # distance where p(correct) = 2/3, the threshold criterion


class DistanceTask(BernoulliTaskLikelihood):
    """p(correct) = sigmoid(SLOPE * (distance - RADIUS) + log 2): 2/3 at RADIUS."""

    def predict(self, params, stimuli, model, *, key=None):
        distance = jnp.linalg.norm(stimuli[1] - stimuli[0])
        return log_probs_from_logit(SLOPE * (distance - RADIUS) + jnp.log(2.0))


def _distance_model():
    from psyphy.model import WPPM, GaussianNoise, Prior

    return WPPM(
        prior=Prior(input_dim=2, basis_degree=1),
        likelihood=DistanceTask(),
        noise=GaussianNoise(),
    )


def _pair(distance, axis=0):
    """(K=2, d=2) stimuli, reference at the origin, comparison `distance` away."""
    return jnp.stack([jnp.zeros(2), jnp.zeros(2).at[axis].set(distance)])


def test_log_probs_from_logit_matches_the_sigmoid():
    z = jnp.array([-4.0, 0.0, 3.0])
    log_p, log_1mp = log_probs_from_logit(z)
    assert jnp.allclose(jnp.exp(log_p), jax.nn.sigmoid(z), atol=1e-6)
    assert jnp.allclose(jnp.exp(log_1mp), 1.0 - jax.nn.sigmoid(z), atol=1e-6)


def test_minimal_task_through_loglik():
    """loglik takes log p(correct) for a 1 and log p(incorrect) for a 0."""
    import types

    data = types.SimpleNamespace(
        stimuli=jnp.stack([_pair(0.2), _pair(0.0)]),  # (2, K=2, d=2)
        responses=jnp.array([[1], [0]]),
    )
    z = SLOPE * (jnp.array([0.2, 0.0]) - RADIUS) + jnp.log(2.0)  # (2,)
    expected = jnp.log(jax.nn.sigmoid(z[0])) + jnp.log(1.0 - jax.nn.sigmoid(z[1]))

    got = DistanceTask().loglik(None, data, None)

    assert float(got) == pytest.approx(float(expected), abs=1e-5)


def test_loglik_holds_each_trial_within_1e_6_of_0_and_1():
    """A saturated trial adds log(1e-6) when wrong and log(1 - 1e-6) when right."""
    import types

    far = jnp.stack([_pair(1.0)])  # (1, K=2, d=2); logit about 46

    wrong = DistanceTask().loglik(
        None, types.SimpleNamespace(stimuli=far, responses=jnp.array([[0]])), None
    )
    right = DistanceTask().loglik(
        None, types.SimpleNamespace(stimuli=far, responses=jnp.array([[1]])), None
    )

    # unclipped, wrong would be about -46 and right about -1e-20
    assert float(wrong) == pytest.approx(float(jnp.log(1e-6)), abs=1e-5)
    assert float(right) == pytest.approx(float(jnp.log1p(-1e-6)), abs=1e-7)


def test_minimal_task_through_simulate():
    """simulate returns (responses, (p_correct,)), with p_correct = exp(log_p)."""
    stimuli = jnp.stack([_pair(RADIUS), _pair(1.0)])  # p = 2/3, then p rounds to 1
    responses, (p_correct,) = DistanceTask().simulate(
        None, stimuli, None, key=jr.PRNGKey(0)
    )

    assert responses.shape == (2,)
    assert p_correct.shape == (2,)
    assert float(p_correct[0]) == pytest.approx(2.0 / 3.0, abs=1e-5)
    assert int(responses[1]) == 1  # p_correct == 1.0, so every draw is correct


def test_minimal_task_through_predict_prob():
    model = _distance_model()
    params = model.init_params(jr.PRNGKey(0))  # DistanceTask ignores them

    p = model.predict_prob(params, (jnp.zeros(2), jnp.array([RADIUS, 0.0])))

    assert float(p) == pytest.approx(2.0 / 3.0, abs=1e-5)


def test_minimal_task_through_the_predictive_posterior():
    """Both modes: p(correct) at paired stimuli, and the recovered threshold."""
    from psyphy.posterior import MAPPosterior, ThresholdConfig, WPPMPredictivePosterior

    model = _distance_model()
    posterior = MAPPosterior(model, model.init_params(jr.PRNGKey(0)))

    X_pairs = jnp.stack([_pair(RADIUS, axis=0), _pair(RADIUS, axis=1)])  # (2, K=2, d=2)
    probs = WPPMPredictivePosterior(posterior, X_pairs, n_samples=2, key=jr.PRNGKey(0))
    assert jnp.allclose(probs.mean, 2.0 / 3.0, atol=1e-5)

    # grid step 0.01 with RADIUS on the grid, so the threshold is a circle of radius RADIUS
    cfg = ThresholdConfig(n_theta=8, n_length=20, bounds=(0.01, 0.2))
    X_refs = jnp.array([[0.0, 0.0], [0.5, 0.5]])  # (n_test=2, d=2)
    thresholds = WPPMPredictivePosterior(
        posterior,
        X_refs,
        n_samples=1,
        threshold_pred=True,
        threshold_config=cfg,
        key=jr.PRNGKey(0),
    )
    # off by about 2e-9 here; one grid step off moves an entry by 2e-3, 20 times the tolerance
    assert jnp.allclose(thresholds.mean, RADIUS**2 * jnp.eye(2), atol=1e-4)
