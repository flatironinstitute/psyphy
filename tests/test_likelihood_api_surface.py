"""Characterization test for the public surface of psyphy.model.likelihood.

Written before the module was split into a package (issue #01) and run green
against the pre-split code, so that "the split changed no behaviour" is a
checked claim rather than an intention. It pins three things:

1. every public name, at each of the three import paths it is reachable from;
2. the class hierarchy, because two `isinstance` gates elsewhere depend on it
   (`WPPM.predict_prob` and `WPPMPredictivePosterior` both require a
   `BernoulliTaskLikelihood`, so a task that quietly stops being one keeps
   fitting while silently breaking every prediction path);
3. the `predict` return contract, which is a tuple of sufficient statistics.
   A bare array here is not a type error: `BernoulliTaskLikelihood.loglik`
   indexes element 0 after vmapping, so a scalar return silently collapses
   every trial onto the first trial's probability.

This file should keep passing unchanged through the package split.
"""

from __future__ import annotations

import importlib

import jax.numpy as jnp
import jax.random as jr
import pytest

# --------------------------------------------------------------------------
# 1. public names, at every path they are importable from
# --------------------------------------------------------------------------

LIKELIHOOD_NAMES = [
    "TaskLikelihood",
    "BernoulliTaskLikelihood",
    "GaussianTaskLikelihood",
    "OddityTask",
    "OddityTaskConfig",
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

    assert isinstance(out, tuple), (
        "predict must return a tuple of sufficient statistics; a bare array makes "
        "BernoulliTaskLikelihood.loglik silently use trial 0 for every trial"
    )
    assert len(out) == 1, "Bernoulli tasks return exactly (p_correct,)"
    assert jnp.ndim(out[0]) == 0, "p_correct is a scalar per trial"
    assert 0.0 < float(out[0]) < 1.0


def test_loglik_uses_each_trial_separately():
    """Guards the failure a constant-probability mock cannot detect.

    Two trials with different p: the result must be the sum of two
    different log terms, not twice the first one.
    """
    import types

    from psyphy.model.likelihood import BernoulliTaskLikelihood

    class VaryingTask(BernoulliTaskLikelihood):
        def predict(self, params, stimuli, model, *, key=None):
            return (jnp.where(stimuli[0, 0] > 0.5, 0.9, 0.1),)

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
