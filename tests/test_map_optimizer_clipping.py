"""
test_map_optimizer_clipping.py
------------------------------

Gradient-clipping behaviour of :class:`~psyphy.inference.MAPOptimizer`.

Clipping is off by default. When it is on, the useful question is not whether
it ran but *how often it bound*: a threshold below the model's typical gradient
norm binds on every step, which fixes the step length at
``learning_rate * max_grad_norm`` and turns SGD into normalized gradient
descent. These tests pin both regimes via ``clip_rate``.
"""

import warnings

import jax.numpy as jnp
import jax.random as jr
import pytest

from psyphy.data import ResponseData
from psyphy.inference import MAPOptimizer
from psyphy.model import WPPM, Prior
from psyphy.model.likelihood import OddityTask
from psyphy.model.noise import GaussianNoise

# Thresholds chosen to be unambiguously below / above any plausible gradient
# norm, so these tests do not depend on the model's actual scale.
ALWAYS_CLIPS = 1e-8
NEVER_CLIPS = 1e12


@pytest.fixture
def model():
    return WPPM(
        prior=Prior(input_dim=2, basis_degree=3),
        likelihood=OddityTask(),
        noise=GaussianNoise(),
    )


@pytest.fixture
def data():
    k1, k2 = jr.split(jr.PRNGKey(42))
    refs = jr.normal(k1, (20, 2))
    comps = refs + jr.normal(k2, (20, 2)) * 0.3
    d = ResponseData()
    for i in range(20):
        d.add_trial((refs[i], comps[i]), 1)
    return d


def test_clipping_is_off_by_default():
    """Clipping is opt-in: a threshold is model-specific, so there is no safe default."""
    assert MAPOptimizer().max_grad_norm is None


def test_clip_rate_is_none_when_clipping_off(model, data):
    opt = MAPOptimizer(steps=5)
    opt.fit(model, data, seed=0)
    assert opt.clip_rate is None
    assert opt.n_clipped_steps == 0


def test_clip_rate_one_and_warns_when_threshold_below_gradient_scale(model, data):
    """The failure mode worth catching: clipping silently replacing the optimizer."""
    opt = MAPOptimizer(steps=5, max_grad_norm=ALWAYS_CLIPS)
    with pytest.warns(UserWarning, match="clipping bound on"):
        opt.fit(model, data, seed=0)
    assert opt.clip_rate == 1.0
    assert opt.n_clipped_steps == 5


def test_clip_rate_zero_and_silent_when_threshold_above_gradient_scale(model, data):
    opt = MAPOptimizer(steps=5, max_grad_norm=NEVER_CLIPS)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        opt.fit(model, data, seed=0)
    assert opt.clip_rate == 0.0
    assert opt.n_clipped_steps == 0


def test_clip_counters_reset_between_fits(model, data):
    """A second fit must not inherit the first fit's counts."""
    opt = MAPOptimizer(steps=4, max_grad_norm=ALWAYS_CLIPS)
    with pytest.warns(UserWarning):
        opt.fit(model, data, seed=0)
    assert opt.n_clipped_steps == 4
    with pytest.warns(UserWarning):
        opt.fit(model, data, seed=1)
    assert opt.n_clipped_steps == 4  # not 8
    assert opt.clip_rate == 1.0


def test_clipping_does_not_change_the_returned_parameter_tree(model, data):
    """Clipping affects step size, not structure -- guards the optax.chain wiring."""
    a = MAPOptimizer(steps=3, max_grad_norm=None).fit(model, data, seed=0)
    b = MAPOptimizer(steps=3, max_grad_norm=NEVER_CLIPS).fit(model, data, seed=0)
    assert jax_tree_keys(a.params) == jax_tree_keys(b.params)
    assert jnp.asarray(a.params["W"]).shape == jnp.asarray(b.params["W"]).shape


def jax_tree_keys(params):
    return sorted(params.keys())


@pytest.mark.parametrize("bad", ["average", "Mean", "", None])
def test_reduction_is_validated(bad):
    with pytest.raises(ValueError, match="reduction must be"):
        MAPOptimizer(reduction=bad)
