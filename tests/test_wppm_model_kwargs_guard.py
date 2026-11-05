"""Tests for WPPM constructor guards around base-model kwargs.

These tests protect the "single source of truth" policy:
- task behavior knobs (e.g. MC controls) live in task configs, not in WPPM ctor kwargs.
- prior knobs (input_dim, embedding size, variance_scale, decay_rate) live on Prior.
- Base model lifecycle knobs (online learning) are explicitly named (online_config).

"""

import pytest

from psyphy.model import WPPM, Prior
from psyphy.model.likelihood import OddityTask


def _make_minimal_wppm():
    input_dim = 2
    prior = Prior(input_dim=2, basis_degree=3)
    task = OddityTask()
    return input_dim, prior, task


@pytest.mark.parametrize("bad_kw", ["num_samples", "bandwidth"])
def test_wppm_constructor_rejects_task_knobs_via_model_kwargs(bad_kw: str):
    input_dim, prior, task = _make_minimal_wppm()
    bad_value = 123  # misuse: these belong to OddityTaskConfig

    with pytest.raises(TypeError, match=r"Do not pass task-specific kwargs"):
        bad_kwargs = {bad_kw: bad_value}
        WPPM(
            prior=prior,
            likelihood=task,
            **bad_kwargs,
        )


@pytest.mark.parametrize(
    "bad_kw", ["input_dim", "extra_dims", "variance_scale", "decay_rate"]
)
def test_wppm_constructor_rejects_prior_knobs_via_model_kwargs(bad_kw: str):
    _, prior, task = _make_minimal_wppm()

    with pytest.raises(TypeError, match=r"no longer takes"):
        WPPM(prior=prior, likelihood=task, **{bad_kw: 1})
