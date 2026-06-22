"""Tests for WPPM constructor guards around base-model kwargs.

These tests protect the "single source of truth" policy:
- task behavior knobs (e.g. MC controls) live in task configs, not in WPPM ctor kwargs.
- Base model lifecycle knobs (online learning) are explicitly named (online_config).

"""

import pytest

from psyphy.model import WPPM, Prior
from psyphy.model.likelihood import ContinuousTouchTask, OddityTask


def _make_minimal_wppm(task_type):
    input_dim = 2
    prior = Prior(input_dim=2, basis_degree=3)
    task = task_type
    return input_dim, prior, task


@pytest.mark.parametrize("bad_kw", ["num_samples", "bandwidth"])
@pytest.mark.parametrize("task_type", [OddityTask(), ContinuousTouchTask()])
def test_wppm_constructor_rejects_task_knobs_via_model_kwargs(task_type, bad_kw: str):
    input_dim, prior, task = _make_minimal_wppm(task_type)
    bad_value = 123  # misuse: these belong to OddityTaskConfig

    with pytest.raises(TypeError, match=r"Do not pass task-specific kwargs"):
        bad_kwargs = {bad_kw: bad_value}
        WPPM(
            input_dim=input_dim,
            prior=prior,
            likelihood=task,
            **bad_kwargs,
        )
