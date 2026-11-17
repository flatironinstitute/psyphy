"""
tests/test_inference_engine.py
------------------------------

The contract every psyphy `InferenceEngine` shares: one `fit` signature, and
one place to set the seed. A scientist who swaps MAPOptimizer for NUTSSampler
types the same call, so a mismatch here breaks their script.
"""

from __future__ import annotations

import inspect

import pytest

import psyphy.inference as inference
from psyphy.inference import (
    InferenceEngine,
    LangevinSampler,
    LaplaceApproximation,
    MAPOptimizer,
    NUTSSampler,
)

ENGINES = [MAPOptimizer, NUTSSampler, LangevinSampler, LaplaceApproximation]


def _fit_params(cls):
    return list(inspect.signature(cls.fit).parameters)


@pytest.mark.parametrize("cls", ENGINES, ids=lambda c: c.__name__)
def test_fit_signature_matches_abc(cls):
    # same names in the same order, so positional and keyword calls bind alike
    assert _fit_params(cls) == _fit_params(InferenceEngine)
    assert _fit_params(InferenceEngine) == [
        "self",
        "model",
        "data",
        "init_params",
        "seed",
    ]


@pytest.mark.parametrize("cls", ENGINES, ids=lambda c: c.__name__)
def test_fit_seed_defaults_to_none(cls):
    sig = inspect.signature(cls.fit)
    assert sig.parameters["seed"].default is None
    assert sig.parameters["init_params"].default is None


def test_nuts_takes_no_seed_in_constructor():
    # the seed belongs to a run, so it goes to fit only
    assert "seed" not in inspect.signature(NUTSSampler.__init__).parameters
    with pytest.raises(TypeError):
        NUTSSampler(seed=1)


def test_laplace_constructs_and_fit_is_not_implemented():
    engine = LaplaceApproximation()
    with pytest.raises(NotImplementedError):
        engine.fit(model=None, data=None)


def test_no_engine_registry():
    assert not hasattr(inference, "INFERENCE_ENGINES")
    assert "INFERENCE_ENGINES" not in inference.__all__


def test_numpyro_stub_removed():
    import psyphy.inference.langevin as langevin

    assert not hasattr(langevin, "NumpyroSampler")
