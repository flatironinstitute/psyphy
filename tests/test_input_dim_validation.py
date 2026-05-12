import jax.numpy as jnp
import pytest

from psyphy.model.likelihood import BernoulliTaskLikelihood
from psyphy.model.prior import Prior
from psyphy.model.wppm import WPPM


class MockTask(BernoulliTaskLikelihood):
    def predict(self, params, stimuli, model, noise=None):
        return jnp.array(0.5)

    def loglik(self, params, data, model, noise=None):
        return jnp.array(0.0)


def test_input_dim_mismatch_raises_error():
    """Test that initializing WPPM with mismatching input_dim raises ValueError."""
    input_dim_prior = 2
    input_dim_model = 3

    prior = Prior(input_dim=input_dim_prior)
    task = MockTask()

    with pytest.raises(ValueError, match="Dimension mismatch"):
        WPPM(input_dim=input_dim_model, prior=prior, likelihood=task)


def test_input_dim_match_success():
    """Test that initializing WPPM with matching input_dim succeeds."""
    input_dim = 2

    prior = Prior(input_dim=input_dim)
    task = MockTask()

    model = WPPM(input_dim=input_dim, prior=prior, likelihood=task)
    assert model.input_dim == input_dim
    assert model.prior.input_dim == input_dim
