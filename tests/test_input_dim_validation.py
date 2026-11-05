import jax.numpy as jnp
import pytest

from psyphy.model.likelihood import BernoulliTaskLikelihood, GaussianTaskLikelihood
from psyphy.model.prior import Prior
from psyphy.model.wppm import WPPM


class MockTask(BernoulliTaskLikelihood):
    def predict(self, params, stimuli, model, noise=None):
        return jnp.array(0.5)  # p_correct

    def loglik(self, params, data, model, noise=None):
        return jnp.array(0.0)


class MockGaussianTask(GaussianTaskLikelihood):
    def predict(self, params, stimuli, model, noise=None):
        return (jnp.array(0.0), jnp.array([[1.0]]))  # 1D mu, sigma

    def loglik(self, params, data, model, noise=None):
        return jnp.array(0.0)


class MockMultiGaussianTask(GaussianTaskLikelihood):
    def predict(self, params, stimuli, model, noise=None):
        return (jnp.array([0.0, 0.0]), jnp.eye(2))  # 2D mu, sigma

    def loglik(self, params, data, model, noise=None):
        return jnp.array(0.0)


def test_input_dim_reads_from_prior():
    """WPPM.input_dim delegates to prior.input_dim; there is no second copy to mismatch."""
    prior = Prior(input_dim=3)
    model = WPPM(prior=prior, likelihood=MockTask())
    assert model.input_dim == 3
    assert model.prior.input_dim == 3


def test_input_dim_kwarg_on_wppm_raises():
    """A caller cannot pass input_dim to WPPM; it only takes it through Prior."""
    with pytest.raises(TypeError, match="no longer takes"):
        WPPM(prior=Prior(input_dim=2), likelihood=MockTask(), input_dim=2)


def test_input_dim_match_success_gaussians():
    """WPPM.input_dim agrees with the prior for both Gaussian task shapes."""
    input_dim = 2

    prior = Prior(input_dim=input_dim)
    task = MockGaussianTask()
    multi_task = MockMultiGaussianTask()

    model = WPPM(prior=prior, likelihood=task)
    assert model.input_dim == input_dim
    assert model.prior.input_dim == input_dim
    multi_model = WPPM(prior=prior, likelihood=multi_task)
    assert multi_model.input_dim == input_dim
    assert multi_model.prior.input_dim == input_dim
