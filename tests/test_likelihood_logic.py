"""
test_likelihood_logic.py
---------------------

Tests to ensure that the TaskLikelihoods that are implemented behave as expected

Currently tests cases for: BernoulliTaskLikelihood, GaussianTaskLikelihood

"""

import math

import jax.numpy as jnp
import pytest

from psyphy.data import TrialData
from psyphy.model.likelihood import BernoulliTaskLikelihood, GaussianTaskLikelihood


class MockBernoulliTask(BernoulliTaskLikelihood):
    def predict(self, params, stimuli, model, key, noise=None):
        return jnp.array(0.4)  # p_correct


class MockGaussianTask(GaussianTaskLikelihood):
    def predict(self, params, stimuli, model, key, noise=None):
        return (jnp.array(0), jnp.array(1))  # 1D mu, sigma


class MockMultiGaussianTask(GaussianTaskLikelihood):
    def predict(self, params, stimuli, model, key, noise=None):
        return jnp.array([0, 0]), jnp.array([[1, 0], [0, 1]])  # 2D mu, sigma


class MockEvilGaussianTask(GaussianTaskLikelihood):
    """
    This predict function generates a covariance matrix that is not positive
    definite. Gaussian abstract subclass should throw an error in this case, as
    concrete tasks should always implement 'predict' such that it gives positive
    definite covariance matrices.
    """

    def predict(self, params, stimuli, model, key, noise=None):
        return jnp.array([0, 0]), jnp.array([[1, 1], [1, 1]])


class TestBernoulli:
    """
    Ensure Bernoulli Task likelihood calculation has sound logic
    """

    @pytest.fixture
    def data(self):
        """Create simple data for testing.
        Note that the shape of responses for a single trial can be (N,1) or (N,)
        for loglik. TrialData will normalize to (N,1) allowing for appropriate
        shape checks. If (N,) is passed directly instead, then checks may be
        bypassed.
        """
        stimuli = jnp.array([[[1, 1], [0, 0]], [[1, 1], [0, 0]]])
        responses = jnp.array([1, 0])
        return TrialData(stimuli=stimuli, responses=responses)

    def test_bernoulli_likelihood(self, data):
        """ensure Bernoulli log-likelihood calculation works as expected"""
        true_loglik = jnp.sum(jnp.log(jnp.array([0.4, 0.6])))  # by construction
        task = MockBernoulliTask()

        assert task.loglik(params="NA", data=data, model="NA") == true_loglik


class TestGaussian:
    """
    Ensure Gaussian Task likelihood calculation has sound logic
    """

    @pytest.fixture
    def single_data(self):
        """Create simple 1D response data for testing."""
        stimuli = jnp.array([[[1, 5], [0.2, 0]], [[1, 1], [0, 0]]])
        responses = jnp.array([0, 0])
        return TrialData(stimuli=stimuli, responses=responses)

    @pytest.fixture
    def multi_data(self):
        """Create simple 2D response data for testing."""
        stimuli = jnp.array([[[1, 5], [0.2, 0]], [[1, 1], [0, 0]]])
        responses = jnp.array([[0, 0], [0, 0]])
        return TrialData(stimuli=stimuli, responses=responses)

    def test_gaussian_likelihood(self, single_data):
        """ensure 1D Gaussian log-likelihood calculation works as expected"""
        task = MockGaussianTask()

        loglik_at_mean = -0.5 * math.log(2 * math.pi)

        true_loglik = jnp.sum(
            jnp.array([loglik_at_mean, loglik_at_mean])
        )  # by construction

        assert task.loglik(params="NA", data=single_data, model="NA") == true_loglik

    def test_multi_gaussian_likelihood(self, multi_data):
        """ensure 2D Gaussian log-likelihood calculation works as expected"""
        task = MockMultiGaussianTask()

        loglik_at_mean = -math.log(2 * math.pi)
        true_loglik = jnp.sum(
            jnp.array([loglik_at_mean, loglik_at_mean])
        )  # by construction

        assert task.loglik(params="NA", data=multi_data, model="NA") == true_loglik

    def test_gaussian_likelihood_errors(self, multi_data):
        """ensure Gaussian log-likelihood calculation throws appropriate errors."""
        task = MockEvilGaussianTask()

        with pytest.warns(UserWarning):
            assert task.loglik(params="NA", data=multi_data, model="NA")
