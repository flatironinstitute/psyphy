"""
test_noise_models.py
--------------------

Tests for different noise models (Gaussian vs Student-t) in WPPM.
"""

import jax.numpy as jnp
import jax.random as jr
import pytest

from psyphy.data import ResponseData
from psyphy.model import (
    WPPM,
    ContinuousTouchRule,
    ContinuousTouchTask,
    ContinuousTouchTaskConfig,
    GaussianNoise,
    OddityTask,
    OddityTaskConfig,
    Prior,
    StudentTNoise,
)

# Define the oddity task used for testing
oddity_task = OddityTask(config=OddityTaskConfig(num_samples=2000, bandwidth=1e-2))

# Define the touch tasks used for testing.
# Since we want to test noise across different reasonable uses, we will include
# two touch tasks: one with closed-form and the other with simulation

# closed form touch task:
touch_task = ContinuousTouchTask()


# MC simulation touch task:
def rule_func(x):
    return jnp.array([jnp.exp(x[0]), x[1] * x[0] * x[0]])


mc_touch_task = ContinuousTouchTask(
    config=ContinuousTouchTaskConfig(
        num_samples=2000, rule=ContinuousTouchRule(nonlinear_func=rule_func)
    )
)
# Define appropriate data for each type
oddity_data = ResponseData()
oddity_data.add_trial(input=(jnp.array([0.0, 0.0]), jnp.array([0.1, 0.1])), resp=1)
touch_data = ResponseData()
touch_data.add_trial(input=(jnp.array([0.1, 0.1]),), resp=jnp.array([0.09, 0.11]))


@pytest.mark.parametrize(
    "task, data",
    [(oddity_task, oddity_data), (touch_task, touch_data), (mc_touch_task, touch_data)],
)
class TestNoiseModels:
    """Test that different noise models affect MC likelihood."""

    @pytest.fixture
    def model_gaussian(self, task):
        return WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=3),
            likelihood=task,
            noise=GaussianNoise(sigma=0.03),
        )

    @pytest.fixture
    def model_student_t(self, task):
        return WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=3),
            likelihood=task,
            noise=StudentTNoise(df=3.0, scale=0.03),
        )

    @pytest.fixture
    def data(self, data):
        return data

    def test_noise_models_give_different_results(
        self, model_gaussian, model_student_t, data
    ):
        """
        Test that Gaussian and Student-t noise produce different likelihoods.

        Student-t has heavier tails, so it should behave differently, especially
        for outliers or when discriminability is marginal.
        """
        params = model_gaussian.init_params(jr.PRNGKey(42))

        # Compute likelihood with Gaussian noise
        ll_gaussian = model_gaussian.likelihood.loglik(
            params=params,
            data=data,
            model=model_gaussian,
            key=jr.PRNGKey(0),
        )

        # Compute likelihood with Student-t noise
        ll_student = model_student_t.likelihood.loglik(
            params=params,
            data=data,
            model=model_student_t,
            key=jr.PRNGKey(0),
        )

        # They should be different
        # Note: We use a large sample size to ensure difference isn't just MC noise
        assert not jnp.isclose(ll_gaussian, ll_student, rtol=1e-3), (
            f"Gaussian ({ll_gaussian}) and Student-t ({ll_student}) gave same likelihood!"
        )

    def test_student_t_heavy_tails(self, model_student_t, data):
        """
        Test that Student-t noise works and produces finite likelihoods.
        """
        if isinstance(model_student_t.likelihood, ContinuousTouchTask):
            worse_mc_task = ContinuousTouchTask(
                ContinuousTouchTaskConfig(
                    num_samples=500, rule=ContinuousTouchRule(nonlinear_func=rule_func)
                )
            )
        else:
            worse_mc_task = OddityTask(
                OddityTaskConfig(num_samples=500, bandwidth=1e-2)
            )

        params = model_student_t.init_params(jr.PRNGKey(0))

        # Override MC fidelity for this test via task config.
        worse_model_student_t = WPPM(
            input_dim=model_student_t.input_dim,
            prior=model_student_t.prior,
            likelihood=worse_mc_task,
            noise=model_student_t.noise,
        )

        ll = worse_model_student_t.likelihood.loglik(
            params=params,
            data=data,
            model=worse_model_student_t,
            key=jr.PRNGKey(0),
        )

        assert jnp.isfinite(ll)
        assert ll <= 0.0
