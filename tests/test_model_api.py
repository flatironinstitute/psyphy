"""
test_model_api.py
-----------------

Tests for the Model façade API (Issue #1):
- Model.fit() with hybrid inference configuration
- Model.posterior() for predictive and parameter posteriors
- Model.condition_on_observations() for online learning
- OnlineConfig strategies (full, sliding_window, reservoir)
"""

import jax.numpy as jnp
import jax.random as jr
import pytest

from psyphy.data import ResponseData
from psyphy.inference import MAPOptimizer
from psyphy.model import WPPM, Prior
from psyphy.model.likelihood import ContinuousTouchTask, OddityTask
from psyphy.model.noise import GaussianNoise
from psyphy.posterior import ParameterPosterior

# Generate Data for each condition:
n = 20
key = jr.PRNGKey(42)
key, subkey = jr.split(key)
refs = jr.normal(subkey, (n, 2))
key, subkey = jr.split(key)
comparisons = refs + jr.normal(subkey, (n, 2)) * 0.3
y = jnp.ones(n, dtype=int)  # All correct

# Create ResponseData object
oddity_data = ResponseData()
for i in range(n):
    oddity_data.add_trial((refs[i], comparisons[i]), int(y[i]))

touch_data = ResponseData()
for i in range(n):
    touch_data.add_trial((refs[i]), refs[i] * 0.3)  # treat these as stim and resp

# Generate regular test data:
refs_test = jnp.array([[0.0, 0.0], [1.0, 1.0]])
comparisons = jnp.array([[0.5, 0.0], [1.5, 1.0]])
oddity_X_test = jnp.stack([refs_test, comparisons], axis=1)
touch_X_test = jnp.expand_dims(refs_test, axis=1)
# Generate longer test data for shape testing:
refs_test_shape = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
comparisons_shape = jnp.array([[0.5, 0.0], [1.5, 1.0], [2.5, 2.0]])
oddshape_X_test = jnp.stack([refs_test_shape, comparisons_shape], axis=1)
ctsshape_X_test = jnp.expand_dims(refs_test_shape, axis=1)
# Note the correct shapes for mean & variance of each posterior respectively
oddity_shapes = ((3,), (3,))
touch_shapes = ((3, 2), (3, 2, 2))


@pytest.mark.parametrize(
    "task, data", [(OddityTask(), oddity_data), (ContinuousTouchTask(), touch_data)]
)
class TestModelFit:
    """Test Model.fit() with different inference configurations."""

    @pytest.fixture
    def model(self, task):
        """Create a WPPM model."""
        return WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=3),
            likelihood=task,
            noise=GaussianNoise(),
        )

    @pytest.fixture
    def data_arrays(self, data):
        """Synthetic data arrays in ResponseData format."""
        return data

    def test_optimizer_fit(self, model, data_arrays):
        """Optimizer.fit() returns a posterior."""
        optimizer = MAPOptimizer(steps=10)
        posterior = optimizer.fit(model, data_arrays)
        assert posterior is not None
        assert isinstance(posterior, ParameterPosterior)

    def test_fit_with_different_steps(self, model, data_arrays):
        """Optimizer is configurable."""
        optimizer = MAPOptimizer(steps=50)
        posterior = optimizer.fit(model, data_arrays)
        assert posterior.params is not None


@pytest.mark.parametrize(
    "task, data, X_test, X_test_shape, moment_shapes",
    [
        (OddityTask(), oddity_data, oddity_X_test, oddshape_X_test, oddity_shapes),
        (
            ContinuousTouchTask(),
            touch_data,
            touch_X_test,
            ctsshape_X_test,
            touch_shapes,
        ),
    ],
)
class TestModelPosterior:
    """Test Posterior predictions."""

    @pytest.fixture
    def posterior(self, task, data):
        """Create a fitted posterior."""
        model = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=3),
            likelihood=task,
            noise=GaussianNoise(),
        )

        optimizer = MAPOptimizer(steps=20)
        return optimizer.fit(model, data)

    @pytest.fixture
    def test_data(self, X_test, X_test_shape, moment_shapes):
        return {
            "X_test": X_test,
            "X_test_shape": X_test_shape,
            "moment_shapes": moment_shapes,
        }

    def test_posterior_predictive_default(self, posterior, test_data):
        """posterior.predict() returns PredictivePosterior by default."""
        X_test = test_data["X_test"]

        # In the new API (see WPPMPredictivePosterior), prediction is creating
        # a new predictive object around the parameter posterior.
        from psyphy.posterior.predictive_posterior import WPPMPredictivePosterior

        # It's not a method on posterior, but a wrapper class construction
        pred = WPPMPredictivePosterior(posterior, X_test)

        # Check duck typing
        assert hasattr(pred, "mean")
        assert hasattr(pred, "variance")

    def test_posterior_predictive_has_correct_shape(self, posterior, test_data):
        """posterior.params access and predictive posterior has correct shape."""
        params = posterior.params
        assert isinstance(params, dict)

        from psyphy.posterior.predictive_posterior import WPPMPredictivePosterior

        X_test_shape = test_data["X_test_shape"]
        moment_shapes = test_data["moment_shapes"]

        pred = WPPMPredictivePosterior(posterior, X_test_shape)

        # Triggers lazy computation
        mean = pred.mean
        var = pred.variance

        assert mean.shape == moment_shapes[0]
        assert var.shape == moment_shapes[1]


@pytest.mark.parametrize(
    "task, data", [(OddityTask(), oddity_data), (ContinuousTouchTask(), touch_data)]
)
class TestConditionOnObservations:
    """Test online learning via creating new posteriors."""

    @pytest.fixture
    def model(self, task):
        """Create a model."""
        return WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=3),
            likelihood=task,
            noise=GaussianNoise(),
        )

    @pytest.fixture
    def initial_data(self, data):
        """Generate initial training data."""
        return data

    def test_initial_fit(self, model, initial_data):
        optimizer = MAPOptimizer(steps=10)
        posterior = optimizer.fit(model, initial_data)
        assert posterior is not None


class TestIntegrationWorkflow:
    """Integration tests for complete workflows."""

    @pytest.mark.parametrize(
        "task, data, X_test, moment_shapes",
        [
            (OddityTask(), oddity_data, oddity_X_test, ((2,), (2,))),
            (ContinuousTouchTask(), touch_data, touch_X_test, ((2, 2), (2, 2, 2))),
        ],
    )
    def test_full_new_api_workflow(self, task, data, X_test, moment_shapes):
        """Test complete workflow with new API."""
        # 1. Create model
        model = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=3),
            likelihood=task,
            noise=GaussianNoise(),
        )

        # 3. Fit model (new API)
        optimizer = MAPOptimizer(steps=50)
        posterior = optimizer.fit(model, data)

        # 4. Get predictive posterior

        from psyphy.posterior.predictive_posterior import WPPMPredictivePosterior

        pred = WPPMPredictivePosterior(posterior, X_test)

        # 5. Make predictions
        mean = pred.mean
        var = pred.variance

        assert mean.shape == moment_shapes[0]
        assert var.shape == moment_shapes[1]
        if isinstance(task, OddityTask):
            assert jnp.all((mean >= 0) & (mean <= 1))

    def test_oddity_manual_online_loop(self):
        """Test online learning manually.
        Currently only implemented for OddityTask."""
        # 1. Create model
        model = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=3),
            likelihood=OddityTask(),
            noise=GaussianNoise(),
        )
        optimizer = MAPOptimizer(steps=10)
        data = ResponseData()

        key = jr.PRNGKey(456)

        # Online loop
        for i in range(5):
            key, subkey = jr.split(key)
            ref = jr.normal(subkey, (2,))
            key, subkey = jr.split(key)
            comp = ref + jr.normal(subkey, (2,)) * 0.1

            data.add_trial((ref, comp), 1)

            # Re-fit every step (naive online learning)
            posterior = optimizer.fit(model, data)
            assert posterior is not None
            assert len(data.responses) == i + 1
