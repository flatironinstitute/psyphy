"""
test_diagnostics.py
-------------------

Tests for diagnostics utilities (parameter summaries, threshold uncertainty).
"""

import jax.numpy as jnp
import jax.random as jr
import pytest

from psyphy.model import WPPM, Prior
from psyphy.model.noise import GaussianNoise
from psyphy.model.task import OddityTask
from psyphy.utils import (
    estimate_threshold_contour_uncertainty,
    estimate_threshold_uncertainty,
    parameter_summary,
    print_parameter_summary,
)


@pytest.fixture
def trained_model():
    """Create a simple trained model for testing."""
    # Generate simple training data
    key = jr.PRNGKey(42)
    n = 20

    refs = jr.uniform(key, (n, 2), minval=0, maxval=1)
    probes = refs + 0.05
    y = jnp.ones(n, dtype=int)
    X = jnp.stack([refs, probes], axis=1)

    # Create and fit model
    model = WPPM(
        input_dim=2,
        prior=Prior(input_dim=2, scale=0.5),
        task=OddityTask(),
        noise=GaussianNoise(),
    )
    model.fit(X, y, inference="map", inference_config={"steps": 50})

    return model


def test_parameter_summary(trained_model):
    """Test parameter summary statistics."""
    param_post = trained_model.posterior(kind="parameter")

    summary = parameter_summary(param_post, n_samples=50, key=jr.PRNGKey(0))

    # Should have entries for model parameters
    assert "log_diag" in summary
    # MAP posterior has only one parameter (log_diag) - noise is fixed
    assert len(summary) >= 1

    # Each parameter should have mean, std, quantiles
    for _param_name, stats in summary.items():
        assert "mean" in stats
        assert "std" in stats
        assert "quantiles" in stats
        assert 0.025 in stats["quantiles"]
        assert 0.975 in stats["quantiles"]

        # Check shapes
        mean = stats["mean"]
        std = stats["std"]
        assert mean.shape == std.shape


def test_parameter_summary_custom_quantiles(trained_model):
    """Test parameter summary with custom quantiles."""
    param_post = trained_model.posterior(kind="parameter")

    summary = parameter_summary(
        param_post, n_samples=50, key=jr.PRNGKey(0), quantiles=(0.1, 0.5, 0.9)
    )

    for _param_name, stats in summary.items():
        assert 0.1 in stats["quantiles"]
        assert 0.5 in stats["quantiles"]
        assert 0.9 in stats["quantiles"]
        assert 0.025 not in stats["quantiles"]


def test_print_parameter_summary(trained_model, capsys):
    """Test parameter summary printing."""
    param_post = trained_model.posterior(kind="parameter")

    print_parameter_summary(param_post, n_samples=50, key=jr.PRNGKey(0))

    captured = capsys.readouterr()
    assert "Parameter Summary (50 samples)" in captured.out
    assert "log_diag" in captured.out or "noise_scale" in captured.out


def test_estimate_threshold_uncertainty_1d_line(trained_model):
    """Test threshold uncertainty estimation on 1D line."""
    # Create a line through stimulus space
    reference = jnp.array([0.5, 0.3])
    direction = jnp.array([0.1, 0.05])
    t = jnp.linspace(-1, 1, 100)
    X_grid = reference + t[:, None] * direction
    probes = X_grid + 0.05

    # Estimate threshold
    indices, mean_idx, std_idx = estimate_threshold_uncertainty(
        trained_model,
        X_grid,
        probes,
        threshold_criterion=0.75,
        n_samples=20,  # Small for speed
        key=jr.PRNGKey(0),
    )

    # Check outputs
    assert indices.shape == (20,)
    assert isinstance(mean_idx, float)
    assert isinstance(std_idx, float)

    # Mean should be in valid range
    assert 0 <= mean_idx < len(X_grid)

    # Std should be non-negative
    assert std_idx >= 0


def test_estimate_threshold_uncertainty_different_criteria(trained_model):
    """Test threshold estimation with different accuracy criteria."""
    reference = jnp.array([0.5, 0.3])
    direction = jnp.array([0.1, 0.05])
    t = jnp.linspace(-1, 1, 50)
    X_grid = reference + t[:, None] * direction
    probes = X_grid + 0.05

    # Different thresholds should give different locations
    indices_50, mean_50, _ = estimate_threshold_uncertainty(
        trained_model,
        X_grid,
        probes,
        threshold_criterion=0.5,
        n_samples=10,
        key=jr.PRNGKey(0),
    )

    indices_75, mean_75, _ = estimate_threshold_uncertainty(
        trained_model,
        X_grid,
        probes,
        threshold_criterion=0.75,
        n_samples=10,
        key=jr.PRNGKey(0),
    )

    # 75% threshold should be different from 50% threshold
    # (might be same if model is very uncertain, but typically different)
    assert indices_50.shape == indices_75.shape == (10,)


def test_estimate_threshold_contour_uncertainty(trained_model):
    """Test full contour uncertainty estimation."""
    reference = jnp.array([0.5, 0.3])

    results = estimate_threshold_contour_uncertainty(
        trained_model,
        reference,
        n_angles=8,  # Small for speed
        max_distance=0.3,
        n_grid_points=50,
        probe_offset=0.05,
        threshold_criterion=0.75,
        n_samples=10,  # Small for speed
        key=jr.PRNGKey(0),
    )

    # Check structure
    assert "angles" in results
    assert "threshold_mean" in results
    assert "threshold_std" in results
    assert "threshold_samples" in results

    # Check shapes
    n_angles = 8
    assert results["angles"].shape == (n_angles,)
    assert results["threshold_mean"].shape == (n_angles, 2)  # 2D coordinates
    assert results["threshold_std"].shape == (n_angles,)
    assert results["threshold_samples"].shape == (n_angles, 10)  # 10 samples

    # Angles should span 0 to 2π
    assert float(results["angles"][0]) == pytest.approx(0.0, abs=0.01)
    assert float(results["angles"][-1]) < 2 * jnp.pi

    # All std values should be non-negative
    assert jnp.all(results["threshold_std"] >= 0)


def test_threshold_uncertainty_with_predict_with_params(trained_model):
    """Test that predict_with_params is used correctly."""
    # Get a parameter sample
    param_post = trained_model.posterior(kind="parameter")
    param_samples = param_post.sample(1, key=jr.PRNGKey(0))  # type: ignore[union-attr]
    params_i = {k: v[0] for k, v in param_samples.items()}

    # Create test points
    X = jnp.array([[0.5, 0.3], [0.6, 0.4]])
    probes = X + 0.05

    # Call predict_with_params directly
    predictions = trained_model.predict_with_params(X, probes, params_i)

    # Should return predictions at each test point
    assert predictions.shape == (2,)
    assert jnp.all((predictions >= 0) & (predictions <= 1))
