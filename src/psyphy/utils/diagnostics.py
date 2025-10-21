"""
diagnostics.py
--------------

Model diagnostics and parameter analysis utilities.

Provides tools for:
- Parameter posterior summaries (mean, std, quantiles)
- Threshold uncertainty estimation
- Parameter sensitivity analysis
- Model comparison diagnostics

Examples
--------
>>> # Parameter summary
>>> from psyphy.utils.diagnostics import parameter_summary
>>> summary = parameter_summary(param_post, n_samples=1000)
>>> print(
...     f"Noise: {summary['noise_scale']['mean']:.3f} "
...     f"± {summary['noise_scale']['std']:.3f}"
... )

>>> # Threshold uncertainty
>>> from psyphy.utils.diagnostics import estimate_threshold_uncertainty
>>> threshold_locs, mean_loc, std_loc = estimate_threshold_uncertainty(
...     model, X_grid, probes, n_samples=200, key=jr.PRNGKey(0)
... )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import jax.random as jr

if TYPE_CHECKING:
    from psyphy.model import Model
    from psyphy.posterior import ParameterPosterior


def parameter_summary(
    param_posterior: ParameterPosterior,
    n_samples: int = 1000,
    *,
    key: Any | None = None,
    quantiles: tuple[float, ...] = (0.025, 0.25, 0.5, 0.75, 0.975),
) -> dict[str, dict[str, jnp.ndarray]]:
    """
    Compute summary statistics for all model parameters.

    Parameters
    ----------
    param_posterior : ParameterPosterior
        Parameter posterior to summarize
    n_samples : int, default=1000
        Number of Monte Carlo samples
    key : JAX PRNGKey, optional
        Random key for sampling (auto-generated if None)
    quantiles : tuple of floats, default=(0.025, 0.25, 0.5, 0.75, 0.975)
        Quantiles to compute

    Returns
    -------
    summary : dict[str, dict[str, jnp.ndarray]]
        Dictionary with keys for each parameter, values are dicts with:
        - "mean": Mean of posterior samples
        - "std": Standard deviation
        - "quantiles": Dict mapping quantile to value

    Examples
    --------
    >>> param_post = model.posterior(kind="parameter")
    >>> summary = parameter_summary(param_post, n_samples=500)
    >>> print(
    ...     f"Noise: {summary['noise_scale']['mean']:.3f} "
    ...     f"± {summary['noise_scale']['std']:.3f}"
    ... )
    >>> print(
    ...     f"95% CI: [{summary['noise_scale']['quantiles'][0.025]:.3f}, "
    ...     f"{summary['noise_scale']['quantiles'][0.975]:.3f}]"
    ... )
    """
    if key is None:
        import time

        key = jr.PRNGKey(int(time.time() * 1e6) % 2**32)

    # Sample parameters
    samples = param_posterior.sample(n_samples, key=key)

    # Compute statistics for each parameter
    summary = {}
    for param_name, param_samples in samples.items():
        summary[param_name] = {
            "mean": jnp.mean(param_samples, axis=0),
            "std": jnp.std(param_samples, axis=0),
            "quantiles": {
                q: jnp.percentile(param_samples, 100 * q, axis=0) for q in quantiles
            },
        }

    return summary


def print_parameter_summary(
    param_posterior: ParameterPosterior,
    n_samples: int = 1000,
    *,
    key: Any | None = None,
) -> None:
    """
    Print a human-readable parameter summary.

    Examples
    --------
    >>> param_post = model.posterior(kind="parameter")
    >>> print_parameter_summary(param_post)
    Parameter Summary (1000 samples):

    log_diag:
      Mean: [0.12, -0.03]
      Std:  [0.05,  0.02]
    """
    summary = parameter_summary(param_posterior, n_samples, key=key)

    print(f"Parameter Summary ({n_samples} samples):\n")

    for param_name, stats in summary.items():
        print(f"{param_name}:")

        mean = stats["mean"]
        std = stats["std"]
        q025 = stats["quantiles"][0.025]
        q975 = stats["quantiles"][0.975]

        # Handle different shapes
        if mean.ndim == 0:  # Scalar
            print(f"  Mean: {float(mean):.3f} ± {float(std):.3f}")
            print(f"  95% CI: [{float(q025):.3f}, {float(q975):.3f}]")

        elif mean.ndim == 1:  # Vector
            print(f"  Mean: {mean}")
            print(f"  Std:  {std}")

        elif mean.ndim == 2:  # Matrix (e.g., W)
            # For matrices, report Frobenius norm
            mean_norm = jnp.linalg.norm(mean, "fro")
            std_norm = jnp.linalg.norm(std, "fro")
            q025_norm = jnp.linalg.norm(q025, "fro")
            q975_norm = jnp.linalg.norm(q975, "fro")
            print(f"  Mean (Frobenius norm): {mean_norm:.3f} ± {std_norm:.3f}")
            print(f"  95% CI (norm): [{q025_norm:.3f}, {q975_norm:.3f}]")
            print(f"  Shape: {mean.shape}")

        print()


def estimate_threshold_uncertainty(
    model: Model,
    X_grid: jnp.ndarray,
    probes: jnp.ndarray,
    threshold_criterion: float = 0.75,
    n_samples: int = 100,
    *,
    key: Any,
) -> tuple[jnp.ndarray, float, float]:
    """
    Estimate threshold location and uncertainty via parameter sampling.

    For each parameter sample θᵢ ~ p(θ | data), finds where the model
    predicts threshold_criterion accuracy. The distribution of these
    threshold locations gives us uncertainty about the threshold.

    Parameters
    ----------
    model : Model
        Fitted model (must support predict_with_params)
    X_grid : jnp.ndarray, shape (n_grid, input_dim)
        Grid of test points to search over (e.g., line through stimulus space)
    probes : jnp.ndarray, shape (n_grid, input_dim)
        Probe at each grid point
    threshold_criterion : float, default=0.75
        Target accuracy level (e.g., 0.75 for 75% correct threshold)
    n_samples : int, default=100
        Number of parameter samples for Monte Carlo estimation
    key : JAX random key
        Random key for parameter sampling

    Returns
    -------
    threshold_locations : jnp.ndarray, shape (n_samples,)
        Grid index of threshold for each parameter sample
    threshold_mean : float
        Mean threshold location (as grid index)
    threshold_std : float
        Standard deviation of threshold location (quantifies uncertainty)

    Examples
    --------
    >>> # Create a line through stimulus space
    >>> reference = jnp.array([0.5, 0.3])
    >>> direction = jnp.array([0.1, 0.05])
    >>> t = jnp.linspace(-1, 1, 200)
    >>> X_grid = reference + t[:, None] * direction
    >>> probes = X_grid + 0.05  # Small probe offset
    >>>
    >>> # Estimate threshold uncertainty
    >>> indices, mean_idx, std_idx = estimate_threshold_uncertainty(
    ...     model,
    ...     X_grid,
    ...     probes,
    ...     threshold_criterion=0.75,
    ...     n_samples=200,
    ...     key=jr.PRNGKey(0),
    ... )
    >>>
    >>> # Convert to coordinates
    >>> threshold_coords = X_grid[int(mean_idx)]
    >>> print(f"75% threshold at: {threshold_coords}")
    >>> print(f"Uncertainty: ±{std_idx * (t[1] - t[0]):.3f} stimulus units")
    >>>
    >>> # Plot distribution
    >>> import matplotlib.pyplot as plt
    >>> plt.hist(X_grid[indices, 0], bins=30, alpha=0.7)
    >>> plt.axvline(
    ...     threshold_coords[0], color="r", linestyle="--", label="Mean threshold"
    ... )
    >>> plt.xlabel("Threshold location (dimension 1)")
    >>> plt.ylabel("Frequency")
    >>> plt.title(f"{int(threshold_criterion * 100)}% Threshold Distribution")
    >>> plt.legend()

    Notes
    -----
    This function quantifies **threshold uncertainty** - how uncertain we are
    about the threshold location given the observed data.

    This is different from **prediction uncertainty** at a fixed location:
    - pred_post.variance tells you: "uncertainty about p(correct) at X"
    - estimate_threshold_uncertainty tells you: "uncertainty about where the threshold is"

    Use this for:
    - Reporting threshold estimates with confidence intervals
    - Visualizing threshold contour uncertainty
    - Experimental design (test near uncertain thresholds)
    """
    # Get parameter posterior and sample
    param_post = model.posterior(kind="parameter")
    param_samples = param_post.sample(n_samples, key=key)  # type: ignore[union-attr]

    threshold_indices = []

    for i in range(n_samples):
        # Extract i-th parameter sample
        params_i = {k: v[i] for k, v in param_samples.items()}

        # Evaluate model at all grid points with these specific parameters
        predictions_i = model.predict_with_params(X_grid, probes, params_i)

        # Find where this crosses threshold
        diffs = jnp.abs(predictions_i - threshold_criterion)
        threshold_idx = jnp.argmin(diffs)

        threshold_indices.append(int(threshold_idx))

    threshold_indices = jnp.array(threshold_indices)

    return (
        threshold_indices,
        float(jnp.mean(threshold_indices)),
        float(jnp.std(threshold_indices)),
    )


def estimate_threshold_contour_uncertainty(
    model: Model,
    reference: jnp.ndarray,
    n_angles: int = 16,
    max_distance: float = 0.5,
    n_grid_points: int = 100,
    probe_offset: float = 0.05,
    threshold_criterion: float = 0.75,
    n_samples: int = 100,
    *,
    key: Any,
) -> dict[str, Any]:
    """
    Estimate threshold contour and its uncertainty around a reference point.

    Searches radially in multiple directions to find threshold locations
    and their uncertainty.

    Parameters
    ----------
    model : Model
        Fitted model
    reference : jnp.ndarray, shape (input_dim,)
        Reference stimulus (center of contour)
    n_angles : int, default=16
        Number of directions to search
    max_distance : float, default=0.5
        Maximum search distance from reference
    n_grid_points : int, default=100
        Grid resolution per direction
    probe_offset : float, default=0.05
        Probe offset for discrimination
    threshold_criterion : float, default=0.75
        Target accuracy level
    n_samples : int, default=100
        Parameter samples for uncertainty estimation
    key : JAX random key

    Returns
    -------
    results : dict
        Dictionary with keys:
        - "angles": (n_angles,) - angles in radians
        - "threshold_mean": (n_angles, input_dim) - mean threshold coords
        - "threshold_std": (n_angles,) - std of threshold distance
        - "threshold_samples": (n_angles, n_samples) - all sample indices

    Examples
    --------
    >>> # Estimate full contour
    >>> reference = jnp.array([0.5, 0.3])
    >>> results = estimate_threshold_contour_uncertainty(
    ...     model, reference, n_angles=16, n_samples=200, key=jr.PRNGKey(0)
    ... )
    >>>
    >>> # Plot with uncertainty
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(figsize=(8, 8))
    >>> for i, angle in enumerate(results["angles"]):
    ...     mean_coord = results["threshold_mean"][i]
    ...     std_dist = results["threshold_std"][i]
    ...     ax.plot(*mean_coord, "ro", markersize=8)
    ...     # Plot uncertainty as error bar
    ...     direction = jnp.array([jnp.cos(angle), jnp.sin(angle)])
    ...     lower = mean_coord - 2 * std_dist * direction
    ...     upper = mean_coord + 2 * std_dist * direction
    ...     ax.plot([lower[0], upper[0]], [lower[1], upper[1]], "r-", alpha=0.3)
    >>> ax.plot(*reference, "k*", markersize=20)
    >>> ax.set_aspect("equal")
    """
    angles = jnp.linspace(0, 2 * jnp.pi, n_angles, endpoint=False)

    threshold_means = []
    threshold_stds = []
    all_samples = []

    for angle in angles:
        # Direction vector
        direction = jnp.array([jnp.cos(angle), jnp.sin(angle)])

        # Grid along this direction
        t = jnp.linspace(0, max_distance, n_grid_points)
        X_grid = reference + t[:, None] * direction
        probes = X_grid + probe_offset * direction

        # Estimate threshold
        key, subkey = jr.split(key)
        indices, mean_idx, std_idx = estimate_threshold_uncertainty(
            model,
            X_grid,
            probes,
            threshold_criterion=threshold_criterion,
            n_samples=n_samples,
            key=subkey,
        )

        # Store results
        threshold_coord = X_grid[int(mean_idx)]
        threshold_means.append(threshold_coord)
        threshold_stds.append(std_idx * (t[1] - t[0]))  # Convert to distance
        all_samples.append(indices)

    return {
        "angles": angles,
        "threshold_mean": jnp.array(threshold_means),
        "threshold_std": jnp.array(threshold_stds),
        "threshold_samples": jnp.array(all_samples),
    }
