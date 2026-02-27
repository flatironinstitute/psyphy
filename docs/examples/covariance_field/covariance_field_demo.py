r"""
Covariance Field Demo: Visualizing Wishart Process Perceptual Thresholds
-------------------------------------------------------------------------

This script demonstrates the WPPMCovarianceField abstraction for the full
Wishart Process model with spatially-varying covariance.

What this demo shows:
1. Creating covariance fields from custom parameters (Wishart mode)
2. Evaluating covariance at single points and batches
3. Visualizing threshold ellipses around reference points
4. Visualizing covariance field variations across stimulus space
5. Using the callable interface: field(x) for JAX compatibility

The Wishart Process covariance field:
    Σ(x) = U(x) @ U(x)^T + λI
where:
    U(x) = Σ_ij W_ij * φ_ij(x)  (Chebyshev basis expansion)

This produces spatially-varying, full (non-diagonal) covariance matrices.

"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes

# Allow running script from repo root
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
)

# --8<-- [start:imports]
from psyphy.model import WPPM, GaussianNoise, OddityTask, Prior
from psyphy.model.covariance_field import WPPMCovarianceField

# --8<-- [end:imports]

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


# ============================================================================
# Utilities for ellipse visualization
# ============================================================================

# Pre-compute unit circle for ellipse plotting
_THETAS = jnp.linspace(0, 2 * jnp.pi, 100)
_UNIT_CIRCLE = jnp.vstack([jnp.cos(_THETAS), jnp.sin(_THETAS)])


def matrix_sqrt(Sigma: jnp.ndarray) -> jnp.ndarray:
    """
    Compute matrix square root via eigendecomposition.

    For a positive semi-definite matrix Σ = Q Λ Q^T,
    returns sqrt(Σ) = Q sqrt(Λ) Q^T.

    Parameters
    ----------
    Sigma : jnp.ndarray, shape (2, 2)
        Covariance matrix

    Returns
    -------
    jnp.ndarray, shape (2, 2)
        Matrix square root such that sqrt(Σ) @ sqrt(Σ)^T = Σ
    """
    eigvals, eigvecs = jnp.linalg.eigh(Sigma)
    sqrt_eigvals = jnp.sqrt(jnp.maximum(eigvals, 0))  # Ensure non-negative
    return eigvecs @ jnp.diag(sqrt_eigvals) @ eigvecs.T


def plot_ellipse_at_point(
    ax: Axes,
    center: np.ndarray,
    Sigma: jnp.ndarray,
    scale: float = 1.0,
    color: str = "blue",
    alpha: float = 1.0,
    linewidth: float = 2.5,
    label: str | None = None,
):
    """
    Plot threshold ellipse using matrix square root method.

    We directly transform a unit circle by sqrt(Σ):
    ellipse_points = scale * sqrt(Σ) @ [cos(θ), sin(θ)]

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    center : np.ndarray, shape (2,)
        Center of ellipse (reference point)
    Sigma : jnp.ndarray, shape (2, 2)
        Covariance matrix
    scale : float
        Scaling factor for ellipse size
        - scale=1.0: 1σ ellipse (~68% confidence)
        - scale=2.0: 2σ ellipse (~95% confidence)
    color : str
        Color of ellipse
    alpha : float
        Transparency
    linewidth : float
        Line width for ellipse contour
    label : str, optional
        Label for legend


    """
    # Get matrix square root of covariance
    sqrt_Sigma = matrix_sqrt(Sigma)

    # Transform unit circle: sqrt(Σ) @ unit_circle
    ellipse_points = scale * (sqrt_Sigma @ _UNIT_CIRCLE)

    # Translate to center
    x_coords = center[0] + np.array(ellipse_points[0])
    y_coords = center[1] + np.array(ellipse_points[1])

    # Plot ellipse as a line (contour only)
    ax.plot(
        x_coords, y_coords, color=color, alpha=alpha, linewidth=linewidth, label=label
    )

    # Mark center point
    ax.plot(center[0], center[1], "o", color=color, markersize=8, zorder=10)


def plot_ellipse_field(
    field: WPPMCovarianceField,
    grid_points: jnp.ndarray,
    scale: float = 0.05,
    save_path: str | None = None,
):
    """
    Visualize covariance field as a grid of ellipses.

    Parameters
    ----------
    field : WPPMCovarianceField
        Covariance field to visualize
    grid_points : jnp.ndarray, shape (n, 2)
        Grid points at which to evaluate covariance
    scale : float
        Scaling factor for ellipse size
    save_path : str, optional
        Path to save figure
    """
    # Evaluate covariance at all grid points
    Sigmas_full = field.cov_batch(grid_points)  # (n, d, d) where d might be >2
    # Extract 2x2 blocks (input dimensions only)
    Sigmas = Sigmas_full[:, :2, :2]  # (n, 2, 2)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))

    for point, Sigma in zip(grid_points, Sigmas):
        plot_ellipse_at_point(
            ax,
            center=np.array(point),
            Sigma=Sigma,
            scale=scale,
            color="steelblue",
        )

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect("equal")
    ax.set_xlabel("Model space dimension 1", fontsize=12)
    ax.set_ylabel("Model space dimension 2", fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"    Saved ellipse field to {save_path}")

    return fig


# ============================================================================
# Example 1: Single Point - Threshold Ellipse
# ============================================================================

print("=" * 70)
print("Example 1: Single Reference Point with Threshold Ellipse")
print("=" * 70)

# --8<-- [start:single_point]
# Create Wishart model
model = WPPM(
    input_dim=2,
    prior=Prior(input_dim=2, basis_degree=4, variance_scale=0.03, decay_rate=0.3),
    task=OddityTask(),
    noise=GaussianNoise(sigma=0.1),
    basis_degree=4,  # Wishart mode with 5x5 basis functions
    extra_dims=1,
    diag_term=1e-3,
)

# Sample covariance field from prior
key = jr.PRNGKey(42)
field = WPPMCovarianceField.from_prior(model, key)

# Evaluate at a single reference point
x_ref = jnp.array([0.5, 0.5])
Sigma_full = field(x_ref)  # callable interface!
# Extract the 2x2 block (input dimensions only)
Sigma_ref = Sigma_full[:2, :2]
# --8<-- [end:single_point]
print("\n[1.1] Covariance field created (Wishart mode)")
print(f"\n[1.2] Reference point: {x_ref}")
print(f"      Σ(x_ref) [2x2 block] = \n{Sigma_ref}")


# Visualize ellipse at single point
fig, ax = plt.subplots(figsize=(8, 8))

plot_ellipse_at_point(
    ax,
    center=np.array(x_ref),
    Sigma=Sigma_ref,
    scale=0.4,  #  scale to fit in [0, 1] plot
    color="royalblue",
    label="Threshold Ellipse",
)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("equal")
ax.set_xlabel("Model space dimension 1", fontsize=12)
ax.set_ylabel("Model space dimension 2", fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()

save_path = os.path.join(PLOTS_DIR, "single_point_ellipse.png")
fig.savefig(save_path, dpi=200, bbox_inches="tight")
print(f"[1.4] Saved single point ellipse to {save_path}")
plt.show()


# ============================================================================
# Example 2: Multiple Points - Ellipse Grid
# ============================================================================

print("\n" + "=" * 70)
print("Example 2: Multiple Reference Points - Ellipse Grid")
print("=" * 70)

# --8<-- [start:grid_points]
# Create grid of reference points
n_grid = 5
x_vals = jnp.linspace(0.15, 0.85, n_grid)
y_vals = jnp.linspace(0.15, 0.85, n_grid)
X_grid, Y_grid = jnp.meshgrid(x_vals, y_vals)
grid_points = jnp.stack([X_grid.ravel(), Y_grid.ravel()], axis=1)

# Evaluate covariance at all points using batch method
Sigmas_full = field.cov_batch(grid_points)
# Extract 2x2 blocks
Sigmas_grid = Sigmas_full[:, :2, :2]

# Also works with JAX vmap on the callable!
Sigmas_vmap_full = jax.vmap(field)(grid_points)
Sigmas_vmap = Sigmas_vmap_full[:, :2, :2]
# --8<-- [end:grid_points]
assert jnp.allclose(Sigmas_grid, Sigmas_vmap)
print(f"\n[2.1] Created grid of {len(grid_points)} reference points")
print(f"      Covariances shape: {Sigmas_grid.shape}")
print("      field.cov_batch(X) == jax.vmap(field)(X)")


# Visualize ellipse grid

fig = plot_ellipse_field(
    field,
    grid_points,
    scale=0.15,
    save_path=os.path.join(PLOTS_DIR, "ellipse_grid.png"),
)
plt.show()


# ============================================================================
# Example 3: Custom Parameters
# ============================================================================

print("\n" + "=" * 70)
print("Example 3: Creating Field from Custom Parameters")
print("=" * 70)


# Sample a different covariance field from prior
print("\n[3.1] Creating new covariance field from different prior sample...")
# --8<-- [start:custom]
key_custom = jr.PRNGKey(999)
custom_field = WPPMCovarianceField.from_prior(model, key_custom)
# Evaluate at center point
x_center = jnp.array([0.5, 0.5])
Sigma_custom_full = custom_field(x_center)
Sigma_custom = Sigma_custom_full[:2, :2]
# --8<-- [end:custom]
print("\n[3.2] Σ(0.5, 0.5) from custom field [2x2 block]:")

print(f"      {Sigma_custom}")


# Visualize custom field
print("\n Visualizing custom covariance field...")

fig = plot_ellipse_field(
    custom_field,
    grid_points,
    scale=0.15,
    save_path=os.path.join(PLOTS_DIR, "custom_field_ellipses.png"),
)
plt.show()
