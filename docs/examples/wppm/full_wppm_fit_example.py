"""
Full WPPM Example: Fitting a Spatially-Varying Covariance Field
---------------------------------------------------------------

This example demonstrates how to use the full Wishart Process
Psychophysical Model (WPPM)to fit a spatially-varying covariance
field to synthetic 2D data. It visualizes the ground-truth
covariance field, the initial prior sample, and the fitted
field as ellipsoid contours.

"""

from __future__ import annotations

import os
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# --8<-- [start:imports]
# (imports above are included via mkdocs-snippets)
# --8<-- [end:imports]

# Ensure local src is importable when running directly
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src"))
)
import jax.random as jr

# --8<-- [start:imports]
from psyphy.data.dataset import TrialData  # (batched trial container)
from psyphy.inference.map_optimizer import MAPOptimizer  # fitter
from psyphy.model.covariance_field import (
    WPPMCovarianceField,  # (fast (\Sigma) evaluation)
)
from psyphy.model.noise import GaussianNoise  # for model
from psyphy.model.prior import Prior
from psyphy.model.task import OddityTask, OddityTaskConfig
from psyphy.model.wppm import WPPM

# --8<-- [end:imports]
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")


# --8<-- [start:jax_device_setup]
# Prefer GPU/TPU if available; otherwise fall back to CPU.
try:
    has_accel = any(
        getattr(d, "platform", "").lower() in ("gpu", "cuda", "tpu")
        for d in jax.devices()
    )
except Exception:
    has_accel = False

if not has_accel:
    # Force CPU backend if no accelerator detected (or JAX not yet initialized).
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
else:
    # Remove any forced setting so JAX can use the accelerator.
    os.environ.pop("JAX_PLATFORM_NAME", None)


# print device used
print("DEVICE USED:", jax.devices()[0])
# Helper: invert criterion to d* for Oddity task
# --8<-- [end:jax_device_setup]


# # Robust ellipse plotting utilities
# def matrix_sqrt(Sigma: jnp.ndarray) -> jnp.ndarray:
#     eigvals, eigvecs = jnp.linalg.eigh(Sigma)
#     sqrt_eigvals = jnp.sqrt(jnp.maximum(eigvals, 0))
#     return eigvecs @ jnp.diag(sqrt_eigvals) @ eigvecs.T


# def plot_ellipse_at_point(
#     ax,
#     center: jnp.ndarray,
#     Sigma: jnp.ndarray,
#     scale: float = 1.0,
#     color: str = "blue",
#     alpha: float = 1.0,
#     linewidth: float = 0.5,
#     label: str | None = None,
# ):
#     sqrt_Sigma = matrix_sqrt(Sigma)
#     ellipse_points = scale * (sqrt_Sigma @ _UNIT_CIRCLE)
#     x_coords = center[0] + ellipse_points[0]
#     y_coords = center[1] + ellipse_points[1]
#     ax.plot(
#         x_coords, y_coords, color=color, alpha=alpha, linewidth=linewidth, label=label
#     )


_THETAS = jnp.linspace(0, 2 * jnp.pi, 100)
_UNIT_CIRCLE = jnp.vstack([jnp.cos(_THETAS), jnp.sin(_THETAS)])


def _ellipse_segments_from_covs(
    centers_xy: jnp.ndarray,
    covs: jnp.ndarray,
    *,
    scale: float,
    plot_jitter: float,
    unit_circle: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Convert batched covariances into polyline segments for LineCollection.

    Speed improvements through:
      - all heavy math (jittering, eigvals, Cholesky, circle transform) is done
        in one batched JAX computation rather than inside a Python loop
      - Matplotlib then receives one big array (n_lines, n_pts, 2) instead of
        creating hundreds of individual Line2D artists via ax.plot

    Returns
    -------
    segments : (n_valid, n_theta, 2)
        Polyline segments for each ellipse.
    valid_mask : (n_centers,)
        Which centers were PD after jitter.
    """
    # Add a small diagonal jitter for plotting stability.
    covs = covs + plot_jitter * jnp.eye(covs.shape[-1])

    # PD check: for a 2x2 SPD matrix, eigvalsh is cheap and reliable.
    eigvals = jnp.linalg.eigvalsh(covs)
    valid = jnp.all(eigvals > 0, axis=-1)

    # Cholesky is faster than eigendecomposition for SPD matrices.
    # we only use it for plotting ellipses (shape/orientation), not inference.
    def _cov_to_points(cov: jnp.ndarray, center: jnp.ndarray) -> jnp.ndarray:
        L = jnp.linalg.cholesky(cov)
        pts = scale * (L @ unit_circle)  # (2, n_theta)
        return (center[:, None] + pts).T  # (n_theta, 2)

    # Compute for all, then filter to valid in one shot.
    all_segments = jax.vmap(_cov_to_points)(covs, centers_xy)  # (n, n_theta, 2)
    return all_segments[valid], valid


# 1) Ground truth: Wishart process field

# Original constants describing simulation:
# NUM_GRID_PTS = jnp.float32(10)      # Number of reference points over stimulus space.
# MC_SAMPLES = jnp.float32(50)        # Number of simulated trials to compute likelihood.
# NUM_TRIALS = jnp.float32(4000)      # Number of trials in simulated dataset.
# # MIN_LR = jnp.float32(-7)
# # MAX_LR = jnp.float32(-3)
#
# # python simulate_2d.py --optimizer sgd --learning_rate 5e-5 --momentum 0.9 --mc_samples 500 --bandwidth 1e-2 --total_steps 1000


NUM_GRID_PTS = 10  # Number of reference points over stimulus space.
MC_SAMPLES = 500  # Number of Monte Carlo samples per trial in the likelihood. # 500
NUM_TRIALS_Per_Ref = 4000  # Total number of trials in the simulated dataset.
# 4000 trials does not work on cpu


print("[1/5] Setting up ground-truth WPPM and simulating data...")
input_dim = 2
basis_degree = 4  # controls smoothness/complexity
extra_dims = 1  # embedding dim for Wishart process
decay_rate = 0.4  # decay rate for basis functions
variance_scale = 4e-3
diag_term = 1e-9
# for ground-truth model
bandwidth = 1e-2
learning_rate = 5e-5
num_steps = 2000

task = OddityTask(
    config=OddityTaskConfig(num_samples=int(MC_SAMPLES), bandwidth=float(bandwidth))
)
noise = GaussianNoise(sigma=0.1)
# ---- "Truth" model (ground truth) ----
# This is the synthetic observer we will:
#   1) sample parameters from (truth_params)
#   2) use to generate responses (data)
# Later we fit a *separate* WPPM instance to that data and compare fields.
#

# --8<-- [start:truth_model]
# Set all Wishart process hyperparameters in Prior
truth_prior = Prior(
    input_dim=input_dim,  # (2D)
    basis_degree=basis_degree,  # (5)
    extra_embedding_dims=extra_dims,  # (1)
    decay_rate=decay_rate,  # for basis functions
    variance_scale=variance_scale,  # how big covariance matrices
    # are before fitting
)
truth_model = WPPM(
    input_dim=input_dim,
    extra_dims=extra_dims,
    prior=truth_prior,
    task=task,  # oddity task ("pick the odd-one out among 3 stimuli")
    noise=noise,  # (Gaussian noise)
    diag_term=diag_term,  # ensure positive-definite covariances
)

# Sample ground-truth Wishart process weights
truth_params = truth_model.init_params(jax.random.PRNGKey(123))
# --8<-- [end:truth_model]


# 2) Simulate synthetic data from the ground-truth field
#
# Here we generate y ~ Bernoulli(p_correct) where p_correct is computed by the
# same MC simulation used by the odditiy task:
# `OddityTask.loglik`
#   - samples internal reps around the stimulus means (ref / comparison)
#   - uses the 3-stimulus oddity decision rule via Mahalanobis distances under
#     an averaged covariance
#   - uses a logistic CDF smoothing with a configurable bandwidth
#


# --8<-- [start:simulate_data]
num_trials_per_ref = NUM_TRIALS_Per_Ref  # (trials per reference point)
n_ref_grid = 5  # NUM_GRID_PTS
ref_grid = jnp.linspace(-1, 1, n_ref_grid)  # [-1,1] space
ref_points = jnp.stack(jnp.meshgrid(ref_grid, ref_grid), axis=-1).reshape(-1, 2)

# --- Stimulus design: covariance-scaled probe displacements ---
#
# Rather than sampling probes at a fixed Euclidean radius, we scale the probe
# displacement by sqrt(Σ_ref). This tends to equalize trial difficulty across
# space (roughly constant Mahalanobis radius).

seed = 3
key = jr.PRNGKey(seed)

# Build a batched reference list by repeating each grid point.
n_ref = ref_points.shape[0]
refs = jnp.repeat(ref_points, repeats=num_trials_per_ref, axis=0)  # (N, 2)
num_trials_total = int(refs.shape[0])

# Evaluate Σ(ref) in batch using the psyphy covariance-field wrapper.
truth_field = WPPMCovarianceField(truth_model, truth_params)
Sigmas_ref = truth_field(refs)  # (N, 2, 2)

# Sample unit directions on the circle.
k_dir, k_pred, k_y = jr.split(key, 3)
angles = jr.uniform(k_dir, shape=(num_trials_total,), minval=0.0, maxval=2.0 * jnp.pi)
unit_dirs = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)  # (N, 2)


# displacement scale/orientation follows the local ellipse
# (constant-ish Mahalanobis radius).
# a constant Mahalanobis radius for generating probes around reference points
# MAHAL_RADIUS * chol(Sigma_ref) @ unit_dir
MAHAL_RADIUS = 2.8

# Compute sqrt(Σ) via Cholesky (Σ should be SPD; diag_term/noise keep it stable).
L = jnp.linalg.cholesky(Sigmas_ref)  # (N, 2, 2)
deltas = MAHAL_RADIUS * jnp.einsum("nij,nj->ni", L, unit_dirs)  # (N, 2)
comparisons = jnp.clip(refs + deltas, -1.0, 1.0)

# Compute p(correct) in batch. We vmap the single-trial predictor.
trial_pred_keys = jr.split(k_pred, num_trials_total)


def _p_correct_one(ref: jnp.ndarray, comp: jnp.ndarray, kk: jnp.ndarray) -> jnp.ndarray:
    # Task MC settings (num_samples/bandwidth) come from OddityTaskConfig.
    # Only the randomness is threaded dynamically.
    return task._simulate_trial_mc(
        params=truth_params,
        ref=ref,
        comparison=comp,
        model=truth_model,
        noise=truth_model.noise,
        num_samples=int(task.config.num_samples),
        bandwidth=float(task.config.bandwidth),
        key=kk,
    )


p_correct = jax.vmap(_p_correct_one)(refs, comparisons, trial_pred_keys)

# Sample observed y ~ Bernoulli(p_correct) in batch.
ys = jr.bernoulli(k_y, p_correct, shape=(num_trials_total,)).astype(jnp.int32)

# Build the canonical batched dataset for compute.
#
# Notes:
# - This is equivalent to storing X with shape (N, 2, d) and y with shape (N,)
#   where X[:,0,:]=refs and X[:,1,:]=comparisons.
# - We keep named fields because it's currently native to OddityTask.
data = TrialData(refs=refs, comparisons=comparisons, responses=ys)

# --8<-- [end:simulate_data]

# 3) Model to fit and MAPOptimizer

print("[2/5] Building model and optimizer...")
# --8<-- [start:build_model]
prior = Prior(
    input_dim=input_dim,  # (2D)
    basis_degree=basis_degree,  # 5
    extra_embedding_dims=extra_dims,  # 1
    decay_rate=decay_rate,  # for basis functions (how quickly they vary)
    variance_scale=variance_scale,  # how big covariance matrices
    # are before fitting
)
model = WPPM(
    input_dim=input_dim,
    prior=prior,
    task=task,
    noise=noise,  # Gaussian
    diag_term=1e-4,  # ensure positive-definite covariances
)
# --8<-- [end:build_model]


# 3.5) Prior
# --8<-- [start:prior]
# Initialize at prior sample
init_params = model.init_params(jax.random.PRNGKey(42))
init_field = WPPMCovarianceField(model, init_params)
# Prior over covariance field
covs_prior = init_field(ref_points)  # (25, 2, 2)
# --8<-- [end:prior]
print(f"shape of covs_prior: {covs_prior.shape}")


# 4) Fit using optimizer

print("[3/5] Fitting via MAPOptimizer ...")

steps = num_steps
lr = learning_rate
momentum = 0.9
# --8<-- [start:fit_map]
map_optimizer = MAPOptimizer(
    steps=steps, learning_rate=lr, momentum=momentum, track_history=True, log_every=1
)

map_posterior = map_optimizer.fit(
    model,
    data,
    init_params=init_params,
)
# --8<-- [end:fit_map]


print(
    "MAPOptimizer settings:",
    f"steps={map_optimizer.steps}",
    f"track_history={map_optimizer.track_history}",
    f"log_every={map_optimizer.log_every}",
)
# NOTE: MAPOptimizer.fit(...) optimizes a *point estimate* (MAP), not a sampled
# posterior. The returned object is a MAPPosterior (delta distribution at theta_MAP).
#
# MC-only likelihood note:
# - During optimization, the log-likelihood is evaluated via MC simulation inside
#   OddityTask.loglik.
# - MC samples are *created on the fly* each time loglik is called; they are not
#   stored in the posterior.


# 5) Visualize ellipsoid field: overlay ground truth, prior, and fit at each grid point
# --8<-- [start:plot_ellipses]
print("[4/5] Plotting covariance field ellipses ...")

# Visualization-only stabilization: if a covariance is numerically slightly
# indefinite, we add a small diagonal jitter for plotting.
_PLOT_JITTER_DEFAULT = 0  # 1e-6
_PLOT_JITTER_FIT = 0  # 1e-5

# Grid for field visualization in [-1,1] space
n_grid = 12
grid_x = jnp.linspace(-1, 1, n_grid)
grid_y = jnp.linspace(-1, 1, n_grid)
centers = jnp.stack(jnp.meshgrid(grid_x, grid_y), axis=-1).reshape(-1, 2)


# JAX-native covariance extraction
# ------------------------------
# Instead of custom helpers like truth_local_cov_batch / fit_local_cov_batch, we
# reuse the library abstraction `WPPMCovarianceField`.
#
# Why it's helpful here:
#   - It *binds* a (model, params) pair into a single callable object, so we don't
#     accidentally use truth_params with the fit model, or vice versa.
#   - It supports both single-point and batched evaluation via `field(x)` where
#     x can be shape (d,) or (..., d).
#   - It uses a vmapped + jitted batch path internally, which is typically faster
#     than a Python loop and stays efficient when you evaluate many points.

truth_field = WPPMCovarianceField(truth_model, truth_params)

# --8<-- [start:cov_fields]
map_field = WPPMCovarianceField(model, map_posterior.params)
# evaluate any covariance field object like this at either a single point
# or a batch of points
covs_map = map_field(ref_points)  # to get the fitted covariances
# --8<-- [end:cov_fields]

# ---- Plot scaling (purely for visualization) ----
# We want ellipses to be visually comparable across the plot, so we choose a
# single global scale factor based on the *ground-truth* covariances.
#
# gt_covs: stacked covariances Σ_truth(x) at each reference point.
# Shape: (n_ref_points, 2, 2)
gt_covs = truth_field(ref_points)
# gt_means: for each covariance matrix, compute the mean eigenvalue.
# For a 2x2 SPD matrix this correlates with the "overall variance" (typical radius).
# Shape: (n_ref_points,)
gt_means = jax.vmap(lambda cov: jnp.mean(jnp.linalg.eigvalsh(cov)))(gt_covs)
# gt_scales: sqrt(mean eigenvalue) produces a quantity in "standard deviation" units.
# This is not used directly for each ellipse; we only use its average to set a
# reasonable fixed plotting scale.
gt_scales = jnp.sqrt(gt_means)

avg_scale = float(jnp.mean(gt_scales))
# ellipse_scale: scalar multiplier applied to sqrt(Σ) when drawing ellipses.
# We keep this constant across all ellipses so only *shape/orientation* varies.
# If you want ellipses sized proportionally to local variance, you could multiply by
# something like jnp.sqrt(jnp.mean(eigvals)) per point instead.
ellipse_scale = 0.4  # 0.3 * avg_scale   # 0.3

fig, ax = plt.subplots(figsize=(7, 7))
non_pd_counts = [0, 0, 0]
labels = ["Ground Truth", "Prior Sample (init)", "Fitted (MAP)", "Reference Points"]
colors = ["k", "b", "r", "g"]
params_list = [truth_params, init_params, map_posterior.params]
field_list = [truth_field, init_field, map_field]
legend_handles = []
for i, (field, _params, color, label) in enumerate(
    zip(field_list, params_list, colors, labels)
):
    plot_jitter = _PLOT_JITTER_DEFAULT
    if label.startswith("Fitted"):
        plot_jitter = _PLOT_JITTER_FIT

    # Batch-evaluate covariances at all reference points in one shot.
    # This is typically much faster than calling field(center) inside a Python loop.
    covs = field(ref_points)  # (n_ref_points, 2, 2)

    # Convert batched covariances into batched polyline segments.
    segments, valid = _ellipse_segments_from_covs(
        ref_points,
        covs,
        scale=ellipse_scale,
        plot_jitter=plot_jitter,
        unit_circle=_UNIT_CIRCLE,
    )
    non_pd_counts[i] = int((~valid).sum())

    # Draw all ellipses for this layer as one artist. This reduces Matplotlib
    # overhead dramatically vs. one ax.plot() call per ellipse.
    segments_np = jax.device_get(segments)
    lc = LineCollection(
        segments_np,
        colors=color,
        linewidths=1.2,
        alpha=0.5,
    )
    ax.add_collection(lc)  # type: ignore[arg-type]

    # Legend handle
    h = ax.plot([], [], color=color, alpha=0.5, linewidth=0.8, label=label)[0]
    legend_handles.append(h)

print(
    f"Ellipse plot skips (Fit/MAP): {non_pd_counts[2]} / {len(ref_points)} "
    f"(plot jitter={_PLOT_JITTER_FIT:g})"
)
# Plot reference points
ref_scatter = ax.scatter(
    ref_points[:, 0], ref_points[:, 1], c="g", s=3, zorder=5, label="Reference Points"
)
legend_handles.append(ref_scatter)
ax.set_title(
    f"Covariance field  \nSkipped non-PD: GT={non_pd_counts[0]}, Prior={non_pd_counts[1]}, Fit={non_pd_counts[2]}"
    f"\n lr={lr}, steps={steps} - MC-samples={MC_SAMPLES}, num-trials-total={num_trials_total}"
)
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("Model space dimension 1")
ax.set_ylabel("Model space dimension 2")
ax.grid(True, alpha=0.3)
ax.legend(handles=legend_handles, loc="upper right")
plt.tight_layout()

os.makedirs(PLOTS_DIR, exist_ok=True)
fig.savefig(
    os.path.join(PLOTS_DIR, "ellipses.png"),
    dpi=200,
    bbox_inches="tight",
)


# --- Prior-only ellipsoid plot (fresh prior draw) ---
# This is a convenience plot to show what the *Prior hyperparameters* imply for
# the covariance field before seeing any data.
print("[4b/5] Plotting prior-only covariance field (fresh prior draw) ...")
prior_only_params = model.init_params(jax.random.PRNGKey(7))
prior_only_field = WPPMCovarianceField(model, prior_only_params)

prior_covs = prior_only_field(ref_points)  # (n_ref_points, 2, 2)
prior_segments, prior_valid = _ellipse_segments_from_covs(
    ref_points,
    prior_covs,
    scale=ellipse_scale,
    plot_jitter=_PLOT_JITTER_DEFAULT,
    unit_circle=_UNIT_CIRCLE,
)

fig_prior, ax_prior = plt.subplots(figsize=(7, 7))
lc_prior = LineCollection(
    jax.device_get(prior_segments),
    colors="b",
    linewidths=1.2,
    alpha=0.5,
)
ax_prior.add_collection(lc_prior)  # type: ignore[arg-type]
ax_prior.scatter(
    ref_points[:, 0],
    ref_points[:, 1],
    c="g",
    s=3,
    zorder=5,
    label="Reference Points",
)

prior_non_pd = int((~prior_valid).sum())
ax_prior.set_aspect("equal", adjustable="box")
ax_prior.set_xlabel("Model space dimension 1")
ax_prior.set_ylabel("Model space dimension 2")
ax_prior.grid(True, alpha=0.3)
ax_prior.legend(loc="upper right")
ax_prior.set_title(
    "Prior draw (covariance field)"
    f"\ninput_dim={input_dim}, basis_degree={basis_degree}, extra_dims={extra_dims}"
    f"\ndecay_rate={decay_rate:g}, variance_scale={variance_scale:g}, diag_term={model.diag_term:g}"
    f"\nSkipped non-PD: {prior_non_pd}"
)
plt.tight_layout()
fig_prior.savefig(
    os.path.join(PLOTS_DIR, "prior_sample.png"),
    dpi=200,
    bbox_inches="tight",
)

# --8<-- [end:plot_ellipses]

# Learning curve
print("[5/5] Plotting learning curve...")
# --8<-- [start:plot_learning_curve]
steps_hist, loss_hist = map_optimizer.get_history()
print(f"num steps: {len(steps_hist)}, num losses: {len(loss_hist)}")
# --8<-- [start:plot_learning_curve]
if steps_hist:
    print(f"history step range: [{steps_hist[0]}, {steps_hist[-1]}]")
if steps_hist and loss_hist:
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.set_xlim(steps_hist[0], steps_hist[-1])
    ax2.plot(steps_hist, loss_hist, color="#4444aa")
    ax2.set_title(
        f"Learning curve \n lr={lr}, steps={steps} - MC-samples={MC_SAMPLES}, num-trials-per-ref={num_trials_per_ref}"
    )
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Neg log likelihood")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    fig2.savefig(
        os.path.join(PLOTS_DIR, "learning_curve.png"),
        dpi=200,
        bbox_inches="tight",
    )
else:
    print("No history recorded — set track_history=True in MAPOptimizer to enable.")

# --8<-- [end:plot_learning_curve]
