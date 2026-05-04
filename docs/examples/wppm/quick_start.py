"""
Quick-Start WPPM Example: Fitting a Covariance Ellipse at a Single Point
------------------------------------------------------------------------

This is a minimal, fast version of the full WPPM example. It demonstrates
the complete workflow — simulate data, fit a model, visualize results —
at a **single reference point** with reduced MC samples and fewer optimizer
steps so it runs in seconds on CPU.

For the full spatially-varying field example (25 reference points, GPU), see
:doc:`full_wppm_fit_example`.

"""

from __future__ import annotations

import os
import sys

# --8<-- [start:jax_device_setup]
# Must be set BEFORE importing JAX, as JAX locks in its backend on first import.
# Unset any forced CPU override so JAX can auto-detect GPU/TPU if available.
os.environ.pop("JAX_PLATFORM_NAME", None)
# --8<-- [end:jax_device_setup]

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# Ensure local src is importable when running directly
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src"))
)

# --8<-- [start:imports]
from psyphy.data import TrialData  # batched trial container
from psyphy.inference import MAPOptimizer  # fitter
from psyphy.model import (
    WPPM,
    GaussianNoise,
    OddityTask,
    OddityTaskConfig,
    Prior,
    WPPMCovarianceField,  # fast Σ(x) evaluation
)

# --8<-- [end:imports]

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")

print("DEVICE USED:", jax.devices()[0])

# ---------------------------------------------------------------------------
# Ellipse-plotting helpers (shared with full example)
# ---------------------------------------------------------------------------

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
    """Convert batched covariances into polyline segments for LineCollection."""
    covs = covs + plot_jitter * jnp.eye(covs.shape[-1])
    eigvals = jnp.linalg.eigvalsh(covs)
    valid = jnp.all(eigvals > 0, axis=-1)

    def _cov_to_points(cov: jnp.ndarray, center: jnp.ndarray) -> jnp.ndarray:
        L = jnp.linalg.cholesky(cov)
        pts = scale * (L @ unit_circle)  # (2, n_theta)
        return (center[:, None] + pts).T  # (n_theta, 2)

    all_segments = jax.vmap(_cov_to_points)(covs, centers_xy)  # (n, n_theta, 2)
    return all_segments[valid], valid


# ---------------------------------------------------------------------------
# Compute settings  — deliberately small for a fast CPU run
# ---------------------------------------------------------------------------

# --8<-- [start:compute_settings]
MC_SAMPLES = 50  # MC samples per trial in the likelihood (full example: 500)
NUM_TRIALS = 100  # total simulated trials (full example: 4000 × 25)
NUM_STEPS = 200  # optimizer steps (full example: 2000)

learning_rate = 5e-4  # full example: 5e-5. The smaller the lr, the more steps
# are required.
# --8<-- [end:compute_settings]

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------

# input_dim = 2
# basis_degree = 4  # smoothness / complexity of the basis
# extra_dims = 1  # embedding dim for the Wishart process
# decay_rate = 0.4  # how quickly high-frequency basis coefficients are shrunk
# variance_scale = 4e-3  # 1e-9    # prior scale for the covariance matrices
# diag_term = 1e-4  # small diagonal jitter to keep covariances PD
# bandwidth = 1e-2  # logistic-CDF bandwidth in the oddity task
# momentum = 0.9

# ---------------------------------------------------------------------------
# Step 1 — Ground-truth model
# ---------------------------------------------------------------------------

print("[1/5] Setting up ground-truth WPPM and simulating data...")

# --8<-- [start:truth_model]
task = OddityTask(config=OddityTaskConfig(num_samples=int(MC_SAMPLES)))
noise = GaussianNoise(sigma=0.1)

# Set all Wishart process hyperparameters in Prior
truth_prior = Prior()
truth_model = WPPM(
    prior=truth_prior,
    likelihood=task,
    noise=noise,
)

# Sample ground-truth Wishart process weights
truth_params = truth_model.init_params(jax.random.PRNGKey(123))
# --8<-- [end:truth_model]

# ---------------------------------------------------------------------------
# Step 2 — Simulate data at a *single* reference point
# ---------------------------------------------------------------------------


# Single reference point at the centre of the stimulus space.
ref_point = jnp.array([[0.0, 0.0]])  # shape (1, 2) — kept as a batch for generality

seed = 3
key = jr.PRNGKey(seed)

# Repeat the reference point for every trial.
refs = jnp.repeat(ref_point, repeats=NUM_TRIALS, axis=0)  # (NUM_TRIALS, 2)

# Evaluate Σ at the reference point.
truth_field = WPPMCovarianceField(truth_model, truth_params)
Sigmas_ref = truth_field(refs)  # (NUM_TRIALS, 2, 2)

# Sample unit directions and build covariance-scaled probe displacements.
k_dir, k_sim = jr.split(key)
angles = jr.uniform(k_dir, shape=(NUM_TRIALS,), minval=0.0, maxval=2.0 * jnp.pi)
unit_dirs = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)  # (N, 2)

# Constant Mahalanobis radius: probe = ref + MAHAL_RADIUS * chol(Σ_ref) @ unit_dir
MAHAL_RADIUS = 2.8
L = jnp.linalg.cholesky(Sigmas_ref)  # (N, 2, 2)
# location of comparisons = ref+delta
deltas = MAHAL_RADIUS * jnp.einsum("nij,nj->ni", L, unit_dirs)  # (N, 2)
comparisons = jnp.clip(refs + deltas, -1.0, 1.0)

# --8<-- [start:simulate_data]
# Simulate observed responses using the likelihood implied by the task
ys, p_correct = task.simulate(truth_params, refs, comparisons, truth_model, key=k_sim)
# --8<-- [end:simulate_data]

# --8<-- [start:data]
inputs = jnp.stack([refs, comparisons], axis=1)
data = TrialData(inputs=inputs, responses=ys)  # contains 2 JAX arrays
# --8<-- [end:data]


print(
    f"  Simulated {NUM_TRIALS} trials at ref={ref_point[0].tolist()}, "
    f"mean p(correct)={float(p_correct.mean()):.3f}"
)

# ---------------------------------------------------------------------------
# Step 3 — Build the model to fit
# ---------------------------------------------------------------------------

print("[2/5] Building model and optimizer...")

# --8<-- [start:build_model]
prior = Prior()

model = WPPM(
    prior=prior,
    likelihood=task,
    noise=noise,  # we use the same Gaussian noise as for the ground truth
)
# --8<-- [end:build_model]

# --8<-- [start:prior]
# Initialize parameters at a sample from the prior
init_params = model.init_params(
    jax.random.PRNGKey(42)
)  # intitialize with a draw from the prior
prior_field = WPPMCovarianceField(model, init_params)
# Evaluate prior covariance at the reference point
covs_prior = prior_field(ref_point)  # (1, 2, 2)
# --8<-- [end:prior]
print(f"  shape of covs_prior: {covs_prior.shape}")

# ---------------------------------------------------------------------------
# Step 4 — MAP optimization
# ---------------------------------------------------------------------------

print("[3/5] Fitting via MAPOptimizer ...")

# --8<-- [start:fit_map]
inference = MAPOptimizer(
    steps=NUM_STEPS,
    learning_rate=learning_rate,
    track_history=True,
    log_every=1,
)

map_estimate = inference.fit(model, data, init_params=init_params, seed=4)
# Protocol: ParameterPosterior, here point estimate

# optional: for visualization:
map_cov_field = WPPMCovarianceField(model, map_estimate.params)
# OUTPUT: Covariance Matrices (N, 2, 2) for plotting
# --8<-- [end:fit_map]

# ---------------------------------------------------------------------------
# Step 5 — Visualize covariance ellipses (truth / prior / fit)
# ---------------------------------------------------------------------------

print("[4/5] Plotting covariance ellipses ...")

# --8<-- [start:plot_ellipses]
_PLOT_JITTER = 0.0


# --8<-- [start:cov_fields]
# Evaluate any covariance-field object at a single point or a batch of points.
covs_truth = truth_field(ref_point)  # (N, 2, 2)
covs_prior = prior_field(ref_point)  # (N, 2, 2)
covs_map = map_cov_field(ref_point)  # (N, 2, 2)
# here: N=1 for fast computation
# --8<-- [end:cov_fields]

# Scale ellipses so they are visually readable.
gt_scale = float(jnp.sqrt(jnp.mean(jnp.linalg.eigvalsh(covs_truth[0]))))
ellipse_scale = max(0.3, 0.4 * gt_scale / 0.01)  # keep readable on the unit square

fig, ax = plt.subplots(figsize=(6, 6))

labels = ["Ground Truth", "Prior Sample (init)", "Fitted (MAP)"]
colors = ["k", "b", "r"]
fields = [truth_field, prior_field, map_cov_field]
non_pd_counts = []

for field, color, label in zip(fields, colors, labels):
    covs = field(ref_point)
    segments, valid = _ellipse_segments_from_covs(
        ref_point,
        covs,
        scale=ellipse_scale,
        plot_jitter=_PLOT_JITTER,
        unit_circle=_UNIT_CIRCLE,
    )
    non_pd_counts.append(int((~valid).sum()))
    lc = LineCollection(
        jax.device_get(segments),
        colors=color,
        linewidths=2.0,
        alpha=0.8,
    )
    ax.add_collection(lc)
    ax.plot([], [], color=color, alpha=0.8, linewidth=1.5, label=label)

ax.scatter(
    ref_point[:, 0], ref_point[:, 1], c="g", s=40, zorder=5, label="Reference Point"
)
ax.set_xlim(-0.6, 0.6)
ax.set_ylim(-0.6, 0.6)
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("Stimulus dimension 1")
ax.set_ylabel("Stimulus dimension 2")
ax.set_title(
    f"Covariance ellipse at ref={ref_point[0].tolist()}\n"
    f"lr={learning_rate}, steps={NUM_STEPS}, MC-samples={MC_SAMPLES}, trials={NUM_TRIALS}"
)
ax.grid(True, alpha=0.3)
ax.legend(loc="upper right")
plt.tight_layout()

os.makedirs(PLOTS_DIR, exist_ok=True)
fig.savefig(
    os.path.join(PLOTS_DIR, "quick_start_ellipses.png"), dpi=200, bbox_inches="tight"
)
print(f"  Saved → {PLOTS_DIR}/quick_start_ellipses.png")
# --8<-- [end:plot_ellipses]

# ---------------------------------------------------------------------------
# Step 6 — Learning curve
# ---------------------------------------------------------------------------

print("[5/5] Plotting learning curve ...")

# --8<-- [start:plot_learning_curve]
steps_hist, loss_hist = inference.get_history()
# --8<-- [end:plot_learning_curve]

if steps_hist and loss_hist:
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(steps_hist, loss_hist, color="#4444aa")
    ax2.set_xlim(steps_hist[0], steps_hist[-1])
    ax2.set_title(
        f"Learning curve\n"
        f"lr={learning_rate}, steps={NUM_STEPS}, MC-samples={MC_SAMPLES}, trials={NUM_TRIALS}"
    )
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Neg log likelihood")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    fig2.savefig(
        os.path.join(PLOTS_DIR, "quick_start_learning_curve.png"),
        dpi=200,
        bbox_inches="tight",
    )
    print(f"  Saved → {PLOTS_DIR}/quick_start_learning_curve.png")
else:
    print("  No history recorded — set track_history=True in MAPOptimizer to enable.")

print("Done.")
