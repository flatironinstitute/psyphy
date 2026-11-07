"""
Quick-Start WPPM Example: Fitting a Covariance Ellipse at a Single Point
------------------------------------------------------------------------

This is a minimal, fast version of the full WPPM example. It demonstrates
the complete workflow (simulate data, fit a model, visualize results)
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
from psyphy.viz import plot_ellipses  # covariance-ellipse plotting

# --8<-- [end:imports]

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")

print("DEVICE USED:", jax.devices()[0])

# ---------------------------------------------------------------------------
# Compute settings: small, for a fast CPU run
# ---------------------------------------------------------------------------

# --8<-- [start:compute_settings]
MC_SAMPLES = 50  # MC samples per trial in the likelihood (full example: 500)
NUM_TRIALS = 1000  # total simulated trials (full example: 4000 × 25)
NUM_STEPS = 600  # optimizer steps (full example: 2000)

learning_rate = 1e-4  # full example: 5e-5. The smaller the lr, the more steps
# are required.
#
# These four are not free choices -- they were swept (across both the jax
# version floor CI pins, jax==0.4.28, and a current jax, and across several
# seeds each) for the cheapest setting that recovers the ground truth reliably:
# fitted/true ellipse area ratio within roughly [0.6, 1.6] and no sign of
# collapse or divergence, ~4-13 s CPU depending on jax version.
#
#   * NUM_TRIALS is the binding constraint. All trials sit at a *single*
#     reference point, so the likelihood is weak, and the outcome at low
#     trial counts is not a clean "collapses to zero" failure -- it is
#     chaotic: depending on the exact random draw (which differs across jax
#     versions for the same PRNGKey, not just across seeds) the same trial
#     count can collapse toward zero, overshoot by several times, or blow up
#     numerically. Below ~500 trials this instability shows up often enough
#     to make the fit unreliable; 1000 clears it with margin on every
#     jax/seed combination tested.
#   * NUM_STEPS below ~600 overshoots rather than under-fits.
#   * learning_rate above ~2e-3 diverges to a non-finite loss within 25 steps.
#
# MAPOptimizer defaults to reduction="mean" (a per-trial objective), so this
# learning rate does not need rescaling when you change NUM_TRIALS.
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
# Step 1: Ground-truth model
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
# Step 2: Simulate data at a *single* reference point
# ---------------------------------------------------------------------------


# Single reference point at the centre of the stimulus space.
ref_point = jnp.array([[0.0, 0.0]])  # shape (1, 2), kept as a batch for generality

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

# Stack refs and comparisons to form representation of all stimuli used in the task.
stimuli = jnp.stack([refs, comparisons], axis=1)

# --8<-- [start:simulate_data]
# Simulate observed responses using the likelihood implied by the task
ys, prob_params = task.simulate(truth_params, stimuli, truth_model, key=k_sim)
p_correct = prob_params[0]  # <- for Bernoulli tasks, p_correct is the only prob_param
# --8<-- [end:simulate_data]

# --8<-- [start:data]
data = TrialData(stimuli=stimuli, responses=ys)  # contains 2 JAX arrays
# --8<-- [end:data]


print(
    f"  Simulated {NUM_TRIALS} trials at ref={ref_point[0].tolist()}, "
    f"mean p(correct)={float(p_correct.mean()):.3f}"
)

# ---------------------------------------------------------------------------
# Step 3: Build the model to fit
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
# Step 4: MAP optimization
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
# Step 5: Visualize covariance ellipses (truth / prior / fit)
# ---------------------------------------------------------------------------

print("[4/5] Plotting covariance ellipses ...")

# --8<-- [start:cov_fields]
# Evaluate any covariance-field object at a single point or a batch of points.
covs_truth = truth_field(ref_point)  # (N, 2, 2)
covs_prior = prior_field(ref_point)  # (N, 2, 2)
covs_map = map_cov_field(ref_point)  # (N, 2, 2)
# here: N=1 for fast computation
# --8<-- [end:cov_fields]

fig, ax = plt.subplots(figsize=(6, 6))

# --8<-- [start:plot_ellipses]
# All three fields in one call, at true size. `scale="auto"` is for *grids* of
# reference points -- it needs at least two centers to measure their spacing --
# and there is only one here. At this model's scale the ellipses are readable
# unmagnified anyway, which also means the figure can be read for absolute size.
plot_ellipses(
    ref_point,
    [covs_truth, covs_prior, covs_map],
    ax=ax,
    scale=1.0,
    colors=["k", "C0", "C3"],
    linestyles=["-", "--", "-"],
    linewidths=[2.0, 1.5, 2.0],
    labels=["Ground Truth", "Prior Sample (init)", "Fitted (MAP)"],
    alpha=0.8,
    show_centers=True,
)
# --8<-- [end:plot_ellipses]

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
print(f"  Saved -> {PLOTS_DIR}/quick_start_ellipses.png")

# ---------------------------------------------------------------------------
# Step 6: Learning curve
# ---------------------------------------------------------------------------

print("[5/5] Plotting learning curve ...")

# --8<-- [start:plot_learning_curve]
steps_hist, loss_hist = inference.get_history()
# --8<-- [end:plot_learning_curve]

if steps_hist and loss_hist:
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(steps_hist, loss_hist, color="C0")
    ax2.set_xlim(steps_hist[0], steps_hist[-1])
    ax2.set_title(
        f"Learning curve\n"
        f"lr={learning_rate}, steps={NUM_STEPS}, MC-samples={MC_SAMPLES}, trials={NUM_TRIALS}"
    )
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Neg log posterior (per trial)")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    fig2.savefig(
        os.path.join(PLOTS_DIR, "quick_start_learning_curve.png"),
        dpi=200,
        bbox_inches="tight",
    )
    print(f"  Saved -> {PLOTS_DIR}/quick_start_learning_curve.png")
else:
    print("  No history recorded: set track_history=True in MAPOptimizer to enable.")

print("Done.")
