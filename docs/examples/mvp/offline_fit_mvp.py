"""
MVP offline example: fit WPPM to synthetic 2D data and visualize thresholds
----------------------------------------------------------------------------

This example:
- Builds a 2D WPPM (MVP) with Oddity task
- Simulates synthetic trials from a known diagonal covariance (ground truth)
- Fits the model offline via MAP
- Plots three threshold contours around a reference: ground-truth, init, fitted

Note on visibility: With the MVP Oddity mapping used here, p(d=0) = 2/3.
So if you choose criterion=2/3, the implied discriminability d* is 0 and the
ellipse collapses to a point. We therefore use criterion=0.75 for a visible
contour.

Run:
  python docs/examples/mvp/offline_fit_mvp.py

Quick tour of what this script does:
1) Define a 2D MVP model (WPPM) with an Oddity task and Gaussian noise.
2) Simulate synthetic responses around a reference from a known (diagonal) covariance.
3) Initialize model parameters from a simple Gaussian prior (on log variances).
4) Fit via MAP with a tiny JAX+Optax loop and record the learning curve (loss vs. step).
5) Compute threshold contours at a chosen criterion (0.75) and optionally include noise
    in the displayed contour via Σ_eff = Σ + \sigma^2 I so plots match the likelihood.
6) Plot trials and 3 contours (truth/init/fitted), and save both plots.
"""

from __future__ import annotations

import os
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src")))
# --8<-- [start:imports]
from psyphy.data.dataset import ResponseData
from psyphy.model.noise import GaussianNoise
from psyphy.model.prior import Prior
from psyphy.model.task import OddityTask
from psyphy.model.wppm import WPPM

# --8<-- [end:imports]

# Allow running the script directly from repo root without installing the package.
# (Alternative: export PYTHONPATH=$PWD/src)

# ---------- Utilities ----------

# Where to save figures
# Will save threshold and learning-curve plots here, with filenames that encode lr/steps.
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
INCLUDE_NOISE_IN_THRESHOLDS = True  # if True, use Σ_eff = Σ + \sigma^2 I for plotting

def _fmt_float_sci(x: float) -> str:
    """Return compact scientific notation string for filenames (e.g., 2e-4)."""
    s = f"{x:.0e}"
    s = s.replace("e+0", "e").replace("e+", "e").replace("e-0", "e-")
    return s

def invert_oddity_criterion_to_d(criterion: float, slope: float = 1.5) -> float:
    """Invert OddityTask mapping to get discriminability d* at target p=criterion.

    Mapping: p = 1/3 + (2/3)*0.5*(tanh(slope*d) + 1)
    Note: p(d=0) = 2/3 (chance for Oddity). If you pick criterion close to 2/3,
    the ellipse collapses; here we choose 0.75 for a visible contour.
    """
    chance = 1.0 / 3.0
    perf_range = 1.0 - chance
    g = (criterion - chance) / perf_range
    val = np.clip(2.0 * float(g) - 1.0, -0.999999, 0.999999)
    return float(np.arctanh(val) / slope)


def ellipse_contour_from_cov(ref: np.ndarray, cov: np.ndarray, d_threshold: float, n_points: int = 180) -> np.ndarray:
    """Return points on the isoperformance ellipse: (x-ref)^T Σ^{-1} (x-ref) = d^2."""
    L = np.linalg.cholesky(cov)  # Σ^{1/2}
    angles = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    unit = np.stack([np.cos(angles), np.sin(angles)], axis=0)  # (2,n)
    contour = ref.reshape(2, 1) + d_threshold * (L @ unit)
    return contour.T  # (n,2)


def simulate_response(prob_correct: float, rng: np.random.Generator) -> int:
    """Draw a binary response with P(correct) = prob_correct."""
    return int(rng.uniform() < prob_correct)


# ---------- 1) Ground truth setup ----------
# Define the "true" covariance we will use to generate synthetic data.
# We parameterize the model in log-variance space (log_diag) for stability.
print("[1/6] Setting up ground-truth model and parameters...")
# --8<-- [start:truth]
ref_np = np.array([0.0, 0.0], dtype=float)
log_diag_true = jnp.array([np.log(0.9), np.log(0.01)])  # variances 0.04, 0.01
Sigma_true = np.diag(np.exp(np.array(log_diag_true)))

task = OddityTask(slope=1.5)         # tanh mapping from d to P(correct)
noise = GaussianNoise(sigma=1.0)     # additive isotropic Gaussian noise
truth_prior = Prior.default(input_dim=2)  # not used to generate, but WPPM requires a prior
truth_model = WPPM(input_dim=2, prior=truth_prior, task=task, noise=noise)
truth_params = {"log_diag": log_diag_true}
# --8<-- [end:truth]
print("    Ground truth log_diag:", np.array(log_diag_true))
print("    Ground truth covariance diag:", np.exp(np.array(log_diag_true)))

# ---------- 2) Simulate synthetic trials ----------
# Sample probes around the reference in polar coordinates, compute P(correct)
# under the true model, then Bernoulli sample the responses.
print("[2/6] Simulating synthetic trials around reference:", ref_np)
# random polar probes around ref; compute p(correct) with
#  true Σ and the Oddity mapping; sample 0/1 responses.
# --8<-- [start:data]
rng = np.random.default_rng(0)
data = ResponseData()
num_trials = 400
max_radius = 0.25
for _ in range(num_trials):
    angle = rng.uniform(0.0, 2.0 * np.pi)
    radius = rng.uniform(0.0, max_radius)
    delta = np.array([np.cos(angle), np.sin(angle)]) * radius
    probe_np = ref_np + delta
    p = float(truth_model.predict_prob(truth_params, (jnp.array(ref_np), jnp.array(probe_np))))
    y = simulate_response(p, rng)
    data.add_trial(ref=ref_np.copy(), probe=probe_np.copy(), resp=y)
# --8<-- [end:data]
print(f"    Generated {num_trials} trials (max radius {max_radius}).")


fig, ax = plt.subplots(figsize=(6, 6))
refs_np, probes_np, responses_np = data.to_numpy()
probes_rel = probes_np - refs_np

ax.scatter(probes_rel[responses_np == 1, 0], probes_rel[responses_np == 1, 1], s=12, c="#1b9e77", alpha=0.6, label="Response = 1 (correct)")
ax.scatter(probes_rel[responses_np == 0, 0], probes_rel[responses_np == 0, 1], s=12, c="#d95f02", alpha=0.6, label="Response = 0 (incorrect)")

ax.scatter([0.0], [0.0], c="k", s=30, zorder=5, label="Reference")
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("Delta x (probe relative to ref)")
ax.set_ylabel("Delta y (probe relative to ref)")
ax.set_title(
    f"Simulated Trials \n"
)
ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="gray", fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()

# Save thresholds figure with a filename encoding lr and steps
os.makedirs(PLOTS_DIR, exist_ok=True)
base = f"simulated_trials"
thresh_path = os.path.join(PLOTS_DIR, f"{base}.png")
fig.savefig(thresh_path, dpi=200, bbox_inches="tight")
print(f"    Saved thresholds plot to {thresh_path}")
plt.show()

# ---------- 3) Model + prior + init ----------
# Build the model we will fit. The prior controls regularization; increasing
# `scale` weakens the pull of the prior. We initialize from this prior.
print("[3/6] Initializing model from prior...")
# --8<-- [start:model]
# from psyphy.model.prior import Prior
# from psyphy.model.wppm import WPPM

prior = Prior.default(input_dim=2, scale=0.5)
model = WPPM(input_dim=2, prior=prior, task=task, noise=noise)
init_params = model.init_params(jax.random.PRNGKey(42))
# --8<-- [end:model]
print("    Init log_diag:", np.array(init_params["log_diag"]))
print("    Init covariance diag:", np.exp(np.array(init_params["log_diag"])))

# ---------- 4) MAP fit (with learning curve) ----------
# Optimize the negative log posterior with Optax; record loss every 10 steps
# to display a learning curve. We JIT-compile a single step for speed.
# --8<-- [start:training]
steps = 1000
lr = 2e-2

opt = optax.sgd(learning_rate=lr, momentum=0.9)

# Define loss = negative log posterior (minimize it)
def _loss_fn(params):
    return -model.log_posterior_from_data(params, data)

# Start from prior init
params = init_params # PyTree of parameters (dict of arrays)
opt_state = opt.init(params)


# perform a single optimization step
# with jit-compiling the function for efficient execution on CPU/GPU/TPU
# making the optimization loop much faster
@jax.jit
def _step(params, opt_state):
    # Ensure params and opt_state are JAX PyTrees for JIT compatibility
    # (e.g., dicts of arrays, not custom Python objects)
    loss, grads = jax.value_and_grad(_loss_fn)(params)
    updates, opt_state = opt.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    # Only return JAX-compatible types (PyTrees of arrays, scalars)
    return params, opt_state, loss

# Track loss every 10 steps
loss_iters: list[int] = []
loss_values: list[float] = []
for i in range(steps):
    params, opt_state, loss = _step(params, opt_state)
    if (i % 10 == 0) or (i == steps - 1):
        loss_iters.append(i)
        loss_values.append(float(loss))

fitted_params = params
# --8<-- [end:training]
print(f"[4/6] Running MAP optimization ({steps} steps, SGD+momentum, lr={lr})...")
print("    Fitted (MAP) log_diag:", np.array(fitted_params["log_diag"]))
print("    Fitted covariance diag:", np.exp(np.array(fitted_params["log_diag"])))

# ---------- 5) Compute contours ----------
# Convert the target criterion (0.75) to a discriminability threshold d*.
# Optionally add \sigma^2 I to Σ in the displayed ellipses to match the likelihood's
# effective covariance (Σ_eff = Σ + \sigma^2 I) when noise is additive in stimulus space.
criterion = 0.75
d_thr = invert_oddity_criterion_to_d(criterion, slope=task.slope)
print(f"[5/6] Computing threshold contours at criterion={criterion:.3f} -> d*={d_thr:.4f}")

Sigma_init = np.diag(np.exp(np.array(init_params["log_diag"], dtype=float)))
Sigma_fit = np.diag(np.exp(np.array(fitted_params["log_diag"], dtype=float)))

# Use effective covariance if requested: Σ_eff = Σ + \sigma^2 I
sigma = float(noise.sigma)
I2 = np.eye(2, dtype=float)
if INCLUDE_NOISE_IN_THRESHOLDS:
    Sigma_true_plot = Sigma_true + (sigma ** 2) * I2
    Sigma_init_plot = Sigma_init + (sigma ** 2) * I2
    Sigma_fit_plot = Sigma_fit + (sigma ** 2) * I2
    print(f"    Using effective covariance in plots: Σ + \sigma^2 I (\sigma={sigma})")
else:
    Sigma_true_plot = Sigma_true
    Sigma_init_plot = Sigma_init
    Sigma_fit_plot = Sigma_fit
    print("    Using model Σ only in plots (no noise added)")

contour_true = ellipse_contour_from_cov(ref_np, Sigma_true_plot, d_threshold=d_thr)
contour_init = ellipse_contour_from_cov(ref_np, Sigma_init_plot, d_threshold=d_thr)
contour_fit = ellipse_contour_from_cov(ref_np, Sigma_fit_plot, d_threshold=d_thr)

axes_true = d_thr * np.sqrt(np.diag(Sigma_true_plot))
axes_init = d_thr * np.sqrt(np.diag(Sigma_init_plot))
axes_fit = d_thr * np.sqrt(np.diag(Sigma_fit_plot))
print("    True ellipse semi-axes:", axes_true)
print("    Init ellipse semi-axes:", axes_init)
print("    Fit  ellipse semi-axes:", axes_fit)

# ---------- 6) Plot ----------
# plot 1: the synthetic trials (relative to the reference) and 3 threshold ellipses
# plot 2: learning curve (neg log posterior vs. step).
print("[6/6] Rendering scatter and contours...")

fig, ax = plt.subplots(figsize=(6, 6))
refs_np, probes_np, responses_np = data.to_numpy()
probes_rel = probes_np - refs_np

ax.scatter(probes_rel[responses_np == 1, 0], probes_rel[responses_np == 1, 1], s=12, c="#1b9e77", alpha=0.6, label="Response = 1 (correct)")
ax.scatter(probes_rel[responses_np == 0, 0], probes_rel[responses_np == 0, 1], s=12, c="#d95f02", alpha=0.6, label="Response = 0 (incorrect)")

ax.plot(contour_true[:, 0] - ref_np[0], contour_true[:, 1] - ref_np[1], color="#4daf4a", lw=2.0, label="Ground-truth threshold")
ax.plot(contour_init[:, 0] - ref_np[0], contour_init[:, 1] - ref_np[1], color="#7f7f7f", lw=1.5, ls="--", label="Init threshold")
ax.plot(contour_fit[:, 0] - ref_np[0], contour_fit[:, 1] - ref_np[1], color="#377eb8", lw=2.0, label="Fitted threshold (MAP)")

ax.scatter([0.0], [0.0], c="k", s=30, zorder=5, label="Reference")
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("Delta x (probe relative to ref)")
ax.set_ylabel("Delta y (probe relative to ref)")
ax.set_title(
    f"MVP WPPM Fit, criterion={criterion:.3f} (d*={d_thr:.3f})\n"
    f"True log_diag={np.array(log_diag_true)}\n"
    f"Init log_diag={np.array(init_params['log_diag'])}\n"
    f"Fitted log_diag={np.array(fitted_params['log_diag'])}"
)
ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="gray", fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()

# Save thresholds figure with a filename encoding lr and steps
os.makedirs(PLOTS_DIR, exist_ok=True)
lr_tag = _fmt_float_sci(lr)
base = f"offline_fit_mvp_lr{lr_tag}_steps{steps}"
thresh_path = os.path.join(PLOTS_DIR, f"{base}_thresholds.png")
fig.savefig(thresh_path, dpi=200, bbox_inches="tight")
print(f"    Saved thresholds plot to {thresh_path}")
plt.show()

# plot 2: Learning curve figure
fig2, ax2 = plt.subplots(figsize=(6, 4))
ax2.plot(loss_iters, loss_values, color="#4444aa")
ax2.set_title(f"Learning curve (neg log posterior) — lr={lr}, steps={steps}")
ax2.set_xlabel("Step")
ax2.set_ylabel("Loss")
ax2.grid(True, alpha=0.3)
plt.tight_layout()

# Save learning-curve figure
lc_path = os.path.join(PLOTS_DIR, f"{base}_learning_curve.png")
fig2.savefig(lc_path, dpi=200, bbox_inches="tight")
print(f"    Saved learning curve to {lc_path}")

plt.show()
