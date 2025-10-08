"""
MVP offline example using MAPOptimizer (hides JAX loop)
------------------------------------------------------

This example mirrors offline_fit_mvp.py but delegates optimization to
MAPOptimizer so the JAX training loop stays under the hood. It also plots
loss history recorded by the optimizer.
"""
from __future__ import annotations

import os
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# Ensure local src is importable when running directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src")))

from psyphy.data.dataset import ResponseData
from psyphy.inference.map_optimizer import MAPOptimizer
from psyphy.model.noise import GaussianNoise
from psyphy.model.prior import Prior
from psyphy.model.task import OddityTask
from psyphy.model.wppm import WPPM

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")


def invert_oddity_criterion_to_d(criterion: float, slope: float = 1.5) -> float:
    chance = 1.0 / 3.0
    perf_range = 1.0 - chance
    g = (criterion - chance) / perf_range
    val = np.clip(2.0 * float(g) - 1.0, -0.999999, 0.999999)
    return float(np.arctanh(val) / slope)


def ellipse_contour_from_cov(ref: np.ndarray, cov: np.ndarray, d_threshold: float, n_points: int = 180) -> np.ndarray:
    L = np.linalg.cholesky(cov)
    angles = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    unit = np.stack([np.cos(angles), np.sin(angles)], axis=0)
    contour = ref.reshape(2, 1) + d_threshold * (L @ unit)
    return contour.T


def simulate_response(prob_correct: float, rng: np.random.Generator) -> int:
    return int(rng.uniform() < prob_correct)


# 1) Ground truth and synthetic data
print("[1/4] Setting up truth and simulating data...")
ref_np = np.array([0.0, 0.0], dtype=float)
log_diag_true = jnp.array([np.log(0.9), np.log(0.01)])
Sigma_true = np.diag(np.exp(np.array(log_diag_true)))

task = OddityTask(slope=1.5)
noise = GaussianNoise(sigma=1.0)
truth_prior = Prior.default(input_dim=2)
truth_model = WPPM(input_dim=2, prior=truth_prior, task=task, noise=noise)
truth_params = {"log_diag": log_diag_true}

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

# 2) Model to fit and MAPOptimizer (with history tracking)
print("[2/4] Building model and optimizer...")
prior = Prior.default(input_dim=2, scale=0.5) # Gaussian prior on log_diag
noise = GaussianNoise(sigma=1.0)     # additive isotropic Gaussian noise
model = WPPM(input_dim=2, prior=prior, task=task, noise=noise)


# 3) Fit using optimizer (using psyphy.inference.MAPOptimizer)
print("[3/4] Fitting via MAPOptimizer...")
# --8<-- [start:training]
# optimizer hyperparameters:
steps = 1000
lr = 2e-2
momentum = 0.9

# from psyphy.inference.map_optimizer import MAPOptimizer
optimizer = MAPOptimizer(steps=steps, learning_rate=lr, momentum=momentum, track_history=True, log_every=10)

# [Optional] Initialize parameters explicitly (otherwise falls back to prior sample with seed=0)
init_params = model.init_params(jax.random.PRNGKey(42))

# Fit model to data, returns a Posterior wrapper around the fitted params and model
# To see the training loop that is used, check the source of MAPOptimizer.fit() or the 
# code snippet below.
posterior = optimizer.fit(model, data, init_params=init_params)
# --8<-- [end:training]

# 4) Plot thresholds and learning curve using recorded history
print("[4/4] Plotting...")
criterion = 0.75
d_thr = invert_oddity_criterion_to_d(criterion, slope=task.slope)


# --8<-- [start:params_extraction]

# Collect fitted parameters, posterior holds the fitted params.
fitted = posterior.params if hasattr(posterior, "params") else None

# --8<-- [end:params_extraction]
if fitted is None:
    raise RuntimeError("Posterior missing 'params' attribute; expected by this example.")

# reconstruct fitted covariance
Sigma_fit = np.diag(np.exp(np.array(fitted["log_diag"], dtype=float)))

# adding the noise variance to the diagonal for plotting
Sigma_fit_plot = Sigma_fit + (float(noise.sigma) ** 2) * np.eye(2) # fitted covariance with noise
Sigma_init = np.diag(np.exp(np.array(init_params["log_diag"], dtype=float)))
Sigma_init_plot = Sigma_init + (float(noise.sigma) ** 2) * np.eye(2) # initial covariance with noise
Sigma_true_plot = Sigma_true + float(noise.sigma) ** 2 * np.eye(2) # simulated true covariance



# --8<-- [start:plot_contours]
# Compute contours for truth and fit (fit params are inside posterior.model/params)
# creating threshold contours with convenience function defined above
contour_true = ellipse_contour_from_cov(ref_np, Sigma_true_plot, d_threshold=d_thr)
contour_init = ellipse_contour_from_cov(ref_np, Sigma_init_plot, d_threshold=d_thr)
contour_fit = ellipse_contour_from_cov(ref_np, Sigma_fit_plot, d_threshold=d_thr)

# Scatter + contours
refs_np, probes_np, responses_np = data.to_numpy()
probes_rel = probes_np - refs_np # position of probe relative to reference
# ... plotting
# --8<-- [end:plot_contours]
fig, ax = plt.subplots(figsize=(6, 6))
# simulated data points
ax.scatter(probes_rel[responses_np == 1, 0], probes_rel[responses_np == 1, 1], s=12, c="#1b9e77", alpha=0.6, label="Response = 1")
ax.scatter(probes_rel[responses_np == 0, 0], probes_rel[responses_np == 0, 1], s=12, c="#d95f02", alpha=0.6, label="Response = 0")
# threshold contours
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
    f"Fitted log_diag={np.array(fitted['log_diag'])}"
)
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)
plt.tight_layout()

os.makedirs(PLOTS_DIR, exist_ok=True)
fig.savefig(os.path.join(PLOTS_DIR, "offline_fit_mvp_mapopt_thresholds.png"), dpi=200, bbox_inches="tight")

# --8<-- [start:loss_history]
# Learning curve from optimizer history
steps_hist, loss_hist = optimizer.get_history()
# plotting
# --8<-- [end:loss_history]
if steps_hist and loss_hist:
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(steps_hist, loss_hist, color="#4444aa")
    ax2.set_title(f"Learning curve (neg log posterior) — lr={lr}, steps={steps}")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Loss")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()

    fig2.savefig(os.path.join(PLOTS_DIR, "offline_fit_mvp_mapopt_learning_curve.png"), dpi=200, bbox_inches="tight")
else:
    print("No history recorded — set track_history=True in MAPOptimizer to enable.")
