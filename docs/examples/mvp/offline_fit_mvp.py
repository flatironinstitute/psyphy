"""
MVP offline example: fit WPPM to synthetic 2D data and visualize thresholds
----------------------------------------------------------------------------

This script demonstrates a full synthetic simulation and fitting pipeline:

1. Define a 'ground-truth' WPPM with known parameters θ* (log_diag_true).
2. Generate synthetic behavioral data (reference–probe pairs with binary responses).
3. Fit a new WPPM to the simulated data via MAP estimation.

For each trial, the model computes:
    p_correct = p(y=1 | ref, probe, θ, task)

where y ∈ {0,1} is a binary response (1 = correct, 0 = incorrect).
This probability is obtained from a *closed-form* mapping between
Mahalanobis discriminability d(ref, probe; θ) and expected 3AFC performance.
No Monte Carlo sampling over internal representations is used here.

Thus, the likelihood for each trial is:
    y_i ~ Bernoulli(p_correct_i)
and the overall dataset likelihood is ∏_i p(y_i | ref_i, probe_i, θ, task).

Note:
- The 'truth_model' and 'fitted_model' share the same class (WPPM), 
  but the truth_model uses known parameters to *generate* data,
  while the fitted model infers parameters from that data.
- (Using the same model to simulate and fit (a well-specified setting)
  provides a controlled test of parameter recovery.)
- Note on visibility: With the MVP Oddity mapping used here, p(d=0) = 2/3.
  So if you choose criterion=2/3, the implied discriminability d* is 0 and the
  ellipse collapses to a point. We therefore use criterion=0.75 for a visible
  contour.


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
from psyphy.model.prior import Prior, WishartPrior
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
    """
    Given a target performance criterion (probability correct), invert the OddityTask
    mapping to compute the corresponding discriminability threshold d*.

    The OddityTask maps Mahalanobis distance d to probability correct p via:
        p = 1/3 + (2/3) * 0.5 * (tanh(slope * d) + 1)
    where:
        - 1/3 is chance performance for 3AFC oddity,
        - slope controls the steepness of the psychometric function.

    This function solves for d given a desired p (criterion).

    Parameters
    ----------
    criterion : float
        Target probability correct (e.g., 0.75).
    slope : float, optional
        Slope parameter of the OddityTask psychometric function (default: 1.5).

    Returns
    -------
    float
        Discriminability threshold d* such that p(correct | d*) = criterion.
    """
    chance_level = 1.0 / 3.0  # Chance performance for 3AFC oddity task
    performance_range = 1.0 - chance_level  # Range above chance (2/3)
    # normalize criterion to [0, 1] range above chance
    normalized_perf = (criterion - chance_level) / performance_range
    # Invert the mapping: solve for tanh(slope * d)
    # The mapping is: normalized_perf = 0.5 * (tanh(slope * d) + 1)
    # Rearranged: tanh(slope * d) = 2 * normalized_perf - 1
    tanh_arg = 2.0 * float(normalized_perf) - 1.0
    # Clip to avoid numerical issues with arctanh at +/-1
    tanh_arg = np.clip(tanh_arg, -0.999999, 0.999999)
    # Solve for d: tanh(slope * d) = tanh_arg => d = arctanh(tanh_arg) / slope
    d_star = float(np.arctanh(tanh_arg) / slope)
    return d_star



# def ellipse_contour_from_cov(ref: np.ndarray, cov: np.ndarray, d_threshold: float, n_points: int = 180) -> np.ndarray:
#     """Return points on the isoperformance ellipse: (x-ref)^T Σ^{-1} (x-ref) = d^2."""
#     L = np.linalg.cholesky(cov)  # Σ^{1/2}
#     angles = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
#     unit = np.stack([np.cos(angles), np.sin(angles)], axis=0)  # (2,n)
#     L_unit = L @ unit
#     contour = ref.reshape(2, 1) + d_threshold * L_unit  # (2,n)
#     return contour.T  # (n,2)


def ellipse_contour_from_cov(ref: np.ndarray, cov: np.ndarray, d_threshold: float, n_points: int = 180) -> np.ndarray:
    """
    Return points on the isoperformance ellipse: (x-ref)^T Σ^{-1} (x-ref) = d^2.

    Parameters
    ----------
    ref : np.ndarray
        The center of the ellipse (mean of the distribution), shape (2,).
    cov : np.ndarray
        The 2x2 covariance matrix (Σ), must be symmetric and positive-definite.
    d_threshold : float
        The Mahalanobis distance threshold (radius of the ellipse).
    n_points : int, optional
        Number of points to generate along the ellipse contour (default: 180).

    Returns
    -------
    np.ndarray
        Array of shape (n_points, 2) containing the (x, y) coordinates of the ellipse contour.
    """
    # Compute the Cholesky decomposition of the covariance matrix.
    # This gives a lower-triangular matrix L such that cov = L @ L.T
    # L can be seen as a transformation that maps the unit circle to the ellipse.
    L = np.linalg.cholesky(cov)  # Σ^{1/2}

    # Generate n_points angles evenly spaced between 0 and 2π (not including 2π).
    # These angles represent points around a unit circle.
    angles = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)

    # For each angle, compute the (x, y) coordinates on the unit circle.
    # Shape: (2, n_points), where first row is cosines, second is sines.
    unit = np.stack([np.cos(angles), np.sin(angles)], axis=0)  # (2, n_points)

    # Transform the unit circle points by the Cholesky factor to get the ellipse shape.
    # This scales and rotates the unit circle to match the covariance.
    L_unit = L @ unit  # (2, n_points)

    # Scale the ellipse by the Mahalanobis distance threshold (d_threshold)
    # and shift it to be centered at 'ref'.
    # ref.reshape(2, 1) ensures broadcasting to all points.
    contour = ref.reshape(2, 1) + d_threshold * L_unit  # (2, n_points)

    # Transpose the result to shape (n_points, 2) for easier plotting/use.
    return contour.T  # (n_points, 2)

def simulate_response(prob_correct: float, rng: np.random.Generator) -> int:
    """Draw a binary response with P(correct) = prob_correct."""
    return int(rng.uniform() < prob_correct)


# ---------- 1) Ground truth setup ----------
print("[1/6] Setting up ground-truth model and parameters...")
# --8<-- [start:truth]
# Ground truth setup:
# We instantiate a WPPM with known parameters log_diag_true.
# The covariance at any stimulus location is:
#     Σ(r; θ*) = diag(exp(log_diag_true))
ref_np = np.array([0.0, 0.0], dtype=float)
log_diag_true = jnp.array([np.log(0.9), np.log(0.01)])  # variances 0.9, 0.01
Sigma_true = np.diag(np.exp(np.array(log_diag_true))) # true covariance matrix

# Define the true model: Oddity task + Gaussian noise + WPPM
# from psyphy.model.task import OddityTask
# from psyphy.model.noise import GaussianNoise
# from psyphy.model.wppm import WPPM

task = OddityTask(slope=1.5)         # tanh mapping from d to P(correct)
noise = GaussianNoise(sigma=1.0)     # additive isotropic Gaussian noise
truth_prior = Prior.default(input_dim=2)  # not used to generate, but WPPM requires a prior

# Build the true model
truth_model = WPPM(input_dim=2, prior=truth_prior, task=task, noise=noise)
truth_params = {"log_diag": log_diag_true}
# --8<-- [end:truth]
print("    Ground truth log_diag:", np.array(log_diag_true))
print("    Ground truth covariance diag:", np.exp(np.array(log_diag_true)))

# ---------- 2) Simulate synthetic trials ----------
print("[2/6] Simulating synthetic trials around reference:", ref_np)
# random polar probes around ref; compute p(correct) with
#  true Σ and the Oddity mapping; sample 0/1 responses.
# --8<-- [start:data]
# Simulate synthetic trials:
# Sample probes around the reference in polar coordinates, 
# compute  P(correct|probe, ref) under the true model, 
# then Bernoulli sample the responses.
rng = np.random.default_rng(0)
data = ResponseData() # to store trials
num_trials = 400
max_radius = 0.25 # max radius of probe from reference

for _ in range(num_trials):
    angle = rng.uniform(0.0, 2.0 * np.pi)   # random angle
    radius = rng.uniform(0.0, max_radius)  # random radius
    delta = np.array([np.cos(angle), np.sin(angle)]) * radius # delta from ref
    probe_np = ref_np + delta # probe position

    # For each (ref, probe) pair, we compute the conditional probability:
    # p_correct_truth = p(y=1 | ref, probe, θ*, task)  aka P(correct|probe, ref):
    p_correct_truth = float(truth_model.predict_prob(truth_params, (jnp.array(ref_np), jnp.array(probe_np)))) 
    #  sample responses y ~ Bernoulli(p_correct_truth) to form a simulated dataset.
    y = simulate_response(p_correct_truth, rng) # draw 0/1 response with p_correct_truth
    data.add_trial(ref=ref_np.copy(), probe=probe_np.copy(), resp=y) # store trial (ref, probe, response)
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
    "Simulated Trials \n"
)
ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="gray", fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()

# Save thresholds figure with a filename encoding lr and steps
os.makedirs(PLOTS_DIR, exist_ok=True)
base = "simulated_trials"
thresh_path = os.path.join(PLOTS_DIR, f"{base}.png")
fig.savefig(thresh_path, dpi=200, bbox_inches="tight")
print(f"    Saved thresholds plot to {thresh_path}")
plt.show()

# ---------- 3) Model + prior + init ----------
# Build the model we will fit. The prior controls regularization; increasing
# `scale` weakens the pull of the prior. We initialize from this prior.
print("[3/6] Initializing model from prior...")
# --8<-- [start:model]
# Here we fit a new WPPM to the simulated data:
# For each trial i with response y_i, the model evaluates:
#     log p(y_i | ref_i, probe_i, θ, task)
# where p(y=1|·) is obtained using the same closed-form mapping as in simulation,
# but with parameters θ (to be estimated).

# from psyphy.model.prior import Prior
# from psyphy.model.noise import GaussianNoise
# from psyphy.model.wppm import WPPM
# You can switch to a full-covariance prior; below we center it near the ground-truth diagonal mean
nu = 5.0
V_center = jnp.diag(jnp.array([0.9, 0.01])) / nu  # E[Σ] = ν V ≈ diag(0.9, 0.01)
prior = WishartPrior.default(input_dim=2, nu=nu, V=V_center)
# prior = Prior.default(input_dim=2, scale=0.5) # Gaussian prior on log_diag (back-compat MVP)
noise = GaussianNoise(sigma=1.0)   # additive isotropic Gaussian noise
model = WPPM(input_dim=2, prior=prior, task=task, noise=noise)
init_params = model.init_params(jax.random.PRNGKey(42))
# --8<-- [end:model]
def _sigma_from_params(params) -> np.ndarray:
    # Build Σ from either diagonal log-variances or packed Cholesky params
    if "chol_params" in params:
        vec = np.array(params["chol_params"], dtype=float)  # length 3 for 2D: [a, b, c]
        a, b, c = vec
        # Add tiny floors to diagonals to avoid underflow to exact zero
        L = np.array([[np.exp(a) + 1e-12, 0.0], [b, np.exp(c) + 1e-12]], dtype=float)
        return L @ L.T
    elif "log_diag" in params:
        return np.diag(np.exp(np.array(params["log_diag"], dtype=float)))
    raise KeyError("Expected 'chol_params' or 'log_diag' in params")

Sigma_init = _sigma_from_params(init_params)
print("    Init covariance diag:", np.diag(Sigma_init))

# ---------- 4) MAP fit (with learning curve) ----------
# Optimize the negative log posterior with Optax; record loss every 10 steps
# to display a learning curve. We JIT-compile a single step for speed.
# --8<-- [start:training]
# optimizer hyperparameters:
steps = 100
lr = 2e-2
momentum = 0.9

# Use a safer optimizer when using full-covariance (chol_params)
if "chol_params" in init_params:
    steps = 300
    lr = 1e-3
    opt = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=lr),
    )
else:
    # Use SGD+momentum from Optax for diagonal MVP
    opt = optax.sgd(learning_rate=lr, momentum=momentum)

# Define loss = negative log posterior (minimize it)
def _loss_fn(params):
    return -model.log_posterior_from_data(params, data)

# Start from prior init
params = init_params # PyTree of parameters (dict of arrays)
opt_state = opt.init(params) # PyTree of optimizer state


# Perform a single optimization step:
# Details: each step computes gradients via automatic differentiation (jax.value_and_grad), 
# updates parameters, and returns new ones — all as jax PyTrees, 
# which are lightweight nested structures of arrays that 
# Jax can efficiently traverse and transform.
@jax.jit
def _step(params, opt_state):
    # Ensure params and opt_state are Jax PyTrees for JIT compatibility
    # (e.g., dicts of arrays, not custom Python objects)
    loss, grads = jax.value_and_grad(_loss_fn)(params) # auto-diff
    updates, opt_state = opt.update(grads, opt_state, params) # optimizer update
    params = optax.apply_updates(params, updates) # apply updates
    # Only return jax-compatible types (PyTrees of arrays, scalars)
    return params, opt_state, loss

# Training loop: run steps of SGD+momentum
# and track loss every 10 steps
loss_iters: list[int] = []
loss_values: list[float] = []
for i in range(steps):
    params, opt_state, loss = _step(params, opt_state) # single JIT-compiled step
    # Break if loss becomes non-finite (stability guard)
    if not np.isfinite(float(loss)):
        print("    Warning: non-finite loss encountered; stopping early.")
        break
    if (i % 10 == 0) or (i == steps - 1):
        loss_iters.append(i)
        loss_values.append(float(loss))

fitted_params = params # maximum a posteriori (MAP) estimate after training of θ
# --8<-- [end:training]
print(f"[4/6] Running MAP optimization ({steps} steps, SGD+momentum, lr={lr})...")
Sigma_fit = _sigma_from_params(fitted_params)
print("    Fitted covariance diag:", np.diag(Sigma_fit))

# ---------- 5) Compute contours ----------
# Convert the target criterion (0.75) to a discriminability threshold d*.
# Optionally add \sigma^2 I to Σ in the displayed ellipses to match the likelihood's
# effective covariance (Σ_eff = Σ + \sigma^2 I) when noise is additive in stimulus space.
criterion = 0.75
d_thr = invert_oddity_criterion_to_d(criterion, slope=task.slope)
print(f"[5/6] Computing threshold contours at criterion={criterion:.3f} -> d*={d_thr:.4f}")

# Sigma_init already computed; Sigma_fit already computed

# Use effective covariance if requested: Σ_eff = Σ + \sigma^2 I
sigma = float(noise.sigma)
I2 = np.eye(2, dtype=float)
if INCLUDE_NOISE_IN_THRESHOLDS:
    Sigma_true_plot = Sigma_true + (sigma ** 2) * I2
    Sigma_init_plot = Sigma_init + (sigma ** 2) * I2
    Sigma_fit_plot = Sigma_fit + (sigma ** 2) * I2
    print(f"    Using effective covariance in plots: Σ + σ^2 I (σ={sigma})")
else:
    Sigma_true_plot = Sigma_true
    Sigma_init_plot = Sigma_init
    Sigma_fit_plot = Sigma_fit
    print("    Using model Σ only in plots (no noise added)")

# Add a tiny ridge to ensure numerical PD for plotting
Sigma_true_plot = Sigma_true_plot + 1e-12 * I2
Sigma_init_plot = Sigma_init_plot + 1e-12 * I2
Sigma_fit_plot = Sigma_fit_plot + 1e-12 * I2

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
    f"True Σ diag={np.diag(Sigma_true)}\n"
    f"Init Σ diag={np.diag(Sigma_init)}\n"
    f"Fitted Σ diag={np.diag(Sigma_fit)}"
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
plt.show()
