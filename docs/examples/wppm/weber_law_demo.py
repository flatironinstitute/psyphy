"""
Weber's Law Recovery with 1D WPPM
----------------------------------

Demonstrates that a 1D WPPM can recover Weber's law — the finding that the
just-noticeable difference (JND) is a constant fraction of the stimulus level
— purely from binary MC-based oddity-task responses.

The core claim
--------------
Simulate binary oddity responses from a 1D ground-truth observer whose internal
variance satisfies Sigma(s) = (k*s)^2.  Fit a 1D WPPM to those responses.
If the fitted model recovers a variance function whose implied threshold
scales linearly with stimulus magnitude, the Wishart process is flexible enough
to represent Weber's law in 1D.

Why sqrt(Sigma(s)) is the right diagnostic
------------------------------------------
The WPPM oddity task uses a pooled covariance
    Sigma_avg = (2/3)*Sigma_ref + (1/3)*Sigma_comp
and three noisy internal representations to compute pairwise Mahalanobis
distances. This is NOT a simple d' = |delta|/sigma(s_ref). However, in the
1D equal-variance limit (Sigma_ref = Sigma_comp = Sigma), the threshold
displacement at which performance reaches a criterion level is proportional to
sqrt(Sigma(s)). So sqrt(Sigma_hat(s)) is a valid proxy for the implied JND,
with the understanding that it is a diagnostic of the covariance field, not a
direct readout of the decision model's threshold.

Weber's law in the WPPM representation
---------------------------------------
Weber's law: JND(s) = k*s, i.e., Sigma(s) = (k*s)^2.
Since Sigma = U*U^T and Weber requires sqrt(Sigma) = k*s (linear in s),
U(s) = k*s is itself linear — degree 1 in the Chebyshev basis.

Why basis_degree=2 is the principled choice
--------------------------------------------
U(s) = W_0*T_0(s) + W_1*T_1(s) + W_2*T_2(s)
     = W_0 + W_1*s + W_2*(2s^2-1)   [degree-2 Chebyshev expansion of U]
Sigma(s) = U(s)^2                    [degree 4 in s — can represent k^2*s^2]

Weber's law needs only degree-1 U (i.e., basis_degree=1 is the minimum).
We use basis_degree=2 because:
  1. It gives the model one extra degree of freedom to fit deviations from
     Weber's law (e.g., a constant offset — Near's law / high-threshold).
  2. It still forces the prior to strongly regularise: only 3 Chebyshev
     coefficients (W shape (3, 1, 1) = 3 parameters with extra_dims=0).
  3. It is the smallest degree that can represent both Weber (linear U) and
     Near's law (affine U = W_0 + W_1*s), so the test is more principled:
     the model cannot cheat by relying on higher-frequency components.
  basis_degree=4 with extra_dims=1 (the previous setting) had W shape
  (5, 1, 2) = 10 parameters, with the extra capacity driven by the prior
  rather than by data.

Coordinate system and normalisation
-------------------------------------
WPPM does NOT normalise inputs. It enforces x in [-1, 1] and raises ValueError
otherwise. The Chebyshev polynomials are orthogonal on [-1, 1]; using a
sub-interval wastes basis capacity and mis-calibrates the smoothness prior.

We therefore normalise all physical stimuli s in [S_MIN, S_MAX] to
x in [-1, 1] before passing them to the WPPM:
    x = 2*(s - S_MIN)/(S_MAX - S_MIN) - 1

WeberGroundTruth receives the same normalized x. Its covariance encodes
Weber's law in physical units:
    Sigma_gt(s(x)) = (k * s(x))^2  where  s(x) = to_physical(x)

All plotting is done in physical units s for interpretability.

Plots (5 panels + learning curve)
------------------------------------
  1. Trial data scatter  — raw binary responses; x=reference level s,
                           y=displacement delta (both in physical units).
  2. JND recovery        — sqrt(Sigma_hat(s)) vs s: should track k*s.
  3. Weber fraction      — JND(s)/s vs s: should be flat at k.
  4. Fechner's law       — integral of 1/JND: should follow log(s).
  5. Psychometric curves — p(correct) vs delta for three reference levels;
                           sigmoid curves shift right as s grows (Weber).

Toggle SAVE_INDIVIDUAL_PANELS = True to save each panel as its own PNG.
"""

from __future__ import annotations

import os
import sys

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src"))
)

from psyphy.data import TrialData
from psyphy.inference import MAPOptimizer
from psyphy.model import WPPM, GaussianNoise, OddityTask, Prior, WPPMCovarianceField
from psyphy.model.likelihood import OddityTaskConfig

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")

# ---------------------------------------------------------------------------
# Toggle: save each panel as its own PNG in addition to the combined figure
# ---------------------------------------------------------------------------
SAVE_INDIVIDUAL_PANELS = True

print("DEVICE:", jax.devices()[0])

# ---------------------------------------------------------------------------
# Physical stimulus range and coordinate transforms
# ---------------------------------------------------------------------------
# All stimuli live on a positive magnitude axis [S_MIN, S_MAX].
# The WPPM Chebyshev basis requires inputs in [-1, 1].
# We map between physical (s) and normalized (x) coordinates explicitly.

S_MIN: float = 0.2  # minimum physical stimulus level (must be > 0 for Weber)
S_MAX: float = 1.0  # maximum physical stimulus level


def to_norm(s: jnp.ndarray) -> jnp.ndarray:
    """Map physical stimulus s in [S_MIN, S_MAX] -> normalized x in [-1, 1]."""
    return 2.0 * (s - S_MIN) / (S_MAX - S_MIN) - 1.0


def to_phys(x: jnp.ndarray) -> jnp.ndarray:
    """Map normalized x in [-1, 1] -> physical stimulus s in [S_MIN, S_MAX]."""
    return 0.5 * (x + 1.0) * (S_MAX - S_MIN) + S_MIN


# ---------------------------------------------------------------------------
# Ground-truth model: Weber's law  Sigma(s) = (k*s)^2,  in normalized coords
# ---------------------------------------------------------------------------
# WeberGroundTruth receives normalized x in [-1, 1] (same as the WPPM).
# It converts back to physical s internally to compute the covariance.
# This ensures simulation and fitting use the same coordinate system.


class WeberGroundTruth:
    """Ground-truth observer for Weber's law, operating in normalized coordinates.

    Covariance in physical units: Sigma(s) = (k*s)^2
    Receives normalized x in [-1, 1]; converts to physical s internally.

    Implements only the interface needed by OddityTask:
      _compute_sqrt(params, x) -> U of shape (1, embedding_dim),
      such that Sigma = U @ U^T = (k * s(x))^2
    """

    def __init__(self, k: float = 0.2, extra_dims: int = 0):
        self.k = k
        self.input_dim = 1
        self.extra_dims = extra_dims
        self.diag_term = 0.0
        self.noise = GaussianNoise(sigma=0.0)
        self.basis_degree = 2

    def _compute_sqrt(self, _params, x: jnp.ndarray) -> jnp.ndarray:
        # _params unused: WeberGroundTruth has no learned parameters.
        # x is normalized in [-1, 1]; convert to physical s for Weber's law
        s = to_phys(x[0])  # scalar physical stimulus
        sigma = self.k * s  # sqrt(Sigma) = k*s  (Weber)
        embedding_dim = self.input_dim + self.extra_dims
        return sigma * jnp.eye(self.input_dim, embedding_dim)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

K_WEBER = 0.2  # Weber fraction (ground truth)
N_TRIALS = 1000
MC_SAMPLES = 600  # MC samples for simulation and fitting

# basis_degree=2 is the  minimum that works well for this demo:
# - degree 1 U(x) is sufficient to represent Weber (linear U -> quadratic Sigma)
# - With extra_dims=0: W shape (3, 1, 1) = 3 parameters total (minimum possible)
# See module docstring for the full argument.
BASIS_DEGREE = 2

NUM_STEPS = 200  # 1000 gets close for 3 param model (0 additional embedding dims)
LEARNING_RATE = 5e-4

# Reference levels for the psychometric function panel (in physical units)
PSYCH_LEVELS_PHYS = [0.3, 0.6, 0.9]
PSYCH_COLORS = ["#2166ac", "#d6604d", "#4dac26"]

# ---------------------------------------------------------------------------
# Unified axis label constants
# ---------------------------------------------------------------------------
# All panels that share a quantity on an axis use the same label string so
# that readers can immediately identify what is being shown.
#
#   LABEL_S     — x-axis of panels 1–4: the reference stimulus level
#   LABEL_DELTA — x-axis of panel 5 / y-axis of panel 1: comparison offset
#   LABEL_JND   — y-axis of panel 2: WPPM-implied JND proxy
#   LABEL_WF    — y-axis of panel 3: Weber fraction (JND / s)
#   LABEL_PSI   — y-axis of panel 4: Fechner perceived magnitude
#   LABEL_P     — y-axis of panel 5: probability correct

LABEL_S = r"Reference level  $s_\mathrm{ref} \in$  [0.2, 1.0]"
LABEL_DELTA = r"Stimulus difference  $\delta = s_\mathrm{comp} - s_\mathrm{ref}$"
LABEL_JND = r"JND proxy  $\sqrt{\hat{\Sigma}(s_\mathrm{ref})}$"
LABEL_WF = r"Weber fraction  $\sqrt{\hat{\Sigma}(s_\mathrm{ref})}\,/\,s_\mathrm{ref}$"
LABEL_PSI = r"Perceived magnitude  $\hat{\psi}(s_\mathrm{ref})$  [normalized]"
LABEL_P = r"$P(\mathrm{correct})$"

# ---------------------------------------------------------------------------
# Step 1 — Simulate Weber's law data
# ---------------------------------------------------------------------------

print("[1/5] Simulating oddity-task data from WeberGroundTruth ...")

weber_gt = WeberGroundTruth(k=K_WEBER)
task = OddityTask(config=OddityTaskConfig(num_samples=MC_SAMPLES))

key = jr.PRNGKey(0)
k_refs, k_radii, k_sim = jr.split(key, 3)

# Sample reference stimuli uniformly in physical units, then normalise
refs_s = jr.uniform(k_refs, (N_TRIALS,), minval=S_MIN, maxval=S_MAX)  # physical
refs_x = to_norm(refs_s)  # normalized, passed to models
refs = refs_x[:, None]  # (N, 1) for WPPM / OddityTask

# Random Mahalanobis radii so the scatter spans a 2D region in (s, delta) space.
# delta = r * JND(s) = r * k * s  (in physical units)
r_vals = jr.uniform(k_radii, (N_TRIALS,), minval=0.5, maxval=4.0)
delta_s = r_vals * K_WEBER * refs_s  # displacement in physical units

# Comparisons in normalized coordinates (WPPM and OddityTask expect normalized x)
comps_x = to_norm(refs_s + delta_s)
comparisons = comps_x[:, None]  # (N, 1)

# Simulate binary responses via the MC-based oddity decision process.
# WeberGroundTruth receives normalized x; internally converts to physical s.
responses, p_correct_sim = task.simulate(
    params=None, refs=refs, comparisons=comparisons, model=weber_gt, key=k_sim
)
data = TrialData(refs=refs, comparisons=comparisons, responses=responses)

print(
    f"  {N_TRIALS} trials simulated, "
    f"mean p(correct) = {float(p_correct_sim.mean()):.3f}"
)

# ---------------------------------------------------------------------------
# Step 2 — Build and fit the 1D WPPM
# ---------------------------------------------------------------------------

print("[2/5] Fitting 1D WPPM via MAPOptimizer ...")

prior = Prior(input_dim=1, basis_degree=BASIS_DEGREE, extra_embedding_dims=0)
model = WPPM(
    input_dim=1,
    extra_dims=0,  # works well for embedding_dim =input_dim
    prior=prior,
    likelihood=task,
    noise=GaussianNoise(sigma=0.0),
)

init_params = model.init_params(jr.PRNGKey(1))

optimizer = MAPOptimizer(
    steps=NUM_STEPS,
    learning_rate=LEARNING_RATE,
    track_history=True,
    log_every=50,
)

map_posterior = optimizer.fit(model, data, init_params=init_params, seed=2)
print("  Fitting done.")

# ---------------------------------------------------------------------------
# Step 3 — Derived quantities (computed in physical units for interpretability)
# ---------------------------------------------------------------------------

print("[3/5] Computing derived quantities ...")

# Dense grid: physical s -> normalized x for WPPM, physical s for plotting
s_grid = jnp.linspace(S_MIN, S_MAX, 300)  # physical
x_grid = to_norm(s_grid)[:, None]  # normalized, shape (300, 1)

fitted_cov_fn = WPPMCovarianceField(model, map_posterior.params)
variances = fitted_cov_fn(x_grid)  # (300, 1, 1) — 1x1 "matrix" per grid point;
# [:, 0, 0] extracts the scalar Sigma(s)
# sqrt(Sigma) from WPPM — proxy for implied JND in physical units
# (valid in the 1D equal-variance limit; see module docstring)
jnd_fitted = jnp.sqrt(variances[:, 0, 0])
jnd_truth = K_WEBER * s_grid  # ground truth: k*s

# Weber fraction: JND(s)/s — should be flat at K_WEBER
weber_fraction_fitted = jnd_fitted / s_grid

# Fechner's law: psi(s) = integral_{S_MIN}^{s} 1/JND(s') ds'
# When JND = k*s, psi = (1/k)*log(s/S_MIN) — the logarithmic sensation scale
ds = float(s_grid[1] - s_grid[0])
psi_fitted = jnp.cumsum(1.0 / jnd_fitted) * ds
psi_truth = jnp.cumsum(1.0 / jnd_truth) * ds
psi_log = jnp.log(s_grid / s_grid[0])


def _norm01(x):
    return (x - x[0]) / (x[-1] - x[0])


psi_fitted = _norm01(psi_fitted)
psi_truth = _norm01(psi_truth)
psi_log = _norm01(psi_log)

# Psychometric functions in physical units
# Use 500 MC samples for smooth curves (evaluation only, not fitting)
k_psych = jr.PRNGKey(42)
n_delta = 80
task_smooth = OddityTask(config=OddityTaskConfig(num_samples=500))
psych_data = {}

for s_ref in PSYCH_LEVELS_PHYS:
    jnd_gt = K_WEBER * s_ref
    delta_sweep = jnp.linspace(0.01 * jnd_gt, 4.0 * jnd_gt, n_delta)  # physical

    # Convert to normalized coordinates for model calls
    refs_psych = to_norm(jnp.full(n_delta, s_ref))[:, None]
    comps_psych = to_norm(jnp.full(n_delta, s_ref) + delta_sweep)[:, None]

    p_fit = jax.vmap(
        lambda r, c: task_smooth.predict(map_posterior.params, r, c, model, key=k_psych)
    )(refs_psych, comps_psych)

    p_gt = jax.vmap(
        lambda r, c: task_smooth.predict(None, r, c, weber_gt, key=k_psych)
    )(refs_psych, comps_psych)

    # Bin actual trial data near this reference level (physical units)
    tol = 0.12
    mask = jnp.abs(refs_s - s_ref) < tol
    d_sel = np.asarray(delta_s[mask])
    r_sel = np.asarray(responses[mask])
    n_bins = 8
    bin_edges = np.linspace(float(delta_sweep[0]), float(delta_sweep[-1]), n_bins + 1)
    bin_centers, bin_pcorrect, bin_counts = [], [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        idx = (d_sel >= lo) & (d_sel < hi)
        if idx.sum() >= 3:
            bin_centers.append(0.5 * (lo + hi))
            bin_pcorrect.append(r_sel[idx].mean())
            bin_counts.append(idx.sum())

    psych_data[s_ref] = {
        "delta_sweep": delta_sweep,  # physical units
        "p_fit": p_fit,
        "p_ground_truth": p_gt,
        "bin_centers": np.array(bin_centers),
        "bin_pcorrect": np.array(bin_pcorrect),
        "bin_counts": np.array(bin_counts),
    }

# ---------------------------------------------------------------------------
# Helper: save a single-panel figure
# ---------------------------------------------------------------------------


def _save_panel(fig_fn, name: str) -> None:
    fig_p, ax_p = plt.subplots(figsize=(5.5, 4.2))
    fig_fn(ax_p)
    fig_p.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, f"weber_{name}.png")
    fig_p.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig_p)
    print(f"  Saved individual panel -> {path}")


# ---------------------------------------------------------------------------
# Panel drawing functions (all axes in physical units)
# ---------------------------------------------------------------------------

rng_jitter = np.random.default_rng(0)


def draw_trial_scatter(ax):
    """Panel 1: raw trial data in physical units.

    x: reference level s (absolute stimulus intensity, physical units).
    y: displacement delta = s_comp - s_ref (how different the comparison is).

    Weber's law predicts the JND boundary is a straight line through the origin.
    Trials above the line tend to be correct; below tend to be incorrect.
    """
    correct = np.asarray(responses) == 1
    s_np = np.asarray(refs_s)  # physical
    d_np = np.asarray(delta_s)  # physical

    jitter_amp = 0.003
    jitter_c = rng_jitter.uniform(-jitter_amp, jitter_amp, correct.sum())
    jitter_i = rng_jitter.uniform(-jitter_amp, jitter_amp, (~correct).sum())

    ax.scatter(
        s_np[~correct],
        d_np[~correct] + jitter_i,
        s=5,
        alpha=0.3,
        color="#d6604d",
        label="Incorrect",
        rasterized=True,
    )
    ax.scatter(
        s_np[correct],
        d_np[correct] + jitter_c,
        s=5,
        alpha=0.3,
        color="#054907",
        label="Correct",
        rasterized=True,
    )
    ax.plot(
        s_grid,
        jnd_truth,
        "k--",
        linewidth=2,
        label=r"JND$_{ground thruth}(s) = k \cdot s$",
    )
    ax.set_xlabel(LABEL_S)
    ax.set_ylabel(LABEL_DELTA)
    ax.set_title("Raw trial data\nTrials above JND line -> mostly correct")
    ax.legend(markerscale=2.5, fontsize=8)
    ax.grid(True, alpha=0.25)


def draw_jnd_recovery(ax):
    """Panel 2: JND curve — sqrt(Sigma_hat(s)) vs s in physical units.

    sqrt(Sigma) is a proxy for the implied JND under the 1D equal-variance
    approximation (see module docstring). The behavioral verification is
    panel 5 (psychometric curves).
    """
    ax.plot(
        s_grid,
        jnd_truth,
        "k--",
        linewidth=2,
        label=r"Weber's law: $k \cdot s$  (truth)",
    )
    ax.plot(
        s_grid,
        jnd_fitted,
        color="#c0392b",
        linewidth=2,
        label=r"WPPM: $\sqrt{\hat{\Sigma}(s)}$",
    )
    ax.set_xlabel(LABEL_S)
    ax.set_ylabel(LABEL_JND)
    ax.set_title(r"JND recovery — $\sqrt{\hat{\Sigma}(s)}$ vs $s$")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)


def draw_weber_fraction(ax):
    """Panel 3: Weber fraction JND(s)/s — flat = Weber's law holds."""
    ax.axhline(
        K_WEBER,
        color="k",
        linestyle="--",
        linewidth=2,
        label=f"Weber fraction = {K_WEBER}  (truth), to be recovered",
    )
    ax.plot(
        s_grid,
        weber_fraction_fitted,
        color="#c0392b",
        linewidth=2,
        label=r"WPPM: $\sqrt{\hat{\Sigma}(s)}\,/\,s$",
    )
    ax.set_ylim(0, 0.5)
    ax.set_xlabel(LABEL_S)
    ax.set_ylabel(LABEL_WF)
    ax.set_title("Weber fraction\n(flat = Weber's law recovered)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)


def draw_fechner(ax):
    """Panel 4: Fechner's law — cumulative integral of 1/JND(s) gives log scale."""
    ax.plot(s_grid, psi_log, "k--", linewidth=2, label=r"$\log(s/s_0)$  (Fechner)")
    ax.plot(
        s_grid,
        psi_truth,
        color="#888888",
        linewidth=1.5,
        linestyle=":",
        label="Integrated truth JND",
    )
    ax.plot(
        s_grid,
        psi_fitted,
        color="#c0392b",
        linewidth=2,
        label=r"WPPM: $\int \frac{1}{\sqrt{\hat\Sigma(s)}}\,ds$",
    )
    ax.set_xlabel(LABEL_S)
    ax.set_ylabel(LABEL_PSI)
    ax.set_title("Fechner's law\n(integral of JND = log scale)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)


def draw_psychometric(ax):
    """Panel 5: psychometric functions — p(correct) vs delta (physical units).

    PRIMARY behavioral test: if WPPM has recovered Weber's law, the sigmoid
    curves for different reference levels s should be shifted right in proportion
    to s (higher s -> larger delta needed for same performance).

    Chance level for a 3-alternative oddity task is 1/3 (not 1/2 as in 2AFC):
    with three stimuli presented, a random observer picks the odd one correctly
    1 out of 3 times.
    """
    for s_ref, color in zip(PSYCH_LEVELS_PHYS, PSYCH_COLORS):
        d = psych_data[s_ref]
        ax.plot(d["delta_sweep"], d["p_fit"], color=color, linewidth=2.2)
        ax.plot(
            d["delta_sweep"],
            d["p_ground_truth"],
            color=color,
            linewidth=1.4,
            linestyle="--",
            alpha=0.6,
        )
        if len(d["bin_centers"]) > 0:
            sizes = 20 + 3 * d["bin_counts"]
            ax.scatter(
                d["bin_centers"],
                d["bin_pcorrect"],
                color=color,
                s=sizes,
                zorder=5,
                edgecolors="white",
                linewidths=0.5,
            )

    ax.axhline(
        1 / 3,
        color="gray",
        linewidth=0.9,
        linestyle=":",
        label="Chance (1/3, 3-alternative oddity)",
    )
    ax.set_xlabel(LABEL_DELTA)
    ax.set_ylabel(LABEL_P)
    ax.set_title(
        "Psychometric functions \n"
        r"Curves shift right as $s$ grows -> Weber's law"
    )
    level_handles = [
        Line2D([0], [0], color=c, linewidth=2.2, label=f"s = {s}")
        for s, c in zip(PSYCH_LEVELS_PHYS, PSYCH_COLORS)
    ]
    style_handles = [
        Line2D([0], [0], color="k", linewidth=2.2, label="WPPM fit"),
        Line2D(
            [0],
            [0],
            color="k",
            linewidth=1.4,
            linestyle="--",
            alpha=0.6,
            label="Ground truth",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="k",
            linestyle="none",
            markersize=6,
            label="Binned trial data",
        ),
    ]
    ax.legend(handles=level_handles + style_handles, fontsize=7, ncol=2)
    ax.grid(True, alpha=0.25)


# ---------------------------------------------------------------------------
# Step 4 — Combined 5-panel figure
# ---------------------------------------------------------------------------

print("[4/5] Plotting combined figure ...")

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

draw_trial_scatter(axes[0])
draw_jnd_recovery(axes[1])
draw_weber_fraction(axes[2])
draw_fechner(axes[3])
draw_psychometric(axes[4])

ax6 = axes[5]
steps_hist, loss_hist = optimizer.get_history()
if steps_hist:
    ax6.plot(steps_hist, loss_hist, color="#4444aa", linewidth=1.5)
    ax6.set_xlabel("Optimisation step")
    ax6.set_ylabel("Neg log posterior")
    ax6.set_title("Learning curve")
    ax6.grid(True, alpha=0.25)

fig.suptitle(
    f"1D WPPM — Weber's Law Recovery  "
    f"(k={K_WEBER}, N={N_TRIALS} trials, basis_degree={BASIS_DEGREE})\n"
    f"Stimuli normalized to [-1,1]; plots in physical units s∈[{S_MIN},{S_MAX}]",
    fontsize=12,
    fontweight="bold",
)
fig.tight_layout()

os.makedirs(PLOTS_DIR, exist_ok=True)
combined_path = os.path.join(PLOTS_DIR, "weber_law_recovery.png")
fig.savefig(combined_path, dpi=200, bbox_inches="tight")
print(f"  Saved combined figure -> {combined_path}")

# ---------------------------------------------------------------------------
# Step 5 — Individual panels
# ---------------------------------------------------------------------------

if SAVE_INDIVIDUAL_PANELS:
    print("[5/5] Saving individual panels ...")
    _save_panel(draw_trial_scatter, "panel1_trial_scatter")
    _save_panel(draw_jnd_recovery, "panel2_jnd_recovery")
    _save_panel(draw_weber_fraction, "panel3_weber_fraction")
    _save_panel(draw_fechner, "panel4_fechner")
    _save_panel(draw_psychometric, "panel5_psychometric")
else:
    print(
        "[5/5] Individual panels skipped (set SAVE_INDIVIDUAL_PANELS=True to enable)."
    )

print("Done.")
