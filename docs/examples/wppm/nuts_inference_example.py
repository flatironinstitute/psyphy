"""
NUTS Posterior Inference for WPPM
----------------------------------

Demonstrates full posterior inference using the No-U-Turn Sampler (NUTS)
for the Wishart Process Psychophysical Model (WPPM).

Unlike MAP estimation (which returns a single point estimate θ_MAP), NUTS
draws samples from the full posterior p(θ | data), enabling uncertainty
quantification over the covariance field Σ(x).

Steps
-----
1.  Setup: synthetic ground-truth model + simulated data
2.  MAP warm-start: fast initialization near the posterior mode
3.  NUTS sampling: draw posterior samples with BlackJAX
4.  ArviZ diagnostics: R-hat, ESS, trace plots
5.  Posterior predictive: ensemble of covariance field ellipses
6.  MAP vs NUTS mean comparison
7.  All four layers overlaid in one axes

Requirements
------------
    pip install 'psyphy[sampling,diagnostics,viz]'
    # sampling    -> blackjax>=1.4,<2   (requires Python >= 3.11)
    # diagnostics -> arviz>=1.0         (requires Python >= 3.12)
    # viz         -> matplotlib>=3.5
    #
    # There is no `nuts` extra. On a GPU add `cuda` (jax[cuda12]), and do NOT
    # `module load cuda cudnn` alongside it -- see CONTRIBUTING.md.
    #
    # Add `progress` for the progress bars; without it `show_progress=True` is
    # silently a no-op and a long fit prints nothing for hours.

Usage
-----
    python nuts_inference_example.py                    # quick: laptop smoke test
    python nuts_inference_example.py --mode balanced    # a real fit, ~GPU-hour
    python nuts_inference_example.py --mode full        # paper-scale, batch job

`quick` is the default so the docs build and a laptop run stay fast. It is NOT a
fit to interpret -- 450 trials and 50 MC samples leave the frozen-key posterior
visibly biased, and 150 draws cannot support R-hat or ESS. Use `balanced` or
`full` for anything you intend to read. Only `quick` writes the canonical plot
filenames the documentation embeds; other modes suffix them with the mode.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time

os.environ.pop("JAX_PLATFORM_NAME", None)  # allow GPU/TPU if available

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src"))
)

# --8<-- [start:imports]
from psyphy.data import TrialData
from psyphy.inference import MAPOptimizer, NUTSSampler
from psyphy.model import (
    WPPM,
    GaussianNoise,
    OddityTask,
    OddityTaskConfig,
    Prior,
    WPPMCovarianceField,
)
from psyphy.viz import plot_ellipses

# --8<-- [end:imports]

_parser = argparse.ArgumentParser(description="NUTS posterior inference for WPPM")
_parser.add_argument(
    "--mode",
    choices=("quick", "balanced", "full"),
    default="quick",
    help="compute/fidelity preset; see MODES below (default: quick)",
)
_parser.add_argument("--seed", type=int, default=42, help="master PRNG seed")
_args = _parser.parse_args()
MODE = _args.mode
SEED = _args.seed

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def plot_path(name: str) -> str:
    """Canonical filename for `quick`, mode-suffixed otherwise.

    The documentation embeds plots/nuts_*.png, and those committed figures come
    from the default (quick) run. A heavier run must not silently replace them,
    so every other mode writes plots/nuts_*_<mode>.png -- the same convention
    hong2025_reproduction.py uses for its own full-mode figures.
    """
    if MODE == "quick":
        return os.path.join(PLOTS_DIR, name)
    stem, ext = os.path.splitext(name)
    return os.path.join(PLOTS_DIR, f"{stem}_{MODE}{ext}")


print("DEVICE:", jax.devices()[0])
print("MODE  :", MODE)

# ---------------------------------------------------------------------------
# Model. IDENTICAL IN EVERY MODE, deliberately: --mode changes how carefully
# the posterior is explored, never what is being fit, so two modes are
# comparable. Change these to change the science, not the budget.
# ---------------------------------------------------------------------------
INPUT_DIM = 2
BASIS_DEGREE = 2
EXTRA_DIMS = 1
DECAY_RATE = 0.4
VARIANCE_SCALE = 4e-3
DIAG_TERM = 1e-4
BANDWIDTH = 1e-2

# --8<-- [start:compute_settings]
# Compute/fidelity presets. Same MODES convention as hong2025_reproduction.py.
#
# Why each knob is where it is in `full`:
#   mc_samples 2000  the value hong2025.PAPER_HYPERPARAMS uses for this model
#                    class. It is the knob that matters most here: the MC
#                    likelihood is only deterministic because NUTSSampler
#                    freezes one PRNG key, which makes the sampled posterior
#                    the one implied by that noise draw. The bias shrinks as
#                    mc_samples grows -- 50 is a smoke test, 200 the documented
#                    minimum, 2000 the paper's.
#   trials 5000      25 reference points x 200 trials. 450 trials (quick) is
#                    thin for this many coefficients, and under-determined
#                    posteriors are wide for reasons that have nothing to do
#                    with the sampler.
#   4 x 1000 draws   the standard NUTS configuration, and what the shapes in
#                    the design note assume. R-hat needs >= 2 chains (4 is
#                    convention) and ESS > 100 per parameter needs the draws.
#   warmup 1000      dual averaging tunes the step size AND the mass matrix;
#                    for WPPM the mass matrix matters a lot, because
#                    coefficients at different Chebyshev degrees have genuinely
#                    different prior scales. 100 warmup steps cannot do it.
MODES = {
    "quick": {
        "mc_samples": 50,
        "trials_per_ref": 50,
        "n_ref_grid": 3,
        "map_steps": 300,
        "warmup": 100,
        "draws": 150,
        "chains": 2,
        "ensemble": 30,
    },
    "balanced": {
        "mc_samples": 500,
        "trials_per_ref": 100,
        "n_ref_grid": 5,
        "map_steps": 800,
        "warmup": 500,
        "draws": 500,
        "chains": 4,
        "ensemble": 100,
    },
    "full": {
        "mc_samples": 2000,
        "trials_per_ref": 200,
        "n_ref_grid": 5,
        "map_steps": 1500,
        "warmup": 1000,
        "draws": 1000,
        "chains": 4,
        "ensemble": 200,
    },
}
# --8<-- [end:compute_settings]

_cfg = MODES[MODE]
MC_SAMPLES = _cfg["mc_samples"]
NUM_TRIALS_PER_REF = _cfg["trials_per_ref"]
N_REF_GRID = _cfg["n_ref_grid"]

# MAP warm-start settings. Step count scales with mode; the learning rate and
# momentum are tuning, not budget, so they are fixed.
MAP_STEPS = _cfg["map_steps"]
MAP_LR = 5e-5
MAP_MOMENTUM = 0.9

# NUTS settings
# --8<-- [start:nuts_settings]
NUM_WARMUP = _cfg["warmup"]
NUM_SAMPLES = _cfg["draws"]
NUM_CHAINS = _cfg["chains"]
# Tree-depth cap: a draw costs at most 2**MAX_NUM_DOUBLINGS - 1 gradient
# evaluations (255 here; BlackJAX's default 10 would allow 1023). Check
# sampler_stats["n_steps"] after the run -- if most draws hit 255, raise it.
MAX_NUM_DOUBLINGS = 8
# --8<-- [end:nuts_settings]
N_ENSEMBLE = _cfg["ensemble"]

print(
    f"        mc={MC_SAMPLES}, {N_REF_GRID**2}x{NUM_TRIALS_PER_REF} trials, "
    f"MAP {MAP_STEPS} steps, NUTS {NUM_CHAINS}x({NUM_WARMUP}+{NUM_SAMPLES})"
)

# ---------------------------------------------------------------------------
# 1. Ground-truth model + synthetic data
# ---------------------------------------------------------------------------
print("[1/7] Setting up ground-truth WPPM and simulating data...")

task = OddityTask(config=OddityTaskConfig(num_samples=MC_SAMPLES, bandwidth=BANDWIDTH))
noise = GaussianNoise(sigma=0.1)

truth_prior = Prior(
    input_dim=INPUT_DIM,
    basis_degree=BASIS_DEGREE,
    extra_embedding_dims=EXTRA_DIMS,
    decay_rate=DECAY_RATE,
    variance_scale=VARIANCE_SCALE,
)
truth_model = WPPM(
    input_dim=INPUT_DIM,
    extra_dims=EXTRA_DIMS,
    prior=truth_prior,
    likelihood=task,
    noise=noise,
    diag_term=DIAG_TERM,
)
truth_params = truth_model.init_params(jr.PRNGKey(123))
truth_field = WPPMCovarianceField(truth_model, truth_params)

# Build reference grid
ref_grid = jnp.linspace(-0.8, 0.8, N_REF_GRID)
ref_points = jnp.stack(jnp.meshgrid(ref_grid, ref_grid), axis=-1).reshape(-1, 2)
n_ref = ref_points.shape[0]
refs = jnp.repeat(ref_points, NUM_TRIALS_PER_REF, axis=0)

# Covariance-scaled probe displacements (constant Mahalanobis radius)
key = jr.PRNGKey(3)
k_dir, k_sim = jr.split(key)
n_total = refs.shape[0]
Sigmas_ref = truth_field(refs)
angles = jr.uniform(k_dir, (n_total,), minval=0.0, maxval=2 * jnp.pi)
unit_dirs = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)
L = jnp.linalg.cholesky(Sigmas_ref)
deltas = 2.8 * jnp.einsum("nij,nj->ni", L, unit_dirs)
comparisons = jnp.clip(refs + deltas, -1.0, 1.0)

# Stimuli are packed (N, K, d) with K=2 -- slot 0 reference, slot 1 comparison.
# The oddity task presents three stimuli but stores two: the reference is shown
# twice, and that duplication lives in OddityTask, not in the data.
stimuli = jnp.stack([refs, comparisons], axis=1)
ys, _ = task.simulate(truth_params, stimuli, truth_model, key=k_sim)

# --8<-- [start:data]
data = TrialData(stimuli=stimuli, responses=ys)
# --8<-- [end:data]

print(
    f"  {n_total} trials simulated ({N_REF_GRID**2} ref points × {NUM_TRIALS_PER_REF} trials)"
)

# ---------------------------------------------------------------------------
# 2. Fit model (shared between MAP warm-start and NUTS)
# ---------------------------------------------------------------------------
print("[2/7] Building model...")

# --8<-- [start:build_model]
prior = Prior(
    input_dim=INPUT_DIM,
    basis_degree=BASIS_DEGREE,
    extra_embedding_dims=EXTRA_DIMS,
    decay_rate=DECAY_RATE,
    variance_scale=VARIANCE_SCALE,
)
model = WPPM(
    input_dim=INPUT_DIM,
    prior=prior,
    likelihood=task,
    noise=noise,
    diag_term=DIAG_TERM,
)
# --8<-- [end:build_model]

# ---------------------------------------------------------------------------
# 3. MAP warm-start
# Initializing NUTS near the posterior mode greatly reduces warmup steps.
# ---------------------------------------------------------------------------
print("[3/7] MAP warm-start (fast initialization near posterior mode)...")

# --8<-- [start:map_warmstart]
map_optimizer = MAPOptimizer(
    steps=MAP_STEPS,
    learning_rate=MAP_LR,
    momentum=MAP_MOMENTUM,
    track_history=True,
    show_progress=True,
)
map_posterior = map_optimizer.fit(model, data, seed=SEED)
# --8<-- [end:map_warmstart]

print(f"  MAP done. Final loss: {map_optimizer.loss_history[-1]:.4f}")

# --8<-- [start:cost_probe]
# Project the sampling cost before committing to it. NUTS costs
# O(2^tree_depth) gradient evaluations per draw and the depth is data
# dependent, so the honest output is a bracket, not a number. Cheap insurance
# against submitting a batch job that cannot finish inside its time limit --
# and the only way to know whether --mode full is hours or days on this
# hardware, since the MC likelihood dominates and GPUs do not scale linearly
# with mc_samples.
_probe = jax.jit(
    jax.value_and_grad(
        lambda q: model.log_posterior_from_data(q, data, key=jr.PRNGKey(0))
    )
)
_, _grad = _probe(map_posterior.params)
jax.block_until_ready(_grad["W"])  # compile first, outside the timing
_t0 = time.perf_counter()
for _ in range(5):
    _, _grad = _probe(map_posterior.params)
jax.block_until_ready(_grad["W"])
_per_grad = (time.perf_counter() - _t0) / 5
_total_draws = NUM_CHAINS * (NUM_WARMUP + NUM_SAMPLES)
print(
    f"  cost probe: one value_and_grad = {_per_grad * 1e3:.1f} ms"
    f"  (mc={MC_SAMPLES}, {n_total} trials)"
)
# Chains are vmapped (one batched call), so on a GPU the wall clock is closer
# to the per-chain figure below than to the all-chains total; on a CPU it is
# closer to the total. The upper bracket is the tree-depth cap: no draw can
# cost more than 2**MAX_NUM_DOUBLINGS - 1 gradient evaluations.
print(f"  {_total_draws} draws total ({NUM_CHAINS} chains vmapped)")
for _g_per_draw in (16, 64, 2**MAX_NUM_DOUBLINGS - 1):
    _hours = _per_grad * _total_draws * _g_per_draw / 3600.0
    print(f"    at {_g_per_draw:2d} grad evals/draw -> {_hours:7.2f} h")
# The probe times *execution* of an already-compiled gradient. It therefore
# EXCLUDES XLA compilation of the warmup scan, which is a one-off cost but can
# take minutes and, on a short run like --mode quick, dominates everything the
# projection above reports. If the run seems stuck right after "[4/7]", it is
# almost certainly still compiling -- not hung.
print("    (excludes one-off XLA compilation of the warmup scan: can be minutes)")
# --8<-- [end:cost_probe]

# ---------------------------------------------------------------------------
# 4. NUTS sampling
# ---------------------------------------------------------------------------
print("[4/7] Running NUTS posterior sampling (BlackJAX)...")
print(f"  {NUM_CHAINS} chains × {NUM_WARMUP} warmup + {NUM_SAMPLES} draws")
print("  Note: warmup compiles once for all chains — may take ~30s+ before it runs.")

# --8<-- [start:nuts_fit]
sampler = NUTSSampler(
    num_warmup=NUM_WARMUP,
    num_samples=NUM_SAMPLES,
    num_chains=NUM_CHAINS,
    logdensity_key_seed=0,  # fixed MC key → deterministic Hamiltonian
    target_acceptance_rate=0.8,
    max_num_doublings=MAX_NUM_DOUBLINGS,  # cap: <= 255 grad evals per draw
    # Sample z = W / prior_std so warmup starts from a metric matched to the
    # prior scales instead of the identity (the default; shown for clarity).
    prior_preconditioning=True,
    show_progress=True,
)
nuts_posterior = sampler.fit(
    model,
    data,
    init_params=map_posterior.params,  # warm-start from MAP
    seed=7,
)
# --8<-- [end:nuts_fit]

print(
    f"  Sampling done: {nuts_posterior.n_chains} chains × {nuts_posterior.n_draws} draws"
)
acc = nuts_posterior.sampler_stats["acceptance_rate"]
print(f"  Mean acceptance rate: {float(jnp.mean(acc)):.3f}")
# Fraction of post-warmup draws whose trajectory was cut off by the depth cap.
# A large fraction means max_num_doublings is limiting mixing.
_n_steps = nuts_posterior.sampler_stats["n_steps"]
_at_cap = float(jnp.mean(_n_steps >= 2**MAX_NUM_DOUBLINGS - 1))
print(
    f"  Leapfrog steps/draw: mean {float(jnp.mean(_n_steps)):.0f}, "
    f"{_at_cap:.0%} at the cap ({2**MAX_NUM_DOUBLINGS - 1})"
)

# ---------------------------------------------------------------------------
# 5. ArviZ diagnostics
# ---------------------------------------------------------------------------
print("[5/7] Computing ArviZ diagnostics...")

try:
    import arviz as az

    # --8<-- [start:arviz_diagnostics]
    idata = nuts_posterior.to_arviz()
    summary = az.summary(idata, var_names=["W"])
    print(summary[["mean", "sd", "r_hat", "ess_bulk"]].describe())
    r_hat_max = float(summary["r_hat"].max())
    ess_min = float(summary["ess_bulk"].min())
    print(f"  R-hat max: {r_hat_max:.3f}  (target < 1.01)")
    print(f"  ESS min:   {ess_min:.0f}    (target > 100 per chain)")
    # --8<-- [end:arviz_diagnostics]

    # Trace plot. ArviZ draws one panel per *scalar* element of W, and W here has
    # 3*3*2*3 = 54 of them -- past rcParams["plot.max_subplots"] (40), and
    # illegible even if that limit is raised. So plot one readable slice: the
    # 3x3 basis coefficients at the first Model and embedding dimension.
    #
    # NOTE (arviz >= 1.0): plot_trace returns an arviz_plots PlotCollection, not
    # an array of matplotlib axes, so it is saved with pc.savefig(). The 0.x
    # kwargs `compact=` and `figsize=` no longer exist and raise if passed.
    # How to read the resulting figure:
    #   * W has shape (3, 3, 2, 3) = (deg_x, deg_y, U_row, U_col). The first two
    #     axes index the Chebyshev basis -- degree along Model dimension 1 and
    #     along dimension 2 -- and the last two index the entry of U(x), where
    #     Sigma = U U^T + diag_term * I.
    #   * coords={"W_dim_2": 0, "W_dim_3": 0} fixes U[0, 0] and leaves the 3x3
    #     coefficient grid, so `col_wrap=3` lays the panels out AS that grid:
    #     panel title "i, j" is the coefficient of degree i x degree j.
    #   * x axis = draw index WITHIN a chain, post-warmup (warmup is discarded).
    #   * one coloured line per chain.
    #   * black ticks along the bottom are divergences, which appear only because
    #     sampler_stats now carries `diverging` under ArviZ's standard name.
    #   * what you want to see: chains overlapping, flat, and mixing fast -- a
    #     "fuzzy caterpillar". A chain drifting steadily across the panel means
    #     warmup was too short and the sampler is still finding the typical set.
    #
    # NOTE (arviz >= 1.0): plot_trace returns an arviz_plots PlotCollection, not
    # an array of matplotlib axes, so it is saved with pc.savefig(). The 0.x
    # kwargs `compact=` and `figsize=` no longer exist and raise if passed.
    # `visuals={"xlabel": False}` drops the per-panel "Draw" label, which in a
    # multi-row layout collides with the next row's title.
    # --8<-- [start:trace_plot]
    pc = az.plot_trace(
        idata,
        var_names=["W"],
        coords={"W_dim_2": 0, "W_dim_3": 0},
        col_wrap=3,
        visuals={"xlabel": False},
    )
    # Reaching into arviz_plots internals for the caption: cosmetic only
    with contextlib.suppress(Exception):
        pc.viz["figure"].item().suptitle(
            "Posterior traces for W[deg_x, deg_y, 0, 0]. 1 panel per Chebyshev "
            "coefficient\ncolours = chains · x = draw index (post-warmup)",
            fontsize=10,
        )
    pc.savefig(plot_path("nuts_trace_plot.png"))
    # --8<-- [end:trace_plot]
    print("  Saved: nuts_trace_plot.png  (W[:, :, 0, 0] slice)")


except ImportError:
    print(
        "  ArviZ not installed — skipping diagnostics. "
        "Install with: pip install 'psyphy[diagnostics]'"
    )

# ---------------------------------------------------------------------------
# 6. Posterior predictive: ensemble of covariance field ellipses
# ---------------------------------------------------------------------------
print("[6/7] Plotting posterior predictive covariance field...")

ELLIPSE_SCALE = 1.2  # magnification; ellipses are not drawn at true size


def ensemble_covs(param_samples, points):
    """Covariance field at `points` for every posterior draw.

    Returns (n_samples, n_points, 2, 2), which is the "several fields" spelling
    plot_ellipses accepts, so the whole ensemble draws in one call.
    """
    return jnp.stack(
        [
            WPPMCovarianceField(model, {"W": param_samples["W"][i]})(points)
            for i in range(param_samples["W"].shape[0])
        ]
    )


# Draw posterior parameter samples
# --8<-- [start:posterior_samples]
n_ensemble = N_ENSEMBLE
param_samples = nuts_posterior.sample(n_ensemble, key=jr.PRNGKey(0))
# param_samples["W"].shape == (n_ensemble, *W_shape)
# --8<-- [end:posterior_samples]

mean_field = WPPMCovarianceField(model, nuts_posterior.params)
covs_mean = mean_field(ref_points)
covs_truth = truth_field(ref_points)

fig, ax = plt.subplots(figsize=(7, 7))

# --8<-- [start:plot_uncertainty]
# The posterior spread, drawn first so it sits behind the two lines. One call:
# (n_ensemble, n_ref, 2, 2) is the "several fields" shape plot_ellipses takes.
plot_ellipses(
    jax.device_get(ref_points),
    jax.device_get(ensemble_covs(param_samples, ref_points)),
    ax=ax,
    scale=ELLIPSE_SCALE,
    colors="#4444cc",
    linewidths=0.5,
    alpha=0.15,
)

# Posterior mean and ground truth, overlaid as a list of fields so each gets its
# own colour, width and legend entry.
plot_ellipses(
    jax.device_get(ref_points),
    [jax.device_get(covs_mean), jax.device_get(covs_truth)],
    ax=ax,
    scale=ELLIPSE_SCALE,
    colors=["#cc2222", "k"],
    labels=["Posterior mean", "Ground truth"],
    linewidths=[1.5, 1.2],
    alpha=[0.9, 0.7],
)
# --8<-- [end:plot_uncertainty]

ax.scatter(ref_points[:, 0], ref_points[:, 1], c="g", s=5, zorder=5)
ax.plot(
    [],
    [],
    color="#4444cc",
    alpha=0.5,
    linewidth=1,
    label=f"Posterior samples (n={n_ensemble})",
)
ax.set_aspect("equal")
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.set_xlabel("Model dimension 1")
ax.set_ylabel("Model dimension 2")
ax.set_title(
    f"Posterior predictive covariance field\n"
    f"chains={NUM_CHAINS}, draws={NUM_SAMPLES}, MC_samples={MC_SAMPLES}"
)
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(
    plot_path("nuts_ellipses_uncertainty.png"),
    dpi=200,
    bbox_inches="tight",
)
plt.close()
print("  Saved: nuts_ellipses_uncertainty.png")

# ---------------------------------------------------------------------------
# 7. MAP vs NUTS mean comparison (side-by-side)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
# (field, colour, label, draw_the_posterior_ensemble_behind_it)
#
# The ensemble goes on the NUTS panel only, and that asymmetry IS the point of
# the figure: MAP can only ever draw one contour, because its posterior is a
# delta at theta_MAP. NUTS draws a band from the same downstream code. Showing
# the mean alone makes the two look like equivalent point estimates, which is
# exactly the misreading this example exists to prevent.
configs = [
    (truth_field, "k", "Ground truth", False),
    (
        WPPMCovarianceField(model, map_posterior.params),
        "#cc7700",
        f"MAP ({MAP_STEPS} steps)",
        False,
    ),
    (
        mean_field,
        "#cc2222",
        f"NUTS mean ({NUM_CHAINS}×{NUM_SAMPLES} draws) + {n_ensemble} posterior samples",
        True,
    ),
]
# Computed once, reused by the NUTS panel below.
ensemble = jax.device_get(ensemble_covs(param_samples, ref_points))

for ax, (field, color, label, with_ensemble) in zip(axes, configs):
    if with_ensemble:
        plot_ellipses(
            jax.device_get(ref_points),
            ensemble,
            ax=ax,
            scale=ELLIPSE_SCALE,
            colors="#4444cc",
            linewidths=0.5,
            alpha=0.15,
        )
        # A proxy handle: the ensemble is n_ensemble fields sharing one colour,
        # so labelling it in plot_ellipses would give n_ensemble identical
        # legend entries.
        ax.plot(
            [],
            [],
            color="#4444cc",
            alpha=0.5,
            linewidth=1,
            label=f"Posterior samples (n={n_ensemble})",
        )
    plot_ellipses(
        jax.device_get(ref_points),
        jax.device_get(field(ref_points)),
        ax=ax,
        scale=ELLIPSE_SCALE,
        colors=color,
        linewidths=1.2,
        alpha=0.8,
    )
    ax.scatter(ref_points[:, 0], ref_points[:, 1], c="g", s=5, zorder=5)
    ax.set_aspect("equal")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_title(label, fontsize=10)
    ax.grid(True, alpha=0.3)
    if with_ensemble:
        ax.legend(loc="lower left", fontsize=8)

plt.suptitle(
    "Ground truth vs MAP (point estimate) vs NUTS (posterior mean + spread)", y=1.01
)
plt.tight_layout()
fig.savefig(plot_path("nuts_vs_map_comparison.png"), dpi=200, bbox_inches="tight")
plt.close()
print("  Saved: nuts_vs_map_comparison.png")

# ---------------------------------------------------------------------------
# 8. Everything in one axes: truth, MAP, posterior mean, posterior spread
#
# The three-panel figure above compares like with like; this one is for reading
# the *disagreements* off directly, which side-by-side panels make you do from
# memory. Four layers in two plot_ellipses calls: the ensemble as a stack, then
# the three means as a list of fields with per-field styling.
# ---------------------------------------------------------------------------
print("[7/7] Overlaying truth, MAP and posterior in one axes...")

# --8<-- [start:overlay]
# Posterior spread first, so it sits behind the three lines.
fig, ax = plt.subplots(figsize=(8, 8))
plot_ellipses(
    jax.device_get(ref_points),
    ensemble,
    ax=ax,
    scale=ELLIPSE_SCALE,
    colors="#4444cc",
    linewidths=0.5,
    alpha=0.15,
)
plot_ellipses(
    jax.device_get(ref_points),
    [
        jax.device_get(truth_field(ref_points)),
        jax.device_get(WPPMCovarianceField(model, map_posterior.params)(ref_points)),
        jax.device_get(covs_mean),
    ],
    ax=ax,
    scale=ELLIPSE_SCALE,
    colors=["k", "#cc7700", "#cc2222"],
    labels=["Ground truth", f"MAP ({MAP_STEPS} steps)", "NUTS posterior mean"],
    linewidths=[1.2, 1.4, 1.6],
    linestyles=["-", "--", "-"],
    alpha=[0.8, 0.9, 0.95],
    show_centers=True,
)
# --8<-- [end:overlay]

# plot_ellipses builds its legend from `labels`, one entry per field. The
# ensemble is n_ensemble fields sharing one colour, so it gets a proxy handle
# instead -- labelling it there would produce n_ensemble identical entries.
ax.plot(
    [],
    [],
    color="#4444cc",
    alpha=0.5,
    linewidth=1,
    label=f"Posterior samples (n={n_ensemble})",
)
ax.legend(loc="upper left", fontsize=9)
ax.set_aspect("equal")
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.set_xlabel("Model dimension 1")
ax.set_ylabel("Model dimension 2")
ax.set_title(
    f"Truth vs MAP vs posterior  ({MODE}: mc={MC_SAMPLES}, "
    f"{NUM_CHAINS}×{NUM_SAMPLES} draws, ellipses ×{ELLIPSE_SCALE})"
)
ax.grid(True, alpha=0.3)
fig.savefig(plot_path("nuts_overlay.png"), dpi=200, bbox_inches="tight")
plt.close()
print("  Saved: nuts_overlay.png")

print("\nDone. All plots saved to:", PLOTS_DIR)
