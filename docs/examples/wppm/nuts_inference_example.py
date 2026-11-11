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
    # `module load cuda cudnn` alongside it; see CONTRIBUTING.md.
    #
    # Add `progress` for the progress bars; without it `show_progress=True` is
    # silently a no-op and a long fit prints nothing for hours.

Usage
-----
    python nuts_inference_example.py                    # quick: laptop smoke test
    python nuts_inference_example.py --mode balanced    # a real fit, ~GPU-hour
    python nuts_inference_example.py --mode full        # paper-scale, batch job
    python nuts_inference_example.py --mode balanced --from-fit   # figures only

`quick` is the default so the docs build and a laptop run stay fast. It is NOT a
fit to interpret: 450 trials and 50 MC samples leave the frozen-key posterior
visibly biased, and 150 draws cannot support R-hat or ESS. Use `balanced` or
`full` for anything you intend to read.

The documentation embeds the `balanced` figures, i.e. the mode-suffixed
plots/nuts_*_balanced.{png,jpg}. `quick` writes the unsuffixed names and its
output is throwaway.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.pop("JAX_PLATFORM_NAME", None)  # allow GPU/TPU if available

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np

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
    WPPMCovarianceFieldSamples,
    contour_radii,
    covariance_to_arviz,
)
from psyphy.posterior import MAPPosterior, MCMCPosterior
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
_parser.add_argument(
    "--from-fit",
    action="store_true",
    help="skip MAP and NUTS; rebuild every figure and diagnostic from the "
    "cached fit for this mode (see fit_path). Needs no GPU.",
)
_args = _parser.parse_args()
MODE = _args.mode
SEED = _args.seed
FROM_FIT = _args.from_fit

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def plot_path(name: str) -> str:
    """Canonical filename for `quick`, mode-suffixed otherwise.

    The documentation embeds the `--mode balanced` figures,
    plots/nuts_*_balanced.{png,jpg}: a `quick` fit is not converged, so its
    plots show little beyond "not converged yet". Every mode other than `quick`
    therefore suffixes its filenames with the mode, the same convention
    hong2025_reproduction.py uses. `quick` writes the unsuffixed names as throwaway
    smoke-test output, which must never be what the docs embed.
    """
    if MODE == "quick":
        return os.path.join(PLOTS_DIR, name)
    stem, ext = os.path.splitext(name)
    return os.path.join(PLOTS_DIR, f"{stem}_{MODE}{ext}")


FITS_DIR = os.path.join(os.path.dirname(__file__), "fits")


def fit_path() -> str:
    """Where this mode's cached fit lives.

    Same convention as hong2025_reproduction.py: one .npz per mode. The
    `balanced` file is the fit this tutorial is written from and is committed
    (~0.5 MB), so every figure and every printed diagnostic on the page can be
    regenerated with --from-fit on a laptop. The `quick` file is smoke-test
    output and is gitignored.
    """
    return os.path.join(FITS_DIR, f"nuts_{MODE}_fit.npz")


def save_ellipse_figure(fig, name: str) -> None:
    """Save an ellipse figure as JPEG (150 dpi, quality 85) and close it.
    JPEG is more compact than PNG for these figures, and the quality
    loss is visually negligible.
    """
    fig.savefig(
        plot_path(f"{name}.jpg"),
        dpi=150,
        bbox_inches="tight",
        # Passed through to Pillow's JPEG encoder. optimize=True costs a
        # little time and saves a few percent of size at no loss.
        pil_kwargs={"quality": 85, "optimize": True},
    )
    plt.close(fig)
    print(f"  Saved: {os.path.basename(plot_path(name + '.jpg'))}")


print("DEVICE:", jax.devices()[0])
print("MODE  :", MODE)

# ---------------------------------------------------------------------------
# Model, identical in every mode: --mode changes how carefully
# the posterior is explored, never what is being fit, so two modes stay
# comparable. Change these to change the science, not the budget.
# ---------------------------------------------------------------------------
INPUT_DIM = 2
BASIS_DEGREE = 2
EXTRA_DIMS = 1
DECAY_RATE = 0.4
VARIANCE_SCALE = 4e-3
DIAG_TERM = 1e-4
BANDWIDTH = 1e-2

# Compute/fidelity presets. Same MODES convention as hong2025_reproduction.py.
#
# Why each knob is where it is in `full`:
#   mc_samples 2000  the value hong2025.PAPER_HYPERPARAMS uses for this model
#                    class. It is the knob that matters most here: the MC
#                    likelihood is only deterministic because NUTSSampler
#                    freezes one PRNG key, which makes the sampled posterior
#                    the one implied by that noise draw. The bias shrinks as
#                    mc_samples grows. 50 is a smoke test, 200 the documented
#                    minimum, 2000 the paper's.
#   trials 5000      25 reference points x 200 trials. 450 trials (quick) is
#                    thin for this many coefficients, and under-determined
#                    posteriors are wide for reasons that have nothing to do
#                    with the sampler.
#   4 x 1000 samples the standard NUTS configuration, and what the shapes in
#                    the design note assume. R-hat needs >= 2 chains (4 is
#                    convention) and ESS > 100 per parameter needs the samples.
#   warmup 1000      dual averaging tunes the step size AND the mass matrix;
#                    for WPPM the mass matrix matters a lot, because
#                    coefficients at different Chebyshev degrees have genuinely
#                    different prior scales. 100 warmup steps cannot do it.
# --8<-- [start:compute_settings]
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
# Tree-depth cap: a sample costs at most 2**MAX_NUM_DOUBLINGS - 1 gradient
# evaluations (255 here; BlackJAX's default 10 would allow 1023). Check
# sampler_stats["n_steps"] after the run; if most samples hit 255, raise it.
MAX_NUM_DOUBLINGS = 8
# Sampling seed, and the frozen key the Monte Carlo likelihood is evaluated
# with. Both define *which* posterior was sampled, not just how, so they are
# named here rather than inlined: the fit cache fingerprints them.
NUTS_SEED = 7
LOGDENSITY_KEY_SEED = 0
TARGET_ACCEPTANCE_RATE = 0.8
# --8<-- [end:nuts_settings]
N_ENSEMBLE = _cfg["ensemble"]

print(
    f"        mc={MC_SAMPLES}, {N_REF_GRID**2}x{NUM_TRIALS_PER_REF} trials, "
    f"MAP {MAP_STEPS} steps, NUTS {NUM_CHAINS}x({NUM_WARMUP}+{NUM_SAMPLES})"
)

# Everything that changes what was fitted. Stored with the cache and compared
# on load: a figure captioned `balanced` that was drawn from `quick` samples is
# precisely the failure this cache could otherwise introduce, so a mismatch is
# a hard error naming the field rather than a warning nobody reads.
RUN_SETTINGS = {
    "mode": MODE,
    "seed": SEED,
    "mc_samples": MC_SAMPLES,
    "bandwidth": BANDWIDTH,
    "trials_per_ref": NUM_TRIALS_PER_REF,
    "n_ref_grid": N_REF_GRID,
    "chains": NUM_CHAINS,
    "warmup": NUM_WARMUP,
    "draws": NUM_SAMPLES,
    "max_num_doublings": MAX_NUM_DOUBLINGS,
    "nuts_seed": NUTS_SEED,
    "logdensity_key_seed": LOGDENSITY_KEY_SEED,
    "target_acceptance_rate": TARGET_ACCEPTANCE_RATE,
    "map_steps": MAP_STEPS,
    "map_lr": MAP_LR,
    "map_momentum": MAP_MOMENTUM,
    "input_dim": INPUT_DIM,
    "basis_degree": BASIS_DEGREE,
    "extra_dims": EXTRA_DIMS,
    "decay_rate": DECAY_RATE,
    "variance_scale": VARIANCE_SCALE,
    "diag_term": DIAG_TERM,
}


def save_fit(map_posterior, nuts_posterior, ref_points, truth_params) -> None:
    """Persist the fit so the figures can be rebuilt without re-sampling.

    Stores the draws with their chain structure intact (diagnostics need it),
    the sampler statistics, the MAP estimate the chains were warm-started
    from, and the ground truth and reference grid the figures are drawn
    against. It also stores RUN_SETTINGS, which is what makes the file
    self-describing rather than a bag of arrays whose provenance you must remember.
    """
    os.makedirs(FITS_DIR, exist_ok=True)
    stats = {
        f"stat_{k}": np.asarray(jax.device_get(v))
        for k, v in nuts_posterior.sampler_stats.items()
    }
    np.savez_compressed(
        fit_path(),
        settings=json.dumps(RUN_SETTINGS, sort_keys=True),
        draws_W=np.asarray(jax.device_get(nuts_posterior.samples["W"])),
        map_W=np.asarray(jax.device_get(map_posterior.params["W"])),
        truth_W=np.asarray(jax.device_get(truth_params["W"])),
        ref_points=np.asarray(jax.device_get(ref_points)),
        **stats,
    )
    _mb = os.path.getsize(fit_path()) / 1e6
    print(f"  Cached fit: {os.path.basename(fit_path())} ({_mb:.2f} MB)")


def load_fit(model, ref_points, truth_params):
    """Rebuild (map_posterior, nuts_posterior) from the cache for this mode.

    Refuses rather than guesses: a missing file, a settings mismatch or a
    ground truth that does not match the one just rebuilt all raise, naming
    what differs.
    """
    path = fit_path()
    if not os.path.exists(path):
        raise SystemExit(
            f"--from-fit: no cached fit at {path}\n"
            f"            run `python {os.path.basename(__file__)} "
            f"--mode {MODE}` once to create it."
        )
    blob = np.load(path, allow_pickle=False)
    cached = json.loads(str(blob["settings"]))
    differs = [
        f"            {k}: cached {cached.get(k, '<missing>')!r} != current {v!r}"
        for k, v in RUN_SETTINGS.items()
        if cached.get(k, "<missing>") != v
    ]
    if differs:
        raise SystemExit(
            f"--from-fit: {os.path.basename(path)} was produced with different "
            "settings:\n" + "\n".join(differs)
        )
    # The data are re-simulated rather than stored, so check that they came out
    # the same: identical truth and grid mean identical trials downstream.
    for name, cached_arr, current in (
        ("truth_W", blob["truth_W"], truth_params["W"]),
        ("ref_points", blob["ref_points"], ref_points),
    ):
        if not np.allclose(cached_arr, jax.device_get(current)):
            raise SystemExit(
                f"--from-fit: {name} differs from the cached run. The cache "
                "does not describe this script's current ground truth."
            )
    stats = {
        k[len("stat_") :]: jnp.asarray(blob[k])
        for k in blob.files
        if k.startswith("stat_")
    }
    return (
        MAPPosterior({"W": jnp.asarray(blob["map_W"])}, model),
        MCMCPosterior({"W": jnp.asarray(blob["draws_W"])}, model, sampler_stats=stats),
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

# Stimuli are packed (N, K, d) with K=2: slot 0 reference, slot 1 comparison.
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
    prior=prior,
    likelihood=task,
    noise=noise,
    diag_term=DIAG_TERM,
)
# --8<-- [end:build_model]

# ---------------------------------------------------------------------------
# 3. MAP warm-start, and 4. NUTS sampling
#
# Both are skipped under --from-fit: they are the only expensive steps, and
# everything after them is post-processing of the samples. Initializing NUTS near
# the posterior mode greatly reduces warmup steps.
# ---------------------------------------------------------------------------
if FROM_FIT:
    print(f"[3/7] --from-fit: skipping MAP ({os.path.basename(fit_path())})")
    print("[4/7] --from-fit: skipping NUTS; loading cached samples...")
    map_posterior, nuts_posterior = load_fit(model, ref_points, truth_params)
else:
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
    # O(2^tree_depth) gradient evaluations per sample and the depth is data
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
    # closer to the total. The upper bracket is the tree-depth cap: no sample can
    # cost more than 2**MAX_NUM_DOUBLINGS - 1 gradient evaluations.
    print(f"  {_total_draws} samples total ({NUM_CHAINS} chains vmapped)")
    for _g_per_draw in (16, 64, 2**MAX_NUM_DOUBLINGS - 1):
        _hours = _per_grad * _total_draws * _g_per_draw / 3600.0
        print(f"    at {_g_per_draw:2d} grad evals/sample -> {_hours:7.2f} h")
    # The probe times *execution* of an already-compiled gradient. It therefore
    # EXCLUDES XLA compilation of the warmup scan, which is a one-off cost but can
    # take minutes and, on a short run like --mode quick, dominates everything the
    # projection above reports. If the run seems stuck right after "[4/7]", it is
    # almost certainly still compiling, not hung.
    print("    (excludes one-off XLA compilation of the warmup scan: can be minutes)")
    # --8<-- [end:cost_probe]

    # ---------------------------------------------------------------------------
    # 4. NUTS sampling
    # ---------------------------------------------------------------------------
    print("[4/7] Running NUTS posterior sampling (BlackJAX)...")
    print(f"  {NUM_CHAINS} chains × {NUM_WARMUP} warmup + {NUM_SAMPLES} samples")
    print("  Note: warmup compiles once for all chains: may take ~30s+ before it runs.")

    # --8<-- [start:nuts_fit]
    sampler = NUTSSampler(
        num_warmup=NUM_WARMUP,
        num_samples=NUM_SAMPLES,
        num_chains=NUM_CHAINS,
        logdensity_key_seed=LOGDENSITY_KEY_SEED,  # fixed MC key -> deterministic
        target_acceptance_rate=TARGET_ACCEPTANCE_RATE,
        max_num_doublings=MAX_NUM_DOUBLINGS,  # cap: <= 255 grad evals per sample
        # Sample z = W / prior_std so warmup starts from a metric matched to the
        # prior scales instead of the identity (the default; shown for clarity).
        prior_preconditioning=True,
        show_progress=True,
    )
    nuts_posterior = sampler.fit(
        model,
        data,
        init_params=map_posterior.params,  # warm-start from MAP
        seed=NUTS_SEED,
    )
    # --8<-- [end:nuts_fit]

    # Cache before plotting: the sampling is the expensive part, and a crash in
    # a figure must not cost it. A failure to cache is not a failure of the run.
    try:
        save_fit(map_posterior, nuts_posterior, ref_points, truth_params)
    except OSError as _exc:
        print(f"  WARNING: could not cache the fit ({_exc}); continuing")

print(
    f"  Posterior: {nuts_posterior.n_chains} chains × {nuts_posterior.n_draws} samples"
)
acc = nuts_posterior.sampler_stats["acceptance_rate"]
print(f"  Mean acceptance rate: {float(jnp.mean(acc)):.3f}")
# Divergences first: a divergent transition means the integrator blew up rather
# than following the posterior's geometry, so the kept samples are biased in a way
# more sampling does not fix. Any non-zero count makes the run suspect whatever
# R-hat says, which is why it is printed before the convergence diagnostics.
# --8<-- [start:divergence_check]
_div = nuts_posterior.sampler_stats["diverging"]
_n_div = int(jnp.sum(_div))
print(
    f"  Divergences: {_n_div} / {_div.size} samples ({_n_div / _div.size:.1%})"
    f"{'  <-- investigate; try target_acceptance_rate=0.9' if _n_div else ''}"
)
# Fraction of post-warmup samples whose trajectory was cut off by the depth cap.
# A large fraction means max_num_doublings is limiting mixing.
_n_steps = nuts_posterior.sampler_stats["n_steps"]
_at_cap = float(jnp.mean(_n_steps >= 2**MAX_NUM_DOUBLINGS - 1))
print(
    f"  Leapfrog steps/sample: mean {float(jnp.mean(_n_steps)):.0f}, "
    f"{_at_cap:.0%} at the cap ({2**MAX_NUM_DOUBLINGS - 1})"
)
# --8<-- [end:divergence_check]

# ---------------------------------------------------------------------------
# 5. Posterior predictive: ensemble of covariance field ellipses
# ---------------------------------------------------------------------------
print("[5/7] Plotting posterior samples and the posterior mean field...")

ELLIPSE_SCALE = 1.2  # magnification; ellipses are not drawn at true size

# Posterior samples as covariance fields
# --8<-- [start:posterior_samples]
fields = WPPMCovarianceFieldSamples(nuts_posterior)  # every stored sample
# (n_samples, n_ref, 2, 2): one field per sample
covs_samples = fields(ref_points)
# every k-th sample, so the plot shows n_ensemble contours from all chains
n_ensemble = N_ENSEMBLE
covs_ensemble = covs_samples[:: fields.n_samples // n_ensemble]
# --8<-- [end:posterior_samples]

# Posterior mean covariance field, E[Σ(x)]: Σ is computed for every sample and
# the matrices are averaged. Not
# WPPMCovarianceField(model, nuts_posterior.params): params is the mean
# parameter E[W], and Σ is nonlinear in W, so Σ(E[W]) comes out far too
# small. See the WPPMCovarianceFieldSamples docstring.
# --8<-- [start:posterior_mean]
covs_mean = fields.mean(ref_points)  # (n_ref, 2, 2)
# --8<-- [end:posterior_mean]
covs_truth = truth_field(ref_points)

fig, ax = plt.subplots(figsize=(7, 7))

# --8<-- [start:plot_posterior_samples]
# We first draw the posterior spread, so it's behind the two lines. 1 call:
# (n_ensemble, n_ref, 2, 2) is the "several fields" shape plot_ellipses takes
plot_ellipses(
    jax.device_get(ref_points),
    jax.device_get(covs_ensemble),
    ax=ax,
    scale=ELLIPSE_SCALE,
    colors="C0",
    linewidths=0.5,
    alpha=0.15,
)

# Posterior mean and ground truth, overlaid as a list of fields so each gets its
# own color, width and legend entry:
plot_ellipses(
    jax.device_get(ref_points),
    [jax.device_get(covs_mean), jax.device_get(covs_truth)],
    ax=ax,
    scale=ELLIPSE_SCALE,
    colors=["C3", "k"],
    labels=["Posterior mean", "Ground truth"],
    linewidths=[1.5, 1.2],
    alpha=[0.9, 0.7],
)
# --8<-- [end:plot_posterior_samples]

ax.scatter(ref_points[:, 0], ref_points[:, 1], c="C2", s=5, zorder=5)
ax.plot(
    [],
    [],
    color="C0",
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
    f"chains={NUM_CHAINS}, samples={NUM_SAMPLES}, MC_samples={MC_SAMPLES}"
)
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)
plt.tight_layout()
save_ellipse_figure(fig, "nuts_ellipses_uncertainty")

# ---------------------------------------------------------------------------
# 5b. Pointwise credible band on the threshold contour
#
# The ensemble figure above shows a cloud of contours. A cloud is not what gets
# reported in a paper; an interval is. Writing each sample's contour as a radius
# per direction turns the cloud into a quantile band.
# ---------------------------------------------------------------------------
print("[5b/7] Pointwise credible band on the threshold contours...")


# --8<-- [start:credible_band]
CI_PROB = 0.89  # ArviZ's default, so the band and az.summary match

band = fields.band(ref_points, ci_prob=CI_PROB)
# band.lower / .median / .upper  (n_ref, n_directions): contour radius per direction
# band.directions  (n_directions, 2): closed, so a filled polygon has no seam
# --8<-- [end:credible_band]

r_truth = contour_radii(covs_truth, band.directions)  # (n_ref, n_directions)

_u = jax.device_get(band.directions)
_refs = jax.device_get(ref_points)
_lo, _med, _hi, _truth = (
    jax.device_get(band.lower),
    jax.device_get(band.median),
    jax.device_get(band.upper),
    jax.device_get(r_truth),
)

fig, ax = plt.subplots(figsize=(7, 7))
for k in range(_refs.shape[0]):
    outer = _refs[k] + ELLIPSE_SCALE * _hi[k][:, None] * _u
    inner = _refs[k] + ELLIPSE_SCALE * _lo[k][:, None] * _u
    # One path, inner contour reversed: nonzero winding leaves the hole unfilled,
    # so the fill is an annulus between the two quantile contours.
    ax.fill(
        np.concatenate([outer[:, 0], inner[::-1, 0]]),
        np.concatenate([outer[:, 1], inner[::-1, 1]]),
        color="C0",
        alpha=0.25,
        linewidth=0,
    )
    for r, color, lw in ((_med[k], "C0", 1.0), (_truth[k], "k", 1.0)):
        pts = _refs[k] + ELLIPSE_SCALE * r[:, None] * _u
        ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=lw, alpha=0.85)

ax.plot([], [], color="C0", alpha=0.25, linewidth=8, label=f"{CI_PROB:.0%} CI")
ax.plot([], [], color="C0", linewidth=1, label="Posterior median contour")
ax.plot([], [], color="k", linewidth=1, label="Ground truth")
ax.scatter(_refs[:, 0], _refs[:, 1], c="C2", s=5, zorder=5)
ax.set_aspect("equal")
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.set_xlabel("Model dimension 1")
ax.set_ylabel("Model dimension 2")
ax.set_title(
    f"Pointwise {CI_PROB:.0%} credible band on the threshold contour\n"
    f"chains={NUM_CHAINS}, samples={NUM_SAMPLES}, MC_samples={MC_SAMPLES}"
)
ax.legend(loc="upper right", fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
save_ellipse_figure(fig, "nuts_threshold_band")


# ---------------------------------------------------------------------------
# 5c. Nested bands: several probability levels in one call
# ---------------------------------------------------------------------------
print("[5c/7] Nested credible bands at one reference point...")

# --8<-- [start:nested_bands]
# A sequence of probabilities gives one interval per level in a single call,
# over the same samples the band above used:
nested = fields.band(ref_points, ci_prob=[0.5, 0.8, 0.95])
# nested.lower / .median / .upper  (n_levels, n_ref, n_directions)
# nested.ci_prob                    (0.5, 0.8, 0.95), the order of that axis
# One shape for all three, so nested.lower[i] and nested.median[i] line up and
# a single loop over i draws every level.
# --8<-- [end:nested_bands]

# Show one reference point, the least constrained one, so the three levels are
# far enough apart to tell apart by eye.
_rel_width = (nested.upper[-1] - nested.lower[-1]) / nested.median[-1]
_k = int(jnp.argmax(jnp.mean(_rel_width, axis=1)))  # widest band, (n_ref,) -> index

_ref_k = jax.device_get(ref_points[_k])
_dirs = jax.device_get(nested.directions)  # (n_directions, 2)
fig, ax = plt.subplots(figsize=(5.5, 5.5))
# Widest level first so the narrower fills draw on top of it.
for level in reversed(range(len(nested.ci_prob))):
    lo = jax.device_get(nested.lower[level][_k])  # (n_directions,)
    hi = jax.device_get(nested.upper[level][_k])
    outer = _ref_k + ELLIPSE_SCALE * hi[:, None] * _dirs
    inner = _ref_k + ELLIPSE_SCALE * lo[:, None] * _dirs
    ax.fill(
        np.concatenate([outer[:, 0], inner[::-1, 0]]),
        np.concatenate([outer[:, 1], inner[::-1, 1]]),
        color="C0",
        alpha=0.22,
        linewidth=0,
        label=f"{nested.ci_prob[level]:.0%} CI",
    )
_med_k = jax.device_get(nested.median[0][_k])
_truth_k = jax.device_get(r_truth[_k])
for r, color, lab in (
    (_med_k, "C0", "Posterior median"),
    (_truth_k, "k", "Ground truth"),
):
    pts = _ref_k + ELLIPSE_SCALE * r[:, None] * _dirs
    ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=1.2, label=lab)
ax.scatter(*_ref_k, c="C2", s=20, zorder=5)
ax.set_aspect("equal")
ax.set_xlabel("Model dimension 1")
ax.set_ylabel("Model dimension 2")
ax.set_title(
    "Nested credible bands on one threshold contour\n"
    f"ci_prob=[0.5, 0.8, 0.95], {fields.n_samples} samples"
)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], loc="upper right", fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
save_ellipse_figure(fig, "nuts_threshold_band_nested")

# ---------------------------------------------------------------------------
# 6. ArviZ diagnostics
# ---------------------------------------------------------------------------
print("[6/7] Computing ArviZ diagnostics...")

try:
    import arviz as az

    # Diagnostics are computed on the covariance field Σ(x), not on W. W is
    # only identified up to a rotation of its embedding axis (W -> W R leaves
    # every Σ unchanged), so chains can disagree about W while agreeing about
    # every ellipse, and diagnostics on W report a failure that does not
    # exist. Σ is a function of x, so we must choose WHERE to evaluate it: here
    # the reference points, i.e. exactly the ellipses this script plots and
    # reports. covariance_to_arviz has no default for `points`.
    # --8<-- [start:arviz_diagnostics]
    dt_sigma = covariance_to_arviz(nuts_posterior, ref_points)
    summary = az.summary(dt_sigma, var_names=["Sigma"])
    print(summary[["mean", "sd", "r_hat", "ess_bulk"]].describe())
    r_hat_max = float(summary["r_hat"].max())
    ess_min = float(summary["ess_bulk"].min())
    print(f"  R-hat max on Σ: {r_hat_max:.3f}  (target < 1.01)")
    print(f"  ESS min on Σ:   {ess_min:.0f}    (target > 100 per chain)")
    # --8<-- [end:arviz_diagnostics]

    # Trace plot at the reference point with the highest R-hat (the worst
    # case, picked by the numbers rather than by hand): one panel per quantity
    # (log_var_0, log_var_1, corr_0_1), one line per chain. Plain ArviZ output.
    #
    # NOTE (arviz >= 1.0): plot_trace returns an arviz_plots PlotCollection,
    # not an array of matplotlib axes, so it is saved with pc.savefig(). The
    # 0.x kwargs `compact=` and `figsize=` no longer exist and raise if passed.
    # bbox_inches="tight" only affects saving: without it, ArviZ's two-line
    # panel titles are clipped at the top edge of the image.
    # --8<-- [start:trace_plot]
    # pick the reference point with the worst R-hat
    worst_point = int(az.rhat(dt_sigma)["Sigma"].max("quantity").argmax(dim="point"))
    pc = az.plot_trace(dt_sigma, var_names=["Sigma"], coords={"point": worst_point})
    pc.savefig(plot_path("nuts_trace_sigma.png"), bbox_inches="tight")
    # --8<-- [end:trace_plot]
    print(f"  Saved: {os.path.basename(plot_path('nuts_trace_sigma.png'))}")

    # ESS (bulk, tail) and R-hat over ALL Σ quantities, ArviZ defaults: one
    # cumulative curve per diagnostic (height = fraction of quantities at or
    # below that value) and ArviZ's own reference thresholds as dashed lines.
    # --8<-- [start:convergence_plot]
    pc = az.plot_convergence_dist(dt_sigma, var_names=["Sigma"])
    pc.savefig(plot_path("nuts_convergence_sigma.png"), bbox_inches="tight")
    # --8<-- [end:convergence_plot]
    print(f"  Saved: {os.path.basename(plot_path('nuts_convergence_sigma.png'))}")


except ImportError:
    print(
        "  ArviZ not installed, skipping diagnostics. "
        "Install with: pip install 'psyphy[diagnostics]'"
    )

# ---------------------------------------------------------------------------
# 7. MAP vs NUTS mean comparison (side-by-side)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
# (field, color, label, draw_the_posterior_ensemble_behind_it)
#

configs = [
    (truth_field, "k", "Ground truth", False),
    (
        WPPMCovarianceField(model, map_posterior.params),
        "C1",
        f"MAP ({MAP_STEPS} steps)",
        False,
    ),
    (
        fields.mean,
        "C3",
        f"NUTS mean ({NUM_CHAINS}×{NUM_SAMPLES} samples) + {n_ensemble} of them shown",
        True,
    ),
]
# the ensemble from section 5, reused by the NUTS panel below
ensemble = jax.device_get(covs_ensemble)

for ax, (field, color, label, with_ensemble) in zip(axes, configs):
    if with_ensemble:
        plot_ellipses(
            jax.device_get(ref_points),
            ensemble,
            ax=ax,
            scale=ELLIPSE_SCALE,
            colors="C0",
            linewidths=0.5,
            alpha=0.15,
        )
        # A proxy handle: the ensemble is n_ensemble fields sharing one color,
        # so labelling it in plot_ellipses would give n_ensemble identical
        # legend entries.
        ax.plot(
            [],
            [],
            color="C0",
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
    ax.scatter(ref_points[:, 0], ref_points[:, 1], c="C2", s=5, zorder=5)
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
save_ellipse_figure(fig, "nuts_vs_map_comparison")

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
    colors="C0",
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
    colors=["k", "C1", "C3"],
    labels=["Ground truth", f"MAP ({MAP_STEPS} steps)", "NUTS posterior mean"],
    linewidths=[1.2, 1.4, 1.6],
    linestyles=["-", "--", "-"],
    alpha=[0.8, 0.9, 0.95],
    show_centers=True,
)
# --8<-- [end:overlay]

# plot_ellipses builds its legend from `labels`, one entry per field. The
# ensemble is n_ensemble fields sharing one color, so it gets a proxy handle
# instead; labelling it there would produce n_ensemble identical entries.
ax.plot(
    [],
    [],
    color="C0",
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
    f"{NUM_CHAINS}×{NUM_SAMPLES} samples, ellipses ×{ELLIPSE_SCALE})"
)
ax.grid(True, alpha=0.3)
save_ellipse_figure(fig, "nuts_overlay")

print("\nDone. All plots saved to:", PLOTS_DIR)
