"""
Reproducing Hong et al. (2025) with psyphy
------------------------------------------

This  script fits *published human
data from Hong et al. (2025)* and compares against the authors' own published fit.

It runs in five stages, deliberately separated. The first two are given the
paper's own fitted weights and check what psyphy *computes* from them; only the
third asks psyphy to *fit* anything. Everything is per-observer: one subject per
run (``--subject``, default 1 = CH), never pooled.

  Stage 1 -- covariance field, exact. Feed the paper's published weights into
      psyphy's covariance field and compare to their published covariances.
      Deterministic: no optimizer, no Monte Carlo. Seconds.

  Stage 2 -- threshold contours, reproducing the paper's Figure 2B. Invert the
      oddity task to turn that noise field into 66.7%-correct discrimination
      thresholds, and plot them in the paper's own monitor-calibrated colors.
      Monte Carlo but no optimizer. ~20 s on CPU at the default settings.

  Stage 3 -- refit. Start from a prior sample and fit psyphy's WPPM to the
      paper's trials, then compare the resulting *noise* field to theirs, and
      save the fitted weights. At the paper's settings this is a GPU/cluster
      job; `--mode quick` is a seconds-long smoke test that does NOT reproduce
      anything.

  Stage 4 -- end to end. Take the weights stage 3 fit from the raw trials, run
      the same oddity inversion stage 2 runs, and compare the resulting
      *thresholds* to the published ones. This is the whole claim in one line:
      raw data -> our weights -> our contours -> the published figure. Reads
      the saved weights, so it costs ~20 s on CPU and needs no GPU.

  Stage 5 -- the yardstick. "Close enough" needs a criterion, so we use the
      authors' own: they resampled the AEPsych trials 120 times, refit the WPPM
      to each, ranked the fits by summed Normalized Bures Similarity against
      the original, kept the top 114 (95% of 120), and took the union and
      intersection of the retained threshold contours as their 95% CI. We
      reproduce that definition and report how many of our contours fall
      inside. Also ~20 s on CPU; the bootstrap contours ship pre-inverted.

The ordering is the point: stages 1-2 hold the model to account with the
optimizer removed from the picture, so if stage 3 disagrees you already know
the disagreement is the optimizer's and not the model's. Stage 4 then puts the
two halves back together, and stage 5 says whether the result is good enough by
the paper's own standard.

Calibration of stage 5, measured: feeding the paper's *own* published weights
through our inversion puts 49/49 reference contours inside their CI, at 100% of
sampled directions. That is the ceiling the end-to-end number is read against.

Usage
-----
    python hong2025_reproduction.py    # ~1 min CPU; stage 3 is only a smoke test
    python hong2025_reproduction.py --skip-refit    # stages 1-2 only, no fitting
    python hong2025_reproduction.py --mode full     # paper settings, GPU/cluster
    python hong2025_reproduction.py --skip-thresholds   # stages 1 and 3 only

    # on the cluster: fit once, keep the weights
    python hong2025_reproduction.py --mode full

    # then anywhere, as often as you like, without refitting
    python hong2025_reproduction.py --from-fit fits/hong2025_full_fit.npz \
        --no-noise-ellipses --skip-thresholds

Measured runtimes
-----------------
Keep these current when settings change -- they are quoted in the accompanying
markdown page. CPU figures are from an Apple Silicon laptop with JAX using
~12 cores; the GPU figure is a single CUDA device.

    Stage 1  covariance check, 10 609 pts, deterministic  CPU   seconds
    Stage 2  Figure 2B, 49 refs, n_theta=16,
             n_length=300, mc=500                         CPU   20-23 s
    Stage 2  at the paper's settings (n_length=1000,
             mc=2000): 13.4 s per reference point         CPU   ~11 min
    Stages 1+2 together                                   CPU   24-36 s
             (spread is CPU scheduling, not workload)
    Stage 3  quick smoke test, 500 trials, 20 steps,
             mc=50 -- the fit itself                      CPU   0.8 s
    Stage 3  full: 6 000 trials, 1 500 steps, mc=2000,
             3 restarts (3 x ~317 s)                      GPU   ~16 min

For reference, the paper's own SLURM header requests an H100 for 14 h -- but
that covers the main fit *plus* 120 bootstrap refits, not a single fit.

Data is downloaded on first run into ~/.cache/psyphy/ (override with
$PSYPHY_DATA_HOME). psyphy ships no data; see the accompanying markdown page.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

# --8<-- [start:x64]
import jax

# float64 must be enabled before the first JAX array exists (a JAX constraint,
# not ours). We use it because the paper does, and because it is the safe
# default for the stage-3 refit, where gradients accumulate over
# 6000 x 2000 x 1500 and diag_term=0 leaves Sigma unregularised.
#
# Stages 1-2 do NOT require it: measured on the full grid, float32 gives
# max|diff| 6.86e-9 vs float64's 6.78e-9, both inside the 1e-8 gate. Sigma
# entries are ~1e-3, so float32's *relative* 1.2e-7 epsilon resolves them to
# ~5e-10 absolute -- below the published CSV's own 1e-8 rounding.
jax.config.update("jax_enable_x64", True)
# --8<-- [end:x64]

import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402

if not os.environ.get("DISPLAY") and os.name != "nt":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# --8<-- [start:imports]
from psyphy.data.published import hong2025  # noqa: E402
from psyphy.inference import MAPOptimizer  # noqa: E402
from psyphy.model import WPPMCovarianceField  # noqa: E402
from psyphy.posterior import (  # noqa: E402
    MAPPosterior,
    ThresholdConfig,
    WPPMPredictivePosterior,
)
from psyphy.viz import auto_scale, plot_ellipses  # noqa: E402

# --8<-- [end:imports]

PLOTS_DIR = Path(__file__).parent / "plots"

# Stage 3 writes its fitted weights here so stage 4 can invert them without
# paying for the fit again. The fit is the only part that needs a GPU; keeping
# its output on disk means the inversion and the figure can be re-run on a
# laptop as often as you like.
FITS_DIR = Path(__file__).parent / "fits"

# --8<-- [start:modes]
# Stage-3 compute settings. "full" is the paper's own configuration; "quick"
# exists only to prove the pipeline runs -- it will NOT reproduce the paper.
MODES = {
    "quick": {"max_trials": 500, "mc_samples": 50, "steps": 20, "restarts": 1},
    "full": {"max_trials": None, "mc_samples": 2000, "steps": 1500, "restarts": 3},
}
# --8<-- [end:modes]

# --8<-- [start:threshold_settings]
# Threshold-inversion settings, shared by stage 2 (invert the paper's weights)
# and stage 4 (invert ours). `n_theta` is the same in both; only the distance
# grid and the Monte Carlo sample count differ.
#
# "paper" matches Hong et al. exactly and is the default: every committed
# figure and quoted number comes from it. "fast" trades accuracy for ~30x less
# wall clock -- it reproduces the published semi-axes to a median 2.18 %
# (max 10.78 %) in 20-23 s instead of ~11 min, which is what keeps the
# smoke test a smoke test.
THRESHOLD_SETTINGS = {
    "paper": {
        "mc_samples": 2000,
        "config": ThresholdConfig(n_theta=16, n_length=1000),
    },
    "fast": {
        "mc_samples": 500,
        "config": ThresholdConfig(n_theta=16, n_length=300),
    },
}
# --8<-- [end:threshold_settings]


# ---------------------------------------------------------------------------
# Comparison metrics
# ---------------------------------------------------------------------------
# --8<-- [start:metrics]
def _sqrtm_psd(M: np.ndarray) -> np.ndarray:
    """Matrix square root of a symmetric PSD matrix."""
    vals, vecs = np.linalg.eigh(M)
    return vecs @ np.diag(np.sqrt(np.clip(vals, 0.0, None))) @ vecs.T


def normalized_bures_similarity(A: np.ndarray, B: np.ndarray) -> float:
    """Normalized Bures Similarity between two PD matrices (1.0 == identical).

    The similarity measure Hong et al. use to rank bootstrap fits, reproduced
    here so the numbers are directly comparable to the paper's.
    """
    sa = _sqrtm_psd(A)
    inner = _sqrtm_psd(sa @ B @ sa)
    return float(np.trace(inner) / np.sqrt(np.trace(A) * np.trace(B)))


def compare_fields(Sigma_fit: np.ndarray, Sigma_ref: np.ndarray) -> dict[str, float]:
    """Compare two stacks of 2x2 covariances, shape (M, 2, 2).

    Compares Sigma  (never W ). U -> U Q for orthogonal Q leaves Sigma = U U^T
    unchanged and the prior is isotropic in the embedding axis, so the weights
    are not identifiable while the covariance field is.
    """
    rel_frob = np.linalg.norm(Sigma_fit - Sigma_ref, axis=(1, 2)) / np.linalg.norm(
        Sigma_ref, axis=(1, 2)
    )
    # Ellipse area is proportional to sqrt(det Sigma).
    area_ratio = np.sqrt(np.linalg.det(Sigma_fit) / np.linalg.det(Sigma_ref))

    def major_axis_angle(S: np.ndarray) -> np.ndarray:
        vecs = np.linalg.eigh(S)[1][..., -1]  # eigh returns ascending eigenvalues
        return np.arctan2(vecs[..., 1], vecs[..., 0])

    # Ellipse axes are undirected, so angle error lives on [0, 90] degrees.
    d_ang = np.degrees(major_axis_angle(Sigma_fit) - major_axis_angle(Sigma_ref))
    d_ang = np.abs((d_ang + 90.0) % 180.0 - 90.0)

    nbs = np.array(
        [
            normalized_bures_similarity(a, b)
            for a, b in zip(Sigma_ref, Sigma_fit, strict=True)
        ]
    )
    return {
        "rel_frobenius_median": float(np.median(rel_frob)),
        "rel_frobenius_max": float(rel_frob.max()),
        "area_ratio_median": float(np.median(area_ratio)),
        "angle_err_deg_median": float(np.median(d_ang)),
        "nbs_median": float(np.median(nbs)),
        "nbs_min": float(nbs.min()),
    }


# --8<-- [end:metrics]


# ---------------------------------------------------------------------------
# Plotting
#
# One convention across every comparison figure on this page, so the reader
# learns the legend once: the PUBLISHED field is black dashed at low alpha
# (reads as gray) and sits underneath; OURS is solid on top, colored by the
# reference stimulus via the monitor calibration matrix. Stages 2, 3, 4 and 5
# all follow it. The styling is spelled out at each call site rather than
# hidden behind a helper, because one of those calls is quoted in the docs.
# ---------------------------------------------------------------------------
# Every figure names its observer in the legend, on both curves, so a panel
# lifted out of the page cannot be mistaken for a different subject or for a
# group average. The paper fits each of its 8 observers separately; nothing
# here is ever pooled across them.
def _subject_tag(subject: int) -> str:
    """e.g. "subject 1 (CH)"."""
    return f"subject {subject} ({hong2025.SUBJECT_INITIALS.get(subject, '?')})"


def _published_label(subject: int, what: str = "published inversion") -> str:
    """Label for the authors' curve.

    ``what`` matters, because the figures do not all compare against the same
    published object. The threshold figures plot the authors' *published
    threshold table* -- contours they obtained by inverting their own fit -- so
    "published inversion" is the like-for-like counterpart to our inversion.
    The Sigma_noise figure plots no published table at all: it is psyphy's
    covariance field evaluated at their published weights, so it is labelled as
    weights rather than as a fit or an inversion.
    """
    return f"Hong et al. 2025, {what} — {_subject_tag(subject)}"


def _ours_label(subject: int, what: str) -> str:
    return f"psyphy, {what} — {_subject_tag(subject)}"


def _stimulus_colors(coords, M):
    """Per-reference RGB from the monitor calibration, or flat gray with a note.

    Returning the note rather than silently falling back means a gray figure
    cannot be mistaken for a correctly colored one.
    """
    if M is not None:
        return hong2025.w2d_to_rgb(coords, M), ""
    return (
        np.full((len(coords), 3), 0.45),
        "\nneutral gray \u2014 calibration matrix not downloaded",
    )


def _load_calibration():
    """Monitor calibration matrix, or None if it cannot be fetched."""
    try:
        return hong2025.load_calibration_matrix(hong2025.fetch_calibration_matrix())
    except Exception as exc:  # network, or OSF layout change
        print(f"  color calibration unavailable ({exc}); plotting in grey")
        return None


def plot_comparison(
    coords, Sigma_fit, Sigma_ref, out_path, title, scale, M=None, subject=1
):
    """Two noise fields overlaid, published vs fitted.

    Same convention as the threshold figures: published dashed gray underneath,
    ours solid on top colored by reference stimulus.
    """
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    colors, fallback_note = _stimulus_colors(coords, M)
    title = title + fallback_note
    plot_ellipses(
        coords,
        [Sigma_ref, Sigma_fit],
        ax=ax,
        scale=scale,
        colors=["black", colors],
        linestyles=["--", "solid"],
        linewidths=[2.2, 1.6],
        alpha=[0.35, None],
        labels=[
            # Not "published Sigma_noise": this curve is psyphy's covariance
            # field evaluated at their published weights. Stage 1 shows the two
            # agree to 7e-9, but the curve on screen is ours, from their W.
            f"{_published_label(subject, 'published weights')}, \u03a3_noise",
            f"{_ours_label(subject, 'our MAP refit')}, \u03a3_noise",
        ],
        show_centers=True,
    )
    ticks = np.linspace(-0.7, 0.7, 5)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlabel("Model Dimension 1")
    ax.set_ylabel("Model Dimension 2")
    ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path.name}")


# --8<-- [start:plot_thresholds]
def plot_threshold_figure(
    coords,
    Sigma_psyphy,
    Sigma_published,
    out_path,
    scale,
    M,
    title=None,
    label=None,
    subject=1,
):
    """Figure 2B: threshold contours, colored by reference stimulus.

    ``M`` is the monitor calibration matrix; when it is None the plot falls
    back to neutral gray and says so, so a gray figure cannot pass for a
    correctly colored one.

    ``title`` and ``label`` let stage 4 reuse this exact styling for the
    end-to-end figure -- same dashed-published-underneath convention, so the
    two figures on the page can be compared without re-reading the legend.
    """
    fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=150)

    colors, fallback_note = _stimulus_colors(coords, M)

    # Published contours underneath as a dashed outline, ours on top colored by
    # stimulus. One scale for both, so the comparison stays honest.
    # --8<-- [start:plot_call]
    plot_ellipses(
        coords,
        [Sigma_published, Sigma_psyphy],
        ax=ax,
        scale=scale,
        colors=["black", colors],
        linestyles=["--", "solid"],
        linewidths=[2.2, 1.6],
        alpha=[0.35, None],
        labels=[
            _published_label(subject),
            label or _ours_label(subject, "oddity inversion of their weights"),
        ],
        show_centers=True,
    )
    # --8<-- [end:plot_call]

    ticks = np.linspace(-0.7, 0.7, 5)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlim(-0.95, 0.95)
    ax.set_ylim(-0.95, 0.95)
    ax.set_xlabel("Model Dimension 1")
    ax.set_ylabel("Model Dimension 2")
    ax.set_title(
        (
            title
            or " 66.7%-correct discrimination thresholds\n"
            "Figure 2B in Hong et al. 2025 reproduced, subject 1 (CH)"
        )
        + fallback_note,
        fontsize=9,
    )
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path.name}")


# --8<-- [end:plot_thresholds]


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def stage1_exact_check(paths: dict[str, Path]) -> None:
    """Compare psyphy's covariance field to the published one, given the paper's W."""
    print("\n=== Stage 1: psyphy's Sigma(W_org) vs the published field ===")
    if "noise_ellipses" not in paths:
        print("  skipped (re-run with --noise-ellipses to download the 68 MB table)")
        return

    # --8<-- [start:stage1]
    W_org = hong2025.load_reference_W(paths["weights"])  # (5, 5, 2, 3)
    coords, sigma_published = hong2025.load_sigma_table(paths["noise_ellipses"])

    model = hong2025.build_paper_model(mc_samples=1)  # MC unused: no likelihood here
    field = WPPMCovarianceField(model, {"W": W_org})
    sigma_psyphy = np.asarray(field(jnp.asarray(coords)))

    max_abs = float(np.abs(sigma_psyphy - sigma_published).max())
    # --8<-- [end:stage1]

    print(f"  grid points  : {len(coords)}")
    print(f"  max |diff|   : {max_abs:.3e}")
    print(
        f"  mean |diff|  : {float(np.abs(sigma_psyphy - sigma_published).mean()):.3e}"
    )
    verdict = "PASS" if max_abs < 1e-8 else "FAIL"
    print(f"  {verdict} — the published CSV is rounded to 8 decimals, so this is")
    print("  agreement to the precision the published file can express.")


def stage2_thresholds(paths: dict[str, Path], thr: dict, subject: int) -> None:
    """Reproduce Figure 2B: threshold contours from the paper's own weights."""
    print("\n=== Stage 2: threshold contours (Figure 2B) from W_org ===")
    mc_samples, config = thr["mc_samples"], thr["config"]

    # --8<-- [start:thresholds]
    W_org = hong2025.load_reference_W(paths["weights"])
    coords, thres_published = hong2025.load_sigma_table(paths["thres_ellipses"])

    # Model: given weights W, how noisy is perception at each color?
    model = hong2025.build_paper_model(mc_samples=mc_samples)
    # Parameter posterior: which W do we believe? ,
    posterior = MAPPosterior({"W": W_org}, model)

    # Posterior Predictive: given what we believe about W, what do we predict
    # at these points? In threshold mode: how far a comparison must move from
    # each reference to be noticed 2/3 of the time.
    predictive = WPPMPredictivePosterior(
        posterior,
        jnp.asarray(coords),  # reference points only; the search finds comparisons
        n_samples=1,  # a point estimate has only one draw
        threshold_pred=True,
        threshold_config=config,  # search settings: how carefully to look
    )
    thres_psyphy = np.asarray(predictive.mean)  # (49, 2, 2); runs on first access
    # --8<-- [end:thresholds]

    # --8<-- [start:threshold_error]
    # Compare semi-axis lengths: sqrt of the covariance eigenvalues.
    got = np.sqrt(np.linalg.eigvalsh(thres_psyphy))
    want = np.sqrt(np.linalg.eigvalsh(thres_published))
    rel_err = np.abs(got - want) / want
    # --8<-- [end:threshold_error]

    print(f"  reference points : {len(coords)}")
    print(
        f"  semi-axis error  : median {np.median(rel_err) * 100:.2f} %, "
        f"max {rel_err.max() * 100:.2f} %"
    )
    print(
        f"  settings         : n_theta={config.n_theta}, "
        f"n_length={config.n_length}, mc={mc_samples}"
    )

    # The paper colors each ellipse by its reference stimulus, via a monitor
    # calibration matrix published alongside the data. Optional: the figure
    # falls back to neutral grey when it has not been downloaded.
    M = _load_calibration()

    plot_threshold_figure(
        coords,
        thres_psyphy,
        thres_published,
        PLOTS_DIR / "hong2025_thresholds.png",
        scale=auto_scale(coords, thres_published),
        M=M,
        subject=subject,
    )


def stage3_refit(
    paths: dict[str, Path], cfg: dict, mode: str, seed: int, subject: int
) -> Path:
    """Fit psyphy's WPPM to the published trials and compare fields.

    Returns the path of the saved weights, for stage 4 to invert.
    """
    print(f"\n=== Stage 3: refit from a prior sample (mode={mode}) ===")

    # --8<-- [start:load]
    # Only the AEPsych trials were used for the published fit; the MOCS trials
    # in the same file are held-out validation. This is the default.
    data = hong2025.load_trials(
        paths["trials"], max_trials=cfg["max_trials"], seed=seed
    )
    # --8<-- [end:load]
    print(f"  trials: {data.num_trials} (p_correct={float(data.responses.mean()):.4f})")

    # --8<-- [start:fit]
    model = hong2025.build_paper_model(mc_samples=cfg["mc_samples"])
    optimizer = MAPOptimizer(
        steps=cfg["steps"],  # number of gradient steps per restart
        learning_rate=hong2025.PAPER_HYPERPARAMS["learning_rate"],  # 1e-4, step size
        momentum=hong2025.PAPER_HYPERPARAMS["momentum"],  # 0.2, `heavy-ball` momentum
        reduction="mean",  # objective / N: a per-trial loss, so lr is independent of N
        max_grad_norm=None,  # no clipping
    )

    # The paper fits from 3 random initializations and keeps the lowest final
    # objective, guarding against a bad local optimum.
    best = None
    for r in range(cfg["restarts"]):
        t0 = time.time()
        init = model.init_params(jax.random.PRNGKey(seed + 1000 * r))
        posterior = optimizer.fit(model, data, init_params=init)
        _, losses = optimizer.get_history()
        print(
            f"  restart {r}: loss {losses[0]:.5f} -> {losses[-1]:.5f}"
            f"  ({time.time() - t0:.1f}s)"
        )
        if best is None or losses[-1] < best[1]:
            best = (posterior.params, losses[-1], list(losses))
    params, _, loss_hist = best
    # --8<-- [end:fit]

    # --8<-- [start:save_fit]
    # Persist the fitted weights. The fit is the expensive, GPU-bound step; the
    # threshold inversion that turns these weights into Figure 2B is ~20 s on a
    # laptop. Saving here is what lets stage 4 run anywhere, any number of
    # times, without refitting.
    FITS_DIR.mkdir(parents=True, exist_ok=True)
    fit_path = FITS_DIR / f"hong2025_{mode}_fit.npz"
    np.savez(
        fit_path,
        W=np.asarray(params["W"]),
        final_loss=np.asarray(loss_hist[-1]),
        mode=mode,
        seed=seed,
    )
    # --8<-- [end:save_fit]
    print(f"  saved {fit_path.name}")

    # --8<-- [start:compare]
    # Grid coordinates come from the published table, so there is no meshgrid
    # ordering convention to get wrong.
    coords, _ = hong2025.load_sigma_table(paths["thres_ellipses"])
    W_org = hong2025.load_reference_W(paths["weights"])  # original best fit Weights

    Sigma_ref = np.asarray(
        WPPMCovarianceField(model, {"W": W_org})(jnp.asarray(coords))
    )
    Sigma_fit = np.asarray(WPPMCovarianceField(model, params)(jnp.asarray(coords)))
    metrics = compare_fields(Sigma_fit, Sigma_ref)
    # --8<-- [end:compare]

    print(f"\n  --- fitted vs published field, {len(coords)} grid points ---")
    for key, value in metrics.items():
        print(f"    {key:24s} {value: .4f}")

    scale = auto_scale(coords, Sigma_ref)
    plot_comparison(
        coords,
        Sigma_fit,
        Sigma_ref,
        PLOTS_DIR / f"hong2025_{mode}_ellipses.png",
        f"Σ_noise(x) — psyphy MAP refit vs Hong et al. 2025 — {_subject_tag(subject)}\n"
        f" N={data.num_trials}, mc={cfg['mc_samples']}, steps={cfg['steps']}",
        scale=scale,
        M=_load_calibration(),
        subject=subject,
    )

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.plot(loss_hist, color="#4444aa")
    ax.set_xlabel("Step")
    ax.set_ylabel("Neg log posterior (per trial)")
    ax.set_title(f"Learning curve — mode={mode}, N={data.num_trials}", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"hong2025_{mode}_learning_curve.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved hong2025_{mode}_learning_curve.png")

    if mode == "quick":
        print(
            "\n  Quick mode does NOT reproduce the paper: after a few steps the fit\n"
            "  is still close to its prior. Use --mode full on a GPU."
        )

    return fit_path


def stage4_end_to_end(
    paths: dict[str, Path], fit_path: Path, mode: str, thr: dict, subject: int
) -> None:
    """Close the loop: raw trials -> our weights -> our contours -> Figure 2B.

    Stages 1-2 take the paper's weights as given, so they test what psyphy
    *computes*. Stage 3 fits weights but only ever compares noise fields. This
    stage is the end-to-end claim: it takes the weights stage 3 fit from the raw
    trials, runs the same oddity inversion stage 2 runs, and puts the result
    against the published thresholds in the same figure convention.
    """
    print(f"\n=== Stage 4: thresholds from OUR fitted weights (mode={mode}) ===")

    if not fit_path.exists():
        print(f"  skipped: no saved fit at {fit_path}")
        print("  run stage 3 first (--mode full on a GPU), or pass --from-fit")
        return

    # --8<-- [start:end_to_end]
    W_fit = jnp.asarray(np.load(fit_path)["W"])  # from stage 3, not the paper
    coords, thres_published = hong2025.load_sigma_table(paths["thres_ellipses"])

    # Identical to stage 2, except the weights are ours rather than theirs.
    model = hong2025.build_paper_model(mc_samples=thr["mc_samples"])
    predictive = WPPMPredictivePosterior(
        MAPPosterior({"W": W_fit}, model),
        jnp.asarray(coords),
        n_samples=1,
        threshold_pred=True,
        threshold_config=thr["config"],
    )
    thres_fit = np.asarray(predictive.mean)  # (49, 2, 2)

    # Same metric stage 2 reports, so the two numbers are directly comparable:
    # stage 2 isolates inversion error, this one carries fit error on top.
    got = np.sqrt(np.linalg.eigvalsh(thres_fit))
    want = np.sqrt(np.linalg.eigvalsh(thres_published))
    rel_err = np.abs(got - want) / want
    # --8<-- [end:end_to_end]

    print(f"  reference points : {len(coords)}")
    print(
        f"  semi-axis error  : median {np.median(rel_err) * 100:.2f} %, "
        f"max {rel_err.max() * 100:.2f} %"
    )
    print("  (stage 2's error is inversion only; this one is fit + inversion)")

    M = _load_calibration()

    plot_threshold_figure(
        coords,
        thres_fit,
        thres_published,
        PLOTS_DIR / f"hong2025_{mode}_thresholds_end_to_end.png",
        # One scale for both fields, from the published one, exactly as stage 2
        # does -- otherwise the two figures on the page are not comparable.
        scale=auto_scale(coords, thres_published),
        M=M,
        title=(
            " 66.7%-correct discrimination thresholds, end to end\n"
            "raw trials -> psyphy refit -> inversion, vs Hong et al. 2025 "
            f"— {_subject_tag(subject)}"
        ),
        label=_ours_label(subject, "our refit, then inversion"),
        subject=subject,
    )


#: The paper's bootstrap CI keeps the top 95% of 120 refits by NBS score, i.e.
#: 114. The published columns are already sorted that way -- ``rank0`` is the
#: most similar to the main fit -- so the selection is a slice, not a re-ranking.
N_BOOTSTRAP_CI = 114


def _radii(Sigmas: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Contour radius of each ellipse in each direction ``u``.

    The threshold contour is ``x^T Sigma^-1 x = 1``, so along a unit direction
    ``u`` the radius is ``1 / sqrt(u^T Sigma^-1 u)``. Working in radii rather
    than semi-axes is what lets us take the paper's union and intersection of
    *contours* rather than a per-semi-axis interval.

    Non-positive-definite inputs come back as nan rather than raising. A
    reference point whose threshold sweep failed to bracket 2/3 can yield a
    singular or indefinite Sigma, and ``np.linalg.inv`` would abort the whole
    stage -- an expensive way to fail, since this runs after the fit.
    """
    S = np.asarray(Sigmas, dtype=float)
    bad = np.linalg.eigvalsh(S)[..., 0] <= 0.0
    P = np.linalg.inv(np.where(bad[..., None, None], np.eye(2), S))
    q = np.einsum("di,...ij,dj->...d", u, P, u)  # (..., n_dirs)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = 1.0 / np.sqrt(q)
    return np.where(bad[..., None], np.nan, r)


def stage5_bootstrap_envelope(
    paths: dict[str, Path], fit_path: Path, thr: dict, subject: int, n_dirs: int = 180
) -> None:
    """Put our contours inside the paper's own bootstrap confidence interval.

    "Close enough" needs a yardstick, and the authors supply one. From their
    methods: they drew 120 bootstrap resamplings of the AEPsych trials
    (preserving the Sobol'/adaptive/fallback ratio), refit the WPPM to each,
    ranked the fits by summed Normalized Bures Similarity against the original
    fit, kept the top 114 (95% of 120), and defined the CI bounds as the
    **union and intersection of the retained threshold contours**.

    We reproduce that definition rather than inventing one:
      * the published ``Sigmas_thres_grid_btst{b}_rank{r}`` columns are already
        NBS-sorted, so "top 114" is ``rank < 114``;
      * union/intersection is a *radial* envelope -- per direction, the largest
        and smallest contour radius over the retained fits -- not a per-
        semi-axis percentile.

    A contour inside that band is indistinguishable from the authors' own
    resampling variability, which is a much stronger statement than "the
    ellipses look similar".
    """
    print("\n=== Stage 5: our thresholds vs the paper's bootstrap CI ===")

    if not fit_path.exists():
        print(f"  skipped: no saved fit at {fit_path}")
        return

    # --8<-- [start:bootstraps]
    path = paths["thres_ellipses"]
    coords, thres_published = hong2025.load_sigma_table(path)

    # Each bootstrap refit is another column of the file we already loaded, and
    # the authors already inverted them to threshold space for us. The columns
    # are named ..._btst{b}_rank{r}, sorted by NBS: rank 0 is the bootstrap fit
    # most similar to their main fit.
    with open(path, newline="") as fh:
        btst_cols = [c for c in csv.DictReader(fh).fieldnames or [] if "btst" in c]
    btst_cols.sort(key=lambda c: int(c.rsplit("rank", 1)[1]))
    boots_all = np.stack(
        [hong2025.load_sigma_table(path, value_column=c)[1] for c in btst_cols]
    )  # (120, 49, 2, 2)
    boots = boots_all[:N_BOOTSTRAP_CI]  # the paper's 95% CI set
    # --8<-- [end:bootstraps]

    print(f"  bootstrap refits : {len(boots_all)} published, top {len(boots)} kept")

    # Our end-to-end contours, from the weights stage 3 fit.
    W_fit = jnp.asarray(np.load(fit_path)["W"])
    model = hong2025.build_paper_model(mc_samples=thr["mc_samples"])
    thres_fit = np.asarray(
        WPPMPredictivePosterior(
            MAPPosterior({"W": W_fit}, model),
            jnp.asarray(coords),
            n_samples=1,
            threshold_pred=True,
            threshold_config=thr["config"],
        ).mean
    )

    # --8<-- [start:coverage]
    # The paper's bound is the union and intersection of the retained contours.
    # Sampled radially: per reference point and direction, how far out does the
    # outermost retained fit reach, and the innermost?
    theta = np.linspace(0.0, 2.0 * np.pi, n_dirs, endpoint=False)
    u = np.stack([np.cos(theta), np.sin(theta)], axis=1)  # (n_dirs, 2)

    r_boot = _radii(boots, u)  # (114, 49, n_dirs)
    r_ours = _radii(thres_fit, u)  # (49, n_dirs)
    # nanmax/nanmin so one unusable bootstrap cannot void a reference point.
    outer = np.nanmax(r_boot, axis=0)  # union of the retained contours
    inner = np.nanmin(r_boot, axis=0)  # intersection

    within = (r_ours >= inner) & (r_ours <= outer)  # (49, n_dirs); nan -> False
    fully_inside = within.all(axis=1)  # (49,)
    # --8<-- [end:coverage]

    n_in = int(fully_inside.sum())
    n_bad = int((~np.isfinite(r_ours)).any(axis=1).sum())
    if n_bad:
        print(
            f"  WARNING: {n_bad}/{len(coords)} of our threshold covariances were "
            "not positive definite and count as outside"
        )
    print(
        f"  inside the CI    : {n_in}/{len(coords)} reference points entirely, "
        f"{within.mean() * 100:.1f} % of all sampled directions"
    )
    # Reported alongside so the two numbers cannot be confused: the full spread
    # over all 120 is the looser band, and is NOT the paper's CI.
    r_all = _radii(boots_all, u)
    loose = ((r_ours >= r_all.min(axis=0)) & (r_ours <= r_all.max(axis=0))).all(axis=1)
    print(f"  (full 120 spread : {int(loose.sum())}/{len(coords)}, for reference)")

    M = _load_calibration()
    colors, fallback_note = _stimulus_colors(coords, M)
    scale = auto_scale(coords, thres_published)

    fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=150)

    # Layer 1: the CI set. Thin and nearly transparent so 114 fits read as a
    # band rather than 114 distinguishable curves. Labelled once.
    plot_ellipses(
        coords,
        boots,
        ax=ax,
        scale=scale,
        colors="0.55",
        linewidths=0.4,
        alpha=0.10,
        labels=[f"95% bootstrap CI (Hong et al. 2025, {_subject_tag(subject)})"]
        + [None] * (len(boots) - 1),
    )
    # Layer 2: the same convention as every other figure on the page.
    plot_ellipses(
        coords,
        [thres_published, thres_fit],
        ax=ax,
        scale=scale,
        colors=["black", colors],
        linestyles=["--", "solid"],
        linewidths=[2.2, 1.6],
        alpha=[0.35, None],
        labels=[
            _published_label(subject),
            _ours_label(subject, "our refit, then inversion"),
        ],
        show_centers=True,
    )

    ticks = np.linspace(-0.7, 0.7, 5)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlim(-0.95, 0.95)
    ax.set_ylim(-0.95, 0.95)
    ax.set_xlabel("Model Dimension 1")
    ax.set_ylabel("Model Dimension 2")
    ax.set_title(
        f"Our thresholds against the paper's 95% bootstrap CI — "
        f"{_subject_tag(subject)}" + fallback_note,
        fontsize=9,
    )
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    out_path = PLOTS_DIR / "hong2025_bootstrap_envelope.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=sorted(MODES), default="quick")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--noise-ellipses",
        action="store_true",
        default=True,
        help="download the 68 MB noise-covariance table needed by stage 1",
    )
    parser.add_argument(
        "--no-noise-ellipses",
        dest="noise_ellipses",
        action="store_false",
        help="skip stage 1 and the 68 MB download",
    )
    parser.add_argument(
        "--skip-thresholds",
        action="store_true",
        help="skip stage 2 (the Figure 2B threshold inversion, ~20 s on CPU)",
    )
    parser.add_argument(
        "--skip-refit",
        action="store_true",
        help=(
            "skip stage 3. Stages 1-2 reproduce published results on CPU; the "
            "stage-3 refit at paper settings is a GPU/cluster job."
        ),
    )
    parser.add_argument(
        "--from-fit",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "run stage 4 from an existing saved fit (.npz from a previous "
            "stage-3 run) instead of refitting. Implies --skip-refit."
        ),
    )
    parser.add_argument(
        "--skip-end-to-end",
        action="store_true",
        help="skip stage 4 (the end-to-end inversion of our own fitted weights)",
    )
    parser.add_argument(
        "--skip-envelope",
        action="store_true",
        help="skip stage 5 (our contours against the paper's 120 bootstrap fits)",
    )
    parser.add_argument(
        "--threshold-settings",
        choices=sorted(THRESHOLD_SETTINGS),
        default=None,
        help=(
            "inversion settings for stages 2 and 4. 'paper' matches Hong et al. "
            "(~11 min per stage on CPU) and is the default; 'fast' is ~20 s and "
            "is the default under --mode quick."
        ),
    )
    args = parser.parse_args()

    # Paper settings by default, so the committed figures and the numbers quoted
    # on the page come from the paper's own configuration. --mode quick falls
    # back to "fast" unless asked otherwise, so the smoke test stays seconds and
    # not ~22 min of inversion it was never meant to run.
    thr_name = args.threshold_settings or ("fast" if args.mode == "quick" else "paper")
    thr = THRESHOLD_SETTINGS[thr_name]

    print(f"device: {jax.devices()[0]}   x64: {jax.config.read('jax_enable_x64')}")
    print(
        f"threshold settings: {thr_name} "
        f"(n_theta={thr['config'].n_theta}, n_length={thr['config'].n_length}, "
        f"mc={thr['mc_samples']})"
    )

    # --8<-- [start:fetch]
    paths = hong2025.fetch(subject=args.subject, noise_ellipses=args.noise_ellipses)
    # --8<-- [end:fetch]

    stage1_exact_check(paths)
    if args.skip_thresholds:
        print("\n=== Stage 2: skipped (--skip-thresholds) ===")
    else:
        stage2_thresholds(paths, thr, args.subject)
    fit_path = args.from_fit or FITS_DIR / f"hong2025_{args.mode}_fit.npz"

    if args.skip_refit or args.from_fit:
        reason = "--from-fit" if args.from_fit else "--skip-refit"
        print(f"\n=== Stage 3: skipped ({reason}) ===")
    else:
        if args.mode == "full" and jax.devices()[0].platform == "cpu":
            print(
                "\n  WARNING: --mode full on CPU. The paper's settings are a "
                "GPU/cluster job\n  (~16 min on one GPU; far longer here). "
                "Ctrl-C and pass --mode quick to smoke-test."
            )
        fit_path = stage3_refit(
            paths, MODES[args.mode], args.mode, args.seed, args.subject
        )

    if args.skip_end_to_end:
        print("\n=== Stage 4: skipped (--skip-end-to-end) ===")
    else:
        stage4_end_to_end(paths, fit_path, args.mode, thr, args.subject)

    if args.skip_envelope:
        print("\n=== Stage 5: skipped (--skip-envelope) ===")
    else:
        stage5_bootstrap_envelope(paths, fit_path, thr, args.subject)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
