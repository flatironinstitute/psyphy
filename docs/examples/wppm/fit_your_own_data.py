"""
Fit your own data
-----------------

Companion script for ``docs/examples/wppm/fit_your_own_data.md``.

We read oddity-task trials from a CSV file into a TrialData, fit a WPPM by MAP
optimization, and check the fit with no ground truth: the learning curve, a
second fit from another prior sample, and predicted against observed
proportion correct on held-out trials. The script reads the number of stimulus
dimensions from the data, so the same code fits 2-D and 3-D stimuli.

The data are one observer's published trials from Hong et al. (observer CH,
12,000 trials), downloaded from OSF on first use (about 1 MB). We parse the
file with numpy as we would any CSV. The published weights come in only for
the comparison at the end.

Install::

    pip install 'psyphy[viz]'

Usage::

    python fit_your_own_data.py --mode quick   # smoke test, ca. 1 min on a laptop CPU, doesn't converge
    python fit_your_own_data.py --mode full    # Hong et al.'s Monte Carlo settings, needs a GPU

Figures go to ``plots/`` next to this script.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
# --8<-- [start:imports]
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from psyphy import (  # noqa: E402
    WPPM,
    GaussianNoise,
    MAPOptimizer,
    OddityTask,
    OddityTaskConfig,
    Prior,
    TrialData,
)

# --8<-- [end:imports]
from psyphy.data.published import hong2025  # noqa: E402
from psyphy.model import WPPMCovarianceField  # noqa: E402
from psyphy.posterior import (  # noqa: E402
    MAPPosterior,
    ThresholdConfig,
    WPPMPredictivePosterior,
)
from psyphy.viz import plot_ellipses  # noqa: E402

PLOTS_DIR = Path(__file__).parent / "plots"

# --8<-- [start:modes]
MODES = {
    "quick": {"steps": 300, "mc_fit": 20, "mc_eval": 200},  # smoke test, not converged
    "full": {"steps": 1500, "mc_fit": 2000, "mc_eval": 2000},  # Hong et al.'s settings
}
# --8<-- [end:modes]


def rel_frobenius(A, B):
    """Relative Frobenius distance ||A - B|| / ||B|| per point, shape (n_points,)."""
    return np.linalg.norm(A - B, axis=(1, 2)) / np.linalg.norm(B, axis=(1, 2))


def main(mode: str) -> None:
    cfg = MODES[mode]
    print(f"mode={mode} {cfg}, device {jax.devices()[0]}")
    PLOTS_DIR.mkdir(exist_ok=True)
    paths = hong2025.fetch(subject=1)  # cached after the first call
    csv_path = paths["trials"]

    # ---- 1. Data ----
    # --8<-- [start:load_csv]
    # columns TrialType, xref, x1, y; a coordinate is one quoted field: "0.62,-0.67"
    rows = np.loadtxt(csv_path, dtype=str, delimiter=",", quotechar='"', skiprows=1)
    ref = np.array([s.split(",") for s in rows[:, 1]], dtype=float)  # (N, d)
    comparison = np.array([s.split(",") for s in rows[:, 2]], dtype=float)  # (N, d)
    correct = rows[:, 3].astype(int)  # 1 = correct, 0 = incorrect
    held_out = np.char.startswith(rows[:, 0], "MOCS")  # not fitted, for check (c)
    # --8<-- [end:load_csv]

    # --8<-- [start:trial_data]
    def to_trial_data(keep):
        return TrialData(
            stimuli=jnp.asarray(
                np.stack([ref[keep], comparison[keep]], axis=1)
            ),  # (N, 2, d): slot 0 ref, slot 1 comparison
            responses=jnp.asarray(correct[keep][:, None]),  # (N, 1)
        )

    data = to_trial_data(~held_out)  # the AEPsych trials, which Hong et al. fitted
    test = to_trial_data(held_out)  # the MOCS trials
    assert np.abs(data.stimuli).max() <= 1, "rescale the stimuli to [-1, 1] first"
    # --8<-- [end:trial_data]
    print(f"fit on {data.num_trials} trials, hold out {test.num_trials}")

    # ---- 2. Model ----
    # --8<-- [start:model]
    d = data.stimuli.shape[-1]  # number of stimulus dimensions, 2 here
    prior = Prior(
        input_dim=d,
        basis_degree=4,  # Chebyshev degrees 0 to 4 in each dimension
        extra_embedding_dims=1,
        variance_scale=3e-4,
        decay_rate=0.4,
    )
    model = WPPM(
        prior=prior,
        likelihood=OddityTask(
            config=OddityTaskConfig(num_samples=cfg["mc_fit"], bandwidth=5e-3)
        ),
        noise=GaussianNoise(),  # required: without it, fit raises
    )
    # --8<-- [end:model]

    # ---- 3. Fit, twice, from 2 prior samples ----
    t0 = time.time()
    # --8<-- [start:fit]
    optimizer = MAPOptimizer(steps=cfg["steps"], learning_rate=1e-4, momentum=0.2)
    posteriors, losses = [], []
    for seed in (0, 1):  # seed picks the prior sample the fit starts from
        posteriors.append(optimizer.fit(model=model, data=data, seed=seed))
        losses.append(
            np.array(optimizer.get_history()[1])
        )  # per-trial loss at each step
    posterior = posteriors[0]
    # --8<-- [end:fit]
    print(f"fit: {cfg['steps']} steps x 2 fits, {time.time() - t0:.0f} s")
    for seed, loss in zip((0, 1), losses):
        print(
            f"  seed {seed}: loss first 100 steps {loss[:100].mean():.4f}, "
            f"last 100 {loss[-100:].mean():.4f}"
        )

    # ---- 4. Results ----
    # --8<-- [start:field]
    g = np.linspace(-0.7, 0.7, 7)
    refs = np.stack(np.meshgrid(*[g] * d, indexing="ij"), axis=-1).reshape(
        -1, d
    )  # (7**d, d)
    field = WPPMCovarianceField(posterior)
    # field(refs): (1, 7**d, d, d), 1 parameter set after a MAP fit; [0] drops that axis
    covs = np.asarray(
        field(jnp.asarray(refs))[0]
    )  # (7**d, d, d): Sigma(x) at each reference
    # --8<-- [end:field]

    # More MC samples for predictions than for the fit: less noise in P(correct)
    # --8<-- [start:eval_model]
    eval_model = WPPM(
        prior=prior,
        likelihood=OddityTask(
            config=OddityTaskConfig(num_samples=cfg["mc_eval"], bandwidth=5e-3)
        ),
        noise=GaussianNoise(),
    )
    fits = [MAPPosterior(model=eval_model, params=p.params) for p in posteriors]
    # --8<-- [end:eval_model]

    # --8<-- [start:thresholds]
    # directions probed per reference, raise it as d grows
    n_theta = 16 if d == 2 else 64
    thresholds = []
    for fit in fits:
        predictive = WPPMPredictivePosterior(
            param_posterior=fit,
            X=jnp.asarray(refs),  # (n_refs, d): psyphy searches for the comparisons
            n_samples=1,  # a MAP fit is a single point estimate
            threshold_pred=True,
            threshold_config=ThresholdConfig(n_theta=n_theta, n_length=300),
            key=jax.random.PRNGKey(0),
        )
        thresholds.append(np.asarray(predictive.mean))  # (n_refs, d, d), 66.7% correct
    # --8<-- [end:thresholds]

    # ---- 5. Checks ----
    # --8<-- [start:two_fits]
    diff = rel_frobenius(thresholds[1], thresholds[0])  # (n_refs,)
    print(f"seed 1 against seed 0: median {np.median(diff):.3f}, max {diff.max():.3f}")
    # --8<-- [end:two_fits]

    # --8<-- [start:held_out]
    predictive = WPPMPredictivePosterior(
        fits[0], test.stimuli, n_samples=1, key=jax.random.PRNGKey(1)
    )
    p_pred = np.asarray(predictive.mean)  # (n_test,) predicted P(correct) per trial
    observed = np.asarray(test.responses[:, 0])  # (n_test,)
    edges = np.quantile(
        p_pred, np.linspace(0, 1, 11)
    )  # 10 bins, same number of trials each
    bin_of = np.digitize(p_pred, edges[1:-1])  # bin index 0..9 per trial
    pred_bin = np.array([p_pred[bin_of == b].mean() for b in range(10)])
    obs_bin = np.array([observed[bin_of == b].mean() for b in range(10)])
    # --8<-- [end:held_out]
    n_bin = np.array([(bin_of == b).sum() for b in range(10)])
    se_bin = np.sqrt(obs_bin * (1 - obs_bin) / n_bin)
    off = np.abs(obs_bin - pred_bin) / se_bin
    print(f"held out: |observed - predicted| / SE per bin, max {off.max():.1f}")

    # ---- Box: compare with a known field ----
    # --8<-- [start:truth]
    W_published = hong2025.load_reference_W(paths["weights"])  # (5, 5, 2, 3)
    covs_published = np.asarray(
        WPPMCovarianceField.from_params(model, {"W": W_published})(jnp.asarray(refs))[0]
    )  # (n_refs, d, d)
    error = rel_frobenius(covs, covs_published)  # (n_refs,)
    print(
        f"against the published field: median {np.median(error):.3f}, max {error.max():.3f}"
    )
    # --8<-- [end:truth]

    # ---- Figures ----
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for seed, loss in zip((0, 1), losses):
        ax.plot(loss, color=f"C{seed}", lw=0.6, alpha=0.8, label=f"seed {seed}")
    ax.set_xlabel("step")
    ax.set_ylabel("negative log posterior / N")
    ax.legend(title="fit from", frameon=False)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "own_data_learning_curve.png", dpi=150)

    # --8<-- [start:plot]
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    plot_ellipses(
        refs,  # (n_refs, 2)
        thresholds,  # 2 fits, each (n_refs, 2, 2)
        ax=ax,
        colors=["C0", "C1"],
        linestyles=["-", "--"],
        labels=["threshold, fit from seed 0", "threshold, fit from seed 1"],
    )
    # --8<-- [end:plot]
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1, frameon=False)
    ax.set_aspect("equal")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "own_data_thresholds.png", dpi=150)

    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    ax.plot([1 / 3, 1], [1 / 3, 1], color="gray", lw=0.8, label="perfect prediction")
    ax.errorbar(
        pred_bin,
        obs_bin,
        yerr=2 * se_bin,
        fmt="o",
        color="C0",
        ms=4,
        label="1 bin of held-out trials\n(error bar: 2 standard errors)",
    )
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.set_xlabel("predicted P(correct)")
    ax.set_ylabel("observed proportion correct")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "own_data_held_out.png", dpi=150)
    print(f"figures in {PLOTS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--mode", choices=list(MODES), default="quick")
    main(parser.parse_args().mode)
