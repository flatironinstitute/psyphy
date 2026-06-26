"""
Weber's Law Recovery with 1D WPPM
----------------------------------

Demonstrates that a 1D WPPM can recover Weber's law purely from
binary MC-based oddity-task responses: Weber's law is the finding that the
just-noticeable difference (JND) is a constant fraction of the stimulus level.

The core claim
------------
Simulate binary oddity responses from a 1D ground-truth observer whose internal
variance satisfies Sigma(s) = (k*s)^2, ie the Weber Law.  Fit a 1D WPPM
to those responses. If the fitted model recovers a variance function
whose implied threshold scales linearly with stimulus magnitude,
the Wishart process is flexible enough to represent Weber's law in 1D.

sqrt(Sigma(s)) as a JND proxy
------------------------------------------
sqrt(Sigma(s)) is the noise standard deviation at stimulus level s; the scale
of trial-to-trial perceptual fluctuations. Weber's law predicts this grows
linearly with s. We test linearity directly; any unknown constant of
proportionality between noise SD and behavioral threshold cancels in that test.

Weber's law in the WPPM representation
---------------------------------------
Weber's law: JND(s) = k*s, i.e., Sigma(s) = (k*s)^2, where Sigma(s) is the variance.
Since Sigma = U*U^T, Weber requires U(s) = k*s; linear in s, and therefore
also linear in the normalized coordinate x (degree 1 in the Chebyshev basis).

Basis degree of Chebychev
--------------------
The WPPM Chebyshev basis is evaluated in normalized x in [-1, 1], not physical s.
basis_degree is the index of the highest Chebyshev polynomial used; since T_n
has polynomial degree n, it also equals the polynomial degree of U:

  basis_degree=1  ->  U(x) = W_0*T_0(x) + W_1*T_1(x) = W_0 + W_1*x  [linear in x]
                  ->  Sigma(x) = U(x)^2 = (W_0 + W_1*x)^2             [degree 2 in x]


Two free parameters (W shape (2, 1, 1) with extra_dims=0): W_1 sets the Weber
slope k and W_0 a constant noise floor (generalised law JND = k*s + c).
The prior regularizes W_0 toward zero, so the model defaults to pure Weber but
can deviate when data demand it.

Coordinate system, normalization, and domain
---------------------------------------------
WPPM does NOT normalize inputs. It enforces x in [-1, 1] and raises ValueError
otherwise. The Chebyshev polynomials are orthogonal on [-1, 1]; using a
sub-interval wastes basis capacity and mis-calibrates the smoothness prior.

The Chebyshev domain must span ALL stimuli that appear (both references and
comparisons). Comparisons extend above the reference range: at reference s_ref
with jnd_multiples JNDs of displacement, s_comp = s_ref*(1 + jnd_multiples*K_WEBER).
We therefore set S_MAX = 2.0 so that the worst-case comparison
(S_MAX_REF*(1 + max_jnd_multiples*K_WEBER) = 1.0*1.8 = 1.8) stays inside the domain.
References are sampled from [S_MIN, S_MAX_REF] = [0.2, 1.0].

We normalize all physical stimuli using the full domain [S_MIN, S_MAX]:
    x = 2*(s - S_MIN)/(S_MAX - S_MIN) - 1

WeberGroundTruth receives the same normalized x and converts back to physical s
internally, so simulation and fitting share one coordinate system.

Analysis grids and plots are restricted to [S_MIN, S_MAX_REF] = [0.2, 1.0]
where the model is well-determined by training data.

Plots (5 panels + learning curve)
------------------------------------
  1. Trial data scatter ; raw binary responses; x=reference level s,
                           y=displacement delta (both in physical units).
  2. JND recovery       ; sqrt(Sigma_hat(s)) vs s: should be proportional to k*s.
  3. Weber fraction     ; JND(s)/s vs s: should be flat at k.
  4. Fechner's law      ; integral of 1/JND: should follow log(s).
  5. Psychometric curves; p(correct) vs delta for three reference levels;
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

# --8<-- [start:imports]
from psyphy.data import TrialData
from psyphy.inference import MAPOptimizer
from psyphy.model import WPPM, GaussianNoise, OddityTask, Prior, WPPMCovarianceField
from psyphy.model.likelihood import OddityTaskConfig

# --8<-- [end:imports]

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")

# ---------------------------------------------------------------------------
# Toggle: save each panel as its own PNG in addition to the combined figure
# ---------------------------------------------------------------------------
SAVE_INDIVIDUAL_PANELS = True

# Toggle: run the basis-degree sweep with held-out likelihood (Step 6).
# This refits the model several times, so it is the slow part of the demo.
RUN_BASIS_SWEEP = True

print("DEVICE:", jax.devices()[0])

# ---------------------------------------------------------------------------
# Physical stimulus range and coordinate transforms
# ---------------------------------------------------------------------------
# S_MIN/S_MAX define the Chebyshev domain (all stimuli: refs AND comparisons).
# S_MAX_REF is the upper end of the reference range; comparisons can exceed it.
# Constraint: S_MAX >= S_MAX_REF * (1 + max_jnd_multiples * K_WEBER) to avoid extrapolation.
# The WPPM Chebyshev basis requires inputs in [-1, 1]; we map explicitly.

# --8<-- [start:domain]
S_MIN: float = 0.2  # domain minimum (also minimum reference; must be > 0 for Weber)
S_MAX: float = 2.0  # domain maximum; wide enough to cover all comparisons,
# which are an offset from reference stimulus.
# worst-case: S_MAX_REF*(1 + max_jnd_multiples*K_WEBER) = 1.0*1.8 = 1.8 < 2.0.
S_MAX_REF: float = (
    1.0  # maximum reference stimulus; references sampled from [S_MIN, S_MAX_REF].
)


def to_norm(s: jnp.ndarray) -> jnp.ndarray:
    """Map physical s (anywhere in [S_MIN, S_MAX]=[0.2, 2.0]) -> normalized x in [-1, 1].
    The domain covers both references [S_MIN, S_MAX_REF] and comparisons up to S_MAX."""
    return 2.0 * (s - S_MIN) / (S_MAX - S_MIN) - 1.0


def to_phys(x: jnp.ndarray) -> jnp.ndarray:
    """Map normalized x in [-1, 1] -> physical s in [S_MIN, S_MAX] = [0.2, 2.0]."""
    return 0.5 * (x + 1.0) * (S_MAX - S_MIN) + S_MIN


# --8<-- [end:domain]


# ---------------------------------------------------------------------------
# Ground-truth model: Weber's law  Sigma(s) = (k*s)^2,  in normalized coords
# ---------------------------------------------------------------------------
# WeberGroundTruth receives normalized x in [-1, 1] (same as the WPPM).
# It converts back to physical s internally to compute the covariance.
# This ensures simulation and fitting use the same coordinate system.


# --8<-- [start:ground_truth]
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
        self.diag_term = (
            DIAG_TERM  # same jitter as the fitted WPPM (see DIAG_TERM constant).
        )
        self.noise = GaussianNoise(sigma=0.0)
        self.basis_degree = BASIS_DEGREE  # required by OddityTask._simulate_trial_mc

    def _compute_sqrt(self, _params, x: jnp.ndarray) -> jnp.ndarray:
        # _params unused: WeberGroundTruth has no learned parameters.
        # x is normalized in [-1, 1]; convert to physical s for Weber's law
        s = to_phys(x[0])  # scalar physical stimulus
        sigma = self.k * s  # sqrt(Sigma) = k*s  (Weber)
        embedding_dim = self.input_dim + self.extra_dims
        return sigma * jnp.eye(self.input_dim, embedding_dim)


# --8<-- [end:ground_truth]


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

# --8<-- [start:settings]
K_WEBER = 0.2  # Weber fraction (ground truth)
DIAG_TERM = 1e-6  # numerical stability jitter; shared by WeberGroundTruth AND the
# fitted WPPM so the generative model and the fitting model have
# identical parameterisations. Removing from WPPM risks NaN gradients
# when U(x) \approx 0 during early optimisation
N_TRIALS = 2000
MC_SAMPLES = 1000  # MC samples for simulation and fitting

# degree-1 U(x) = W_0 + W_1*x; sufficient for Weber (linear U -> quadratic Sigma).
# W shape (2, 1, 1) = 2 parameters with extra_dims=0. See module docstring.
BASIS_DEGREE = 1

NUM_STEPS = 500  # 1000 gets close for 3 param model (0 additional embedding dims)
LEARNING_RATE = 5e-4
# --8<-- [end:settings]

# Reference levels for the psychometric function panel (in physical units)
PSYCH_LEVELS_PHYS = [0.3, 0.6, 0.9]
PSYCH_COLORS = ["#2166ac", "#d6604d", "#4dac26"]

# ---------------------------------------------------------------------------
# Unified axis label constants
# ---------------------------------------------------------------------------
# All panels that share a quantity on an axis use the same label string so
# that readers can immediately identify what is being shown.
#
#   LABEL_S    ; x-axis of panels 1–4: the reference stimulus level
#   LABEL_DELTA; x-axis of panel 5 / y-axis of panel 1: comparison offset
#   LABEL_JND  ; y-axis of panel 2: WPPM-implied JND proxy
#   LABEL_WF   ; y-axis of panel 3: Weber fraction (JND / s)
#   LABEL_PSI  ; y-axis of panel 4: Fechner perceived magnitude
#   LABEL_P    ; y-axis of panel 5: probability correct

LABEL_S = rf"Reference level  $s_\mathrm{{ref}} \in$  [{S_MIN}, {S_MAX_REF}]"
LABEL_DELTA = r"Stimulus difference  $\delta = s_\mathrm{comp} - s_\mathrm{ref}$"
LABEL_JND = r"JND proxy  $\sqrt{\hat{\Sigma}(s_\mathrm{ref})}$"
LABEL_WF = r"Weber fraction  $\sqrt{\hat{\Sigma}(s_\mathrm{ref})}\,/\,s_\mathrm{ref}$"
LABEL_PSI = r"Perceived magnitude  $\hat{\psi}(s_\mathrm{ref})$  [normalized]"
LABEL_P = r"$P(\mathrm{correct})$"

# ---------------------------------------------------------------------------
# Step 1; Simulate Weber's law data
# ---------------------------------------------------------------------------

print("[1/5] Simulating oddity-task data from WeberGroundTruth ...")

# --8<-- [start:simulate]
weber_gt = WeberGroundTruth(k=K_WEBER)
task = OddityTask(config=OddityTaskConfig(num_samples=MC_SAMPLES))

key = jr.PRNGKey(0)
key_refs, key_radii, key_sim = jr.split(key, 3)

# Sample reference stimuli uniformly in physical units, then normalize
refs_s = jr.uniform(key_refs, (N_TRIALS,), minval=S_MIN, maxval=S_MAX_REF)  # physical
refs_x = to_norm(refs_s)  # normalized, passed to models
refs = refs_x[:, None]  # (N, 1) for WPPM / OddityTask

# jnd_multiples: how many JNDs each comparison is displaced from its reference.
# delta = jnd_multiples * JND(s) = jnd_multiples * k * s  (in physical units)
# Ranging from 0.5–4 JNDs ensures the scatter covers both hard and easy trials.
# Displacements are one-sided (comp = ref + delta, delta > 0); direction does not
# matter in 1D because only |delta|/sqrt(Sigma) enters the decision.
jnd_multiples = jr.uniform(key_radii, (N_TRIALS,), minval=0.5, maxval=4.0)
delta_s = (
    jnd_multiples * K_WEBER * refs_s
)  # physical displacement: jnd_multiples * JND(s)

# Comparisons in normalized coordinates (WPPM and OddityTask expect normalized x)
comps_x = to_norm(refs_s + delta_s)
comparisons = comps_x[:, None]  # (N, 1)

# Simulate binary responses via the MC-based oddity decision process.
# WeberGroundTruth receives normalized x; internally converts to physical s.
stimuli = jnp.stack(
    [refs, comparisons], axis=1
)  # (N, 2, 1): new API requires pre-stacked
responses, prob_params = task.simulate(
    params=None, stimuli=stimuli, model=weber_gt, key=key_sim
)
p_correct_sim = prob_params[0]  # simulate returns (responses, (p_correct, ...))
data = TrialData(stimuli=stimuli, responses=responses, stimulus_names=("ref", "comp"))
# --8<-- [end:simulate]

print(
    f"  {N_TRIALS} trials simulated, "
    f"mean p(correct) = {float(p_correct_sim.mean()):.3f}"
)

# ---------------------------------------------------------------------------
# Step 2; Build and fit the 1D WPPM
# ---------------------------------------------------------------------------
BASIS_DEGREE_FIT = 3
print("[2/5] Fitting 1D WPPM via MAPOptimizer ...")

# --8<-- [start:fit]
prior = Prior(input_dim=1, basis_degree=BASIS_DEGREE_FIT, extra_embedding_dims=0)
model = WPPM(
    input_dim=1,
    extra_dims=0,  # embedding_dim = input_dim (minimal)
    prior=prior,
    likelihood=task,
    noise=GaussianNoise(sigma=0.0),
    diag_term=DIAG_TERM,  # matches WeberGroundTruth for a fair comparison
)

init_params = model.init_params(jr.PRNGKey(1))

optimizer = MAPOptimizer(
    steps=NUM_STEPS,
    learning_rate=LEARNING_RATE,
    track_history=True,
    log_every=1,
)

map_posterior = optimizer.fit(model, data, init_params=init_params, seed=2)
# --8<-- [end:fit]
print("  Fitting done.")

# ---------------------------------------------------------------------------
# Step 3; Derived quantities (computed in physical units for interpretability)
# ---------------------------------------------------------------------------

print("[3/5] Computing derived quantities ...")

# --8<-- [start:jnd]
# Dense grid over the reference range for analysis and plotting.
# Comparisons may extend above S_MAX_REF, but we only evaluate the sub-range
# [S_MIN, S_MAX_REF] where the model is well-determined by training data.
s_grid = jnp.linspace(S_MIN, S_MAX_REF, 300)  # physical, reference range
x_grid = to_norm(s_grid)[:, None]  # normalized, shape (300, 1)

# Bind fitted parameters to the model: creates a callable x -> Sigma(x).
fitted_cov_fn = WPPMCovarianceField(model, map_posterior.params)
variances = fitted_cov_fn(x_grid)  # (300, 1, 1); 1x1 "matrix" per grid point;
# [:, 0, 0] extracts the scalar Sigma(s)
# sqrt(Sigma) from WPPM; proxy for implied JND in physical units
# (valid in the 1D equal-variance limit; see module docstring)
jnd_fitted = jnp.sqrt(variances[:, 0, 0])
jnd_truth = K_WEBER * s_grid  # ground truth: k*s

# Weber fraction: JND(s)/s; should be flat at K_WEBER
weber_fraction_fitted = jnd_fitted / s_grid
# --8<-- [end:jnd]

# ---------------------------------------------------------------------------
# Weight analysis: exact analytical basis change; Chebyshev(x) -> monomial(s).
# ---------------------------------------------------------------------------
# The normalization x = a*s + b is an invertible affine map.  Because T_n(x(s))
# is a degree-n polynomial in s, the Chebyshev representation U(x) = Σ Wₙ Tₙ(x)
# and the physical monomial representation U(s) = Σ cₙ sⁿ span the same space.
# The conversion is a pure linear basis change; no fitting, no approximation:
#   Step 1: cheb2poly converts W -> monomial coefficients in normalized x.
#   Step 2: substituting x = a*s + b (polynomial composition) maps those to s.
W_raw = np.asarray(map_posterior.params["W"]).reshape(-1)  # shape (BASIS_DEGREE+1,)

_a = 2.0 / (S_MAX - S_MIN)  # slope of x(s)
_b = -2.0 * S_MIN / (S_MAX - S_MIN) - 1.0  # intercept of x(s)

mono_in_x = np.polynomial.chebyshev.cheb2poly(W_raw)  # U(x) in monomial basis

phys_coeffs = np.zeros(len(mono_in_x))
for k, ck in enumerate(mono_in_x):
    # (b + a*s)^k as a polynomial in s (ascending coefficient order)
    term = np.polynomial.polynomial.polypow([_b, _a], k)
    phys_coeffs[: len(term)] += ck * term
# phys_coeffs[n] = coefficient of s^n in U(s), for n = 0 … BASIS_DEGREE

print(
    "  Weight analysis; physical coefficients cn of U(s) = sum_n cn*s^n:\n"
    + "".join(
        f"    c_{n} = {phys_coeffs[n]:.4f}"
        + (
            "  <- Weber slope (truth 0.2000)"
            if n == 1
            else "  <- noise floor (truth 0.0000)"
            if n == 0
            else "  <- should be \approx 0"
        )
        + "\n"
        for n in range(len(phys_coeffs))
    )
)

# --8<-- [start:fechner]
# Fechner's law: psi(s) = integral_{S_MIN}^{s} 1/JND(s') ds'
# When JND = k*s, psi = (1/k)*log(s/S_MIN) (the logarithmic magnitude scale)
ds = float(s_grid[1] - s_grid[0])
psi_fitted = jnp.cumsum(1.0 / jnd_fitted) * ds
psi_truth = jnp.cumsum(1.0 / jnd_truth) * ds
psi_log = jnp.log(s_grid / s_grid[0])


def _norm01(x):
    # Affine rescaling to [0, 1] for visual comparison. sqrt(Sigma) is proportional
    # to JND (not equal), so absolute scale is arbitrary. Normalization isolates the
    # shape of the sensation curve (logarithmic growth) from the scale factor d'.
    return (x - x[0]) / (x[-1] - x[0])


psi_fitted = _norm01(psi_fitted)
psi_truth = _norm01(psi_truth)
psi_log = _norm01(psi_log)
# --8<-- [end:fechner]

# Psychometric functions in physical units
# Use 500 MC samples for smooth curves (evaluation only, not fitting)
key_psych = jr.PRNGKey(42)
n_delta = 80
task_smooth = OddityTask(config=OddityTaskConfig(num_samples=500))
psych_data = {}

for s_ref in PSYCH_LEVELS_PHYS:
    jnd_gt = K_WEBER * s_ref
    delta_sweep = jnp.linspace(0.01 * jnd_gt, 4.0 * jnd_gt, n_delta)  # physical

    # Convert to normalized coordinates for model calls
    refs_psych = to_norm(jnp.full(n_delta, s_ref))[:, None]  # (n_delta, 1)
    comps_psych = to_norm(jnp.full(n_delta, s_ref) + delta_sweep)[
        :, None
    ]  # (n_delta, 1)
    stims_psych = jnp.stack([refs_psych, comps_psych], axis=1)  # (n_delta, 2, 1)

    # --8<-- [start:psychometric]
    # predict returns (p_correct,); vmap gives (array,); [0] extracts the array
    # p_fit: psychometric curve from the *fitted* WPPM; p_gt: from the ground truth.
    p_fit = jax.vmap(
        lambda stim: task_smooth.predict(
            map_posterior.params, stim, model, key=key_psych
        )
    )(stims_psych)[0]

    p_gt = jax.vmap(
        lambda stim: task_smooth.predict(None, stim, weber_gt, key=key_psych)
    )(stims_psych)[0]
    # --8<-- [end:psychometric]

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
        label=r"Ground truth: $\sqrt{\Sigma(s)} = k \cdot s$",
    )
    ax.set_xlabel(LABEL_S)
    ax.set_ylabel(LABEL_DELTA)
    ax.set_title("Raw trial data\nTrials above JND line -> mostly correct")
    ax.legend(markerscale=2.5, fontsize=8)
    ax.grid(True, alpha=0.25)


def draw_jnd_recovery(ax):
    """Panel 2: JND curve; sqrt(Sigma_hat(s)) vs s in physical units.

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
    ax.set_title(r"JND recovery: $\sqrt{\hat{\Sigma}(s)}$ vs $s$")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)


def draw_weber_fraction(ax):
    """Panel 3: Weber fraction JND(s)/s; flat = Weber's law holds."""
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


def draw_weber_coeffs_raw(ax):
    """Left panel: raw Chebyshev coefficients W_0 … W_degree in normalized-x space.

    U(x) = sum_n W_n * T_n(x), x in [-1, 1].
    W_1 (linear term) should dominate; W_2, W_3, ... should be near zero,
    showing the model didn't use its extra flexibility.

    Note: W_0 is large even for pure Weber's law; not because of a noise floor,
    but because x=0 maps to the middle of the physical domain (s=1.1), not s=0.
    The physically interpretable decomposition is in the right panel.
    """
    degree = len(W_raw) - 1
    poly_names = ["constant", "linear", "quadratic", "cubic", "quartic", "quintic"]

    labels = []
    for n in range(degree + 1):
        poly = poly_names[n] if n < len(poly_names) else f"degree {n}"
        labels.append(f"$W_{n}$\n{poly}\n($T_{n}$)")

    colors = ["#c0392b" if n == 1 else "#aaaaaa" for n in range(degree + 1)]
    ax.bar(labels, W_raw.tolist(), color=colors, width=0.6)
    ax.axhline(0.0, color="gray", linewidth=0.8)
    ax.set_ylabel("Coefficient (normalized-$x$ units)")
    ax.set_title(
        f"Raw Chebyshev weights  $W_0 \\ldots W_{{{degree}}}$\n"
        r"Higher-order terms $\approx 0$: extra flexibility unused"
    )
    ax.grid(True, axis="y", alpha=0.25)


def draw_weber_coeffs_physical(ax):
    """Right panel: polynomial coefficients of U(s) in physical stimulus units.

    Obtained by fitting a degree-BASIS_DEGREE polynomial to sqrt(Sigma_hat(s))
    vs physical s. Coefficients are ordered ascending: c_0 (constant), c_1
    (linear), c_2 (quadratic), ...

    For Weber's law, c_1 should dominate (\approx K_WEBER) and all other terms
    should be near zero; showing the model recovered a linear law despite
    having the flexibility to fit higher-order curves.
    """
    degree = len(phys_coeffs) - 1
    poly_names = ["constant", "linear", "quadratic", "cubic", "quartic", "quintic"]

    labels = []
    for n in range(degree + 1):
        poly = poly_names[n] if n < len(poly_names) else f"degree {n}"
        labels.append(f"$c_{n}$\n{poly}\n($s^{n}$)")

    colors = ["#c0392b" if n == 1 else "#aaaaaa" for n in range(degree + 1)]
    ax.bar(labels, phys_coeffs.tolist(), color=colors, width=0.6)
    ax.axhline(
        K_WEBER,
        color="k",
        linestyle="--",
        linewidth=2,
        label=f"ground truth $k = {K_WEBER}$",
    )
    ax.axhline(0.0, color="gray", linewidth=0.8)
    ax.set_ylabel("Coefficient (physical units)")
    ax.set_title(
        f"Physical polynomial coefficients  $c_0 \\ldots c_{{{degree}}}$\n"
        r"$c_1$ (linear / Weber) should dominate; others $\approx 0$"
    )
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)


def save_weber_coeffs_comparison():
    """Save a 2-panel figure comparing raw Chebyshev weights and physical recovery."""
    fig_c, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 4.5))
    draw_weber_coeffs_raw(ax_left)
    draw_weber_coeffs_physical(ax_right)
    # fig_c.suptitle(
    #     # f"Can a degree-{BASIS_DEGREE_FIT} model recover Weber's law?  "
    #     # f"(k={K_WEBER}, N={N_TRIALS} trials)\n"
    #     # "Left: what the model learned in basis space.  "
    #     # "Right: what it means in physical units.",
    #     # fontsize=10,
    # )
    fig_c.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, "weber_coeffs_comparison.png")
    fig_c.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig_c)
    print(f"  Saved coefficient comparison -> {path}")


def draw_fechner(ax):
    """Panel 4: Fechner's law; cumulative integral of 1/JND(s) gives log scale."""
    ax.plot(s_grid, psi_log, "k--", linewidth=2, label=r"$\log(s/s_0)$  (Fechner)")
    ax.plot(
        s_grid,
        psi_truth,
        color="#888888",
        linewidth=1.5,
        linestyle=":",
        label=r"Ground truth $\int 1/\sqrt{\Sigma_\mathrm{gt}}\,ds$",
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
    ax.set_title(r"Fechner's law  ($\int 1/\mathrm{JND}(s)\,ds$ = log scale)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)


def draw_psychometric(ax):
    """Panel 5: psychometric functions; p(correct) vs delta (physical units).

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

    ax.axhline(1 / 3, color="gray", linewidth=0.9, linestyle=":")
    ax.axhline(1.0, color="gray", linewidth=0.7, linestyle=":", alpha=0.5)
    ax.set_ylim(0.0, 1.1)  # show full [0, 1] range; ceiling at 1.0 is meaningful
    ax.set_xlabel(LABEL_DELTA)
    ax.set_ylabel(LABEL_P)
    ax.set_title(
        "Psychometric functions \n"
        r"Curves shift right as $s$ grows $\rightarrow$ Weber's law"
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
        Line2D(
            [0],
            [0],
            color="gray",
            linewidth=0.9,
            linestyle=":",
            label="Chance (1/3, 3-AFC oddity)",
        ),
    ]
    ax.legend(handles=level_handles + style_handles, fontsize=7, ncol=2)
    ax.grid(True, alpha=0.25)


# ---------------------------------------------------------------------------
# Step 4; Combined 5-panel figure
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
# --8<-- [start:learning_curve]
steps_hist, loss_hist = optimizer.get_history()  # (step indices, neg-log-posterior)
# --8<-- [end:learning_curve]
if steps_hist:
    ax6.plot(steps_hist, loss_hist, color="#4444aa", linewidth=1.5)
    ax6.set_xlabel("Optimisation step")
    ax6.set_ylabel("Neg log posterior")
    ax6.set_title("Learning curve")
    ax6.grid(True, alpha=0.25)

fig.suptitle(
    f"1D WPPM: Weber's Law Recovery  "
    f"(k={K_WEBER}, N={N_TRIALS} trials, basis_degree={BASIS_DEGREE_FIT})\n"
    f"Domain [{S_MIN},{S_MAX}] normalized to [-1,1]; "
    f"plots in reference range s∈[{S_MIN},{S_MAX_REF}]",
    fontsize=12,
    fontweight="bold",
)
fig.tight_layout()

os.makedirs(PLOTS_DIR, exist_ok=True)
combined_path = os.path.join(PLOTS_DIR, "weber_law_recovery.png")
fig.savefig(combined_path, dpi=200, bbox_inches="tight")
print(f"  Saved combined figure -> {combined_path}")

# ---------------------------------------------------------------------------
# Step 5; Individual panels
# ---------------------------------------------------------------------------

if SAVE_INDIVIDUAL_PANELS:
    print("[5/5] Saving individual panels ...")
    _save_panel(draw_trial_scatter, "panel1_trial_scatter")
    _save_panel(draw_jnd_recovery, "panel2_jnd_recovery")
    _save_panel(draw_weber_fraction, "panel3_weber_fraction")
    _save_panel(draw_fechner, "panel4_fechner")
    _save_panel(draw_psychometric, "panel5_psychometric")
    save_weber_coeffs_comparison()
else:
    print(
        "[5/5] Individual panels skipped (set SAVE_INDIVIDUAL_PANELS=True to enable)."
    )

# ---------------------------------------------------------------------------
# Step 6 (optional): basis-degree sweep with held-out likelihood
# ---------------------------------------------------------------------------
# Model selection question: how much flexibility does the data actually need?
# basis_degree sets the highest Chebyshev term in U(x):
#   degree 0 -> U constant   -> Sigma constant  -> constant JND (cannot do Weber);
#   degree 1 -> U linear      -> Sigma quadratic -> Weber exactly;
#   degree 2+ -> extra curvature the Weber data does not need.
# Training fit alone cannot rank these (more parameters never fit worse), so we
# split trials into train/test, fit on train, and score the log-likelihood on
# held-out test trials. The degree that generalizes best wins.
if RUN_BASIS_SWEEP:
    print("[6/6] Basis-degree sweep with held-out likelihood ...")

    # --8<-- [start:basis_sweep]
    # A lighter MC task keeps the repeated refits tractable.
    task_sweep = OddityTask(config=OddityTaskConfig(num_samples=400))

    n_test = N_TRIALS // 5
    perm = np.asarray(jr.permutation(jr.PRNGKey(7), N_TRIALS))
    test_idx, train_idx = perm[:n_test], perm[n_test:]

    def _subset(idx):
        return TrialData(
            stimuli=stimuli[idx],
            responses=responses[idx],
            stimulus_names=("ref", "comp"),
        )

    train_data, test_data = _subset(train_idx), _subset(test_idx)

    sweep_degrees = [0, 1, 2, 3]
    heldout_ll_per_trial = []
    for degree in sweep_degrees:
        prior_d = Prior(input_dim=1, basis_degree=degree, extra_embedding_dims=0)
        model_d = WPPM(
            input_dim=1,
            extra_dims=0,
            prior=prior_d,
            likelihood=task_sweep,
            noise=GaussianNoise(sigma=0.0),
            diag_term=DIAG_TERM,
        )
        opt_d = MAPOptimizer(
            steps=NUM_STEPS, learning_rate=LEARNING_RATE, track_history=False
        )
        post_d = opt_d.fit(
            model_d,
            train_data,
            init_params=model_d.init_params(jr.PRNGKey(1)),
            seed=2,
        )
        ll = float(
            model_d.log_likelihood_from_data(
                post_d.params, test_data, key=jr.PRNGKey(123)
            )
        )
        heldout_ll_per_trial.append(ll / n_test)
        print(f"    degree {degree}: held-out loglik/trial = {ll / n_test:.4f}")
    # --8<-- [end:basis_sweep]

    fig_s, ax_s = plt.subplots(figsize=(5.5, 4.2))
    ax_s.plot(sweep_degrees, heldout_ll_per_trial, "o-", color="#2166ac", linewidth=2)
    # Parsimony rule: pick the simplest degree whose held-out score is within a
    # small tolerance of the best. Differences below MC noise do not justify the
    # extra parameters, so we prefer the smallest sufficient model.
    best_ll = max(heldout_ll_per_trial)
    tol = 0.005
    chosen = next(i for i, ll in enumerate(heldout_ll_per_trial) if ll >= best_ll - tol)
    ax_s.scatter(
        [sweep_degrees[chosen]],
        [heldout_ll_per_trial[chosen]],
        s=170,
        facecolors="none",
        edgecolors="#c0392b",
        linewidths=2,
        zorder=5,
        label=f"simplest sufficient: degree {sweep_degrees[chosen]}",
    )
    ax_s.set_xticks(sweep_degrees)
    ax_s.set_xlabel("Chebyshev basis degree")
    ax_s.set_ylabel("Held-out log-likelihood / trial")
    ax_s.set_title(
        "How much flexibility does the data need?\n(degree 1 suffices for Weber)"
    )
    ax_s.legend(fontsize=8)
    ax_s.grid(True, alpha=0.25)
    fig_s.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    sweep_path = os.path.join(PLOTS_DIR, "weber_basis_sweep.png")
    fig_s.savefig(sweep_path, dpi=200, bbox_inches="tight")
    plt.close(fig_s)
    print(f"  Saved basis sweep -> {sweep_path}")
else:
    print("[6/6] Basis-degree sweep skipped (set RUN_BASIS_SWEEP=True to enable).")

print("Done.")
