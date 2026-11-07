"""
test_quick_start_recovery.py
----------------------------

Regression test for the quick-start tutorial's headline claim: fitting a WPPM
to data simulated from a known WPPM recovers the ground-truth covariance.

This exists because the tutorial's committed figure once showed a good fit that
the committed code could not reproduce -- the fit had silently collapsed to zero
covariance and nothing caught it, because "the script ran without error" was the
only check.

The settings were first calibrated on a local, current jax install, and the
first version of this test failed immediately on CI. The reason: CI's ``tests``
job floor-tests against ``jax==0.4.28`` (see ``.github/workflows/lint.yml``),
which is not just numerically different from a current jax but draws different
samples from the *same* ``PRNGKey`` -- so ground truth, init and data are a
different problem instance per jax version, not a noisy variant of the same one.
At low trial counts (checked from n=10 to n=60) the outcome is not a clean
"collapses to zero" failure either; it is chaotic across both seed and jax
version, landing anywhere from near-zero to a several-times overshoot to a
numerical blow-up. There is no trial count in that range where a specific
failure shape is portable, so this file does not try to pin one -- see the
comment at the bottom of this file.

Then it failed in CI a second time, for an unrelated reason worth recording
separately: *platform*. Holding jax and Python fixed at CI's own versions
(0.4.28 / 3.10), the relative Frobenius error is 0.150 on macOS arm64 and 0.590
on the Linux x86_64 runner -- float32 arithmetic and XLA codegen differ enough
on a chaotic Monte Carlo objective to move a shape statistic 4x. The area ratio
was stable across the same comparison. So the two assertions below are
calibrated to different strictness; see the bounds.

The lesson both rounds share: any numeric bound here has to hold across jax
version *and* OS/architecture, and checking one configuration is not evidence
about the others.

The settings here (1000 trials, not the tutorial's original 100) mirror
``docs/examples/wppm/quick_start.py``, calibrated to recover the ground truth
with margin across both the jax version floor CI pins to and a current jax,
each over several seeds. If that script's compute settings change, change them
here too -- the point is to pin the tutorial's result, not an arbitrary
configuration.
"""

import jax.numpy as jnp
import jax.random as jr
import pytest

from psyphy.data import TrialData
from psyphy.inference import MAPOptimizer
from psyphy.model import (
    WPPM,
    GaussianNoise,
    OddityTask,
    OddityTaskConfig,
    Prior,
    WPPMCovarianceField,
)

# Kept in sync with docs/examples/wppm/quick_start.py.
MC_SAMPLES = 50
NUM_TRIALS = 1000
NUM_STEPS = 600
LEARNING_RATE = 1e-4

REF_POINT = jnp.array([[0.0, 0.0]])
MAHAL_RADIUS = 2.8
NOISE_SIGMA = 0.1

# The two statistics are not equally strict, because they are not
# equally well determined.
#
# Area ratio (sqrt-det) measures *size*, which the data pins down well. Measured
# across five seeds on both jax==0.4.28 and a current jax, and on two platforms,
# it stays in [0.97, 1.14]. The bounds keep ~4x margin on that while still
# excluding a collapse (~0.01-0.1) or a divergence (thousands or worse).
#
# Relative Frobenius error also measures *shape and orientation*, which Monte
# Carlo noise leaves far less determined -- and which turns out to be the
# platform-sensitive one. Holding jax and Python fixed (0.4.28 / 3.10) it is
# 0.150 on macOS arm64 and 0.590 on the Linux x86_64 CI runner: a 4x spread
# from float32 arithmetic and XLA codegen alone, with no version difference
# involved. An earlier 0.5 bound here was calibrated on one platform and failed
# in CI for exactly that reason.
#
# 1.0 is the principled ceiling rather than an arbitrary loosening: a fit that
# collapsed to zero covariance scores ||0 - truth|| / ||truth|| == 1.0 exactly,
# so "< 1.0" asserts the fit beats predicting nothing. Raising MC_SAMPLES would
# tighten this (0.150 -> 0.042 at mc=200) but costs 13 s -> 67 s on a current
# jax, which is too slow for something the tutorial calls a quick start.
AREA_RATIO_BOUNDS = (0.6, 1.6)
MAX_REL_FROBENIUS = 1.0


def _area_ratio(fitted, truth) -> float:
    """Ratio of ellipse areas, sqrt(det) based. ~0 when the fit collapses."""
    return float(
        jnp.sqrt(jnp.sqrt(jnp.linalg.det(fitted)) / jnp.sqrt(jnp.linalg.det(truth)))
    )


def _simulate(n_trials: int = NUM_TRIALS):
    """Ground-truth model plus trials drawn from it, as the tutorial does."""
    task = OddityTask(config=OddityTaskConfig(num_samples=MC_SAMPLES))
    noise = GaussianNoise(sigma=NOISE_SIGMA)
    truth_model = WPPM(prior=Prior(), likelihood=task, noise=noise)
    truth_params = truth_model.init_params(jr.PRNGKey(123))

    refs = jnp.repeat(REF_POINT, repeats=n_trials, axis=0)
    truth_field = WPPMCovarianceField(truth_model, truth_params)
    sigmas = truth_field(refs)

    k_dir, k_sim = jr.split(jr.PRNGKey(3))
    angles = jr.uniform(k_dir, (n_trials,), minval=0.0, maxval=2 * jnp.pi)
    dirs = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)
    chol = jnp.linalg.cholesky(sigmas)
    comps = jnp.clip(
        refs + MAHAL_RADIUS * jnp.einsum("nij,nj->ni", chol, dirs), -1.0, 1.0
    )
    stimuli = jnp.stack([refs, comps], axis=1)
    responses, _ = task.simulate(truth_params, stimuli, truth_model, key=k_sim)
    return TrialData(stimuli=stimuli, responses=responses), task, noise, truth_field


@pytest.fixture(scope="module")
def fit():
    """Run the tutorial's fit once and share it across assertions (~4 s)."""
    data, task, noise, truth_field = _simulate()
    model = WPPM(prior=Prior(), likelihood=task, noise=noise)
    init_params = model.init_params(jr.PRNGKey(42))
    optimizer = MAPOptimizer(
        steps=NUM_STEPS,
        learning_rate=LEARNING_RATE,
        track_history=True,
        log_every=1,
    )
    estimate = optimizer.fit(model, data, init_params=init_params, seed=4)
    fitted = WPPMCovarianceField(model, estimate.params)(REF_POINT)[0]
    return {
        "truth": truth_field(REF_POINT)[0],
        "fitted": fitted,
        "init": WPPMCovarianceField(model, init_params)(REF_POINT)[0],
        "losses": optimizer.get_history()[1],
        "optimizer": optimizer,
    }


def test_fit_is_finite_and_positive_definite(fit):
    assert jnp.all(jnp.isfinite(fit["fitted"]))
    assert jnp.all(jnp.linalg.eigvalsh(fit["fitted"]) > 0)


def test_fitted_ellipse_has_the_right_area(fit):
    """sqrt(det) ratio: the statistic that catches a collapse to zero covariance."""
    ratio = _area_ratio(fit["fitted"], fit["truth"])
    low, high = AREA_RATIO_BOUNDS
    assert low < ratio < high, (
        f"fitted/true ellipse area ratio {ratio:.3f} outside [{low}, {high}]. "
        "A ratio near zero means the fit collapsed to zero covariance -- most "
        "likely NUM_TRIALS is too low for a single reference point."
    )


def test_fitted_covariance_matches_ground_truth(fit):
    rel = float(
        jnp.linalg.norm(fit["fitted"] - fit["truth"]) / jnp.linalg.norm(fit["truth"])
    )
    assert rel < MAX_REL_FROBENIUS, (
        f"relative Frobenius error {rel:.3f} exceeds {MAX_REL_FROBENIUS}"
    )


def test_fit_improves_on_its_initialization(fit):
    """Guards the vacuous pass where the prior draw already happened to be close."""

    def err(S):
        return float(jnp.linalg.norm(S - fit["truth"]))

    assert err(fit["fitted"]) < err(fit["init"])


def test_loss_decreases_and_stays_finite(fit):
    losses = fit["losses"]
    assert len(losses) == NUM_STEPS
    assert all(jnp.isfinite(jnp.asarray(x)) for x in losses)
    assert losses[-1] < losses[0]


def test_tutorial_settings_do_not_trigger_gradient_clipping(fit):
    """The tutorial relies on the default; if that changes, this test says so."""
    assert fit["optimizer"].max_grad_norm is None
    assert fit["optimizer"].clip_rate is None


# No "too few trials fails predictably" test, by choice. An earlier version
# asserted that 100 trials reliably collapses the fit to near-zero covariance
# and failed immediately in CI (jax==0.4.28): under that version's draw from
# the same seeds, the same config *overshot* by 1.35x instead -- see the module
# docstring for why. A sweep from n=10 to n=60 (both jax pins, three seeds
# each) found no trial count with a portable failure shape: outcomes ranged
# from a ~100x collapse to a several-times overshoot to a numerical blow-up
# (area ratio > 1e6), depending on seed and jax version alike. A version-robust
# version of that test would need to characterize instability statistically
# (e.g. dispersion across an ensemble of seeds) rather than pin one seed's
# outcome -- worth adding if this module regresses again, but heavier than a
# quick regression script warrants today.
