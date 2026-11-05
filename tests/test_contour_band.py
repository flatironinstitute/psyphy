"""
tests/test_contour_band.py
----------------------------

Tests for contour_radii and WPPMCovarianceFieldSamples.band: the
radius-per-direction parameterization of a threshold contour, and pointwise
credible bands over it.

The first test is the pin for the extraction. The credible band used to be
computed inline in docs/examples/wppm/nuts_inference_example.py, and the
figures on that tutorial page come from those numbers. `_reference_band` below
is that inline code, copied verbatim, and the pin asserts the library
reproduces it. If that test fails, the tutorial's published figures no longer
match the library.

Like test_mcmc_posterior.py these tests build an MCMCPosterior by hand, so the
suite runs with no sampling backend installed.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from psyphy.model import WPPM, GaussianNoise, OddityTask, Prior
from psyphy.model.covariance_field import (
    WPPMCovarianceField,
    WPPMCovarianceFieldSamples,
    _eti,
    _hdi,
    contour_radii,
)
from psyphy.model.likelihood import OddityTaskConfig
from psyphy.posterior.mcmc_posterior import MCMCPosterior
from psyphy.posterior.posterior import MAPPosterior


@pytest.fixture(autouse=True)
def _full_matmul_precision():
    """Full float32 matmuls: TF32 on Ampere+ GPUs would blur exact comparisons.

    The pin below compares two implementations elementwise, so it needs the
    same arithmetic from both.
    """
    with jax.default_matmul_precision("float32"):
        yield


@pytest.fixture
def tiny_model():
    """Minimal WPPM: degree=1, 2D, 1 extra dim -> W shape (2, 2, 2, 3)."""
    return WPPM(
        prior=Prior(input_dim=2, basis_degree=1, extra_embedding_dims=1),
        likelihood=OddityTask(config=OddityTaskConfig(num_samples=5)),
        noise=GaussianNoise(),
    )


@pytest.fixture
def posterior(tiny_model):
    """MCMCPosterior with hand-made draws: 2 chains x 25 draws."""
    W = jr.normal(jr.PRNGKey(0), (2, 25, 2, 2, 2, 3)) * 0.3
    return MCMCPosterior({"W": W}, tiny_model)


@pytest.fixture
def points():
    """Four reference points. (n_points, d) = (4, 2)"""
    return jnp.array([[0.0, 0.0], [0.3, -0.2], [-0.4, 0.1], [0.2, 0.5]])


def _fields(posterior, n_samples=None, key=None):
    """The draws object, which is where band() lives."""
    return WPPMCovarianceFieldSamples.from_posterior(
        posterior, n_samples=n_samples, key=key
    )


def _reference_band(posterior, model, points, n_samples, ci_prob, key):
    """The inline computation from nuts_inference_example.py, verbatim.

    Kept here as the reference implementation so the extraction has something
    to be checked against. Returns (r_lo, r_med, r_hi), each
    (n_points, n_directions).
    """
    phi = jnp.linspace(0.0, 2 * jnp.pi, 181)  # radians
    u = jnp.stack([jnp.cos(phi), jnp.sin(phi)], axis=-1)  # (n_phi, 2)

    def contour_radii(cov):
        prec = jnp.linalg.inv(cov)
        return 1.0 / jnp.sqrt(jnp.einsum("pi,ij,pj->p", u, prec, u))

    band_samples = posterior.sample(n_samples, key=key)
    band_covs = jnp.stack(  # (n_samples, n_points, 2, 2)
        [
            WPPMCovarianceField(model, {"W": band_samples["W"][i]})(points)
            for i in range(band_samples["W"].shape[0])
        ]
    )
    radii = jax.vmap(jax.vmap(contour_radii))(band_covs)  # (n_s, n_pts, n_phi)
    q = jnp.array([(1 - ci_prob) / 2, 0.5, 1 - (1 - ci_prob) / 2])
    return jnp.quantile(radii, q, axis=0)


# ---------------------------------------------------------------------------
# The pin
# ---------------------------------------------------------------------------


def test_band_matches_the_inline_tutorial_computation(tiny_model, posterior, points):
    """band reproduces the numbers the tutorial script produced.

    This is the extraction's pin. A failure means the published figures in
    nuts_inference_example.md no longer match what psyphy computes.
    """
    key = jr.PRNGKey(1)
    r_lo, r_med, r_hi = _reference_band(
        posterior, tiny_model, points, n_samples=50, ci_prob=0.89, key=key
    )

    band = _fields(posterior, 50, key).band(points, ci_prob=0.89)

    # The reference loops over samples in Python, band vmaps, so float32
    # accumulates in a different order. The gap that leaves is absolute, about
    # 1.5e-6 on a radius, and does not shrink with the radius: r_lo holds the
    # smallest radii here (~0.06), so an rtol alone would set a tolerance below
    # the noise floor and fail on one platform and not another. atol carries it
    # instead. A real regression moves these by percent, so the pin is still
    # tight: ci_prob 0.90 instead of 0.89 is 1266x this tolerance, and HDI
    # instead of ETI is 10000x.
    assert jnp.allclose(band.lower, r_lo, rtol=1e-5, atol=1e-5)
    assert jnp.allclose(band.median, r_med, rtol=1e-5, atol=1e-5)
    assert jnp.allclose(band.upper, r_hi, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# contour_radii: closed-form cases
# ---------------------------------------------------------------------------


def test_radius_along_an_eigenvector_is_sqrt_eigenvalue():
    """For a diagonal Sigma, the radius along each axis is sqrt of its variance."""
    cov = jnp.diag(jnp.array([4.0, 0.25]))  # (2, 2)
    directions = jnp.array([[1.0, 0.0], [0.0, 1.0]])  # (2, 2)

    r = contour_radii(cov, directions)  # (2,)

    assert jnp.allclose(r, jnp.array([2.0, 0.5]), atol=1e-6)


def test_radii_broadcast_over_leading_axes():
    """Leading axes pass through, so a whole ensemble goes in one call."""
    covs = jnp.broadcast_to(jnp.eye(2), (7, 3, 2, 2))  # (n_draws, n_points, 2, 2)
    directions = jnp.array([[1.0, 0.0], [0.0, 1.0]])

    r = contour_radii(covs, directions)

    assert r.shape == (7, 3, 2)  # (n_draws, n_points, n_directions)
    assert jnp.allclose(r, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# band: shapes, ordering, degenerate case
# ---------------------------------------------------------------------------


def test_band_shapes_for_a_single_level(posterior, points):
    band = _fields(posterior).band(points, n_directions=37, ci_prob=0.89)

    assert band.points.shape == (4, 2)
    assert band.directions.shape == (37, 2)
    for arr in (band.lower, band.median, band.upper):
        assert arr.shape == (4, 37)  # (n_points, n_directions)
    assert band.ci_prob == 0.89
    assert band.ci_kind == "eti"


def test_band_shapes_for_several_levels(posterior, points):
    """A sequence of probabilities adds a leading level axis."""
    band = _fields(posterior).band(points, n_directions=37, ci_prob=[0.5, 0.8, 0.95])

    for arr in (band.lower, band.median, band.upper):
        assert arr.shape == (3, 4, 37)  # (n_levels, n_points, n_directions)
    assert band.ci_prob == (0.5, 0.8, 0.95)


def test_median_is_the_same_at_every_level(posterior, points):
    """The median does not depend on ci_prob, so its levels are copies.

    It is stacked anyway so that lower, median and upper share a shape and a
    loop over levels can index all three.
    """
    band = _fields(posterior).band(points, n_directions=19, ci_prob=[0.5, 0.95])

    assert jnp.allclose(band.median[0], band.median[1], atol=1e-7)


def test_each_level_matches_a_single_level_call(posterior, points):
    """Asking for three levels at once equals asking three times."""
    key = jr.PRNGKey(3)
    fields = _fields(posterior, 40, key)
    many = fields.band(points, ci_prob=[0.5, 0.95])
    for i, p in enumerate((0.5, 0.95)):
        one = fields.band(points, ci_prob=p)
        assert jnp.allclose(many.lower[i], one.lower, atol=1e-6)
        assert jnp.allclose(many.upper[i], one.upper, atol=1e-6)


def test_levels_nest(posterior, points):
    """A wider probability gives a wider interval, everywhere."""
    band = _fields(posterior).band(points, ci_prob=[0.5, 0.95])

    assert jnp.all(band.lower[1] <= band.lower[0] + 1e-6)
    assert jnp.all(band.upper[1] >= band.upper[0] - 1e-6)


def test_band_is_ordered(posterior, points):
    band = _fields(posterior).band(points)

    assert jnp.all(band.lower <= band.median + 1e-6)
    assert jnp.all(band.median <= band.upper + 1e-6)


def test_directions_are_closed_and_unit(posterior, points):
    """The last direction repeats the first, so a filled polygon has no seam."""
    band = _fields(posterior).band(points, n_directions=37)

    assert jnp.allclose(band.directions[0], band.directions[-1], atol=1e-6)
    assert jnp.allclose(jnp.linalg.norm(band.directions, axis=-1), 1.0, atol=1e-6)


def test_map_posterior_band_has_zero_width(tiny_model):
    """A point mass has no spread, so its band collapses onto the median."""
    params = tiny_model.init_params(jr.PRNGKey(0))
    post = MAPPosterior(params, tiny_model)
    pts = jnp.array([[0.0, 0.0], [0.2, 0.2]])

    band = _fields(post, 8, jr.PRNGKey(0)).band(pts, n_directions=19)

    assert jnp.allclose(band.upper - band.lower, 0.0, atol=1e-6)
    assert jnp.allclose(band.median, band.lower, atol=1e-6)


# ---------------------------------------------------------------------------
# ci_kind
# ---------------------------------------------------------------------------


def test_hdi_is_narrower_than_eti_for_skewed_draws():
    """The highest-density interval is the narrowest one holding that mass.

    Tested on a strongly skewed sample, where the two intervals differ by much
    more than the gap between neighbouring draws. Radii are bounded below by
    zero and are often skewed in practice, which is why "hdi" is offered.
    """
    draws = jnp.asarray(
        np.random.default_rng(0).lognormal(0.0, 0.8, (2000, 5))
    )  # (n_draws, n_elements)

    for prob in (0.5, 0.89, 0.95):
        lo_h, hi_h = _hdi(draws, prob)
        lo_e, hi_e = _eti(draws, prob)
        assert jnp.all((hi_h - lo_h) < (hi_e - lo_e))


def test_hdi_covers_the_requested_mass():
    draws = jnp.asarray(np.random.default_rng(1).lognormal(0.0, 0.8, (2000, 5)))

    for prob in (0.5, 0.89, 0.95):
        lo, hi = _hdi(draws, prob)
        inside = jnp.mean((draws >= lo) & (draws <= hi), axis=0)
        assert jnp.all(inside >= prob - 1e-9)


def test_hdi_is_the_narrowest_window_of_that_size():
    """Brute force: no window of the same draw count is narrower.

    This is the defining property, checked without reference to any other
    implementation.
    """
    draws = jnp.asarray(np.random.default_rng(2).lognormal(0.0, 0.5, (200, 1)))
    prob = 0.8
    lo, hi = _hdi(draws, prob)

    ordered = jnp.sort(draws[:, 0])
    width = int(jnp.ceil(prob * ordered.shape[0]))
    widths = [
        float(ordered[i + width - 1] - ordered[i])
        for i in range(ordered.shape[0] - width + 1)
    ]

    assert float(hi[0] - lo[0]) == pytest.approx(min(widths), abs=1e-6)


def test_hdi_flows_through_the_band(posterior, points):
    """ci_kind="hdi" reaches the reduction and is recorded on the result."""
    fields = _fields(posterior, 50, jr.PRNGKey(5))
    eti = fields.band(points, ci_kind="eti")
    hdi = fields.band(points, ci_kind="hdi")

    assert hdi.ci_kind == "hdi"
    assert jnp.all(hdi.lower <= hdi.median + 1e-6)
    assert jnp.all(hdi.median <= hdi.upper + 1e-6)
    assert not jnp.allclose(hdi.lower, eti.lower)


def test_unknown_ci_kind_is_rejected(posterior, points):
    with pytest.raises(ValueError, match="ci_kind"):
        _fields(posterior).band(points, ci_kind="quantile")


def test_ci_prob_out_of_range_is_rejected(posterior, points):
    with pytest.raises(ValueError, match="ci_prob"):
        _fields(posterior).band(points, ci_prob=1.5)


# ---------------------------------------------------------------------------
# Limits of the current implementation
# ---------------------------------------------------------------------------


def test_non_2d_models_are_rejected():
    """The direction grid is a circle, so only 2-D stimulus spaces work today."""
    model = WPPM(
        prior=Prior(input_dim=1, basis_degree=1, extra_embedding_dims=1),
        likelihood=OddityTask(config=OddityTaskConfig(num_samples=5)),
        noise=GaussianNoise(),
    )
    post = MAPPosterior(model.init_params(jr.PRNGKey(0)), model)

    with pytest.raises(NotImplementedError, match="2-D"):
        _fields(post, 4, jr.PRNGKey(0)).band(jnp.array([[0.0]]))


def test_map_posterior_needs_explicit_n_samples(tiny_model):
    """MAPPosterior exposes no n_chains/n_draws, so the default cannot be inferred.

    The draws are taken at construction, so this is where the count is needed.
    """
    post = MAPPosterior(tiny_model.init_params(jr.PRNGKey(0)), tiny_model)

    with pytest.raises(ValueError, match="n_samples"):
        _fields(post)
