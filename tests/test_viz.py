"""Tests for :mod:`psyphy.viz`.

Two halves, mirroring the subpackage's own split:

1. **Geometry tests** need only numpy/scipy and run everywhere.
2. **Drawing tests** need matplotlib, an optional dependency, and skip when it
   is absent -- the same gating idea as the OSF-data tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from psyphy.viz.geometry import auto_scale, ellipse_segments

# ----------------------------------------------------------------------
# Geometry (no plotting backend required)
# ----------------------------------------------------------------------


class TestEllipseSegments:
    def test_identity_covariance_is_a_unit_circle(self):
        segs, valid = ellipse_segments(np.zeros((1, 2)), np.eye(2)[None])
        assert valid.all()
        radii = np.linalg.norm(segs[0], axis=1)
        np.testing.assert_allclose(radii, 1.0, atol=1e-12)

    def test_semi_axes_are_sqrt_eigenvalues(self):
        """The drawn extent must be sqrt(lambda), not lambda.

        Tolerance is loose because the ellipse is a polyline: with n_points
        vertices the sampled angles need not land exactly on the semi-axes, so
        the measured extent is slightly under the true one.
        """
        cov = np.diag([0.04, 0.01])[None]  # semi-axes 0.2 and 0.1
        segs, _ = ellipse_segments(np.zeros((1, 2)), cov)
        np.testing.assert_allclose(np.abs(segs[0][:, 0]).max(), 0.2, rtol=2e-3)
        np.testing.assert_allclose(np.abs(segs[0][:, 1]).max(), 0.1, rtol=2e-3)

    def test_scale_multiplies_linearly(self):
        base, _ = ellipse_segments(np.zeros((1, 2)), np.eye(2)[None])
        scaled, _ = ellipse_segments(np.zeros((1, 2)), np.eye(2)[None], scale=3.0)
        np.testing.assert_allclose(scaled[0], 3.0 * base[0], atol=1e-12)

    def test_centers_offset_the_ellipse(self):
        segs, _ = ellipse_segments(np.array([[1.0, -2.0]]), np.eye(2)[None])
        np.testing.assert_allclose(segs[0].mean(axis=0), [1.0, -2.0], atol=1e-2)

    def test_non_pd_is_skipped_and_reported(self):
        covs = np.stack([np.eye(2), np.diag([-1.0, 1.0])])
        segs, valid = ellipse_segments(np.zeros((2, 2)), covs)
        assert len(segs) == 1
        np.testing.assert_array_equal(valid, [True, False])

    def test_higher_dimensions_raise_not_implemented(self):
        with pytest.raises(NotImplementedError, match="ellipsoid"):
            ellipse_segments(np.zeros((1, 2)), np.eye(3)[None])

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="same length"):
            ellipse_segments(np.zeros((3, 2)), np.eye(2)[None])


class TestAutoScale:
    def test_sizes_median_ellipse_to_a_fraction_of_spacing(self):
        centers = np.stack(
            np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 1, 5)), axis=-1
        ).reshape(-1, 2)
        covs = np.tile(np.eye(2) * 0.01, (len(centers), 1, 1))  # radius 0.1
        s = auto_scale(centers, covs, fraction=0.35)
        np.testing.assert_allclose(s, 0.35 * 0.25 / 0.1, rtol=1e-6)

    def test_can_shrink_not_only_magnify(self):
        """On a dense grid the factor is below 1 -- it is not a magnifier."""
        centers = np.stack(
            np.meshgrid(np.linspace(0, 1, 50), np.linspace(0, 1, 50)), axis=-1
        ).reshape(-1, 2)
        covs = np.tile(np.eye(2) * 0.01, (len(centers), 1, 1))
        assert auto_scale(centers, covs) < 1.0

    def test_single_center_raises(self):
        with pytest.raises(ValueError, match="at least 2 centers"):
            auto_scale(np.zeros((1, 2)), np.eye(2)[None])


# ----------------------------------------------------------------------
# Drawing (requires matplotlib)
# ----------------------------------------------------------------------

mpl = pytest.importorskip("matplotlib", reason="psyphy[viz] not installed")
mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from psyphy.viz import plot_ellipses  # noqa: E402


@pytest.fixture
def field():
    centers = np.stack(
        np.meshgrid(np.linspace(-0.7, 0.7, 4), np.linspace(-0.7, 0.7, 4)), axis=-1
    ).reshape(-1, 2)
    covs = np.tile(np.diag([4e-3, 2e-3]), (len(centers), 1, 1))
    return centers, covs


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


class TestPlotEllipses:
    def test_single_field_draws_one_collection(self, field):
        centers, covs = field
        ax = plot_ellipses(centers, covs)
        assert len(ax.collections) == 1

    def test_list_of_fields_overlays(self, field):
        centers, covs = field
        ax = plot_ellipses(centers, [covs, covs * 1.5])
        assert len(ax.collections) == 2

    def test_stacked_samples_draw_one_collection_each(self, field):
        """(n_samples, n_points, 2, 2) -- e.g. posterior draws."""
        centers, covs = field
        ax = plot_ellipses(centers, np.stack([covs] * 5), alpha=0.2)
        assert len(ax.collections) == 5

    def test_auto_scale_is_shared_across_fields(self, field):
        """Both fields must use one factor, or relative size is meaningless."""
        centers, covs = field
        _, s = plot_ellipses(
            centers, [covs, covs * 100.0], scale="auto", return_scale=True
        )
        expected = auto_scale(centers, covs)  # from the FIRST field only
        np.testing.assert_allclose(s, expected, rtol=1e-9)

    def test_default_scale_is_true_size(self, field):
        centers, covs = field
        _, s = plot_ellipses(centers, covs, return_scale=True)
        assert s == 1.0

    def test_per_point_colors_accepted(self, field):
        centers, covs = field
        rgb = np.linspace(0, 1, len(centers) * 3).reshape(-1, 3)
        ax = plot_ellipses(centers, covs, colors=rgb)
        assert len(ax.collections) == 1

    def test_labels_produce_a_legend(self, field):
        centers, covs = field
        ax = plot_ellipses(centers, [covs, covs], labels=["a", "b"])
        assert ax.get_legend() is not None

    def test_non_pd_warns_and_still_draws(self, field):
        centers, covs = field
        covs = covs.copy()
        covs[0] = np.diag([-1.0, 1.0])
        with pytest.warns(UserWarning, match="non-positive-definite"):
            ax = plot_ellipses(centers, covs)
        assert len(ax.collections) == 1

    def test_non_pd_can_raise_instead(self, field):
        centers, covs = field
        covs = covs.copy()
        covs[0] = np.diag([-1.0, 1.0])
        with pytest.raises(ValueError, match="non-positive-definite"):
            plot_ellipses(centers, covs, skip_non_pd=False)

    def test_draws_into_supplied_axes(self, field):
        centers, covs = field
        _, ax = plt.subplots()
        assert plot_ellipses(centers, covs, ax=ax) is ax

    def test_bad_cov_shape_names_the_accepted_forms(self, field):
        centers, _ = field
        with pytest.raises(ValueError, match=r"n_points, 2, 2"):
            plot_ellipses(centers, np.zeros((len(centers), 2)))

    def test_bad_scale_string_raises(self, field):
        centers, covs = field
        with pytest.raises(ValueError, match="'auto'"):
            plot_ellipses(centers, covs, scale="big")

    @pytest.mark.parametrize("bad", [[1.0, 2.0], (1.0, 2.0), np.array([1.0, 2.0])])
    def test_per_field_scale_is_refused(self, field, bad):
        """One scale for every field is a deliberate constraint, not an oversight.

        Fields magnified independently cannot be compared by eye, which is the
        only reason to draw them on shared axes. Passing one scale per field is
        the natural mistake, so it has to fail loudly rather than silently take
        the first value or broadcast.
        """
        centers, covs = field
        with pytest.raises(TypeError, match="single number"):
            plot_ellipses(centers, [covs, covs], scale=bad)

    def test_scale_error_explains_why(self, field):
        """The message has to carry the rationale; the constraint is non-obvious."""
        centers, covs = field
        with pytest.raises(TypeError) as excinfo:
            plot_ellipses(centers, [covs, covs], scale=[1.0, 2.0])
        assert "independently" in str(excinfo.value)
