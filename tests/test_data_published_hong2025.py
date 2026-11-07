"""Tests for :mod:`psyphy.data.published.hong2025`.

Two clearly separated halves:

1. **Parsing tests** run everywhere. They use small synthetic CSVs written to
   ``tmp_path`` that mimic the published layout, so CI needs no network and no
   third-party data.

2. **The external-validity test** is skipped unless the real OSF data has been
   downloaded. It is the reason this module exists: it asserts that psyphy's
   covariance construction reproduces the field Hong et al. published, given
   their weights. Run ``hong2025.fetch(1, noise_ellipses=True)`` to enable it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from psyphy.data.dataset import TrialData
from psyphy.data.published import hong2025
from psyphy.model.covariance_field import WPPMCovarianceField

# ``enable_x64`` moved from ``jax.experimental`` to the top-level ``jax``
# namespace. psyphy supports ``jax>=0.4.28``, which spans the move, so resolve
# it at import time rather than committing to one spelling: 0.4.28 has only the
# experimental path, recent versions have only the top-level one, and hardcoding
# either breaks the other.
try:
    _enable_x64 = jax.enable_x64
except AttributeError:  # jax < 0.6-ish
    # The ignore is the point of the shim rather than a defect in it: on a recent
    # jax the experimental path does not exist, which is precisely what the `try`
    # above has already established at this line.
    #
    # Imported under its own name and aliased on a separate line.
    # Writing `import enable_x64 as _enable_x64` here needs a second ignore for
    # `no-redef` and pushes the line past 88 characters, at which point
    # ruff-format wraps it in parentheses and the trailing `# type: ignore`
    # lands on a line mypy does not blame, silently doing nothing.
    from jax.experimental import enable_x64  # type: ignore[attr-defined]

    _enable_x64 = enable_x64


@pytest.fixture
def x64():
    """Enable float64 for the duration of a single test, then restore.

    Comparing against the published covariances needs float64 -- in float32 the
    agreement floors around 1e-7 rather than resolving the ~1e-9 that is
    actually there.

    This is a scoped context manager and *not* a module-level
    ``jax.config.update``. pytest imports every test module during collection,
    before running any test, so a module-level flag flip changes precision for
    the entire session. Other modules here assert behaviour that only holds in
    float32 (e.g. ``test_mc_likelihood.py::test_gradients_are_finite_normal_case``),
    and would fail depending on collection order.
    """
    with _enable_x64():
        yield


# ----------------------------------------------------------------------
# Synthetic fixtures mimicking the published CSV layout
# ----------------------------------------------------------------------

_TRIALS_CSV = """TrialType,xref,x1,y
AEPsych_0_Sobol_0,"0.10,-0.20","0.15,-0.25",1
AEPsych_1_adaptive_0,"-0.30,0.40","-0.35,0.45",0
AEPsych_2_adaptive_1,"0.50,0.60","0.55,0.65",1
MOCS_0_cond_0_level_0_trial_0,"0.00,0.00","0.05,0.05",1
MOCS_0_cond_0_level_0_trial_1,"0.00,0.00","0.05,0.05",0
"""

_SIGMA_CSV = """grid_ref,Sigmas_thres_grid_org,Sigmas_thres_grid_btst1_rank0
"-0.70000000,-0.70000000","[[0.008,0.003],[0.003,0.005]]","[[0.009,0.004],[0.004,0.006]]"
"0.00000000,0.00000000","[[0.001,0.000],[0.000,0.002]]","[[0.002,0.001],[0.001,0.003]]"
"""


def _write(tmp_path, name: str, text: str):
    path = tmp_path / name
    path.write_text(text)
    return path


def _weights_csv(shape=(2, 2, 2, 3), *, column: str = "W_org") -> str:
    """A dense weights table over `shape`, values enumerated so gaps are visible."""
    lines = [f'"i,j,k,l",{column},W_btst0_rank0']
    for n, idx in enumerate(np.ndindex(*shape)):
        key = ",".join(str(i) for i in idx)
        lines.append(f'"{key}",{n * 0.001:.6f},{n * 0.002:.6f}')
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------
# load_trials
# ----------------------------------------------------------------------


class TestLoadTrials:
    def test_returns_trial_data_with_expected_shapes(self, tmp_path):
        data = hong2025.load_trials(_write(tmp_path, "t.csv", _TRIALS_CSV))
        assert isinstance(data, TrialData)
        # 3 AEPsych rows kept by default; K=2 (ref, comp) and d=2.
        assert data.stimuli.shape == (3, 2, 2)
        assert data.responses.shape == (3, 1)
        assert data.stimulus_names == ("ref", "comp")

    def test_default_keeps_only_aepsych_trials(self, tmp_path):
        """The published fit excluded MOCS rows; the default must too."""
        path = _write(tmp_path, "t.csv", _TRIALS_CSV)
        assert hong2025.load_trials(path).num_trials == 3
        assert hong2025.load_trials(path, trial_types=None).num_trials == 5
        assert hong2025.load_trials(path, trial_types=("MOCS",)).num_trials == 2

    def test_named_stimulus_slots_map_to_ref_and_comparison(self, tmp_path):
        data = hong2025.load_trials(_write(tmp_path, "t.csv", _TRIALS_CSV))
        np.testing.assert_allclose(data.stimulus("ref")[0], [0.10, -0.20])
        np.testing.assert_allclose(data.stimulus("comp")[0], [0.15, -0.25])

    def test_responses_preserve_row_order(self, tmp_path):
        data = hong2025.load_trials(_write(tmp_path, "t.csv", _TRIALS_CSV))
        np.testing.assert_array_equal(np.asarray(data.responses)[:, 0], [1, 0, 1])

    def test_subsampling_is_reproducible_given_seed(self, tmp_path):
        path = _write(tmp_path, "t.csv", _TRIALS_CSV)
        a = hong2025.load_trials(path, max_trials=2, seed=7)
        b = hong2025.load_trials(path, max_trials=2, seed=7)
        assert a.num_trials == 2
        np.testing.assert_array_equal(np.asarray(a.stimuli), np.asarray(b.stimuli))

    def test_max_trials_above_available_is_a_noop(self, tmp_path):
        path = _write(tmp_path, "t.csv", _TRIALS_CSV)
        assert hong2025.load_trials(path, max_trials=99).num_trials == 3

    def test_missing_column_raises_naming_the_column(self, tmp_path):
        path = _write(tmp_path, "bad.csv", 'TrialType,xref,x1\nA,"0,0","0,0"\n')
        with pytest.raises(ValueError, match="missing expected column"):
            hong2025.load_trials(path)

    def test_no_matching_trial_type_raises(self, tmp_path):
        path = _write(tmp_path, "t.csv", _TRIALS_CSV)
        with pytest.raises(ValueError, match="No trials matched"):
            hong2025.load_trials(path, trial_types=("Nonexistent",))


# ----------------------------------------------------------------------
# load_reference_W
# ----------------------------------------------------------------------


class TestLoadReferenceW:
    def test_shape_is_inferred_from_index_keys(self, tmp_path):
        path = _write(tmp_path, "w.csv", _weights_csv((2, 2, 2, 3)))
        assert hong2025.load_reference_W(path).shape == (2, 2, 2, 3)

    def test_values_land_at_their_index(self, tmp_path):
        path = _write(tmp_path, "w.csv", _weights_csv((2, 2, 2, 3)))
        W = np.asarray(hong2025.load_reference_W(path))
        # Values were enumerated in np.ndindex order.
        for n, idx in enumerate(np.ndindex(2, 2, 2, 3)):
            assert W[idx] == pytest.approx(n * 0.001)

    def test_alternate_column_can_be_selected(self, tmp_path):
        path = _write(tmp_path, "w.csv", _weights_csv((2, 2, 2, 3)))
        main = np.asarray(hong2025.load_reference_W(path))
        btst = np.asarray(hong2025.load_reference_W(path, column="W_btst0_rank0"))
        np.testing.assert_allclose(btst, 2.0 * main)

    def test_duplicate_index_keys_are_rejected(self, tmp_path):
        """A duplicated key plus a dropped one preserves both the row count and
        the per-axis maxima, so the density check alone cannot catch it."""
        rows = _weights_csv((2, 2, 2, 3)).splitlines()
        rows[5] = rows[1]  # duplicate the first index key onto another row
        path = _write(tmp_path, "w.csv", "\n".join(rows) + "\n")
        with pytest.raises(ValueError, match="duplicate index keys"):
            hong2025.load_reference_W(path)

    def test_sparse_table_is_rejected(self, tmp_path):
        rows = _weights_csv((2, 2, 2, 3)).splitlines()
        del rows[3]  # drop a row; maxima unchanged, count now short
        path = _write(tmp_path, "w.csv", "\n".join(rows) + "\n")
        with pytest.raises(ValueError, match="not dense"):
            hong2025.load_reference_W(path)

    def test_unknown_column_raises(self, tmp_path):
        path = _write(tmp_path, "w.csv", _weights_csv((2, 2, 2, 3)))
        with pytest.raises(ValueError, match="not found"):
            hong2025.load_reference_W(path, column="W_nope")

    def test_wrong_index_column_raises(self, tmp_path):
        path = _write(tmp_path, "w.csv", 'idx,W_org\n"0,0,0,0",1.0\n')
        with pytest.raises(ValueError, match="expected first column"):
            hong2025.load_reference_W(path)


# ----------------------------------------------------------------------
# load_sigma_table
# ----------------------------------------------------------------------


class TestLoadSigmaTable:
    def test_parses_coordinates_and_matrices(self, tmp_path):
        coords, sigmas = hong2025.load_sigma_table(
            _write(tmp_path, "s.csv", _SIGMA_CSV)
        )
        assert coords.shape == (2, 2)
        assert sigmas.shape == (2, 2, 2)
        np.testing.assert_allclose(coords[0], [-0.7, -0.7])
        np.testing.assert_allclose(sigmas[0], [[0.008, 0.003], [0.003, 0.005]])

    def test_org_column_is_auto_detected(self, tmp_path):
        _, sigmas = hong2025.load_sigma_table(_write(tmp_path, "s.csv", _SIGMA_CSV))
        # Must pick *_org, not merely the first value column.
        np.testing.assert_allclose(sigmas[1], [[0.001, 0.0], [0.0, 0.002]])

    def test_explicit_column_overrides_auto_detection(self, tmp_path):
        _, sigmas = hong2025.load_sigma_table(
            _write(tmp_path, "s.csv", _SIGMA_CSV),
            value_column="Sigmas_thres_grid_btst1_rank0",
        )
        np.testing.assert_allclose(sigmas[0], [[0.009, 0.004], [0.004, 0.006]])

    def test_fine_grid_coordinate_column_is_accepted(self, tmp_path):
        text = _SIGMA_CSV.replace("grid_ref,", "grid_ref_fine,", 1).replace(
            "Sigmas_thres", "Sigmas_noise"
        )
        coords, _ = hong2025.load_sigma_table(_write(tmp_path, "n.csv", text))
        assert coords.shape == (2, 2)

    def test_unexpected_coordinate_column_raises(self, tmp_path):
        text = _SIGMA_CSV.replace("grid_ref,", "location,", 1)
        with pytest.raises(ValueError, match="expected first column"):
            hong2025.load_sigma_table(_write(tmp_path, "s.csv", text))


# ----------------------------------------------------------------------
# build_paper_model
# ----------------------------------------------------------------------


class TestBuildPaperModel:
    def test_matches_the_published_configuration(self):
        model = hong2025.build_paper_model(mc_samples=4)
        assert model.input_dim == 2
        assert model.basis_degree == 4  # paper's degree=5 counts T0..T4
        assert model.embedding_dim == 3  # input_dim + extra_dims
        assert model.diag_term == 0.0
        assert model.prior.variance_scale == pytest.approx(3e-4)
        assert model.prior.decay_rate == pytest.approx(0.4)

    def test_mc_controls_are_overridable_but_default_to_the_paper(self):
        assert hong2025.build_paper_model().likelihood.config.num_samples == 2000
        assert hong2025.build_paper_model().likelihood.config.bandwidth == 5e-3
        fast = hong2025.build_paper_model(mc_samples=8, bandwidth=1e-2)
        assert fast.likelihood.config.num_samples == 8
        assert fast.likelihood.config.bandwidth == 1e-2

    def test_prior_sample_has_the_published_weight_shape(self):
        model = hong2025.build_paper_model(mc_samples=4)
        params = model.init_params(jax.random.PRNGKey(0))
        # Same shape as the published Bestfit_W table, so the two are swappable.
        assert params["W"].shape == (5, 5, 2, 3)


# ----------------------------------------------------------------------
# Colour transform: load_calibration_matrix / w2d_to_rgb
# ----------------------------------------------------------------------

#: A hand-checkable stand-in for the published matrix: doubles
#: w1, negates w2, and offsets every channel by 0.5.
_FAKE_M_CSV = """2.0,0.0,0.5
0.0,-1.0,0.5
0.0,0.0,0.5
"""


class TestLoadCalibrationMatrix:
    def test_parses_a_3x3_matrix(self, tmp_path):
        path = _write(tmp_path, "M.csv", _FAKE_M_CSV)
        M = hong2025.load_calibration_matrix(path)
        assert M.shape == (3, 3)
        np.testing.assert_allclose(M[0], [2.0, 0.0, 0.5])

    def test_wrong_shape_is_rejected(self, tmp_path):
        path = _write(tmp_path, "M.csv", "1.0,2.0\n3.0,4.0\n")
        with pytest.raises(ValueError, match="expected a 3x3 matrix"):
            hong2025.load_calibration_matrix(path)


class TestW2dToRgb:
    @pytest.fixture
    def M(self, tmp_path):
        return hong2025.load_calibration_matrix(_write(tmp_path, "M.csv", _FAKE_M_CSV))

    def test_single_coordinate_returns_a_single_rgb(self, M):
        rgb = hong2025.w2d_to_rgb(np.array([0.1, -0.2]), M)
        assert rgb.shape == (3,)
        # R = 2*0.1 + 0.5 = 0.7; G = -1*(-0.2) + 0.5 = 0.7; B = 0.5
        np.testing.assert_allclose(rgb, [0.7, 0.7, 0.5])

    def test_batch_shape_is_preserved(self, M):
        coords = np.zeros((5, 2))
        assert hong2025.w2d_to_rgb(coords, M).shape == (5, 3)

    def test_augmentation_supplies_the_dc_offset(self, M):
        """The origin maps to the matrix's third column, not to black."""
        np.testing.assert_allclose(hong2025.w2d_to_rgb(np.zeros(2), M), [0.5, 0.5, 0.5])

    def test_out_of_gamut_values_are_clipped(self, M):
        # R = 2*10 + 0.5 = 20.5 -> 1.0; G = -1*10 + 0.5 = -9.5 -> 0.0
        rgb = hong2025.w2d_to_rgb(np.array([10.0, 10.0]), M)
        np.testing.assert_allclose(rgb, [1.0, 0.0, 0.5])

    def test_bad_coordinate_shape_raises(self, M):
        with pytest.raises(ValueError, match=r"shape \(2,\) or \(N, 2\)"):
            hong2025.w2d_to_rgb(np.zeros((4, 3)), M)


# ----------------------------------------------------------------------
# External validity: requires the OSF download
# ----------------------------------------------------------------------

_SUB1 = hong2025.default_data_dir() / "sub1"
_WEIGHTS = _SUB1 / "Bestfit_W_sub1.csv"
_NOISE = _SUB1 / "Noise_ellipses_sub1.csv"
_CALIBRATION_CSV = (
    hong2025.default_data_dir()
    / "calibration"
    / f"M_2DWToRGB_DELL_{hong2025.CALIBRATION_DATE}_copy.csv"
)

requires_osf_data = pytest.mark.skipif(
    not (_WEIGHTS.exists() and _NOISE.exists()),
    reason=(
        "Published data not downloaded. Run: python -c "
        "'from psyphy.data.published import hong2025; "
        "hong2025.fetch(1, noise_ellipses=True)'"
    ),
)


@requires_osf_data
def test_covariance_field_matches_published_sigma_noise(x64):
    """psyphy must reproduce the paper's covariance field from the paper's weights.

    This is the external-validity check for the whole WPPM covariance path:
    Chebyshev basis, weight layout, the einsum contraction, and
    ``Sigma = U U^T + diag_term I``. It is deterministic -- no optimizer, no
    Monte Carlo -- so any deviation is a real regression, not noise.

    The tolerance is set by the published CSV, which is rounded to 8 decimals;
    agreement is otherwise tighter than this.
    """
    W_org = hong2025.load_reference_W(_WEIGHTS)
    assert W_org.dtype == jnp.float64, "x64 must be enabled for this comparison"

    coords, sigma_published = hong2025.load_sigma_table(_NOISE)
    model = hong2025.build_paper_model(mc_samples=1)
    sigma_psyphy = np.asarray(WPPMCovarianceField(model, {"W": W_org})(coords))

    assert sigma_psyphy.shape == sigma_published.shape
    max_abs = float(np.abs(sigma_psyphy - sigma_published).max())
    assert max_abs < 1e-8, f"max |diff| = {max_abs:.3e} over {len(coords)} points"


@requires_osf_data
def test_published_trials_load_with_expected_counts():
    """Guards the trial-type split the published fit depends on."""
    trials_csv = _SUB1 / "trial_data_pooled_by_type_sub1.csv"
    if not trials_csv.exists():
        pytest.skip("trial data not downloaded")

    fitted = hong2025.load_trials(trials_csv)
    everything = hong2025.load_trials(trials_csv, trial_types=None)
    assert fitted.num_trials == 6000  # AEPsych only -- what the paper fitted
    assert everything.num_trials == 12000  # + held-out MOCS validation trials
    assert float(fitted.responses.mean()) == pytest.approx(0.7077, abs=1e-3)


@pytest.mark.skipif(
    not _CALIBRATION_CSV.exists(),
    reason=(
        "Calibration matrix not downloaded. Run: python -c "
        "'from psyphy.data.published import hong2025; "
        "hong2025.fetch_calibration_matrix()'"
    ),
)
def test_published_calibration_maps_origin_to_neutral_gray():
    """W-space origin must land on neutral monitor gray.

    The transformation-matrix README on OSF states that 2DW ``[0, 0]`` is the
    neutral monitor grey ``[0.5, 0.5, 0.5]`` in linear RGB. That single
    documented anchor pins the whole affine transform: it checks the DC offset
    (the matrix's third column), the augmentation, and the multiply
    orientation all at once. A transposed matrix or a missing augmentation
    would both fail here.
    """
    M = hong2025.load_calibration_matrix(_CALIBRATION_CSV)
    rgb = hong2025.w2d_to_rgb(np.zeros(2), M)
    np.testing.assert_allclose(rgb, [0.5, 0.5, 0.5], atol=1e-5)

    # Over the paper's own reference grid nothing should clip.
    grid = np.stack(
        np.meshgrid(np.linspace(-0.7, 0.7, 7), np.linspace(-0.7, 0.7, 7)), axis=-1
    ).reshape(-1, 2)
    colors = hong2025.w2d_to_rgb(grid, M)
    assert colors.shape == (49, 3)
    assert np.all((colors > 0.0) & (colors < 1.0)), "unexpected gamut clipping"
