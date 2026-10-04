"""
hong2025.py
-----------

Loaders for the published human colour-discrimination dataset of Hong et al.

    Hong, F., Bouhassira, R., Chow, J., Sanders, C., Shvartsman, M., Guan, P.,
    Williams, A. H., & Brainard, D. H. (2026). Comprehensive characterization of
    human color discrimination thresholds. eLife, 14:RP108943.
    https://doi.org/10.7554/eLife.108943.2

Hong et al 2025 fitted a Wishart Process Psychophysical Model to eight observers
 and published both the trial-level data and the fitted Chebyshev weights.

Data availability
-----------------
Hosted on OSF node ``k27js`` (https://osf.io/k27js). **psyphy ships no data.**
:func:`fetch` downloads on request into a user cache directory. At the time of
writing the OSF node carries no explicit license, so please always cite paper!
Typical use
-----------
>>> from psyphy.data.published import hong2025
>>> paths = hong2025.fetch(subject=1)  # doctest: +SKIP
>>> data = hong2025.load_trials(paths["trials"])  # doctest: +SKIP
>>> W_org = hong2025.load_reference_W(paths["weights"])  # doctest: +SKIP

Coordinate convention
---------------------
All stimulus coordinates are in the paper's 2-D "W space" and already lie in the
Chebyshev domain [-1, 1]. No normalisation is applied or needed.

See also
--------
``docs/examples/wppm/hong2025_reproduction.md`` for the end-to-end walkthrough.
"""

from __future__ import annotations

import ast
import csv
import json
import os
import ssl
import sys
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from psyphy.data.dataset import TrialData

__all__ = [
    "CALIBRATION_DATE",
    "PAPER_HYPERPARAMS",
    "SUBJECT_INITIALS",
    "build_paper_model",
    "default_data_dir",
    "fetch",
    "fetch_calibration_matrix",
    "load_calibration_matrix",
    "load_reference_W",
    "load_sigma_table",
    "load_trials",
    "w2d_to_rgb",
]

# ----------------------------------------------------------------------
# DATASET CONSTANTS
# ----------------------------------------------------------------------

OSF_NODE = "k27js"
_OSF_API = "https://api.osf.io/v2"
_ORGANIZED = "Organized data and model predictions"
_CALIBRATION = "Calibration and transformation"
_TRANSFORM_MATRICES = "Transformation matrices"

#: Date stamp of the monitor calibration the published figures used.
#:
#: The transformation-matrix CSVs on OSF are named
#: ``M_<From>To<To>_DELL_<date>_copy.csv``. Only one calibration is published,
#: so this is effectively a constant; it is exposed as a parameter because the
#: filenames carry it and a future recalibration would change it.
CALIBRATION_DATE = "02242025"

#: Subject number -> observer initials, as used in the paper and the OSF tree.
SUBJECT_INITIALS: Mapping[int, str] = {
    1: "CH",
    2: "ME",
    4: "SG",
    6: "DK",
    7: "BH",
    8: "FM",
    10: "HG",
    11: "FW",
}

#: Trial-type prefixes used for the published fit.
#:
#: The pooled CSV holds 12 000 trials per observer, but the published fit used
#: only the 6 000 ``AEPsych_*`` rows (adaptive placement plus pre-generated
#: Sobol). The 6 000 ``MOCS_*`` rows are held-out validation data. See
#: ``ellipsoids/fit/fit_4d_human.py``, which fits ``combined_data`` as returned
#: by ``load_expt_data.load_combine_AEPsych_pregSobol()``. Fitting all 12 000
#: rows does not reproduce the paper.
FIT_TRIAL_TYPES: tuple[str, ...] = ("AEPsych",)

#: Model and optimizer settings transcribed from ``fit_4d_human.py`` SECTION 4.
#:
#: The paper constructs ``WishartProcessModel(5, 2, 1, 3e-4, 0.4, 0)``. Its
#: ``degree=5`` counts basis *functions* (T0..T4), whereas psyphy's
#: ``basis_degree`` is the *maximum degree*, hence 4 here. Both describe the
#: same 5x5 coefficient grid.
# --8<-- [start:hyperparams]
PAPER_HYPERPARAMS: Mapping[str, Any] = {
    # -- Covariance field: what shapes the noise field can take ------------
    # Highest Chebyshev degree, i.e. T0..T4 per input dimension -> a 5x5
    # coefficient grid. The paper writes this as degree=5, counting basis
    # *functions* rather than degrees. Same model, different convention.
    "basis_degree": 4,
    # Stimulus dimensionality: the 2-D isoluminant chromatic plane.
    "input_dim": 2,
    # Extra embedding dimensions for U(x), where Sigma = U U^T + diag_term * I
    # and U has shape (input_dim, input_dim + extra_dims). The extra column
    # lets Sigma stay full rank instead of being a bare rank-2 outer product.
    "extra_dims": 1,
    # -- Prior over the basis weights W ----------------------------------
    # Prior variance of a degree-0 coefficient: the overall scale of the noise
    # field before any data is seen.
    "variance_scale": 3e-4,
    # A degree-d coefficient has prior variance variance_scale * decay_rate^d,
    # so higher-frequency terms are shrunk harder. Smaller -> smoother field.
    "decay_rate": 0.4,
    # The delta added to Sigma's diagonal. psyphy's default is 1e-6,
    # which is safer for numerical stability.
    "diag_term": 0.0,
    # -- Oddity likelihood: how P(correct) is estimated --------------------
    # Monte Carlo draws per trial. P(correct) has no closed form, so it is
    # estimated by sampling; this dominates the cost of a fit.
    "mc_samples": 2000,
    # Width of the logistic that smooths the hard "is the odd one furthest?"
    # comparison, so the MC estimate stays differentiable for gradient descent
    "bandwidth": 5e-3,
    # -- MAP optimizer, used by the refit only -----------------------------
    "learning_rate": 1e-4,  # gradient step size
    "momentum": 0.2,  # 'heavy-ball' momentum
    "total_steps": 1500,  # gradient steps per restart
    "n_restarts": 1,  # independent prior draws; the lowest final loss wins
    # Hong et al tried 3, but 1 is enough to reproduce the published fit.
    # -- Reporting ---------------------------------------------------------
    # The criterion the paper's thresholds are defined at: midway between
    # chance (1/3) and perfect, for a 3-alternative task
    "target_pC": 0.667,
}
# --8<-- [end:hyperparams]


def default_data_dir() -> Path:
    """Return the directory :func:`fetch` downloads into.

    ``$PSYPHY_DATA_HOME`` if set, otherwise ``~/.cache/psyphy/hong2025``.
    Deliberately outside the repository so downloaded data is never staged for
    commit.
    """
    root = os.environ.get("PSYPHY_DATA_HOME")
    base = Path(root) if root else Path.home() / ".cache" / "psyphy"
    return base / "hong2025"


# ----------------------------------------------------------------------
# DOWNLOAD
# ----------------------------------------------------------------------
# Known system CA bundle locations (macOS, Debian/Ubuntu, RHEL, OpenSUSE).
# python.org builds on macOS ship their own OpenSSL that ignores the system
# keychain, so the default context fails with CERTIFICATE_VERIFY_FAILED.
_CA_PATHS = (
    "/etc/ssl/cert.pem",
    "/etc/ssl/certs/ca-certificates.crt",
    "/etc/pki/tls/certs/ca-bundle.crt",
    "/etc/ssl/ca-bundle.pem",
    "/usr/local/etc/openssl/cert.pem",
)


def _ssl_context() -> ssl.SSLContext:
    """Build an SSL context that works on macOS python.org installs and Linux."""
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        pass
    for ca_file in _CA_PATHS:
        if Path(ca_file).exists():
            try:
                return ssl.create_default_context(cafile=ca_file)
            except ssl.SSLError:
                continue
    return ssl.create_default_context()


def _get_json(url: str, ctx: ssl.SSLContext) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
            return json.loads(resp.read().decode())  # type: ignore[no-any-return]
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"OSF returned HTTP {exc.code} for {url}") from exc


def _list_folder(url: str, ctx: ssl.SSLContext) -> list[dict[str, Any]]:
    """Return every item in a (paginated) OSF folder listing."""
    items: list[dict[str, Any]] = []
    while url:
        payload = _get_json(url, ctx)
        items.extend(payload.get("data", []))
        url = payload.get("links", {}).get("next") or ""
    return items


def _subfolder_url(items: Sequence[dict[str, Any]], name: str) -> str:
    for item in items:
        attrs = item.get("attributes", {})
        if attrs.get("name") == name and attrs.get("kind") == "folder":
            return str(
                item.get("relationships", {})
                .get("files", {})
                .get("links", {})
                .get("related", {})
                .get("href", "")
            )
    return ""


def _find_file(items: Sequence[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for item in items:
        attrs = item.get("attributes", {})
        if attrs.get("name") == name and attrs.get("kind") == "file":
            return item
    return None


def _match_files(items: Sequence[dict[str, Any]], pattern: str) -> list[dict[str, Any]]:
    """Return every file entry whose name matches a glob pattern.

    Used for the calibration matrices, whose names carry both a monitor label
    and a ``_copy`` suffix (``M_2DWToRGB_DELL_02242025_copy.csv``). Matching a
    pattern rather than constructing the exact name keeps this robust to those
    incidental parts.
    """
    return [
        item
        for item in items
        if item.get("attributes", {}).get("kind") == "file"
        and fnmatch(str(item.get("attributes", {}).get("name", "")), pattern)
    ]


def _download(
    item: dict[str, Any], dest: Path, ctx: ssl.SSLContext, *, verbose: bool
) -> None:
    """Download one OSF file, skipping when a complete copy already exists."""
    expected = item.get("attributes", {}).get("size")
    if dest.exists() and (expected is None or dest.stat().st_size == expected):
        if verbose:
            print(f"  skip  {dest.name}")
        return

    url = item.get("links", {}).get("download", "")
    if not url:
        raise RuntimeError(f"OSF entry for {dest.name} has no download link.")
    if verbose:
        size = f" ({expected:,} bytes)" if expected else ""
        print(f"  fetch {dest.name}{size}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temporary name and rename only on a complete read, so an
    # interrupted download cannot leave a short file that the skip-check above
    # would happily accept next time.
    tmp = dest.with_suffix(dest.suffix + ".partial")
    req = urllib.request.Request(url)
    with (
        urllib.request.urlopen(req, timeout=300, context=ctx) as resp,
        open(tmp, "wb") as fh,
    ):
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
    tmp.replace(dest)


def fetch(
    subject: int = 1,
    data_dir: str | Path | None = None,
    *,
    noise_ellipses: bool = False,
    verbose: bool = True,
) -> dict[str, Path]:
    """Download one observer's files from OSF node ``k27js``.

    Parameters
    ----------
    subject : int, default=1
        Observer number. See :data:`SUBJECT_INITIALS`; ``1`` is observer CH.
    data_dir : str or Path, optional
        Destination root. Defaults to :func:`default_data_dir`.
    noise_ellipses : bool, default=False
        Also fetch ``Noise_ellipses_sub{N}.csv``. This is ~68 MB, and holds the
        published noise covariances on a 103x103 grid. Needed only to verify
        psyphy against the published covariance field exactly.
    verbose : bool, default=True
        Print progress.

    Returns
    -------
    dict[str, Path]
        Keys ``"trials"``, ``"weights"``, ``"thres_ellipses"``, and — when
        requested — ``"noise_ellipses"``.

    Notes
    -----
    Requires network access; nothing else in this module does. Files already
    present at the expected size are skipped, so repeated calls are cheap.


    """
    targets = {
        "trials": f"trial_data_pooled_by_type_sub{subject}.csv",
        "weights": f"Bestfit_W_sub{subject}.csv",
        "thres_ellipses": f"Thres_ellipses_sub{subject}.csv",
    }
    if noise_ellipses:
        targets["noise_ellipses"] = f"Noise_ellipses_sub{subject}.csv"

    root_dir = Path(data_dir) if data_dir is not None else default_data_dir()
    dest_dir = root_dir / f"sub{subject}"

    ctx = _ssl_context()
    if verbose:
        initials = SUBJECT_INITIALS.get(subject, "?")
        print(f"OSF {OSF_NODE} sub{subject} ({initials}) -> {dest_dir}")

    root_items = _list_folder(f"{_OSF_API}/nodes/{OSF_NODE}/files/osfstorage/", ctx)
    org_url = _subfolder_url(root_items, _ORGANIZED)
    if not org_url:
        raise RuntimeError(f"'{_ORGANIZED}' not found on OSF node {OSF_NODE}.")

    org_items = _list_folder(org_url, ctx)
    sub_url = _subfolder_url(org_items, f"sub{subject}")
    if not sub_url:
        available = sorted(
            i["attributes"]["name"]
            for i in org_items
            if i.get("attributes", {}).get("kind") == "folder"
        )
        raise ValueError(f"sub{subject} not found on OSF. Available: {available}")

    sub_items = _list_folder(sub_url, ctx)
    paths: dict[str, Path] = {}
    for key, name in targets.items():
        item = _find_file(sub_items, name)
        if item is None:
            raise RuntimeError(f"'{name}' not found under sub{subject} on OSF.")
        dest = dest_dir / name
        _download(item, dest, ctx, verbose=verbose)
        paths[key] = dest
    return paths


def fetch_calibration_matrix(
    data_dir: str | Path | None = None,
    *,
    file_date: str = CALIBRATION_DATE,
    verbose: bool = True,
) -> Path:
    """Download the 2-D W space -> RGB transformation matrix from OSF.

    Separate from :func:`fetch` because this file is not per-subject and does
    not live under the same OSF folder: it sits in
    ``Calibration and transformation/Transformation matrices/``, alongside the
    monitor colorimetry matrices, and one copy serves every observer.

    Parameters
    ----------
    data_dir : str or Path, optional
        Destination root. Defaults to :func:`default_data_dir`; the file lands
        in a ``calibration/`` subdirectory of it.
    file_date : str, default=:data:`CALIBRATION_DATE`
        Calibration date stamp embedded in the filename.
    verbose : bool, default=True
        Print progress.

    Returns
    -------
    Path
        Local path to ``M_2DWToRGB_DELL_<file_date>_copy.csv``.

    See Also
    --------
    load_calibration_matrix : read the downloaded file.
    w2d_to_rgb : apply it.
    """
    root_dir = Path(data_dir) if data_dir is not None else default_data_dir()
    dest_dir = root_dir / "calibration"

    ctx = _ssl_context()
    if verbose:
        print(f"OSF {OSF_NODE} calibration -> {dest_dir}")

    root_items = _list_folder(f"{_OSF_API}/nodes/{OSF_NODE}/files/osfstorage/", ctx)
    cal_url = _subfolder_url(root_items, _CALIBRATION)
    if not cal_url:
        raise RuntimeError(f"'{_CALIBRATION}' not found on OSF node {OSF_NODE}.")

    cal_items = _list_folder(cal_url, ctx)
    matrices_url = _subfolder_url(cal_items, _TRANSFORM_MATRICES)
    if not matrices_url:
        raise RuntimeError(
            f"'{_TRANSFORM_MATRICES}' not found under '{_CALIBRATION}' "
            f"on OSF node {OSF_NODE}."
        )

    matrix_items = _list_folder(matrices_url, ctx)
    pattern = f"M_2DWToRGB*{file_date}*.csv"
    matches = _match_files(matrix_items, pattern)
    if not matches:
        available = sorted(
            str(i["attributes"]["name"])
            for i in matrix_items
            if i.get("attributes", {}).get("kind") == "file"
        )
        raise RuntimeError(
            f"No file matching '{pattern}' under '{_TRANSFORM_MATRICES}'. "
            f"Available: {available}"
        )
    if len(matches) > 1:
        names = sorted(str(m["attributes"]["name"]) for m in matches)
        raise RuntimeError(
            f"'{pattern}' matched more than one file: {names}. "
            "Pass a more specific file_date."
        )

    item = matches[0]
    dest = dest_dir / str(item["attributes"]["name"])
    _download(item, dest, ctx, verbose=verbose)
    return dest


def load_calibration_matrix(path: str | Path) -> np.ndarray:
    """Load a 3x3 transformation matrix CSV.

    Parameters
    ----------
    path : str or Path
        Path to a ``M_*To*_DELL_*.csv`` file from
        :func:`fetch_calibration_matrix`.

    Returns
    -------
    np.ndarray, shape (3, 3)
        The matrix, oriented to be applied to **column** vectors from the
        left, exactly as stored.

    Notes
    -----
    Header-less, comma-delimited, three rows. The OSF folder's own README
    documents ``np.loadtxt`` as the intended reader.
    """
    M = np.loadtxt(path, delimiter=",", dtype=np.float64)
    if M.shape != (3, 3):
        raise ValueError(f"{path}: expected a 3x3 matrix, got shape {M.shape}.")
    return M


def w2d_to_rgb(coords: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Convert 2-D W space coordinates to linear monitor RGB.

    The transform is affine, handled by augmenting each coordinate with a
    constant 1 so that the matrix's third column supplies the DC offset
    between W-space origin and monitor gray::

        rgb = clip(M @ [w1, w2, 1], 0, 1)

    Parameters
    ----------
    coords : np.ndarray, shape (2,) or (N, 2)
        Stimulus coordinates in the paper's 2-D W space.
    M : np.ndarray, shape (3, 3)
        Matrix from :func:`load_calibration_matrix`.

    Returns
    -------
    np.ndarray, shape (3,) or (N, 3)
        Linear RGB in [0, 1], matching the input's dimensionality.

    Notes
    -----
    **These are linear, non-gamma-corrected RGB values**, which is what the
    paper's own plotting code produces and passes to matplotlib. Applying an
    sRGB gamma would give more perceptually even colours but would no longer
    match the published figures.

    **The matrix is monitor-specific** -- one Dell display, calibrated on the
    date in :data:`CALIBRATION_DATE`. It reproduces *this paper's* figures; it
    is not a general W-space-to-RGB conversion, and a different display needs
    its own calibration.

    Values outside the monitor gamut are clipped, so a returned colour on a
    channel boundary may not be the exact requested chromaticity. Over the
    paper's own [-0.7, 0.7] reference grid no clipping occurs.

    Examples
    --------
    >>> M = load_calibration_matrix(fetch_calibration_matrix())  # doctest: +SKIP
    >>> w2d_to_rgb(np.zeros(2), M)  # doctest: +SKIP
    array([0.5, 0.49999396, 0.5])
    """
    xy = np.asarray(coords, dtype=np.float64)
    single = xy.ndim == 1
    if single:
        xy = xy[None, :]
    if xy.ndim != 2 or xy.shape[-1] != 2:
        raise ValueError(
            f"coords must have shape (2,) or (N, 2), got {np.shape(coords)}."
        )

    augmented = np.concatenate([xy, np.ones((xy.shape[0], 1))], axis=1)  # (N, 3)
    rgb = np.clip(augmented @ np.asarray(M, dtype=np.float64).T, 0.0, 1.0)
    return rgb[0] if single else rgb


# ----------------------------------------------------------------------
# PARSING
# ----------------------------------------------------------------------
def _parse_vec(text: str) -> tuple[float, ...]:
    """Parse a stringified coordinate such as ``"0.61996435,-0.67159398"``.

    ``ast.literal_eval`` already returns a tuple for a bare comma-separated list
    of numbers, so no hand-rolled parser is needed. Unlike ``eval`` it is safe
    on untrusted input.
    """
    return tuple(float(v) for v in ast.literal_eval(text))


def load_trials(
    path: str | Path,
    *,
    trial_types: Sequence[str] | None = FIT_TRIAL_TYPES,
    max_trials: int | None = None,
    seed: int = 0,
) -> TrialData:
    """Load ``trial_data_pooled_by_type_sub{N}.csv`` into a :class:`TrialData`.

    Parameters
    ----------
    path : str or Path
        Path to the pooled trial CSV (the ``"trials"`` entry from :func:`fetch`).
    trial_types : sequence of str, optional
        Keep only rows whose ``TrialType`` starts with one of these prefixes.
        Defaults to :data:`FIT_TRIAL_TYPES`, reproducing the published fit. Pass
        ``None`` to keep every row (AEPsych and MOCS).
    max_trials : int, optional
        Take a reproducible random subsample of this many trials. Useful for
        quick runs; ``None`` (default) keeps all matching trials.
    seed : int, default=0
        Seed for the subsample.

    Returns
    -------
    TrialData
        ``stimuli`` has shape (N, 2, 2) with slot 0 the reference and slot 1 the
        comparison; ``responses`` has shape (N, 1), 1 meaning a correct oddity
        identification. ``stimulus_names`` is ``("ref", "comp")``.

    Notes
    -----
    The oddity task presents three stimuli (reference, reference, comparison)
    but only two *distinct* means, so K=2. The duplication is encoded in the
    task likelihood, not the data container i.e., the same convention psyphy uses
    elsewhere, and the same one the paper uses (its data tuple is
    ``(y, mref, mprobe)``).
    """
    refs: list[tuple[float, ...]] = []
    comps: list[tuple[float, ...]] = []
    resps: list[int] = []
    prefixes = tuple(trial_types) if trial_types is not None else None

    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        missing = {"TrialType", "xref", "x1", "y"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{path} is missing expected column(s) {sorted(missing)}; "
                f"found {reader.fieldnames}."
            )
        for row in reader:
            if prefixes is not None and not row["TrialType"].startswith(prefixes):
                continue
            refs.append(_parse_vec(row["xref"]))
            comps.append(_parse_vec(row["x1"]))
            resps.append(int(row["y"]))

    if not refs:
        raise ValueError(
            f"No trials matched trial_types={trial_types!r} in {path}. "
            'Expected prefixes such as "AEPsych" or "MOCS".'
        )

    stimuli = np.stack([np.asarray(refs), np.asarray(comps)], axis=1)  # (N, 2, 2)
    responses = np.asarray(resps)[:, None]  # (N, 1)

    if max_trials is not None and max_trials < stimuli.shape[0]:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(stimuli.shape[0], size=max_trials, replace=False))
        stimuli, responses = stimuli[idx], responses[idx]

    return TrialData(
        stimuli=jnp.asarray(stimuli),
        responses=jnp.asarray(responses),
        stimulus_names=("ref", "comp"),
    )


def load_reference_W(path: str | Path, column: str = "W_org") -> jnp.ndarray:
    """Load published Chebyshev weights from ``Bestfit_W_sub{N}.csv``.

    Parameters
    ----------
    path : str or Path
        Path to the best-fit weights CSV (the ``"weights"`` entry from
        :func:`fetch`).
    column : str, default="W_org"
        Which fit to read. ``"W_org"`` is the main published fit; the file also
        carries 120 bootstrap columns named ``W_btst{b}_rank{r}``.

    Returns
    -------
    jnp.ndarray
        Shape (5, 5, 2, 3) — exactly psyphy's ``params["W"]`` layout for
        ``basis_degree=4, input_dim=2, extra_embedding_dims=1``, so it can be
        used directly as ``{"W": load_reference_W(...)}`` with no reshaping.

    Raises
    ------
    ValueError
        If the index column is malformed, the requested column is absent, index
        keys are duplicated, or the table is not dense.

    Notes
    -----
    Enable ``jax_enable_x64`` before calling if you intend to compare against
    published covariances; in float32 the agreement floors around 1e-7.
    """
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        index_col = fields[0] if fields else None
        if index_col != "i,j,k,l":
            raise ValueError(
                f"{path}: expected first column 'i,j,k,l', got {index_col!r}."
            )
        if column not in fields:
            available = [f for f in fields if f != index_col]
            raise ValueError(
                f"{path}: column {column!r} not found. "
                f"{len(available)} available, e.g. {available[:3]}."
            )
        entries = [
            (tuple(int(v) for v in row[index_col].split(",")), float(row[column]))
            for row in reader
        ]

    if not entries:
        raise ValueError(f"{path}: no rows.")

    shape = tuple(max(idx[axis] for idx, _ in entries) + 1 for axis in range(4))
    expected = int(np.prod(shape))
    if len(entries) != expected:
        raise ValueError(
            f"{path}: got {len(entries)} rows but the index range implies shape "
            f"{shape} ({expected} entries) -- the table is not dense."
        )

    # A row count alone is not enough: one duplicated index key plus one missing
    # key preserves both the count and the per-axis maxima, and the missing cell
    # would then hold whatever the allocation happened to contain. Check
    # uniqueness, and fill with NaN so any remaining gap is loud rather than
    # plausible-looking.
    indices = [idx for idx, _ in entries]
    if len(set(indices)) != len(indices):
        raise ValueError(f"{path}: duplicate index keys in column {column!r}.")

    W = np.full(shape, np.nan, dtype=np.float64)
    for idx, value in entries:
        W[idx] = value
    if not np.isfinite(W).all():
        n_bad = int((~np.isfinite(W)).sum())
        raise ValueError(f"{path}: {n_bad} unfilled entries in column {column!r}.")
    return jnp.asarray(W)


def load_sigma_table(
    path: str | Path, *, value_column: str | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Load a published covariance table.

    Handles both ``Thres_ellipses_sub{N}.csv`` (49 rows, the 7x7 reference grid,
    66.7%-correct threshold covariances) and ``Noise_ellipses_sub{N}.csv``
    (10 609 rows, a 103x103 grid, model noise covariances). They share a layout:
    one row per reference location, a first column holding the coordinate as
    ``"x,y"``, then one column per fit holding a 2x2 matrix as
    ``"[[a,b],[c,d]]"``.

    Parameters
    ----------
    path : str or Path
        Path to either ellipse CSV.
    value_column : str, optional
        Which fit to read. Defaults to the file's main fit — whichever
        ``*_org`` column is present.

    Returns
    -------
    coords : np.ndarray, shape (M, 2)
        Reference locations in W space.
    Sigmas : np.ndarray, shape (M, 2, 2)
        Covariance matrix at each location.

    Notes
    -----
    Because every row carries its own coordinate, the row/column ordering
    ambiguity that affects the published ``.pkl`` files — which store bare
    (7, 7, 2, 2) arrays with no coordinates attached — does not arise here.
    """
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        coord_col = fields[0] if fields else None
        if coord_col not in ("grid_ref", "grid_ref_fine"):
            raise ValueError(
                f"{path}: expected first column 'grid_ref' or 'grid_ref_fine', "
                f"got {coord_col!r}."
            )
        if value_column is None:
            candidates = [f for f in fields if f.endswith("_org")]
            if not candidates:
                raise ValueError(f"{path}: no '*_org' column found in {fields[:4]}.")
            value_column = candidates[0]
        elif value_column not in fields:
            raise ValueError(f"{path}: column {value_column!r} not found.")

        coords, sigmas = [], []
        for row in reader:
            coords.append(_parse_vec(row[coord_col]))
            sigmas.append(ast.literal_eval(row[value_column]))

    if not coords:
        raise ValueError(f"{path}: no rows.")
    return np.asarray(coords, dtype=np.float64), np.asarray(sigmas, dtype=np.float64)


# ----------------------------------------------------------------------
# MODEL
# ----------------------------------------------------------------------
def build_paper_model(
    *, mc_samples: int | None = None, bandwidth: float | None = None
) -> Any:
    """Build a :class:`~psyphy.model.wppm.WPPM` configured as in the paper.

    Every setting comes from :data:`PAPER_HYPERPARAMS`, itself a transcription
    of ``ellipsoids/fit/fit_4d_human.py`` SECTION 4. Keeping it here rather than
    in the example script gives the tutorial and the regression test a single
    source of truth for the paper-to-psyphy mapping.

    Parameters
    ----------
    mc_samples : int, optional
        Monte Carlo samples per trial in the oddity likelihood. Defaults to the
        paper's 2000, which is expensive; lower it for quick runs.
    bandwidth : float, optional
        Logistic smoothing bandwidth. Defaults to the paper's 5e-3.

    Returns
    -------
    WPPM
        Ready to fit, or to wrap in a
        :class:`~psyphy.model.covariance_field.WPPMCovarianceField` for
        evaluating covariances from published weights.

    Notes
    -----
    Imported lazily so that ``psyphy.data`` does not depend on ``psyphy.model``
    at import time.
    """
    from psyphy.model.likelihood import OddityTask, OddityTaskConfig
    from psyphy.model.noise import GaussianNoise
    from psyphy.model.prior import Prior
    from psyphy.model.wppm import WPPM

    prior = Prior(
        input_dim=PAPER_HYPERPARAMS["input_dim"],
        basis_degree=PAPER_HYPERPARAMS["basis_degree"],
        extra_embedding_dims=PAPER_HYPERPARAMS["extra_dims"],
        variance_scale=PAPER_HYPERPARAMS["variance_scale"],
        decay_rate=PAPER_HYPERPARAMS["decay_rate"],
    )
    task = OddityTask(
        config=OddityTaskConfig(
            num_samples=int(
                mc_samples
                if mc_samples is not None
                else PAPER_HYPERPARAMS["mc_samples"]
            ),
            bandwidth=float(
                bandwidth if bandwidth is not None else PAPER_HYPERPARAMS["bandwidth"]
            ),
        )
    )
    return WPPM(
        input_dim=PAPER_HYPERPARAMS["input_dim"],
        extra_dims=PAPER_HYPERPARAMS["extra_dims"],
        prior=prior,
        likelihood=task,
        noise=GaussianNoise(),
        diag_term=PAPER_HYPERPARAMS["diag_term"],
    )


def _cli() -> int:
    """``python -m psyphy.data.published.hong2025 --subject 1`` convenience entry."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download the Hong et al. (2025) data."
    )
    parser.add_argument("--subject", type=int, nargs="+", default=[1])
    parser.add_argument("--data-dir", default=None)

    # Noise_ellipses corresponds to Fig. S3 (Supplements) in Hong et al 2025.
    # The internal noise ellipses are larger in file size and are not
    # necessary to plot the threshold contours.
    parser.add_argument(
        "--noise-ellipses",
        action="store_true",
        help="also fetch Noise_ellipses_sub{N}.csv (~68 MB)",
    )
    args = parser.parse_args()
    for subject in args.subject:
        fetch(
            subject,
            args.data_dir,
            noise_ellipses=args.noise_ellipses,
        )
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
