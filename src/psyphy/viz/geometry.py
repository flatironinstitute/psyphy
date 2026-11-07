"""
geometry
========

Covariance-to-ellipse geometry, with no plotting dependency.

These helpers turn stacks of 2x2 covariance matrices into polyline vertices
that any plotting backend can draw, and compute a magnification that makes a
field of ellipses legible on its own grid. They import only numpy and scipy,
both core psyphy dependencies, so they can be used (and tested) without
matplotlib installed.

See :mod:`psyphy.viz.ellipses` for the matplotlib drawing layer.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

__all__ = ["auto_scale", "ellipse_segments"]


def _validate_field(
    centers: np.ndarray, covs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Coerce and check one (centers, covariances) pair. 2-D only."""
    centers = np.asarray(centers, dtype=float)
    covs = np.asarray(covs, dtype=float)

    if centers.ndim != 2 or centers.shape[-1] != 2:
        raise ValueError(
            f"centers must have shape (n_points, 2); got {centers.shape}. "
            "Only 2-D stimulus spaces can be drawn as ellipses."
        )
    if covs.ndim != 3 or covs.shape[-2:] != (2, 2):
        if covs.ndim == 3 and covs.shape[-1] == covs.shape[-2]:
            raise NotImplementedError(
                f"covariances are {covs.shape[-1]}x{covs.shape[-1]}; only 2-D "
                "(2x2) fields can be drawn as ellipses. Higher dimensions "
                "would need ellipsoid rendering, which is not implemented."
            )
        raise ValueError(f"covs must have shape (n_points, 2, 2); got {covs.shape}.")
    if len(centers) != len(covs):
        raise ValueError(
            f"centers and covs must have the same length; got "
            f"{len(centers)} and {len(covs)}."
        )
    return centers, covs


def ellipse_segments(
    centers: np.ndarray,
    covs: np.ndarray,
    *,
    scale: float = 1.0,
    n_points: int = 100,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Convert a field of covariances into closed polylines.

    Each covariance is drawn as its 1-standard-deviation ellipse,
    ``{c + scale * L u : |u| = 1}`` where ``L`` is the Cholesky factor of the
    covariance. The ellipse's semi-axes are therefore ``scale * sqrt(lambda_i)``
    for eigenvalues ``lambda_i``.

    Parameters
    ----------
    centers : array_like, shape (n_points, 2)
        Ellipse centers, typically the reference stimuli.
    covs : array_like, shape (n_points, 2, 2)
        Covariance matrix per center.
    scale : float, default=1.0
        Multiplies every semi-axis. ``1.0`` draws true size; see
        :func:`auto_scale` for a grid-aware value.
    n_points : int, default=100
        Vertices per ellipse.

    Returns
    -------
    segments : list of np.ndarray, each shape (n_points, 2)
        One array per **positive-definite** covariance, in input order.
        Shorter than ``centers`` when some are skipped.
    valid : np.ndarray of bool, shape (n_points,)
        Which covariances were positive-definite and therefore drawn.

    Notes
    -----
    Non-positive-definite covariances are skipped rather than raising. Check
    ``valid.sum() < len(valid)`` and report it rather than ignore it: a
    silently thinned field is hard to notice by eye.

    Examples
    --------
    >>> import numpy as np
    >>> segs, valid = ellipse_segments(np.zeros((1, 2)), np.eye(2)[None])
    >>> bool(valid.all()), segs[0].shape
    (True, (100, 2))
    """
    centers, covs = _validate_field(centers, covs)

    valid = np.all(np.linalg.eigvalsh(covs) > 0, axis=-1)
    theta = np.linspace(0.0, 2 * np.pi, n_points)
    circle = np.vstack([np.cos(theta), np.sin(theta)])

    segments = [
        (c[:, None] + scale * (np.linalg.cholesky(S) @ circle)).T
        for c, S, ok in zip(centers, covs, valid, strict=True)
        if ok
    ]
    return segments, valid


def auto_scale(
    centers: np.ndarray, covs: np.ndarray, *, fraction: float = 0.35
) -> float:
    """Scale factor that fits a field of ellipses to its own grid.

    Sizes the *median* ellipse so its typical radius is ``fraction`` of the
    median nearest-neighbor spacing between centers. The same call fits a
    coarse 7x7 grid and a dense 103x103 one.

    Parameters
    ----------
    centers : array_like, shape (n_points, 2)
        Ellipse centers. At least two are required, since the spacing is a
        nearest-neighbor distance.
    covs : array_like, shape (n_points, 2, 2)
        Covariance matrix per center.
    fraction : float, default=0.35
        Target ratio of typical ellipse radius to typical center spacing.

    Returns
    -------
    float
        Multiplier to pass as ``scale``.

    Notes
    -----
    **This is not always a magnification.** On a dense grid the factor can be
    well below 1, shrinking ellipses so neighbors do not overlap. On the Hong
    et al. (2025) data it is ~1.24 for the 49-point threshold grid and ~0.22
    for the 10 609-point noise grid.

    Because it rescales, a figure drawn with it cannot be read for *absolute*
    size. When comparing several fields, compute the factor **once** and apply
    it to all of them, or relative sizes become meaningless;
    :func:`psyphy.viz.plot_ellipses` enforces this.

    Spacing comes from a KD-tree query, so no (n_points, n_points) distance
    matrix is built.
    """
    centers, covs = _validate_field(centers, covs)
    if len(centers) < 2:
        raise ValueError(
            "auto_scale needs at least 2 centers to measure spacing; got "
            f"{len(centers)}. Pass an explicit scale instead."
        )

    dists, _ = cKDTree(centers).query(centers, k=2)  # k=2: [0] is the point itself
    spacing = float(np.median(dists[:, 1]))
    typical_radius = float(np.median(np.sqrt(np.linalg.eigvalsh(covs).mean(-1))))

    if not np.isfinite(typical_radius) or typical_radius <= 0.0:
        raise ValueError(
            "could not determine a typical ellipse radius (non-finite or "
            "non-positive covariances); pass an explicit scale."
        )
    return fraction * spacing / typical_radius
