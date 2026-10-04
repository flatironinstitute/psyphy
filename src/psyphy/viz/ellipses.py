"""
ellipses
========

Matplotlib drawing for fields of 2-D covariance ellipses.

matplotlib is an *optional* dependency (``pip install psyphy[viz]``) and is
imported inside the functions here, so importing psyphy never pulls it in.
The geometry these functions draw lives in :mod:`psyphy.viz.geometry` and needs
no plotting backend at all.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from .geometry import auto_scale, ellipse_segments

__all__ = ["plot_ellipses"]

_INSTALL_HINT = (
    "plotting requires matplotlib, which is an optional dependency of psyphy. "
    "Install it with:  pip install 'psyphy[viz]'"
)


def _require_matplotlib() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised only without mpl
        raise ImportError(_INSTALL_HINT) from exc
    return plt


def _as_fields(covs: Any) -> list[np.ndarray]:
    """Normalize the several accepted ``covs`` spellings to a list of fields."""
    if isinstance(covs, (list, tuple)):
        return [np.asarray(c, dtype=float) for c in covs]

    arr = np.asarray(covs, dtype=float)
    if arr.ndim == 4:  # (n_fields, n_points, 2, 2) -- e.g. posterior samples
        return list(arr)
    if arr.ndim == 3:  # (n_points, 2, 2) -- a single field
        return [arr]
    raise ValueError(
        f"covs has shape {arr.shape}, which is not a covariance field. Expected "
        "(n_points, 2, 2) for one field, (n_fields, n_points, 2, 2) for several "
        "(e.g. posterior samples), or a list of (n_points, 2, 2) arrays."
    )


def _per_field(value: Any, n_fields: int) -> list[Any]:
    """Broadcast a per-field option, leaving arrays (per-point colors) intact."""
    if value is None:
        return [None] * n_fields
    if isinstance(value, np.ndarray):
        return [value] * n_fields
    if isinstance(value, (list, tuple)) and len(value) == n_fields and n_fields > 1:
        return list(value)
    return [value] * n_fields


def plot_ellipses(
    centers: np.ndarray,
    covs: Any,
    *,
    ax: Any = None,
    scale: float | str = 1.0,
    colors: Any = None,
    labels: Any = None,
    linestyles: Any = None,
    linewidths: Any = None,
    alpha: Any = None,
    show_centers: bool = False,
    skip_non_pd: bool = True,
    return_scale: bool = False,
    n_points: int = 100,
) -> Any:
    """Draw one or more fields of 2-D covariance ellipses.

    Parameters
    ----------
    centers : array_like, shape (n_points, 2)
        Ellipse centers, shared by every field.
    covs : array_like or list
        The covariances to draw, in any of three spellings:

        - ``(n_points, 2, 2)`` -- a single field.
        - ``(n_samples, n_points, 2, 2)`` -- several fields, e.g. draws from
          :meth:`~psyphy.posterior.WPPMPredictivePosterior.rsample`.
        - a list of ``(n_points, 2, 2)`` arrays -- several fields to overlay,
          e.g. a published field and a fitted one.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. A new figure and axes are created when omitted.
        Nothing is ever saved or shown; the caller owns the figure.
    scale : float or "auto", default=1.0
        Multiplies every semi-axis. ``1.0`` draws true size. ``"auto"`` calls
        :func:`~psyphy.viz.geometry.auto_scale` on the **first** field and
        applies that one factor to all of them, so relative sizes stay
        comparable. Note ``"auto"`` may shrink as well as magnify.
    colors : color or array or list, optional
        A matplotlib color applied to a whole field, or an ``(n_points, 3|4)``
        array giving one color per ellipse (e.g. from
        :func:`psyphy.data.published.hong2025.w2d_to_rgb`). Pass a list of
        length ``n_fields`` to set each field separately. Defaults to
        matplotlib's color cycle.
    labels : str or list of str, optional
        Legend entries, one per field. A legend is drawn only if any is given.
    linestyles, linewidths, alpha : optional
        Per-field line styling, broadcast the same way as ``colors``.
    show_centers : bool, default=False
        Also scatter the center points.
    skip_non_pd : bool, default=True
        Skip non-positive-definite covariances (and warn, naming the count)
        rather than raising.
    return_scale : bool, default=False
        Also return the scale factor actually used -- worth doing with
        ``scale="auto"``, so the figure can report its own magnification.
    n_points : int, default=100
        Vertices per ellipse.

    Returns
    -------
    matplotlib.axes.Axes, or (Axes, float) when ``return_scale=True``

    Examples
    --------
    One field, true size::

        ax = plot_ellipses(coords, Sigma)

    Two fields overlaid, sized to the grid, reporting the factor::

        ax, s = plot_ellipses(
            coords,
            [Sigma_published, Sigma_fit],
            scale="auto",
            colors=["black", "crimson"],
            linestyles=["--", "-"],
            labels=["published", "psyphy"],
            return_scale=True,
        )

    Posterior draws as an uncertainty band::

        ax = plot_ellipses(coords, cov_samples, alpha=0.15)
    """
    plt = _require_matplotlib()
    from matplotlib.collections import LineCollection

    centers = np.asarray(centers, dtype=float)
    fields = _as_fields(covs)
    n_fields = len(fields)
    if n_fields == 0:
        raise ValueError("covs contained no covariance fields.")

    if isinstance(scale, str):
        if scale != "auto":
            raise ValueError(f"scale must be a float or 'auto'; got {scale!r}.")
        # One factor from the first field, applied to all: independently scaled
        # fields cannot be compared by eye.
        scale_value = auto_scale(centers, fields[0])
    else:
        # Deliberately one scalar for every field, never one per field:
        # independently magnified fields cannot be compared by eye, which is
        # the whole point of drawing them on shared axes.
        try:
            scale_value = float(scale)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"scale must be a single number or 'auto'; got {type(scale).__name__}. "
                "One factor is applied to every field on purpose -- scaling them "
                "independently would make unlike fields look alike."
            ) from exc

    colors_pf = _per_field(colors, n_fields)
    labels_pf = _per_field(labels, n_fields)
    styles_pf = _per_field(linestyles, n_fields)
    widths_pf = _per_field(linewidths, n_fields)
    alpha_pf = _per_field(alpha, n_fields)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
    n_skipped_total = 0

    for i, field in enumerate(fields):
        segments, valid = ellipse_segments(
            centers, field, scale=scale_value, n_points=n_points
        )
        n_skipped = int((~valid).sum())
        if n_skipped and not skip_non_pd:
            raise ValueError(
                f"field {i} has {n_skipped} non-positive-definite covariance(s); "
                "pass skip_non_pd=True to draw the rest."
            )
        n_skipped_total += n_skipped

        color = colors_pf[i] if colors_pf[i] is not None else cycle[i % len(cycle)]
        seg_colors = (
            np.asarray(color)[valid]
            if isinstance(color, np.ndarray) and np.ndim(color) == 2
            else color
        )

        ax.add_collection(
            LineCollection(
                segments,
                colors=seg_colors,
                linestyles=styles_pf[i] or "solid",
                linewidths=widths_pf[i] if widths_pf[i] is not None else 1.4,
                alpha=alpha_pf[i],
            )
        )
        if show_centers:
            ax.scatter(
                centers[:, 0],
                centers[:, 1],
                c=color,
                s=10,
                zorder=5,
                edgecolors="none",
            )
        if labels_pf[i] is not None:
            # Proxy artist: a LineCollection does not produce a legend entry.
            proxy_color = color if not isinstance(color, np.ndarray) else "0.3"
            ax.plot(
                [],
                [],
                color=proxy_color,
                linestyle=styles_pf[i] or "solid",
                linewidth=widths_pf[i] if widths_pf[i] is not None else 1.4,
                label=labels_pf[i],
            )

    if n_skipped_total:
        warnings.warn(
            f"{n_skipped_total} non-positive-definite covariance(s) were not "
            "drawn; the plotted field is incomplete.",
            stacklevel=2,
        )

    ax.autoscale_view()
    ax.set_aspect("equal")
    if any(x is not None for x in labels_pf):
        ax.legend(fontsize=8)

    return (ax, scale_value) if return_scale else ax
