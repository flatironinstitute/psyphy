"""
viz
===

Plotting helpers for psyphy, kept separate from the modeling code.

This subpackage provides:

- :func:`~psyphy.viz.ellipses.plot_ellipses` -- draw one or more fields of
  2-D covariance ellipses (noise fields, threshold contours, posterior draws).
- :func:`~psyphy.viz.geometry.ellipse_segments` -- covariance to polyline
  vertices, with no plotting backend.
- :func:`~psyphy.viz.geometry.auto_scale` -- a grid-aware size factor.

Design
------
Free functions taking **arrays**, not methods on model objects: much of what
you want to draw (published reference data, for instance) never passes through
a psyphy model, and should still be plottable.

Geometry and drawing are split. ``geometry`` depends only on numpy and scipy,
so it is usable and testable without matplotlib; ``ellipses`` imports
matplotlib lazily, inside the functions, so ``import psyphy`` stays free of it.
Install the backend with ``pip install 'psyphy[viz]'``.

Naming is geometric: a *covariance* field and a *threshold* field
are different quantities that happen to share a drawing routine, so the
function is named for the shape it draws, not for either model concept.

Future extensions
-----------------
- Ellipsoid rendering for 3-D stimulus spaces (currently raises).
- Convenience wrappers that take a covariance field or predictive posterior
  directly and delegate here.
"""

from psyphy.viz.ellipses import plot_ellipses
from psyphy.viz.geometry import auto_scale, ellipse_segments

__all__ = ["auto_scale", "ellipse_segments", "plot_ellipses"]
