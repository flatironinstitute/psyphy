"""
Plotting covariance ellipse fields with psyphy.viz
--------------------------------------------------

Three figures, each showing one thing `plot_ellipses` is for:

  1. a single field, at true size
  2. two fields overlaid, sized to the grid
  3. posterior draws of one field, as an uncertainty band

Everything here is synthetic, so it runs in about a second with no downloads.

Usage
-----
    python ellipse_plots.py
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
import numpy as np

if not os.environ.get("DISPLAY") and os.name != "nt":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# --8<-- [start:imports]
from psyphy.viz import plot_ellipses  # noqa: E402

# --8<-- [end:imports]

PLOTS_DIR = Path(__file__).parent / "plots"
RNG = np.random.default_rng(0)


# --8<-- [start:field]
def example_field(n_side: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """A grid of reference points and one 2x2 covariance at each.

    Ellipses grow with distance from the origin and are oriented radially --
    loosely the structure seen in color-discrimination data.
    """
    axis = np.linspace(-0.7, 0.7, n_side)
    centers = np.stack(np.meshgrid(axis, axis), axis=-1).reshape(-1, 2)

    radius = np.linalg.norm(centers, axis=1)
    angle = np.arctan2(centers[:, 1], centers[:, 0])
    covs = np.empty((len(centers), 2, 2))
    for i, (r, a) in enumerate(zip(radius, angle, strict=True)):
        lengths = np.diag([(0.02 + 0.05 * r) ** 2, (0.02 + 0.015 * r) ** 2])
        rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        covs[i] = rot @ lengths @ rot.T
    return centers, covs


# --8<-- [end:field]


def save(fig, name: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / name, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  saved {name}")


def main() -> int:
    # --8<-- [start:single]
    centers, covs = example_field()

    # One field. The default scale=1.0 draws true size. ellipses are small
    # relative to the grid.
    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    plot_ellipses(centers, covs, ax=ax, colors="C0", show_centers=True)
    # --8<-- [end:single]
    ax.set_title("One field, true size (scale=1.0)", fontsize=9)
    ax.grid(True, alpha=0.2)
    save(fig, "viz_single_field.png")

    # --8<-- [start:overlay]
    # Two fields overlaid. scale="auto" sizes them to the grid, and returns the
    # factor so the figure can report it. Both fields share that one factor --
    # otherwise their relative sizes would be meaningless.
    inflated = covs * 1.6**2  # 1.6x larger radii
    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    ax, scale = plot_ellipses(
        centers,
        [covs, inflated],
        ax=ax,
        scale="auto",
        colors=["black", "crimson"],
        linestyles=["--", "solid"],
        labels=["reference", "comparison"],
        return_scale=True,
    )
    # --8<-- [end:overlay]
    ax.set_title(f"Two fields, scale='auto' ({scale:.2f}x)", fontsize=9)
    ax.grid(True, alpha=0.2)
    save(fig, "viz_overlay.png")

    # --8<-- [start:samples]
    # A (n_samples, n_points, 2, 2) stack draws one translucent field per
    # sample. This is what posterior draws would look like.
    samples = covs[None] * RNG.normal(1.0, 0.12, size=(12, len(centers), 1, 1)) ** 2
    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    plot_ellipses(centers, samples, ax=ax, scale="auto", colors="C3", alpha=0.25)
    # --8<-- [end:samples]
    ax.set_title("12 draws of one field (alpha=0.25)", fontsize=9)
    ax.grid(True, alpha=0.2)
    save(fig, "viz_samples.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
