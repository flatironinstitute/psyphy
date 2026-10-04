"""The short version of the Hong et al. (2025) reproduction.

This file exists so the orientation block at the top of
``hong2025_reproduction.md`` is quoted from code that runs, rather than typed
into the Markdown and left to rot. It is deliberately minimal: no argparse, no
figures, no refit. For the full pipeline see ``hong2025_reproduction.py``.

Run it with::

    python hong2025_recipe.py

At the paper's settings this takes ~11 min on a laptop for all 49 reference
points. Pass ``--quick`` to do three of them instead, which is enough to prove
the code path works.
"""

from __future__ import annotations

import sys

# --8<-- [start:recipe]
import jax

jax.config.update("jax_enable_x64", True)  # the authors used float64

import jax.numpy as jnp  # noqa: E402

from psyphy.data.published import hong2025  # noqa: E402
from psyphy.posterior import (  # noqa: E402
    MAPPosterior,
    ThresholdConfig,
    WPPMPredictivePosterior,
)

paths = hong2025.fetch(subject=1)  # download from OSF
W = hong2025.load_reference_W(paths["weights"])  # the paper's fitted weights
# W = jnp.asarray(np.load("fits/hong2025_full_fit.npz")["W"]) # for full refit
coords, published = hong2025.load_sigma_table(paths["thres_ellipses"])

# Model: given weights W, how noisy is perception at each color?
model = hong2025.build_paper_model(mc_samples=2000)

# Parameter posterior: which W do we believe?
posterior = MAPPosterior({"W": W}, model)

# Search settings: how carefully to look for each threshold
# These are the paper's own: 16 directions, 1000 distances along each
config = ThresholdConfig(n_theta=16, n_length=1000)

# Predictive posterior: given what we believe about W, what do we predict here?
thresholds = WPPMPredictivePosterior(
    posterior,
    jnp.asarray(coords),  # reference points only
    n_samples=1,
    threshold_pred=True,  # ask for thresholds
    threshold_config=config,
).mean  # -> (49, 2, 2)

# This used the authors' weights, so it reproduces their published inversion
# rather than the figure at the top of the page. To go end to end instead, fit
# your own weights (see Refit) and change only where W comes from:
#
#   W = jnp.asarray(np.load("fits/hong2025_full_fit.npz")["W"])
#
# The model, the config and the call above are identical either way.
# --8<-- [end:recipe]


if __name__ == "__main__":
    if "--quick" in sys.argv:
        #  three reference points, so the path can be checked in
        # seconds rather than minutes.
        thresholds = WPPMPredictivePosterior(
            posterior,
            jnp.asarray(coords[:3]),
            n_samples=1,
            threshold_pred=True,
            threshold_config=config,
        ).mean
    print(f"thresholds: {thresholds.shape}")
    print(f"published : {published.shape}")
