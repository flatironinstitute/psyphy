"""
candidates.py
-------------

Utilities for generating candidate stimulus pools.

Definition
----------
A candidate pool is the set of all possible (reference, probe) pairs
that an adaptive placement strategy may select from.

Separation of concerns
----------------------
- Candidate generation (this module) defines *what* stimuli are possible.
- Trial placement strategies (e.g., GreedyMAPPlacement, InfoGainPlacement)
  define *which* of those candidates to present *next*.

Why this matters
----------------
- Researchers: think of the candidate pool as the "menu" of allowable trials.
- Developers: placement strategies should not generate candidates
  but only select from a given pool.

MVP implementation
------------------
- Grid-based candidates (probes on circles around a reference).
- Sobol sequence candidates (low-discrepancy exploration).
- Custom user-defined candidate pools.

Full WPPM mode
--------------
- Candidate generation could adaptively refine itself based on posterior
  uncertainty (e.g., dynamic grids).
- Candidate pools could be constrained by device gamut or subject-specific calibration.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

# Stimulus = (reference, probe)
Stimulus = tuple[jnp.ndarray, jnp.ndarray]


def grid_candidates(
    reference: jnp.ndarray, radii: list[float], directions: int = 16
) -> list[Stimulus]:
    """
    Generate grid-based candidate probes around a reference.

    Parameters
    ----------
    reference : jnp.ndarray, shape (D,)
        Reference stimulus in model space.
    radii : list of float
        Distances from reference to probe.
    directions : int, default=16
        Number of angular directions.

    Returns
    -------
    list of Stimulus
        Candidate (reference, probe) pairs.

    Notes
    -----
    - MVP: probes lie on concentric circles around reference.
    - Full WPPM mode: could adaptively refine grid around regions of
      high posterior uncertainty.
    """
    candidates = []
    angles = jnp.linspace(0, 2 * jnp.pi, directions, endpoint=False)
    for r in radii:
        probes = [reference + r * jnp.array([jnp.cos(a), jnp.sin(a)]) for a in angles]
        candidates.extend([(reference, p) for p in probes])
    return candidates


def sobol_candidates(
    reference: jnp.ndarray, n: int, bounds: list[tuple[float, float]], seed: int = 0
) -> list[Stimulus]:
    """
    Generate Sobol quasi-random candidates within bounds.

    Parameters
    ----------
    reference : jnp.ndarray, shape (D,)
        Reference stimulus.
    n : int
        Number of candidates to generate.
    bounds : list of (low, high)
        Bounds per dimension.
    seed : int, default=0
        Random seed.

    Returns
    -------
    list of Stimulus
        Candidate (reference, probe) pairs.

    Notes
    -----
    - MVP: uniform coverage of space using low-discrepancy Sobol sequence.
    - Full WPPM mode: Sobol could be used for initialization,
      then hand off to posterior-aware strategies.
    """
    from scipy.stats.qmc import Sobol

    dim = len(bounds)
    engine = Sobol(d=dim, scramble=True, seed=seed)
    raw = engine.random(n)
    scaled = [low + (high - low) * raw[:, i] for i, (low, high) in enumerate(bounds)]
    probes = np.stack(scaled, axis=-1)
    return [(reference, jnp.array(p)) for p in probes]


def custom_candidates(
    reference: jnp.ndarray, probe_list: list[jnp.ndarray]
) -> list[Stimulus]:
    """
    Wrap a user-defined list of probes into candidate pairs.

    Parameters
    ----------
    reference : jnp.ndarray, shape (D,)
        Reference stimulus.
    probe_list : list of jnp.ndarray
        Explicitly chosen probe vectors.

    Returns
    -------
    list of Stimulus
        Candidate (reference, probe) pairs.

    Notes
    -----
    - Useful when hardware constraints (monitor gamut, auditory frequencies)
      restrict the set of valid stimuli.
    - Full WPPM mode: this pool could be pruned or expanded dynamically
      depending on posterior fit quality.
    """
    return [(reference, probe) for probe in probe_list]
