"""
trial_placement
===============

Non-acquisition-based trial placement strategies.

This module provides classical (non-Bayesian-optimization) placement strategies:

- GridPlacement: Fixed grid designs for systematic exploration
- SobolPlacement: Quasi-random low-discrepancy sequences (space-filling)
- StaircasePlacement: Adaptive staircase procedures (e.g., 1-up-2-down)

For acquisition-based adaptive designs (Bayesian optimization), use:
- psyphy.acquisition: Expected Improvement, UCB, Mutual Information
- See: psyphy.acquisition.optimize_acqf_discrete() for trial selection

Examples
--------
>>> # Fixed grid design
>>> from psyphy.trial_placement import GridPlacement
>>> placement = GridPlacement(n_points_per_dim=10, bounds=[[-1, 1], [-1, 1]])
>>> trials = placement.propose(batch_size=5)

>>> # Quasi-random exploration
>>> from psyphy.trial_placement import SobolPlacement
>>> placement = SobolPlacement(bounds=[[-1, 1], [-1, 1]])
>>> trials = placement.propose(batch_size=10)

>>> # For adaptive designs with acquisition functions, see:
>>> from psyphy.acquisition import expected_improvement, optimize_acqf_discrete
"""

from psyphy.trial_placement.grid import GridPlacement
from psyphy.trial_placement.sobol import SobolPlacement
from psyphy.trial_placement.staircase import StaircasePlacement

__all__ = [
    "GridPlacement",
    "SobolPlacement",
    "StaircasePlacement",
]
