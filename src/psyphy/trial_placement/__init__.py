"""
trial_placement
===============

Strategies for selecting the next set of trials in an experiment.

This subpackage provides:
- TrialPlacement : abstract base class.
- GridPlacement : fixed non-adaptive design.
- SobolPlacement : quasi-random low-discrepancy exploration.
- StaircasePlacement : classical adaptive rule (1-up-2-down).
- GreedyMAPPlacement : adaptive design based on MAP point estimate.
- InfoGainPlacement : adaptive design based on expected information gain.

MVP implementation
------------------
- Simple grid, Sobol, and staircase procedures.
- Greedy placement uses MAP only.
- InfoGain uses entropy-style heuristic with placeholder logic.

Full WPPM mode
--------------
- InfoGainPlacement will integrate with posterior.sample() from
  LaplacePosterior or MCMCPosterior.
- StaircasePlacement can be extended to multi-dimensional, task-aware rules.
- Hybrid strategies: exploration (Sobol) -> exploitation (InfoGain).
"""

from .base import TrialPlacement
from .greedy_map import GreedyMAPPlacement
from .grid import GridPlacement
from .info_gain import InfoGainPlacement
from .sobol import SobolPlacement
from .staircase import StaircasePlacement

__all__ = [
    "TrialPlacement",
    "GridPlacement",
    "SobolPlacement",
    "StaircasePlacement",
    "GreedyMAPPlacement",
    "InfoGainPlacement",
]
