"""
session
=======

Experiment orchestration.

This subpackage provides:
- ExperimentSession : a high-level controller that coordinates data collection,
  model fitting, posterior updates, and adaptive trial placement.

MVP implementation
------------------
- Wraps model, inference engine, and placement strategy.
- Stores data in a ResponseData object.
- Provides initialize(), update(), and next_batch() methods.

Full WPPM mode
--------------
- Will support richer workflows:
  * Batch vs online updates.
  * Integration with live experimental computers (e.g., resuming sessions after breaks).
"""

from .experiment_session import ExperimentSession

__all__ = ["ExperimentSession"]
