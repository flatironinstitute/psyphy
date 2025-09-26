"""
psyphy: A package for psychophysical modeling and adaptive trial placement.


Top-level API for psychophysical modeling and adaptive trial placement.

Exposes the most commonly used classes and functions so users can do:

    from psyphy import WPPM, Prior, OddityTask, ExperimentSession
    from psyphy.trial_placement import InfoGainPlacement

See submodules for specialized components:
- psyphy.data          Trial data and transformations
- psyphy.model         WPPM, priors, noise, task likelihoods
- psyphy.inference     Inference engines (MAP, Langevin, Laplace, Slice)
- psyphy.posterior     Posterior wrapper classes and diagnostics
- psyphy.trial_placement  Adaptive trial placement strategies
- psyphy.session       Experiment orchestration
- psyphy.utils         Shared helpers

"""

# Data
from .data.dataset import ResponseData, TrialBatch
from .inference.langevin import LangevinSampler
from .inference.laplace import LaplaceApproximation

# Inference
from .inference.map_optimizer import MAPOptimizer
from .model.noise import GaussianNoise, StudentTNoise
from .model.prior import Prior
from .model.task import OddityTask, TwoAFC
from .model.wppm import WPPM

# Posterior
from .posterior.posterior import Posterior

# Experiment orchestration
from .session.experiment_session import ExperimentSession

__all__ = [
    # Core model
    "WPPM", # needs task for likelihood and noise model
    "Prior",
    "OddityTask", 
    "TwoAFC",
    "GaussianNoise", #default noise model
    "StudentTNoise",
    # Inference
    "MAPOptimizer",
    "LangevinSampler",
    "LaplaceApproximation",
    # Posterior
    "Posterior",
    # Session orchestration
    "ExperimentSession",
    # Data handling
    "ResponseData",
    "TrialBatch",
]

"""
psyphy
======

Psychophysical modeling and adaptive trial placement.

This package implements the Wishart Process Psychophysical Model (WPPM)
with modular components for priors, task likelihoods, and noise models.

----------------------------------------------------------------------
Workflow
----------------------------------------------------------------------

Core design
-----------
1. WPPM (model/wppm.py)
   - Structural definition of the psychophysical model.
   - Maintains parameterization of local covariance fields.
   - Computes discriminability between stimuli.
   - Delegates trial likelihoods and predictions to the task.

2. Prior (model/prior.py)
   - Defines the distribution over model parameters.
   - MVP: Gaussian prior over diagonal log-variances.
   - Full WPPM mode: structured prior over basis weights and
     lengthscale-controlled covariance fields.

3. TaskLikelihood (model/task.py)
   - Encodes the psychophysical decision rule.
   - MVP: OddityTask (3AFC) and TwoAFC with sigmoid mappings.
   - Full WPPM mode: loglik and predict implemented via Monte Carlo
     observer simulations, using the noise model explicitly.

4. NoiseModel (model/noise.py)
   - Defines the distribution of internal representation noise.
   - MVP: GaussianNoise (zero mean, isotropic).
   - Full WPPM mode: StudentTNoise and beyond (heavy-tailed, anisotropic).

Data flow
---------
- A ResponseData object (psyphy.data) contains trial stimuli and responses.
- WPPM.init_params(prior) samples parameter initialization.
- Inference engines optimize the log posterior:
      log_posterior = task.loglik(params, data, model=WPPM, noise=NoiseModel)
                    + prior.log_prob(params)
- Posterior predictions (p(correct), threshold ellipses) are always obtained
  through WPPM delegating to TaskLikelihood.

Extensibility
-------------
- To add a new task: subclass TaskLikelihood, implement predict/loglik.
- To add a new noise model: subclass NoiseModel, implement logpdf/sample.
- To upgrade from MVP → Full WPPM mode: replace local_covariance and
  discriminability with basis-expansion Wishart process + MC simulation.

MVP vs Full WPPM mode
---------------------
- MVP is a diagonal-covariance, closed-form scaffold that runs out of the box.
- Full WPPM mode matches the published research model:
  - Smooth covariance fields (Wishart process priors).
  - Monte Carlo likelihood evaluation.
  - Explicit noise model in predictions.
- The API is stable across both: developers swap internals without changing
  external calls.

----------------------------------------------------------------------
"""
