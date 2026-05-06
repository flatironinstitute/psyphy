"""
psyphy
======

Psychophysical modeling and adaptive trial placement.

This package implements the Wishart Process Psychophysical Model (WPPM)
with modular components for priors, task likelihoods, and noise models,
which can be fitted to incoming subject data and used to adaptively
select new trials to present to the subject next.
This is useful for efficiently estimating psychophysical parameters
(e.g. threshold contours) with minimal trials.

----------------------------------------------------------------------
Workflow
----------------------------------------------------------------------

Core design
-----------
1. WPPM (model/wppm.py):
   - Structural definition of the psychophysical model.
   - Maintains parameterization of local covariance fields.
   - Computes discriminability between stimuli.
   - Delegates trial likelihoods and predictions to the task.

2. Prior (model/prior.py):
   - Defines the distribution over model parameters.
   - WPPM: structured prior over basis weights and
     decay_rate-controlled covariance fields.

3. TaskLikelihood (model/likelihood.py):
   - Encodes the psychophysical decision rule.
   - WPPM: loglik and predict implemented via Monte Carlo
     observer simulations, using the noise model explicitly.

4. NoiseModel (model/noise.py):
   - Defines the distribution of internal representation noise.
   - WPPM: GaussianNoise or StudentTNoise option.

Unified import style
--------------------
Top-level (core models + session):
  from psyphy import WPPM, Prior, OddityTask, GaussianNoise, MAPOptimizer
  from psyphy import ExperimentSession, ResponseData, TrialBatch

Subpackages:
  from psyphy.model import WPPM, Prior, OddityTask, GaussianNoise, StudentTNoise
  from psyphy.inference import MAPOptimizer, LangevinSampler, LaplaceApproximation
  from psyphy.acquisition import expected_improvement, upper_confidence_bound, mutual_information
  from psyphy.acquisition import optimize_acqf, optimize_acqf_discrete, optimize_acqf_random
  from psyphy.trial_placement import GridPlacement, SobolPlacement
  from psyphy.utils import grid_candidates, sobol_candidates, custom_candidates, chebyshev_basis

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



----------------------------------------------------------------------
"""

# Data
# Re-export subpackages for unified import style (e.g., psyphy.model, psyphy.inference)
# from . import acquisition as acquisition
from . import data as data
from . import inference as inference
from . import model as model
from . import posterior as posterior

# from . import session as session
from . import trial_placement as trial_placement
from . import utils as utils
from .data.dataset import ResponseData, TrialBatch, TrialData
from .inference.langevin import LangevinSampler
from .inference.laplace import LaplaceApproximation

# Inference
from .inference.map_optimizer import MAPOptimizer
from .model.likelihood import OddityTask, OddityTaskConfig
from .model.noise import GaussianNoise, StudentTNoise
from .model.prior import Prior
from .model.wppm import WPPM

# Posterior
from .posterior.parameter_posterior import ParameterPosterior
from .posterior.posterior import MAPPosterior
from .posterior.predictive_posterior import PredictivePosterior, WPPMPredictivePosterior

# Experiment orchestration
# from .session.experiment_session import ExperimentSession

__all__ = [
    # Core model
    "WPPM",  # needs task for likelihood and noise model
    "Prior",
    "OddityTask",
    "OddityTaskConfig",
    "GaussianNoise",  # default noise model
    "StudentTNoise",
    # Inference
    "MAPOptimizer",
    "LangevinSampler",
    "LaplaceApproximation",
    # Data handling
    "ResponseData",
    "TrialBatch",
    "TrialData",
    # Subpackages
    "model",
    "inference",
    "posterior",
    "trial_placement",
    "utils",
    "data",
]
