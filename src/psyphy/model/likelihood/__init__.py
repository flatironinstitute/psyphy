"""psyphy.model.likelihood

Task likelihoods for psychophysical experiments: the mapping from a model and a
set of stimuli to the likelihood of an observed response.

Likelihoods are grouped by response distribution. The log-likelihood is written
once per distribution, and a new task only implements ``predict``:

Modules
-------
base    : TaskLikelihood, BernoulliTaskLikelihood, GaussianTaskLikelihood
oddity  : OddityTaskConfig, OddityTask (3-AFC, Monte Carlo)

Every public name is re-exported here, so existing imports are unaffected::

    from psyphy.model.likelihood import OddityTask, TaskLikelihood
"""

from .base import (
    BernoulliTaskLikelihood,
    GaussianTaskLikelihood,
    TaskLikelihood,
)
from .oddity import OddityTask, OddityTaskConfig

__all__ = [
    # Base hierarchy
    "TaskLikelihood",
    "BernoulliTaskLikelihood",
    "GaussianTaskLikelihood",
    # Concrete tasks
    "OddityTaskConfig",
    "OddityTask",
]
