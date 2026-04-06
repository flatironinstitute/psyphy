"""psyphy.model.likelihood
--------------------------

Task likelihoods for psychophysical experiments.

Modules
-------
base    : TaskLikelihood ABC (predict, loglik, simulate)
oddity  : OddityTaskConfig, OddityTask (MC-based)
neural  : NeuralSurrogateTask (ABC), NeuralSurrogateOddityTask (NN-based)

All public names are re-exported here so existing imports are unaffected:

    from psyphy.model.likelihood import TaskLikelihood, OddityTask, OddityTaskConfig
    from psyphy.model.likelihood import NeuralSurrogateTask, NeuralSurrogateOddityTask
"""

from .base import TaskLikelihood
from .neural import NeuralSurrogateOddityTask, NeuralSurrogateTask
from .oddity import OddityTask, OddityTaskConfig

__all__ = [
    # Base
    "TaskLikelihood",
    # MC-based
    "OddityTaskConfig",
    "OddityTask",
    # NN-based
    "NeuralSurrogateTask",
    "NeuralSurrogateOddityTask",
]
