"""
langevin.py
-----------

Langevin samplers for posterior inference.

Implements:
- Overdamped (unadjusted) Langevin Algorithm (ULA)
- Underdamped Langevin (with BAOAB splitting scheme?)

Used for posterior-aware trial placement (InfoGain).


MVP implementation:
- Stub that returns an initial Posterior.
- Future: implement underdamped Langevin dynamics (e.g. BAOAB integrator).
"""

from psyphy.posterior.posterior import Posterior


class LangevinSampler:
    """
    Langevin sampler (stub).

    Parameters
    ----------
    steps : int, default=1000
        Number of Langevin steps.
    step_size : float, default=1e-3
        Integration step size.
    temperature : float, default=1.0
        Noise scale (temperature).
    """

    def __init__(self, steps: int = 1000, step_size: float = 1e-3, temperature: float = 1.0):
        self.steps = steps
        self.step_size = step_size
        self.temperature = temperature

    def fit(self, model, data) -> Posterior:
        """
        Fit model parameters with Langevin dynamics (stub).

        Parameters
        ----------
        model : WPPM
            Model instance.
        data : ResponseData
            Observed trials.

        Returns
        -------
        Posterior
            Posterior wrapper (MVP: params from init).
        """
        return Posterior(params=model.init_params(None), model=model)
