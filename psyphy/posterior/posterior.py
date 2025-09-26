"""
posterior.py
------------

Posterior representation for WPPM.

MVP implementation:
- Posterior = MAPPosterior (point estimate only).
- Stores one parameter set and delegates predictions to WPPM.

Design note
-----------
- This class inherits from BasePosterior.
- In the future, other subclasses (LaplacePosterior, MCMCPosterior) will
  also inherit from BasePosterior, each implementing the same interface.
"""

from __future__ import annotations

import jax.numpy as jnp

from psyphy.posterior.base_posterior import BasePosterior


class Posterior(BasePosterior):
    """
    MVP Posterior (MAP only).

    Parameters
    ----------
    params : dict
        MAP parameter dictionary.
    model : WPPM
        Model instance used for predictions.

    Notes
    -----
    - This is effectively a MAPPosterior.
    - Future subclasses (LaplacePosterior, MCMCPosterior) will extend
      BasePosterior with real sampling logic.
    """

    def __init__(self, params, model):
        self.params = params
        self.model = model

    # ------------------------------------------------------------------
    # ACCESSORS (expose internal information)
    # ------------------------------------------------------------------
    def MAP_params(self):
        """
        Return the MAP parameters.

        Returns
        -------
        dict
            Parameter dictionary.
        """
        return self.params

    def sample(self, num_samples: int = 1):
        """
        Draw parameter samples from the posterior.

        Parameters
        ----------
        num_samples : int, default=1
            Number of samples.

        Returns
        -------
        list of dict
            Parameter sets.

        MVP
        ---
        Returns MAP params repeated n times.

        Future
        ------
        - LaplacePosterior: draw from N(mean, cov).
        - MCMCPosterior: return stored samples.
        """
        return [self.params] * num_samples

    # ------------------------------------------------------------------
    # PREDICTIONS: delegates to model
    # ------------------------------------------------------------------
    def predict_prob(self, stimulus):
        """
        Predict probability of correct response for a stimulus.

        Parameters
        ----------
        stimulus : tuple
            (reference, probe).

        Returns
        -------
        jnp.ndarray
            Probability of correct response.

        Notes
        -----
        Delegates to WPPM.predict_prob().
        This is not recursion: Posterior calls WPPM’s method with stored params.
        """
        return self.model.predict_prob(self.params, stimulus)

    def predict_thresholds(self, reference, criterion: float = 0.667, directions: int = 16):
        """
        Predict discrimination threshold contour around a reference stimulus.

        Parameters
        ----------
        reference : jnp.ndarray
            Reference point in model space.
        criterion : float, default=0.667
            Target performance (e.g., 2/3 for oddity).
        directions : int, default=16
            Number of directions to probe.

        Returns
        -------
        jnp.ndarray
            Contour points (MVP: unit circle).

        MVP
        ---
        Returns a placeholder unit circle.

        Future
        ------
        - Search outward in each direction until performance crosses criterion.
        - Average over posterior samples (Laplace, MCMC) to get credible intervals.
        """
        angles = jnp.linspace(0, 2 * jnp.pi, directions, endpoint=False)
        contour = jnp.stack([reference + jnp.array([jnp.cos(a), jnp.sin(a)]) for a in angles])
        return contour
