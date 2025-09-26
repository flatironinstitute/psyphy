"""
base_posterior.py
-----------------

Abstract base class for posterior representations in psyphy.

Defines the common interface for all posterior types:

- MAPPosterior      : point estimate only
- LaplacePosterior  : Gaussian approximation around MAP
- MCMCPosterior     : posterior samples from Langevin/HMC

Why this matters
----------------
Different inference methods yield very different posterior objects
(single point, Gaussian, samples). A common interface ensures that
downstream code (Session, TrialPlacement, ...) can interact with them
uniformly without having to worry about how they were computed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import jax.numpy as jnp


class BasePosterior(ABC):
    """
    Abstract base class for posterior wrappers.

    Notes
    -----
    - Each concrete posterior (MAP, Laplace, MCMC) must inherit from this class.
    - All must provide MAP_params, sample, predict_prob, and predict_thresholds.
    - This ensures polymorphism: different inference engines, same interface.
    """

    # ------------------------------------------------------------------
    # ACCESSORS: expose internal information
    # ------------------------------------------------------------------
    @abstractmethod
    def MAP_params(self):
        """
        Return a representative point estimate of parameters.

        Notes
        -----
        - MAPPosterior : return MAP params directly.
        - LaplacePosterior : return Gaussian mean (MAP).
        - MCMCPosterior : return posterior mean or MAP sample.
        """
        ...

    @abstractmethod
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

        Notes
        -----
        - MAPPosterior : repeat MAP params.
        - LaplacePosterior : draw from N(mean, cov).
        - MCMCPosterior : return stored samples.
        """
        ...

    # ------------------------------------------------------------------
    # PREDICTIONS (delegates to model)
    # ------------------------------------------------------------------
    @abstractmethod
    def predict_prob(self, stimulus) -> jnp.ndarray:
        """
        Predict probability of correct response for a stimulus.

        Parameters
        ----------
        stimulus : tuple
            (reference, probe) in model space.

        Returns
        -------
        jnp.ndarray
            Probability of correct response.

        Notes
        -----
        Delegates to WPPM, using stored parameters.
        """
        ...

    @abstractmethod
    def predict_thresholds(self, reference, criterion: float = 0.667, directions: int = 16):
        """
        Predict discrimination threshold contour around a reference.

        Parameters
        ----------
        reference : jnp.ndarray
            Reference point in model space.
        criterion : float, default=0.667
            Target performance (e.g., 2/3 for oddity).
        directions : int, default=16
            Number of directions (rays) to probe.

        Returns
        -------
        jnp.ndarray
            Array of threshold contour points.

        Notes
        -----
        - MVP : placeholder (e.g., unit circle).
        - Full : step outward in each direction until p(correct) crosses criterion.
        - With samples : average over posterior draws to get credible regions.
        """
        ...
