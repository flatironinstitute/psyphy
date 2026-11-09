"""
base.py
-------

Base class for psychophysical models.

Design
------
Models in PsyPhy are **stateless configuration objects**.
They consist of a:
1. Prior (init_params) and a
2. Likelihood (log_likelihood)

They do NOT hold data or fitted parameters.
Inference is handled by external engines (e.g. MAPOptimizer) which
return a Posterior object containing the results.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp

if TYPE_CHECKING:
    from psyphy.data import ResponseData


class Model(ABC):
    """
    Abstract base class for psychophysical models.

    Subclasses must implement:
    - init_params(key) --> sample initial parameters (Prior)
    - log_likelihood_from_data(params, data) --> compute likelihood
    """

    # ------------------------------------------------------------------
    # Abstract methods (must be implemented by subclasses)
    # ------------------------------------------------------------------

    @abstractmethod
    def init_params(self, key: Any) -> dict:  # jax.Array
        """
        Sample initial parameters from prior.

        Parameters
        ----------
        key : jax.Array
            PRNG key

        Returns
        -------
        dict
            Parameter PyTree
        """
        ...

    @abstractmethod
    def log_likelihood_from_data(self, params: dict, data: ResponseData) -> jnp.ndarray:
        """
        Compute log p(data | params).

        Parameters
        ----------
        params : dict
            Model parameters
        data : ResponseData
            Observed trials

        Returns
        -------
        jnp.ndarray, shape ()
            Log-likelihood (scalar)
        """
        ...
