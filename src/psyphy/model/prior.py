"""
prior.py
--------

Prior distributions for WPPM parameters

MVP implementation:
- Gaussian prior over diagonal log-variances

Forward compatibility (Full WPPM mode):
- Exposes hyperparameters that will be used when the full Wishart Process
  covariance field is implemented:
    * variance_scale : global scaling factor for covariance magnitude
    * lengthscale    : smoothness/length-scale controlling spatial variation
    * extra_embedding_dims : embedding dimension for basis expansions

Connections
-----------
- WPPM calls Prior.sample_params() to initialize model parameters
- WPPM adds Prior.log_prob(params) to task log-likelihoods to form the log posterior
- In Full WPPM mode, Prior will generate structured parameters for basis expansions
  and lengthscale-controlled smooth covariance fields
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import jax.random as jr

Params = dict[str, jnp.ndarray]


@dataclass
class Prior:
    """
    Prior distribution over WPPM parameters

    Parameters
    ----------
    input_dim : int
        Dimensionality of the model space (same as WPPM.input_dim)
    scale : float, default=0.5
        Stddev of Gaussian prior for log_diag entries (MVP only).
    variance_scale : float, default=1.0
        Forward-compatible stub for Full WPPM mode. Will scale covariance magnitudes
    lengthscale : float, default=1.0
        Forward-compatible stub for Full WPPM mode;
        controls smoothness of covariance field:
        - small lengthscale --> rapid variation across space
        - large lengthscale --> smoother field, long-range correlations.
    extra_embedding_dims : int, default=0
        Forward-compatible stub for Full WPPM mode. Will expand embedding space.
    """

    input_dim: int
    scale: float = 0.5
    variance_scale: float = 1.0
    lengthscale: float = 1.0
    extra_embedding_dims: int = 0

    @classmethod
    def default(cls, input_dim: int, scale: float = 0.5) -> Prior:
        """Convenience constructor with MVP defaults."""
        return cls(input_dim=input_dim, scale=scale)

    def sample_params(self, key: jr.KeyArray) -> Params:
        """
        Sample initial parameters from the prior.

        MVP:
            Returns {"log_diag": shape (input_dim,)}.
        Full WPPM mode:
            Will also include basis weights, structured covariance params,
            and hyperparameters for GP (variance_scale, lengthscale).
        """
        log_diag = jr.normal(key, shape=(self.input_dim,)) * self.scale
        return {"log_diag": log_diag}

    def log_prob(self, params: Params) -> jnp.ndarray:
        """
        Compute log prior density (up to a constant)

        MVP:
            Isotropic Gaussian on log_diag
        Full WPPM mode:
            Will implement structured prior over basis weights and
            lengthscale-regularized covariance fields
        """
        log_diag = params["log_diag"]
        var = self.scale**2
        return -0.5 * jnp.sum((log_diag**2) / var)
