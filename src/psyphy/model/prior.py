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
from typing import Dict, Optional

import jax.numpy as jnp
import jax.random as jr

Params = Dict[str, jnp.ndarray]


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
    def default(cls, input_dim: int, scale: float = 0.5) -> "Prior":
        """Convenience constructor with MVP defaults."""
        return cls(input_dim=input_dim, scale=scale)

    def sample_params(self, key) -> Params:
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


@dataclass
class WishartPrior:
    """
    Wishart prior over full SPD covariance Σ via a Cholesky parameterization.

    Parameters
    ----------
    input_dim : int
        Dimensionality p of the covariance Σ (Σ is p x p).
    nu : float, default=3.0
        Degrees of freedom (must be > p - 1). Controls prior strength.
    V : jnp.ndarray | None, default=None
        Scale matrix (p x p, SPD). If None, uses Identity.
        Note E[Σ] = nu * V.
    init_scale : float, default=0.1
        Stddev for initializing unconstrained Cholesky parameters.
    """

    input_dim: int
    nu: float = 3.0
    V: Optional[jnp.ndarray] = None
    init_scale: float = 0.1

    @classmethod
    def default(cls, input_dim: int, nu: float = 3.0, V: Optional[jnp.ndarray] = None, init_scale: float = 0.1) -> "WishartPrior":
        return cls(input_dim=input_dim, nu=nu, V=V, init_scale=init_scale)

    def _pack_size(self) -> int:
        p = int(self.input_dim)
        return p * (p + 1) // 2

    def sample_params(self, key) -> Params:
        size = self._pack_size()
        vec = jr.normal(key, shape=(size,)) * self.init_scale
        # initialize diagonals near zero so exp(diag) ≈ 1
        # leave off-diagonals centered at 0
        return {"chol_params": vec}

    def log_prob(self, params: Params) -> jnp.ndarray:
        p = int(self.input_dim)
        vec = params["chol_params"]
        # Reconstruct L from packed vector
        L = jnp.zeros((p, p), dtype=vec.dtype)
        idx = 0
        for i in range(p):
            for j in range(i):
                L = L.at[i, j].set(vec[idx])
                idx += 1
            L = L.at[i, i].set(jnp.exp(vec[idx]))
            idx += 1
        # Σ and log|Σ|
        Sigma = L @ L.T
        # Safe log|Σ| using diagonal of L; add small epsilon to avoid log(0)
        eps = jnp.array(1e-12, dtype=Sigma.dtype)
        sum_log_diag_L = jnp.sum(jnp.log(jnp.diag(L) + eps))
        log_det_Sigma = 2.0 * sum_log_diag_L
        # Scale matrix and trace term
        V = self.V if self.V is not None else jnp.eye(p, dtype=Sigma.dtype)
        Vinv = jnp.linalg.inv(V)
        trace_term = jnp.trace(Vinv @ Sigma)
        # Wishart log density up to constant terms
        coeff = 0.5 * (self.nu - p - 1.0)
        logdet_term = coeff * log_det_Sigma
        return logdet_term - 0.5 * trace_term
