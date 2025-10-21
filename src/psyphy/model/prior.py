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
from typing import Any

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
    basis_degree : int | None, default=None
        Degree of Chebyshev basis for Wishart process.
        If None, uses MVP mode with log_diag parameters.
        If set, uses Wishart mode with W coefficients.
    variance_scale : float, default=1.0
        Prior variance for degree-0 (constant) coefficient in Wishart mode.
        Controls overall scale of covariances.
    decay_rate : float, default=0.5
        Geometric decay rate for prior variance over higher-degree coefficients.
        Prior variance for degree-d coefficient = variance_scale * (decay_rate^d).
        Smaller decay_rate → stronger smoothness prior.
    lengthscale : float, default=1.0
        Alias for decay_rate (kept for backward compatibility).
        If both specified, decay_rate takes precedence.
    extra_embedding_dims : int, default=0
        Additional latent dimensions in U matrices beyond input dimensions.
        Allows richer ellipsoid shapes in Wishart mode.
    """

    input_dim: int
    scale: float = 0.5
    basis_degree: int | None = None
    variance_scale: float = 1.0
    decay_rate: float = 0.5
    lengthscale: float = 1.0  # Kept for compatibility
    extra_embedding_dims: int = 0

    def __post_init__(self):
        """Validate and normalize parameters."""
        if self.basis_degree is not None and self.basis_degree < 0:
            raise ValueError("basis_degree must be non-negative or None")

        # Use lengthscale as decay_rate if only lengthscale specified
        # (for backward compatibility with old API)
        if self.decay_rate == 0.5 and self.lengthscale != 1.0:
            self.decay_rate = self.lengthscale

    @classmethod
    def default(cls, input_dim: int, scale: float = 0.5) -> Prior:
        """Convenience constructor with MVP defaults."""
        return cls(input_dim=input_dim, scale=scale)

    def _compute_basis_degree_grid(self) -> jnp.ndarray:
        """
        Compute total degree for each basis function coefficient.

        For 2D: basis functions are products φᵢ(x₁) * φⱼ(x₂)
        Total degree of φᵢⱼ is i + j.

        Note: For basis_degree=d, we have (d+1) basis functions [T_0, ..., T_d] per dimension,
        so the grid has shape (d+1, d+1) for 2D.

        Returns
        -------
        basis_degrees : jnp.ndarray
            Array of total degrees, shape (degree+1, degree+1) for 2D
        """
        if self.basis_degree is None:
            return jnp.array([])

        if self.input_dim == 2:
            # 2D: degree[i,j] = i + j for i,j in 0..basis_degree
            basis_degrees = (
                jnp.arange(self.basis_degree + 1)[:, None]
                + jnp.arange(self.basis_degree + 1)[None, :]
            )
        elif self.input_dim == 3:
            # 3D: degree[i,j,k] = i + j + k
            basis_degrees = (
                jnp.arange(self.basis_degree + 1)[:, None, None]
                + jnp.arange(self.basis_degree + 1)[None, :, None]
                + jnp.arange(self.basis_degree + 1)[None, None, :]
            )
        else:
            raise NotImplementedError(
                f"Wishart process only supports 2D and 3D. Got input_dim={self.input_dim}"
            )

        return basis_degrees

    def _compute_W_prior_variances(self) -> jnp.ndarray:
        """
        Compute prior variances for W coefficients with smoothness prior.

        Prior variance = variance_scale * (decay_rate^total_degree)
        This enforces smoothness: higher frequency components have lower variance.

        Returns
        -------
        variances : jnp.ndarray
            Prior variances, shape matches basis degree grid
        """
        basis_degrees = self._compute_basis_degree_grid()
        return self.variance_scale * (self.decay_rate**basis_degrees)

    def sample_params(self, key: Any) -> Params:
        """
        Sample initial parameters from the prior.

        MVP mode (basis_degree=None):
            Returns {"log_diag": shape (input_dim,)}

        Wishart mode (basis_degree set):
            Returns {"W": shape (degree, degree, embedding_dim, embedding_dim + extra_dims)}
            for 2D, where embedding_dim = input_dim * (basis_degree + 1)

        Parameters
        ----------
        key : JAX random key

        Returns
        -------
        params : dict
            Parameter dictionary
        """
        if self.basis_degree is None:
            # MVP mode: simple diagonal covariance
            log_diag = jr.normal(key, shape=(self.input_dim,)) * self.scale
            return {"log_diag": log_diag}

        # Wishart mode: basis function coefficients W
        variances = self._compute_W_prior_variances()
        embedding_dim = self.input_dim * (self.basis_degree + 1)

        if self.input_dim == 2:
            # Sample W ~ Normal(0, variances) for each matrix entry
            # Shape: (degree+1, degree+1, embedding_dim, embedding_dim + extra_dims)
            # Note: degree+1 to match number of basis functions [T_0, ..., T_degree]
            W = jnp.sqrt(variances)[:, :, None, None] * jr.normal(
                key,
                shape=(
                    self.basis_degree + 1,
                    self.basis_degree + 1,
                    embedding_dim,
                    embedding_dim + self.extra_embedding_dims,
                ),
            )
        elif self.input_dim == 3:
            # Shape: (degree+1, degree+1, degree+1, embedding_dim, embedding_dim + extra_dims)
            W = jnp.sqrt(variances)[:, :, :, None, None] * jr.normal(
                key,
                shape=(
                    self.basis_degree + 1,
                    self.basis_degree + 1,
                    self.basis_degree + 1,
                    embedding_dim,
                    embedding_dim + self.extra_embedding_dims,
                ),
            )
        else:
            raise NotImplementedError(
                f"Wishart process only supports 2D and 3D. Got input_dim={self.input_dim}"
            )

        return {"W": W}

    def log_prob(self, params: Params) -> jnp.ndarray:
        """
        Compute log prior density (up to a constant)

        MVP mode:
            Isotropic Gaussian on log_diag

        Wishart mode:
            Gaussian prior on W with smoothness via decay_rate
            log p(W) = Σᵢⱼ log N(Wᵢⱼ | 0, σᵢⱼ²) where σᵢⱼ² = prior variance

        Parameters
        ----------
        params : dict
            Parameter dictionary

        Returns
        -------
        log_prob : float
            Log prior probability (up to normalizing constant)
        """
        if "log_diag" in params:
            # MVP mode
            log_diag = params["log_diag"]
            var = self.scale**2
            return -0.5 * jnp.sum((log_diag**2) / var)

        if "W" in params:
            # Wishart mode
            W = params["W"]
            variances = self._compute_W_prior_variances()

            # Gaussian log probability for each entry
            # log N(x | 0, σ²) = -0.5 * (x²/σ² + log(2πσ²))
            # Up to constant: -0.5 * x²/σ²

            if self.input_dim == 2:
                # Each W[i,j,:,:] ~ Normal(0, variance[i,j] * I)
                return -0.5 * jnp.sum((W**2) / (variances[:, :, None, None] + 1e-10))
            elif self.input_dim == 3:
                return -0.5 * jnp.sum((W**2) / (variances[:, :, :, None, None] + 1e-10))

        raise ValueError("params must contain either 'log_diag' (MVP) or 'W' (Wishart)")
