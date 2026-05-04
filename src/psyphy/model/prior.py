"""
prior.py
--------

Prior distributions for WPPM parameters

Hyperparameters:
    * variance_scale : global scaling factor for covariance magnitude
    * decay_rate     : smoothness controlling spatial variation
    * extra_embedding_dims : embedding dimension for basis expansions

Connections
-----------
- WPPM calls Prior.sample_params() to initialize model parameters
- WPPM adds Prior.log_prob(params) to task log-likelihoods to form the log posterior
- Prior will generate structured parameters for basis expansions
  and decay_rate-controlled smooth covariance fields
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
    basis_degree : int | None, default=None
        Degree of Chebyshev basis for Wishart process.
        If set, uses Wishart mode with W coefficients.
    variance_scale : float, default=1.0
        Prior variance for degree-0 (constant) coefficient in Wishart mode.
        Controls overall scale of covariances.
    decay_rate : float, default=0.5
        Geometric decay rate for prior variance over higher-degree coefficients.
        Prior variance for degree-d coefficient = variance_scale * (decay_rate^d).
        Smaller decay_rate -> stronger smoothness prior.
    extra_embedding_dims : int, default=0
        Additional latent dimensions in U matrices beyond input dimensions.
        Allows richer ellipsoid shapes in Wishart mode.
    """

    input_dim: int = 2
    basis_degree: int = 4  # | None = None
    variance_scale: float = 4e-3
    decay_rate: float = 0.4
    extra_embedding_dims: int = 1

    def __post_init__(self):
        """Validate and normalize parameters."""
        if self.basis_degree is not None and self.basis_degree < 0:
            raise ValueError("basis_degree must be non-negative or None")

    def _compute_basis_degree_grid(self) -> jnp.ndarray:
        """
        Compute total degree for each basis function coefficient.

        For 2D: basis functions are products φ_i(x_1) * φ_j(x_2)
        Total degree of φ_ij is i + j.

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
        elif self.input_dim == 1:
            # 1D: degree[i] = i for i in 0..basis_degree
            basis_degrees = jnp.arange(self.basis_degree + 1)  # shape (degree+1,)
        else:
            raise NotImplementedError(
                f"Wishart process only supports 1D, 2D and 3D. Got input_dim={self.input_dim}"
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


        Returns {"W": shape (degree+1, degree+1, input_dim, embedding_dim)}
        for 2D, where embedding_dim = input_dim + extra_embedding_dims

        Note: The 3rd dimension is input_dim (output space dimension).
        This matches the einsum in _compute_sqrt:
        U = einsum("ijde,ij->de", W, phi) where d indexes input_dim.

        Parameters
        ----------
        key : JAX random key

        Returns
        -------
        params : dict
            Parameter dictionary
        """
        if self.basis_degree is None:
            raise ValueError(
                "'basis_degree' is None; please set "
                "`Prior.basis_degree` to an integer >0."
            )

        # Basis function coefficients W
        variances = self._compute_W_prior_variances()
        embedding_dim = self.input_dim + self.extra_embedding_dims

        if self.input_dim == 2:
            # Sample W ~ Normal(0, variances) for each matrix entry
            # Shape: (degree+1, degree+1, input_dim, embedding_dim)
            # Note: degree+1 to match number of basis functions [T_0, ..., T_degree]
            W = jnp.sqrt(variances)[:, :, None, None] * jr.normal(
                key,
                shape=(
                    self.basis_degree + 1,
                    self.basis_degree + 1,
                    self.input_dim,
                    embedding_dim,
                ),
            )
        elif self.input_dim == 3:
            # Shape: (degree+1, degree+1, degree+1, input_dim, embedding_dim)
            W = jnp.sqrt(variances)[:, :, :, None, None] * jr.normal(
                key,
                shape=(
                    self.basis_degree + 1,
                    self.basis_degree + 1,
                    self.basis_degree + 1,
                    self.input_dim,
                    embedding_dim,
                ),
            )
        elif self.input_dim == 1:
            # Shape: (degree+1, input_dim, embedding_dim)
            W = jnp.sqrt(variances)[:, None, None] * jr.normal(
                key,
                shape=(self.basis_degree + 1, self.input_dim, embedding_dim),
            )
        else:
            raise NotImplementedError(
                f"Wishart process only supports 1D, 2D and 3D. Got input_dim={self.input_dim}"
            )

        return {"W": W}

    def log_prob(self, params: Params) -> jnp.ndarray:
        """
        Compute log prior density (up to a constant)

        Gaussian prior on W with smoothness via decay_rate
            log p(W) = Σ_ij log N(W_ij | 0, σ_ij^2) where σ_ij^2 = prior variance

        Parameters
        ----------
        params : dict
            Parameter dictionary

        Returns
        -------
        log_prob : float
            Log prior probability (up to normalizing constant)
        """

        if "W" in params:
            # Wishart mode
            W = params["W"]
            variances = self._compute_W_prior_variances()

            # Gaussian log probability for each entry
            # log N(x | 0, σ^2) = -0.5 * (x^2/σ^2 + log(2πσ^2))
            # Up to constant: -0.5 * x^2/σ^2

            if self.input_dim == 2:
                # Each W[i,j,:,:] ~ Normal(0, variance[i,j] * I)
                return -0.5 * jnp.sum((W**2) / (variances[:, :, None, None] + 1e-10))
            elif self.input_dim == 3:
                return -0.5 * jnp.sum((W**2) / (variances[:, :, :, None, None] + 1e-10))
            elif self.input_dim == 1:
                return -0.5 * jnp.sum((W**2) / (variances[:, None, None] + 1e-10))

        raise ValueError("params must contain weights 'W'")
