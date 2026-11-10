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
- Prior.sample(n, key=...) draws n of them at once
- WPPM adds Prior.log_prob(params) to task log-likelihoods to form the log posterior
- Prior will generate structured parameters for basis expansions
  and decay_rate-controlled smooth covariance fields
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr

Params = dict[str, jnp.ndarray]


@dataclass
class Prior:
    """
    Prior distribution over WPPM parameters

    Parameters
    ----------
    input_dim : int, default=2
        Dimensionality of the model space. WPPM.input_dim reads this.
    basis_degree : int, default=4
        Degree of the Chebyshev basis. U(x) is a weighted sum of basis functions
        up to this degree, with weights W. Must be 0 or more.
    variance_scale : float, default=4e-3
        Prior variance for degree-0 (constant) coefficient.
        Controls overall scale of covariances. WPPM.variance_scale reads this.
    decay_rate : float, default=0.4
        Geometric decay rate for prior variance over higher-degree coefficients.
        Prior variance for degree-d coefficient = variance_scale * (decay_rate^d).
        Smaller decay_rate -> stronger smoothness prior. WPPM.decay_rate reads this.
    extra_embedding_dims : int, default=1
        Additional latent dimensions in U matrices beyond input dimensions.
        Allows richer ellipsoid shapes. WPPM.embedding_dim reads this.
    """

    input_dim: int = 2
    basis_degree: int = 4
    variance_scale: float = 4e-3
    decay_rate: float = 0.4
    extra_embedding_dims: int = 1

    def __post_init__(self):
        """Validate and normalize parameters."""
        if (
            not isinstance(self.basis_degree, int)
            or isinstance(self.basis_degree, bool)
            or self.basis_degree < 0
        ):
            raise ValueError(
                f"basis_degree must be an int of 0 or more, got {self.basis_degree!r}. "
                "Pass e.g. Prior(basis_degree=4)."
            )

    def _compute_basis_degree_grid(self) -> jnp.ndarray:
        """
        Compute total degree for each basis function coefficient.

        For 2D: basis functions are products φ_i(x_1) * φ_j(x_2)
        Total degree of φ_ij is i + j.

        Note: For basis_degree=d, we have (d+1) basis functions [T_0, ..., T_d] per dimension,
        so the grid has shape (d+1, d+1) for 2D.

        Returns
        -------
        basis_degrees : jnp.ndarray, shape (degree+1,) * input_dim
            Array of total degrees.
        """
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
        variances : jnp.ndarray, shape (degree+1,) * input_dim
            Prior variances, same shape as ``_compute_basis_degree_grid``'s output.
        """
        basis_degrees = self._compute_basis_degree_grid()
        return self.variance_scale * (self.decay_rate**basis_degrees)

    def param_scales(self) -> Params:
        """
        Prior standard deviation of every parameter, as a PyTree shaped like
        ``sample_params()``.

        The prior std depends only on a coefficient's basis degree, so it is
        broadcast over the trailing (input_dim, embedding_dim) axes of W.

        Used by NUTSSampler to precondition the sampler: coefficients at
        different degrees have prior scales that differ by orders of magnitude,
        and sampling W / scale instead of W puts them all on a common O(1)
        scale.

        Returns
        -------
        scales : dict
            {"W": array with the same shape as sample_params()["W"]}
        """
        std = jnp.sqrt(self._compute_W_prior_variances())
        embedding_dim = self.input_dim + self.extra_embedding_dims
        # std.shape is the basis grid, e.g. (degree+1, degree+1) for 2D;
        # append the two trailing W axes and broadcast over them.
        W_shape = (*std.shape, self.input_dim, embedding_dim)
        return {"W": jnp.broadcast_to(std[..., None, None], W_shape)}

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

    def sample(self, n: int, *, key: Any) -> Params:
        """
        Sample n independent parameter draws from the prior.

        The vectorized form of :meth:`sample_params`: draw ``i`` equals
        ``sample_params(jax.random.split(key, n)[i])``.

        Parameters
        ----------
        n : int
            Number of draws.
        key : jax.Array
            PRNG key, split n ways.

        Returns
        -------
        params : dict
            Parameter dictionary with a leading axis of n, so for 2D
            {"W": (n, degree+1, degree+1, input_dim, embedding_dim)}

        Notes
        -----
        Satisfies :class:`psyphy.posterior.ParameterSource`.

        Examples
        --------
        >>> import jax.random as jr
        >>> from psyphy.model import Prior
        >>> prior = Prior(input_dim=2, basis_degree=2)
        >>> prior.sample(5, key=jr.PRNGKey(0))["W"].shape
        (5, 3, 3, 2, 3)
        """
        return jax.vmap(self.sample_params)(jr.split(key, n))

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
