"""
covariance_field.py
-------------------

Covariance field abstraction for spatially-varying Σ(x).

A covariance field represents a function from stimulus locations to covariance
matrices, encapsulating the spatially-varying perceptual uncertainty in WPPM.

Design
------
This module provides:
- CovarianceField: Protocol defining the interface
- WPPMCovarianceField: Concrete implementation for WPPM models
- construction from priors, posteriors, or arbitrary parameters

Mathematical Background
-----------------------
In the Wishart Process Psychophysical Model:
    Σ(x) = U(x) @ U(x)^T + λI

where:
    U(x) = Σ_ij W_ij * φ_ij(x)  (basis expansion)
    φ_ij(x) are Chebyshev basis functions
    W_ij are learned coefficients
    λ is a numerical stabilizer (diag_term)

In MVP mode, Σ(x) is constant: Σ(x) = diag(exp(log_diag)).

Usage Examples
--------------
# Sample from prior
>>> model = WPPM(input_dim=2, prior=Prior(basis_degree=5), ...)
>>> field = WPPMCovarianceField.from_prior(model, key)
>>> Sigma = field.cov(x)

# From fitted posterior
>>> posterior = model.fit(data, optimizer=MAPOptimizer())
>>> field = posterior.get_covariance_field()
>>> Sigmas = field.cov_batch(X_grid)

# Access square root
>>> U = field.sqrt_cov(x)  # Only in Wishart mode
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    import jax.random as jr


@runtime_checkable
class CovarianceField(Protocol):
    """
    Protocol for spatially-varying covariance fields Σ(x).

    A covariance field maps stimulus locations x ∈ R^d to
    covariance matrices Σ(x) ∈ R^{dxd}.

    Methods
    -------
    cov(x)
        Evaluate Σ(x) at stimulus location x.
    sqrt_cov(x)
        Evaluate U(x) such that Σ(x) = U(x) @ U(x)^T + λI.
    cov_batch(X)
        Vectorized evaluation at multiple locations.

    Notes
    -----
    This protocol enables polymorphic use of covariance fields from different
    sources (prior samples, fitted posteriors, custom parameterizations).
    """

    def cov(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate covariance matrix Σ(x) at stimulus location x.

        Parameters
        ----------
        x : jnp.ndarray, shape (input_dim,)
            Stimulus location in [0, 1]^d

        Returns
        -------
        jnp.ndarray, shape (dim, dim)
            Covariance matrix Σ(x)
            - MVP mode: dim = input_dim
            - Wishart mode: dim = embedding_dim
        """
        ...

    def sqrt_cov(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate "square root" matrix U(x) such that Σ(x) = U(x) @ U(x)^T + λI.

        Parameters
        ----------
        x : jnp.ndarray, shape (input_dim,)
            Stimulus location

        Returns
        -------
        jnp.ndarray, shape (embedding_dim, embedding_dim + extra_dims)
            Square root matrix U(x)

        Raises
        ------
        ValueError
            If called in MVP mode (no U representation available).

        Notes
        -----
        Only available in Wishart mode (basis_degree set).
        MVP mode uses direct diagonal parameterization without U matrices.
        """
        ...

    def cov_batch(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate covariance at multiple locations (vectorized).

        Parameters
        ----------
        X : jnp.ndarray, shape (n_points, input_dim)
            Multiple stimulus locations

        Returns
        -------
        jnp.ndarray, shape (n_points, dim, dim)
            Covariance matrices at each location
        """
        ...


class WPPMCovarianceField:
    """
    Covariance field for WPPM with Wishart process or MVP mode.

    Encapsulates model + parameters to provide clean evaluation interface
    for Σ(x) and U(x).

    Parameters
    ----------
    model : WPPM
        Model providing evaluation logic (local_covariance, _compute_U)
    params : dict
        Model parameters:
        - MVP: {"log_diag": (input_dim,)}
        - Wishart: {"W": (degree+1, degree+1, embedding_dim, embedding_dim+extra_dims)}

    Attributes
    ----------
    model : WPPM
        Associated model instance
    params : dict
        Parameter dictionary

    Examples
    --------
    >>> # From prior
    >>> model = WPPM(input_dim=2, prior=Prior(basis_degree=5), ...)
    >>> field = WPPMCovarianceField.from_prior(model, key)
    >>> Sigma = field.cov(jnp.array([0.5, 0.3]))
    >>>
    >>> # From posterior
    >>> posterior = model.fit(data, optimizer=MAPOptimizer())
    >>> field = posterior.get_covariance_field()
    >>> Sigmas = field.cov_batch(X_grid)
    >>>
    >>> # Access square root (Wishart mode only)
    >>> U = field.sqrt_cov(jnp.array([0.5, 0.3]))

    Notes
    -----
    Implements the CovarianceField protocol for polymorphic use.
    """

    def __init__(self, model, params: dict):
        """
        Construct covariance field from model and parameters.

        Parameters
        ----------
        model : WPPM
            Model providing evaluation logic
        params : dict
            Parameter dictionary
        """
        self.model = model
        self.params = params

    @classmethod
    def from_prior(cls, model, key: jr.KeyArray) -> WPPMCovarianceField:
        """
        Sample a covariance field from the prior.

        Parameters
        ----------
        model : WPPM
            Model defining prior distribution
        key : jax.random.KeyArray
            PRNG key for sampling

        Returns
        -------
        WPPMCovarianceField
            Field sampled from p(Σ(x))

        Examples
        --------
        >>> model = WPPM(input_dim=2, prior=Prior(basis_degree=5), ...)
        >>> field = WPPMCovarianceField.from_prior(model, jr.PRNGKey(42))
        >>> Sigma = field.cov(jnp.array([0.5, 0.5]))
        """
        params = model.init_params(key)
        return cls(model, params)

    @classmethod
    def from_posterior(cls, posterior) -> WPPMCovarianceField:
        """
        Create covariance field from fitted posterior.

        Parameters
        ----------
        posterior : ParameterPosterior
            Fitted posterior (e.g., from model.fit())

        Returns
        -------
        WPPMCovarianceField
            Field representing posterior estimate of Σ(x)

        Notes
        -----
        For MAP posteriors, uses θ_MAP.
        For variational posteriors, could use posterior mean or sample.

        Examples
        --------
        >>> posterior = model.fit(data, optimizer=MAPOptimizer())
        >>> field = WPPMCovarianceField.from_posterior(posterior)
        >>> Sigma = field.cov(x)
        """
        return cls(posterior._model, posterior.params)

    @classmethod
    def from_params(cls, model, params: dict) -> WPPMCovarianceField:
        """
        Create field from arbitrary parameters.

        Useful for:
        - Custom initialization
        - Posterior samples
        - Intermediate optimization checkpoints
        - Testing

        Parameters
        ----------
        model : WPPM
            Model providing evaluation logic
        params : dict
            Parameter dictionary

        Returns
        -------
        WPPMCovarianceField

        Examples
        --------
        >>> params = {"log_diag": jnp.array([0.1, 0.2])}
        >>> field = WPPMCovarianceField.from_params(model, params)
        """
        return cls(model, params)

    def cov(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate Σ(x) at stimulus location x.

        Parameters
        ----------
        x : jnp.ndarray, shape (input_dim,)
            Stimulus location

        Returns
        -------
        jnp.ndarray, shape (dim, dim)
            Covariance matrix Σ(x)
            - MVP: shape (input_dim, input_dim)
            - Wishart: shape (embedding_dim, embedding_dim)

        Examples
        --------
        >>> field = WPPMCovarianceField.from_prior(model, key)
        >>> x = jnp.array([0.5, 0.3])
        >>> Sigma = field.cov(x)
        >>> print(Sigma.shape)  # (2, 2) for input_dim=2
        """
        return self.model.local_covariance(self.params, x)

    def sqrt_cov(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate U(x) such that Σ(x) = U(x) @ U(x)^T + diag_term*I.

        Parameters
        ----------
        x : jnp.ndarray, shape (input_dim,)
            Stimulus location

        Returns
        -------
        jnp.ndarray, shape (embedding_dim, embedding_dim + extra_dims)
            Square root matrix U(x)

        Raises
        ------
        ValueError
            If in MVP mode (basis_degree=None).

        Notes
        -----
        Only available in Wishart mode. MVP mode uses diagonal parameterization
        without explicit U matrices.

        Examples
        --------
        >>> # Wishart mode
        >>> model = WPPM(input_dim=2, basis_degree=5, extra_dims=1, ...)
        >>> field = WPPMCovarianceField.from_prior(model, key)
        >>> U = field.sqrt_cov(jnp.array([0.5, 0.3]))
        >>> print(U.shape)  # (2, 3) for embedding_dim=2, extra_dims=1
        >>>
        >>> # Verify: Σ = U @ U^T + λI
        >>> Sigma_from_U = U @ U.T + model.diag_term * jnp.eye(2)
        >>> Sigma_direct = field.cov(x)
        >>> assert jnp.allclose(Sigma_from_U, Sigma_direct)
        """
        if "W" not in self.params:
            raise ValueError(
                "sqrt_cov only available in Wishart mode. "
                "Set basis_degree when creating WPPM to use Wishart process."
            )
        return self.model._compute_U(self.params, x)

    def cov_batch(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Vectorized evaluation of Σ(x) at multiple locations.

        Uses JAX vmap for efficient batched computation.

        Parameters
        ----------
        X : jnp.ndarray, shape (n_points, input_dim)
            Multiple stimulus locations

        Returns
        -------
        jnp.ndarray, shape (n_points, dim, dim)
            Covariance matrices at each location

        Examples
        --------
        >>> X_grid = jnp.array([[0.1, 0.2], [0.5, 0.5], [0.9, 0.8]])
        >>> Sigmas = field.cov_batch(X_grid)
        >>> print(Sigmas.shape)  # (3, 2, 2)
        >>>
        >>> # Can also use with meshgrid
        >>> X_grid = jnp.stack(
        ...     jnp.meshgrid(jnp.linspace(0, 1, 50), jnp.linspace(0, 1, 50)), axis=-1
        ... )  # (50, 50, 2)
        >>> Sigmas = field.cov_batch(X_grid.reshape(-1, 2))  # (2500, 2, 2)
        """
        return jax.vmap(self.cov)(X)

    def sqrt_cov_batch(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Vectorized evaluation of U(x) at multiple locations.

        Parameters
        ----------
        X : jnp.ndarray, shape (n_points, input_dim)
            Multiple stimulus locations

        Returns
        -------
        jnp.ndarray, shape (n_points, embedding_dim, embedding_dim + extra_dims)
            Square root matrices at each location

        Raises
        ------
        ValueError
            If in MVP mode.

        Examples
        --------
        >>> X_grid = jnp.array([[0.1, 0.2], [0.5, 0.5], [0.9, 0.8]])
        >>> U_batch = field.sqrt_cov_batch(X_grid)
        >>> print(U_batch.shape)  # (3, 2, 3) for embedding_dim=2, extra_dims=1
        """
        if "W" not in self.params:
            raise ValueError("sqrt_cov_batch only available in Wishart mode")
        return jax.vmap(self.sqrt_cov)(X)

    @property
    def is_wishart_mode(self) -> bool:
        """
        Check if using Wishart process (spatially-varying covariance).

        Returns
        -------
        bool
            True if params contain "W" (Wishart mode), False otherwise (MVP).

        Examples
        --------
        >>> field_wishart = WPPMCovarianceField.from_prior(wishart_model, key)
        >>> assert field_wishart.is_wishart_mode
        >>>
        >>> field_mvp = WPPMCovarianceField.from_prior(mvp_model, key)
        >>> assert not field_mvp.is_wishart_mode
        """
        return "W" in self.params

    @property
    def is_mvp_mode(self) -> bool:
        """
        Check if using MVP diagonal mode (constant covariance).

        Returns
        -------
        bool
            True if params contain "log_diag" (MVP), False otherwise (Wishart).

        Examples
        --------
        >>> field_mvp = WPPMCovarianceField.from_prior(mvp_model, key)
        >>> assert field_mvp.is_mvp_mode
        >>>
        >>> field_wishart = WPPMCovarianceField.from_prior(wishart_model, key)
        >>> assert not field_wishart.is_mvp_mode
        """
        return "log_diag" in self.params
