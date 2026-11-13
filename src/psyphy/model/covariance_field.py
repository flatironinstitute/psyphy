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
- WPPMCovarianceFieldSamples: N draws of the field, with E[Σ(x)] and a
  credible band on its contour as reductions over them (see its docstring on
  why averaging W instead gives too-small ellipses)
- covariance_to_arviz: per-sample Σ(x) at chosen points as an ArviZ DataTree,
  for convergence diagnostics that are not fooled by the non-identifiable
  orientation of W (see its docstring)
- contour_radii / ContourBand: the one-standard-deviation ellipse
  (Mahalanobis-1 contour) of a covariance as a radius per direction. For the
  oddity task that radius is the discrimination threshold

Mathematical Background
-----------------------
In the Wishart Process Psychophysical Model:
    Σ(x) = U(x) @ U(x)^T + λI

where:
    U(x) = Σ_ij W_ij * φ_ij(x)  (basis expansion)
    φ_ij(x) are Chebyshev basis functions
    W_ij are learned coefficients
    λ is a numerical stabilizer (diag_term)


Usage Examples
--------------
>>> import jax.numpy as jnp
>>> import jax.random as jr
>>> from psyphy.model import WPPM, OddityTask, Prior, WPPMCovarianceField
>>> model = WPPM(Prior(input_dim=2, basis_degree=2), OddityTask())
>>> field = WPPMCovarianceField.from_prior(model, jr.PRNGKey(0))
>>> field(jnp.array([0.5, 0.3])).shape
(2, 2)
>>> field(jnp.zeros((10, 2))).shape
(10, 2, 2)
>>> field.sqrt_cov(jnp.array([0.5, 0.3])).shape  # (input_dim, embedding_dim)
(2, 3)
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, NamedTuple, Protocol, runtime_checkable

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    pass


def _draws_from(posterior, n_samples: int | None, key) -> tuple[dict, int]:
    """Parameter draws for a reduction over them, and how many there are.

    Two paths, picked by ``n_samples``:

    - ``None`` pools every stored draw from ``posterior.samples``, in stored
      order. No key, since a mean or a quantile over the whole pool does not
      depend on the order it arrives in.
    - an int calls ``posterior.sample(n_samples, key=key)``, which subsamples.
      ``key`` is required.

    Parameters
    ----------
    posterior : ParameterPosterior
        For the pooled path, one exposing ``samples``, ``n_chains`` and
        ``n_draws``, which today means an ``MCMCPosterior``.
    n_samples : int or None
        Draws to use, or None for every stored draw.
    key : jax.Array or None
        PRNG key. Read only when ``n_samples`` is an int.

    Returns
    -------
    draws : dict
        Parameter PyTree with a leading axis of n_samples.
    n_samples : int
        The resolved count.

    Raises
    ------
    ValueError
        If ``n_samples`` is None and the posterior stores no fixed set of draws,
        or if ``n_samples`` is given without a key.
    """
    if n_samples is None:
        #  get stored posterior values  if available, else raise set to None
        samples = getattr(posterior, "samples", None)
        n_chains = getattr(posterior, "n_chains", None)
        n_draws = getattr(posterior, "n_draws", None)
        if samples is None or n_chains is None or n_draws is None:
            raise ValueError(
                f"{type(posterior).__name__} stores no fixed set of draws, so "
                "n_samples cannot be inferred. Pass n_samples and a key."
            )
        total = int(n_chains) * int(n_draws)
        # (n_chains, n_draws, *param_shape) -> (n_chains * n_draws, *param_shape)
        pooled = jax.tree.map(lambda x: x.reshape(total, *x.shape[2:]), dict(samples))
        return pooled, total

    if key is None:
        raise ValueError(
            f"n_samples={n_samples} subsamples the draws, so pass key=. "
            "Omit n_samples to use every stored draw, which needs no key."
        )
    return posterior.sample(int(n_samples), key=key), int(n_samples)


def _sigma(model, params: dict, X: jnp.ndarray) -> jnp.ndarray:
    """Σ at every location for one parameter set.

    Parameters
    ----------
    model : WPPM
        Model providing ``local_covariance``.
    params : dict
        One parameter set, with no leading draws axis.
    X : jnp.ndarray, shape (..., d)
        Stimulus locations, any number of batch dimensions.

    Returns
    -------
    jnp.ndarray, shape (..., d, d)
        One covariance matrix per location.
    """
    d = model.input_dim
    X_flat = X.reshape(-1, d)  # (n_flat, d)
    # (n_flat, d, d)
    Sigmas_flat = jax.vmap(lambda x: model.local_covariance(params, x))(X_flat)
    return Sigmas_flat.reshape(X.shape[:-1] + (d, d))


_READ_ONLY = "{name} is fixed when the object is built. Build a new one: {build}"


class _PriorSource:
    """A model's prior with the 2 members WPPMCovarianceFieldSamples reads."""

    def __init__(self, model):
        self.model = model

    def sample(self, n: int, *, key) -> dict:
        return self.model.prior.sample(n, key=key)


@runtime_checkable
class CovarianceField(Protocol):
    """
    Protocol for spatially-varying covariance fields Σ(x).

    A covariance field maps stimulus locations x ∈ R^d to
    covariance matrices Σ(x) ∈ R^{dxd}.

    Methods
    -------
    __call__(x)
        Evaluate field at one or more locations. Supports both single points
        and arbitrary batch dimensions.
    sqrt_cov(x)
        Evaluate U(x) such that Σ(x) = U(x) @ U(x)^T + λI.

    Notes
    -----
    This protocol enables polymorphic use of covariance fields from different
    sources (prior samples, fitted posteriors, custom parameterizations).

    The field is callable for mathematical elegance and JAX compatibility:
        Sigma = field(x)  # Single point or batch
    """

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate covariance field at one or more stimulus locations."""
        ...

    def sqrt_cov(self, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate "square root" matrix U(x) such that Σ(x) = U(x) @ U(x)^T + λI."""
        ...


class WPPMCovarianceField:
    """
    Covariance field for WPPM with Wishart process
    Encapsulates model + parameters to provide clean evaluation interface
    for Σ(x) and U(x).

    Parameters
    ----------
    model : WPPM
        Model providing evaluation logic (local_covariance, _compute_sqrt)
    params : dict
        Model parameters:
        - Wishart: {"W": (degree+1, degree+1, input_dim, embedding_dim)}
          where embedding_dim = input_dim + extra_embedding_dims

          Note: The 3rd dimension is input_dim (output/stimulus space) and not
          embedding_dim. This is analogous to the einsum in
          _compute_sqrt where U(x) has shape (input_dim, embedding_dim).

    Attributes
    ----------
    model : WPPM
        Associated model instance
    params : dict
        Parameter dictionary

    Both attributes are fixed when the object is built. Assigning to one
    raises ``AttributeError``. After changing an attribute of the model in
    place, such as ``model.diag_term``, build a new field. The field copies
    ``params`` at construction, so changing the dict you passed in afterwards
    changes nothing.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import jax.random as jr
    >>> from psyphy.model import WPPM, OddityTask, Prior, WPPMCovarianceField
    >>> model = WPPM(Prior(input_dim=2, basis_degree=2), OddityTask())
    >>> # from the prior
    >>> field = WPPMCovarianceField.from_prior(model, jr.PRNGKey(0))
    >>> field(jnp.array([0.5, 0.3])).shape
    (2, 2)
    >>> # from a MAP fit
    >>> from psyphy.inference import MAPOptimizer
    >>> map_posterior = MAPOptimizer().fit(model, data)  # doctest: +SKIP
    >>> field = WPPMCovarianceField.from_posterior(map_posterior)  # doctest: +SKIP

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
        self._model = model
        self._params = dict(params)
        # Pre-compile JIT batch evaluation path for performance
        self._eval_batch_jitted = jax.jit(self._eval_batch_impl)

    @property
    def model(self):
        """The model that evaluates the field. Fixed when the object is built."""
        return self._model

    @model.setter
    def model(self, value):
        raise AttributeError(
            _READ_ONLY.format(name="model", build="WPPMCovarianceField(model, params)")
        )

    @property
    def params(self) -> dict:
        """A shallow copy of the parameter dict. Fixed when the object is built."""
        return dict(self._params)

    @params.setter
    def params(self, value):
        raise AttributeError(
            _READ_ONLY.format(name="params", build="WPPMCovarianceField(model, params)")
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate covariance field at one or more stimulus locations.

        Automatically dispatches to single-point or batch evaluation based on input shape.
        Last axis must have size input_dim. Any number of batch dimensions supported.

        Parameters
        ----------
        x : jnp.ndarray, shape (input_dim,) or (..., input_dim)
            The first for a single point, the second for a batch with
            arbitrary leading batch dimensions.

        Returns
        -------
        jnp.ndarray, shape (d, d) or (..., d, d), where d = model.input_dim
            Square covariance matrix/matrices: the first shape for a single
            point, the second for a batch.

            For input_dim=2: returns (2, 2) or (..., 2, 2)
            For input_dim=3: returns (3, 3) or (..., 3, 3)

        Raises
        ------
        ValueError
            If last axis size != input_dim

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> from psyphy.model import WPPM, OddityTask, Prior, WPPMCovarianceField
        >>> # 2D stimulus space (e.g., isoluminant plane)
        >>> model_2d = WPPM(Prior(input_dim=2, basis_degree=2), OddityTask())
        >>> field_2d = WPPMCovarianceField.from_prior(model_2d, jr.PRNGKey(0))
        >>>
        >>> # Single 2D point
        >>> x = jnp.array([0.5, 0.3])  # Shape (2,)
        >>> Sigma = field_2d(x)  # Shape (2, 2)
        >>>
        >>> # Batch of 2D points
        >>> X = jnp.array([[0.1, 0.2], [0.5, 0.5]])  # Shape (2, 2)
        >>> Sigmas = field_2d(X)  # Shape (2, 2, 2)
        >>>
        >>> # 2D spatial grid in 2D stimulus space
        >>> X_grid = jnp.ones((10, 10, 2))  # Shape (10, 10, 2)
        >>> Sigmas_grid = field_2d(X_grid)  # Shape (10, 10, 2, 2)
        >>>
        >>> # 3D stimulus space (e.g., RGB)
        >>> model_3d = WPPM(Prior(input_dim=3, basis_degree=2), OddityTask())
        >>> field_3d = WPPMCovarianceField.from_prior(model_3d, jr.PRNGKey(0))
        >>>
        >>> # Single 3D point
        >>> x_3d = jnp.array([0.5, 0.3, 0.2])  # Shape (3,)
        >>> Sigma_3d = field_3d(x_3d)  # Shape (3, 3)
        >>>
        >>> # Batch of 3D points
        >>> X_3d = jnp.ones((5, 10, 3))  # Shape (5, 10, 3)
        >>> Sigmas_3d = field_3d(X_3d)  # Shape (5, 10, 3, 3)

        Notes
        -----
        For input_dim=1, single points must have shape (1,) not (). Use x[None]
        if needed to convert scalar to shape (1,).
        """
        # Validate input dimension
        if x.shape[-1] != self.model.input_dim:
            raise ValueError(
                f"Last axis must be input_dim={self.model.input_dim}, got {x.shape[-1]}"
            )

        # Dispatch logic:
        #   - x.ndim == 1: Single point -> call _eval_single()
        #   - x.ndim >= 2: Batch -> flatten, vmap, reshape

        # Dispatch based on shape
        if x.ndim == 1:
            # Single point: shape (d,) -> (d, d) where d = model.input_dim
            # E.g., input_dim=2: (2,) -> (2,2), input_dim=3: (3,) -> (3,3)
            return self._eval_single(x)
        else:
            # Batch: shape (..., d) -> (..., d, d) where d = model.input_dim
            # Returns square covariance matrices for each point in the batch
            return self._eval_batch_jitted(x)

    def _eval_single(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate covariance at a single stimulus location.

        Parameters
        ----------
        x : jnp.ndarray, shape (input_dim,)
            Single stimulus location

        Returns
        -------
        jnp.ndarray, shape (input_dim, input_dim)
            Covariance matrix Σ(x)
        """
        return self.model.local_covariance(self.params, x)

    def _eval_batch_impl(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Batch evaluation, compiled in ``__init__``.

        Parameters
        ----------
        X : jnp.ndarray, shape (..., input_dim)
            Batch of stimulus locations with arbitrary batch dimensions

        Returns
        -------
        jnp.ndarray, shape (..., input_dim, input_dim)
            Batch of covariance matrices
        """
        return _sigma(self.model, self.params, X)

    @classmethod
    def from_prior(cls, model, key: jax.Array) -> WPPMCovarianceField:
        """
        Sample a covariance field from the prior.

        Parameters
        ----------
        model : WPPM
            Model defining prior distribution
        key : jax.Array
            PRNG key for sampling

        Returns
        -------
        WPPMCovarianceField
            Field sampled from p(Σ(x))

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> from psyphy.model import WPPM, OddityTask, Prior, WPPMCovarianceField
        >>> model = WPPM(Prior(input_dim=2, basis_degree=2), OddityTask())
        >>> field = WPPMCovarianceField.from_prior(model, jr.PRNGKey(42))
        >>> field(jnp.array([0.5, 0.5])).shape
        (2, 2)
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
        Builds the field from ``posterior.params``, i.e. a single parameter set.

        - MAP posteriors: ``params`` is θ_MAP, so this is Σ(θ_MAP).
        - Sample-based posteriors (e.g. ``MCMCPosterior``): ``params`` is the
          mean parameter E[W], and Σ(E[W]) is NOT the posterior mean of Σ. It
          is systematically too small and can collapse toward λI (see
          ``WPPMCovarianceFieldSamples`` for why). A ``UserWarning`` is
          raised in that case. Use
          ``WPPMCovarianceFieldSamples(posterior).mean(x)``
          for E[Σ], or ``posterior.sample(n, key=...)`` for individual samples.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> from psyphy.model import WPPM, OddityTask, Prior, WPPMCovarianceField
        >>> model = WPPM(Prior(input_dim=2, basis_degree=2), OddityTask())
        >>> from psyphy.inference import MAPOptimizer
        >>> map_posterior = MAPOptimizer().fit(model, data)  # doctest: +SKIP
        >>> field = WPPMCovarianceField.from_posterior(map_posterior)  # doctest: +SKIP
        >>> field(jnp.array([0.5, 0.5])).shape  # doctest: +SKIP
        (2, 2)
        """
        # Sample-based posteriors are recognized by duck typing (they expose a
        # sample count), the same criterion _draws_from uses.
        # isinstance(posterior, MCMCPosterior) would need an import from
        # psyphy.posterior, which already imports psyphy.model -> cycle.
        #
        # Warns and does not switch to the posterior-mean field, which has a
        # different return type and no sqrt_cov.
        if getattr(posterior, "n_draws", None) is not None:
            warnings.warn(
                f"from_posterior() on a {type(posterior).__name__} builds Σ from the "
                "mean parameter E[W]. Σ is nonlinear in W, so Σ(E[W]) is not the "
                "posterior mean of Σ and comes out systematically too small. Use "
                "WPPMCovarianceFieldSamples(posterior).mean(x) "
                "for E[Σ].",
                UserWarning,
                stacklevel=2,
            )
        return cls(posterior.model, posterior.params)

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
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> from psyphy.model import WPPM, OddityTask, Prior, WPPMCovarianceField
        >>> model = WPPM(Prior(input_dim=2, basis_degree=2), OddityTask())
        >>> params = model.init_params(jr.PRNGKey(0))  # {"W": (3, 3, 2, 3)}
        >>> field = WPPMCovarianceField.from_params(model, params)
        >>> field(jnp.array([0.5, 0.5])).shape
        (2, 2)
        """
        return cls(model, params)

    def sqrt_cov(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate U(x) such that Σ(x) = U(x) @ U(x)^T + diag_term*I.

        Parameters
        ----------
        x : jnp.ndarray, shape (input_dim,)
            Stimulus location

        Returns
        -------
        jnp.ndarray, shape (input_dim, embedding_dim)
            Rectangular square root matrix U(x).
            embedding_dim = input_dim + extra_embedding_dims

        Notes
        -----
        U is of shape (input_dim, embedding_dim).

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> from psyphy.model import WPPM, OddityTask, Prior, WPPMCovarianceField
        >>> model = WPPM(Prior(input_dim=2, basis_degree=2), OddityTask())
        >>> field = WPPMCovarianceField.from_prior(model, jr.PRNGKey(0))
        >>> x = jnp.array([0.5, 0.3])
        >>> U = field.sqrt_cov(x)
        >>> U.shape  # (input_dim, embedding_dim) for input_dim=2, extra_embedding_dims=1
        (2, 3)
        >>> # Σ = U @ U^T + λI
        >>> bool(jnp.allclose(U @ U.T + model.diag_term * jnp.eye(2), field(x)))
        True
        """
        if "W" not in self.params:
            raise ValueError(
                "params has no 'W', so sqrt_cov cannot evaluate U(x). "
                "Pass params with a 'W' entry, e.g. from WPPM.init_params."
            )
        return self.model._compute_sqrt(self.params, x)

    def sqrt_cov_batch(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Vectorized evaluation of U(x) at multiple locations.

        Parameters
        ----------
        X : jnp.ndarray, shape (n_points, input_dim)
            Multiple stimulus locations

        Returns
        -------
        jnp.ndarray, shape (n_points, input_dim, embedding_dim)
            Rectangular square root matrices at each location.
            embedding_dim = input_dim + extra_embedding_dims

        Notes
        -----
        In the rectangular design (Hong et al.), U is (input_dim, embedding_dim).

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> from psyphy.model import WPPM, OddityTask, Prior, WPPMCovarianceField
        >>> model = WPPM(Prior(input_dim=2, basis_degree=2), OddityTask())
        >>> field = WPPMCovarianceField.from_prior(model, jr.PRNGKey(0))
        >>> X_grid = jnp.array([[0.1, 0.2], [0.5, 0.5], [0.9, 0.8]])  # (3, 2)
        >>> field.sqrt_cov_batch(X_grid).shape  # (n_points, input_dim, embedding_dim)
        (3, 2, 3)
        """
        if "W" not in self.params:
            raise ValueError(
                "params has no 'W', so sqrt_cov_batch cannot evaluate U(x)."
            )
        return jax.vmap(self.sqrt_cov)(X)


def covariance_to_arviz(posterior, points):
    r"""
    Per-sample covariance Σ(x) at chosen points, as an ArviZ DataTree.

    Use this, not ``posterior.to_arviz()``, for convergence diagnostics
    (R-hat, ESS, trace plots) of a WPPM fit.

    Why Σ and not W
    ---------------
    W is only identified up to an orthogonal rotation R of its embedding
    axis: W -> W R gives U(x) -> U(x) R and leaves Σ(x) = U Uᵀ + λI, and the
    prior, unchanged. Chains can therefore settle at different orientations
    that describe exactly the same covariance field. R-hat on W reports that
    as non-convergence, although no ellipse, threshold or prediction depends
    on the orientation. Σ(x) is invariant, so its diagnostics measure what the
    fit is actually used for.

    Why points must be chosen
    -------------------------
    Unlike W, Σ is a function of the stimulus location x, so a diagnostic of Σ
    is a diagnostic *at some x*. This function has no default for
    ``points``: which locations to check is a modeling decision. Sensible
    choices are the locations you report (e.g. the reference points of the
    plotted ellipses), optionally plus a grid covering the stimulus space,
    since away from the data the posterior relies more on the prior. Report
    the worst value over all points. A modest number of points suffices: Σ is
    a low-degree polynomial in x, so dense grids add mostly near-duplicates.

    Quantities
    ----------
    For each sample and point, the d x d matrix Σ is summarized by its d log
    variances and d(d-1)/2 correlations, which together determine Σ:

    - ``log_var_i``  = log Σ_ii
    - ``corr_i_j``   = Σ_ij / sqrt(Σ_ii Σ_jj), for i < j

    (2D: ``log_var_0``, ``log_var_1``, ``corr_0_1``.) The off-diagonal is not
    counted twice. Logs make the variances less skewed; note that ArviZ's
    default rank-normalized R-hat is unchanged by such monotone transforms
    anyway.

    Parameters
    ----------
    posterior : MCMCPosterior
        A sample-based posterior with ``to_arviz()`` (chain structure is
        needed for R-hat; ``sample()`` would pool the chains).
    points : array_like, shape (n_points, input_dim)
        Stimulus locations at which to evaluate Σ.

    Returns
    -------
    xarray.DataTree
        ``posterior`` group with variable ``Sigma`` of dims
        ``(chain, draw, point, quantity)``, coordinates ``quantity`` (names as
        above) and ``location_0 ... location_{d-1}`` (the evaluation points, one
        coordinate per stimulus axis, along ``point``). The posterior's
        ``sample_stats`` group (divergences, energy, ...) is carried over.

    Raises
    ------
    TypeError
        If the posterior has no ``to_arviz()`` (e.g. a MAP posterior: no
        chains, nothing to diagnose).
    ValueError
        If ``points`` is not of shape ``(n_points, input_dim)``.
    ImportError
        If ArviZ is not installed (``pip install 'psyphy[diagnostics]'``).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import jax.random as jr
    >>> from psyphy.model import WPPM, OddityTask, Prior, WPPMCovarianceField
    >>> model = WPPM(Prior(input_dim=2, basis_degree=2), OddityTask())
    >>> from psyphy.posterior import MCMCPosterior
    >>> # stand-in for NUTSSampler().fit(model, data): 2 chains of 50 prior samples
    >>> keys = jr.split(jr.PRNGKey(1), 100)
    >>> W = jnp.stack([model.init_params(k)["W"] for k in keys])  # (100, 3, 3, 2, 3)
    >>> nuts_posterior = MCMCPosterior({"W": W.reshape(2, 50, *W.shape[1:])}, model)
    >>> ref_points = jnp.array([[-0.5, -0.5], [0.5, 0.5]])  # (n_points, 2)
    >>> import arviz as az
    >>> from psyphy.model import covariance_to_arviz
    >>> dt = covariance_to_arviz(nuts_posterior, ref_points)
    >>> worst = az.rhat(dt)["Sigma"].max()  # worst point and quantity
    >>> az.plot_trace(dt, var_names=["Sigma"], coords={"point": 0})  # doctest: +SKIP
    """
    to_arviz = getattr(posterior, "to_arviz", None)
    if to_arviz is None:
        raise TypeError(
            f"{type(posterior).__name__} has no to_arviz(); convergence diagnostics "
            "need a sample-based posterior with chains (e.g. MCMCPosterior)."
        )

    model = posterior.model
    d = model.input_dim
    points = jnp.asarray(points)
    if points.ndim != 2:
        raise ValueError(
            f"points must have shape (n_points, input_dim), got {points.shape}. "
            "For a single location use points[None, :]."
        )
    if points.shape[-1] != d:
        raise ValueError(
            f"Last axis of points must be input_dim={d}, got {points.shape[-1]}"
        )

    # Chain-structured samples via the public API (raises ImportError without
    # ArviZ). Every posterior variable keeps its leading (chain, draw) axes,
    # e.g. {"W": (C, S, *W_shape)}.
    idata = to_arviz()
    import arviz as az
    import numpy as np

    params = {
        name: jnp.asarray(var.values)
        for name, var in idata["posterior"].data_vars.items()
    }

    def sigma_one_draw(p):
        # Σ at every point for ONE draw: (n_points, d, d)
        return jax.vmap(lambda x: model.local_covariance(p, x))(points)

    # vmap over samples, then chains: (C, S, n_points, d, d)
    Sig = jax.vmap(jax.vmap(sigma_one_draw))(params)

    # Identifiable summary of each Σ: log variances, then correlations (i < j).
    variances = jnp.diagonal(Sig, axis1=-2, axis2=-1)  # (C, S, P, d)
    columns = [jnp.log(variances[..., i]) for i in range(d)]
    names = [f"log_var_{i}" for i in range(d)]
    for i in range(d):
        for j in range(i + 1, d):
            columns.append(
                Sig[..., i, j] / jnp.sqrt(variances[..., i] * variances[..., j])
            )
            names.append(f"corr_{i}_{j}")
    feats = np.asarray(jnp.stack(columns, axis=-1))  # (C, S, P, n_quantities)

    data = {"posterior": {"Sigma": feats}}
    if "sample_stats" in idata.children:
        data["sample_stats"] = {
            k: v.values for k, v in idata["sample_stats"].data_vars.items()
        }
    dt = az.from_dict(
        data, dims={"Sigma": ["point", "quantity"]}, coords={"quantity": names}
    )
    # Record where Σ was evaluated, one 1-D coordinate per stimulus axis along
    # `point`. (A single (point, axis) coordinate would not attach to Sigma,
    # which has no `axis` dimension.)
    pts = np.asarray(points)
    dt["posterior"] = (
        dt["posterior"]
        .to_dataset()
        .assign_coords({f"location_{i}": ("point", pts[:, i]) for i in range(d)})
    )
    return dt


class ContourBand(NamedTuple):
    """A pointwise credible band on the contour radii, with its provenance.

    Attributes
    ----------
    points : jnp.ndarray, shape (n_points, d)
        Stimulus locations the band was evaluated at.
    directions : jnp.ndarray, shape (n_directions, d)
        Unit directions the radii are measured along, in order. Closed, so the
        last direction repeats the first and a filled polygon has no seam.
    lower, median, upper : jnp.ndarray, shape (n_points, n_directions) or (n_levels, n_points, n_directions)
        Interval endpoints and the posterior median radius: the first shape
        for a single ``ci_prob``, the second when a sequence was requested.
        All three carry the same shape, so a loop over levels can index all
        of them. The median does not depend on the interval probability, so
        its levels are identical copies.
    ci_prob : float | tuple[float, ...]
        Nominal probability each interval carries. A tuple when several levels
        were requested, in the order the arrays are stacked.
    ci_kind : str
        ``"eti"`` for equal-tailed, ``"hdi"`` for highest-density.

    Notes
    -----
    The interval is **pointwise**: it covers the true radius in each direction
    separately with probability ``ci_prob``. It does NOT cover the whole
    contour jointly at that probability, which is a stronger (simultaneous)
    statement requiring a different construction. ``ci_prob`` and ``ci_kind``
    travel with the arrays so a figure can state which interval it draws.
    """

    points: jnp.ndarray
    directions: jnp.ndarray
    lower: jnp.ndarray
    median: jnp.ndarray
    upper: jnp.ndarray
    ci_prob: float | tuple[float, ...]
    ci_kind: str


def contour_radii(covs: jnp.ndarray, directions: jnp.ndarray) -> jnp.ndarray:
    r"""Distance from the center to the one-standard-deviation ellipse, per direction.

    For a covariance :math:`\Sigma`, the contour
    :math:`\{v : v^\top \Sigma^{-1} v = 1\}` meets the ray along a unit
    direction :math:`u` at distance

    .. math:: r(u) = 1 / \sqrt{u^\top \Sigma^{-1} u}

    Writing a point on the ray as :math:`v = r\,u` turns the contour condition
    into :math:`r^2 u^\top \Sigma^{-1} u = 1`, which is the expression above.
    Along an eigenvector this is :math:`\sqrt{\lambda}`, i.e. exactly the
    semi-axis :func:`psyphy.viz.ellipse_segments` draws at ``scale=1``. For the
    oddity task this radius is the discrimination threshold, the just
    noticeable difference, in direction ``u``.

    Parameters
    ----------
    covs : jnp.ndarray, shape (..., d, d)
        One or more covariance matrices. Leading axes are broadcast over.
    directions : jnp.ndarray, shape (n_directions, d)
        Unit directions. We do not normalize them, and a non-unit direction
        rescales the radius.

    Returns
    -------
    jnp.ndarray, shape (..., n_directions)

    Notes
    -----
    This is a *polar* parameterization of the same ellipse that
    ``ellipse_segments`` produces, and the two are not interchangeable. That
    function maps a circle through the Cholesky factor, so its parameter angle
    is not the direction from the center. Any statistic taken across samples at
    fixed angle, a quantile or a mean, must use this parameterization, or it
    compares points lying in different directions.
    """
    prec = jnp.linalg.inv(covs)
    quad = jnp.einsum("ki,...ij,kj->...k", directions, prec, directions)
    return 1.0 / jnp.sqrt(quad)


def _eti(radii: jnp.ndarray, prob: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Equal-tailed interval over axis 0: the (1-p)/2 and 1-(1-p)/2 quantiles."""
    tail = (1.0 - prob) / 2.0
    lo, hi = jnp.quantile(radii, jnp.array([tail, 1.0 - tail]), axis=0)
    return lo, hi


def _hdi(radii: jnp.ndarray, prob: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Highest-density interval over axis 0: the narrowest window holding p mass.

    Sorts the samples and takes the narrowest window of ``ceil(prob * n)`` of
    them. This is the standard sample HDI. ArviZ computes the same quantity
    with a slightly different window convention, so the two agree closely but
    not bit for bit.
    """
    n = radii.shape[0]
    width = max(int(jnp.ceil(prob * n)), 2)  # samples inside the interval
    ordered = jnp.sort(radii, axis=0)
    los = ordered[: n - width + 1]  # (n_windows, ...)
    his = ordered[width - 1 :]  # (n_windows, ...)
    best = jnp.argmin(his - los, axis=0)[None]  # narrowest window per element
    return (
        jnp.take_along_axis(los, best, axis=0)[0],
        jnp.take_along_axis(his, best, axis=0)[0],
    )


_CI_KINDS = {"eti": _eti, "hdi": _hdi}


class WPPMCovarianceFieldSamples:
    r"""
    N draws of the WPPM covariance field, with the reductions over them.

    A fit gives many parameter draws, so it gives many covariance fields. This
    class holds the draws and evaluates them: ``fields(x)`` for every draw,
    :meth:`mean` for :math:`\mathbb{E}[\Sigma(x)]`, :meth:`band` for a credible
    band on the contour. The draws are taken when the object is built, so the
    three reductions are three views of one sample.

    :meth:`mean` averages :math:`\Sigma` over the draws. A field built from the
    averaged parameters gives contours several times too small, because
    :math:`\Sigma(x) = U(x) U(x)^\top + \lambda I` is nonlinear in :math:`W`.

    There is no ``sqrt_cov``. An average of :math:`U U^\top` is not
    :math:`U U^\top` for any single :math:`U`, so no matrix here is the field's
    square root. For a factor of the mean matrix, use
    ``jnp.linalg.cholesky(fields.mean(x))``. :class:`WPPMCovarianceField` has
    ``sqrt_cov``, for one parameter set.

    Parameters
    ----------
    posterior : ParameterPosterior
        A fitted posterior, typically an ``MCMCPosterior``. Its ``model``
        evaluates the samples. For samples from the prior, use
        :meth:`from_prior`.
    n_samples : int, optional
        How many samples to take. Omit it for every stored sample, which needs
        the posterior to expose ``samples``, ``n_chains`` and ``n_draws``. A
        point-mass posterior such as ``MAPPosterior`` has to pass this.
    key : jax.Array, optional
        PRNG key for the subsample. Required when ``n_samples`` is given, and
        unread otherwise, since every stored sample is taken in order.

    Raises
    ------
    ValueError
        If ``n_samples`` is None and the posterior stores no fixed set of
        samples, or if ``n_samples`` is given without a key.

    Attributes
    ----------
    model : WPPM
        The model the draws belong to.
    samples : dict
        The draws, leading axis ``n_samples``. Public, so a per-draw quantity
        psyphy does not ship can be had with ``jax.vmap`` over it.
    n_samples : int
        Draws held.

    All three are fixed when the object is built. Assigning to one raises
    ``AttributeError``. After changing an attribute of the model in place, such
    as ``model.diag_term``, build a new object.

    See Also
    --------
    WPPMCovarianceField : one parameter set, so one field and its square root.
    covariance_to_arviz : per-draw Σ with the chain axis kept, for diagnostics.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import jax.random as jr
    >>> from psyphy.model import WPPM, OddityTask, Prior, WPPMCovarianceFieldSamples
    >>> from psyphy.posterior import MCMCPosterior
    >>> model = WPPM(Prior(input_dim=2, basis_degree=2), OddityTask())
    >>> keys = jr.split(jr.PRNGKey(1), 100)
    >>> W = jnp.stack([model.init_params(k)["W"] for k in keys])  # (100, 3, 3, 2, 3)
    >>> posterior = MCMCPosterior({"W": W.reshape(2, 50, *W.shape[1:])}, model)
    >>> WPPMCovarianceFieldSamples(posterior).n_samples
    100
    >>> fields = WPPMCovarianceFieldSamples(posterior, n_samples=50, key=jr.PRNGKey(0))
    >>> ref_points = jnp.array([[-0.5, -0.5], [0.5, 0.5]])  # (n_points=2, d=2)
    >>> fields(ref_points).shape  # (n_samples, n_points, d, d)
    (50, 2, 2, 2)
    >>> fields.mean(ref_points).shape  # (n_points, d, d)
    (2, 2, 2)
    >>> fields.band(ref_points).lower.shape  # (n_points, n_directions)
    (2, 181)
    """

    def __init__(self, posterior, *, n_samples: int | None = None, key=None):
        samples, _ = _draws_from(posterior, n_samples, key)
        leaves = jax.tree.leaves(samples)
        if not leaves:
            raise ValueError(
                f"{type(posterior).__name__} returned no parameter arrays. "
                'Its samples need a leading axis, for example {"W": (n_samples, *W_shape)}.'
            )
        self._model = posterior.model
        self._samples = samples
        self._n_samples = int(leaves[0].shape[0])
        # Pre-compile the batch paths, as in WPPMCovarianceField
        self._eval_batch_jitted = jax.jit(self._eval_batch_impl)
        self._mean_batch_jitted = jax.jit(self._mean_batch_impl)

    @property
    def model(self):
        """The model the draws belong to. Fixed when the object is built."""
        return self._model

    @model.setter
    def model(self, value):
        raise AttributeError(
            _READ_ONLY.format(
                name="model", build="WPPMCovarianceFieldSamples(posterior)"
            )
        )

    @property
    def samples(self) -> dict:
        """A shallow copy of the draws dict. Fixed when the object is built."""
        return dict(self._samples)

    @samples.setter
    def samples(self, value):
        raise AttributeError(
            _READ_ONLY.format(
                name="samples", build="WPPMCovarianceFieldSamples(posterior)"
            )
        )

    @property
    def n_samples(self) -> int:
        """Draws held. Fixed when the object is built."""
        return self._n_samples

    @n_samples.setter
    def n_samples(self, value):
        raise AttributeError(
            _READ_ONLY.format(
                name="n_samples", build="WPPMCovarianceFieldSamples(posterior)"
            )
        )

    @classmethod
    def from_prior(cls, model, *, n_samples: int, key) -> WPPMCovarianceFieldSamples:
        """
        Samples from the model's prior.

        A prior stores no samples, so ``n_samples`` and ``key`` are both required.

        Parameters
        ----------
        model : WPPM
            Model whose prior we draw from.
        n_samples : int
            Draws to take.
        key : jax.Array
            PRNG key, split n_samples ways.

        Returns
        -------
        WPPMCovarianceFieldSamples

        Examples
        --------
        >>> import jax.random as jr
        >>> from psyphy.model import WPPM, OddityTask, Prior, WPPMCovarianceFieldSamples
        >>> model = WPPM(Prior(input_dim=2, basis_degree=2), OddityTask())
        >>> fields = WPPMCovarianceFieldSamples.from_prior(
        ...     model, n_samples=8, key=jr.PRNGKey(0)
        ... )
        >>> fields.samples["W"].shape  # (n_samples, degree+1, degree+1, d, m)
        (8, 3, 3, 2, 3)
        """
        return cls(_PriorSource(model), n_samples=n_samples, key=key)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate Σ(x) for every draw.

        Parameters
        ----------
        x : jnp.ndarray, shape (d,) or (..., d)
            One stimulus location, or any batch of them. The last axis must
            have size ``input_dim``.

        Returns
        -------
        Sigma : jnp.ndarray, shape (n_samples, d, d) or (n_samples, ..., d, d)
            One covariance matrix per draw per location. The draws come first,
            which is the stacking :func:`psyphy.viz.plot_ellipses` takes.

        Notes
        -----
        The result holds ``n_samples * n_points * d * d`` floats, so a dense
        grid with many draws is large: 4000 draws on a 50 x 50 grid of 2x2
        matrices is 160 MB. For a grid, pass ``n_samples`` when building the object.
        """
        self._check_input_dim(x)
        if x.ndim == 1:
            return self._eval_batch_impl(x)
        return self._eval_batch_jitted(x)

    def mean(self, x: jnp.ndarray) -> jnp.ndarray:
        r"""
        Posterior mean of the covariance field, :math:`\mathbb{E}[\Sigma(x)]`.

        Evaluates :math:`\Sigma(x)` for every draw and averages the resulting
        matrices. Averaging the parameters first instead gives contours several
        times too small, because :math:`\Sigma` is nonlinear in :math:`W`.

        Parameters
        ----------
        x : jnp.ndarray, shape (d,) or (..., d)
            One stimulus location, or any batch of them.

        Returns
        -------
        Sigma : jnp.ndarray, shape (d, d) or (..., d, d)
            The mean matrix per location, with no draws axis.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> from psyphy.model import WPPM, OddityTask, Prior, WPPMCovarianceFieldSamples
        >>> model = WPPM(Prior(input_dim=2, basis_degree=2), OddityTask())
        >>> fields = WPPMCovarianceFieldSamples.from_prior(
        ...     model, n_samples=16, key=jr.PRNGKey(0)
        ... )
        >>> fields.mean(jnp.array([0.5, 0.3])).shape
        (2, 2)
        """
        self._check_input_dim(x)
        if x.ndim == 1:
            return self._mean_single(x)
        return self._mean_batch_jitted(x)

    def band(
        self,
        points: jnp.ndarray,
        *,
        n_directions: int = 181,
        ci_prob: float | Sequence[float] = 0.89,
        ci_kind: str = "eti",
    ) -> ContourBand:
        r"""Pointwise credible band on the one-standard-deviation ellipse at each point.

        The posterior-interval counterpart to :meth:`mean` (a posterior *mean*)
        and :func:`covariance_to_arviz` (posterior *samples*, for diagnostics).

        Every draw gives one covariance field, hence one contour per point.
        Writing each contour as a radius per direction
        (:func:`contour_radii`) makes it a scalar per draw, so an interval
        follows from quantiles over draws. For a :math:`1-\alpha` interval, at
        each direction separately, the band runs between
        :math:`q_{\alpha/2}\{r_i\}` and :math:`q_{1-\alpha/2}\{r_i\}` over the
        :math:`N` draws.

        Parameters
        ----------
        points : jnp.ndarray, shape (n_points, 2)
            Stimulus locations, usually the reference points we report.
        n_directions : int, default=181
            Directions per contour. The grid is closed, so the last direction
            repeats the first, and 181 gives 180 distinct directions, one every
            two degrees. The number is a magic default with nothing behind it:
            large enough that a filled band looks smooth, small enough to stay
            cheap. Nothing in the result depends on the choice.
        ci_prob : float or sequence of float, default=0.89
            How much posterior mass the interval holds. At 0.89, 89% of the
            draws of the radius lie between ``lower`` and ``upper``, in each
            direction separately. 0.89 is ArviZ's default too, so a band and
            ``az.summary`` agree. Pass a sequence for nested bands, and each
            level gets its own interval stacked on a leading axis.
        ci_kind : {"eti", "hdi"}, default="eti"
            Which interval of that mass to take, since many of them hold it.
            ``"eti"`` is equal-tailed: it cuts the same probability off each
            end, so ``lower`` and ``upper`` are quantiles. ``"hdi"`` is
            highest-density: it takes the narrowest interval holding the mass.
            The two differ when the draws are skewed. Radii usually are,
            because a radius cannot go below zero.

        Returns
        -------
        ContourBand

        Raises
        ------
        NotImplementedError
            If the model is not 2-D. The direction grid is a circle, and higher
            dimensions have no single analogous grid.
        ValueError
            If ``ci_kind`` is unknown, or if any ``ci_prob`` is outside (0, 1).

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> from psyphy.model import WPPM, OddityTask, Prior, WPPMCovarianceFieldSamples
        >>> model = WPPM(Prior(input_dim=2, basis_degree=2), OddityTask())
        >>> ref_points = jnp.array([[-0.5, -0.5], [0.5, 0.5]])  # (n_points, 2)
        >>> fields = WPPMCovarianceFieldSamples.from_prior(
        ...     model, n_samples=50, key=jr.PRNGKey(1)
        ... )
        >>> fields.band(ref_points).lower.shape  # (n_points, n_directions)
        (2, 181)
        >>> nested = fields.band(ref_points, ci_prob=[0.5, 0.8, 0.95])
        >>> nested.upper.shape  # (n_levels, n_points, n_directions)
        (3, 2, 181)
        """
        if ci_kind not in _CI_KINDS:
            raise ValueError(
                f"ci_kind must be one of {sorted(_CI_KINDS)}, got {ci_kind!r}."
            )

        if isinstance(ci_prob, (int, float)):
            scalar_level = True
            probs: tuple[float, ...] = (float(ci_prob),)
        else:
            scalar_level = False
            probs = tuple(float(p) for p in ci_prob)
        if not probs:
            raise ValueError("ci_prob must hold at least one probability.")
        for p in probs:
            if not 0.0 < p < 1.0:
                raise ValueError(f"ci_prob must lie in (0, 1), got {p}.")

        if self.model.input_dim != 2:
            raise NotImplementedError(
                "band supports 2-D stimulus spaces only, got "
                f"input_dim={self.model.input_dim}. contour_radii is "
                "d-dimensional, so call it with explicit directions instead."
            )

        # Closed grid, so the last angle repeats the first. n_directions is a magic
        # default (181 -> one direction every two degrees); nothing depends on it.
        phi = jnp.linspace(0.0, 2 * jnp.pi, n_directions)  # radians
        directions = jnp.stack([jnp.cos(phi), jnp.sin(phi)], axis=-1)  # (n_dir, 2)

        self._check_input_dim(points)
        # (n_samples, n_points, 2, 2)
        covs = self._eval_batch_impl(points)
        # (n_samples, n_points, n_directions)
        radii = jax.vmap(lambda c: contour_radii(c, directions))(covs)

        reduce = _CI_KINDS[ci_kind]
        bounds = [reduce(radii, p) for p in probs]
        lower = jnp.stack([lo for lo, _ in bounds])  # (n_levels, n_pts, n_dir)
        upper = jnp.stack([hi for _, hi in bounds])
        # One median whatever ci_prob holds, repeated per level so lower,
        # median and upper share a shape.
        median = jnp.median(radii, axis=0)  # (n_points, n_directions)
        median = jnp.broadcast_to(median, lower.shape)  # (n_levels, n_pts, n_dir)

        return ContourBand(
            points=points,
            directions=directions,
            lower=lower[0] if scalar_level else lower,
            median=median[0] if scalar_level else median,
            upper=upper[0] if scalar_level else upper,
            ci_prob=probs[0] if scalar_level else probs,
            ci_kind=ci_kind,
        )

    # ----------------------------------------------------------------------
    # internals
    # ----------------------------------------------------------------------
    def _check_input_dim(self, x: jnp.ndarray) -> None:
        if x.shape[-1] != self.model.input_dim:
            raise ValueError(
                f"Last axis must be input_dim={self.model.input_dim}, got {x.shape[-1]}"
            )

    def _eval_batch_impl(self, X: jnp.ndarray) -> jnp.ndarray:
        """Σ at every location for every draw: (n_samples, ..., d, d)."""
        # vmap over the leading draw axis of every leaf of self.samples
        return jax.vmap(lambda params: _sigma(self.model, params, X))(self.samples)

    def _mean_single(self, x: jnp.ndarray) -> jnp.ndarray:
        """E[Σ(x)] at one location: Σ for every draw, THEN the mean."""
        per_draw = self._eval_batch_impl(x)  # (n_samples, d, d)
        return jnp.mean(per_draw, axis=0)  # (d, d)

    def _mean_batch_impl(self, X: jnp.ndarray) -> jnp.ndarray:
        """Flatten -> vmap -> reshape, as in ``WPPMCovarianceField``."""
        input_dim = self.model.input_dim
        batch_shape = X.shape[:-1]
        X_flat = X.reshape(-1, input_dim)
        Sigmas_flat = jax.vmap(self._mean_single)(X_flat)
        return Sigmas_flat.reshape(batch_shape + (input_dim, input_dim))
