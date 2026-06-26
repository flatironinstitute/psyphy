"""
wppm.py
-------

Wishart Process Psychophysical Model (WPPM)

Goals
-----
Wishart Process Psychophysical Model (WPPM):
   - Expose hyperparameters needed to for example use Model config used in Hong et al.:
       * extra_dims: embedding size for basis expansions
       * variance_scale: global covariance scale
       * decay_rate: smoothness/length-scale for covariance field
       * diag_term: numerical stabilizer added to covariance diagonals

All numerics use JAX (jax.numpy as jnp) to support autodiff and optax optimizers
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from psyphy.utils.math import chebyshev_basis

from .base import Model
from .likelihood import BernoulliTaskLikelihood, TaskLikelihood
from .prior import Prior

# Type aliases for readability
Params = dict[str, jnp.ndarray]
# A "stimulus" is a pair (reference, comparison) in model space (shape: (input_dim,))
Stimulus = tuple[jnp.ndarray, jnp.ndarray]


class WPPM(Model):
    """
    Wishart Process Psychophysical Model (WPPM).

    Parameters
    ----------
    input_dim : int
        Dimensionality of the *input stimulus space* (e.g., 2 for isoluminant plane,
        3 for RGB). Both reference and comparison live in R^{input_dim}.
    prior : Prior
        Prior distribution over model parameters. Controls basis_degree in WPPM (basis expansion).
        The WPPM delegates
        to prior.basis_degree to ensure consistency between parameter sampling and
        basis evaluation.
    likelihood : TaskLikelihood
        Psychophysical task mapping that defines how discriminability translates
        to p(correct) and how log-likelihood of responses is computed.
        (e.g., OddityTask)
    noise : Any, optional
        Noise model describing internal representation noise (e.g., GaussianNoise).

    hyperparameters
    -----------------------------------
    extra_dims : int, default=0
        Additional embedding dimensions for basis expansions (beyond input_dim).
        embedding_dim = input_dim + extra_dims.
    variance_scale : float, default=1.0
        Global scaling factor for covariance magnitude
    decay_rate : float, default=1.0
        Smoothness/length-scale for spatial covariance variation
    diag_term : float, default=1e-6
        Small positive value added to the covariance diagonal for numerical stability.

    **model_kwargs : Any
        Reserved for *future* keyword arguments accepted by the base `Model.__init__`.
        Do not pass WPPM math knobs or task/likelihood knobs here.
    """

    def __init__(
        self,
        prior: Prior,
        likelihood: TaskLikelihood,
        noise: Any | None = None,
        *,  # everything after here is keyword-only
        input_dim: int = 2,
        extra_dims: int = 1,
        variance_scale: float = 4e-3,
        decay_rate: float = 0.4,
        diag_term: float = 1e-6,
        **model_kwargs: Any,
    ) -> None:
        # Base-model configuration.
        #
        # `model_kwargs` is reserved for *future* base `Model.__init__` kwargs.
        # It should NOT be used for WPPM-specific math (e.g. alternative covariance
        # parameterizations) or for task-specific likelihood knobs.
        if model_kwargs:
            known_misuses = {"num_samples", "bandwidth", "online_config"}
            bad = sorted(known_misuses.intersection(model_kwargs.keys()))
            if bad:
                raise TypeError(
                    "Do not pass task-specific kwargs via WPPM(..., **model_kwargs). "
                    f"Move {bad} into the task config (e.g. OddityTaskConfig)."
                )

        super().__init__(**model_kwargs)

        # --- core components ---
        self.input_dim = int(input_dim)  # stimulus-space dimensionality
        self.prior = prior  # prior over parameter PyTree

        if self.prior.input_dim != self.input_dim:
            raise ValueError(
                f"Dimension mismatch: Model initialized with input_dim={self.input_dim}, "
                f"but Prior expects input_dim={self.prior.input_dim}."
            )

        self.likelihood = likelihood  # task mapping and likelihood
        self.noise = noise  # noise model

        self.extra_dims = int(extra_dims)
        self.variance_scale = float(variance_scale)
        self.decay_rate = float(decay_rate)
        self.diag_term = float(diag_term)

    @property
    def basis_degree(self) -> int | None:
        """
        Chebyshev polynomial degree for Wishart process basis expansion.

        This property delegates to self.prior.basis_degree to ensure consistency
        between parameter sampling and basis evaluation.

        Returns
        -------
        int | None
            Degree of Chebyshev polynomial basis (0 = constant, 1 = linear, etc.)

        Notes
        -----
        WPPM gets its basis_degree parameter from Prior.basis_degree.
        """
        return self.prior.basis_degree

    @property
    def embedding_dim(self) -> int:
        """
        Dimension of the embedding space.

        embedding_dim = input_dim + extra_dims.
        this represents the full perceptual space where:
        - First input_dim dimensions correspond to observable stimulus features
        - Remaining extra_dims are latent  dimensions

        Returns
        -------
        int
            input_dim + extra_dims

        Notes
        -----
        This is a computed property, not a constructor parameter.
        """
        if self.basis_degree is None:
            raise ValueError("Basis degree must be set for embedding.")
        # Wishart mode: full perceptual space
        return self.input_dim + self.extra_dims

    def _validate_basis_input(self, x: jnp.ndarray) -> None:
        """Validate that inputs are in the expected basis domain.

        Design contract (current):
            - WPPM does **not** normalize stimuli.
            - Users must provide stimulus coordinates already in the basis domain.

        For the current Chebyshev basis implementation, the domain is ``[-1, 1]``
        per dimension.

        Notes
        -----
        This is intentionally a lightweight, eager validator (Python exceptions).
        If we later need fully-jittable checks, we can replace this with
        ``jax.experimental.checkify``.
        """

        x = jnp.asarray(x)
        if x.ndim != 1 or x.shape[0] != self.input_dim:
            raise ValueError(
                f"Expected x with shape ({self.input_dim},), got {tuple(x.shape)}"
            )

        # NOTE: This method may be called from jitted code paths.
        # Avoid Python bool/float conversions on JAX tracers.
        #
        # Strategy:
        # - If x is concrete (non-traced), do normal Python ('eager') exceptions.
        # - If x is traced (e.g., under jit/vmap), skip eager checks to
        #   remain JIT-compatible.
        # JAX tracing produces Tracer objects (not concrete values). Eager validation
        # (Python exceptions) is only safe on concrete arrays.
        if isinstance(x, jax.core.Tracer):
            return

        if jnp.any(~jnp.isfinite(x)):
            raise ValueError("Stimulus x contains NaN/Inf.")

        # Chebyshev basis expects inputs in [-1, 1].
        tol = 1e-6
        if jnp.any(x < (-1.0 - tol)) or jnp.any(x > (1.0 + tol)):
            x_min = float(jnp.min(x))
            x_max = float(jnp.max(x))
            raise ValueError(
                "Stimulus x is outside the Chebyshev domain [-1, 1]. "
                f"Observed min={x_min:g}, max={x_max:g}. "
                "Normalize your data to align with domain of basis "
                "in order to avoid unexpected behavior."
            )

    # ----------------------------------------------------------------------
    # PARAMETERS
    # ----------------------------------------------------------------------
    def init_params(self, key: jax.Array) -> Params:
        """Sample initial parameters from the prior.

        Returns
        -------
        params : dict[str, jnp.ndarray]
        """
        return self.prior.sample_params(key)

    # ----------------------------------------------------------------------
    # WISHART PROCESS COVARIANCE
    # ----------------------------------------------------------------------
    def _evaluate_basis_at_point(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate all Chebyshev basis functions at point x, keeping structure for einsum.

        For 2D: returns phi_ij(x) = T_i(x_1) * T_j(x_2) with shape (degree+1, degree+1)
        For 3D: returns phi_ijk(x) = T_i(x_1) * T_j(x_2) * T_k(x_3) with shape (degree+1, degree+1, degree+1)

        Note: chebyshev_basis(x, degree=d) returns (degree+1) basis functions [T_0, ..., T_d].

        Parameters
        ----------
        x : jnp.ndarray, shape (input_dim,)
            Stimulus coordinates in the Chebyshev domain ``[-1, 1]``.
            The model does not normalize inputs; callers must normalize upstream.

        Returns
        -------
        phi : jnp.ndarray
            Basis function values with structured shape for efficient einsum.
            Shape is (degree+1, degree+1) for 2D or (degree+1, degree+1, degree+1) for 3D.
        """

        if self.basis_degree is None:
            raise ValueError(
                "Cannot evaluate basis: basis_degree is None. "
                "Set basis_degree to a valid integer."
            )

        self._validate_basis_input(x)
        x_norm = x

        if self.input_dim == 2:
            # Evaluate basis functions: \phi_ij(x) = T_i(x_1) * T_j(x_2)
            # chebyshev_basis returns (1, degree+1) for each dimension
            cheb_0 = chebyshev_basis(x_norm[0:1], degree=self.basis_degree)[
                0, :
            ]  # (degree+1,)
            cheb_1 = chebyshev_basis(x_norm[1:2], degree=self.basis_degree)[
                0, :
            ]  # (degree+1,)
            phi = cheb_0[:, None] * cheb_1[None, :]  # (degree+1, degree+1)

        elif self.input_dim == 3:
            # 3D case: \phi_ijk(x) = T_i(x_1) * T_j(x_2) * T_k(x_3)
            # phi.shape = (degree+1, degree+1, degree+1)
            cheb_0 = chebyshev_basis(x_norm[0:1], degree=self.basis_degree)[0, :]
            cheb_1 = chebyshev_basis(x_norm[1:2], degree=self.basis_degree)[0, :]
            cheb_2 = chebyshev_basis(x_norm[2:3], degree=self.basis_degree)[0, :]
            phi = cheb_0[:, None, None] * cheb_1[None, :, None] * cheb_2[None, None, :]

        elif self.input_dim == 1:
            # 1D case: \phi_i(x) = T_i(x_1)
            # phi.shape = (degree+1,)
            phi = chebyshev_basis(x_norm[0:1], degree=self.basis_degree)[0, :]

        else:
            raise NotImplementedError(
                f"Wishart process currently only supports 1D, 2D and 3D. Got input_dim={self.input_dim}"
            )

        return phi

    def _compute_sqrt(self, params: Params, x: jnp.ndarray) -> jnp.ndarray:
        r"""
        Compute "square root" matrix U(x) from basis expansion.

        This is the core of the Wishart process: U(x) = Σ_ij W_ij * \phi_ij(x)
        where W_ij are learned coefficients and \phi_ij are Chebyshev basis functions.

        The covariance is then Σ(x) = U(x) @ U(x)^T + diag_term * I, which is
        guaranteed to be positive definite.

        Parameters
        ----------
        params : dict
            Model parameters. Must contain "W" for Wishart mode.
        x : jnp.ndarray, shape (input_dim,)
            Stimulus location

        Returns
        -------
        U : jnp.ndarray, shape (input_dim, embedding_dim)
            Rectangular square root matrix if extra_dims > 0.
            embedding_dim = input_dim + extra_dims

        Raises
        ------
        ValueError
            If params doesn't contain "W" (not in Wishart mode)

        Notes
        -----
        U is rectangular when extra_dims > 0.
        When multiplied U @ U^T, this produces covariance in stimulus space:
        Σ(x) \in R^(input_dim x input_dim)
        """
        if "W" not in params:
            raise ValueError("Cannot compute U(x): params missing 'W'. ")

        W = params["W"]
        phi = self._evaluate_basis_at_point(x)

        # Linear combination: U(x) = Σ_ij W_ij * phi_ij(x)
        # Einstein summation over basis function indices
        if self.input_dim == 2:
            # W[i,j,d,e] * phi[i,j] -> U[d,e]
            # W is (degree+1, degree+1, input_dim, embedding_dim)
            # U is rectangular if extra_dims > 0: (input_dim, embedding_dim)
            U = jnp.einsum("ijde,ij->de", W, phi)
        elif self.input_dim == 3:
            # W[i,j,k,d,e] * phi[i,j,k] -> U[d,e]
            # W is (degree+1, degree+1, degree+1, input_dim, embedding_dim)
            # U is rectangular if extra_dims > 0: (input_dim, embedding_dim)
            U = jnp.einsum("ijkde,ijk->de", W, phi)
        elif self.input_dim == 1:
            # W[i,d,e] * phi[i] -> U[d,e]
            # W is (degree+1, input_dim, embedding_dim)
            U = jnp.einsum("ide,i->de", W, phi)
        else:
            raise NotImplementedError(
                f"Wishart process only supports 1D, 2D and 3D. Got input_dim={self.input_dim}"
            )

        return U

    # ----------------------------------------------------------------------
    # LOCAL COVARIANCE (Σ(x))
    # ----------------------------------------------------------------------
    def local_covariance(self, params: Params, x: jnp.ndarray) -> jnp.ndarray:
        """
        Return local covariance Σ(x) at stimulus location x.


        Wishart mode (basis_degree set):
            Σ(x) = U(x) @ U(x)^T + diag_term * I
            where U(x) is rectangular (input_dim, embedding_dim) if extra_dims > 0.
            - Varies smoothly with x
            - Guaranteed positive-definite
            - Returns stimulus covariance directly (input_dim, input_dim)

        Parameters
        ----------
        params : dict
            Model parameters:
            - WPPM: {"W": (degree+1, ..., input_dim, embedding_dim)}
        x : jnp.ndarray, shape (input_dim,)
            Stimulus location

        Returns
        -------
        Σ : jnp.ndarray, shape (input_dim, input_dim)
            Covariance matrix in stimulus space.
        """
        # WPPM: spatially-varying covariance
        if "W" in params:
            U = self._compute_sqrt(params, x)  # (input_dim, embedding_dim)
            # Σ(x) = U(x) @ U(x)^T + diag_term * I
            # Result is (input_dim, input_dim)
            Sigma = U @ U.T + self.diag_term * jnp.eye(self.input_dim)
            return Sigma

        raise ValueError("params must contain 'W' (weights of WPPM)")

    # ----------------------------------------------------------------------
    # PREDICTION (delegates to likelihood)
    # ----------------------------------------------------------------------
    def predict_prob(
        self, params: Params, stimulus: Stimulus, **likelihood_kwargs: Any
    ) -> jnp.ndarray:
        """
        Predict probability of a correct response for a single stimulus.

        Design choice:
            WPPM computes discriminability & covariance; the LIKELIHOOD defines how
            that translates to performance. We therefore delegate to:
                likelihood.predict(params, stimulus, model=self, noise=self.noise)

        Parameters
        ----------
        params : dict
        stimulus : tuple[jnp.ndarray, jnp.ndarray]
             (reference, comparison) pair in model space.

        Returns
        -------
        p_correct : jnp.ndarray
        """
        # Strict task-owned configuration:
        # - MC control knobs (e.g. num_samples/bandwidth) live in the task config.
        # - WPPM.predict_prob therefore does not accept task-specific kwargs.
        if likelihood_kwargs:
            raise TypeError(
                f"WPPM.predict_prob got unexpected kwargs: {likelihood_kwargs}. "
                "Configure likelihood behavior via the TaskLikelihood object itself."
            )

        if not isinstance(self.likelihood, BernoulliTaskLikelihood):
            raise NotImplementedError(
                "WPPMP currently only supports "
                "BernoulliTaskLikelihood. Gaussian support requires "
                "updating to handle (mu, sigma) returns."
            )

        stimuli = jnp.stack(stimulus, axis=1)
        return self.likelihood.predict(params, stimuli, self)[0]

    # ----------------------------------------------------------------------
    # LIKELIHOOD (delegates to likelihood component)
    # ----------------------------------------------------------------------
    # TODO: add random key
    def log_likelihood_from_data(
        self, params: Params, data: Any, *, key: jax.Array | None = None
    ) -> jnp.ndarray:
        """Compute log-likelihood directly from a batched data object.

        Why delegate to the likelihood?
            - The likelihood knows the decision rule (oddity, 2AFC, ...).
            - The likelihood can use the model (this WPPM) to fetch discriminabilities.
            - The likelihood can use the noise model if it needs MC simulation.

        Parameters
        ----------
        params : dict
            Model parameters.
        data : TrialData (or any object with refs/comparisons/responses arrays)
            Collected trial data.
        key : jax.Array | None, optional
            JAX random key for MC likelihood evaluation. When provided, a fresh
            noise realization is drawn every call — required for correct stochastic
            gradient estimates during optimization. When None, the task falls back
            to ``OddityTaskConfig.default_key_seed`` (useful for fixed evaluation
            and testing, but should not be used during gradient-based optimization).

        Returns
        -------
        loglik : jnp.ndarray
            Scalar log-likelihood (task-only; add prior outside if needed).
        """
        return self.likelihood.loglik(params, data, self, key=key)

    # ----------------------------------------------------------------------
    # POSTERIOR-STYLE CONVENIENCE (OPTIONAL)
    # ----------------------------------------------------------------------
    def log_posterior_from_data(
        self, params: Params, data: Any, *, key: jax.Array | None = None
    ) -> jnp.ndarray:
        """Compute log posterior from data.

        This simply adds the prior log-probability to the task log-likelihood.
        Inference engines (e.g., MAP optimizer) typically optimize this quantity.

        Parameters
        ----------
        params : dict
            Model parameters.
        data : TrialData
            Collected trial data.
        key : jax.Array | None, optional
            JAX random key for the MC likelihood. Must be provided during
            optimization so each gradient step uses a fresh noise realization.
            When None, falls back to ``OddityTaskConfig.default_key_seed``.

        Returns
        -------
        jnp.ndarray
            Scalar log posterior = loglik(params | data) + log_prior(params).
        """
        return self.log_likelihood_from_data(
            params, data, key=key
        ) + self.prior.log_prob(params)
