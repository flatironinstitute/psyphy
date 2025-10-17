"""
wppm.py
-------

Wishart Process Psychophysical Model (WPPM) — MVP-style implementation with
forward-compatible hooks for the full WPPM model.

Goals
-----
1) MVP that runs today:
   - Local covariance Σ(x) is diagonal and *constant* across the space.
   - Discriminability is Mahalanobis distance under Σ(reference).
   - Task mapping (e.g., Oddity, 2AFC) converts discriminability -> p(correct).
   - Likelihood is delegated to the TaskLikelihood (no Bernoulli code here).

2) Forward compatibility with full WPPM model:
   - Expose hyperparameters needed to for example use Model config used in Hong et al.:
       * extra_dims: embedding size for basis expansions (unused in MVP)
       * variance_scale: global covariance scale (unused in MVP)
       * lengthscale: smoothness/length-scale for covariance field (unused in MVP)
       * diag_term: numerical stabilizer added to covariance diagonals (used in MVP)
   - Later, replace `local_covariance` with a basis-expansion Wishart process
     and swap discriminability/likelihood with MC observer simulation.

All numerics use JAX (jax.numpy as jnp) to support autodiff and Optax optimizers
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from .prior import Prior
from .task import TaskLikelihood

# Type aliases for readability
Params = dict[str, jnp.ndarray]
# A "stimulus" is a pair (reference, probe) in model space (shape: (input_dim,))
Stimulus = tuple[jnp.ndarray, jnp.ndarray]


class WPPM:
    """
    Wishart Process Psychophysical Model (WPPM).

    Parameters
    ----------
    input_dim : int
        Dimensionality of the *input stimulus space* (e.g., 2 for isoluminant plane,
        3 for RGB). Both reference and probe live in R^{input_dim}.
    prior : Prior
        Prior distribution over model parameters. MVP uses a simple Gaussian prior
        over diagonal log-variances (see Prior.sample_params()).
    task : TaskLikelihood
        Psychophysical task mapping that defines how discriminability translates
        to p(correct) and how log-likelihood of responses is computed.
        (e.g., OddityTask, TwoAFC)
    noise : Any, optional
        Noise model describing internal representation noise (e.g., GaussianNoise).
        Not used in MVP mapping but passed to the task interface for future MC sims.

    Forward-compatible hyperparameters (MVP stubs)
    ----------------------------------------------
    extra_dims : int, default=0
        Additional embedding dimensions for basis expansions (unused in MVP).
    variance_scale : float, default=1.0
        Global scaling factor for covariance magnitude (unused in MVP).
    lengthscale : float, default=1.0
        Smoothness/length-scale for spatial covariance variation (unused in MVP).
        (formerly "decay_rate")
    diag_term : float, default=1e-6
        Small positive value added to the covariance diagonal for numerical stability.
        MVP uses this in matrix solves; the research model will also use it.
    """

    def __init__(
        self,
        input_dim: int,
        prior: Prior,
        task: TaskLikelihood,
        noise: Any | None = None,
        *,
        extra_dims: int = 0,
        variance_scale: float = 1.0,
        lengthscale: float = 1.0,
        diag_term: float = 1e-6,
    ) -> None:
        # --- core components ---
        self.input_dim = int(input_dim)  # stimulus-space dimensionality
        self.prior = prior  # prior over parameter PyTree
        self.task = task  # task mapping and likelihood
        self.noise = noise  # noise model

        # --- forward-compatible hyperparameters (stubs in MVP) ---
        self.extra_dims = int(extra_dims)
        self.variance_scale = float(variance_scale)
        self.lengthscale = float(lengthscale)
        self.diag_term = float(diag_term)

    # ----------------------------------------------------------------------
    # PARAMETERS
    # ----------------------------------------------------------------------
    def init_params(self, key: Any) -> Params:
        """
        Sample initial parameters from the prior.

        MVP parameters:
            {"log_diag": shape (input_dim,)}
        which defines a constant diagonal covariance across the space.

        Returns
        -------
        params : dict[str, jnp.ndarray]
        """
        return self.prior.sample_params(key)

    # ----------------------------------------------------------------------
    # LOCAL COVARIANCE (Σ(x)), to be replaced by basis-expansion Wishart process
    # ----------------------------------------------------------------------
    def local_covariance(self, params: Params, x: jnp.ndarray) -> jnp.ndarray:
        """
        Return local covariance Σ(x) at stimulus location x.

        MVP:
            Σ(x) = diag(exp(log_diag)), constant across x.
            - Positive-definite because exp(log_diag) > 0.
        Future (full WPPM mode):
            Σ(x) varies smoothly with x via basis expansions and a Wishart-process
            prior controlled by (extra_dims, variance_scale, lengthscale). Those
            hyperparameters are exposed here but not used in MVP.

        Parameters
        ----------
        params : dict
            model parameters (MVP expects "log_diag": (input_dim,)).
        x : jnp.ndarray
            Stimulus location (unused in MVP because Σ is constant).

        Returns
        -------
        Σ : jnp.ndarray, shape (input_dim, input_dim)
        """
        log_diag = params["log_diag"]  # unconstrained diagonal log-variances
        diag = jnp.exp(log_diag)  # enforce positivity
        return jnp.diag(diag)  # constant diagonal covariance

    # ----------------------------------------------------------------------
    # DISCRIMINABILITY (d), later implemented via MC simulation
    # ----------------------------------------------------------------------
    def discriminability(self, params: Params, stimulus: Stimulus) -> jnp.ndarray:
        """
        Compute scalar discriminability d >= 0 for a (reference, probe) pair

        MVP:
            d = sqrt( (probe - ref)^T Σ(ref)^{-1} (probe - ref) )
            with Σ(ref) the local covariance at the reference,
            - We add `diag_term * I` for numerical stability before inversion
        Future (full WPPM mode):
            d is implicit via Monte Carlo simulation of internal noisy responses
            under the task's decision rule (no closed form). In that case, tasks
            will directly implement predict/loglik with MC, and this method may be
            used only for diagnostics.

        Parameters
        ----------
        params : dict
            Model parameters.
        stimulus : tuple
            (reference, probe) arrays of shape (input_dim,).

        Returns
        -------
        d : jnp.ndarray
            Nonnegative scalar discriminability.
        """
        ref, probe = stimulus
        delta = probe - ref  # difference vector in input space
        Sigma = self.local_covariance(params, ref)  # local covariance at reference
        # Add jitter for stable solve; diag_term is configurable
        jitter = self.diag_term * jnp.eye(self.input_dim)
        # Solve (Σ + jitter)^{-1} delta using a PD-aware solver
        x = jax.scipy.linalg.solve(Sigma + jitter, delta, assume_a="pos")
        d2 = jnp.dot(delta, x)  # quadratic form
        # Guard against tiny negative values from numerical error
        return jnp.sqrt(jnp.maximum(d2, 0.0))

    # ----------------------------------------------------------------------
    # PREDICTION (delegates to task)
    # ----------------------------------------------------------------------
    def predict_prob(self, params: Params, stimulus: Stimulus) -> jnp.ndarray:
        """
        Predict probability of a correct response for a single stimulus.

        Design choice:
            WPPM computes discriminability & covariance; the TASK defines how
            that translates to performance. We therefore delegate to:
                task.predict(params, stimulus, model=self, noise=self.noise)

        Parameters
        ----------
        params : dict
        stimulus : (reference, probe)

        Returns
        -------
        p_correct : jnp.ndarray
        """
        return self.task.predict(params, stimulus, self, self.noise)

    # ----------------------------------------------------------------------
    # LIKELIHOOD (delegates to task)
    # ----------------------------------------------------------------------
    def log_likelihood(
        self,
        params: Params,
        refs: jnp.ndarray,
        probes: jnp.ndarray,
        responses: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute the log-likelihood for arrays of trials.

        IMPORTANT:
            We delegate to the TaskLikelihood to avoid duplicating Bernoulli (MPV)
            or MC likelihood logic in multiple places. This keeps responsibilities
            clean and makes adding new tasks straightforward.

        Parameters
        ----------
        params : dict
            Model parameters.
        refs : jnp.ndarray, shape (N, input_dim)
        probes : jnp.ndarray, shape (N, input_dim)
        responses : jnp.ndarray, shape (N,)
            Typically 0/1; task may support richer encodings.

        Returns
        -------
        loglik : jnp.ndarray
            Scalar log-likelihood (task-only; add prior outside if needed)
        """
        # We need a ResponseData-like object. To keep this method usable from
        # array inputs, we construct one on the fly. If you already have a
        # ResponseData instance, prefer `log_likelihood_from_data`.
        from psyphy.data.dataset import ResponseData  # local import to avoid cycles

        data = ResponseData()
        # ResponseData.add_trial(ref, probe, resp)
        for r, p, y in zip(refs, probes, responses):
            data.add_trial(r, p, int(y))
        return self.task.loglik(params, data, self, self.noise)

    def log_likelihood_from_data(self, params: Params, data: Any) -> jnp.ndarray:
        """
        Compute log-likelihood directly from a ResponseData object.

        Why delegate to the task?
            - The task knows the decision rule (oddity, 2AFC, ...).
            - The task can use the model (this WPPM) to fetch discriminabilities
            - and the task can use the noise model if it needs MC simulation

        Parameters
        ----------
        params : dict
            Model parameters.
        data : ResponseData
            Collected trial data.

        Returns
        -------
        loglik : jnp.ndarray
            scalar log-likelihood (task-only; add prior outside if needed)
        """
        return self.task.loglik(params, data, self, self.noise)

    # ----------------------------------------------------------------------
    # POSTERIOR-STYLE CONVENIENCE (OPTIONAL)
    # ----------------------------------------------------------------------
    def log_posterior_from_data(self, params: Params, data: Any) -> jnp.ndarray:
        """
        Convenience helper if you want log posterior in one call (MVP).

        This simply adds the prior log-probability to the task log-likelihood.
        Inference engines (e.g., MAP optimizer) typically optimize this quantity.

        Returns
        -------
        jnp.ndarray : scalar log posterior = loglik(params | data) + log_prior(params)
        """
        return self.log_likelihood_from_data(params, data) + self.prior.log_prob(params)
