"""psyphy.model.likelihood.oddity
---------------------------------

Three-alternative forced-choice oddity task (MC-based likelihood).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr

from .base import TaskLikelihood


@dataclass(frozen=True, slots=True)
class OddityTaskConfig:
    """Configuration for :class:`OddityTask`.

    This is the single source of truth for MC likelihood controls.

    Attributes
    ----------
    num_samples : int
        Number of Monte Carlo samples per trial.
    bandwidth : float
        Logistic CDF smoothing bandwidth.
    default_key_seed : int
        Seed used when no key is provided (keeps behavior deterministic by
        default while allowing reproducibility control upstream).
    """

    num_samples: int = 1000
    bandwidth: float = 1e-2
    default_key_seed: int = 0

    def __post_init__(self) -> None:
        if int(self.num_samples) <= 0:
            raise ValueError(f"num_samples must be > 0, got {self.num_samples}")
        if float(self.bandwidth) <= 0:
            raise ValueError(f"bandwidth must be > 0, got {self.bandwidth}")


class OddityTask(TaskLikelihood):
    """
    Three-alternative forced-choice oddity task (MC-based only).

    Implements the full 3-stimulus oddity task using Monte Carlo simulation:
        - Samples three internal representations per trial (z0, z1, z2)
        - Uses proper oddity decision rule with three pairwise distances
        - Suitable for complex covariance structures

    Notes
    -----
    MC simulation in loglik() (full 3-stimulus oddity):
        1. Sample three internal representations: z_ref, z_refprime ~ N(ref, Σ_ref), z_comparison ~ N(comparison, Σ_comparison)
        2. Compute average covariance: Σ_avg = (2/3) Σ_ref + (1/3) Σ_comparison
        3. Compute three pairwise Mahalanobis distances:
           - d^2(z_ref, z_refprime) = distance between two reference samples
           - d^2(z_ref, z_comparison) = distance from ref to comparison
           - d^2(z_refprime, z_comparison) = distance from reference_prime to comparison
        4. Apply oddity decision rule: delta = min(d^2(z_ref,z_comparison), d^2(z_refprime,z_comparison)) - d^2(z_ref,z_refprime)
        5. Logistic smoothing: P(correct) \approx logistic.cdf(delta / bandwidth)
        6. Average over samples

    Examples
    --------
    >>> from psyphy.model.likelihood import OddityTask, OddityTaskConfig
    >>> from psyphy.model import WPPM, Prior
    >>> from psyphy.model.noise import GaussianNoise
    >>> import jax.numpy as jnp
    >>> import jax.random as jr
    >>>
    >>> likelihood = OddityTask(
    ...     config=OddityTaskConfig(num_samples=1000, bandwidth=1e-2)
    ... )
    >>> model = WPPM(
    ...     input_dim=2,
    ...     prior=Prior(input_dim=2),
    ...     likelihood=likelihood,
    ...     noise=GaussianNoise(),
    ... )
    >>> params = model.init_params(jr.PRNGKey(0))
    >>> from psyphy.data.dataset import ResponseData
    >>> data = ResponseData()
    >>> data.add_trial(
    ...     ref=jnp.array([0.0, 0.0]), comparison=jnp.array([0.3, 0.2]), resp=1
    ... )
    >>> ll = likelihood.loglik(params, data, model, key=jr.PRNGKey(42))
    >>> print(f"Log-likelihood (MC): {ll:.4f}")
    """

    def __init__(self, config: OddityTaskConfig | None = None) -> None:
        # No analytical parameters in MC-only mode.
        self.config = config or OddityTaskConfig()

    def predict(
        self,
        params: Any,
        ref: jnp.ndarray,
        comparison: jnp.ndarray,
        model: Any,
        *,
        key: Any = None,
    ) -> jnp.ndarray:
        """Return p(correct) for a single (ref, comparison) trial via MC simulation.

        MC controls (``num_samples``, ``bandwidth``) are read from
        :class:`OddityTaskConfig`. Pass ``key`` to control randomness; when
        None, ``config.default_key_seed`` is used.
        """
        num_samples = int(self.config.num_samples)
        bandwidth = float(self.config.bandwidth)
        if key is None:
            key = jr.PRNGKey(int(self.config.default_key_seed))

        return self._simulate_trial_mc(
            params=params,
            ref=ref,
            comparison=comparison,
            model=model,
            num_samples=num_samples,
            bandwidth=bandwidth,
            key=key,
        )

    def _simulate_trial_mc(
        self,
        params: Any,
        ref: jnp.ndarray,
        comparison: jnp.ndarray,
        model: Any,
        num_samples: int,
        bandwidth: float,
        key: Any,
    ) -> jnp.ndarray:
        """
        Simulate a single 3-stimulus oddity trial via Monte Carlo.

        This implements the FULL oddity task where the observer sees three stimuli:
        two identical references and one comparison. The task is to identify which
        stimulus is the "odd one out" (the comparison).

        Parameters
        ----------
        params : Any
            Model parameters as expected by ``model._compute_sqrt``.
        ref : jnp.ndarray, shape (input_dim,)
            Reference stimulus (2 samples represented)
        comparison : jnp.ndarray, shape (input_dim,)
            Probe stimulus (1 sample represented, the "odd one out")
        model : WPPM
            Model instance providing covariance structure and ``model.noise``.
        num_samples : int
            Number of Monte Carlo samples for estimating P(correct)
        bandwidth : float
            Logistic smoothing parameter (controls decision sharpness)
        key : PRNGKey
            JAX random key for sampling

        Returns
        -------
        float
            Estimated P(correct) for this trial, in range [0, 1]

        Notes
        -----
        **Full 3-stimulus oddity task algorithm:**

        1. Sample three internal representations:
           - z_ref, z_refprime ~ N(ref, Σ_ref)     [two samples from reference]
           - z_comparison ~ N(comparison, Σ_comparison)        [one sample from comparison]

        2. Compute covariance for distance metric:
           - Σ_avg = (2/3) * Σ_ref + (1/3) * Σ_comparison
           - Weighted by stimulus frequency (2 refs, 1 comparison)

        3. Compute three pairwise Mahalanobis distances:
           - d^2(z_ref, z_refprime) = (z_ref - z_refprime).T @ Σ_avg^{-1} @ (z_ref - z_refprime)
           - d^2(z_ref, z_comparison) = (z_ref - z_comparison).T @ Σ_avg^{-1} @ (z_ref - z_comparison)
           - d^2(z_refprime, z_comparison) = (z_refprime - z_comparison).T @ Σ_avg^{-1} @ (z_refprime - z_comparison)

        4. Decision rule (correct response):
           - delta = min(d^2(z_ref,z_comparison), d^2(z_refprime,z_comparison)) - d^2(z_ref,z_refprime)
           - delta > 0 indicates correct identification of comparison as odd

        5. Smooth decision with logistic CDF:
           - P(correct | sample) \approx sigmoid(delta / bandwidth)

        6. Monte Carlo average:
           - P(correct) \approx mean over num_samples
        """
        input_dim = ref.shape[0]
        if model.basis_degree is None:
            raise ValueError(
                "(Expected a basis degree, got None. model.basis_degree must not be None)."
            )

        # STEP 1: Compute covariance structures at ref and comparison locations
        U_ref = model._compute_sqrt(params, ref)  # (input_dim, embedding_dim)
        U_comparison = model._compute_sqrt(
            params, comparison
        )  # (input_dim, embedding_dim)

        diag_term = model.diag_term
        sqrt_diag = jnp.sqrt(diag_term)

        # STEP 2: Sample internal representations (3 samples from 2 distributions)
        keys = jr.split(key, 6)
        embed_dim = U_ref.shape[1]  # type: ignore

        n_ref_embed = model.noise.sample_standard(keys[0], (num_samples, embed_dim))
        n_refprime_embed = model.noise.sample_standard(
            keys[1], (num_samples, embed_dim)
        )
        n_comparison_embed = model.noise.sample_standard(
            keys[2], (num_samples, embed_dim)
        )
        n_ref_diag = model.noise.sample_standard(keys[3], (num_samples, input_dim))
        n_refprime_diag = model.noise.sample_standard(keys[4], (num_samples, input_dim))
        n_comparison_diag = model.noise.sample_standard(
            keys[5], (num_samples, input_dim)
        )

        # Reparameterization: z = n_embed @ U.T + mean + sqrt(diag_term) * n_diag
        z_ref = n_ref_embed @ U_ref.T + ref[None, :] + sqrt_diag * n_ref_diag  # type: ignore
        z_refprime = (
            n_refprime_embed @ U_ref.T + ref[None, :] + sqrt_diag * n_refprime_diag
        )  # type: ignore
        z_comparison = (  # type: ignore
            n_comparison_embed @ U_comparison.T
            + comparison[None, :]
            + sqrt_diag * n_comparison_diag
        )

        # STEP 3: Compute average covariance for Mahalanobis distance
        Sigma_ref_full = U_ref @ U_ref.T + diag_term * jnp.eye(input_dim)  # type: ignore
        Sigma_comparison_full = U_comparison @ U_comparison.T + diag_term * jnp.eye(
            input_dim
        )  # type: ignore
        Sigma_avg = (2.0 / 3.0) * Sigma_ref_full + (1.0 / 3.0) * Sigma_comparison_full

        # STEP 4: Compute three pairwise Mahalanobis distances
        diff_ref_refprime = z_ref - z_refprime
        diff_ref_comparison = z_ref - z_comparison
        diff_refprime_comparison = z_refprime - z_comparison

        diffs_stacked = jnp.stack(
            [diff_ref_refprime, diff_ref_comparison, diff_refprime_comparison], axis=0
        )
        solved = jax.vmap(lambda d: jnp.linalg.solve(Sigma_avg, d.T))(diffs_stacked)

        d_sq_ref_reference_prime = jnp.sum(diff_ref_refprime * solved[0].T, axis=1)
        d_sq_ref_comparison = jnp.sum(diff_ref_comparison * solved[1].T, axis=1)
        d_sq_refprime_comparison = jnp.sum(
            diff_refprime_comparison * solved[2].T, axis=1
        )

        # STEP 5: Oddity decision rule
        delta = (
            jnp.minimum(d_sq_ref_comparison, d_sq_refprime_comparison)
            - d_sq_ref_reference_prime
        )

        # STEP 6: Logistic smoothing + Monte Carlo average
        prob_correct_per_sample = jax.scipy.stats.logistic.cdf(delta / bandwidth)
        prob = jnp.mean(prob_correct_per_sample)

        eps = 1e-6
        return jnp.clip(prob, eps, 1.0 - eps)
