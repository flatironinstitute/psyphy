"""psyphy.model.task

Task likelihoods for psychophysical experiments.

This module defines task-specific mappings from a model (e.g., WPPM) and stimuli
to response likelihoods.

Current direction
-----------------
`OddityTask`: the log-likelihood is computed via Monte Carlo observer
simulation of the full 3-stimulus oddity decision rule (two identical references,
one comparison).

The public API is:

- ``TaskLikelihood.predict(params, stimuli, model, noise)``
    Optional fast predictor for p(correct). For MC-only tasks this may be
    unimplemented.

- ``TaskLikelihood.loglik(params, data, model, noise, **kwargs)``
    Compute log-likelihood of observed responses under this task.

Connections
-----------
- WPPM delegates to the task to compute likelihood.
- Noise models are passed through so tasks can simulate observer responses.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr

Stimulus = tuple[jnp.ndarray, jnp.ndarray]


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


class TaskLikelihood(ABC):
    """
    Abstract base class for task likelihoods
    """

    @abstractmethod
    def predict(
        self, params: Any, stimuli: Stimulus, model: Any, noise: Any
    ) -> jnp.ndarray:
        """Predict probability of correct response for a stimulus."""
        ...

    @abstractmethod
    def loglik(
        self, params: Any, data: Any, model: Any, noise: Any, **kwargs: Any
    ) -> jnp.ndarray:
        """Compute log-likelihood of observed responses under this task.

        Why ``**kwargs``?
        - Different tasks may need different optional runtime controls.
        - MC-based tasks may need parameters such as a PRNG ``key``.
            In particular, :class:`OddityTask` takes Monte Carlo controls
            (``num_samples`` and ``bandwidth``) exclusively from
            :class:`OddityTaskConfig` to avoid silent mismatch bugs.

        Notes
        -----
        - Task implementations should document which kwargs they accept.
        - Callers should not assume arbitrary kwargs are supported.
        """
        ...


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
    >>> from psyphy.model.task import OddityTask
    >>> from psyphy.model.task import OddityTaskConfig
    >>> from psyphy.model import WPPM, Prior
    >>> from psyphy.model.noise import GaussianNoise
    >>> import jax.numpy as jnp
    >>> import jax.random as jr
    >>>
    >>> # Create task and model (task-owned MC controls)
    >>> task = OddityTask(config=OddityTaskConfig(num_samples=1000, bandwidth=1e-2))
    >>> model = WPPM(
    ...     input_dim=2, prior=Prior(input_dim=2), task=task, noise=GaussianNoise()
    ... )
    >>> params = model.init_params(jr.PRNGKey(0))

    >>> # MC simulation
    >>> from psyphy.data.dataset import ResponseData
    >>> data = ResponseData()
    >>> data.add_trial(ref, comparison, resp=1)
    >>> ll_mc = task.loglik(params, data, model, model.noise, key=jr.PRNGKey(42))
    >>> print(f"Log-likelihood (MC): {ll_mc:.4f}")
    """

    def __init__(self, config: OddityTaskConfig | None = None) -> None:
        # No analytical parameters in MC-only mode.
        self.config = config or OddityTaskConfig()

    def predict(
        self, params: Any, stimuli: Stimulus, model: Any, noise: Any
    ) -> jnp.ndarray:
        """Predict p(correct) for a single (ref, comparison) stimulus.

        Even though OddityTask is *MC-only*, we still implement ``predict``.
        Reason: large parts of the library (posterior predictive, acquisition
        functions, diagnostics, etc.) need a forward model that returns
        p(correct) at candidate stimuli. Historically this used an analytical
        approximation, but in MC-only mode we compute it via simulation.

        Notes
        -----
        - This method is intentionally lightweight: it performs the same
          single-trial Monte Carlo simulation used by ``loglik``.
                - If you need to control MC fidelity/smoothing, set
                    ``OddityTaskConfig(num_samples=..., bandwidth=...)`` when you
                    construct the task.
                - If you need reproducible randomness, pass ``key=...`` to ``loglik``.
        """

        num_samples = int(self.config.num_samples)
        bandwidth = float(self.config.bandwidth)
        key = jr.PRNGKey(int(self.config.default_key_seed))

        ref, comparison = stimuli
        return self._simulate_trial_mc(
            params=params,
            ref=ref,
            comparison=comparison,
            model=model,
            noise=noise,
            num_samples=num_samples,
            bandwidth=bandwidth,
            key=key,
        )

    # Intentionally no `predict_with_kwargs`: we want a single source of truth.

    def loglik(
        self, params: Any, data: Any, model: Any, noise: Any, **kwargs: Any
    ) -> jnp.ndarray:
        """
            Compute log-likelihood via Monte Carlo observer simulation.

            This method implements the FULL 3-stimulus oddity task. Instead of using
            an analytical approximation, we:
            1. Sample three internal noisy representations per trial:
               - z_ref, z_refprime ~ N(ref, Σ_ref)  [two samples from reference]
               - z_comparison ~ N(comparison, Σ_comparison)           [one sample from comparison]
            2. Compute three pairwise Mahalanobis distances
            3. Apply oddity decision rule: comparison is odd if it's farther from BOTH ref and reference_prime
            4. Apply logistic smoothing to approximate P(correct)
            5. Average over MC samples

            Parameters
            ----------
            params : Any
                Model parameters as expected by ``model._compute_sqrt``.
            data : ResponseData
                Trial data with refs, comparisons, and responses
            model : WPPM
                Model instance providing ``_compute_sqrt`` for covariance computation.
            noise : NoiseModel
                Observer noise model (provides ``sample_standard``).
            key : jax.random.PRNGKey, optional
                Random key for reproducible sampling.
                If None, uses ``OddityTaskConfig.default_key_seed``.

            Returns
            -------
            jnp.ndarray
                Scalar sum of log-likelihoods over all trials.
                Same shape and interpretation as ``loglik``.

            Raises
            ------
            TypeError
                If ``num_samples`` or ``bandwidth`` are provided as kwargs.
            ValueError
                If the task configuration is invalid (e.g. ``num_samples <= 0``).

            Notes
            -----
            Monte Carlo controls (``num_samples``, ``bandwidth``) are owned by the
            task configuration:

            - Create the task with ``OddityTask(config=OddityTaskConfig(...))``.
            - Pass only the PRNG ``key`` at call time when you want to control
              randomness.

            Notes
            -----
            **Full 3-stimulus oddity task algorithm:**

            For each trial (ref, comparison, response):
            1. Compute covariances:
               - Σ_ref = U_ref @ U_ref.T + σ^2 I
               - Σ_comparison = U_comparison @ U_comparison.T + σ^2 I
               - Σ_avg = (2/3) Σ_ref + (1/3) Σ_comparison  [weighted by stimulus frequency]

            2. Sample three internal representations:
               - z_ref, z_refprime ~ N(ref, Σ_ref)  [2 samples from reference, num_samples times each]
               - z_comparison ~ N(comparison, Σ_comparison)           [1 sample from comparison, num_samples times]

            3. Compute three pairwise Mahalanobis distances:
               - d^2(z_ref, z_refprime) = (z_ref - z_refprime).T @ Σ_avg^{-1} @ (z_ref - z_refprime)  [ref vs reference_prime]
               - d^2(z_ref, z_comparison) = (z_ref - z_comparison).T @ Σ_avg^{-1} @ (z_ref - z_comparison)  [ref vs comparison]
               - d^2(z_refprime, z_comparison) = (z_refprime - z_comparison).T @ Σ_avg^{-1} @ (z_refprime - z_comparison)  [reference_prime vs comparison]

            4. Apply oddity decision rule:
               - delta = min(d^2(z_ref,z_comparison), d^2(z_refprime,z_comparison)) - d^2(z_ref,z_refprime)
               - delta > 0 means comparison is farther from BOTH ref and reference_prime -> correct identification

            5. Apply logistic smoothing:
               - P(correct) \approx mean(logistic.cdf(delta / bandwidth))

            6. Bernoulli log-likelihood:
               - LL = Σ [y * log(p) + (1-y) * log(1-p)]

            Performance:
            - Memory: O(num_samples * input_dim) per trial
            - Vectorized across trials using jax.vmap for GPU acceleration
        - Can be JIT-compiled for additional speed (future optimization)

            Examples
            --------
            >>> import jax.numpy as jnp
            >>> import jax.random as jr
            >>> from psyphy.model import WPPM, Prior
            >>> from psyphy.model.task import OddityTask
            >>> from psyphy.model.noise import GaussianNoise
            >>> from psyphy.data.dataset import ResponseData
            >>>
            >>> # Setup
            >>> model = WPPM(
            ...     input_dim=2,
            ...     prior=Prior(input_dim=2, basis_degree=3),
            ...     task=OddityTask(),
            ...     noise=GaussianNoise(sigma=0.03),
            ... )
            >>> params = model.init_params(jr.PRNGKey(0))
            >>>
            >>> # Create trial data
            >>> data = ResponseData()
            >>> data.add_trial(
            ...     ref=jnp.array([0.0, 0.0]), comparison=jnp.array([0.3, 0.2]), resp=1
            ... )
            >>>
            >>> loglik = model.task.loglik(
            ...     params,
            ...     data,
            ...     model,
            ...     model.noise,
            ...     num_samples=5000,
            ...     bandwidth=1e-3,
            ...     key=jr.PRNGKey(42),
            ... )
            >>> print(f"MC (N=5000): {loglik:.4f}")


        """
        # Task is the single source of truth for MC controls.
        num_samples = int(self.config.num_samples)
        bandwidth = float(self.config.bandwidth)

        # Only PRNG key is accepted dynamically.
        key = kwargs.pop("key", None)
        if "num_samples" in kwargs or "bandwidth" in kwargs:
            raise TypeError(
                "OddityTask.loglik does not accept 'num_samples'/'bandwidth' overrides. "
                "Configure them via OddityTaskConfig when constructing the task."
            )
        if kwargs:
            unexpected = ", ".join(sorted(kwargs.keys()))
            raise TypeError(
                f"Unexpected keyword arguments for OddityTask.loglik: {unexpected}"
            )

        if num_samples <= 0:
            raise ValueError(f"num_samples must be > 0, got {num_samples}")
        if bandwidth <= 0:
            raise ValueError(f"bandwidth must be > 0, got {bandwidth}")

        if key is None:
            key = jr.PRNGKey(int(self.config.default_key_seed))

        # Unpack trial data.
        #
        # Compute-efficient contract: `data` should be a batch of arrays.
        # We accept either a TrialData dataclass or any object with
        # `.refs/.comparisons/.responses` array attributes.
        refs = jnp.asarray(data.refs)
        comparisons = jnp.asarray(data.comparisons)
        responses = jnp.asarray(data.responses)
        n_trials = int(refs.shape[0])

        # Split keys for each trial (ensures independent sampling)
        trial_keys = jr.split(key, n_trials)

        # Vectorized computation of P(correct) for all trials
        # This processes all trials in parallel using jax.vmap
        # Note: probabilities are already clipped in _simulate_trial_mc()
        probs = self._simulate_trials_mc_vectorized(
            params=params,
            refs=refs,
            comparisons=comparisons,
            model=model,
            noise=noise,
            num_samples=num_samples,
            bandwidth=bandwidth,
            trial_keys=trial_keys,
        )

        # Bernoulli log-likelihood: LL = Σ [y log(p) + (1-y) log(1-p)]
        # Probabilities are already clipped to [eps, 1-eps] so log is safe
        log_likelihoods = jnp.where(
            responses == 1,
            jnp.log(probs),  # Correct response
            jnp.log(1.0 - probs),  # Incorrect response
        )

        return jnp.sum(log_likelihoods)

    def _simulate_trials_mc_vectorized(
        self,
        params: Any,
        refs: jnp.ndarray,
        comparisons: jnp.ndarray,
        model: Any,
        noise: Any,
        num_samples: int,
        bandwidth: float,
        trial_keys: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Vectorized Monte Carlo simulation across all trials.

        This method processes multiple trials in parallel using JAX's vmap,
        which is much faster than a Python loop (especially on GPU/TPU).

        Parameters
        ----------
        params : Any
            Model parameters as expected by ``model._compute_sqrt``.
        refs : jnp.ndarray, shape (n_trials, input_dim)
            Reference stimuli for all trials
        comparisons : jnp.ndarray, shape (n_trials, input_dim)
            Probe stimuli for all trials
        model : WPPM
            Model instance
        noise : NoiseModel
            Observer noise model
        num_samples : int
            Number of Monte Carlo samples per trial
        bandwidth : float
            Logistic smoothing bandwidth
        trial_keys : jnp.ndarray, shape (n_trials, 2)
            Random keys for each trial

        Returns
        -------
        probs : jnp.ndarray, shape (n_trials,)
            Estimated P(correct) for each trial
        """

        # Create a vectorized version of the single-trial simulation
        # vmap automatically batches over the first axis of each input
        def simulate_single(ref, comparison, key):
            return self._simulate_trial_mc(
                params, ref, comparison, model, noise, num_samples, bandwidth, key
            )

        # Apply to all trials in parallel
        probs = jax.vmap(simulate_single)(refs, comparisons, trial_keys)

        return probs

    def _simulate_trial_mc(
        self,
        params: Any,
        ref: jnp.ndarray,
        comparison: jnp.ndarray,
        model: Any,
        noise: Any,
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
            Model instance providing covariance structure
        noise : NoiseModel
            Observer noise (currently unused, noise in covariance?)
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
           - d^2(z_ref, z_refprime) = (z_ref - z_refprime).T @ Σ_avg^{-1} @ (z_ref - z_refprime)  [ref vs reference_prime]
           - d^2(z_ref, z_comparison) = (z_ref - z_comparison).T @ Σ_avg^{-1} @ (z_ref - z_comparison)  [ref vs comparison]
           - d^2(z_refprime, z_comparison) = (z_refprime - z_comparison).T @ Σ_avg^{-1} @ (z_refprime - z_comparison)  [reference_prime vs comparison]

        4. Decision rule (correct response):
           - The comparison (z_comparison) is the odd one if it's farther from BOTH ref and reference_prime
             than ref and reference_prime are from each other
           - delta = min(d^2(z_ref,z_comparison), d^2(z_refprime,z_comparison)) - d^2(z_ref,z_refprime)
           - delta > 0 indicates correct identification of comparison as odd

        5. Smooth decision with logistic CDF:
           - P(correct | sample) \approx sigmoid(delta / bandwidth)
           - Approximates noisy threshold decision

        6. Monte Carlo average:
           - P(correct) \approx mean over num_samples

        """
        # Get input dimension and require Wishart mode.
        # OddityTask is intentionally MC-only and currently only supports the
        # WPPM/Wishart covariance parameterization.
        input_dim = ref.shape[0]
        if model.basis_degree is not None:
            raise ValueError(
                "(Expected a basis degree, got None. model.basis_degree must not be None)."
            )

        # ========================================================================
        # STEP 1: Compute covariance structures at ref and comparison locations
        # ========================================================================

        # Wishart mode: Spatially-varying covariance
        # Compute U matrices that define covariances at each location
        # These U matrices define the two distributions:

        # U_ref defines DISTRIBUTION 1 covariance: Σ_ref = U_ref @ U_ref.T + diag_term * I
        U_ref = model._compute_sqrt(params, ref)  # (input_dim, embedding_dim)

        # U_comparison defines DISTRIBUTION 2 covariance: Σ_comparison = U_comparison @ U_comparison.T + diag_term * I
        U_comparison = model._compute_sqrt(
            params, comparison
        )  # (input_dim, embedding_dim)

        # Diagonal noise term (small regularization, same for both distributions)
        diag_term = model.diag_term
        sqrt_diag = jnp.sqrt(diag_term)

        # ========================================================================
        # STEP 2: Sample internal representations (3 samples from 2 distributions)
        # ========================================================================
        # Split random key: 3 for embedding samples + 3 for diagonal noise
        keys = jr.split(key, 6)

        # manual sampling using reparameterization trick
        # Covariance structure: Σ = U @ U.T + diag_term * I
        # Reparameterization: z = n_embed @ U.T + mean + sqrt(diag_term) * n_diag

        embed_dim = U_ref.shape[1]  # type: ignore

        # Samples from standard normal  (will be transformed to our target distributions 1 and 2)
        n_ref_embed = noise.sample_standard(keys[0], (num_samples, embed_dim))
        n_refprime_embed = noise.sample_standard(keys[1], (num_samples, embed_dim))
        n_comparison_embed = noise.sample_standard(keys[2], (num_samples, embed_dim))

        # Sample diagonal noise (independent across dimensions)
        n_ref_diag = noise.sample_standard(keys[3], (num_samples, input_dim))
        n_refprime_diag = noise.sample_standard(keys[4], (num_samples, input_dim))
        n_comparison_diag = noise.sample_standard(keys[5], (num_samples, input_dim))

        # =================================================================
        # SAMPLING: Transform standard normals to samples from our 2 distributions
        # =================================================================

        # SAMPLE 1 & 2: From DISTRIBUTION 1 (Reference), z_ref ~ N(ref, Σ_ref)
        # Both z_ref and z_refprime sampled from N(ref, Σ_ref)
        # where Σ_ref = U_ref @ U_ref.T + diag_term * I
        z_ref = n_ref_embed @ U_ref.T + ref[None, :] + sqrt_diag * n_ref_diag  # type: ignore
        #       ^^^^^^        ^^^^^       ^^^^^^^
        #       |                |          |
        #       |                |          +--- MEAN: ref (same for z_ref and z_refprime)
        #       |                +--- COVARIANCE: Uses U_ref (defines Σ_ref)
        #       +--- Independent noise (different from z_refprime, but same distribution)

        #  z_refprime ~ N(ref, Σ_ref)
        z_refprime = (
            n_refprime_embed @ U_ref.T + ref[None, :] + sqrt_diag * n_refprime_diag
        )  # type: ignore
        #              ^^^^^                   ^^^^      ^^^^^^^^^^
        #              |                     |          |
        #              |                     |          +--- MEAN: ref (SAME as z_ref!)
        #              |                     +--- COVARIANCE: Uses U_ref (SAME as z_ref!)
        #              +--- Independent noise (different from z_ref)

        # SAMPLE 3: From DISTRIBUTION 2 (Probe), z_comparison ~ N(comparison, Σ_comparison)
        # z_comparison sampled from N(comparison, Σ_comparison)
        # where Σ_comparison = U_comparison @ U_comparison.T + diag_term * I
        z_comparison = (
            n_comparison_embed @ U_comparison.T
            + comparison[None, :]
            + sqrt_diag * n_comparison_diag
        )  # type: ignore
        #         ^^^^^^^^        ^^^^^^        ^^^^^^^^^^
        #         |               |             |
        #         |               |             +--- MEAN: comparison (DIFFERENT from ref!)
        #         |               +--- COVARIANCE: Uses U_comparison (DIFFERENT from U_ref!)
        #         +--- independent noise (different distribution from z_ref and z_refprime)

        # ========================================================================
        # STEP 3: Compute average covariance for Mahalanobis distance
        # ========================================================================
        # we need a single covariance matrix for computing Mahalanobis distances
        # Weight by frequency: we sampled 2 times from N(ref, Σ_ref) and 1 time from N(comparison, Σ_comparison)
        # -> so we use weights (2/3) for reference distribution and (1/3) for comparison distribution

        # For Wishart mode, explicitly construct full covariances from U matrices
        # Σ_ref = U_ref @ U_ref.T + diag_term * I  (covariance of DISTRIBUTION 1)
        Sigma_ref_full = U_ref @ U_ref.T + diag_term * jnp.eye(input_dim)  # type: ignore

        # Σ_comparison = U_comparison @ U_comparison.T + diag_term * I  (covariance of DISTRIBUTION 2)
        Sigma_comparison_full = U_comparison @ U_comparison.T + diag_term * jnp.eye(
            input_dim
        )  # type: ignore

        # Weighted average: (2/3) * Σ_ref + (1/3) * Σ_comparison
        Sigma_avg = (2.0 / 3.0) * Sigma_ref_full + (1.0 / 3.0) * Sigma_comparison_full
        #           ^^^^^^^                         ^^^^^^^
        #           2 samples from ref               1 sample from comparison

        # ========================================================================
        # STEP 4: Compute three pairwise Mahalanobis distances
        # ========================================================================
        # difference vectors for all sample pairs, all of shape (num_samples, input_dim)
        diff_ref_refprime = z_ref - z_refprime  # distance ref to reference_prime
        diff_ref_comparison = z_ref - z_comparison  # distance ref to comparison
        diff_refprime_comparison = (
            z_refprime - z_comparison
        )  # distance reference_prime to comparison

        # Mahalanobis distance formula: d^2(x) = x^T @ Σ^{-1} @ x, where x is the difference vector, e.g., (z_ref - z_refprime)
        # We compute this efficiently without explicit matrix inversion.

        # Mathematical trick:
        # 1. Let y = Σ^{-1} @ x
        # 2. Then Σ @ y = x  (linear system)
        # 3. We find y using jnp.linalg.solve(Σ, x), which is O(D^3) but numerically stable.
        # 4. then, d^2 = x^T @ y = dot(x, y)

        # Stack differences for batch processing: (3, num_samples, input_dim)
        diffs_stacked = jnp.stack(
            [diff_ref_refprime, diff_ref_comparison, diff_refprime_comparison], axis=0
        )

        # Vectorized Solve via vmap:
        # We apply solve(Sigma_avg, d.T) to each of the 3 difference sets.
        # - Input d: (num_samples, input_dim)
        # - d.T: (input_dim, num_samples) -> acts as a batch of column vectors
        # - solve(Sigma, d.T): Solves Σ y_i = x_i for all i simultaneously.
        # - Result 'solved': (3, input_dim, num_samples) containing the vectors Σ^{-1} x
        solved = jax.vmap(lambda d: jnp.linalg.solve(Sigma_avg, d.T))(diffs_stacked)

        # Compute Dot Products: x^T @ y
        # We perform element-wise multiplication and sum over dimensions.
        # - diff_ref_refprime: x (num_samples, input_dim)
        # - solved[0].T: y (num_samples, input_dim)
        # - sum(x * y, axis=1): equivalent to dot product for each sample
        d_sq_ref_reference_prime = jnp.sum(
            diff_ref_refprime * solved[0].T, axis=1
        )  # (num_samples,)
        d_sq_ref_comparison = jnp.sum(
            diff_ref_comparison * solved[1].T, axis=1
        )  # (num_samples,)
        d_sq_refprime_comparison = jnp.sum(
            diff_refprime_comparison * solved[2].T, axis=1
        )  # (num_samples,)

        # ========================================================================
        # STEP 5: apply oddity decision rule
        # ========================================================================
        # Correct response: comparison (z_comparison) is farther from BOTH ref and reference_prime than they are from each other
        # delta > 0 means: min[d(z_ref,z_comparison), d(z_refprime,z_comparison)] > d(z_ref,z_refprime)
        # -> z_comparison is the outlier

        #
        delta = (
            jnp.minimum(d_sq_ref_comparison, d_sq_refprime_comparison)
            - d_sq_ref_reference_prime
        )  # (num_samples,)

        # ========================================================================
        # STEP 6: Smooth decision with logistic CDF
        # ========================================================================
        # Logistic CDF approximates a noisy threshold decision
        # bandwidth controls decision noise:
        #   - small bandwidth (~1e-3): sharp threshold (nearly deterministic)
        #   - large bandwidth (~5e-2): smooth, gradual transition
        # P(correct | delta) \approx sigmoid(delta / bandwidth)

        prob_correct_per_sample = jax.scipy.stats.logistic.cdf(delta / bandwidth)

        # ========================================================================
        # STEP 7: Monte Carlo average
        # ========================================================================
        # by law of large numbers: mean(samples) -> E[P(correct)]
        prob = jnp.mean(prob_correct_per_sample)

        # Clip probability to avoid numerical issues with log(0) or log(1)
        # - Use eps=1e-6  for safety (or smaller epsilon and increase precison from float32 to float64)
        # - Clipping here (before return) ensures gradients stay finite
        # - Without this, prob=1.0 -> log(1.0)=0.0 -> grad through clip at boundary -> NaN
        eps = 1e-6
        return jnp.clip(prob, eps, 1.0 - eps)
