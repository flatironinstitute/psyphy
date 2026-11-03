"""
predictive_posterior.py
----------------------

Predictive posterior distributions p(f(X*) | data) at test stimuli.

This module defines posteriors over **predictions** (not parameters),
used by acquisition functions for Bayesian optimization.

Design
------
PredictivePosterior wraps a ParameterPosterior and computes predictions via:
    E[f(X*) | data] \approx (1/N) Σ_i f(X*; θ_i) where θ_i ~ p(θ | data)

This separates concerns:
- ParameterPosterior: represents uncertainty over θ
- PredictivePosterior: represents uncertainty over f(X*) (decision-making)
- effectively decoupling how we FIT the model from how we USE the fitted model
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import jax.random as jr

if TYPE_CHECKING:
    from psyphy.posterior.parameter_posterior import ParameterPosterior

from ..model.likelihood import BernoulliTaskLikelihood


@dataclass(frozen=True, slots=True)
class ThresholdConfig:
    """Configuration for :class:`WPPMPredictivePosterior`'s ``threshold_pred`` mode.

    Controls the oddity-task inversion that recovers a threshold covariance
    Σ_thres(x), the contour where the probability correct reaches
    ``target_prob_correct``, from a
    fitted noise field. There is no closed form for this inversion (see
    :class:`~psyphy.model.likelihood.OddityTask`), so it is done numerically:
    probe ``n_theta`` directions around each reference point, brute-force
    search ``n_length`` candidate distances per direction for the one closest
    to ``target_prob_correct``, then fit an ellipsoid to the resulting boundary
    points.

    Attributes
    ----------
    n_theta : int
        Number of directions probed around each reference point. Must be at
        least ``d * (d + 1) // 2`` (the number of free parameters of a
        symmetric ``d x d`` matrix) or the ellipsoid fit is underdetermined.
    n_length : int
        Number of candidate distances probed per direction. Brute-force
        search rather than bisection: P(correct) is only monotone in
        distance up to Monte Carlo noise, so a root-finder can walk off a
        noisy plateau where a dense-grid argmin cannot.
    bounds : tuple[float, float]
        ``(min, max)`` candidate distance, in the model's stimulus units.
    target_prob_correct : float
        The probability correct that defines the threshold contour. 2/3 is the
        oddity task's standard criterion.
    chunk : int
        Batch size for the Monte Carlo sweep, bounding peak memory;
        ``n_theta * n_length`` candidate trials are evaluated per
        (posterior sample, reference point) pair.

    Notes
    -----
    Cost per (posterior sample, reference point) pair is
    ``n_theta * n_length`` calls to the task's MC-simulated ``predict``, each
    itself an average over ``OddityTaskConfig.num_samples`` draws. Total cost
    for a call to :class:`WPPMPredictivePosterior` scales as
    ``n_samples * n_test * n_theta * n_length * num_samples``; a warning is
    emitted when ``n_samples * n_test`` alone exceeds 200.
    """

    n_theta: int = 16
    n_length: int = 200
    bounds: tuple[float, float] = (5e-4, 0.3)
    target_prob_correct: float = 2.0 / 3.0
    chunk: int = 2000

    def __post_init__(self) -> None:
        if int(self.n_theta) <= 0:
            raise ValueError(f"n_theta must be > 0, got {self.n_theta}")
        if int(self.n_length) <= 0:
            raise ValueError(f"n_length must be > 0, got {self.n_length}")
        if not (0.0 < float(self.target_prob_correct) < 1.0):
            raise ValueError(
                f"target_prob_correct must be in (0, 1), got {self.target_prob_correct}"
            )
        if int(self.chunk) <= 0:
            raise ValueError(f"chunk must be > 0, got {self.chunk}")
        lo, hi = self.bounds
        if not (0.0 < float(lo) < float(hi)):
            raise ValueError(f"bounds must satisfy 0 < min < max, got {self.bounds}")


@runtime_checkable
class PredictivePosterior(Protocol):
    """
    Protocol for predictive distributions p(f(X*) | data) at test stimuli.

    Returned by Model.posterior(X) for use in acquisition functions.
    """

    @property
    def mean(self) -> jnp.ndarray:
        """
        Posterior predictive mean E[f(X*) | data].

        Returns
        -------
        jnp.ndarray
            Shape (n_test,) for scalar outputs
            Shape (n_test, output_dim) for vector outputs (future)

        Notes
        -----
        Computed via Monte Carlo integration over parameter posterior.

        :class:`WPPMPredictivePosterior` in ``threshold_pred=True`` mode is a
        documented exception to the shapes above: it returns matrix-valued
        moments, shape (n_test, input_dim, input_dim) -- a threshold
        covariance per test point, not a scalar or vector. See its own
        ``mean``/``variance`` docstrings.
        """
        ...

    @property
    def variance(self) -> jnp.ndarray:
        """
        Posterior predictive marginal variances Var[f(X*) | data].

        Returns
        -------
        jnp.ndarray
            Shape (n_test,) for scalar outputs
            Shape (n_test, output_dim) for vector outputs (future)

        Notes
        -----
        Captures both aleatoric (model) and epistemic (parameter) uncertainty.

        Same WPPM ``threshold_pred=True`` exception as :attr:`mean` applies.
        """
        ...

    def rsample(self, sample_shape: tuple = (), *, key: jax.Array) -> jnp.ndarray:
        """
        Reparameterized samples from p(f(X*) | data).

        Parameters
        ----------
        sample_shape : tuple, default=()
            Shape of sample batch
        key : jax.Array
            PRNG key

        Returns
        -------
        jnp.ndarray
            Shape (*sample_shape, n_test) for scalar outputs
            Shape (*sample_shape, n_test, output_dim) for vector outputs

        Notes
        -----
        Enables gradient-based acquisition optimization via reparameterization trick.
        """
        ...


class WPPMPredictivePosterior:
    """
    Predictive posterior for WPPM models.

    Computes p(f(X*) | data) via Monte Carlo integration over
    parameter posterior p(θ | data).

    Parameters
    ----------
    param_posterior : ParameterPosterior
        Posterior over model parameters
    X : jnp.ndarray
        Test stimuli. Shape depends on ``threshold_pred``:

        - ``threshold_pred=False``: ``(n_test, k_stimuli, input_dim)``:
          pre-paired (reference, comparison) stimuli, scored directly via
          ``model.predict_prob``.
        - ``threshold_pred=True``: ``(n_test, input_dim)``: bare reference
          points. Comparison stimuli are *not* supplied; the oddity-task
          inversion generates its own by sweeping directions and distances
          around each reference (see :class:`ThresholdConfig`). Passing the
          paired-stimulus shape here raises ``ValueError``.
    n_samples : int, default=100
        Number of posterior samples for MC integration
    threshold_pred: bool, default = False
        Whether to compute threshold covariances (Σ_thres, the
        contour where the probability correct reaches
        ``target_prob_correct``) instead of predicted
        probabilities. Changes both ``X``'s expected shape (above) and
        ``mean``/``variance``'s returned shape (see their docstrings).
    threshold_config : ThresholdConfig, optional
        Settings for the threshold inversion. Only used when
        ``threshold_pred=True``; defaults to ``ThresholdConfig()``.
    key : jax.Array, keyword-only
        PRNG key, split once here into a key for drawing parameters and a key
        for the threshold sweep's Monte Carlo. ``mean`` and ``variance`` are
        properties, so this is where their randomness enters.

    Attributes
    ----------
    param_posterior : ParameterPosterior
        Wrapped parameter posterior
    X : jnp.ndarray
        Test stimuli
    n_samples : int
        MC sample count
    threshold_pred: bool
        Whether this instance predicts thresholds rather than probabilities
    threshold_config : ThresholdConfig or None
        Threshold-inversion settings (``threshold_pred=True`` only)

    Notes
    -----
    Uses lazy evaluation: moments computed on first access.
    """

    def __init__(
        self,
        param_posterior: ParameterPosterior,
        X: jnp.ndarray,
        n_samples: int = 100,
        threshold_pred: bool = False,
        threshold_config: ThresholdConfig | None = None,
        *,
        key: jax.Array,
    ):
        self.param_posterior = param_posterior
        self.X = X
        self.n_samples = n_samples
        self.threshold_pred = threshold_pred
        self.threshold_config = threshold_config

        # Split by purpose at construction: `mean` and `variance` are
        # properties, so they have nowhere to take a key, and both read these.
        self._draw_key, self._sweep_key = jr.split(key)

        # Lazy evaluation cache. The annotation keeps `mean`/`variance`
        # from reading as returning None.
        self._mean: jnp.ndarray | None = None
        self._variance: jnp.ndarray | None = None
        self._computed = False

    def _recover_threshold_covariances(
        self, param_samples, X: jnp.ndarray
    ) -> jnp.ndarray:
        """Recover threshold covariances Σ_thres(x) at each reference in X.

        For every (posterior parameter sample, reference point) pair, probes
        ``threshold_config.n_theta`` directions around the reference,
        brute-force searches ``n_length`` candidate distances per direction
        for the one closest to ``target_prob_correct`` (see :class:`ThresholdConfig`
        for why brute force rather than bisection), then fits a covariance
        to the resulting boundary points via linear least squares: a point
        at radius r in direction u on the ellipsoid {c + Lu : |u|=1}
        satisfies ``u^T Sigma^-1 u = 1/r^2``, which is *linear* in Sigma^-1's
        ``d*(d+1)/2`` free entries (d = input_dim) -- no optimizer needed.

        Parameters
        ----------
        param_samples : pytree
            Batched model parameters, leading axis = number of samples.
        X : jnp.ndarray, shape (n_test, input_dim)
            Bare reference points (not paired stimuli).

        Returns
        -------
        jnp.ndarray, shape (n_param_samples, n_test, input_dim, input_dim)
        """
        if X.ndim != 2:
            raise ValueError(
                "threshold_pred=True expects bare reference points, shape "
                f"(n_test, input_dim); got X.shape={X.shape}, which looks "
                "like paired (reference, comparison) stimuli -- that shape "
                "is for threshold_pred=False."
            )

        model = self.param_posterior.model
        if not isinstance(model.likelihood, BernoulliTaskLikelihood):
            raise NotImplementedError(
                "Threshold prediction currently only supports "
                "BernoulliTaskLikelihood (e.g. OddityTask), whose predict() "
                "returns a single p(correct)."
            )
        task = model.likelihood

        cfg = self.threshold_config or ThresholdConfig()
        n_test, d = X.shape
        n_param_samples = jax.tree_util.tree_leaves(param_samples)[0].shape[0]

        min_theta = d * (d + 1) // 2
        if cfg.n_theta < min_theta:
            raise ValueError(
                f"threshold_config.n_theta={cfg.n_theta} is too small for "
                f"input_dim={d}: a symmetric {d}x{d} covariance has "
                f"{min_theta} free parameters, so the ellipsoid fit needs "
                f"at least {min_theta} probed directions."
            )

        cost = n_param_samples * n_test
        if cost > 200:
            warnings.warn(
                f"threshold_pred cost is n_samples * n_test * n_theta * "
                f"n_length * OddityTaskConfig.num_samples MC draws; "
                f"n_samples * n_test = {cost} here. Consider lowering "
                "n_samples, the number of reference points, or "
                "threshold_config.n_length if this is slow.",
                stacklevel=2,
            )

        # Directions: evenly-spaced angles in 2D (matches the validated
        # prototype exactly); random unit vectors otherwise, deterministic
        # and independent of the MC sampling key below.
        if d == 2:
            theta = jnp.linspace(0.0, 2 * jnp.pi, cfg.n_theta, endpoint=False)
            dirs = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=1)
        else:
            raw = jr.normal(jr.PRNGKey(0), (cfg.n_theta, d))
            dirs = raw / jnp.linalg.norm(raw, axis=1, keepdims=True)

        lengths = jnp.linspace(cfg.bounds[0], cfg.bounds[1], cfg.n_length)
        tri_i, tri_j = jnp.triu_indices(d)
        # design[:, k] = u_i * u_j (doubled off-diagonal) for the k-th free
        # entry of the symmetric matrix -- generalizes the prototype's 2D
        # [dx^2, 2*dx*dy, dy^2] row to arbitrary d.
        off_diag_weight = jnp.where(tri_i == tri_j, 1.0, 2.0)
        design = dirs[:, tri_i] * dirs[:, tri_j] * off_diag_weight[None, :]

        chunk = cfg.chunk

        @jax.jit
        def _p_correct_chunk(params, stim_chunk, key_chunk):
            return jax.vmap(lambda s, k: task.predict(params, s, model, key=k)[0])(
                stim_chunk, key_chunk
            )

        def _sweep_one(params, ref, key):
            comps = ref[None, None, :] + dirs[:, None, :] * lengths[None, :, None]
            comps_flat = comps.reshape(-1, d)
            refs_flat = jnp.broadcast_to(ref, comps_flat.shape)
            stimuli = jnp.stack([refs_flat, comps_flat], axis=1)  # (n_pairs, 2, d)

            n_pairs = stimuli.shape[0]
            keys = jr.split(key, n_pairs)
            outs = [
                _p_correct_chunk(params, stimuli[i : i + chunk], keys[i : i + chunk])
                for i in range(0, n_pairs, chunk)
            ]
            prob_correct = jnp.concatenate(outs).reshape(cfg.n_theta, cfg.n_length)

            # Brute force rather than bisection: P(correct) is monotone in
            # distance only up to Monte Carlo noise; argmin over a dense
            # grid is robust to that in a way a root-finder is not.
            radii = lengths[
                jnp.argmin(jnp.abs(prob_correct - cfg.target_prob_correct), axis=1)
            ]

            y = 1.0 / radii**2
            coeffs = jnp.linalg.solve(design.T @ design, design.T @ y)
            Sigma_inv = (
                jnp.zeros((d, d))
                .at[tri_i, tri_j]
                .set(coeffs)
                .at[tri_j, tri_i]
                .set(coeffs)
            )
            return jnp.linalg.inv(Sigma_inv)

        # Nested Python loop over (posterior sample, reference point), not
        # vmap: vmapping either axis would multiply peak memory by exactly
        # the dimension `chunk` exists to bound. `_p_correct_chunk` is
        # compiled once (shapes are static across every iteration) and
        # reused, so the loop's overhead is dispatch only, not compilation.
        sweep_keys = jr.split(self._sweep_key, n_param_samples * n_test)
        results = []
        for s in range(n_param_samples):
            params_s = jax.tree_util.tree_map(lambda leaf, s=s: leaf[s], param_samples)
            row = [
                _sweep_one(params_s, X[t], sweep_keys[s * n_test + t])
                for t in range(n_test)
            ]
            results.append(jnp.stack(row))
        return jnp.stack(results)

    def _ensure_computed(self):
        """Compute moments via MC integration (lazy)."""
        if self._computed:
            return

        # Sample parameters from posterior
        param_samples = self.param_posterior.sample(self.n_samples, key=self._draw_key)

        model = self.param_posterior.model

        if self.threshold_pred:
            # cov_samples: (n_samples, n_test, input_dim, input_dim)
            cov_samples = self._recover_threshold_covariances(param_samples, self.X)
            self._mean = jnp.mean(cov_samples, axis=0)
            self._variance = jnp.var(cov_samples, axis=0)
            self._computed = True
            return

        # Vectorized prediction over parameter samples
        def predict_batch(params):
            """Predict distribution parameters for given params
            For OddityTask, this is p(correct) for all (ref, probe) pairs given params."""

            if not isinstance(model.likelihood, BernoulliTaskLikelihood):
                raise NotImplementedError(
                    "WPPMPredictivePosterior currently only supports "
                    "BernoulliTaskLikelihood. Gaussian support requires "
                    "updating to handle (mu, sigma) returns."
                )

            return jax.vmap(lambda x: model.predict_prob(params, x))(self.X)

        # predictions: shape (n_samples, n_test)
        predictions = jax.vmap(predict_batch)(param_samples)

        # Compute moments
        self._mean = jnp.mean(predictions, axis=0)
        self._variance = jnp.var(predictions, axis=0)
        self._computed = True

    @property
    def mean(self) -> jnp.ndarray:
        """E[f(X*) | data].

        Shape (n_test,): predicted probabilities, unless
        ``threshold_pred=True``, in which case shape is
        (n_test, input_dim, input_dim): the posterior-mean threshold
        covariance Σ_thres at each reference point, averaged arithmetically
        over posterior parameter samples (the PD cone is convex, so this
        stays PD; not a geometric/Bures mean).
        """
        self._ensure_computed()
        assert self._mean is not None  # set by _ensure_computed
        return self._mean

    @property
    def variance(self) -> jnp.ndarray:
        """Var[f(X*) | data].

        Shape (n_test,) unless ``threshold_pred=True``, in which case shape
        is (n_test, input_dim, input_dim): elementwise variance of the
        per-sample threshold covariances (not a variance *of* a matrix in
        any distributional sense; just ``jnp.var`` over the sample axis,
        entry by entry).
        """
        self._ensure_computed()
        assert self._variance is not None  # set by _ensure_computed
        return self._variance

    def rsample(self, sample_shape: tuple = (), *, key: jax.Array) -> jnp.ndarray:
        """
        Sample predictions from p(f(X*) | data).

        Parameters
        ----------
        sample_shape : tuple
            Batch shape
        key : jax.Array
            PRNG key

        Returns
        -------
        jnp.ndarray
            Shape (*sample_shape, n_test); or, when ``threshold_pred=True``,
            (*sample_shape, n_test, input_dim, input_dim): one recovered
            threshold covariance per draw. Cost scales linearly in the number
            of draws requested; see :class:`ThresholdConfig`.
        """
        n = int(jnp.prod(jnp.array(sample_shape))) if sample_shape else 1
        param_samples = self.param_posterior.sample(n, key=key)

        model = self.param_posterior.model

        if self.threshold_pred:
            # cov_samples: (n, n_test, input_dim, input_dim)
            cov_samples = self._recover_threshold_covariances(param_samples, self.X)
            if sample_shape:
                return cov_samples.reshape(*sample_shape, *cov_samples.shape[1:])
            return cov_samples

        def predict_one(params):
            """Predict for all test points with given params."""
            return jax.vmap(lambda x: model.predict_prob(params, x))(self.X)

        samples = jax.vmap(predict_one)(param_samples)

        if sample_shape:
            return samples.reshape(*sample_shape, -1)
        return samples
