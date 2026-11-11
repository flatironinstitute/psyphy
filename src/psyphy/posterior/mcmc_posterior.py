"""
mcmc_posterior.py
-----------------

MCMCPosterior: backend-agnostic container for MCMC samples.

Satisfies the ParameterPosterior protocol -->  plugs directly into
WPPMPredictivePosterior and any future acquisition functions.

Design
------
MCMCPosterior stores raw chain draws as a dict of JAX arrays with shape
(n_chains, n_draws, *param_shape). It is produced by any MCMC inference
engine (NUTSSampler, future MALASampler, etc.) and consumed identically
to MAPPosterior by all downstream code.
"""

from __future__ import annotations

import sys

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np


class MCMCPosterior:
    """
    Backend-agnostic container for MCMC posterior samples.

    Satisfies the ParameterPosterior protocol:
    - ``params``: posterior mean *parameter* over all chains and draws (for
      WPPM, do not turn this into a covariance field -- see ``params``)
    - ``model``: the fitted WPPM instance
    - ``sample(n, key)``: draw n parameter vectors from the chain pool

    Parameters
    ----------
    samples : dict
        Parameter samples. Each value has shape ``(n_chains, n_draws, *param_shape)``.
        For WPPM this is ``{"W": ndarray of shape (n_chains, n_draws, d0, d1, d2, d3)}``.
    model : WPPM
        The model instance used during inference.
    sampler_stats : dict | None, optional
        Per-chain diagnostics returned by the sampler, each of shape
        ``(n_chains, n_draws)``. NUTSSampler supplies ArviZ's ``sample_stats``
        names: ``acceptance_rate``, ``diverging``, ``energy``, ``n_steps``.

    Notes
    -----
    Implements the ParameterPosterior protocol (i.e., structural, not inherited).
    """

    def __init__(
        self,
        samples: dict,
        model,
        *,
        sampler_stats: dict | None = None,
    ):
        # samples: dict of JAX arrays, each shaped (n_chains, n_draws, *param_shape)
        # For WPPM: {"W": (n_chains, n_draws, degree+1, degree+1, input_dim, embedding_dim)}
        # e.g. 4 chains × 1000 draws, degree=1, input_dim=2, extra_dims=1:
        #      {"W": (4, 1000, 2, 2, 2, 3)}
        # This shape flows unchanged from NUTSSampler -> here -> ArviZ.
        self._samples = samples
        self._model = model
        # sampler_stats: optional dict of per-chain diagnostics from the sampler.
        # Keys from NUTSSampler: acceptance_rate, diverging, energy, n_steps --
        # each (n_chains, n_draws), named per ArviZ's sample_stats convention so
        # to_arviz() feeds az.plot_energy and divergence reporting directly.
        self.sampler_stats = sampler_stats or {}

    # ------------------------------------------------------------------
    # ParameterPosterior protocol
    # ------------------------------------------------------------------

    @property
    def params(self) -> dict:
        """Posterior mean of the parameters over all chains and draws.

        Returns
        -------
        dict
            Parameter PyTree with the same structure as a single draw.
            For WPPM: ``{"W": ndarray of shape (*W_shape)}``.
            e.g. {"W": (2, 2, 2, 3)}, mean over axes 0 (chains) and 1 (draws).

        Warnings
        --------
        For WPPM, do **not** build a covariance field from this, e.g.
        ``WPPMCovarianceField(model, posterior.params)``. That gives Σ(E[W]),
        which is not the posterior mean of Σ and is systematically too small:

        - Σ = U Uᵀ + λI is quadratic in W, so E[Σ] = Σ(E[W]) + Cov[U] and
          averaging W first drops the (positive semidefinite) Cov[U] term.
        - W is only identified up to an orthogonal rotation of its embedding
          axis (W -> W R leaves Σ unchanged), so draws at different
          orientations partly cancel in E[W], pulling Σ(E[W]) toward λI.

        For the posterior mean covariance field use
        ``WPPMCovarianceFieldSamples(posterior).mean(x)``; for
        individual covariance draws use ``sample(n)``. E[W] remains a valid parameter set, e.g. as
        ``init_params`` for another fit.
        """
        # _samples["W"]: (n_chains, n_draws, *param_shape)
        # mean over (0, 1)  ->  (*param_shape)
        # Kept as the plain parameter mean: the ParameterPosterior
        # protocol requires params to be a parameter PyTree (NUTSSampler and
        # MAPOptimizer accept it as init_params), and E[Σ] cannot be expressed
        # as any single W. The fix for covariance summaries therefore lives in
        # WPPMCovarianceFieldSamples, not here.
        return jax.tree.map(lambda x: jnp.mean(x, axis=(0, 1)), self._samples)

    @property
    def model(self):
        """The WPPM instance used during inference."""
        return self._model

    def sample(self, n: int, key) -> dict:
        """Draw n parameter vectors from the pooled chain samples.

        Pools all chains and draws into a flat collection of
        ``n_chains * n_draws`` samples, then randomly subsamples ``n``
        of them (without replacement if n <= total, with replacement otherwise).

        Parameters
        ----------
        n : int
            Number of samples to return.
        key : jax.Array
            PRNG key for the subsample.

        Returns
        -------
        dict
            Parameter PyTree with leading dimension ``n``.
            For WPPM: ``{"W": ndarray of shape (n, *W_shape)}``.

        Notes
        -----
        For every stored sample, read :attr:`samples` instead. It keeps the chain
        axis and takes no key.
        """

        def _subsample(x):
            # x     : (n_chains, n_draws, *param_shape)
            #           e.g. (4, 1000, 2, 2, 2, 3)
            # pool  : (n_chains * n_draws, *param_shape)
            #           e.g. (4000, 2, 2, 2, 3)  : all draws from all chains flattened
            # output: (n, *param_shape)
            #           e.g. (100, 2, 2, 2, 3)   : n draws subsampled from pool
            n_chains, n_draws = x.shape[:2]
            event_shape = x.shape[2:]
            pool = x.reshape(n_chains * n_draws, *event_shape)
            total = n_chains * n_draws
            replace = n > total
            indices = jr.choice(key, total, shape=(n,), replace=replace)
            return pool[indices]

        return jax.tree.map(_subsample, self._samples)

    # Convenience properties
    # -----------------------------------------------

    @property
    def samples(self) -> dict:
        """All draws, chain structure preserved.

        Returns
        -------
        dict
            ``{name: array of shape (n_chains, n_draws, *param_shape)}``.
            For WPPM: ``{"W": (n_chains, n_draws, d0, d1, d2, d3)}``.

        Notes
        -----
        This is the raw posterior, not a summary: use it to persist a fit, to
        pass the draws to another library, or for any computation that needs
        the per-chain layout (diagnostics do). For a random subset of draws use
        :meth:`sample`; for the posterior mean parameter use :attr:`params`.

        The returned dict is a fresh mapping, so adding or removing keys does
        not affect the posterior, but the arrays themselves are shared (JAX
        arrays are immutable).
        """
        return dict(self._samples)

    @property
    def n_chains(self) -> int:
        """Number of chains."""
        first = next(iter(self._samples.values()))
        return int(first.shape[0])

    @property
    def n_draws(self) -> int:
        """Number of draws per chain."""
        first = next(iter(self._samples.values()))
        return int(first.shape[1])

    # -------------------------------------------------------------
    # ArviZ conversion

    def to_arviz(self):
        """Convert MCMC samples to an ArviZ InferenceData object.

        Requires `arviz` to be installed::

            pip install 'psyphy[diagnostics]'

        Returns
        -------
        xarray.DataTree
            ArviZ >= 1.0 returns xarray's DataTree; ``arviz.InferenceData`` was
            the 0.x type and is gone (it is still importable on 1.x but emits a
            MigrationWarning). Measured: on arviz 0.23.4 every access pattern
            documented below raises. The extra pins ``arviz>=1.0``.
            Contains a ``posterior`` group with variables shaped
            ``(n_chains, n_draws, *param_shape)``.
            For WPPM: variable ``W`` of shape
            ``(n_chains, n_draws, d0, d1, d2, d3)``.

        Notes
        -----
        ArviZ handles arbitrary-dimensional variables natively, no
        flattening required. R-hat and ESS are computed per scalar element.

        If ``sampler_stats`` were provided at construction (e.g.
        ``acceptance_rate``), they are included in the ``sample_stats``
        group.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import arviz as az
        >>> from psyphy.model import WPPM, OddityTask, Prior
        >>> from psyphy.posterior import MCMCPosterior
        >>> model = WPPM(Prior(input_dim=2, basis_degree=2), OddityTask())
        >>> # stand-in for NUTSSampler().fit(model, data): 2 chains of 50 prior draws
        >>> keys = jr.split(jr.PRNGKey(1), 100)
        >>> W = jnp.stack(
        ...     [model.init_params(k)["W"] for k in keys]
        ... )  # (100, 3, 3, 2, 3)
        >>> nuts_posterior = MCMCPosterior({"W": W.reshape(2, 50, *W.shape[1:])}, model)
        >>> idata = nuts_posterior.to_arviz()
        >>> print(az.summary(idata))  # doctest: +SKIP
        """
        try:
            import arviz as az
        except ImportError as e:
            raise ImportError(
                "ArviZ is required for MCMCPosterior.to_arviz(). "
                "Install it with: pip install 'psyphy[diagnostics]'\n"
                f"Note: to_arviz() needs arviz >= 1.0, and every arviz 1.x "
                f"requires Python >= 3.12 (this is {sys.version_info.major}."
                f"{sys.version_info.minor}). On an older interpreter the "
                "'diagnostics' extra resolves to nothing and this error appears "
                "even after installing it. The chains are still available via "
                "`.sample()` / `._samples` without ArviZ."
            ) from e

        # -- ArviZ 1.0 API boundary --------------------------------
        # az.from_dict(data) where data is a nested dict:
        #   {
        #     "posterior":    {var_name: np.ndarray of shape (n_chains, n_draws, *param_shape)},
        #     "sample_stats": {stat_name: np.ndarray of shape (n_chains, n_draws)},  # optional
        #   }
        #
        # For WPPM with 4 chains, 1000 draws, W shape (2,2,2,3):
        #   {"posterior": {"W": np.ndarray (4, 1000, 2, 2, 2, 3)}}
        #
        # Returns: xarray.DataTree  (ArviZ 1.0; was InferenceData in ArviZ 0.x)
        #   idata["posterior"]["W"].values  -> np.ndarray (4, 1000, 2, 2, 2, 3)
        #   list(idata.children)            -> ["posterior", "sample_stats"]
        #   az.rhat(idata).ds.data_vars     -> {"W"}  (R-hat per scalar element)
        #
        # Note: must convert JAX arrays to numpy; ArviZ uses numpy/xarray internally.
        # ----------------------------------------------------------------
        data: dict = {"posterior": {k: np.asarray(v) for k, v in self._samples.items()}}

        if self.sampler_stats:
            # sample_stats values shape: (n_chains, n_draws)
            # e.g. {"acceptance_rate": (4, 1000), "diverging": (4, 1000), ...}
            data["sample_stats"] = {
                k: np.asarray(v) for k, v in self.sampler_stats.items()
            }

        return az.from_dict(data)
