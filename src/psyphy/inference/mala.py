"""
mala.py
-------

MALA (Metropolis-adjusted Langevin algorithm) inference engine backed by BlackJAX.

Each step moves the parameters along the gradient of the log posterior plus
Gaussian noise, then accepts or rejects the move. One gradient evaluation per
step, so a step costs far less than a NUTS step, and the chain needs more
steps to explore the posterior.

Shares its chain loop with :class:`NUTSSampler`, in ``psyphy.inference.mcmc``.

Requires
--------
blackjax >= 1.0  (install with: pip install 'psyphy[sampling]')
"""

from __future__ import annotations

from psyphy.inference.base import InferenceEngine
from psyphy.inference.mcmc import (
    _fit_chains,
    _import_blackjax,
    _make_pbar,
    _run_fixed_kernel,
)
from psyphy.posterior.mcmc_posterior import MCMCPosterior


class MALASampler(InferenceEngine):
    """
    MALA (Metropolis-adjusted Langevin algorithm) using BlackJAX.

    Draws samples from the full posterior p(θ | data). Returns an
    :class:`MCMCPosterior` with the same layout as :class:`NUTSSampler`, so
    ``covariance_to_arviz``, :class:`WPPMCovarianceField` and the
    plotting functions take either.

    MALA does not tune itself. The step size is fixed for the whole run,
    burn-in included, and there is no mass matrix.

    Parameters
    ----------
    num_warmup : int, default=500
        Burn-in steps per chain. Their samples are thrown away.
    num_samples : int, default=1000
        Number of posterior samples per chain (after burn-in).
    num_chains : int, default=4
        Number of independent chains, run together.
    step_size : float
        Size of each Langevin step. Each step moves ``step_size`` times the
        gradient and adds noise with standard deviation ``sqrt(2 * step_size)``.
        With ``prior_preconditioning=True`` both are in units of each
        parameter's prior standard deviation. Required, since a good value
        depends on the model and on how much data there is. Start at 1e-4 and
        adjust until the mean of ``sampler_stats["acceptance_rate"]`` is about
        0.6: lower the step size when it is below, raise it when it is near 1.
    logdensity_key_seed : int, default=0
        Seed for the fixed Monte Carlo key baked into the log density. See
        :class:`NUTSSampler`.
    prior_preconditioning : bool, default=True
        If True and the model exposes ``model.prior.param_scales()`` (WPPM
        does), sample z = W / prior_std instead of W. Every coordinate then
        has the same scale a priori. Results are returned in W coordinates
        either way.
    show_progress : bool, default=True
        Display a tqdm progress bar. Requires `tqdm`, from the ``progress``
        extra (``pip install 'psyphy[progress]'``). Skipped without a message
        if tqdm is not installed.

    Notes
    -----
    ``sampler_stats`` has 2 keys, each shaped (num_chains, num_samples):

    - ``acceptance_rate``: probability of accepting each proposed step
    - ``is_accepted``: whether the step was accepted

    MALA has no ``diverging``, ``energy`` or ``n_steps``, so ArviZ's energy
    plot and divergence marks are not available. R-hat and effective sample
    size work as for NUTS.

    Examples
    --------
    >>> from psyphy.inference import MALASampler
    >>> sampler = MALASampler(num_warmup=2000, num_samples=2000, step_size=1e-4)
    >>> mala_posterior = sampler.fit(
    ...     model, data, init_params=map_posterior.params
    ... )  # doctest: +SKIP
    >>> print(mala_posterior.n_chains, mala_posterior.n_draws)  # doctest: +SKIP
    4 2000
    """

    def __init__(
        self,
        num_warmup: int = 500,
        num_samples: int = 1000,
        num_chains: int = 4,
        *,
        step_size: float,
        logdensity_key_seed: int = 0,
        prior_preconditioning: bool = True,
        show_progress: bool = True,
    ):
        self.num_warmup = int(num_warmup)
        self.num_samples = int(num_samples)
        self.num_chains = int(num_chains)
        if float(step_size) <= 0:
            raise ValueError(f"step_size must be > 0, got {step_size}")
        self.step_size = float(step_size)
        self.logdensity_key_seed = int(logdensity_key_seed)
        self.prior_preconditioning = bool(prior_preconditioning)
        self.show_progress = bool(show_progress)

    def fit(
        self,
        model,
        data,
        init_params: dict | None = None,
        seed: int | None = None,
    ) -> MCMCPosterior:
        """
        Run MALA and return a posterior over model parameters.

        Parameters
        ----------
        model : WPPM
            Model instance providing log_posterior_from_data.
        data : TrialData
            Observed trials.
        init_params : dict | None, optional
            Initial parameter PyTree, shared by all chains. If None, one draw
            from the model prior with the seed. Starting from the MAP
            (``map_posterior.params``) shortens the burn-in needed.
        seed : int | None, optional
            Seed for chain initialization and sampling. None means 0.

        Returns
        -------
        MCMCPosterior
            Samples shaped (num_chains, num_samples, *param_shape), and
            ``sampler_stats`` with ``acceptance_rate`` and ``is_accepted``.
        """
        blackjax = _import_blackjax("MALASampler")

        def run_chains(logdensity_fn, init_position, chain_keys):
            mala = blackjax.mala(logdensity_fn, step_size=self.step_size)
            return _run_fixed_kernel(
                mala,
                init_position,
                chain_keys,
                num_warmup=self.num_warmup,
                num_samples=self.num_samples,
                stats_fn=_mala_stats,
                pbar=_make_pbar(
                    self.show_progress,
                    total=1,
                    desc=f"MALA ({self.num_chains} chains, vmapped)",
                ),
            )

        return _fit_chains(
            model,
            data,
            init_params,
            seed,
            num_chains=self.num_chains,
            logdensity_key_seed=self.logdensity_key_seed,
            precondition=self.prior_preconditioning,
            run_chains=run_chains,
        )


def _mala_stats(info) -> dict:
    """Per-draw MALA statistics."""
    return {
        "acceptance_rate": info.acceptance_rate,
        "is_accepted": info.is_accepted,
    }
