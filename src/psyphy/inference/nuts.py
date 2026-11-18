r"""
nuts.py
-------

NUTS (No-U-Turn Sampler) inference engine backed by BlackJAX

Uses gradient-based MCMC to draw samples from the full posterior
p(θ| data), unlike MAP which returns only a point estimate.

Connections
-----------
- Shares its chain loop with every psyphy MCMC engine, in
  ``psyphy.inference.mcmc``: the fixed MC key in the log density, prior
  preconditioning, vmapped chains, and the returned MCMCPosterior. The
  notes on the stochastic likelihood are there.

Notes on warmup cost
--------------------
Adaptive warmup used to dominate wall-clock time for three reasons, each
addressed here:

1. Badly scaled starting metric. window_adaptation starts from an identity
   mass matrix, but WPPM coefficients have prior stds from ~0.06 (degree 0)
   down to ~0.01 (high degree), and the posterior is narrower still. With an
   identity metric the step size must shrink to the smallest posterior std,
   so every trajectory needs O(1/step_size) leapfrog steps and NUTS runs to
   its maximum tree depth (1023 gradients per draw at BlackJAX's default)
   until the first mass-matrix update ~100 steps in.
   Fix: *prior preconditioning*. The sampler works in z = W / prior_std,
   where every coordinate is O(1) a priori. This is the same as starting
   adaptation from inverse_mass_matrix = prior variance, but needs no
   BlackJAX-version-specific keyword. Draws are mapped back to W before
   they are returned. See ``prior_preconditioning``.
2. Unbounded tree depth. Even with a good metric, a stiff region can push
   a single draw to 2^10 gradients. Fix: ``max_num_doublings`` caps it.
3. Sequential chains with per-chain recompilation. Chains ran in a Python
   loop that rebuilt the warmup and kernel closures every iteration, so XLA
   recompiled both scans once per chain. Fix: warmup and sampling are each
   ``jax.vmap``-ed over chains and ``jax.jit``-ed once: one compilation,
   and all chains advance together.

Requires
--------
blackjax >= 1.0  (install with: pip install 'psyphy[sampling]')
"""

from __future__ import annotations

import contextlib
from functools import partial

import jax
import jax.numpy as jnp

from psyphy.inference.base import InferenceEngine
from psyphy.inference.mcmc import (
    _fit_chains,
    _import_blackjax,
    _make_pbar,
    _run_fixed_kernel,
)
from psyphy.posterior.mcmc_posterior import MCMCPosterior


class NUTSSampler(InferenceEngine):
    """
    NUTS (No-U-Turn Sampler) using BlackJAX.

    Draws samples from the full posterior p(θ | data) via gradient-based
    MCMC. Returns an MCMCPosterior with shape
    (num_chains, num_samples, *param_shape).

    Parameters
    ----------
    num_warmup : int, default=500
        Number of warmup/burn-in steps per chain.
    num_samples : int, default=1000
        Number of posterior draws per chain (after warmup).
    num_chains : int, default=4
        Number of independent chains to run.
    logdensity_key_seed : int, default=0
        Seed for the fixed JAX key baked into the logdensity closure.
        Controls which MC noise realization is used throughout sampling.
        See module docstring for rationale.
    target_acceptance_rate : float, default=0.8
        Target Metropolis acceptance rate for window adaptation
        (adaptive warmup only). Typical values: 0.-0.9.
    step_size : float | None, default=None
        If None (default), use blackjax.window_adaptation to
        automatically tune step_size and mass matrix (adaptive warmup).
        If a float, skip adaptation and use this fixed step_size for all
        chains; warmup becomes plain burn-in and all chains are vmapped
        in parallel (fixed step size). The caller is responsible for choosing a
        suitable value.
    max_num_doublings : int, default=8
        Maximum NUTS tree depth. One draw costs at most
        2**max_num_doublings - 1 gradient evaluations (255 at the default;
        BlackJAX's own default of 10 allows 1023). Applies to both modes,
        warmup and sampling. If ``sampler_stats["n_steps"]`` sits at the cap
        for most draws, the trajectories are being truncated: mixing gets
        worse (lower ESS) but the chain is still a valid MCMC sampler.
        Raise the cap, or improve the parameterization.
    prior_preconditioning : bool, default=True
        Adaptive warmup only. If True and the model exposes ``model.prior.param_scales()``
        (WPPM does), sample z = W / prior_std instead of W, so adaptation
        starts from a mass matrix matched to the prior scales rather than
        the identity. Results are returned in W coordinates either way. Set
        False to reproduce the old identity-start behavior.
    show_progress : bool, default=True
        Display a tqdm progress bar during warmup and sampling.
        Requires `tqdm`, which ships in the ``progress`` extra
        (``pip install 'psyphy[progress]'``) and is NOT a core dependency.
        Silently skipped if tqdm is not installed, so on a clean install this
        default is a no-op and a long fit prints nothing.

    Notes
    -----
    **Two warmup modes:**

    *Adaptive warmup* (step_size=None, default): blackjax.window_adaptation
    adapts step_size and inverse_mass_matrix independently per chain.
    The adaptation state is all JAX arrays, so warmup is vmapped over
    chains and JIT-compiled once; sampling is likewise one vmapped,
    jitted lax.scan using each chain's own adapted parameters.

    *Fixed step size* (step_size=<float>): No adaptation. All chains share the
    same fixed step_size and a diagonal mass matrix of ones. Warmup is
    pure burn-in. All chains are fully vmapped -> single compiled call.
    Faster for many chains but requires manual step_size tuning.

    Examples
    --------
    >>> from psyphy.inference import NUTSSampler
    >>> sampler = NUTSSampler(num_warmup=200, num_samples=500, num_chains=4)
    >>> nuts_posterior = sampler.fit(
    ...     model, data, init_params=map_posterior.params
    ... )  # doctest: +SKIP
    >>> print(nuts_posterior.n_chains, nuts_posterior.n_draws)  # doctest: +SKIP
    4 500
    """

    def __init__(
        self,
        num_warmup: int = 500,
        num_samples: int = 1000,
        num_chains: int = 4,
        *,
        logdensity_key_seed: int = 0,
        target_acceptance_rate: float = 0.8,
        step_size: float | None = None,
        max_num_doublings: int = 8,
        prior_preconditioning: bool = True,
        show_progress: bool = True,
    ):
        self.num_warmup = int(num_warmup)
        self.num_samples = int(num_samples)
        self.num_chains = int(num_chains)
        self.logdensity_key_seed = int(logdensity_key_seed)
        self.target_acceptance_rate = float(target_acceptance_rate)
        self.step_size = float(step_size) if step_size is not None else None
        if int(max_num_doublings) < 1:
            raise ValueError(f"max_num_doublings must be >= 1, got {max_num_doublings}")
        self.max_num_doublings = int(max_num_doublings)
        self.prior_preconditioning = bool(prior_preconditioning)
        self.show_progress = bool(show_progress)

    # ------------------------------------------------------------------
    # InferenceEngine protocol

    def fit(
        self,
        model,
        data,
        init_params: dict | None = None,
        seed: int | None = None,
    ) -> MCMCPosterior:
        """
        Run NUTS sampling and return a posterior over model parameters.

        Parameters
        ----------
        model : WPPM
            Model instance providing log_posterior_from_data.
        data : TrialData
            Observed trials.
        init_params : dict | None, optional
            Initial parameter PyTree. If None, samples from the model prior
            using the seed. Providing MAP params as init_params
            dramatically reduces the warmup needed.
        seed : int | None, optional
            Master PRNG seed for chain initialization and sampling keys.
            None means 0.

        Returns
        -------
        MCMCPosterior
            Samples shaped (num_chains, num_samples, *param_shape),
            sampler_stats with acceptance_rate.
        """
        blackjax = _import_blackjax("NUTSSampler")

        if self.step_size is None:
            run_chains = partial(self._run_adaptive, blackjax)
            # Prior preconditioning is for adaptive warmup only. A fixed step
            # size was picked for W coordinates.
            precondition = self.prior_preconditioning
        else:
            run_chains = partial(self._run_fixed_stepsize, blackjax)
            precondition = False

        return _fit_chains(
            model,
            data,
            init_params,
            seed,
            num_chains=self.num_chains,
            logdensity_key_seed=self.logdensity_key_seed,
            precondition=precondition,
            run_chains=run_chains,
        )

    # ------------------------------------------------------------------
    # Adaptive warmup (window_adaptation), chains vmapped
    #
    # window_adaptation adapts two things during warmup, per chain:
    #   1. step_size via dual averaging -- after each proposal it checks
    #      whether the acceptance rate is above/below target and nudges the
    #      step size up/down until it produces target_acceptance_rate.
    #   2. inverse_mass_matrix via the empirical variance of the warmup
    #      draws -- scales each parameter by its marginal posterior variance,
    #      so parameters with very different magnitudes (W at different
    #      basis degrees) all move at comparable rates.
    #
    # All of that adaptation state is JAX arrays carried through a lax.scan
    # inside BlackJAX, so the whole warmup is a pure function of
    # (rng_key, init_position) and can be jax.vmap-ed over chains. Each chain
    # still adapts independently: vmap batches the computation, it does not
    # pool the statistics.
    #
    # Earlier versions ran chains sequentially in a Python loop on the
    # (incorrect) assumption that adaptation state was Python-level. That
    # loop also rebuilt the warmup and kernel closures every iteration, so
    # XLA recompiled both scans once per chain -- minutes per chain for the
    # expensive MC likelihood.
    #
    # Cost model under vmap: NUTS builds its trajectory in a while_loop, and
    # a vmapped while_loop keeps running until EVERY chain has finished its
    # tree. So each step costs the deepest tree across chains -- one reason
    # max_num_doublings matters more here than in the sequential version.
    # ------------------------------------------------------------------

    def _run_adaptive(self, blackjax, logdensity_fn, init_position, chain_keys):
        """Run adaptive NUTS -- all chains vmapped, one compilation per phase.

        BlackJAX inputs per chain
        -------------------------
        logdensity_fn   : dict -> scalar   (see fit() for full contract; may
                                            be the preconditioned z-space one)
        init_position   : dict             e.g. {"W": (2, 2, 2, 3)}; shared by
                                            all chains (typically the MAP)
        chain_keys      : (num_chains, 2)  one PRNGKey per chain

        Returns
        -------
        positions       : dict            {"W": (num_chains, num_samples, ...)}
        sampler_stats   : dict            each (num_chains, num_samples)
        """
        # Built ONCE and shared by every chain. Its closures (and therefore
        # the jitted functions below) are created once per fit() call, so XLA
        # compiles warmup once and sampling once, regardless of num_chains.
        #
        # max_num_doublings is forwarded by window_adaptation to blackjax.nuts
        # as an extra parameter, so the cap applies during warmup too --
        # which is exactly where the deep trees happen.
        #
        # NOTE: no `progress_bar=` here. BlackJAX's own progress
        # bar needs an extra package (fastprogress on 1.5, jax-tap on 1.6+)
        # that is not a default dependency, so passing progress_bar=True on a
        # plain `pip install psyphy[sampling]` raises ImportError from inside
        # a JAX callback. This class draws its own tqdm bar via _make_pbar().
        warmup = blackjax.window_adaptation(
            blackjax.nuts,
            logdensity_fn,
            target_acceptance_rate=self.target_acceptance_rate,
            max_num_doublings=self.max_num_doublings,
        )

        def warmup_one_chain(warmup_key):
            # warmup.run returns: (state, parameters), info
            #   state      : NUTSState at the end of warmup (the sampling start)
            #   parameters : {"step_size": (), "inverse_mass_matrix": (n_params,),
            #                 ...plus the extra parameters we passed in, i.e.
            #                 "max_num_doublings"}
            # `info` holds per-step adaptation traces; it is dropped so the
            # vmapped output does not carry (num_chains, num_warmup, ...) arrays.
            (state, parameters), _ = warmup.run(
                warmup_key, init_position, num_steps=self.num_warmup
            )
            # Keep only the two adapted quantities. The echoed
            # "max_num_doublings" comes back from vmap as a batched int array;
            # passing it on to blackjax.nuts would make the tree-depth bound a
            # traced value instead of a Python int, so it is dropped here and
            # self.max_num_doublings is passed explicitly in sample_one_chain.
            return state, (parameters["step_size"], parameters["inverse_mass_matrix"])

        def sample_one_chain(state, step_size, inverse_mass_matrix, sample_key):
            # Adapted parameters are frozen for sampling. Under vmap they are
            # per-chain tracers, which blackjax.nuts accepts (they are only
            # used arithmetically inside the integrator).
            # kernel signature: (PRNGKey, NUTSState) -> (NUTSState, NUTSInfo)
            kernel = blackjax.nuts(
                logdensity_fn,
                step_size=step_size,
                inverse_mass_matrix=inverse_mass_matrix,
                max_num_doublings=self.max_num_doublings,
            ).step

            def one_step(state, rng_key):
                state, info = kernel(rng_key, state)
                return state, (state.position, _nuts_stats(info))

            # lax.scan carry : NUTSState
            # lax.scan xs    : (num_samples, 2) PRNGKeys
            # lax.scan ys    : positions {"W": (num_samples, *shape)}, stats
            #                  dict, each (num_samples,)
            sample_keys = jax.random.split(sample_key, self.num_samples)
            _, outputs = jax.lax.scan(one_step, state, sample_keys)
            return outputs

        # Each chain key is split into an independent warmup and sampling key.
        # split_keys: (num_chains, 2, 2) -> [:, 0] warmup keys, [:, 1] sample keys
        split_keys = jax.vmap(jax.random.split)(chain_keys)
        warmup_keys, sample_keys = split_keys[:, 0], split_keys[:, 1]

        pbar = _make_pbar(
            self.show_progress,
            total=self.num_chains * (self.num_warmup + self.num_samples),
            desc="NUTS (adaptive)",
        )

        # Warmup, all chains at once. The bar can only move when the
        # whole compiled call returns -- there is no per-step host callback --
        # so the description says what is running while it sits still.
        if pbar is not None:
            with contextlib.suppress(Exception):
                pbar.set_description(f"NUTS warmup ({self.num_chains} chains, vmapped)")
        states, (step_sizes, inverse_mass_matrices) = jax.jit(
            jax.vmap(warmup_one_chain)
        )(warmup_keys)
        # states                : NUTSState with a leading (num_chains,) axis
        # step_sizes            : (num_chains,)
        # inverse_mass_matrices : (num_chains, n_params)
        if pbar is not None:
            with contextlib.suppress(Exception):
                pbar.update(self.num_chains * self.num_warmup)
                pbar.set_description(
                    f"NUTS sampling ({self.num_chains} chains, vmapped)"
                )

        # Sampling, all chains at once, each with its own adapted
        # step size and mass matrix.
        positions, sampler_stats = jax.jit(jax.vmap(sample_one_chain))(
            states, step_sizes, inverse_mass_matrices, sample_keys
        )

        if pbar is not None:
            with contextlib.suppress(Exception):
                pbar.update(self.num_chains * self.num_samples)
                pbar.close()

        # vmap already stacked the chain axis first:
        #   positions : {"W": (n_chains, n_draws, *param_shape)}
        #   stats     : each (n_chains, n_draws)
        return positions, sampler_stats

    # ------------------------------------------------------------------
    # Fixed step size, all chains vmapped
    #
    # WHY vmappable (contrast with adaptive warmup):
    # step_size and inverse_mass_matrix are fixed constants provided upfront.
    # Every piece of state (NUTSState, step_size, M^-1) is a JAX array
    # from the start. jax.vmap can stack these along a new chain axis and
    # compile all chains into a single XLA kernel (one compiled call).
    # Trade-off: the caller must supply a reasonable step_size manually, and
    # without mass matrix adaptation chains may mix slowly if parameters
    # have very different scales (identity M^-1 treats all params equally).
    # ------------------------------------------------------------------

    def _run_fixed_stepsize(self, blackjax, logdensity_fn, init_position, chain_keys):
        """Run NUTS with fixed step_size: all chains vmapped in one call.

        Returns
        -------
        positions       : dict            {"W": (num_chains, num_samples, ...)}
        sampler_stats   : dict            each (num_chains, num_samples)
        """
        # inverse_mass_matrix: (n_params,), diagonal identity, no per-parameter scaling
        # For WPPM {"W": (2,2,2,3)}: n_params = 2*2*2*3 = 48.
        n_params = sum(x.size for x in jax.tree.leaves(init_position))
        nuts = blackjax.nuts(
            logdensity_fn,
            step_size=self.step_size,
            inverse_mass_matrix=jnp.ones(n_params),
            max_num_doublings=self.max_num_doublings,
        )
        return _run_fixed_kernel(
            nuts,
            init_position,
            chain_keys,
            num_warmup=self.num_warmup,
            num_samples=self.num_samples,
            stats_fn=_nuts_stats,
            pbar=_make_pbar(
                self.show_progress, total=1, desc="NUTS (fixed step_size, vmapped)"
            ),
        )


def _nuts_stats(info) -> dict:
    """Per-draw NUTS statistics, keyed in ArviZ's sample_stats spelling.

    ``diverging`` is the first NUTS diagnostic to check. ``n_steps`` pinned at
    2**max_num_doublings - 1 means trajectories are truncated.
    """
    return {
        "acceptance_rate": info.acceptance_rate,
        "diverging": info.is_divergent,
        "energy": info.energy,
        "n_steps": info.num_integration_steps,
    }
