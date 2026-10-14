r"""
nuts.py
-------

NUTS (No-U-Turn Sampler) inference engine backed by BlackJAX

Uses gradient-based MCMC to draw samples from the full posterior
p(θ| data), unlike MAP which returns only a point estimate.

Connections
-----------
-  BlackJax needs a log density and calls
    WPPM.log_posterior_from_data(params, data, key=fixed_key) as the
  log-density. A fixed key is baked in at construction time to make the
  stochastic MC likelihood deterministic for NUTS (see Notes).
- Returns an MCMCPosterior wrapping the sampled chains and model.

Notes on stochastic likelihood
-------------------------------
WPPM's current oddity-task-likelihood is estimated
via Monte Carlo inside OddityTask.loglik.
NUTS requires a (nearly) deterministic log-density to compute consistent
Hamiltonian dynamics. The chosen approach: a single fixed JAX key is baked
into the logdensity closure so every gradient evaluation uses the same MC
noise realization. This bias decreases as MC_SAMPLES -> \inf (>=200 recommended).

Alternatives (not implemented here):
- fresh key per NUTS *trajectory* (pseudo-marginal / noisy HMC)
- Neural surrogate likelihood
(exact gradients, see NeuralSurrogateTask, not yet implemented)

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
import sys

import jax
import jax.numpy as jnp

from psyphy.inference.base import InferenceEngine
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
        (Mode A only). Typical values: 0.-0.9.
    step_size : float | None, default=None
        If None (default), use blackjax.window_adaptation to
        automatically tune step_size and mass matrix (Mode A).
        If a float, skip adaptation and use this fixed step_size for all
        chains — warmup becomes plain burn-in and all chains are vmapped
        in parallel (Mode B). The user is responsible for choosing a
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
        Mode A only. If True and the model exposes ``model.prior.param_scales()``
        (WPPM does), sample z = W / prior_std instead of W, so adaptation
        starts from a mass matrix matched to the prior scales rather than
        the identity. Results are returned in W coordinates either way. Set
        False to reproduce the old identity-start behaviour.
    seed : int, default=0
        Master PRNG seed for chain initialization and sampling keys.
    show_progress : bool, default=True
        Display a tqdm progress bar during warmup and sampling.
        Requires `tqdm`, which ships in the ``progress`` extra
        (``pip install 'psyphy[progress]'``) and is NOT a core dependency.
        Silently skipped if tqdm is not installed, so on a clean install this
        default is a no-op and a long fit prints nothing.

    Notes
    -----
    **Two warmup modes:**

    *Mode A* (step_size=None, default): blackjax.window_adaptation
    adapts step_size and inverse_mass_matrix independently per chain.
    The adaptation state is all JAX arrays, so warmup is vmapped over
    chains and JIT-compiled once; sampling is likewise one vmapped,
    jitted lax.scan using each chain's own adapted parameters.

    *Mode B* (step_size=<float>): No adaptation. All chains share the
    same fixed step_size and a diagonal mass matrix of ones. Warmup is
    pure burn-in. All chains are fully vmapped -> single compiled call.
    Faster for many chains but requires manual step_size tuning.

    Examples
    --------
    >>> from psyphy.inference import NUTSSampler
    >>> sampler = NUTSSampler(num_warmup=200, num_samples=500, num_chains=4)
    >>> nuts_posterior = sampler.fit(model, data, init_params=map_posterior.params)
    >>> print(nuts_posterior.n_chains, nuts_posterior.n_draws)
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
        seed: int = 0,
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
        self.seed = int(seed)
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
            using the master seed. Providing MAP params as init_params
            dramatically reduces the warmup needed.
        seed : int | None, optional
            Override the master PRNG seed set at construction.

        Returns
        -------
        MCMCPosterior
            Samples shaped (num_chains, num_samples, *param_shape),
            sampler_stats with acceptance_rate.
        """
        try:
            import blackjax
        except ImportError as e:
            raise ImportError(
                "BlackJAX is required for NUTSSampler. "
                "Install it with: pip install 'psyphy[sampling]'\n"
                f"Note: blackjax >= 1.4 requires Python >= 3.11 (this is "
                f"{sys.version_info.major}.{sys.version_info.minor}), so on an "
                "older interpreter the 'sampling' extra resolves to nothing and "
                "this error appears even after installing it."
            ) from e

        rng_seed = self.seed if seed is None else int(seed)
        master_key = jax.random.PRNGKey(rng_seed)
        init_key, chain_key = jax.random.split(master_key)

        # init_position: dict matching the model's parameter PyTree structure.
        # For WPPM: {"W": jnp.ndarray of shape (degree+1, degree+1, input_dim, embedding_dim)}
        # e.g. degree=1, input_dim=2, extra_dims=1 -> {"W": shape (2, 2, 2, 3)}
        # BlackJAX requires this as the starting point for Hamiltonian dynamics.
        init_position = (
            init_params if init_params is not None else model.init_params(init_key)
        )

        # Fixed MC key baked into logdensity — makes the stochastic
        # likelihood deterministic for NUTS gradient evaluations.
        fixed_mc_key = jax.random.PRNGKey(self.logdensity_key_seed)

        # -- BlackJAX API boundary --------------------------------
        # logdensity_fn is the ONLY thing BlackJAX needs from psyphy.
        # Contract:
        #   input:  params  — dict, same structure as init_position
        #                      e.g. {"W": (2, 2, 2, 3)}
        #   output: scalar  — log p(θ | data), a JAX float (shape ())
        #   must be: JAX-differentiable (jax.grad must work on it)
        #            (nearly) deterministic — hence the fixed_mc_key
        # -----------------------------------------
        def logdensity_fn(params: dict) -> jnp.ndarray:
            return model.log_posterior_from_data(params, data, key=fixed_mc_key)

        # chain_keys: shape (num_chains, 2) — one PRNGKey per chain,
        # split further inside each mode into warmup_key and sample_key.
        chain_keys = jax.random.split(chain_key, self.num_chains)

        # blackjax is passed as an argument rather than imported again in each
        # private method. Because it is a soft dependency, the single import
        # (with its error-handling) lives here in fit(). In Python, modules are
        # first-class objects — 'import blackjax' just binds the module to a
        # local variable, which can be passed around like any other value.
        if self.step_size is None:
            # Prior preconditioning (Mode A only -- in Mode B the user picked
            # step_size for W coordinates, and silently rescaling would change
            # what that number means). See "Notes on warmup cost" above.
            scales = (
                self._prior_scales(model, init_position)
                if self.prior_preconditioning
                else None
            )
            if scales is None:
                positions, sampler_stats = self._run_adaptive(
                    blackjax, logdensity_fn, init_position, chain_keys
                )
            else:
                # Change of variables W = scales * z. The Jacobian of a fixed
                # linear map is a constant, so it is dropped: it does not
                # change the posterior or any accept/reject decision.
                def z_logdensity_fn(z: dict) -> jnp.ndarray:
                    return logdensity_fn(jax.tree.map(jnp.multiply, z, scales))

                init_z = jax.tree.map(jnp.divide, init_position, scales)
                z_positions, sampler_stats = self._run_adaptive(
                    blackjax, z_logdensity_fn, init_z, chain_keys
                )
                # Map draws back to W. z leaves are (n_chains, n_draws, *shape)
                # and scales leaves are (*shape), so they broadcast over the
                # two leading axes. Every downstream consumer (MCMCPosterior,
                # ArviZ, plots) therefore only ever sees W.
                positions = jax.tree.map(jnp.multiply, z_positions, scales)
        else:
            positions, sampler_stats = self._run_fixed_stepsize(
                blackjax, logdensity_fn, init_position, chain_keys
            )

        # positions: dict, each value shaped (n_chains, n_draws, *param_shape)
        #   e.g. {"W": (num_chains, num_samples, 2, 2, 2, 3)}
        # sampler_stats: each value shaped (num_chains, num_samples) — one scalar
        # per draw per chain. Keys follow ArviZ's sample_stats convention.
        return MCMCPosterior(positions, model, sampler_stats=sampler_stats)

    # ------------------------------------------------------------------
    # Prior preconditioning helper
    # ------------------------------------------------------------------

    @staticmethod
    def _prior_scales(model, init_position):
        """Per-parameter prior stds shaped like ``init_position``, or None.

        Returns None (-> no preconditioning, identity-start adaptation) when
        the model has no ``prior.param_scales()`` or when its output does not
        line up leaf-for-leaf with ``init_position``. Silently falling back
        is deliberate: preconditioning is an optimization, never a
        requirement, so an unfamiliar model must still sample.
        """
        prior = getattr(model, "prior", None)
        if not hasattr(prior, "param_scales"):
            return None
        scales = prior.param_scales()
        # Same PyTree structure (e.g. both {"W": ...}) and same leaf shapes,
        # otherwise the elementwise W = scales * z below would be wrong.
        if jax.tree.structure(scales) != jax.tree.structure(init_position):
            return None
        if any(
            s.shape != jnp.shape(p)
            for s, p in zip(jax.tree.leaves(scales), jax.tree.leaves(init_position))
        ):
            return None
        return scales

    # ------------------------------------------------------------------
    # Mode A: adaptive warmup (window_adaptation), chains vmapped
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
        # NOTE: deliberately no `progress_bar=` here. BlackJAX's own progress
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
                # Names match ArviZ's sample_stats convention (diverging, energy,
                # n_steps) so to_arviz() feeds az.plot_energy / divergence
                # reporting directly. `diverging` is the single most important
                # NUTS diagnostic: any divergence makes a run suspect regardless
                # of what R-hat says. `n_steps` is how to check whether
                # max_num_doublings is truncating trajectories: values pinned
                # at 2**max_num_doublings - 1 mean it is.
                return state, (
                    state.position,
                    info.acceptance_rate,
                    info.is_divergent,
                    info.energy,
                    info.num_integration_steps,
                )

            # lax.scan carry : NUTSState
            # lax.scan xs    : (num_samples, 2) PRNGKeys
            # lax.scan ys    : positions {"W": (num_samples, *shape)} + 4 stats,
            #                  each (num_samples,)
            sample_keys = jax.random.split(sample_key, self.num_samples)
            _, outputs = jax.lax.scan(one_step, state, sample_keys)
            return outputs

        # Each chain key is split into an independent warmup and sampling key.
        # split_keys: (num_chains, 2, 2) -> [:, 0] warmup keys, [:, 1] sample keys
        split_keys = jax.vmap(jax.random.split)(chain_keys)
        warmup_keys, sample_keys = split_keys[:, 0], split_keys[:, 1]

        pbar = self._make_pbar(
            total=self.num_chains * (self.num_warmup + self.num_samples),
            desc="NUTS (adaptive)",
        )

        # Phase 1: warmup, all chains at once. The bar can only move when the
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

        # Phase 2: sampling, all chains at once, each with its own adapted
        # step size and mass matrix.
        positions, acc_rates, diverging, energy, n_steps = jax.jit(
            jax.vmap(sample_one_chain)
        )(states, step_sizes, inverse_mass_matrices, sample_keys)

        if pbar is not None:
            with contextlib.suppress(Exception):
                pbar.update(self.num_chains * self.num_samples)
                pbar.close()

        # vmap already stacked the chain axis first:
        #   positions : {"W": (n_chains, n_draws, *param_shape)}
        #   stats     : (n_chains, n_draws)
        return positions, {
            "acceptance_rate": acc_rates,
            "diverging": diverging,
            "energy": energy,
            "n_steps": n_steps,
        }

    # ------------------------------------------------------------------
    # Mode B: fixed step_size, all chains vmapped
    #
    # WHY vmappable (contrast with Mode A):
    # step_size and inverse_mass_matrix are fixed constants provided upfront.
    # Every piece of state — NUTSState, step_size, M^-1 — is a JAX array
    # from the start. jax.vmap can stack these along a new chain axis and
    # compile all chains into a single XLA kernel (one compiled call).
    # Trade-off: the user must supply a reasonable step_size manually, and
    # without mass matrix adaptation chains may mix slowly if parameters
    # have very different scales (identity M^-1 treats all params equally).
    # ------------------------------------------------------------------

    def _run_fixed_stepsize(self, blackjax, logdensity_fn, init_position, chain_keys):
        """Run NUTS with fixed step_size — all chains vmapped in one call.

        BlackJAX inputs
        ---------------
        logdensity_fn        : dict -> scalar   (see fit() for full contract)
        init_position        : dict             e.g. {"W": (2, 2, 2, 3)}
        step_size            : float            user-supplied; no adaptation
        inverse_mass_matrix  : (n_params,)      diagonal identity — no per-parameter scaling

        BlackJAX outputs (via jax.vmap over chain_keys)
        ------------------------------------------------
        all_positions : dict  e.g. {"W": (num_chains, num_samples, 2, 2, 2, 3)}
        all_acc_rates : (num_chains, num_samples)
        """
        # n_params: total scalar count across all parameter leaves.
        # For WPPM {"W": (2,2,2,3)}: n_params = 2*2*2*3 = 48.
        # inverse_mass_matrix shape: (n_params,) — diagonal of M^-1 (identity = no scaling).
        flat_leaves = jax.tree.leaves(init_position)
        n_params = sum(x.size for x in flat_leaves)
        inverse_mass_matrix = jnp.ones(n_params)  # shape: (n_params,)

        # blackjax.nuts(...) returns an object with:
        #   .init(position)             -> NUTSState
        #   .step(PRNGKey, NUTSState)   -> (NUTSState, NUTSInfo)
        # All parameters are JAX arrays -> entire run_one_chain is vmappable.
        # max_num_doublings: same tree-depth cap as Mode A -- at most
        # 2**max_num_doublings - 1 gradient evaluations per draw.
        nuts = blackjax.nuts(
            logdensity_fn,
            step_size=self.step_size,
            inverse_mass_matrix=inverse_mass_matrix,
            max_num_doublings=self.max_num_doublings,
        )

        def run_one_chain(chain_key):
            # chain_key: PRNGKey shape (2,) — one per chain, vmapped over
            warmup_key, sample_key = jax.random.split(chain_key)

            # nuts.init: position dict -> NUTSState
            # NUTSState holds: position (dict), momentum, potential_energy, ...
            state = nuts.init(init_position)

            # lax.scan carry : NUTSState
            # lax.scan xs    : keys shape (num_steps, 2)
            # lax.scan ys    : (position dict, scalar acc_rate)
            def one_step(state, rng_key):
                state, info = nuts.step(rng_key, state)
                # state.position : dict  e.g. {"W": (2, 2, 2, 3)}
                # stats          : ArviZ sample_stats names -- see Mode A's note
                return state, (
                    state.position,
                    info.acceptance_rate,
                    info.is_divergent,
                    info.energy,
                    info.num_integration_steps,
                )

            # Burn-in: num_warmup steps, no adaptation, positions discarded.
            # (contrast Mode A: here there is no step_size tuning, just mixing)
            burn_keys = jax.random.split(warmup_key, self.num_warmup)
            state, _ = jax.lax.scan(one_step, state, burn_keys)

            # Sampling: positions kept
            # positions : dict  e.g. {"W": (num_samples, 2, 2, 2, 3)}
            # acc_rates : (num_samples,)
            s_keys = jax.random.split(sample_key, self.num_samples)
            _, (positions, acc_rates, diverging, energy, n_steps) = jax.lax.scan(
                one_step, state, s_keys
            )
            return positions, acc_rates, diverging, energy, n_steps

        pbar = self._make_pbar(total=1, desc="NUTS (fixed step_size, vmapped)")

        # jax.vmap(run_one_chain)(chain_keys):
        #   input  : chain_keys shape (num_chains, 2)
        #   output : positions dict e.g. {"W": (num_chains, num_samples, 2, 2, 2, 3)}
        #            acc_rates shape (num_chains, num_samples)
        all_positions, all_acc, all_div, all_energy, all_n_steps = jax.vmap(
            run_one_chain
        )(chain_keys)

        if pbar is not None:
            with contextlib.suppress(Exception):
                pbar.update(1)
                pbar.close()

        return all_positions, {
            "acceptance_rate": all_acc,
            "diverging": all_div,
            "energy": all_energy,
            "n_steps": all_n_steps,
        }

    # ------------------------------------------------------------------
    # Progress bar helper
    # ------------------------------------------------------------------

    def _make_pbar(self, *, total: int, desc: str):
        if not self.show_progress:
            return None
        try:
            from tqdm.auto import tqdm

            return tqdm(total=total, desc=desc, leave=False)
        except Exception:
            return None
