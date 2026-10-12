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
    adapts step_size and inverse_mass_matrix per chain. Adaptation uses
    Python-level averaging and is NOT vmappable — chains run
    sequentially in a Python for loop. After warmup, each chain's
    sampling is JIT-compiled via jax.lax.scan.

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
        seed: int = 0,
        show_progress: bool = True,
    ):
        self.num_warmup = int(num_warmup)
        self.num_samples = int(num_samples)
        self.num_chains = int(num_chains)
        self.logdensity_key_seed = int(logdensity_key_seed)
        self.target_acceptance_rate = float(target_acceptance_rate)
        self.step_size = float(step_size) if step_size is not None else None
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
            positions, sampler_stats = self._run_adaptive(
                blackjax, logdensity_fn, init_position, chain_keys
            )
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
    # Mode A: adaptive warmup (window_adaptation), sequential chains
    #
    # WHY sequential and not vmapped:
    # window_adaptation adapts two things during warmup:
    #   1. step_size via dual averaging — after each proposal it checks
    #      whether acceptance rate is above/below target and nudges ε
    #      up/down. Converges to the ε that produces target_acceptance_rate.
    #   2. inverse_mass_matrix via empirical covariance — scales each
    #      parameter by its marginal posterior variance, so params with very
    #      different magnitudes (e.g. W at different basis degrees) all move
    #      at comparable rates.
    # Both accumulators are plain Python floats updated in a Python for-loop
    # inside blackJAX — they live outside the JAX computational graph.
    # jax.vmap requires ALL state to be JAX arrays (so JAX can stack them
    # along a batch axis). Python floats cannot be stacked -> not vmappable.
    # Therefore chains run sequentially. Once warmup ends the adapted values
    # are frozen as JAX scalars and sampling runs as a pure lax.scan.
    # ------------------------------------------------------------------

    def _run_adaptive(self, blackjax, logdensity_fn, init_position, chain_keys):
        """Run adaptive NUTS — one chain at a time (window_adaptation not vmappable).

        BlackJAX inputs per chain
        -------------------------
        logdensity_fn   : dict -> scalar   (see fit() for full contract)
        init_position   : dict             e.g. {"W": (2, 2, 2, 3)}
        warmup_key      : PRNGKey          shape (2,)

        BlackJAX outputs per chain
        --------------------------
        state           : NUTSState       internal HMC state (position + momentum + ...)
        parameters      : dict            {"step_size": float,
                                           "inverse_mass_matrix": (n_params,)}
                                          adapted by dual averaging / empirical cov
        positions       : dict            {"W": (num_samples, 2, 2, 2, 3)}
                                          one param dict per draw from lax.scan
        acc_rates       : (num_samples,)  Metropolis acceptance rate per draw
        """
        # all_positions: list of num_chains dicts, each {"W": (num_samples, *param_shape)}
        # all_acc_rates: list of num_chains arrays, each shape (num_samples,)
        all_positions = []
        all_acc_rates = []
        all_diverging = []
        all_energy = []
        all_n_steps = []

        pbar = self._make_pbar(
            total=self.num_chains * (self.num_warmup + self.num_samples),
            desc="NUTS (adaptive)",
        )

        for i, chain_key in enumerate(chain_keys):
            warmup_key, sample_key = jax.random.split(chain_key)

            if pbar is not None:
                with contextlib.suppress(Exception):
                    pbar.set_description(f"NUTS warmup chain {i + 1}/{self.num_chains}")

            # BlackJAX warmup: runs dual averaging + empirical mass estimation.
            # NOT vmappable — adaptation state is Python-level, not JAX arrays.
            # NOTE: deliberately no `progress_bar=` here. BlackJAX's own progress
            # bar needs an extra package (fastprogress on 1.5, jax-tap on 1.6+)
            # that is not a default dependency, so passing progress_bar=True on a
            # plain `pip install psyphy[sampling]` raises ImportError from inside a
            # JAX callback -- i.e. NUTSSampler() with default arguments crashed.
            # This class already draws its own tqdm bar via _make_pbar(), so
            # asking BlackJAX for a second one only added a dependency.
            warmup = blackjax.window_adaptation(
                blackjax.nuts,
                logdensity_fn,
                target_acceptance_rate=self.target_acceptance_rate,
            )
            # warmup.run returns: (state, parameters), info
            #   state.position : dict  — final warmup position (same shape as init_position)
            #   parameters     : {"step_size": float, "inverse_mass_matrix": (n_params,)}
            #                    these are now frozen JAX scalars/arrays
            (state, parameters), _ = warmup.run(
                warmup_key, init_position, num_steps=self.num_warmup
            )

            if pbar is not None:
                with contextlib.suppress(Exception):
                    pbar.update(self.num_warmup)

            # Sampling phase: parameters are frozen -> pure lax.scan, fully JIT-compiled.
            # blackjax.nuts(**parameters).step unpacks to:
            #   step_size=<float>, inverse_mass_matrix=<(n_params,)>
            # kernel signature: (PRNGKey, NUTSState) -> (NUTSState, NUTSInfo)
            kernel = blackjax.nuts(logdensity_fn, **parameters).step

            # lax.scan carry : NUTSState
            # lax.scan xs    : sample_keys shape (num_samples, 2)
            # lax.scan ys    : (positions dict, acc_rates)
            #   positions    : {"W": (num_samples, 2, 2, 2, 3)}
            #   acc_rates    : (num_samples,)
            # _kernel=kernel : early binding — freezes this chain's kernel in the
            # closure so the loop variable isn't captured by reference (late binding
            # would mean the next iteration's kernel is used if scan ran later).
            def one_step(state, rng_key, _kernel=kernel):
                state, info = _kernel(rng_key, state)
                # Names match ArviZ's sample_stats convention (diverging, energy,
                # n_steps) so to_arviz() feeds az.plot_energy / divergence
                # reporting directly. `diverging` is the single most important
                # NUTS diagnostic: any divergence makes a run suspect regardless
                # of what R-hat says.
                return state, (
                    state.position,
                    info.acceptance_rate,
                    info.is_divergent,
                    info.energy,
                    info.num_integration_steps,
                )

            sample_keys = jax.random.split(sample_key, self.num_samples)
            _, (positions, acc_rates, diverging, energy, n_steps) = jax.lax.scan(
                one_step, state, sample_keys
            )

            if pbar is not None:
                with contextlib.suppress(Exception):
                    pbar.update(self.num_samples)

            all_positions.append(positions)
            all_acc_rates.append(acc_rates)
            all_diverging.append(diverging)
            all_energy.append(energy)
            all_n_steps.append(n_steps)

        if pbar is not None:
            with contextlib.suppress(Exception):
                pbar.close()

        # Stack chains: list of num_chains dicts -> dict of (n_chains, n_draws, *shape)
        # e.g. [{"W": (S,2,2,2,3)}, ...] x C  ->  {"W": (C, S, 2, 2, 2, 3)}
        stacked_positions = jax.tree.map(
            lambda *arrays: jnp.stack(arrays, axis=0), *all_positions
        )
        stacked_stats = {
            "acceptance_rate": jnp.stack(all_acc_rates, axis=0),  # (n_chains, n_draws)
            "diverging": jnp.stack(all_diverging, axis=0),
            "energy": jnp.stack(all_energy, axis=0),
            "n_steps": jnp.stack(all_n_steps, axis=0),
        }
        return stacked_positions, stacked_stats

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
        nuts = blackjax.nuts(
            logdensity_fn,
            step_size=self.step_size,
            inverse_mass_matrix=inverse_mass_matrix,
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
