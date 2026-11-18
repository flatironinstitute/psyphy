r"""
mcmc.py
-------

The chain loop that every psyphy MCMC engine shares. Private: the engines
(:class:`NUTSSampler`, :class:`MALASampler`) are the public names.

An engine builds its BlackJAX kernel and hands it to these functions. Everything
else is the same for every engine:

- a log density with a fixed Monte Carlo key (see below)
- prior preconditioning, sampling z = W / prior_std and mapping back to W
- all chains vmapped, burn-in then sampling
- the result wrapped in an :class:`MCMCPosterior`, draws shaped
  (n_chains, n_draws, *param_shape)

Notes on stochastic likelihood
------------------------------
WPPM's oddity-task likelihood is estimated by Monte Carlo inside
``OddityTask.loglik``. A gradient-based sampler needs a (nearly) deterministic
log density, so one fixed JAX key is baked into the log density and every
gradient evaluation uses the same MC noise realization. The bias this leaves
shrinks as the number of MC samples grows (>= 200 recommended). It applies to
every engine here.
"""

from __future__ import annotations

import contextlib
import sys
from collections.abc import Callable

import jax
import jax.numpy as jnp

from psyphy.posterior.mcmc_posterior import MCMCPosterior


def _import_blackjax(engine_name: str):
    """Import BlackJAX, or raise an ImportError naming the extra."""
    try:
        import blackjax
    except ImportError as e:
        raise ImportError(
            f"BlackJAX is required for {engine_name}. "
            "Install it with: pip install 'psyphy[sampling]'\n"
            f"Note: blackjax >= 1.4 requires Python >= 3.11 (this is "
            f"{sys.version_info.major}.{sys.version_info.minor}), so on an "
            "older interpreter the 'sampling' extra resolves to nothing and "
            "this error appears even after installing it."
        ) from e
    return blackjax


def _fit_chains(
    model,
    data,
    init_params: dict | None,
    seed: int | None,
    *,
    num_chains: int,
    logdensity_key_seed: int,
    precondition: bool,
    run_chains: Callable,
) -> MCMCPosterior:
    """Run an engine's chains on a model and wrap the draws in a posterior.

    Parameters
    ----------
    model : WPPM
        Model providing ``log_posterior_from_data`` and ``init_params``.
    data : TrialData
        Observed trials.
    init_params : dict | None
        Starting parameter PyTree, shared by all chains. If None, one draw from
        the prior with the seed.
    seed : int | None
        Seed for chain initialization and sampling. None means 0.
    num_chains : int
        Number of chains.
    logdensity_key_seed : int
        Seed of the fixed MC key baked into the log density.
    precondition : bool
        If True and the model exposes ``prior.param_scales()``, sample
        z = W / prior_std. Draws come back in W either way.
    run_chains : callable
        The engine's sampler, ``run_chains(logdensity_fn, init_position,
        chain_keys) -> (positions, sampler_stats)``. ``chain_keys`` has shape
        (num_chains, 2). Each value of ``positions`` has shape
        (num_chains, num_samples, *param_shape) and each value of
        ``sampler_stats`` has shape (num_chains, num_samples).

    Returns
    -------
    MCMCPosterior
        Draws in W coordinates, with the engine's ``sampler_stats``.
    """
    rng_seed = 0 if seed is None else int(seed)
    master_key = jax.random.PRNGKey(rng_seed)
    init_key, chain_key = jax.random.split(master_key)

    # init_position: dict matching the model's parameter PyTree structure.
    # e.g. degree=1, input_dim=2, extra_dims=1 -> {"W": shape (2, 2, 2, 3)}
    init_position = (
        init_params if init_params is not None else model.init_params(init_key)
    )

    fixed_mc_key = jax.random.PRNGKey(logdensity_key_seed)

    # -- BlackJAX API boundary --------------------------------
    # logdensity_fn is the only thing BlackJAX needs from psyphy.
    #   input:  params, a dict with the same structure as init_position
    #   output: scalar log p(θ | data), shape ()
    #   must be: differentiable with jax.grad, and (nearly) deterministic,
    #            hence the fixed_mc_key
    # -----------------------------------------
    def logdensity_fn(params: dict) -> jnp.ndarray:
        return model.log_posterior_from_data(params, data, key=fixed_mc_key)

    chain_keys = jax.random.split(chain_key, num_chains)  # (num_chains, 2)

    scales = _prior_scales(model, init_position) if precondition else None
    if scales is None:
        positions, sampler_stats = run_chains(logdensity_fn, init_position, chain_keys)
    else:
        # Change of variables W = scales * z. The Jacobian of a fixed linear
        # map is a constant, so it is dropped: it changes neither the
        # posterior nor any accept/reject decision.
        def z_logdensity_fn(z: dict) -> jnp.ndarray:
            return logdensity_fn(jax.tree.map(jnp.multiply, z, scales))

        init_z = jax.tree.map(jnp.divide, init_position, scales)
        z_positions, sampler_stats = run_chains(z_logdensity_fn, init_z, chain_keys)
        # z leaves (n_chains, n_draws, *shape) times scales leaves (*shape)
        positions = jax.tree.map(jnp.multiply, z_positions, scales)

    # positions: {"W": (n_chains, n_draws, *param_shape)}
    # sampler_stats: each (n_chains, n_draws), keys in ArviZ's sample_stats spelling
    return MCMCPosterior(model, positions, sampler_stats=sampler_stats)


def _prior_scales(model, init_position):
    """Per-parameter prior stds shaped like ``init_position``, or None.

    Returns None, and the engine samples W directly, when the model has no
    ``prior.param_scales()`` or when its output does not match
    ``init_position`` leaf for leaf.
    """
    prior = getattr(model, "prior", None)
    if not hasattr(prior, "param_scales"):
        return None
    scales = prior.param_scales()
    # same PyTree structure and leaf shapes, or W = scales * z is wrong
    if jax.tree.structure(scales) != jax.tree.structure(init_position):
        return None
    if any(
        s.shape != jnp.shape(p)
        for s, p in zip(jax.tree.leaves(scales), jax.tree.leaves(init_position))
    ):
        return None
    return scales


def _run_fixed_kernel(
    kernel,
    init_position: dict,
    chain_keys: jnp.ndarray,
    *,
    num_warmup: int,
    num_samples: int,
    stats_fn: Callable,
    pbar=None,
):
    """Run a kernel with fixed settings on all chains at once.

    Each chain starts at ``init_position``, takes ``num_warmup`` burn-in steps
    that are thrown away, then ``num_samples`` steps that are kept.

    Parameters
    ----------
    kernel : blackjax.SamplingAlgorithm
        Built with its settings, e.g. ``blackjax.mala(logdensity_fn, step_size)``.
        Provides ``.init(position)`` and ``.step(key, state)``.
    init_position : dict
        Starting PyTree, e.g. {"W": (2, 2, 2, 3)}.
    chain_keys : jnp.ndarray, shape (num_chains, 2)
        One PRNG key per chain.
    num_warmup, num_samples : int
        Burn-in and kept steps per chain.
    stats_fn : callable
        Maps the kernel's per-step info to a dict of scalars, keyed in ArviZ's
        ``sample_stats`` spelling.
    pbar : tqdm | None
        Advanced by one when all chains are done.

    Returns
    -------
    positions : dict
        Each leaf shaped (num_chains, num_samples, *param_shape).
    sampler_stats : dict
        Each value shaped (num_chains, num_samples).
    """

    def run_one_chain(chain_key):
        warmup_key, sample_key = jax.random.split(chain_key)
        state = kernel.init(init_position)

        def one_step(state, rng_key):
            state, info = kernel.step(rng_key, state)
            return state, (state.position, stats_fn(info))

        # burn-in, positions discarded
        burn_keys = jax.random.split(warmup_key, num_warmup)
        state, _ = jax.lax.scan(one_step, state, burn_keys)

        # positions: {"W": (num_samples, *param_shape)}, stats: each (num_samples,)
        s_keys = jax.random.split(sample_key, num_samples)
        _, (positions, stats) = jax.lax.scan(one_step, state, s_keys)
        return positions, stats

    positions, sampler_stats = jax.vmap(run_one_chain)(chain_keys)

    if pbar is not None:
        with contextlib.suppress(Exception):
            pbar.update(1)
            pbar.close()

    return positions, sampler_stats


def _make_pbar(show_progress: bool, *, total: int, desc: str):
    """A tqdm bar, or None when progress is off or tqdm is not installed."""
    if not show_progress:
        return None
    try:
        from tqdm.auto import tqdm

        return tqdm(total=total, desc=desc, leave=False)
    except Exception:
        return None
