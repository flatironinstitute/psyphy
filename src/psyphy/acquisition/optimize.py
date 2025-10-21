"""
optimize.py
-----------

Optimization utilities for acquisition functions.

Provides functional interface for maximizing acquisition functions:
- optimize_acqf_discrete: Exhaustive search over candidate set
- optimize_acqf: Gradient-based optimization (continuous)
- optimize_acqf_random: Random search baseline

Design
------
Following BoTorch's functional API:
    X_next, acq_value = optimize_acqf(acq_fn, bounds, q=1)

But adapted for psyphy:
- Support for both continuous and discrete optimization
- JAX-based gradient descent (Optax)
- Batch acquisition (q > 1) support
"""

from __future__ import annotations

from typing import Any, Callable, Literal

import jax.numpy as jnp
import jax.random as jr


def optimize_acqf_discrete(
    acq_fn: Callable[[jnp.ndarray], jnp.ndarray],
    candidates: jnp.ndarray,
    q: int = 1,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Optimize acquisition function over discrete candidate set.

    This is the simplest and most common approach for psychophysics,
    where stimulus space is often discretized.

    Parameters
    ----------
    acq_fn : callable
        Acquisition function. Takes (n_candidates, input_dim) array,
        returns (n_candidates,) scores.
    candidates : jnp.ndarray, shape (n_candidates, input_dim)
        Discrete candidate points to evaluate
    q : int, default=1
        Batch size (number of points to select)

    Returns
    -------
    X_next : jnp.ndarray, shape (q, input_dim)
        Selected candidate points
    acq_values : jnp.ndarray, shape (q,)
        Acquisition values of selected points

    Examples
    --------
    >>> # Simple discrete optimization
    >>> candidates = jnp.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
    >>> def acq_fn(X):
    ...     return expected_improvement(model.posterior(X), best_f=0.5)
    >>>
    >>> X_next, acq_val = optimize_acqf_discrete(acq_fn, candidates, q=1)

    >>> # Batch acquisition
    >>> X_batch, acq_vals = optimize_acqf_discrete(acq_fn, candidates, q=3)
    """
    # Evaluate all candidates
    acq_values = acq_fn(candidates)

    # Select top-q by acquisition value
    top_indices = jnp.argsort(acq_values)[-q:][::-1]  # Descending order
    X_next = candidates[top_indices]
    selected_values = acq_values[top_indices]

    return X_next, selected_values


def optimize_acqf(
    acq_fn: Callable[[jnp.ndarray], jnp.ndarray],
    bounds: jnp.ndarray,
    q: int = 1,
    *,
    method: Literal["gradient", "random"] = "gradient",
    num_restarts: int = 10,
    raw_samples: int = 100,
    optim_steps: int = 100,
    lr: float = 0.01,
    key: Any = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Optimize acquisition function over continuous domain.

    Uses multi-start gradient descent to find global optimum.

    Parameters
    ----------
    acq_fn : callable
        Acquisition function. Takes (n_points, input_dim) array,
        returns (n_points,) scores.
    bounds : jnp.ndarray, shape (input_dim, 2)
        Box constraints [[x1_min, x1_max], [x2_min, x2_max], ...]
    q : int, default=1
        Batch size (number of points to select)
    method : {"gradient", "random"}, default="gradient"
        Optimization method
    num_restarts : int, default=10
        Number of random restarts for gradient descent
    raw_samples : int, default=100
        Number of random samples to initialize restarts
    optim_steps : int, default=100
        Number of optimization steps per restart
    lr : float, default=0.01
        Learning rate for gradient descent
    key : jax.random.KeyArray | None
        PRNG key for random initialization

    Returns
    -------
    X_next : jnp.ndarray, shape (q, input_dim)
        Optimized points
    acq_values : jnp.ndarray, shape (q,)
        Acquisition values at X_next

    Examples
    --------
    >>> # Simple continuous optimization
    >>> def acq_fn(X):
    ...     return expected_improvement(model.posterior(X), best_f=0.5)
    >>>
    >>> bounds = jnp.array([[0.0, 1.0], [0.0, 1.0]])  # 2D unit square
    >>> X_next, acq_val = optimize_acqf(acq_fn, bounds, q=1, method="gradient")

    >>> # Batch acquisition with multiple restarts
    >>> X_batch, acq_vals = optimize_acqf(
    ...     acq_fn,
    ...     bounds,
    ...     q=3,
    ...     num_restarts=20,
    ...     optim_steps=200,
    ... )

    Notes
    -----
    For gradient-based optimization, ensure your acquisition function
    is differentiable through JAX. Use jax.grad() or jax.value_and_grad().

    For non-differentiable acquisition functions, use method="random"
    or optimize_acqf_discrete() with a candidate grid.
    """
    if key is None:
        key = jr.PRNGKey(0)

    if method == "random":
        return optimize_acqf_random(
            acq_fn, bounds, q=q, num_samples=raw_samples, key=key
        )
    elif method == "gradient":
        return _optimize_acqf_gradient(
            acq_fn,
            bounds,
            q=q,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            optim_steps=optim_steps,
            lr=lr,
            key=key,
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'gradient' or 'random'.")


def optimize_acqf_random(
    acq_fn: Callable[[jnp.ndarray], jnp.ndarray],
    bounds: jnp.ndarray,
    q: int = 1,
    *,
    num_samples: int = 1000,
    key: Any = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Optimize acquisition function via random search.

    Simple baseline: sample random points, evaluate, select best.

    Parameters
    ----------
    acq_fn : callable
        Acquisition function
    bounds : jnp.ndarray, shape (input_dim, 2)
        Box constraints
    q : int, default=1
        Batch size
    num_samples : int, default=1000
        Number of random samples to evaluate
    key : jax.random.KeyArray | None
        PRNG key

    Returns
    -------
    X_next : jnp.ndarray, shape (q, input_dim)
        Best random samples
    acq_values : jnp.ndarray, shape (q,)
        Acquisition values

    Examples
    --------
    >>> X_next, acq_val = optimize_acqf_random(acq_fn, bounds, q=1, num_samples=5000)
    """
    if key is None:
        key = jr.PRNGKey(0)

    # Generate random samples in bounds
    input_dim = bounds.shape[0]
    key, subkey = jr.split(key)
    random_samples = jr.uniform(subkey, (num_samples, input_dim))

    # Scale to bounds
    lower = bounds[:, 0]
    upper = bounds[:, 1]
    random_samples = lower + random_samples * (upper - lower)

    # Evaluate and select best
    return optimize_acqf_discrete(acq_fn, random_samples, q=q)


def _optimize_acqf_gradient(
    acq_fn: Callable[[jnp.ndarray], jnp.ndarray],
    bounds: jnp.ndarray,
    q: int,
    num_restarts: int,
    raw_samples: int,
    optim_steps: int,
    lr: float,
    key: Any,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Internal: Gradient-based optimization with multiple restarts.

    Uses JAX autodiff + simple gradient ascent.
    For production, could integrate Optax for adaptive learning rates.
    """
    import jax

    if key is None:
        key = jr.PRNGKey(0)

    input_dim = bounds.shape[0]
    lower = bounds[:, 0]
    upper = bounds[:, 1]

    # Initialize candidates from random samples
    key, subkey = jr.split(key)
    init_samples = jr.uniform(subkey, (raw_samples, input_dim))
    init_samples = lower + init_samples * (upper - lower)

    # Evaluate initial samples
    init_acq = acq_fn(init_samples)
    best_init_indices = jnp.argsort(init_acq)[-num_restarts:]

    # Run gradient ascent from best initializations
    best_X = None
    best_acq_value = -jnp.inf

    # Negative acquisition for minimization (gradient descent)
    def loss_fn(X):
        return -jnp.sum(acq_fn(X))  # Negative for descent

    grad_fn = jax.grad(loss_fn)

    for restart_idx in best_init_indices:
        X_current = init_samples[restart_idx : restart_idx + 1]  # Shape: (1, input_dim)

        # Gradient descent
        for _ in range(optim_steps):
            grad = grad_fn(X_current)
            X_current = X_current - lr * grad

            # Project to bounds
            X_current = jnp.clip(X_current, lower, upper)

        # Evaluate final
        final_acq = acq_fn(X_current)

        if final_acq[0] > best_acq_value:
            best_acq_value = float(final_acq[0])  # Convert to Python float
            best_X = X_current

    # For q > 1, run q independent optimizations
    if q > 1:
        # Simple approach: take top-q from all restarts
        # (Better approach: sequential greedy or batch acquisition)
        all_X = []
        all_acq = []

        for restart_idx in best_init_indices[:q]:
            X_current = init_samples[restart_idx : restart_idx + 1]

            for _ in range(optim_steps):
                grad = grad_fn(X_current)
                X_current = X_current - lr * grad
                X_current = jnp.clip(X_current, lower, upper)

            all_X.append(X_current)
            all_acq.append(acq_fn(X_current))

        X_next = jnp.concatenate(all_X, axis=0)
        acq_values = jnp.concatenate(all_acq, axis=0)

        return X_next, acq_values
    else:
        assert best_X is not None, "Optimization failed to find valid solution"
        return best_X, jnp.array([best_acq_value])
