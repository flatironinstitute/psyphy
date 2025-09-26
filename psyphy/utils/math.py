"""
math.py
-------

Math utilities for psyphy.

Includes:
- chebyshev_basis : compute Chebyshev polynomial basis.
- mahalanobis_distance : discriminability metric used in WPPM MVP.
- rbf_kernel : kernel function, useful in Full WPPM mode covariance priors.

All functions use JAX (jax.numpy) for compatibility with autodiff.

Notes
-----
- math.chebyshev_basis is relevant when implementing Full WPPM mode,
  where covariance fields are expressed in a basis expansion.
- math.mahalanobis_distance is directly used in WPPM MVP discriminability.
- math.rbf_kernel is a placeholder for Gaussian-process-style covariance priors.

Examples
--------
>>> import jax.numpy as jnp
>>> from psyphy.utils import math
>>> x = jnp.linspace(-1, 1, 5)
>>> math.chebyshev_basis(x, degree=3).shape
(5, 4)
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import lax


def chebyshev_basis(x: jnp.ndarray, degree: int) -> jnp.ndarray:
    """
    Construct the Chebyshev polynomial basis matrix T_0..T_degree evaluated at x.

    Parameters
    ----------
    x : jnp.ndarray
        Input points of shape (N,). For best numerical properties, values should lie in [-1, 1].
    degree : int
        Maximum polynomial degree (>= 0). The output includes columns for T_0 through T_degree.

    Returns
    -------
    jnp.ndarray
        Array of shape (N, degree + 1) where column j contains T_j(x).

    Raises
    ------
    ValueError
        If `degree` is negative or `x` is not 1-D.

    Notes
    -----
    Uses the three-term recurrence:
        T_0(x) = 1
        T_1(x) = x
        T_{n+1}(x) = 2 x T_n(x) - T_{n-1}(x)
    The Chebyshev polynomials are orthogonal on [-1, 1] with weight (1 / sqrt(1 - x^2)).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> x = jnp.linspace(-1, 1, 5)
    >>> B = chebyshev_basis(x, degree=3)  # columns: T0, T1, T2, T3
    """
    if degree < 0:
        raise ValueError("degree must be >= 0")
    if x.ndim != 1:
        raise ValueError("x must be 1-D (shape (N,))")

    # Ensure a floating dtype (Chebyshev recurrences are polynomial in x)
    x = x.astype(jnp.result_type(x, 0.0))

    N = x.shape[0]

    # Handle small degrees explicitly.
    if degree == 0:
        return jnp.ones((N, 1), dtype=x.dtype)
    if degree == 1:
        return jnp.stack([jnp.ones_like(x), x], axis=1)

    # Initialize T0 and T1 columns.
    T0 = jnp.ones_like(x)
    T1 = x

    # Scan to generate T2..T_degree in a JIT-friendly way (avoids Python-side loops).
    def step(carry, _):
        # compute next Chebyshev polynomial
        Tm1, Tm = carry
        Tnext = 2.0 * x * Tm - Tm1
        return (Tm, Tnext), Tnext # new carry, plus an output to collect

    # Jax friendly loop
    (final_Tm1_ignored, final_Tm_ignored), Ts = lax.scan(step, (T0, T1), xs=None, length=degree - 1)
    # Ts has shape (degree-1, N) and holds [T2, T3, ..., T_degree]
    B = jnp.concatenate(
        [T0[:, None], T1[:, None], jnp.swapaxes(Ts, 0, 1)],
        axis=1
    )
    return B



def mahalanobis_distance(x: jnp.ndarray, mean: jnp.ndarray, cov_inv: jnp.ndarray) -> jnp.ndarray:
    """
    Compute squared Mahalanobis distance between x and mean.

    Parameters
    ----------
    x : jnp.ndarray
        Data vector, shape (D,).
    mean : jnp.ndarray
        Mean vector, shape (D,).
    cov_inv : jnp.ndarray
        Inverse covariance matrix, shape (D, D).

    Returns
    -------
    jnp.ndarray
        Scalar squared Mahalanobis distance.

    Notes
    -----
    - Formula: d^2 = (x - mean)^T Σ^{-1} (x - mean)
    - Used in WPPM discriminability calculations.
    """
    delta = x - mean
    return jnp.dot(delta, cov_inv @ delta)


def rbf_kernel(x1: jnp.ndarray, x2: jnp.ndarray, lengthscale: float = 1.0) -> jnp.ndarray:
    """
    Radial Basis Function (RBF) kernel between two sets of points.
    

    Parameters
    ----------
    x1 : jnp.ndarray
        First set of points, shape (N, D).
    x2 : jnp.ndarray
        Second set of points, shape (M, D).
    lengthscale : float, default=1.0
        Length-scale parameter controlling smoothness.

    Returns
    -------
    jnp.ndarray
        Kernel matrix of shape (N, M).

    Notes
    -----
    - RBF kernel: k(x, x') = exp(-||x - x'||^2 / (2 * lengthscale^2))
    - Default used for Gaussian processes for smooth covariance priors in Full WPPM mode.
    """
    sqdist = jnp.sum((x1[:, None, :] - x2[None, :, :])**2, axis=-1)
    return jnp.exp(-0.5 * sqdist / (lengthscale**2))
