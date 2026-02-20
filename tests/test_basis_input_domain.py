"""test_basis_input_domain.py

Tests for basis-domain validation in WPPM.

Contract (current):
- In Wishart mode, WPPM expects stimuli in the Chebyshev domain [-1, 1].
- The model does not normalize inputs; it validates and raises.

This keeps the library honest and makes it easier to swap in other basis
families later (each basis can define its own expected domain).
"""

import jax.numpy as jnp
import pytest

from psyphy.model import WPPM, Prior
from psyphy.model.noise import GaussianNoise
from psyphy.model.task import OddityTask


@pytest.mark.parametrize("input_dim", [2, 3])
def test_evaluate_basis_at_point_accepts_chebyshev_domain(input_dim: int) -> None:
    model = WPPM(
        input_dim=input_dim,
        prior=Prior(input_dim=input_dim, basis_degree=3),
        task=OddityTask(),
        noise=GaussianNoise(),
    )

    x = jnp.linspace(-1.0, 1.0, input_dim)
    phi = model._evaluate_basis_at_point(x)

    d = int(model.basis_degree)  # type: ignore[arg-type]
    if input_dim == 2:
        assert phi.shape == (d + 1, d + 1)
    else:
        assert phi.shape == (d + 1, d + 1, d + 1)


@pytest.mark.parametrize("bad_x", [jnp.array([1.1, 0.0]), jnp.array([-1.2, 0.0])])
def test_evaluate_basis_at_point_rejects_out_of_range_2d(bad_x: jnp.ndarray) -> None:
    model = WPPM(
        input_dim=2,
        prior=Prior(input_dim=2, basis_degree=2),
        task=OddityTask(),
        noise=GaussianNoise(),
    )

    with pytest.raises(ValueError, match=r"Chebyshev domain \[-1, 1\]"):
        model._validate_basis_input(bad_x)


def test_evaluate_basis_at_point_rejects_wrong_shape() -> None:
    model = WPPM(
        input_dim=2,
        prior=Prior(input_dim=2, basis_degree=2),
        task=OddityTask(),
        noise=GaussianNoise(),
    )

    with pytest.raises(ValueError, match=r"Expected x with shape \(2,\),"):
        _ = model._evaluate_basis_at_point(jnp.ones((3,)))
