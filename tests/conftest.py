"""
Central pytest configuration for this project.

This file is automatically discovered by pytest and is intended for:

- **Fixtures**: reusable objects or setup logic shared across multiple test files.
- **Pytest hooks**: project-wide customizations of pytest behavior (e.g., modifying CLI options, adding markers).

Notes
-----
- Contributors should 
  install the package in editable mode (`pip install -e .`) so that imports are resolved
  consistently in local dev and CI environments.
- Keep this file focused on test setup. Do not add application logic here.
"""



import jax.numpy as jnp
import pytest


@pytest.fixture
def chebyshev_inputs():
    """Return evenly spaced test points in [-1, 1] for basis function tests."""
    return jnp.linspace(-1, 1, 5)


@pytest.fixture
def default_degree():
    """Default polynomial degree for testing."""
    return 3
