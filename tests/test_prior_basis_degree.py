"""Tests that Prior accepts only an int basis_degree of 0 or more."""

import pytest

from psyphy.model import Prior


@pytest.mark.parametrize("bad_degree", [None, -1, 2.5, "4", True])
def test_prior_rejects_invalid_basis_degree(bad_degree):
    # a Prior that builds with basis_degree=None would only raise at first use
    with pytest.raises(ValueError, match=r"basis_degree must be an int of 0 or more"):
        Prior(basis_degree=bad_degree)


def test_prior_accepts_degree_zero():
    assert Prior(basis_degree=0).basis_degree == 0
