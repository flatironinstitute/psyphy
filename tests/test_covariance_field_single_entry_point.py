"""
Tests for single entry point to covariance field API with automatic batch handling.

This test suite validates the single entry point design where field(x)
automatically dispatches based on input shape:
- Single point: x.ndim == 1 -> direct evaluation
- Any batch: x.ndim >= 2 -> flatten -> vmap -> reshape

Key convention: Last axis must be input_dim.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from psyphy.model import WPPM, OddityTask, Prior
from psyphy.model.covariance_field import WPPMCovarianceField

# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def wishart_field():
    """Create Wishart mode covariance field for testing."""
    model = WPPM(
        input_dim=2,
        prior=Prior(input_dim=2, basis_degree=3, extra_embedding_dims=1),
        task=OddityTask(),
        extra_dims=1,
    )
    key = jr.PRNGKey(123)
    return WPPMCovarianceField.from_prior(model, key)


@pytest.fixture
def field_3d():
    """Create 3D covariance field for testing."""
    model = WPPM(
        input_dim=3,
        prior=Prior(input_dim=3, basis_degree=2),
        task=OddityTask(),
    )
    key = jr.PRNGKey(456)
    return WPPMCovarianceField.from_prior(model, key)


# ==============================================================================
# Test 1: Single Point Dispatch
# ==============================================================================


class TestSinglePointDispatch:
    """Test that field(x) works for single points."""

    def test_single_point_wishart(self, wishart_field):
        """field(x) evaluates single point in Wishart mode."""
        x = jnp.array([0.7, 0.2])

        Sigma = wishart_field(x)

        assert Sigma.shape == (2, 2)
        eigvals = jnp.linalg.eigvalsh(Sigma)
        assert jnp.all(eigvals > 0)

    def test_single_point_3d(self, field_3d):
        """field(x) works for 3D input."""
        x = jnp.array([0.5, 0.5, 0.5])

        Sigma = field_3d(x)

        assert Sigma.shape == (3, 3)
        eigvals = jnp.linalg.eigvalsh(Sigma)
        assert jnp.all(eigvals > 0)


# ==============================================================================
# Test 2: 1D Batch Dispatch
# ==============================================================================


class TestOneDimensionalBatch:
    """Test field(X) for 1D batches (n_points, input_dim)."""

    def test_1d_batch_wishart(self, wishart_field):
        """field(X) handles 1D batch in Wishart mode."""
        X = jnp.array([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]])

        Sigmas = wishart_field(X)

        assert Sigmas.shape == (3, 2, 2)
        # Should vary spatially (Wishart)
        assert not jnp.allclose(Sigmas[0], Sigmas[1], atol=1e-6)
        assert not jnp.allclose(Sigmas[1], Sigmas[2], atol=1e-6)


# ==============================================================================
# Test 3: 2D Grid Dispatch
# ==============================================================================


class TestTwoDimensionalGrid:
    """Test field(X) for 2D grids (h, w, input_dim)."""

    def test_2d_grid_wishart(self, wishart_field):
        """field(X_grid) handles 2D grid for visualization."""
        # Create 10x10 grid
        x_vals = jnp.linspace(0, 1, 10)
        y_vals = jnp.linspace(0, 1, 10)
        X_grid = jnp.stack(jnp.meshgrid(x_vals, y_vals, indexing="ij"), axis=-1)

        Sigmas = wishart_field(X_grid)

        assert Sigmas.shape == (10, 10, 2, 2)
        # Check a few points are positive definite
        assert jnp.all(jnp.linalg.eigvalsh(Sigmas[0, 0]) > 0)
        assert jnp.all(jnp.linalg.eigvalsh(Sigmas[5, 5]) > 0)
        assert jnp.all(jnp.linalg.eigvalsh(Sigmas[9, 9]) > 0)

    def test_2d_grid_large(self, wishart_field):
        """field(X) handles large grids (50x50)."""
        X_grid = jnp.ones((50, 50, 2)) * 0.5

        Sigmas = wishart_field(X_grid)

        assert Sigmas.shape == (50, 50, 2, 2)


# ==============================================================================
# Test 4: 3D and Higher Batch Dimensions
# ==============================================================================


class TestHigherDimensionalBatches:
    """Test field(X) for 3D+ batches."""

    def test_3d_batch(self, wishart_field):
        """field(X) handles 3D batches (e.g., temporal × grid)."""
        # Shape: (time, height, width, input_dim)
        X_3d = jnp.ones((5, 10, 10, 2)) * 0.5

        Sigmas = wishart_field(X_3d)

        assert Sigmas.shape == (5, 10, 10, 2, 2)

    def test_4d_batch(self, wishart_field):
        """field(X) handles 4D batches."""
        X_4d = jnp.ones((2, 3, 5, 5, 2)) * 0.5

        Sigmas = wishart_field(X_4d)

        assert Sigmas.shape == (2, 3, 5, 5, 2, 2)

    def test_3d_input_space(self, field_3d):
        """field(X) works with 3D input space."""
        X = jnp.ones((10, 3)) * 0.5

        Sigmas = field_3d(X)

        assert Sigmas.shape == (10, 3, 3)


# ==============================================================================
# Test 5: Parametrized Arbitrary Batch Dimensions
# ==============================================================================


class TestArbitraryBatchDimensions:
    """Parametrized tests for various batch structures."""

    @pytest.mark.parametrize(
        "batch_shape",
        [
            (),  # Single point (special case)
            (10,),  # 1D batch
            (5, 5),  # 2D grid
            (3, 5, 5),  # 3D
            (2, 3, 5, 5),  # 4D
            (100,),  # Large 1D
            (50, 50),  # Large 2D
        ],
    )
    def test_arbitrary_batch_shapes(self, wishart_field, batch_shape):
        """field(x) handles arbitrary batch dimensions."""
        input_dim = 2

        if len(batch_shape) == 0:
            # Single point
            input_shape = (input_dim,)
            expected_output_shape = (input_dim, input_dim)
        else:
            # Batch
            input_shape = batch_shape + (input_dim,)
            expected_output_shape = batch_shape + (input_dim, input_dim)

        X = jnp.ones(input_shape) * 0.5
        Sigmas = wishart_field(X)

        assert Sigmas.shape == expected_output_shape


# ==============================================================================
# Test 6: Input Validation
# ==============================================================================


class TestInputValidation:
    """Test that field(x) validates inputs properly."""

    def test_wrong_input_dim_raises_error(self, wishart_field):
        """field(x) raises clear error for wrong input_dim."""
        # Model expects input_dim=2, give it 3
        x_wrong = jnp.ones(3)

        with pytest.raises(ValueError, match="Last axis must be input_dim=2"):
            wishart_field(x_wrong)

    def test_wrong_input_dim_batch_raises_error(self, wishart_field):
        """field(X) raises error for wrong input_dim in batch."""
        X_wrong = jnp.ones((10, 3))  # Should be (10, 2)

        with pytest.raises(ValueError, match="Last axis must be input_dim=2"):
            wishart_field(X_wrong)

    def test_3d_field_validates_input_dim(self, field_3d):
        """3D field validates input dimension."""
        x_wrong = jnp.ones(2)  # Should be 3

        with pytest.raises(ValueError, match="Last axis must be input_dim=3"):
            field_3d(x_wrong)


# ==============================================================================
# Test 7: Equivalence to Manual vmap
# ==============================================================================


class TestEquivalenceToVmap:
    """Test that field(X) gives same result as manual vmap."""

    def test_1d_batch_equivalent_to_vmap(self, wishart_field):
        """field(X) equivalent to jax.vmap(field)(X) for 1D batch."""
        X = jnp.array([[0.1, 0.2], [0.5, 0.5], [0.9, 0.8]])

        Sigmas_direct = wishart_field(X)
        Sigmas_vmap = jax.vmap(wishart_field)(X)

        assert jnp.allclose(Sigmas_direct, Sigmas_vmap, rtol=5e-3, atol=1e-3)

    def test_2d_grid_equivalent_to_nested_vmap(self, wishart_field):
        """field(X_grid) equivalent to nested vmap for 2D grid."""
        X_grid = jnp.ones((5, 5, 2)) * 0.5

        Sigmas_direct = wishart_field(X_grid)

        # Manual nested vmap
        vmap_inner = jax.vmap(wishart_field)
        vmap_outer = jax.vmap(vmap_inner)
        Sigmas_nested = vmap_outer(X_grid)

        assert jnp.allclose(Sigmas_direct, Sigmas_nested, rtol=5e-4, atol=1e-4)


# ==============================================================================
# Test 8: JIT Compatibility
# ==============================================================================


class TestJITCompatibility:
    """Test that field(X) is JIT-compatible."""

    def test_single_point_jit_compatible(self, wishart_field):
        """field(x) works under JIT for single point."""
        jitted_field = jax.jit(wishart_field)
        x = jnp.array([0.5, 0.3])

        Sigma_eager = wishart_field(x)
        Sigma_jitted = jitted_field(x)

        assert jnp.allclose(Sigma_eager, Sigma_jitted)

    def test_batch_jit_compatible(self, wishart_field):
        """field(X) is JIT-compatible for batches."""
        jitted_field = jax.jit(wishart_field)
        X = jnp.ones((10, 2)) * 0.5

        Sigmas_eager = wishart_field(X)
        Sigmas_jitted = jitted_field(X)

        assert jnp.allclose(Sigmas_eager, Sigmas_jitted)

    def test_2d_grid_jit_compatible(self, wishart_field):
        """field(X_grid) is JIT-compatible for 2D grids."""
        jitted_field = jax.jit(wishart_field)
        X_grid = jnp.ones((10, 10, 2)) * 0.5

        Sigmas_eager = wishart_field(X_grid)
        Sigmas_jitted = jitted_field(X_grid)

        assert jnp.allclose(Sigmas_eager, Sigmas_jitted)


# ==============================================================================
# Test 9: Edge Cases
# ==============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_element_batch(self, wishart_field):
        """field(X) handles batch of size 1."""
        X = jnp.array([[0.5, 0.5]])  # Shape (1, 2)

        Sigmas = wishart_field(X)

        assert Sigmas.shape == (1, 2, 2)

    @pytest.mark.skip(reason="Wishart process only supports input_dim >= 2")
    def test_1d_input_space_single_point(self):
        """For input_dim=1, single point has shape (1,)."""
        model = WPPM(
            input_dim=1,
            prior=Prior(input_dim=1, basis_degree=2),
            task=OddityTask(),
        )
        field = WPPMCovarianceField.from_prior(model, jr.PRNGKey(789))

        x = jnp.array([0.5])  # Shape (1,) - single point

        Sigma = field(x)

        assert Sigma.shape == (1, 1)

    @pytest.mark.skip(reason="Wishart process only supports input_dim >= 2")
    def test_1d_input_space_batch(self):
        """For input_dim=1, batch requires shape (n, 1)."""
        model = WPPM(
            input_dim=1,
            prior=Prior(input_dim=1, basis_degree=2),
            task=OddityTask(),
        )
        field = WPPMCovarianceField.from_prior(model, jr.PRNGKey(789))

        X = jnp.array([[0.1], [0.5], [0.9]])  # Shape (3, 1) - batch

        Sigmas = field(X)

        assert Sigmas.shape == (3, 1, 1)


# ==============================================================================
# Test 10: Backward Compatibility (Deprecation Warnings)
# ==============================================================================


class TestBackwardCompatibility:
    """Test that old API still works with deprecation warnings."""

    def test_cov_deprecated(self, wishart_field):
        """cov() still works but warns."""
        x = jnp.array([0.5, 0.3])

        with pytest.warns(DeprecationWarning, match="Use field\\(x\\)"):
            Sigma = wishart_field.cov(x)

        assert Sigma.shape == (2, 2)

    def test_cov_batch_deprecated(self, wishart_field):
        """cov_batch() still works but warns."""
        X = jnp.array([[0.1, 0.2], [0.5, 0.5]])

        with pytest.warns(DeprecationWarning, match="Use field\\(X\\)"):
            Sigmas = wishart_field.cov_batch(X)

        assert Sigmas.shape == (2, 2, 2)

    def test_cov_rejects_batch(self, wishart_field):
        """cov() raises error for batch input."""
        X = jnp.array([[0.1, 0.2], [0.5, 0.5]])

        with (
            pytest.warns(DeprecationWarning),
            pytest.raises(ValueError, match="only accepts single points"),
        ):
            wishart_field.cov(X)

    def test_deprecated_equivalent_to_new(self, wishart_field):
        """Old API gives same results as new API."""
        x = jnp.array([0.5, 0.3])
        X = jnp.array([[0.1, 0.2], [0.5, 0.5]])

        with pytest.warns(DeprecationWarning):
            Sigma_old = wishart_field.cov(x)
        Sigma_new = wishart_field(x)
        assert jnp.allclose(Sigma_old, Sigma_new)

        with pytest.warns(DeprecationWarning):
            Sigmas_old = wishart_field.cov_batch(X)
        Sigmas_new = wishart_field(X)

        assert jnp.allclose(Sigmas_old, Sigmas_new)
