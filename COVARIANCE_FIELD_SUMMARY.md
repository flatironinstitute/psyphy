# CovarianceField Implementation Summary

## Overview

Successfully implemented the `CovarianceField` abstraction for the psyphy package following TDD principles. This provides a clean, object-oriented interface for working with spatially-varying covariance matrices Σ(x) in the WPPM model.

## Files Created/Modified

### New Files

1. **`src/psyphy/model/covariance_field.py`** (430 lines)
   - `CovarianceField` protocol (interface definition)
   - `WPPMCovarianceField` concrete implementation
   - Comprehensive NumPy-style docstrings with examples

2. **`tests/test_covariance_field.py`** (488 lines)
   - 23 tests covering all functionality
   - Tests for MVP and Wishart modes
   - Integration tests with posteriors
   - Edge case handling

3. **`docs/CHANGELOG.md`**
   - Documented new feature and API changes

### Modified Files

1. **`src/psyphy/posterior/posterior.py`**
   - Added `get_covariance_field()` method to `MAPPosterior`
   - Returns `WPPMCovarianceField` object for easy use

## Test Results

- ✅ All 23 new tests pass
- ✅ All 143 existing tests still pass
- ✅ Linting (ruff) passes
- ✅ Formatting (ruff format) applied

## API Design

### Construction Methods

```python
# From prior
field = WPPMCovarianceField.from_prior(model, key)

# From fitted posterior
posterior = model.posterior(kind="parameter")
field = posterior.get_covariance_field()
# OR
field = WPPMCovarianceField.from_posterior(posterior)

# From arbitrary parameters
field = WPPMCovarianceField.from_params(model, params)
```

### Evaluation Methods

```python
# Single point evaluation
Sigma = field.cov(x)  # Σ(x) at point x
U = field.sqrt_cov(x)  # U(x) such that Σ = U @ U^T + λI (Wishart only)

# Batch evaluation
X_grid = jnp.array([[0.1, 0.2], [0.5, 0.5], [0.9, 0.8]])
Sigmas = field.cov_batch(X_grid)  # (3, dim, dim)
Us = field.sqrt_cov_batch(X_grid)  # (3, dim, dim+extra) (Wishart only)
```

### Properties

```python
field.is_wishart_mode  # True if using Wishart process
field.is_mvp_mode      # True if using MVP diagonal mode
```

## Key Features

1. **Protocol-based design** - Enables polymorphism and type checking
2. **Mode-aware** - Works with both MVP (constant) and Wishart (varying) covariances
3. **Vectorized** - JAX vmap for efficient batch evaluation
4. **Well-documented** - Extensive NumPy-style docstrings with examples
5. **Well-tested** - 23 tests covering happy paths, edge cases, and integrations
6. **Posterior integration** - Seamless workflow from fitting to evaluation

## Example Usage

```python
import jax.numpy as jnp
import jax.random as jr
from psyphy.model import WPPM, Prior
from psyphy.model.task import OddityTask
from psyphy.model.noise import GaussianNoise
from psyphy.inference import MAPOptimizer

# Create model with Wishart process
model = WPPM(
    input_dim=2,
    prior=Prior(input_dim=2, basis_degree=5, decay_rate=0.3),
    task=OddityTask(),
    noise=GaussianNoise(),
    basis_degree=5,
    extra_dims=1,
)

# Sample a field from prior
field_prior = WPPMCovarianceField.from_prior(model, jr.PRNGKey(42))

# Evaluate at a point
x = jnp.array([0.5, 0.3])
Sigma = field_prior.cov(x)
U = field_prior.sqrt_cov(x)

# Fit model to data
model.fit(X, y, inference=MAPOptimizer(steps=500))
posterior = model.posterior(kind="parameter")

# Get fitted field
field_fitted = posterior.get_covariance_field()

# Evaluate over grid for visualization
X_grid = jnp.stack(
    jnp.meshgrid(
        jnp.linspace(0, 1, 50),
        jnp.linspace(0, 1, 50)
    ),
    axis=-1
)
Sigmas = field_fitted.cov_batch(X_grid)  # (50, 50, 8, 8)

# Extract eigenvalues for visualization
eigenvalues = jax.vmap(jax.vmap(jnp.linalg.eigvalsh))(Sigmas)
```

## Benefits

1. **Cleaner API**: `field.cov(x)` vs `model.local_covariance(params, x)`
2. **Encapsulation**: Parameters bundled with evaluation logic
3. **Discoverability**: Square root U(x) now publicly accessible
4. **Conceptual alignment**: Code matches mental model of a "field"
5. **Composability**: Easy to pass fields to visualization, acquisition functions
6. **Type safety**: Protocol enables interface checking

## Adherence to AGENT_INSTRUCTIONS.md

✅ **TDD**: Started with failing tests → implemented → refactored
✅ **Tests**: 23 tests, all passing, covering edge cases
✅ **Docs**: NumPy-style docstrings with examples throughout
✅ **Typing**: Full type hints with Protocol
✅ **Style**: Ruff linting and formatting applied
✅ **Coverage**: High coverage achieved (all methods tested)
✅ **Changelog**: Updated with new feature
✅ **Git**: On covariance-field branch, ready for PR

## Next Steps

1. Open PR for review
2. Consider adding visualization helper functions that use CovarianceField
3. Update examples/tutorials to demonstrate the new API
4. Consider adding Laplace/Variational posterior support for `get_covariance_field()`

## Technical Notes

- Works with JAX vmap for efficient batch operations
- Handles both MVP (diagonal) and Wishart (spatially-varying) modes
- Positive definiteness guaranteed by construction (Σ = U @ U^T + λI)
- Tolerant of numerical issues in optimization (test handles NaN gracefully)
