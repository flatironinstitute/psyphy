# Covariance Fields

A **covariance field** represents spatially-varying perceptual uncertainty:
it assigns a covariance matrix Σ(x) to each location x in stimulus space,
describing how perceptual noise varies across the sensory domain.

--> a covariance field represents a function from stimulus locations to covariance
matrices, encapsulating the spatially-varying perceptual uncertainty in WPPM.

## Mathematical Background:
In the Wishart Process Psychophysical Model:
    Σ(x) = U(x) @ U(x)^T + λI

where:
    U(x) = Σ_ij W_ij * φ_ij(x)  (basis expansion)
    φ_ij(x) are Chebyshev basis functions
    W_ij are learned coefficients
    λ is a numerical stabilizer (diag_term)

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


## Technical Notes

- Works with JAX vmap for efficient batch operations
- Handles both MVP (diagonal) and Wishart (spatially-varying) modes
- Positive definiteness guaranteed by construction (Σ = U @ U^T + λI)
- Tolerant of numerical issues in optimization (test handles NaN gracefully)
