# WPPM API Design Document

**Last updated**: January 13, 2026
**Purpose**: Documentation of API design choices, implementation details, and architectural decisions for the Wishart Process Psychophysical Model (WPPM)

---

## Table of Contents

1. [Overview](#overview)
2. [Design Philosophy](#design-philosophy)
3. [Core Abstractions](#core-abstractions)
4. [Parameter Lifecycle](#parameter-lifecycle)
5. [Code Organization](#code-organization)

---

## Overview

### What is Wishart Process Psychophysical Model (WPPM)?

The **Wishart Process Psychophysical Model (WPPM)** is a Bayesian model for perceptual discrimination that represents spatially-varying perceptual thresholds as ellipsoids across stimulus space.

**Key components**:
- **Model**: WPPM class defining forward model and likelihood
- **Prior**: Gaussian prior over coefficient matrices W
- **Covariance Field**: Spatial function $\Sigma(x)$ representing thresholds
- **Inference**: MAP, Laplace, or Langevin MCMC for posterior
- **Acquisition**: Active learning for optimal trial placement


---

## Design Philosophy

### Hybrid OOP + Functional Approach

**Why not pure OOP?**
- Hard to JIT compile stateful objects
- Less composable with JAX transformations
- Inheritance can become unwieldy

**Why not pure functional?**
- Less intuitive API: `evaluate_field(params, model, x)`
- Manual parameter threading everywhere
- Harder for users unfamiliar with functional programming


**Our choice: Hybrid**
```python
# OOP for API clarity
field = WPPMCovarianceField.from_prior(model, key)

# Functional for JAX compatibility
Sigmas = jax.vmap(field)(X_grid)  # field acts as pure function
```

**Benefits**:
- Clean, intuitive API via classes
- JAX transformations work seamlessly
- Best of both paradigms, has scikit learn feel

### Protocol-Based Interfaces

**Pattern**: Define contracts via `Protocol`, not inheritance

```python
@runtime_checkable
class CovarianceField(Protocol):
    def cov(self, x: jnp.ndarray) -> jnp.ndarray: ...
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray: ...
```

**Why?**
-  Structural typing (duck typing with types)
-  No inheritance coupling
-  Easy to add new implementations

### Immutability + Pure Functions

**JAX requirement**: Functions must be pure (no side effects)

**-> Our approach**:
- Parameters stored as immutable JAX arrays
- No in-place modifications
- Return new objects instead of mutating

```python
# good: pure function
def local_covariance(params, x):
    U = compute_sqrt(params, x)
    return U @ U.T + diag_term * I

# bad: Mutation (would break JAX)
def local_covariance(self, x):
    self.cache[x] = compute_sqrt(self.params, x)  # side effect: mutates state of the object and violates JAX's requirement for pure, stateless functions
    return self.cache[x]
```

---

## Core Abstractions

### 1. Model (WPPM)

**Location**: `src/psyphy/model/wppm.py`

**Purpose**: Defines forward model, likelihood, and prediction logic

**Key methods**:
```python
class WPPM(Model):
    def init_params(self, key) -> dict
    def local_covariance(self, params, x) -> jnp.ndarray
    def predict_prob(self, params, stimulus) -> float
    def log_likelihood(self, params, refs, probes, responses) -> float
```

**Design choices**:
- Inherits from `Model` base class (common interface)
- Stateless: all methods take `params` explicitly
- JAX-compatible: pure functions, no hidden state

### 2. Prior

**Location**: `src/psyphy/model/prior.py`

**Purpose**: Defines p(W) and samples initial parameters

**Key methods**:
```python
@dataclass
class Prior:
    def sample_params(self, key) -> dict
    def log_prob(self, params) -> float
```

**Design choices**:
- `@dataclass` for simple parameter container
- Smoothness prior via `decay_rate` hyperparameter where prior distribution for a coefficent degree $d$ is a normal distribution for a with zero mean and variance given by (variance_scale) * (decay_rate^$d$)

### 3. Covariance Field

**Location**: `src/psyphy/model/covariance_field.py`

**Purpose**: Encapsulates model + params for easy covariance field $\Sigma(x)$ evaluation

**Key methods**:
```python
class WPPMCovarianceField:
    @classmethod
    def from_prior(cls, model, key) -> WPPMCovarianceField

    @classmethod
    def from_posterior(cls, posterior) -> WPPMCovarianceField

    @classmethod
    def from_params(cls, model, params) -> WPPMCovarianceField

    def __call__(self, x) -> jnp.ndarray
    def cov(self, x) -> jnp.ndarray
    def sqrt_cov(self, x) -> jnp.ndarray
    def cov_batch(self, X) -> jnp.ndarray
```

**Design choices**:
- Factory methods for different construction paths
- Callable interface: `field(x)` for mathematical elegance
- Protocol compliance for polymorphism
- **Unified Entry Point**: `field(x)` automatically handles both single points and batches (via vmap), so users don't need to switch methods based on input shape.

### 4. Posterior

**Location**: `src/psyphy/posterior/posterior.py`

**Purpose**: Represents p(W | data) after inference

**Key methods**:
```python
class MAPPosterior(BasePosterior):
    @property
    def params(self) -> dict

    def sample(self, n, key) -> dict
    def log_prob(self, params) -> float
    def get_covariance_field(self) -> WPPMCovarianceField
    def predict_prob(self, stimulus) -> float
```

**Design choices**:
- Separate classes for MAP, Laplace, Langevin
- Delegates to model for predictions
- Provides `get_covariance_field()` convenience method

---

## Parameter Lifecycle

### Phase 1: Model Creation (no Parameters yet)

```python
# Define model structure
model = WPPM(
    input_dim=2,
    prior=Prior(basis_degree=4, variance_scale=0.03, decay_rate=0.3),
    task=OddityTask(), # defines likelihood
    basis_degree=4,
    extra_dims=1,
)

# At this point: NO parameters exist!
# Just model specification (hyperparameters)
```

### Phase 2: Parameter Instantiation

**Explicit path**:
```python
key = jr.PRNGKey(12)
params = model.init_params(key)  # <- Parameters created here
```

**Implicit path** (via covariance field):
```python
field = WPPMCovarianceField.from_prior(model, key)
# Internally calls model.init_params(key)
```

**Call chain**:
```
model.init_params(key)        [wppm.py]
  ↓
prior.sample_params(key)                  [prior.py]
  ↓
prior._sample_W_wishart(key)             [prior.py]
  ↓
variances = variance_scale * decay_rate^deg
W = sqrt(variances) * jr.normal(key, (5,5,2,3))  <- Instantiation!
  ↓
return {"W": W}
```

### Phase 3: Parameter Distribution

**Prior sampling** (before data):
```python
W ~ p(W) = Normal(0, Σ_prior)

where Σ_prior[i,j] = variance_scale · (decay_rate)^(i+j)
```

**Posterior optimization** (after data):
```python
# Option 1: High-level API (Façade)
posterior = model.fit(data, inference="map")

# Option 2: Explicit Optimizer API
optimizer = MAPOptimizer(steps=500)
posterior = optimizer.fit(model, data)

# Result:
W_MAP = arg max p(W | data)
```

**Future: Posterior sampling** (uncertainty quantification):
```python
# Option 1: High-level API
posterior = model.fit(data, inference="langevin")

# Option 2: Explicit Optimizer API
optimizer = LangevinOptimizer(steps=1000)
posterior = optimizer.fit(model, data)

# Result:
W_samples ~ p(W | data)  # MCMC samples
```

### Phase 4: Evaluation

```python
# Create field from parameters
field = WPPMCovarianceField(model, params)

# Evaluate covariance (deterministic given params)
x = jnp.array([0.5, 0.5])
Sigma = field(x)  # Calls model.local_covariance(params, x)
```

### Phase 5: Explicit Parameter Access (advanced)

While the API encourages using high-level abstractions like `CovarianceField`, one can access the raw JAX arrays for debugging, visualization, or custom analysis.

**Access methods**:
```python
# From Posterior
raw_params = posterior.params  # -> {"W": DeviceArray(...)}

# From CovarianceField
raw_params = field.params      # -> {"W": DeviceArray(...)}
```

**Design Rationale**:
- **Encapsulation**: The `CovarianceField` abstraction hides the complexity of the underlying parameterization, allowing the backend to change without breaking user code.
- **Safety**: Parameters are wrapped in read-only properties to discourage manual mutation, which breaks JAX's functional purity.
- **Escape Hatch**: We expose the raw `params` dictionary to support access to the model internals.

---

## Code Organization

### Module Structure

```
src/psyphy/
├── model/              # Core model definitions
│   ├── wppm.py        # Wishart Process model
│   ├── prior.py       # Parameter priors
│   ├── covariance_field.py  # Covariance field abstraction
│   ├── task.py        # Task likelihoods (Oddity, 2AFC)
│   ├── noise.py       # Noise models
│   └── base.py        # Base model interface
│
├── inference/         # Posterior inference
│   ├── map_optimizer.py     # MAP estimation
│   ├── laplace.py          # Laplace approximation stub
│   └── langevin.py         # MCMC sampling stub
│
├── posterior/         # Posterior representations
│   ├── posterior.py        # MAPPosterior class
│   ├── parameter_posterior.py  # Protocol definition
│   └── predictive_posterior.py # Predictive distribution stub
│
├── acquisition/       # Active learning stub
│   ├── mutual_information.py
│   ├── upper_confidence_bound.py
│   └── expected_improvement.py
│
├── data/             # Data handling
│   ├── dataset.py   # ResponseData class, TrialData class
│   └── io.py        # Loading/saving
│
└── utils/            # Utilities
    ├── math.py      # Chebyshev basis, etc.
    └── diagnostics.py  # Model checking
```

### Dependency Graph

```
Model (WPPM)
  ├─ requires: Prior, Task, Noise
  └─ used by: Inference, Posterior

Prior
  ├─ requires: (none - self-contained)
  └─ used by: WPPM

CovarianceField
  ├─ requires: WPPM, Params
  └─ used by: User code, Posterior

Posterior
  ├─ requires: WPPM, Params
  └─ used by: User code, Acquisition

Inference
  ├─ requires: WPPM, Data
  └─ produces: Posterior

Acquisition
  ├─ requires: Posterior, Candidates
  └─ produces: Next trial location
```

---
