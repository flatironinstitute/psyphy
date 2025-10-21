# Issue #4: Acquisition Functions - Implementation Summary

**Status**: 🚧 In Progress
**Branch**: `api`

---

## Overview

Refactored trial placement into a clean acquisition function API inspired by BoTorch but adapted for psychophysics.

## Key Changes

### 1. New `acquisition/` Module

Created `src/psyphy/acquisition/` with functional API:

```python
# Old way (class-based, awkward)
placer = InfoGainPlacement(candidate_pool)
trials = placer.propose(posterior, batch_size=1)

# New way (functional, simple)
acq_fn = lambda X: expected_improvement(model.posterior(X), best_f=0.8)
X_next, acq_val = optimize_acqf(acq_fn, bounds, q=1)
```

### 2. Files Created

**Core Infrastructure:**
- `src/psyphy/acquisition/__init__.py` - Module exports
- `src/psyphy/acquisition/base.py` - `AcquisitionFunction` protocol

**Optimization:**
- `src/psyphy/acquisition/optimize.py` - Optimization utilities
  - `optimize_acqf_discrete()` - Exhaustive search over candidates
  - `optimize_acqf()` - Gradient-based continuous optimization
  - `optimize_acqf_random()` - Random search baseline

**Acquisition Functions:**
- `src/psyphy/acquisition/expected_improvement.py` - EI and log-EI
- `src/psyphy/acquisition/upper_confidence_bound.py` - UCB/LCB
- `src/psyphy/acquisition/mutual_information.py` - Information gain (BALD)

### 3. Design Principles

**Functional over Class-Based:**
```python
# Acquisition is just a callable
def my_acquisition(X: jnp.ndarray) -> jnp.ndarray:
    posterior = model.posterior(X)
    return posterior.mean + 2.0 * jnp.sqrt(posterior.variance)

# Works with optimization
X_next = optimize_acqf(my_acquisition, bounds, q=1)
```

**Composition over Inheritance:**
```python
# Easy to combine acquisitions
def combined_acq(X):
    ei = expected_improvement(model.posterior(X), best_f)
    ucb = upper_confidence_bound(model.posterior(X), beta=2.0)
    return 0.5 * ei + 0.5 * ucb  # Weighted combination
```

**Support Both Continuous and Discrete:**
```python
# Discrete: common for psychophysics
X_next = optimize_acqf_discrete(acq_fn, candidates, q=1)

# Continuous: gradient-based
X_next = optimize_acqf(acq_fn, bounds, q=1, method="gradient")
```

---

## API Reference

### Acquisition Functions

#### Expected Improvement
```python
from psyphy.acquisition import expected_improvement

ei = expected_improvement(
    posterior: PredictivePosterior,
    best_f: float,
    maximize: bool = True,
)
```

**Use cases:**
- Most popular for Bayesian optimization
- Balances exploration and exploitation naturally
- Works well for accuracy maximization

#### Upper Confidence Bound
```python
from psyphy.acquisition import upper_confidence_bound

ucb = upper_confidence_bound(
    posterior: PredictivePosterior,
    beta: float = 2.0,
    maximize: bool = True,
)
```

**Use cases:**
- Simple and fast (no CDF/PDF computations)
- Tunable exploration via β parameter
- Good for pure exploration (high β) or exploitation (low β)

#### Mutual Information
```python
from psyphy.acquisition import mutual_information

mi = mutual_information(
    param_posterior: ParameterPosterior,
    X: jnp.ndarray,
    probes: jnp.ndarray | None = None,
    n_samples: int = 100,
)
```

**Use cases:**
- Maximize information gain about parameters
- Ideal for threshold estimation
- Best for scientific experiments (not just optimization)

### Optimization

#### Discrete Optimization
```python
from psyphy.acquisition import optimize_acqf_discrete

X_next, acq_values = optimize_acqf_discrete(
    acq_fn: Callable,
    candidates: jnp.ndarray,
    q: int = 1,
)
```

**When to use:**
- Small candidate sets (< 10,000 points)
- Exhaustive evaluation is feasible
- Most common for psychophysics

#### Continuous Optimization
```python
from psyphy.acquisition import optimize_acqf

X_next, acq_values = optimize_acqf(
    acq_fn: Callable,
    bounds: jnp.ndarray,
    q: int = 1,
    method: str = "gradient",  # or "random"
    num_restarts: int = 10,
    optim_steps: int = 100,
)
```

**When to use:**
- Continuous stimulus spaces
- Large candidate sets (gradient search is faster)
- Need smooth optimal solutions

---

## Examples

### Example 1: Simple Discrete Optimization

```python
import jax.numpy as jnp
from psyphy.acquisition import expected_improvement, optimize_acqf_discrete
from psyphy.model import WPPM, Prior

# Fit model
model = WPPM(input_dim=2, prior=Prior(input_dim=2), ...)
model.fit(X_train, y_train, inference="map")

# Define candidates
candidates = jnp.array([
    [0.0, 0.0],
    [0.5, 0.5],
    [1.0, 1.0],
])
probes = candidates + 0.1

# Optimize acquisition
best_f = jnp.max(y_train)

def acq_fn(X):
    posterior = model.posterior(X, probes=probes)
    return expected_improvement(posterior, best_f)

X_next, ei_val = optimize_acqf_discrete(acq_fn, candidates, q=1)
```

### Example 2: Continuous Gradient-Based

```python
from psyphy.acquisition import upper_confidence_bound, optimize_acqf

# Define bounds
bounds = jnp.array([
    [0.0, 1.0],  # x1 range
    [0.0, 1.0],  # x2 range
])

# Optimize UCB
def acq_fn(X):
    posterior = model.posterior(X, probes=X + 0.1)
    return upper_confidence_bound(posterior, beta=2.0)

X_next, ucb_val = optimize_acqf(
    acq_fn,
    bounds,
    q=1,
    method="gradient",
    num_restarts=10,
    optim_steps=100,
)
```

### Example 3: Batch Acquisition

```python
# Select multiple points at once
X_batch, acq_vals = optimize_acqf_discrete(
    acq_fn,
    candidates,
    q=5,  # Select 5 points
)

# Run all 5 trials in parallel or sequentially
for X_trial in X_batch:
    y_new = run_experiment(X_trial)
    model = model.condition_on_observations(X_trial[None], y_new[None])
```

### Example 4: Information Gain

```python
from psyphy.acquisition import mutual_information

# Get parameter posterior
param_post = model.posterior(kind="parameter")

# Compute MI for candidates
mi = mutual_information(
    param_post,
    X=candidates,
    probes=probes,
    n_samples=200,  # MC samples
)

# Select most informative
X_next = candidates[jnp.argmax(mi)]
```

---

## Migration from Old API

### Old: Class-Based Trial Placement

```python
from psyphy.trial_placement import InfoGainPlacement

placer = InfoGainPlacement(candidate_pool=candidates)
trials = placer.propose(posterior, batch_size=1)
X_next = trials.references[0]
```

### New: Functional Acquisition

```python
from psyphy.acquisition import mutual_information, optimize_acqf_discrete

def acq_fn(X):
    return mutual_information(param_post, X, probes=None, n_samples=100)

X_next, mi_val = optimize_acqf_discrete(acq_fn, candidates, q=1)
```

### Old: Greedy MAP Placement

```python
from psyphy.trial_placement import GreedyMAPPlacement

placer = GreedyMAPPlacement(model)
trials = placer.propose(posterior, batch_size=1)
```

### New: Upper Confidence Bound (β=0 for greedy)

```python
from psyphy.acquisition import upper_confidence_bound, optimize_acqf_discrete

def acq_fn(X):
    posterior = model.posterior(X, probes=probes)
    return upper_confidence_bound(posterior, beta=0.0, maximize=True)

X_next, _ = optimize_acqf_discrete(acq_fn, candidates, q=1)
```

---

## What Stays in `trial_placement/`

Non-acquisition strategies remain:
- `GridPlacement` - Fixed non-adaptive designs
- `SobolPlacement` - Quasi-random exploration
- `StaircasePlacement` - Classical psychophysics (1-up-2-down)

These are NOT acquisition functions (don't use posterior), so they stay separate.

---

## Testing Plan

Need to create `tests/test_acquisition.py`:

1. **Test acquisition functions:**
   - EI: verify formula, boundary cases
   - UCB: verify β=0 (greedy), β>0 (exploration)
   - MI: verify information gain is non-negative

2. **Test optimization:**
   - Discrete: verify selects argmax
   - Continuous: verify gradient descent converges
   - Batch: verify q > 1 works

3. **Integration tests:**
   - Full workflow: fit → posterior → acquisition → optimize
   - Online learning loop with acquisitions

---

## Next Steps

1. ✅ Create acquisition module structure
2. ✅ Implement optimization utilities
3. ✅ Implement EI, UCB, MI
4. ⏳ Write comprehensive tests
5. ⏳ Update documentation
6. ⏳ Deprecate old `InfoGainPlacement` and `GreedyMAPPlacement`

---

## Performance Notes

**Discrete optimization:**
- O(n_candidates) evaluations
- Fast for n < 10,000
- Parallelizable over candidates

**Continuous optimization:**
- O(num_restarts × optim_steps) gradient evaluations
- Faster for large candidate sets
- Requires differentiable acquisition

**Mutual information:**
- O(n_samples × n_candidates) model evaluations
- Most expensive acquisition
- Use with discrete optimization for efficiency

---

## References

- **Expected Improvement**: Mockus et al. (1978)
- **UCB**: Srinivas et al. (2009)
- **BALD**: Houlsby et al. (2011)
- **BoTorch API**: https://botorch.org/api/acquisition.html
