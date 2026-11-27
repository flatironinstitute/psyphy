# Covariance Field: Visualizing Spatially-Varying Internal Noise

[`WPPMCovarianceField`](../../reference/model.md#psyphy.model.WPPMCovarianceField)
evaluates the covariance $\Sigma(x)$ of the Wishart Process Psychophysical Model (WPPM) at
any stimulus location $x$. $\Sigma(x)$ is the observer's internal noise, not the
discrimination threshold: thresholds are computed from it, as in the
[Hong et al. reproduction](../wppm/hong2025_reproduction.md). Here we evaluate the field at
one point and on a grid, and draw each $\Sigma(x)$ as an ellipse.

You can run this complete example yourself with:
```bash
python docs/examples/covariance_field/covariance_field_demo.py
```

---

## Mathematical Background

A **covariance field** $\Sigma(x)$ maps each stimulus location to a covariance matrix, the
internal noise at that location.

### Wishart Process Covariance
The full WPPM uses a Wishart Process to model spatially-varying, full covariance:

$$
\Sigma(x) = U(x) \, U(x)^\top + \lambda I
$$

where:

$$
U(x) = \sum_{ij} W_{ij} \, \phi_{ij}(x)
$$

- $\phi_{ij}(x)$: Chebyshev basis functions
- $W_{ij}$: Learned coefficients
- $\lambda$: Numerical stabilizer (`diag_term`)

---

## Imports

```python title="Required Imports"
--8<-- "docs/examples/covariance_field/covariance_field_demo.py:imports"
```

---

## Example 1: Single point

### Creating a covariance field

We create a WPPM with a Chebyshev basis up to degree 4 in each input dimension (`basis_degree=4`, so $5 \times 5$
basis functions) and sample a covariance field from the prior:

```python title="Create Wishart Model and Sample Field"
--8<-- "docs/examples/covariance_field/covariance_field_demo.py:single_point"
```

- `basis_degree=4` is set on the `Prior`, and the WPPM takes it from there.
- `extra_dims=1` adds one embedding dimension, so $U(x)$ has shape `(2, 3)`.
- `variance_scale=0.03` and `decay_rate=0.3` set how smooth the field is.
- `field(x)` returns $\Sigma(x)$ with the sample axis first, shape `(1, 2, 2)` for our 1 prior
  sample. `[0]` gives the matrix.

### Plotting one ellipse

We draw $\Sigma(x)$ at a single point as an ellipse:


<div align="center">
    <picture>
    <img align="center" src="../plots/single_point_ellipse.png" width="600"/>
    </picture>
    <p><em>Covariance ellipse at (0.5, 0.5), drawn at 0.4 times its true size. Its orientation is the direction of strongest noise correlation, and its size is the noise magnitude.</em></p>
</div>



---

## Example 2: Covariance field on a grid

### Evaluating a batch of points

To see how the noise varies across space, we evaluate the field at a grid of points in one
call:

```python title="Create Grid and Evaluate"
--8<-- "docs/examples/covariance_field/covariance_field_demo.py:grid_points"
```

`field(X)` with `X` of shape `(n, 2)` returns `(1, n, 2, 2)`, since the sample axis comes
first in every call. `jax.vmap(field)(X)` gives the same values with the points first,
`(n, 1, 2, 2)`. So we pass the batch to `field` directly.

### Plotting the field


<div align="center">
    <picture>
    <img align="center" src="../plots/ellipse_grid.png" width="700"/>
    </picture>
    <p><em>Covariance ellipses on a 5x5 grid, drawn at 0.15 times their true size. Note how size, shape and orientation change smoothly from one point to the next.</em></p>
</div>


---

## Example 3: Another prior sample

### Sampling a different field

A different key gives a different field from the same prior. This time we draw the
parameters ourselves and pass them to `from_params`, as we would a ground truth or a
published $W$:

```python title="Custom Field from Different Prior Sample"
--8<-- "docs/examples/covariance_field/covariance_field_demo.py:custom"
```

### Plotting it

<div align="center">
    <picture>
    <img align="center" src="../plots/custom_field_ellipses.png" width="700"/>
    </picture>
    <p><em>A second prior sample, drawn at 0.15 times its true size.</em></p>
</div>


---



## Summary of the API

### Constructors

| Method | Description | Use Case |
|--------|-------------|----------|
| `WPPMCovarianceField(posterior)` | every stored sample of a fit | after a maximum a posteriori (MAP) fit, 1 sample, or an MCMC fit |
| `from_prior(model, n_samples=n, key=key)` | `n` samples from the prior | initialization, visualization |
| `from_params(model, params)` | 1 parameter set, with a sample axis of length 1 | a ground truth, a published $W$, or fitted parameters saved earlier |

After a sample-based fit, for example from
[`NUTSSampler`](../../reference/inference.md#psyphy.inference.NUTSSampler),
`field.mean(x)` evaluates $\Sigma$ for every sample and averages the matrices, giving
$\mathbb{E}[\Sigma(x)]$. See
[Step 5 of the NUTS tutorial](../wppm/nuts_inference_example.md#step-5-plotting-posterior-samples-and-credible-intervals).
Don't pass that fit's `posterior.mean_params` to `from_params`. It's the mean parameter
$\mathbb{E}[W]$, and the field built from it is too small. `posterior.params` raises.

`field.sqrt_cov(x)` returns $U(x)$ for every sample, shape `(n_samples, d, m)` for one point,
with $m$ the embedding dimension.

### Evaluation

| Method | Input Shape | Output Shape | Description |
|--------|-------------|--------------|-------------|
| `field(x)` | `(d,)` | `(n_samples, d, d)` | one point |
| `field(X)` | `(n, d)` | `(n_samples, n, d, d)` | a batch of points |
| `field.mean(X)` | `(n, d)` | `(n, d, d)` | $\mathbb{E}[\Sigma(x)]$ over the samples |
| `jax.vmap(field)(X)` | `(n, d)` | `(n, n_samples, d, d)` | the points first; pass the batch to `field` instead |
| `field.sqrt_cov(x)` | `(d,)` | `(n_samples, d, d + extra_dims)` | $U(x)$ |

`d` is `input_dim`, `n` the number of points, and `n_samples` is 1 after a MAP fit or for a
single parameter set.

### Properties

| Property | Returns | Description |
|----------|---------|-------------|
| `field.model` | `WPPM` | Associated model |
| `field.samples` | `dict` | The parameter sets, leading axis `n_samples` |
| `field.n_samples` | `int` | How many parameter sets the field holds |

---

## In a fitting workflow

The same field, before and after a MAP fit, drawn with
[`plot_ellipses`](../../reference/viz.md#psyphy.viz.ellipses.plot_ellipses):

```python
from psyphy.inference import MAPOptimizer
from psyphy.viz import plot_ellipses

model = WPPM(prior=Prior(input_dim=2, basis_degree=4), likelihood=OddityTask(), noise=GaussianNoise())

# the prior, before any data
prior_field = WPPMCovarianceField.from_prior(model, n_samples=1, key=jr.PRNGKey(0))
plot_ellipses(grid_points, prior_field(grid_points), scale="auto")  # (1, n, 2, 2): 1 field

# fit, then the fitted field
map_posterior = MAPOptimizer(steps=500).fit(model, data)
fitted_field = WPPMCovarianceField(map_posterior)
plot_ellipses(grid_points, fitted_field(grid_points), scale="auto")
```

---

## Evaluating many points

Pass the whole batch in one call rather than looping over points:

```python
Sigmas = field(X_grid)  # (n_samples, n, 2, 2), one call

for x in X_grid:  # n separate calls, much slower
    Sigma = field(x)
```

---
