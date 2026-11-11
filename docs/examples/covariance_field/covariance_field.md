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
- `field(x)` returns $\Sigma(x)$, shape `(2, 2)`.

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

`field(X)` with `X` of shape `(n, 2)` returns `(n, 2, 2)`. `jax.vmap(field)(X)` gives the
same result.

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

A different key gives a different field from the same prior:

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
| `from_prior(model, key)` | Sample from prior | Initialization, visualization |
| `from_posterior(posterior)` | Extract fitted field | Analysis after fitting |
| `from_params(model, params)` | Any parameter set | tests, or fitted parameters saved earlier, e.g. from another subject |

`from_posterior` builds the field from `posterior.params`, a single parameter set. For a
maximum a posteriori (MAP) fit that is $W_\text{MAP}$, which is what you want. For a
sample-based posterior, for example from
[`NUTSSampler`](../../reference/inference.md#psyphy.inference.NUTSSampler), `params` is the
mean parameter $\mathbb{E}[W]$. The field built from it is too small, so `from_posterior`
warns. Use `WPPMCovarianceFieldSamples(posterior).mean(x)` instead. It
evaluates $\Sigma$ for every posterior draw and averages the matrices, giving
$\mathbb{E}[\Sigma(x)]$. See
[Step 5 of the NUTS tutorial](../wppm/nuts_inference_example.md#step-5-plotting-posterior-samples-and-credible-intervals).

### Evaluation

| Method | Input Shape | Output Shape | Description |
|--------|-------------|--------------|-------------|
| `field(x)` | `(d,)` | `(d, d)` | one point |
| `field(X)` | `(n, d)` | `(n, d, d)` | a batch of points |
| `jax.vmap(field)(X)` | `(n, d)` | `(n, d, d)` | same as `field(X)` |
| `field.sqrt_cov(x)` | `(d,)` | `(d, d + extra_dims)` | $U(x)$ |

`d` is `input_dim` and `n` the number of points. `field.cov` and `field.cov_batch` still
work but are deprecated in favor of `field(x)`.

### Properties

| Property | Returns | Description |
|----------|---------|-------------|
| `field.model` | `WPPM` | Associated model |
| `field.params` | `dict` | Parameter dictionary |

---

## In a fitting workflow

The same field, before and after a MAP fit, drawn with
[`plot_ellipses`](../../reference/viz.md#psyphy.viz.ellipses.plot_ellipses):

```python
from psyphy.inference import MAPOptimizer
from psyphy.viz import plot_ellipses

model = WPPM(prior=Prior(input_dim=2, basis_degree=4), likelihood=OddityTask(), noise=GaussianNoise())

# the prior, before any data
prior_field = WPPMCovarianceField.from_prior(model, jr.PRNGKey(0))
plot_ellipses(grid_points, prior_field(grid_points), scale="auto")

# fit, then the fitted field
map_posterior = MAPOptimizer(steps=500).fit(model, data)
fitted_field = WPPMCovarianceField.from_posterior(map_posterior)
plot_ellipses(grid_points, fitted_field(grid_points), scale="auto")
```

---

## Evaluating many points

Pass the whole batch in one call rather than looping over points:

```python
Sigmas = field(X_grid)  # (n, 2, 2), one call

for x in X_grid:  # n separate calls, much slower
    Sigma = field(x)
```

---
