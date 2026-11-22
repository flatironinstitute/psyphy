# Quick start: fit your first covariance ellipse

> **Goal:** run the full `psyphy` workflow in one short script, with no GPU required. We
> simulate data, fit a model and inspect the result.
>
> The complete runnable script is [`quick_start.py`](https://github.com/flatironinstitute/psyphy/blob/main/docs/examples/wppm/quick_start.py).
> For a spatially-varying field over a 2-D stimulus grid, see the [full example](full_wppm_fit_example.md).

---

### Runtime

| Hardware | Approximate time |
|---|---|
| GPU (any modern CUDA device) | < 5 s |
| CPU (laptop / M-series Mac) | < 1 min |

These settings control the runtime:

```python title="Compute settings (quick start defaults)"
--8<-- "docs/examples/wppm/quick_start.py:compute_settings"
```

---

## Step 0: Imports

```python title="Imports"
--8<-- "docs/examples/wppm/quick_start.py:imports"
```

---

## Step 1: Define a ground-truth model and sample parameters

We create a Wishart Process Psychophysical Model (WPPM) with known parameters to act as
the synthetic observer. Its parameters are the ground truth we compare the fit against.

```python title="Ground-truth model"
--8<-- "docs/examples/wppm/quick_start.py:truth_model"
```

---

## Step 2: Simulate trials at a single reference point

We generate `NUM_TRIALS` oddity-task responses at a single reference stimulus
`ref = [0, 0]`. Probe displacements are scaled by the local covariance (constant
Mahalanobis radius), so trial difficulty stays roughly uniform.

```python title="Simulate data"
--8<-- "docs/examples/wppm/quick_start.py:simulate_data"
```

The [`TrialData`](../../reference/data.md#psyphy.data.TrialData) container is the input every fit takes:

```python title="Data container"
--8<-- "docs/examples/wppm/quick_start.py:data"
```

---

## Step 3: Build the model to fit

We build a second WPPM with the same settings and start the fit from a different draw from
the prior.

```python title="Model definition"
--8<-- "docs/examples/wppm/quick_start.py:build_model"
```

```python title="Prior sample (initialization)"
--8<-- "docs/examples/wppm/quick_start.py:prior"
```

---

## Step 4: Fit with MAP optimization

```python title="Fit with MAPOptimizer"
--8<-- "docs/examples/wppm/quick_start.py:fit_map"
```

`MAPOptimizer` runs stochastic gradient descent with momentum and returns a
[`MAPPosterior`](../../reference/posterior.md#psyphy.posterior.MAPPosterior), the maximum a
posteriori (MAP) point estimate $W_\text{MAP}$.

---

## Step 5: Inspect the fitted covariance ellipse

[`WPPMCovarianceField`](../../reference/model.md#psyphy.model.WPPMCovarianceField) builds
a callable from a fit, and the callable returns $\Sigma(x)$ for any stimulus `x`. We pass
it the posterior. For a parameter set with no fit, like the ground truth and the prior
sample here, there's `WPPMCovarianceField.from_params(model, params)`.

A MAP fit is 1 parameter set, so the field holds 1 sample (`n_samples` is 1) and every
result starts with a sample axis of length 1. `[0]` drops it. The posterior stores
$W_\text{MAP}$ the way an MCMC fit stores its samples, with `n_chains` and `n_draws` both
equal to 1. No sampler ran. It's just the layout, and it lets the same line work after
either fit:

```python title="Evaluate covariance fields"
--8<-- "docs/examples/wppm/quick_start.py:cov_fields"
```

The plot below overlays the ground truth, the prior sample we started from, and the MAP fit
at the single reference point:

```python title="Plot ellipses"
--8<-- "docs/examples/wppm/quick_start.py:plot_ellipses"
```

<div align="center">
  <img src="../plots/quick_start_ellipses.png"
       alt="Covariance ellipses: ground truth (black), prior (blue), MAP fit (red)"
       width="480"/>
  <p><em>Black: ground truth. Dashed blue: the prior sample the fit started from. Red: MAP fit. Note how the red ellipse has moved from the blue (init) closer to the ground truth (black).</em></p>
</div>

---

## Step 6: Learning curve

```python title="Access learning curve"
--8<-- "docs/examples/wppm/quick_start.py:plot_learning_curve"
```

<div align="center">
  <img src="../plots/quick_start_learning_curve.png"
       alt="Learning curve"
       width="480"/>
  <p><em>Per-trial negative log-posterior over optimizer steps.</em></p>
</div>

---

## Next steps

- **Spatially-varying field:** scale up to a full 2-D grid in the [full example](full_wppm_fit_example.md).
- **Your own data:** replace the simulated `TrialData` with your own `stimuli` and `responses` arrays. At the moment each trial has to be a reference and a comparison, `stimuli` of shape `(N, 2, d)`.
- **API reference:** see [`MAPOptimizer`](../../reference/inference.md), [`WPPM`](../../reference/model.md), and [`WPPMCovarianceField`](../../reference/model.md).
