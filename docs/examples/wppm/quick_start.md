# Quick start — fit your first covariance ellipse

> **Goal:** run the full `psyphy` workflow — simulate data, fit a model, inspect the result — in one short script with no GPU required.
>
> The complete runnable script is [`quick_start.py`](https://github.com/flatironinstitute/psyphy/blob/main/docs/examples/wppm/quick_start.py).
> For a spatially-varying field over a 2-D stimulus grid, see the [full example](full_wppm_fit_example.md).

---

### Runtime

| Hardware | Approximate time |
|---|---|
| GPU (any modern CUDA device) | < 5 s |
| CPU (laptop / M-series Mac) | < 1 min |

The three knobs that control runtime:

```python title="Compute settings (quick start defaults)"
--8<-- "docs/examples/wppm/quick_start.py:compute_settings"
```

---

## Step 0 — Imports

```python title="Imports"
--8<-- "docs/examples/wppm/quick_start.py:imports"
```

---

## Step 1 — Define a ground-truth model and sample parameters

We create a WPPM with known parameters to act as the synthetic observer.
Data will be generated from it so we have a ground truth to compare against.

```python title="Ground-truth model"
--8<-- "docs/examples/wppm/quick_start.py:truth_model"
```

---

## Step 2 — Simulate trials at a single reference point

We generate `NUM_TRIALS` oddity-task responses at a single reference stimulus
`ref = [0, 0]`.  Probe displacements are scaled by the local covariance
(constant Mahalanobis radius), so trial difficulty stays roughly uniform.

```python title="Simulate data"
--8<-- "docs/examples/wppm/quick_start.py:simulate_data"
```

The `TrialData` container is the canonical input for fitting:

```python title="Data container"
--8<-- "docs/examples/wppm/quick_start.py:data"
```

---

## Step 3 — Build the model to fit

We build a fresh WPPM with the same hyperparameters but independent random
weights, then take one draw from the prior as the starting point for
optimization.

```python title="Model definition"
--8<-- "docs/examples/wppm/quick_start.py:build_model"
```

```python title="Prior sample (initialization)"
--8<-- "docs/examples/wppm/quick_start.py:prior"
```

---

## Step 4 — Fit with MAP optimization

```python title="Fit with MAPOptimizer"
--8<-- "docs/examples/wppm/quick_start.py:fit_map"
```

`MAPOptimizer` runs SGD + momentum and returns a `MAPPosterior` — a point
estimate at $W_\text{MAP}$.

---

## Step 5 — Inspect the fitted covariance ellipse

`WPPMCovarianceField` binds a `(model, params)` pair into a single callable
that returns $\Sigma(x)$ for any stimulus `x`:

```python title="Evaluate covariance fields"
--8<-- "docs/examples/wppm/quick_start.py:cov_fields"
```

The ellipse plot below overlays the ground truth (black), the prior
initialization (blue), and the MAP fit (red) at the single reference point:

```python title="Plot ellipses"
--8<-- "docs/examples/wppm/quick_start.py:plot_ellipses"
```

<div align="center">
  <img src="../plots/quick_start_ellipses.png"
       alt="Covariance ellipses: ground truth (black), prior (blue), MAP fit (red)"
       width="480"/>
  <p><em>Ground truth (black), prior sample (blue), and MAP-fitted (red) covariance ellipses at the single reference point. Note how the red ellipse has moved from the blue (init) closer to the ground truth (black).</em></p>
</div>

---

## Step 6 — Learning curve

```python title="Access learning curve"
--8<-- "docs/examples/wppm/quick_start.py:plot_learning_curve"
```

<div align="center">
  <img src="../plots/quick_start_learning_curve.png"
       alt="Learning curve"
       width="480"/>
  <p><em>Negative log-likelihood over optimizer steps.</em></p>
</div>

---

## Next steps

- **Spatially-varying field:** scale up to a full 2-D grid → [full example](full_wppm_fit_example.md).
- **Your own data:** replace the simulated `TrialData` with your own `inputs` and `responses` arrays. (currently calculations are only supported when the inputs are "refs" and "comparisons.")
- **API reference:** see [`MAPOptimizer`](../../reference/inference.md), [`WPPM`](../../reference/model.md), and [`WPPMCovarianceField`](../../reference/model.md).
