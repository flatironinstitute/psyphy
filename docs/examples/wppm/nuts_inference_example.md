# NUTS Posterior Inference for WPPM

This tutorial demonstrates **full posterior inference** using the No-U-Turn Sampler (NUTS) for the Wishart Process Psychophysical Model (WPPM).

The companion script can be found at
[`nuts_inference_example.py`](https://github.com/flatironinstitute/psyphy/blob/main/docs/examples/wppm/nuts_inference_example.py).

---

## Why MCMC instead of MAP?

MAP estimation returns a single point estimate $\theta_\text{MAP}$:

$$\theta_\text{MAP} = \arg\max_\theta \left[\log p(\mathcal{D} \mid \theta) + \log p(\theta)\right]$$

This tells you *where the posterior peaks*, but not *how wide* it is. If the data are sparse or ambiguous, the posterior may be broad and the MAP estimate alone can't tell us that.

MCMC (and specifically NUTS) draws samples from the **full posterior** $p(\theta \mid \mathcal{D})$. This enables:

- **Uncertainty quantification**: how spread out is our belief over model parameters?
- **Posterior predictive distributions**: how uncertain are our predictions at new stimuli?
- **Convergence diagnostics**: R-hat and ESS tell you whether the chains have mixed.

---

## The posterior predictive distribution

Given posterior samples $\theta_i \sim p(\theta \mid \mathcal{D})$, predictions at new test stimuli $X^*$ are:

$$p(\text{correct} \mid X^*, \mathcal{D}) = \int p(\text{correct} \mid X^*, \theta)\; p(\theta \mid \mathcal{D})\; d\theta \approx \frac{1}{N} \sum_{i=1}^{N} \underbrace{p(\text{correct} \mid X^*, \theta_i)}_{\texttt{OddityTask.predict}(\theta_i,\, X^*)}$$

In psyphy this is computed by `WPPMPredictivePosterior`, which calls `nuts_posterior.sample(N)` to get $N$ parameter draws and vmaps the task likelihood over them. The same class works identically with a `MAPPosterior` (which just returns $N$ copies of $\theta_\text{MAP}$).

---

## A note on the likelihood

In this tutorial, we use the MC-based likelihood which is estimated via Monte Carlo inside `OddityTask.loglik` and expensive.



**Future Alternatives**
- *Neural surrogate likelihood*: replace the MC estimator with a trained neural network for exact gradients (see `NeuralSurrogateTask`).

---

## Step-by-step walkthrough

### Step 0 — Install requirements

```bash
pip install 'psyphy[sampling,diagnostics,viz]'
# sampling    -> blackjax>=1.4,<2   (needs Python >= 3.11)
# diagnostics -> arviz>=1.0         (needs Python >= 3.12)
# viz         -> matplotlib>=3.5
```

!!! note "Python version"
    Both optional dependencies carry a Python floor set by upstream, above
    psyphy's own `>=3.10`: `blackjax` 1.4+ requires **3.11+**, and every
    `arviz` 1.x requires **3.12+**. On an older interpreter the corresponding
    extra resolves to nothing, and `NUTSSampler.fit()` / `to_arviz()` raise an
    `ImportError` that says so. `psyphy` itself is unaffected.

### Step 1 — Simulate data

Same setup as the full MAP example. Data is a `TrialData` object:

```python title="Simulated data"
--8<-- "docs/examples/wppm/nuts_inference_example.py:data"
```

### Step 2 — Build the model

```python title="Model definition"
--8<-- "docs/examples/wppm/nuts_inference_example.py:build_model"
```

### Step 3 — MAP warm-start (recommended)

Starting NUTS near the posterior mode dramatically reduces the warmup needed. Run a short MAP fit first:

```python title="MAP warm-start"
--8<-- "docs/examples/wppm/nuts_inference_example.py:map_warmstart"
```

Then pass `map_posterior.params` as `init_params` to `NUTSSampler.fit()`.

### Step 4 — NUTS sampling

```python title="NUTS sampling with NUTSSampler"
--8<-- "docs/examples/wppm/nuts_inference_example.py:nuts_fit"
```

`NUTSSampler` has the same `fit(model, data)` interface as `MAPOptimizer` — the BlackJAX backend is entirely hidden. It returns an `MCMCPosterior` satisfying the same `ParameterPosterior` protocol as `MAPPosterior`.

**Compute settings.** The script takes `--mode`, which selects a preset. The *model* is
identical in every mode, so only the thoroughness of the fit changes, never what is being
fit:

```bash
python nuts_inference_example.py                    # quick (default): laptop smoke test
python nuts_inference_example.py --mode balanced    # a real fit, roughly a GPU-hour
python nuts_inference_example.py --mode full        # paper-scale MC, 4 x 1000 draws
```

```python
--8<-- "docs/examples/wppm/nuts_inference_example.py:compute_settings"
```

!!! warning "`quick` is not a fit to interpret"
    The default exists so this page builds and a laptop run finishes. With 450 trials and
    50 MC samples the frozen-key posterior (see the determinism note below) is visibly
    biased, and 150 draws cannot support R-hat or ESS. Read the diagnostics from
    `--mode balanced` or `--mode full`, not from the default.

**Runtime notes:**

- **The MC likelihood dominates everything.** Each gradient evaluation simulates
  `mc_samples` Monte Carlo draws for every trial, so cost scales with
  `mc_samples × n_trials`. Going from `quick` to `full` raises that product by ~440×,
  which is why `full` is a batch job and not something to run interactively.
- **NUTS cost per draw is not fixed.** It is `O(2^tree_depth)` gradient evaluations, and
  the depth is data-dependent — commonly 16–64 evaluations per draw. The script therefore
  **times one gradient and prints a bracketed projection** before sampling starts, so you
  can see whether a run fits inside a Slurm time limit before committing to it.
- **Adaptive warmup runs chains sequentially** (see the two warmup modes below), so wall
  clock scales linearly with `chains`. `full`'s four chains cost four times one chain.
- **Progress bars need the `progress` extra.** `show_progress=True` is the default, but the
  bar comes from `tqdm`, which is optional. Without it a multi-hour fit prints nothing and
  looks hung.
- **Only `quick` writes the plot filenames this page embeds.** Other modes suffix them
  (`nuts_trace_plot_full.png`), so a heavy run cannot silently replace the documented
  figures.

#### Two warmup modes

| Mode | When | How |
|------|------|-----|
| Adaptive (default, `step_size=None`) | Best quality; adapts step size and mass matrix per chain via `blackjax.window_adaptation`. Chains run sequentially. | `NUTSSampler(num_warmup=200, num_samples=500)` |
| Fixed (`step_size=<float>`) | Faster with many chains; no adaptation. All chains vmapped in one compiled call. User must choose a good step size. | `NUTSSampler(..., step_size=0.01)` |

### Step 5 — ArviZ diagnostics

```python title="ArviZ diagnostics"
--8<-- "docs/examples/wppm/nuts_inference_example.py:arviz_diagnostics"
```

```python title="Trace plot"
--8<-- "docs/examples/wppm/nuts_inference_example.py:trace_plot"
```

<div align="center">
  <img src="../plots/nuts_trace_plot.png" width="700"/>
  <p><em>Posterior traces for the slice <code>W[:, :, 0, 0]</code>.</em></p>
</div>

**How to read it.** `W` has shape `(3, 3, 2, 3)` = `(deg_x, deg_y, U_row, U_col)`: the
first two axes index the Chebyshev basis — degree along stimulus dimension 1 and along
dimension 2 — and the last two index the entry of `U(x)`, where `Σ = U Uᵀ + diag_term·I`.
That is 54 scalars, past ArviZ's `plot.max_subplots` limit and illegible anyway, so the
figure fixes `U[0, 0]` and plots the remaining 3×3 coefficient grid, laid out *as* that
grid by `col_wrap=3`.

| element of the figure | what it is |
|---|---|
| one panel | one Chebyshev coefficient; the title `i, j` is degree `i` × degree `j` |
| x axis | draw index **within a chain**, post-warmup (warmup draws are discarded, not plotted) |
| y axis | the value of that coefficient |
| one coloured line per panel | one chain |
| black ticks along the bottom | divergences — they appear because `sampler_stats` carries `diverging` under ArviZ's standard name |

**What good looks like:** chains overlapping each other, flat, and wiggling fast — the
"fuzzy caterpillar". A chain that drifts steadily across the panel has not converged:
warmup was too short and the sampler is still finding the typical set. That is the single
most common thing this plot catches, and it is a reason to raise `--mode`, not a reason to
distrust the model.

It is generated by **ArviZ**, from `MCMCPosterior.to_arviz()` — no psyphy plotting code is
involved. On ArviZ ≥ 1.0 `plot_trace` returns an `arviz_plots` `PlotCollection` (not an
array of matplotlib axes), so it is saved with `pc.savefig()`; the 0.x kwargs `compact=`
and `figsize=` no longer exist and raise if passed. For the classic density-plus-trace
pairing, ArviZ 1.x has a separate `az.plot_trace_dist`.

#### Interpreting diagnostics

| Diagnostic | What it measures | Good value |
|---|---|---|
| **R-hat** | Ratio of between-chain to within-chain variance. Values near 1 indicate convergence. | < 1.01 |
| **ESS (bulk)** | Effective sample size — how many independent draws the chains are equivalent to. | > 100 per chain |
| **Acceptance rate** | Fraction of NUTS proposals accepted. Too low → too large step_size. Too high → too small. | 0.6–0.9 |

With `--mode quick` (2 chains × 150 draws) R-hat and ESS will not reach textbook
thresholds, and they are not meant to — use `--mode balanced` or `--mode full`, which exist
for exactly this reason, before drawing any conclusion from these numbers.

### Step 6 — Posterior predictive uncertainty

```python title="Draw posterior parameter samples"
--8<-- "docs/examples/wppm/nuts_inference_example.py:posterior_samples"
```

```python title="Plot ensemble of ellipses"
--8<-- "docs/examples/wppm/nuts_inference_example.py:plot_uncertainty"
```

<div align="center">
  <img src="../plots/nuts_ellipses_uncertainty.png" width="600"/>
  <p><em>Faint blue: 30 covariance fields sampled from the posterior. Red: posterior mean. Black: ground truth. The spread of blue ellipses shows where the model is uncertain.</em></p>
</div>

### Step 7 — MAP vs NUTS comparison

The asymmetry between the panels is the point. **MAP can only ever draw one contour**,
because its posterior is a delta at `θ_MAP`: `MAPPosterior.sample(n)` returns `θ_MAP`
repeated `n` times, so the ensemble it produces has zero spread. NUTS puts a band around
the same contour, from *identical downstream code* — `WPPMPredictivePosterior` never asks
which kind of posterior it was given. Plotting only the two means would make them look like
equivalent point estimates and lose the one thing full posterior inference buys you.

<div align="center">
  <img src="../plots/nuts_vs_map_comparison.png" width="900"/>
  <p><em>Ground truth (left), MAP point estimate (centre), NUTS posterior mean with its
  posterior sample ensemble in faint blue (right).</em></p>
</div>

---

## Architecture recap

```
NUTSSampler.fit(model, data)        ← same interface as MAPOptimizer
        ↓  returns
MCMCPosterior                        ← satisfies ParameterPosterior protocol
  .params        → posterior mean W
  .sample(n,key) → n draws {"W": (n, *W_shape)}
  .to_arviz()    → xarray.DataTree (n_chains, n_draws, *W_shape)
        ↓
WPPMPredictivePosterior(nuts_posterior, X_test, ...)
  .mean     → E[p(correct | X*, θ) | D]
  .variance → Var[p(correct | X*, θ) | D]
```

`WPPMPredictivePosterior` is permanently agnostic about whether it wraps a `MAPPosterior` or `MCMCPosterior` — the `ParameterPosterior` protocol is the firewall.

---

## Limitations and future extensions

- **Fixed MC key bias**: We introduced a fixed key for the MC-based Oddity-Task likelihood because BlackJAX samplers require a deterministic log density. Using a fixed key introduces a slight bias. Mitigate with larger `MC_SAMPLES` or switch to the neural surrogate likelihood.
- **Sequential warmup**: adaptive mode runs one chain at a time. For >8 chains, fixed `step_size` mode + vmap is faster.
- **Other samplers**: any sampler that produces an `MCMCPosterior` plugs into the same downstream pipeline. Future additions: `MALASampler`, `SGLDSampler`, NumPyro integration.

---

## Further reading

- API reference: [`NUTSSampler`](../../reference/inference.md), [`MCMCPosterior`](../../reference/posterior.md)
- MAP fitting tutorial: [`full_wppm_fit_example`](full_wppm_fit_example.md)
- Model details: [`wppm_tutorial`](wppm_tutorial.md)
- BlackJAX documentation: [blackjax-devs.github.io/blackjax](https://blackjax-devs.github.io/blackjax/)
- ArviZ documentation: [python.arviz.org](https://python.arviz.org/)
