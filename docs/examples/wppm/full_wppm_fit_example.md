# Full WPPM fit (end-to-end) — simulated 2D data

This tutorial explains how to use the Wishart Psychophysical Process Model (WPPM) end to end:
> You can treat this as a ``recipe'' for using the Wishart Psychophysical Process Model (WPPM) in your own project: build a model, initialize parameters, fit the model, and visualize fitted predicted thresholds.

This tutorial explains the example that can be found and run
 [`full_wppm_fit_example.py`](https://github.com/flatironinstitute/psyphy/blob/main/docs/examples/wppm/full_wppm_fit_example.py) and exposes key code snippets.



### Notes on runtime, hardware, and key hyperparameters

Runtime depends heavily on your device and on a few computation-driving hyperparameters (see below).
- GPU (A100 40GB): With the default settings below, the script runs in ~3 minutes.
- CPU: To make the script practical on CPU, reduce `MC_SAMPLES` aggressively (e.g., to 5–20 and `NUM_TRIALS_Per_Ref`) and shorten the **optimizer** run (reduce `num_steps`). These two knobs usually give the biggest speedups.
- Apple Silicon / MPS: With the current configuration, the script takes 1h 45 min on  CPU (M4 Max, 64 GB)
- JAX-accelearion on MPS limitation (`jax-metal`): At the moment, JAX on MPS does not support some operations we rely on (e.g., Cholesky decomposition), hence CPU is the only

#### Default compute settings (GPU)
```python
NUM_GRID_PTS = 10  # Number of reference points over stimulus space.
MC_SAMPLES = 500  # Number of Monte Carlo samples per trial in the likelihood.
NUM_TRIALS_Per_Ref = 4000  # Total number of trials in the simulated dataset per Grid/ref point.
# optimizer:
num_steps = 2000
```


### What the [script](full_wppm_fit_example.py) does in a nutshell
- **Task:** Fit a *spatially varying covariance field* $\Sigma(x)$ over a 2D stimulus space $x \in [-1,1]^2$ using the **Wishart Process Psychophysical Model (WPPM)**.
- **Data:** synthetic oddity-task responses simulated from a ``ground-truth'' WPPM.
- **Inference:** MAP (maximum a posteriori) optimization of the WPPM parameters.



---

## What the Wishart Psychophysical Process Model (WPPM) is in a nutshell

WPPM defines a *covariance matrix field* $\Sigma(x)$ over stimulus space (e.g. color represented in RGB). Intuitively, $\Sigma(x)$ describes the local noise/uncertainty ellipse around stimulus $x$ where stimuli within that ellipse will be perceived as identical to the human observer.

The model represents $\Sigma(x)$ as

\[
\Sigma(x) = U(x)U(x)^\top + \varepsilon I,
\]

where $U(x)$ is a smooth, basis-expanded matrix-valued function and $\varepsilon$ is a small diagonal “jitter” (`diag_term`) to avoid numerical issues. Alternatively, in Gaussian Process (GP) terms, you can think of $U(x)$ defining a GP in weight space, i.e., a "Bayesian linear model".

A psychophysical task model (here: `OddityTask`) uses $\Sigma$ to compute probability of a correct response on each trial, and `MAPOptimizer` fits WPPM parameters by maximizing

For more details on the psychophysical task used in this example as well as some more details on the model, please checkout the paper by [Hong et al (2025)](https://elifesciences.org/reviewed-preprints/108943) and this [tutorial](`docs/examples/wppm/wppm_tutorial.md`).

>The most important thing to keep in mind is that the **task** used in the experiment (with humans) implicitly defines the **likelihood**. So, in this context, you can think of task and likelihood as interchangeable.

\[
\log p(W \mid \mathcal{D}) = \log p(\mathcal{D} \mid W) + \log p(W).
\]

---
## Step 0 -- Load data

For OddityTask, we store trials as (ref, comparison) even though the task involves three items, since the assumed stimulus triplet is (ref, ref, comparison), i.e., ref, ref represent two samples from the distribution over ref.

```python title="Load data"
--8<-- "docs/examples/wppm/full_wppm_fit_example.py:data"
```

`psyphy` provides two lightweight containers for trial data (defined in [`src/psyphy/data/dataset.py`](https://github.com/flatironinstitute/psyphy/blob/main/src/psyphy/data/dataset.py)):

**`TrialData` (compute-first; used for fitting):**

- The canonical, (batched) input expected by likelihood evaluation and optimizers (e.g., MAPOptimizer.fit(...)).
It holds JAX arrays:

   - refs: (N, d)
   - comparisons: (N, d)
   - responses: (N,)

where $d$ refers to the input dimension, here 2.

**`ResponseData` (collection/I/O-first; convenient for experiments):**
- A Python-friendly log (stores trials in lists) designed for incremental collection, saving/loading (e.g., CSV), and adaptive experiments but expensive for computation

#### Typical workflow:
Avoid repeatedly converting Python lists → JAX arrays inside tight loops.

- If you’re doing online fitting with adaptive trial placement, it’s usually better to:
-  collect trials in ResponseData (easy incremental updates).
- then convert to `TrialData` with `to_trial_data()` when you’re ready to fit/evaluate a model once or  in batches (e.g. every K trials) before running expensive optimizaiton.


Note that here, we simlulate data, foe details check out  [`full_wppm_fit_example.py`](full_wppm_fit_example.py) directly resulting in a `TrialData` object.


---

## Step 1 — Define the prior (how weights are distributed initially)

The WPPM parameters are basis weights stored as a dict:

- `params = {"W": W}`

where `W` is a tensor of Chebyshev-basis coefficients.

### Prior distribution over weights

See `src/psyphy/model/prior.py`:

- `Prior.sample_params(key)` samples weights `W` from a **zero-mean Gaussian** with a *degree-dependent variance*.

For 2D, the weight tensor shape is

\[
W \in \mathbb{R}^{(d+1) \times (d+1) \times D \times E},
\]

where:

- $d$ = `basis_degree`

      - -> we're representing a function using a Chebychev expansion with terms up to degree `basis_degree`, which is 4 here

      - T_0(x), T_1(x), T_2(x), T_3(x), T_4(x)
- $D$ = `input_dim` (here 2)
- $E$ = `embedding_dim = input_dim + extra_embedding_dims`

The prior variance decays with basis “total degree”. In code:

- `Prior._compute_basis_degree_grid()` constructs degrees $i+j$ (2D) or $i+j+k$ (3D).
- `Prior._compute_W_prior_variances()` returns

\[
\sigma^2_{ij} = \texttt{variance_scale} \cdot (\texttt{decay_rate})^{(i+j)}.
\]

- `Prior.sample_params(...)` then samples

\[
W_{ijde} \sim \mathcal{N}(0, \sigma^2_{ij}).
\]

This is the  state of the  WPPM: **before any data**, WPPM draws smooth random fields because high-frequency coefficients are shrunk by the decay.


```python title="Ground-truth model + prior sample"
--8<-- "docs/examples/wppm/full_wppm_fit_example.py:truth_model"
```


<div align="center">
    <picture>
    <img align="center" src="../plots/prior_sample.png" width="600"/>
    </picture>
    <p><em>A sample from the prior</em></p>
</div>




---

## Step 2 — Build the model (WPPM + task + noise)

Like good Bayesians, we build a model by combining a **prior** and a **likelihood**.


In `psyphy`, `model` acts as  a container for both

- Prior specific hyerparameters, owned by the `Prior`.

- Likelihood-specific hyperparameters are owned by the `Task`

- The `model` also takes compute-specific arguments such as diag_term, which improves numerical stability by encouraging positive-definite matrices.

```python title="Model definition"
--8<-- "docs/examples/wppm/full_wppm_fit_example.py:build_model"
```

---

## Step 3 — A Draw from the Prior a.k.a Evaluate the covariance field $\Sigma(x)$

Now that we have the model, we can evaluate the
the covariance field at $x$
\[
\Sigma(x) = U(x)U(x)^\top + \varepsilon I.
\]

where $x$ is `ref_points` in the code:

```python title="Covariance field evaluation ($\Sigma(x)$), here Prior"
--8<-- "docs/examples/wppm/full_wppm_fit_example.py:prior"
```
<div align="center">
    <picture>
    <img align="center" src="../plots/prior_sample.png" width="600"/>
    </picture>
    <p><em>A sample from the prior</em></p>
</div>


---

## Step 4 — Fit with MAP optimization

We obtain a MAP estimate over weights $W$ by  computing the negative log likelihood  using
SGD + momentum:


```python title="Fitting with psyphy (MAPOptimizer)"
--8<-- "docs/examples/wppm/full_wppm_fit_example.py:fit_map"
```

### What is being optimized

We compute a MAP estimate of weights $W$:

\[
W_\text{MAP} = \arg\max_{W} \big[\log p(\mathcal{D}\mid W) + \log p(W)\big].
\]

- $\log p(W)$ is from `Prior.log_prob(params)` (see `prior.py`).
- $\log p(\mathcal{D}\mid W)$ is computed by the task’s log-likelihood (here via Monte Carlo inside `OddityTask.loglik`).

The result in this example is a `MAPPosterior` object that contains a point estimate `map_posterior.params`.


---

## Step 6 — Visualize fit vs. truth vs. prior sample
To see how we generate the covariance field figures, checkout the plotting code in this [script](https://github.com/flatironinstitute/psyphy/blob/main/docs/examples/wppm/full_wppm_fit_example.py), which reproduces the figures in this tutorial.
<div align="center">
    <picture>
    <img align="center" src="../plots/ellipses.png" width="600"/>
    </picture>
    <p><em>Fitted ellipsoids (red) overlayed with ground truth (gray) and model initialization (blue), a sample from the prior. The fitted ellipsoids (read) are very close to the ground truth (gray). Note the bottom right corner, where they diverge.</em></p>
</div>
---

```python title="Access learning curve"
--8<-- "docs/examples/wppm/full_wppm_fit_example.py:plot_learning_curve"
```
<div align="center">
    <picture>
    <img align="center" src="../plots/learning_curve.png" width="600"/>
    </picture>
    <p><em>Learning curve.</em></p>
</div>

---

## To recap: Minimal recipe (copy/paste mental model)

To use WPPM on your own data, these are the essential calls:

**1. Create** task + noise + prior:

   - `task = OddityTask()`

   - `noise = GaussianNoise(sigma=...)`

   - `prior = Prior(input_dim=..., basis_degree=..., extra_embedding_dims=..., decay_rate=..., variance_scale=...)`

**2. Create** WPPM:

   - `model = WPPM(input_dim=..., prior=prior, task=task, noise=noise, diag_term=...)`

**3. Initialize** parameters:

   - `params0 = model.init_params(jax.random.PRNGKey(...))`  (draws from `Prior.sample_params`)

**4. Load/build** a dataset:

   - `data = TrialData(refs=..., comparisons=..., responses=...)`

**5. Fit**:

   - `map = MAPOptimizer(...).fit(model, data, init_params=params0, ...)`

**6. Inspect** $\Sigma(x)$:

   - `field = WPPMCovarianceField(model, map.params)`
   - `Sigmas = field(xs)`

---

## Notes and pitfalls

- **CPU vs GPU:** this example can be heavy because the oddity likelihood uses Monte Carlo. A GPU can help a lot.
- **Positive definiteness:** `diag_term` is important. If you ever see a non-PD covariance, increase `diag_term` slightly.
- **MC variance:** optimization stability depends on `MC_SAMPLES`. Too small means noisy gradients.


---
## Next places to explore

- Read the API docs in `docs/reference/` (especially model + inference sections).
- Inspect `src/psyphy/model/prior.py` if you want to change smoothness/regularization.
- Inspect `src/psyphy/model/covariance_field.py` if you want faster / vmapped field evaluation patterns.



## If your curious about some of the implementation details, checkout these files:

<!--
Note on links:
MkDocs warns when a Markdown link points outside the `docs/` tree.
For source files under `src/`, we therefore link to the repository (GitHub)
instead of using relative filesystem paths.
-->

- Example script: [`docs/examples/wppm/full_wppm_fit_example.py`](full_wppm_fit_example.py)
- Prior (how weights are initialized / regularized): [`src/psyphy/model/prior.py`](https://github.com/flatironinstitute/psyphy/blob/main/src/psyphy/model/prior.py)
- Model definition: [`src/psyphy/model/wppm.py`](https://github.com/flatironinstitute/psyphy/blob/main/src/psyphy/model/wppm.py) (see `WPPM`)
- Covariance field wrapper: [`src/psyphy/model/covariance_field.py`](https://github.com/flatironinstitute/psyphy/blob/main/src/psyphy/model/covariance_field.py) (see `WPPMCovarianceField`)
- Task / likelihood: [`src/psyphy/model/task.py`](https://github.com/flatironinstitute/psyphy/blob/main/src/psyphy/model/task.py) (see `OddityTask`)
- Noise model: [`src/psyphy/model/noise.py`](https://github.com/flatironinstitute/psyphy/blob/main/src/psyphy/model/noise.py) (see `GaussianNoise`)
- MAP fitting: [`src/psyphy/inference/map_optimizer.py`](https://github.com/flatironinstitute/psyphy/blob/main/src/psyphy/inference/map_optimizer.py) (see `MAPOptimizer`)
- Data container: [`src/psyphy/data/dataset.py`](https://github.com/flatironinstitute/psyphy/blob/main/src/psyphy/data/dataset.py) (see `ResponseData`)

If you want to “follow the call graph”:

1. `WPPM.init_params(...)` (defined in [`src/psyphy/model/wppm.py`](https://github.com/flatironinstitute/psyphy/blob/main/src/psyphy/model/wppm.py)) → delegates to the prior’s `Prior.sample_params(...)` (defined in [`src/psyphy/model/prior.py`](https://github.com/flatironinstitute/psyphy/blob/main/src/psyphy/model/prior.py)).
2. `OddityTask.predict_with_kwargs(...)` / `OddityTask.loglik(...)` (defined in [`src/psyphy/model/task.py`](https://github.com/flatironinstitute/psyphy/blob/main/src/psyphy/model/task.py)) → calls into the model to get $\Sigma(x)$ and then runs the task’s decision rule (Monte Carlo in the full model).
3. `WPPMCovarianceField(model, params)` (defined in [`src/psyphy/model/covariance_field.py`](https://github.com/flatironinstitute/psyphy/blob/main/src/psyphy/model/covariance_field.py)) → provides a callable `field(x)` that returns $\Sigma(x)$ for single points or batches.
4. `MAPOptimizer.fit(...)` (defined in [`src/psyphy/inference/map_optimizer.py`](https://github.com/flatironinstitute/psyphy/blob/main/src/psyphy/inference/map_optimizer.py)) → runs gradient-based optimization of the negative log likelihood.
