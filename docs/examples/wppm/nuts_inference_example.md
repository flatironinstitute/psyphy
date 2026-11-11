# Sampling the posterior with NUTS

The [maximum a posteriori (MAP) tutorial](full_wppm_fit_example.md) gives one threshold
ellipse per reference point. Here we get a distribution of ellipses instead. Tight clusters mean the data
constrain the threshold ellipse and wide spread means they don't. A single MAP ellipse hides
that difference. Crucially, adaptive trial placement depends on this uncertainty, because
the next trial should go where the model is most uncertain.

!!! note "Check the samples before you trust them"
    The example below is converged. Before you trust any set of samples you need to check
    that they are representative of the posterior.
    [Checking MCMC samples with ArviZ](mcmc_diagnostics.md) shows how, and which diagnostics
    to run.

<div align="center">
  <img src="../plots/nuts_ellipses_uncertainty_full.jpg" width="560"/>
  <p><em>What we build in this tutorial. The axes are the two model dimensions, arbitrary up to
  an affine transformation of the input space (e.g., RGB). Each green dot is one of the 25 reference
  points at which the model is compared against the ground truth. Around each of them, black
  is the true threshold contour, red is the posterior mean of the covariance field, \(\mathbb{E}[\Sigma(x)]\), and the
  200 faint blue contours are individual samples from the posterior. Built in <a href="#step-5-plotting-posterior-samples-and-credible-intervals">Step 5</a>.</em></p>
</div>

The companion script is
[`nuts_inference_example.py`](https://github.com/flatironinstitute/psyphy/blob/main/docs/examples/wppm/nuts_inference_example.py).

| step | what we do | what we get |
|---|---|---|
| 1–2 | simulate data, build the Wishart Process Psychophysical Model (WPPM) | a [`TrialData`](../../reference/data.md#psyphy.data.TrialData) and a model, as in the [MAP tutorial](full_wppm_fit_example.md) |
| 3 | run a short MAP fit | a starting point near the posterior peak |
| 4 | run the sampler, here the No-U-Turn Sampler (NUTS), via [BlackJAX](https://blackjax-devs.github.io/blackjax/) | an [`MCMCPosterior`](../../reference/posterior.md#psyphy.posterior.MCMCPosterior): thousands of plausible parameter arrays $W$ |
| 5 | evaluate the covariance field for every sample | a distribution of threshold contours, and a credible band |
| next | check the chains with [ArviZ](https://python.arviz.org/) | evidence that the samples are representative of the posterior, see [Checking MCMC samples with ArviZ](mcmc_diagnostics.md) |

??? note "How to run the script"

    The following 3 dependencies are required, none of which psyphy requires by default:

    ```bash
    pip install 'psyphy[sampling,diagnostics,viz]'
    ```

    | extra | brings in | needed for |
    |---|---|---|
    | `sampling` | `blackjax>=1.4,<2` | [`NUTSSampler`](../../reference/inference.md#psyphy.inference.NUTSSampler) |
    | `diagnostics` | [`arviz>=1.0`](https://python.arviz.org/en/latest/user_guide/getting_started.html) | [`covariance_to_arviz`](../../reference/model.md#psyphy.model.covariance_to_arviz), R-hat, effective sample size (ESS), the diagnostic figures |
    | `viz` | `matplotlib>=3.5` | the figures |

    If you want to run the companion script
    [`nuts_inference_example.py`](https://github.com/flatironinstitute/psyphy/blob/main/docs/examples/wppm/nuts_inference_example.py)
    yourself to reproduce the results shown here, you have three options: `--mode` selects a
    compute preset.

    ```bash
    python nuts_inference_example.py --mode quick      # smoke test, ~1 min on an A100, but possible on CPU (ca. 35 min)
    python nuts_inference_example.py --mode balanced   # good samples but not converged, ~35 min on an A100
    python nuts_inference_example.py --mode full       # 11 h on a GPU, and what we show here, converged
    ```

    Every figure and number here comes from `--mode full`. `quick` is just a smoke test for the
    code path. Its chains are far too short to converge, so its diagnostics are not to be interpreted.

    Only MAP and NUTS are expensive, and the script caches them. The diagnostics and all
    figures are post-processing of the stored samples and don't need a GPU and can easily be done like this:

    ```bash
    python nuts_inference_example.py --mode full --from-fit
    ```

    The cache (`fits/nuts_full_fit.npz`) is committed and records the settings it was
    produced with.

---

## Why sample the posterior

MAP estimation returns the single parameter array that maximizes the posterior:

$$\theta_\text{MAP} = \arg\max_\theta \left[\log p(\mathcal{D} \mid \theta) + \log p(\theta)\right]$$

That locates the peak but doesn't tell us anything about its width. What we want is the posterior
$p(\theta \mid \mathcal{D})$ itself, and Bayes' rule gives it in one line containing an
integral over all of parameter space that we cannot evaluate here.

[Markov chain Monte Carlo](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo) gets
around this by constructing a random walk whose long-run visiting frequency is the
posterior, so the places it has been form a sample from it. A plain random walk mixes too
slowly to be usable in the roughly 100 dimensions of our parameteres $W$, so
[Hamiltonian Monte Carlo. (HMC)](https://en.wikipedia.org/wiki/Hamiltonian_Monte_Carlo) uses the
gradient of the log posterior to move along the distribution's geometry instead. Three
terms will come up repeatedly, also when diagnosing the sampler:

- the path can't be solved exactly, so an *integrator* approximates it in small steps
- the *step size* is how far each step goes
- the sequence of steps is the *trajectory*

That leaves HMC with two things to set: the step size, and how many steps before stopping.
NUTS, of [Hoffman and Gelman (2014)](https://jmlr.org/papers/v15/hoffman14a.html), removes
the second by extending each trajectory until it starts to double back on itself. The step
size is still tuned, during warmup ([Step 4](#step-4-nuts-sampling)). Betancourt's
[conceptual introduction to HMC](https://arxiv.org/abs/1701.02434) is a good resource to skim if you'd like to learn more.

!!! tip "What we get in exchange for the extra compute"

    - **Uncertainty quantification**: how spread out is our belief over model parameters?
    - **Posterior predictive distributions**: how uncertain are our predictions at new test
      points?

??? note "Why the sampler needs a frozen random key"

    NUTS needs the log density to be a fixed function of the parameters, and psyphy's oddity
    likelihood is estimated by simulation, so it returns a slightly different number on every
    call. `NUTSSampler` freezes one pseudo-random number generator (PRNG) key for the whole
    run (`logdensity_key_seed`) so the sampler sees the same density twice.

    The sampler also needs gradients, which is why the log posterior is written in JAX and
    goes straight to BlackJAX.

    What the frozen key costs, and when it stops being needed, is in
    [Developer notes](../../developer-notes.md#the-log-density-uses-one-frozen-prng-key).

---

## Step-by-step walkthrough

### Step 1: Simulate data

Same setup as the [MAP tutorial](full_wppm_fit_example.md): oddity-task trials from a known
ground-truth WPPM, so there is something to compare the fit against.

```python title="Data container"
--8<-- "docs/examples/wppm/nuts_inference_example.py:data"
```

### Step 2: Build the model

```python title="Model definition"
--8<-- "docs/examples/wppm/nuts_inference_example.py:build_model"
```

The model's output is a covariance *field* $\Sigma(x)$: a $2\times2$ covariance matrix at
every point $x$ of the stimulus space, whose contour is the threshold contour we plot. Every
$\Sigma$ below means this, evaluated at one reference point and one posterior sample unless
said otherwise.

??? note "Recap: the model and its parameters"

    The model is a [`Prior`](../../reference/model.md#psyphy.model.Prior) over the parameters, plus a task likelihood and a noise model.
    Every diagnostic and figure below refers to the parameters, so here is what they are.

    **Parameters.** The WPPM fits a covariance *field* $\Sigma(x)$: a $2\times2$ matrix at
    each point $x$ in stimulus space. It is defined as

    $$\Sigma(x) = U(x)\,U(x)^\top + \lambda I, \qquad U(x) = \sum_{j} W_j \,\phi_j(x)$$

    The $\phi_j$ are fixed Chebyshev basis functions, up to degree 2 per input dimension.
    Only the coefficients $W_j$ are fitted. With `input_dim=2`, `basis_degree=2` and
    `extra_embedding_dims=1` they form an array of shape `(3, 3, 2, 3)`. $\lambda$
    (`diag_term`) is a small floor that keeps $\Sigma$ invertible.

    For the basis, the prior and `diag_term` in more depth, see the
    [full WPPM fit](full_wppm_fit_example.md).

### Step 3: MAP warm-start

A sampler started far from the posterior spends its first samples getting
there, and those samples are then thrown away. Starting at the MAP estimate instead avoids
this:

```python title="MAP warm-start"
--8<-- "docs/examples/wppm/nuts_inference_example.py:map_warmstart"
```

`map_posterior.params` then goes to `NUTSSampler.fit()` as `init_params` as shown in the code snippet below:

### Step 4: NUTS sampling

```python title="NUTS sampling with NUTSSampler"
--8<-- "docs/examples/wppm/nuts_inference_example.py:nuts_fit"
```

`NUTSSampler` has the same `fit(model, data)` interface as [`MAPOptimizer`](../../reference/inference.md#psyphy.inference.MAPOptimizer), and the BlackJAX
backend stays hidden. Each chain runs a warmup phase, where the step size and mass matrix
are tuned and the samples discarded, then a sampling phase where the tuning is frozen and
every sample is kept. The code and ArviZ call these draws. We say samples.

What comes back is an `MCMCPosterior`. We use 2 things on it:

| call | returns | used in |
|---|---|---|
| [`WPPMCovarianceFieldSamples`](../../reference/model.md#psyphy.model.WPPMCovarianceFieldSamples)`(nuts_posterior)` | every stored sample as a covariance field, with `.mean(x)` for $\mathbb{E}[\Sigma(x)]$ and `.band(x)` for a credible band. Pass `n_samples` and `key` for a random subset | Step 5, the contour ensemble, the red contours and the band |
| `covariance_to_arviz(nuts_posterior, points)` | $\Sigma$ at chosen points for every sample, in ArviZ's format | [the diagnostics page](mcmc_diagnostics.md) |

One sample is one complete covariance field, so one threshold contour at every reference
point. `MCMCPosterior` and [`MAPPosterior`](../../reference/posterior.md#psyphy.posterior.MAPPosterior) share this interface, so code written for a MAP
fit, [`WPPMPredictivePosterior`](../../reference/posterior.md#psyphy.posterior.WPPMPredictivePosterior) for instance, takes a NUTS fit unchanged.

!!! warning "Do not build a covariance field from `nuts_posterior.params`"
    `nuts_posterior.params` is the mean *parameter* $\mathbb{E}[W]$. It is a valid parameter
    set and works as `init_params`, but the model is nonlinear in its parameters, so
    building a field from the mean parameter is not the same as averaging the fields.
    [`WPPMCovarianceField`](../../reference/model.md#psyphy.model.WPPMCovarianceField)`(model, nuts_posterior.params)`
    gives contours several times too small on a typical fit. Use
    [`WPPMCovarianceFieldSamples`](../../reference/model.md#psyphy.model.WPPMCovarianceFieldSamples)`.mean`
    ([Step 5](#step-5-plotting-posterior-samples-and-credible-intervals)), which averages
    $\Sigma$ rather than $W$. Its docstring has the detail.
    `WPPMCovarianceField.from_posterior` warns when handed an MCMC posterior.

### Step 5: Plotting posterior samples and credible intervals

Every sample is a complete covariance field, so the ensemble of samples is an ensemble of
threshold contours. Built from `nuts_posterior` alone, the object holds every stored sample. Called at the reference
points, it gives one $\Sigma$ per sample and point. We draw every k-th of them, so the
plot shows `n_ensemble` contours from all chains.

```python title="Draw posterior samples as covariance fields"
--8<-- "docs/examples/wppm/nuts_inference_example.py:posterior_samples"
```

For one summary field, `.mean` builds $\Sigma$ from each sample and averages the matrices.
The other route,
`WPPMCovarianceField(model, posterior.params)`, averages the samples of $W$ first and builds a
single field from that mean parameter. The model is nonlinear in its parameters, so the two
orders do not agree: averaging first drops the spread between samples. It reports contours
several times too small at every reference point, and gives no sign that it has. The class
docstring has the identity behind it.

```python title="Posterior mean covariance field"
--8<-- "docs/examples/wppm/nuts_inference_example.py:posterior_mean"
```

```python title="Plot posterior samples"
--8<-- "docs/examples/wppm/nuts_inference_example.py:plot_posterior_samples"
```

<div align="center">
  <img src="../plots/nuts_ellipses_uncertainty_full.jpg" width="560"/>
  <p><em>Faint blue: one threshold contour per posterior sample, 200 of them. Red: the
  posterior mean $\mathbb{E}[\Sigma(x)]$. Black: ground truth. Green: the 25 reference
  points.</em></p>
</div>

One threshold contour per posterior sample, at every reference point. Tight clusters mean the
data constrain the threshold, wide spread means they do not. This is the figure at the top
of the page, repeated here because it is what the snippet above produces.

A cloud of contours is the full answer, but a paper reports an interval. psyphy computes
one with `.band`, a method on the same class, at the points we pass. The ensemble, the
mean and the band all come from the same samples.

??? note "What to pass, and what we get back"

    | argument | what to pass |
    |---|---|
    | `points` | the reference points, `(n_points, 2)` |
    | `ci_prob` | how much posterior mass the interval holds. `0.89` by default, matching ArviZ |
    | `ci_kind` | `"eti"` for equal-tailed, `"hdi"` for the narrowest interval of that mass |
    | `n_directions` | how many directions each contour is measured along. `181` by default, one every two degrees |

    The samples the band uses are the ones the object was built with. Pass `n_samples` and
    `key` when building it for a random subset.

    What we get back is a
    [`ContourBand`](../../reference/model.md#psyphy.model.ContourBand): `lower`,
    `median` and `upper` radii, each `(n_points, n_directions)`, plus the `directions` they
    were measured along and the `ci_prob` and `ci_kind` that produced them, so a figure
    caption can state which interval it draws.

It works by writing each contour as a radius per direction, which is the discrimination
threshold, the just noticeable difference, in that direction. That is a scalar, one number
per sample, so quantiles over samples give an interval at each direction. Filling between them
gives us the credible band.
[`contour_radii`](../../reference/model.md#psyphy.model.contour_radii) computes those radii
on their own, in any dimension, for anyone who wants a quantile psyphy does not offer. Its
docstring has the formula and the derivation.

```python title="Pointwise credible band on the threshold contour"
--8<-- "docs/examples/wppm/nuts_inference_example.py:credible_band"
```

<div align="center">
  <img src="../plots/nuts_threshold_band_full.jpg" width="620"/>
  <p><em>Pointwise 89% credible band on each threshold contour (blue fill), the posterior
  median contour (blue line) and ground truth (black).</em></p>
</div>

The band is narrow where the data constrain the threshold and wide where they do not. A MAP
fit reports the same contours with no way to tell those two cases apart.

??? note "Nested bands: several levels in one call"

    Passing a sequence to `ci_prob` gives one interval per level, from one set of samples:

    ```python title="Nested bands from one call"
    --8<-- "docs/examples/wppm/nuts_inference_example.py:nested_bands"
    ```

    `lower`, `median` and `upper` all gain the same leading level axis, so
    `nested.lower[i]`, `nested.median[i]` and `nested.upper[i]` all belong to
    `nested.ci_prob[i]` and one loop draws every level.

    <div align="center">
      <img src="../plots/nuts_threshold_band_nested_full.jpg" width="520"/>
      <p><em>One reference point, drawn alone. Black: ground truth. Blue line: posterior
      median. Blue fills: the 50%, 80% and 95% bands, darkest where they overlap. Green:
      the reference point.</em></p>
    </div>

!!! warning "This band is pointwise, not simultaneous"
    The interval covers the true radius in each direction separately with probability
    $1-\alpha$. It does not say that the whole true contour lies inside the band with
    probability $1-\alpha$, which is a stronger statement needing a different construction.
    An 89% pointwise band is expected to miss some directions.
## Next steps

- **Check the posterior samples.** How well do the samples represent the posterior distribution?
  [This tutorial shows how to check the MCMC samples with the established library ArviZ](mcmc_diagnostics.md).
   We also look at which sampler arguments to change when one diganostic fails.
- **Predictions at new stimuli.** `WPPMPredictivePosterior` takes this posterior and gives
  the posterior predictive distribution over responses, averaging $P(\text{correct})$ over
  samples instead of over one point estimate.
- **Adaptive trial placement.** The spread shown here is what an acquisition function needs
  in order to choose where the next trial goes.
- **Other samplers.** Nothing downstream of the sampler is NUTS-specific: any psyphy
  [`InferenceEngine`](../../reference/inference.md#psyphy.inference.InferenceEngine)
  returning an `MCMCPosterior` with samples shaped `(n_chains, n_draws, *W_shape)` works with
  `covariance_to_arviz`, `WPPMCovarianceFieldSamples` and `WPPMPredictivePosterior`
  unchanged. Note that note all the diagnostics will carry over, since a sampler without trajectories has for example
  no divergences.

## Further reading

- API reference: [`NUTSSampler`](../../reference/inference.md#psyphy.inference.NUTSSampler),
  [`InferenceEngine`](../../reference/inference.md#psyphy.inference.InferenceEngine),
  [`MCMCPosterior`](../../reference/posterior.md#psyphy.posterior.MCMCPosterior),
  [`covariance_to_arviz`](../../reference/model.md#psyphy.model.covariance_to_arviz),
  [`WPPMCovarianceFieldSamples`](../../reference/model.md#psyphy.model.WPPMCovarianceFieldSamples),
  [`contour_radii`](../../reference/model.md#psyphy.model.contour_radii)
- MAP fitting and the model in more depth:
  [full WPPM fit](full_wppm_fit_example.md).
- [BlackJAX](https://blackjax-devs.github.io/blackjax/) for the sampler and its arguments.
- [ArviZ](https://python.arviz.org/) for the diagnostics, with its
  [getting started guide](https://python.arviz.org/en/latest/user_guide/getting_started.html)
  and the [arviz-stats API](https://python.arviz.org/projects/stats/en/latest/).

> Hoffman, M. D., & Gelman, A. (2014). *The No-U-Turn Sampler: adaptively setting path
> lengths in Hamiltonian Monte Carlo.* Journal of Machine Learning Research, 15, 1593-1623.
> [jmlr.org](https://jmlr.org/papers/v15/hoffman14a.html)
>
> Betancourt, M. (2017). *A Conceptual Introduction to Hamiltonian Monte Carlo.*
> [arXiv:1701.02434](https://arxiv.org/abs/1701.02434)
