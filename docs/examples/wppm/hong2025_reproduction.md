# Reproducing Hong et al. (2025)
This tutorial is accompanied by a
**runnable script:**
[`hong2025_reproduction.py`](https://github.com/flatironinstitute/psyphy/blob/main/docs/examples/wppm/hong2025_reproduction.py).


??? note "How to run the script"

    Everything on this page comes from one script. Pick a mode by how much
    compute you want to spend:

    ```bash
    # stages 1 and 2 only: the exact check and Figure 2B. <1 min on CPU.
    python hong2025_reproduction.py --skip-refit

    # add stage 3, the refit at the paper's settings. Wants a GPU.
    python hong2025_reproduction.py --mode full
    ```

    To check the code path runs on your laptop before committing to any of
    that:

    ```bash
    # a smoke test, not a reproduction: 500 trials and 20 steps
    # leave the fit essentially at its prior.
    python hong2025_reproduction.py --mode quick
    ```


Hong et al. measured how finely people can tell colors apart, across a whole
plane of colors rather than at a handful of points. This page reproduces their
central figure from their own published data, in three stages: an exact
check of the model's arithmetic, the threshold contours of Figure 2B, and a
refit from their raw trials to see whether we get the same final results.

**Who this is for**

- You want a worked example of psyphy on real data, with an external ground
  truth to check against.
- You know the paper and want to see how psyphy reproduces it.

No familiarity with the model is needed to start. The next section introduces it
at a high level, [the simulated-data tutorial](full_wppm_fit_example.md) goes
further, and the paper itself is the full reference:


> Hong, F., Bouhassira, R., Chow, J., Sanders, C., Shvartsman, M., Guan, P.,
> Williams, A. H., & Brainard, D. H. (2025). *Comprehensive characterization of
> human color discrimination thresholds.* eLife 14:RP108943.
> <https://doi.org/10.7554/eLife.108943.2>

---

## Background — what the Whishart Psychophysical Process Model (WPPM) is

Measuring a discrimination threshold the usual way means fixing one color and
asking, over many trials, how far a second color has to move before someone
notices the difference. That tells you about one color. Repeating it across a
whole plane of colors is impractical: too many locations and far too many trials, so we run into the curse of dimensionality.

The WPPM takes a different approach. It assumes the observer's internal noise
changes *smoothly* across color space, or more generally, the input space: nearby colors are confusable in similar
ways. That lets us fit one smooth field over the entire space instead of many
separate measurements, so every trial informs the whole picture. Once fit, we
can evaluate the model at any point in stimulus space, including those we haven't tested!

psyphy implements the Wishart Psychophysical Process Model (WPPM) in general form: any number of
stimulus dimensions, any task you can write a likelihood for. The color setup
here is only one configuration of it, which is why this page doubles as an external
check on psyphy and a worked example of the general pipeline. The WPPM approach carries beyond color to any domain where the noise
limiting performance varies smoothly across input space.

Hong et al. collect each judgement from the human subjects with an **oddity task**: on
every trial the observer sees three stimuli — two identical, one different —
and picks the odd one out. Chance is therefore 1/3, and the threshold is placed
at the usual midpoint between chance and perfect performance,
`P(correct) = 2/3`. That is the 66.7% contour this page reproduces.

---

## The result

Each ellipse is a *Just-Noticeable Difference (JND)* threshold contour around a reference
color at its center: the smallest color difference this observer can reliably
detect. Operationally, it is how far a comparison color must move from the
reference before they pick it out as the odd one 66.7% of the time. It is an
ellipse rather than a circle because sensitivity depends on *direction* — some
color changes are easier to see than others of the same magnitude. The
orientation and elongation of each ellipse are exactly what the WPPM estimates. We
can also see that the sizes of the ellipses increase as you move away from the origin
in the plot below, which corresponds to a gray stimulus. This is a reproduction of the
[Weber–Fechner law](https://en.wikipedia.org/wiki/Weber%E2%80%93Fechner_law).

See [Recovering Weber's Law](weber_law.md) for a worked example reproducing the
classic Weber's Law result on simulated one-dimensional data.


<div align="center">
    <img src="../plots/hong2025_thresholds.png"
         alt="Paper Figure 2B reproduced: 66.7%-correct discrimination threshold contours"
         width="620"/>
    <p><em>Colored ellipses are the contours we recover with psyphy; dashed gray
    are the published ones. Each ellipse takes the color of its own reference
    stimulus (center dot). The dimensions of the figure here are called model dimensions and are arbitrary in that they can result from any affine transformation of the input (RGB) space. </em></p>
</div>



## The whole recipe


```python title="Published data to threshold contours"
import jax
jax.config.update("jax_enable_x64", True)   # the authors used float64
import jax.numpy as jnp

from psyphy.data.published import hong2025
from psyphy.posterior import MAPPosterior, ThresholdConfig, WPPMPredictivePosterior

paths = hong2025.fetch(subject=1)                       # download from OSF
W_org = hong2025.load_reference_W(paths["weights"])     # the paper's fitted weights
coords, published = hong2025.load_sigma_table(paths["thres_ellipses"])

# Model: given weights W, how noisy is perception at each color?
model = hong2025.build_paper_model(mc_samples=2000)

# Parameter posterior: which W do we believe?
posterior = MAPPosterior({"W": W_org}, model)

# Search settings: how carefully to look for each threshold.
# These are the paper's own: 16 directions, 1000 distances along each.
config = ThresholdConfig(n_theta=16, n_length=1000)

# Predictive posterior: given what we believe about W, what do we predict here?
thresholds = WPPMPredictivePosterior(
    posterior,
    jnp.asarray(coords),                                # reference points only
    n_samples=1,
    threshold_pred=True,                                # ask for thresholds
    threshold_config=config,
).mean                                                  # -> (49, 2, 2)
```


The sections below will dive deeper into details, such as how to load the data or how to compute the thresholds.


---

## Data

Psyphy makes it easy to download the published data:

```python title="Download one observer's files"
--8<-- "docs/examples/wppm/hong2025_reproduction.py:fetch"
```

??? note "What each data file is, and how big"

    | File | Size | Used for |
    |---|---|---|
    | `trial_data_pooled_by_type_sub1.csv` | 1 MB | trials, for the refit |
    | `Bestfit_W_sub1.csv` | 212 KB | fitted weights, plus 120 bootstraps |
    | `Thres_ellipses_sub1.csv` | 320 KB | the 7×7 grid and published thresholds |
    | `Noise_ellipses_sub1.csv` | 68 MB | published $\Sigma_{\text{noise}}$ on a 103×103 grid |




`load_trials` loads in the published file and returns psyphy's `TrialData` object, so it will
work directly with our methods:

```python title="Load the trials"
--8<-- "docs/examples/wppm/hong2025_reproduction.py:load"
```

The published data holds 12,000 trials in two equal halves: 6,000 `AEPsych_*`
rows (5,100 adaptive placement plus 900 Sobol) used for fitting, and 6,000
`MOCS_*` rows held out for validation. By default `load_trials` loads only the
rows used for fitting. Pass `trial_types=("MOCS",)` for the held-out half, or
`trial_types=None` for all 12,000.

!!! warning "Fitting all 12,000 trials does not reproduce the paper"
    It gives a plausible result that is not the published one. This is why
    `load_trials` defaults to `trial_types=("AEPsych",)`.

For more information on how the authors did adaptive trial placement using the
library AEPsych, we refer the reader to the paper.

Two conventions psyphy handles for us:

- **Coordinates are already in the Chebyshev domain** `[-1, 1]`, so no
  normalization is needed.
- **Oddity trials are stored with `K=2`, not 3.** Each trial in the task presents three
  stimuli (reference, reference, comparison) but only **two distinct** ones,
  and `K` counts the distinct stimuli. The duplication lives in the likelihood,
  not in the stored data.

---

## Model

`build_paper_model()` assembles a WPPM from the hyperparameters used in the paper, which are stored in the dictionary `PAPER_HYPERPARAMS`

??? note "Paper -> psyphy parameter mapping"

    Most settings map one to one. The ones worth knowing:

    | Paper | psyphy | Note |
    |---|---|---|
    | `degree=5` | `basis_degree=4` | **Off-by-one.** Theirs counts basis *functions* (T₀…T₄); ours is the *maximum degree*. Same 5×5 grid. |
    | `variance_scale=3e-4` | same | psyphy's default is `4e-3` |
    | `diag_term=0` | same | psyphy's default is `1e-6`; theirs leaves Σ unregularized |
    | `mc_samples=2000`, `bandwidth=5e-3` | `OddityTaskConfig` | |
    | `learning_rate=1e-4`, `momentum=0.2`, `total_steps=1500`, 3 restarts | `MAPOptimizer` | refit only |

    The published weight tensor is `(5, 5, 2, 3)` , which is exactly psyphy's `params["W"]` layout.



---

## Thresholds (as in Paper Figure 2B)

The model is parameterized in $\Sigma_{\text{noise}}(x)$, the covariance of the observer's
_internal representation_. The paper reports **thresholds**, i.e., how much do we have to move in stimulus space, until the observer picks it out as the odd one 66.7% of the time. Those are different
objects! The map between them runs in two directions, and only the forward pass is easy:

- **Forward**: given the noise at two points, how often does the observer get
  the trial right? That is what the model computes directly.
- **Inverse**: given that they get it right two-thirds of the time, how far
  apart were the stimuli? That is what Figure 2B plots and it is the
  direction with no closed form.

Written out:

$$
\begin{aligned}
\text{forward (psyphy's OddityTask)}:\qquad
  & \Sigma_{\text{noise}}(x_{\text{ref}}),\ \Sigma_{\text{noise}}(x_{1})
  && \longrightarrow\ P(\text{correct}) \\[4pt]
\text{inverse (what Figure 2B plots)}:\qquad
  & P(\text{correct}) = \tfrac{2}{3}
  && \longrightarrow\ x_{1}
\end{aligned}
$$


There is no closed form for the inverse. For the 3-alternative oddity task the
observer is correct when the two identical stimuli are nearer to each other than
either is to the odd one:

$$
P(\text{correct}) \;=\; \Pr\!\left[\min(d_{02},\, d_{12}) > d_{01}\right]
$$

where $d_{ij}$ is the
[Mahalanobis distance](https://en.wikipedia.org/wiki/Mahalanobis_distance)
between the internal representations of stimuli $i$ and $j$. This is the distance that
measures separation in units of the noise itself, so a step counts as large only
relative to how noisy the representation is in that direction. That probability
has no analytic form, which is why the paper estimates it by
[Monte Carlo](https://en.wikipedia.org/wiki/Monte_Carlo_method) in the
first place. So we invert numerically:

1. Probe `n_theta` directions around each reference point.
2. Along each, evaluate `P(correct)` at `n_length` distances and keep the one
closest to 2/3. We thus have one boundary point per direction.
3. Fit an ellipse to those `n_theta` points. This step does have a closed-form solution and so can be done quickly.

Step 3 needs no optimizer — the ellipse fit is closed-form.

??? note "Why the ellipse fit is closed-form"

    A point at radius `r` in direction `u` satisfies $u^TΣ^{-1}u = 1/r^2$, which
    is **linear** in the three free entries of $Σ^{-1}$. So the fit is least
    squares over those three unknowns, followed by a single matrix inverse to
    recover $Σ$ itself. No iteration, and nothing that can fail to converge.

```python title="Threshold inversion at every published reference point"
--8<-- "docs/examples/wppm/hong2025_reproduction.py:thresholds"
```


??? note "Why threshold mode takes and returns different shapes"

    `threshold_pred` selects which direction of the map above you are asking
    for, so both the input and the output change shape with it.

    | | `threshold_pred=False` | `threshold_pred=True` (used here) |
    |---|---|---|
    | **Direction** | forward | inverse |
    | **`X` you pass** | assembled trials, `(n_test, k_stimuli, input_dim)` | bare reference points, `(n_test, input_dim)` |
    | **output you get** | probability correct per trial, `(n_test,)` | covariance per reference point, `(n_test, input_dim, input_dim)` |

    **Why bare points go in.** Normally you supply the comparison stimulus and
    the model scores that pair. In threshold mode, *finding* the comparison is
    what we want: the threshold is the distance at which `P(correct)` reaches
    2/3. So, supplying one would be handing over the answer. Instead, it generates its own by sweeping `n_theta` directions by `n_length` distances around
    each reference (that is what `ThresholdConfig` controls).




```python title="Compute settings"
--8<-- "docs/examples/wppm/hong2025_reproduction.py:threshold_settings"
```

We run the inversion at the paper's own settings (16 directions, 1,000
distances per direction, 2,000 Monte Carlo samples).

### Plotting it
S3
Both contour fields go on one axes in a single
[`plot_ellipses`](../../reference/viz.md) call: published dashed underneath, ours on
top colored by reference stimulus:

```python title="The plotting call"
--8<-- "docs/examples/wppm/hong2025_reproduction.py:plot_call"
```

`scale` comes from `auto_scale(coords, thres_published)` and `colors` from
`hong2025.w2d_to_rgb(coords, M)`. We recommend only passing  **one** `scale` for both fields because otherwise the comparison independently scaled fields cannot be
compared by eye.

For more detail on this plotting function, including how to use per-ellipse colors
and posterior draws, see [Plotting ellipse fields](../viz/ellipse_plots.md).


---

That reproduces the published figure, but we can test for numeric reproducibility,
not just visual agreement. The process described above has many steps where
error can be introduced.

The next two sections take those away in order. First a fully deterministic
check: published weights straight through psyphy's covariance field, with no
optimizer and no sampling anywhere. Then the refit, with both back in; so that
if *that* disagrees, we already know the disagreement is the optimizer's and
not the model's.


## Exact check
### does psyphy build the same covariance field Hong et al published?

This is fully
deterministic.

```python title="Published weights through psyphy's covariance field"
--8<-- "docs/examples/wppm/hong2025_reproduction.py:stage1"
```


In the above, we're simply computing the difference between our computed
covariances and the values shared by the paper's authors, for all 42,436
ellipses. The maximum value of the differences are shown below:

```
max |diff|   : 6.778e-09
mean |diff|  : 2.538e-09
```

 Our values round to theirs exactly in 96% of
cases and agree to within one unit in the last printed digit in 100%. **This is
agreement to the precision the file can express.**

This runs as a test (`test_covariance_field_matches_published_sigma_noise`),
skipped automatically when the data has not been downloaded, so CI stays
network-free.

---

## Refit
### Does psyphy's fit find the paper's covariance field?

Everything above started from the paper's weights. The stronger question is: given
only the paper's **data**, does psyphy's fit find the paper's covariance field?

The following block of code refits the WPPM's weights from the raw data, computes the covariance field  and then plots resulting ellipses. Looking at the alignment of the ellipses in the figure below, the answer to that question is yes.

```python title="MAP fit with the paper's optimizer settings"
--8<-- "docs/examples/wppm/hong2025_reproduction.py:fit"
```




<div align="center">
    <img src="../plots/hong2025_full_ellipses.png"
         alt="Sigma_noise: the published weights' field vs a full-settings psyphy refit"
         width="560"/>
    <p><em><span class="arithmatex">\(\Sigma_{\text{noise}}(x)\)</span> for subject 1 (CH), in the same convention as the figure at
    the top of this page: dashed gray is the field from the authors' published
    weights, colored solid is our own MAP refit, each ellipse taking the color of
    its reference stimulus. This is the paper's supplementary Figure S3.
    <br/><br/>
    Note: These ellipses look much like the ones at the top of the page, but they are a
    different quantity.
    <span class="arithmatex">\(\Sigma_{\text{noise}}(x) = U(x)U(x)^{\top} + \delta I\)</span>
    is the covariance of the observer's internal representation at stimulus
    <span class="arithmatex">\(x\)</span> — the field the WPPM is
    parameterized in, read off at each grid point. No task enters it. The
    contours at the top are <span class="arithmatex">\(\Sigma_{\text{thres}}\)</span>, one step downstream: <span class="arithmatex">\(\Sigma_{\text{noise}}\)</span> at a reference
    and a comparison feeds the oddity likelihood to give P(correct), and that map
    is inverted for the displacement at which P(correct) = 2/3. We use the same grid and
    plotting convention, but <span class="arithmatex">\(\Sigma_{\text{noise}}\)</span> is the model's parameters evaluated,
    while <span class="arithmatex">\(\Sigma_{\text{thres}}\)</span> is behavior predicted from them at a criterion, here 2/3.</em></p>
</div>


!!! warning "Scope"
    These results are for one subject (CH, 1 of 8) and a single run on one GPU.
    They were not repeated for seed stability and not run for the other seven
    subjects. Read this as "the fitting pipeline reproduces the paper for this
    subject", not as a claim about all eight.


---

## Runtimes

The full refit requires **~16 min** on a single GPU. See the following table for a breakdown of how long each step takes.

??? note "Measured runtimes, step by step"

    CPU figures are an Apple Silicon laptop (M5); GPU is one A100 unless otherwise noted.

    | Step | Hardware | Wall clock | Details |
    |---|---|---|---|
    | Exact covariance check | CPU | seconds | 10,609 points, deterministic |
    | **Thresholds, paper settings** | CPU | **~11 min** | 49 refs, `n_theta=16`, `n_length=1000`, `mc=2000` (13.4 s per ref) |
    | Thresholds, `fast` preset | CPU | 20–23 s | `n_length=300`, `mc=500` — smoke tests only |
    | **Refit — full** | 1 GPU | **~16 min** | 6,000 trials, 1,500 steps, `mc=2000`, 3 restarts |
    | The paper's own run | H100 | 14 h | **one subject**: main fit + 120 bootstrap refits |

    The 14-hour figure is per observer, not for the whole paper. The WPPM is fit
    separately for each participant, and the 120 bootstraps resample that
    participant's own trials, so all eight observers is roughly eight times
    that.


---

## Watch out for

- **$\Sigma_{\text{noise}}$ and $\Sigma_{\text{thres}}$ are different things.** The thresholds
  above are $\Sigma_{\text{thres}}$, as plotted in Figure 2B; the exact check and the
  refit compare $\Sigma_{\text{noise}}$, the noise field, which is plotted in
  supplementary Figure S3. Both arrive as `(49, 2, 2)` stacks on the same grid,
  which makes them easy to conflate.
- **Monte Carlo results are not bit-reproducible across platforms.** The exact
  check is exact anywhere; thresholds and refits reproduce to a neighborhood.
- **Loss values are not comparable to the paper's.** psyphy's `Prior.log_prob`
  drops a constant, which the paper keeps (still  identical gradients but different numbers)


---

## See also

- [Full WPPM fit (simulated data)](full_wppm_fit_example.md) — same machinery with ground truth available.
- [Quick start](quick_start.md) — the minimal version.
- [Plotting ellipse fields](../viz/ellipse_plots.md) — `plot_ellipses` on its own, with synthetic data.
- `psyphy.data.published.hong2025` in [Data](../../reference/data.md); `WPPMPredictivePosterior` and `ThresholdConfig` in [Posterior](../../reference/posterior.md).
