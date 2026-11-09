# Reproducing Hong et al. (2025)
The companion script is
[`hong2025_reproduction.py`](https://github.com/flatironinstitute/psyphy/blob/main/docs/examples/wppm/hong2025_reproduction.py).


??? note "How to run the script"

    The figures need matplotlib, which psyphy does not install by default:

    ```bash
    pip install 'psyphy[viz]'     # psyphy + matplotlib, all this page needs
    pip install 'psyphy[cuda]'    # also, for the refit: JAX's CUDA build
    ```

    `viz` installs matplotlib only. The refit at the paper's settings needs an
    NVIDIA GPU. Everything else runs on a laptop.

    Pick a mode by how much compute you want to spend:

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


We reproduce Hong et al. (2025) with psyphy, starting from their raw trials: we refit the
model, invert it to discrimination thresholds, and compare against the figure they
published. Our threshold contours fall inside the authors' own 95% bootstrap interval at
all 49 of their reference points.

Disclaimer: Their experiment was about color, so this page is too, but the model and psyphy's implementation generalize! See
[Scope](#scope) below.

We build the reproduction up one question at a time, so a disagreement at any point tells
us where it came from:

1. Given the published weights, do we compute the same **covariance field**?
2. Given the published weights, do we recover the same **threshold contours**
   (Figure 2B)?
3. Given only the raw trials, do we refit the same **weights**?
4. Putting 2 and 3 together: from raw trials alone, do we reproduce the same
   **published figure 2B**?
5. And is that agreement **good enough**, measured against the paper's own
   bootstrap confidence interval?

Questions 1 and 2 hand psyphy the paper's own weights, so the expensive optimization
doesn't run. If they fail, the bug is in our model code. Question 3 is the first that fits
anything, and fitting is both the slow part and the part with the most ways to go wrong.

This is a worked example of psyphy on real data, with an external ground truth to check
against. No familiarity with the model is needed to start. The next section introduces it
at a high level. See the paper itself for the full reference:


> Hong, F., Bouhassira, R., Chow, J., Sanders, C., Shvartsman, M., Guan, P.,
> Williams, A. H., & Brainard, D. H. (2025). *Comprehensive characterization of
> human color discrimination thresholds.* eLife 14:RP108943.
> <https://doi.org/10.7554/eLife.108943.2>

---

## Background: the Wishart Process Psychophysical Model (WPPM)

Measuring a discrimination threshold the usual way means fixing one color and
asking, over many trials, how far a second color has to move before someone
notices the difference. That gives the threshold at one color. Repeating it across a whole
plane of colors takes too many locations and far too many trials, which is the curse of
dimensionality.

The WPPM assumes the observer's internal noise changes smoothly across color space: nearby
colors are confusable in similar ways. That lets us fit one smooth field over the entire
space instead of many separate measurements, so every trial informs the whole picture. Once
fit, we can evaluate the model at any point in stimulus space, including those we haven't
tested.

Hong et al. collect each judgment from their participants with an **oddity task**: on
every trial the observer sees three stimuli (two identical, one different)
and picks the odd one out. Chance is therefore 1/3, and the threshold is placed
at the usual midpoint between chance and perfect performance,
`P(correct) = 2/3`.



---

<a id="scope"></a>

!!! note "Scope"
    We describe the WPPM in terms of color because that is what Hong et al.
    measured. The WPPM itself is not specific to color: it models noise varying
    smoothly over any stimulus space, for any task you can write a likelihood
    for. See [Recovering Weber's Law](weber_law.md) for a one-dimensional
    example, or [the simulated-data walkthrough](full_wppm_fit_example.md).

## The result

Each ellipse is a just-noticeable difference (JND) threshold contour around the reference
color at its center: how far a comparison color must move from the reference before the
observer picks it out as the odd one 66.7% of the time. Some color changes are easier to
see than others of the same magnitude, so the contour is an ellipse, and its orientation
and elongation are what the WPPM estimates. In the plot below the ellipses grow away from
the origin, which corresponds to a gray stimulus. This is consistent with the
[Weber-Fechner law](https://en.wikipedia.org/wiki/Weber%E2%80%93Fechner_law).

See [Recovering Weber's Law](weber_law.md) for a worked example reproducing the
classic Weber's Law result on simulated one-dimensional data.


<div align="center">
    <img src="../plots/hong2025_full_thresholds_end_to_end.png"
         alt="Paper Figure 2B reproduced end to end, from raw trials through a psyphy refit"
         width="620"/>
    <p><em>Paper Figure 2B, reproduced end to end for subject 1 (CH). Colored
    ellipses are the contours psyphy recovers, dashed gray the published ones. Each
    ellipse takes the color of its own reference stimulus (center dot). Only the raw
    trials come from the paper: psyphy fits the model's weights to them and inverts the
    oddity task to turn the resulting noise field into 66.7%-correct thresholds. The axes
    are model dimensions, which are arbitrary up to an affine transformation of the input
    (RGB) space.</em></p>
</div>


## In a nutshell

The block below downloads one observer's data, loads the paper's fitted weights and turns
them into threshold contours shown above. Because we use the fitted weights, this runs on a laptop. The sections after it go
through the same steps in detail and in particular, we refit the model, which requires a GPU.

```python title="Published data to threshold contours"
--8<-- "docs/examples/wppm/hong2025_recipe.py:recipe"
```


---

## Data

`hong2025.fetch` downloads one observer's files:

```python title="Download one observer's files"
--8<-- "docs/examples/wppm/hong2025_reproduction.py:fetch"
```

??? note "What each data file is, and how big"

    | File | Size | Used for |
    |---|---|---|
    | `trial_data_pooled_by_type_sub1.csv` | 1 MB | trials, for the refit |
    | `Bestfit_W_sub1.csv` | 212 KB | fitted weights, plus 120 bootstraps |
    | `Thres_ellipses_sub1.csv` | 320 KB | the 7x7 grid and published thresholds |
    | `Noise_ellipses_sub1.csv` | 68 MB | published $\Sigma_{\text{noise}}$ on a 103x103 grid |


`load_trials` reads the published file and returns a psyphy
[`TrialData`](../../reference/data.md#psyphy.data.TrialData):

```python title="Load the trials"
--8<-- "docs/examples/wppm/hong2025_reproduction.py:load"
```

The published data holds 12,000 trials in two equal halves: 6,000 `AEPsych_*`
rows (5,100 adaptive placement plus 900 Sobol) used for fitting, and 6,000
`MOCS_*` rows held out for validation. `load_trials` returns only the fitting rows by
default, and everything here uses them. Pass `trial_types=("MOCS",)` for the held-out half, or
`trial_types=None` for all 12,000.

!!! warning "Fitting all 12,000 trials does not reproduce the paper"
    It gives a plausible result that is not the published one.

The paper describes how the authors placed trials adaptively with AEPsych.

??? note "Conventions in the loaded data"

    psyphy stores trials as a `stimuli` array of shape `(N, K, d)` (trials, stimuli
    per trial, stimulus dimensions) alongside `responses`.

    **Oddity trials are stored with `K=2`, not 3.** Each trial shows three
    stimuli (reference, reference, comparison) but only two distinct ones, and `K`
    counts distinct stimuli. So `data.stimuli` comes back `(6000, 2, 2)` for a
    three-interval task. The oddity likelihood adds the repeated reference.

---

## Model

`build_paper_model()` assembles a WPPM from the settings the paper used, which
we transcribed once into `PAPER_HYPERPARAMS`.

??? note "The paper's hyperparameters in full"

    Grouped by what each one controls.

    ```python title="psyphy.data.published.hong2025"
    --8<-- "src/psyphy/data/published/hong2025.py:hyperparams"
    ```

---

## Exact check

The first question we ask: Does psyphy build the same covariance field as Hong et al.? For this, we hand psyphy the paper's published
weights and compute the covariance field. If this
disagrees, the problem is in the model implementation itself, as so far no optimizer has been involved.

```python title="Published weights through psyphy's covariance field"
--8<-- "docs/examples/wppm/hong2025_reproduction.py:stage1"
```


In the above, we're simply computing the difference between our computed
covariances and the values shared by the paper's authors, at all 10,609 grid points
(42,436 matrix entries). The maximum and mean absolute differences are:

```
max |diff|   : 6.778e-09
mean |diff|  : 2.538e-09
```

Our values agree to all published digits in 96% of the cases and, in the final 4%, only
differ by +/- 1 in the last printed digit.

The same comparison runs as a test, `test_covariance_field_matches_published_sigma_noise`,
which is skipped when the data has not been downloaded.

---

## Thresholds (as in Paper Figure 2B)

The model is parameterized in $\Sigma_{\text{noise}}(x)$, the covariance of the observer's
internal representation. The paper reports **thresholds**: how far we have to move in
stimulus space until the observer picks out the comparison 66.7% of the time. The map
between the two runs in two directions, and only the forward one is easy:

- **Forward**: given the noise at two points, how often does the observer get
  the trial right? That is what the model computes directly.
- **Inverse**: given that they get it right two-thirds of the time, how far
  apart were the stimuli? That is what Figure 2B plots.

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
between the internal representations of stimuli $i$ and $j$. This probability has no
analytic form, and the paper estimates it by
[Monte Carlo](https://en.wikipedia.org/wiki/Monte_Carlo_method). We compute the inverse
numerically, following the paper's procedure:

1. Probe `n_theta` directions around each reference point.
2. Along each, evaluate `P(correct)` at `n_length` distances and keep the one
closest to 2/3. We thus have one boundary point per direction.
3. Fit an ellipse to those `n_theta` points. This step has a closed-form solution.

In psyphy,
[`WPPMPredictivePosterior`](../../reference/posterior.md#psyphy.posterior.WPPMPredictivePosterior)
with `threshold_pred=True` computes this inverse.

??? note "Why the ellipse fit is closed-form"

    A point at radius `r` in direction `u` satisfies $u^T\Sigma^{-1}u = 1/r^2$, which
    is linear in the three free entries of $\Sigma^{-1}$. So, the fit is least
    squares over those three unknowns. We then do a single matrix inverse to
    recover $\Sigma$.

We run the inversion at the paper's own settings (16 directions, 1,000
distances along each, 2,000 Monte Carlo samples per evaluation). That
costs about 11 minutes on CPU for the 49 reference points.

??? note "Threshold settings"

    ```python
    --8<-- "docs/examples/wppm/hong2025_reproduction.py:threshold_settings"
    ```

    `--mode quick` uses 300 distances and 500 Monte Carlo samples instead of 1,000
    and 2,000. It runs the same 49 reference points in about 20 seconds instead of
    11 minutes, and is only for checking that the code path works. The script prints
    which preset is in effect when it starts.



Loading and the model are the same as in the short version above. The new part is asking
the predictive posterior for thresholds:

```python title="Threshold inversion at every published reference point"
--8<-- "docs/examples/wppm/hong2025_reproduction.py:thresholds"
```


---

## Refitting the model from scratch

Everything above started from the paper's weights. Here, we start from the paper's data
only and ask: does psyphy's fit find the paper's covariance field?

The following block of code refits the WPPM's weights from the raw data, computes the covariance field and then plots resulting ellipses.
Note again, that this is the compute-heavy part and requires a GPU.

Looking at the alignment of the ellipses in the figure below, the answer to that question is yes.

```python title="MAP fit with the paper's optimizer settings"
--8<-- "docs/examples/wppm/hong2025_reproduction.py:fit"
```

The fit is the only part that needs a GPU, so we write the weights to disk.

```python title="Keep the fitted weights"
--8<-- "docs/examples/wppm/hong2025_reproduction.py:save_fit"
```


<div align="center">
    <img src="../plots/hong2025_full_ellipses.png"
         alt="Sigma_noise: the published weights' field vs a full-settings psyphy refit"
         width="560"/>
    <p><em><span class="arithmatex">\(\Sigma_{\text{noise}}(x)\)</span> for subject 1 (CH): dashed gray is the field
    from the authors' published weights, red is our own MAP refit. This is the
    paper's supplementary Figure S3.
    <br/><br/>
    Note: These ellipses look much like the ones at the top of the page, but they are a
    different quantity.
    <span class="arithmatex">\(\Sigma_{\text{noise}}(x) = U(x)U(x)^{\top} + \delta I\)</span>
    is the covariance of the observer's internal representation at stimulus
    <span class="arithmatex">\(x\)</span>; the field the WPPM is
    parameterized in, read off at each grid point. No task enters it. The
    contours at the top are <span class="arithmatex">\(\Sigma_{\text{thres}}\)</span>, one step downstream: <span class="arithmatex">\(\Sigma_{\text{noise}}\)</span> at a reference
    and a comparison feeds the oddity likelihood to give P(correct), and that map
    is inverted for the displacement at which P(correct) = 2/3. We use the same grid and
    plotting convention, but <span class="arithmatex">\(\Sigma_{\text{noise}}\)</span> is the model's parameters evaluated,
    while <span class="arithmatex">\(\Sigma_{\text{thres}}\)</span> is behavior predicted from them at a criterion, here 2/3.</em></p>
</div>


---

## End to end: from raw trials to Figure 2B

This produces the figure at the top of the page.

The threshold inversion above started from the authors' published weights, and the refit
compared only noise fields. Here we combine them:

*raw trials -> fit weights -> derive threshold contours -> the published figure 2B*


```python title="Invert our own fitted weights"
--8<-- "docs/examples/wppm/hong2025_reproduction.py:end_to_end"
```

??? note "Plotting it"

    Both contour fields go onto one axes in a single
    [`plot_ellipses`](../../reference/viz.md#psyphy.viz.ellipses.plot_ellipses) call:
    published dashed underneath, ours on top, each ellipse colored by its own
    reference stimulus.

    ```python
    --8<-- "docs/examples/wppm/hong2025_reproduction.py:plot_call"
    ```

`scale` comes from `auto_scale(coords, thres_published)` and `colors` from
`hong2025.w2d_to_rgb(coords, M)`, the monitor calibration published with the
data. For per-ellipse colors, posterior draws and the rest of the API, see
[Plotting ellipse fields](../viz/ellipse_plots.md).


<div align="center">
    <img src="../plots/hong2025_full_thresholds_end_to_end.png"
         alt="End-to-end: threshold contours from our own refit vs the published ones"
         width="520"/>
    <p><em>66.7%-correct threshold contours for subject 1 (CH), computed from the
    weights we fit to the raw trials. Dashed gray is the authors' published inversion,
    colored solid is ours, each ellipse taking the color of its reference
    stimulus.</em></p>
</div>



---

## Comparison with the paper's bootstrap interval

To judge how close is close enough, we use the authors' own bootstrap. They resampled the
trials 120 times, refit the model to each, and kept the 114 fits (95% of 120) that came out
most like their original. The spread of those 114 contours is their 95% confidence
interval. We check whether the thresholds from psyphy's fit lie inside it.

The figure below shows that our fit is indistinguishable from their run-to-run variation at all 49 reference points and every direction tested, and in that
sense *psyphy's refit is indistinguishable from their fit*.

<div align="center">
    <img src="../plots/hong2025_bootstrap_envelope.png"
         alt="Our threshold contours against the paper's 95% bootstrap confidence interval"
         width="620"/>
    <p><em>Our end-to-end contours against the paper's own 95% bootstrap interval
    for subject 1 (CH). The gray band is the 114 retained bootstrap refits,
    dashed gray the published fit, colored solid ours.</em></p>
</div>

??? note "Plotting it:"

    `plot_ellipses` takes a whole stack of fields at once, so all 114 retained
    refits go on in a single call. The published fit and ours are drawn over
    them in the usual convention.

    ```python
    --8<-- "docs/examples/wppm/hong2025_reproduction.py:envelope_plot"
    ```

!!! warning "Scope"
    These results are for one subject (CH, 1 of 8) and a single run on one GPU.
    They were not repeated for seed stability and not run for the other seven
    subjects.


---

## Runtimes

The full refit takes about 16 min on a single GPU. The table breaks the runtime down by
step.

??? note "Measured runtimes:"

    CPU figures are an Apple Silicon laptop (M5); GPU is one A100 unless otherwise noted.

    | Step | Hardware | Wall clock | Details |
    |---|---|---|---|
    | Exact covariance check | CPU | seconds | 10,609 points, deterministic |
    | Thresholds, paper settings | CPU | ~11 min | 49 refs, `n_theta=16`, `n_length=1000`, `mc=2000` (13.4 s per ref) |
    | Thresholds, `quick` preset | CPU | 20–23 s | `n_length=300`, `mc=500`, smoke tests only |
    | Refit, full | 1 GPU | ~16 min | 6,000 trials, 1,500 steps, `mc=2000`, 3 restarts of ~317 s |
    | The paper's own run | H100 | 14 h | one subject: main fit + 120 bootstrap refits |

    The 14-hour figure is per observer, not for the whole paper. The WPPM is fit
    separately for each participant, and the 120 bootstraps resample that
    participant's own trials, so all eight observers is roughly eight times
    that.


---

## Watch out for

- **$\Sigma_{\text{noise}}$ against $\Sigma_{\text{thres}}$.** The thresholds
  above are $\Sigma_{\text{thres}}$, as plotted in Figure 2B; the exact check and the
  refit compare $\Sigma_{\text{noise}}$, the noise field, which is plotted in
  supplementary Figure S3. Both arrive as `(49, 2, 2)` stacks on the same grid,
  which makes them easy to conflate.
- **Seeds across machines.** The same seed gives the same answer on the same machine, but
  not necessarily on a different one, because XLA can reorder floating-point arithmetic
  ([JAX FAQ](https://docs.jax.dev/en/latest/faq.html#jit-changes-the-exact-numerics-of-outputs)).
  The exact check is unaffected, since it compares against a table rounded to 8 decimals.
  The thresholds and the refit can differ in their low-order digits between a laptop and a
  GPU.
- **Loss values.** psyphy's `Prior.log_prob` drops a constant that the paper keeps, so loss
  values are not comparable to the paper's. The gradients are identical.


---

## See also

- [Full WPPM fit (simulated data)](full_wppm_fit_example.md): more on the model math with ground truth available
- [Quick start](quick_start.md): the minimal version
- [Plotting ellipse fields](../viz/ellipse_plots.md): `plot_ellipses` on its own, with synthetic data
- `psyphy.data.published.hong2025` in [Data](../../reference/data.md); `WPPMPredictivePosterior` and [`ThresholdConfig`](../../reference/posterior.md#psyphy.posterior.ThresholdConfig) in [Posterior](../../reference/posterior.md).
