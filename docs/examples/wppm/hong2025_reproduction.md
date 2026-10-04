# Reproducing Hong et al. (2025)
This tutorial is accompanied by a
**runnable script:**
[`hong2025_reproduction.py`](https://github.com/flatironinstitute/psyphy/blob/main/docs/examples/wppm/hong2025_reproduction.py).


??? note "How to run the script"

    An install of psyphy on its own is not enough: the figures here need a
    plotting backend, which psyphy does not depend on.

    ```bash
    pip install 'psyphy[viz]'     # psyphy + matplotlib, all this page needs
    pip install 'psyphy[cuda]'    # also, for the refit: JAX's CUDA build
    ```

    `viz` is matplotlib and nothing else. (There is also a broader `examples`
    extra that adds seaborn and JupyterLab for the other pages; this one does
    not need it.) The refit at the paper's settings wants an NVIDIA GPU .
    Everything else on this page runs on a laptop.

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


Hong et al. measured how finely people can tell colors apart, for more than
just a handful of colors. This page reproduces their central figure end to
end: starting from their raw trial data, psyphy refits the model, inverts it
to discrimination thresholds, and lands inside the authors' own bootstrap
confidence interval.

**How this tutorial is laid out** First, we introduce the task and show the headline
result: the paper's Figure 2B, reproduced. Then we cover the practical parts, e.g.,
loading the published data, and building paper's model. The reproduction itself is then built up one question at a
time, so that a disagreement at any point tells us where it came from:

1. Given the published weights, do we compute the same **covariance field**?
2. Given the published weights, do we recover the same **threshold contours**
   (Figure 2B)?
3. Given only the raw trials, do we refit the same **weights**?
4. Putting 2 and 3 together: from raw trials alone, do we reproduce the same
   **published figure 2B**?
5. And is that agreement **good enough**, measured against the paper's own
   bootstrap confidence interval?

That order is backwards on purpose. Questions 1 and 2 hand psyphy the paper's
own weights, so no optimizer ever runs. If they fail, the bug is in our model
code; there is nowhere else it could be coning from. Question 3 is the first that fits
anything, and fitting is both the slow part and the part with the most ways to
go wrong. Checking the cheap, deterministic parts first means that if question 3 disagrees, the optimizer is the only suspect left.

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
changes *smoothly* across color space: nearby colors are confusable in
similar ways. That lets us fit one smooth field over the entire space instead of many
separate measurements, so every trial informs the whole picture. Once fit, we
can evaluate the model at any point in stimulus space, including those we haven't tested!

psyphy implements the Wishart Psychophysical Process Model (WPPM) in general form: any number of
stimulus dimensions, any task you can write a likelihood for. The color setup
here is only one configuration of it, which is why this page doubles as an external
check on psyphy and a worked example of the general pipeline. The WPPM approach carries beyond color to any domain where the noise
limiting performance varies smoothly across the stimulus space.

Hong et al. collect each judgement from the human subjects with an **oddity task**: on
every trial the observer sees three stimuli (two identical, one different)
and picks the odd one out. Chance is therefore 1/3, and the threshold is placed
at the usual midpoint between chance and perfect performance,
`P(correct) = 2/3`. That is the 66.7% contour this page reproduces.



---

!!! note "Scope"
    For this tutorial we will describe the WPPM in terms of color, because that is what Hong et al.
    measured. The WPPM itself is not specific to color: it models noise varying
    smoothly over any stimulus space, for any task you can write a likelihood
    for. See [Recovering Weber's Law](weber_law.md) for a one-dimensional
    example, or [the simulated-data walkthrough](full_wppm_fit_example.md).

## The result

Each ellipse is a *Just-Noticeable Difference (JND)* threshold contour around a reference
color at its center: the smallest color difference this observer can reliably
detect. Operationally, it is how far a comparison color must move from the
reference before they pick it out as the odd one 66.7% of the time. It is an
ellipse rather than a circle because sensitivity depends on *direction*; some
color changes are easier to see than others of the same magnitude. The
orientation and elongation of each ellipse are exactly what the WPPM estimates. We
can also see that the sizes of the ellipses increase as you move away from the origin
in the plot below, which corresponds to a gray stimulus. This is a reproduction of the
[Weber–Fechner law](https://en.wikipedia.org/wiki/Weber%E2%80%93Fechner_law).

See [Recovering Weber's Law](weber_law.md) for a worked example reproducing the
classic Weber's Law result on simulated one-dimensional data.


<div align="center">
    <img src="../plots/hong2025_full_thresholds_end_to_end.png"
         alt="Paper Figure 2B reproduced end to end, from raw trials through a psyphy refit"
         width="620"/>
    <p><em>Paper Figure 2B, reproduced end to end for subject 1 (CH). Colored
    ellipses are the contours psyphy recovers; dashed gray are the published
    ones. Each ellipse takes the color of its own reference stimulus (center
    dot). Nothing published enters this chain except the raw trials: psyphy
    fits the model's weights from those trials, inverts the oddity task to turn
    the resulting noise field into 66.7%-correct thresholds, and the result is
    what you see. The axes are model dimensions, arbitrary up to an affine
    transformation of the input (RGB) space.</em></p>
</div>


## The whole recipe

The block below is the short version: download one observer's data, load the
paper's fitted weights, and turn them into threshold contours. It runs as it
stands, on a laptop. The sections after it go through the same steps slowly,
and add the refit that produces the figure at the top of this page, and
reproduce the fit from scratch with psyphy's implementation.

```python title="Published data to threshold contours"
--8<-- "docs/examples/wppm/hong2025_recipe.py:recipe"
```


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
    | `Thres_ellipses_sub1.csv` | 320 KB | the 7x7 grid and published thresholds |
    | `Noise_ellipses_sub1.csv` | 68 MB | published $\Sigma_{\text{noise}}$ on a 103x103 grid |


`load_trials` loads in the published file and returns psyphy's `TrialData` object, so it will
work directly with our methods:

```python title="Load the trials"
--8<-- "docs/examples/wppm/hong2025_reproduction.py:load"
```

The published data holds 12,000 trials in two equal halves: 6,000 `AEPsych_*`
rows (5,100 adaptive placement plus 900 Sobol) used for fitting, and 6,000
`MOCS_*` rows held out for validation. For this tutorial and the figures, we
only use the rows used for fitting (by default `load_trials` loads only the
rows used for fitting). Pass `trial_types=("MOCS",)` for the held-out half, or
`trial_types=None` for all 12,000.

!!! warning "Fitting all 12,000 trials does not reproduce the paper"
    It gives a plausible result that is not the published one.

For more information on how the authors did adaptive trial placement using the
library AEPsych, we refer the reader to the paper.

??? note "Two conventions worth knowing when you inspect the loaded data"

    psyphy stores trials as a `stimuli` array of shape `(N, K, d)` — trials x
    stimuli per trial x stimulus dimensions — alongside `responses`; see
    [`TrialData`](../../reference/data.md).

    **Oddity trials are stored with `K=2`, not 3.** Each trial shows three
    stimuli (reference, reference, comparison)but only **two distinct** ones,
    and `K` counts distinct stimuli. So `data.stimuli` comes back `(6000, 2, 2)`
    for a three-interval task. The repetition is applied inside the oddity
    likelihood rather than stored on every row.

---

## Model

`build_paper_model()` assembles a WPPM from the settings the paper used, which
we transcribed once into `PAPER_HYPERPARAMS`.

??? note "The paper's hyperparameters, in full"

    Grouped by what each one controls. These are read straight from the
    library, so the page cannot drift from the values the model is built with.

    ```python title="psyphy.data.published.hong2025"
    --8<-- "src/psyphy/data/published/hong2025.py:hyperparams"
    ```


!!! warning "One convention differs: `degree` counts basis functions, `basis_degree` is the maximum degree"
    The paper builds `WishartProcessModel(5, 2, 1, 3e-4, 0.4, 0)`, where
    `degree=5` is the *number* of Chebyshev basis functions, $T_0$ through
    $T_4$. psyphy's `basis_degree` is instead the *highest degree* used, so the
    same model is `basis_degree=4`. Both describe the same $5 \times 5$
    coefficient grid.


---

## Exact check
### does psyphy build the same covariance field Hong et al published?

With the data loaded and the model built, we start with the question that has
no moving parts. Hand psyphy the paper's own weights and ask it for the
covariance field: no optimizer and nothing random. If this
disagrees, the problem is in the model implementation itself, and everything
downstream would be built on sand.

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

 Our values agree to all published didgits in 96% of the
cases and, in the final 4%, only differ by +/- 1 in the last printed digit.
 **This is agreement to the precision the file can express.**

This runs as a test (`test_covariance_field_matches_published_sigma_noise`),
skipped automatically when the data has not been downloaded, so CI stays
network-free.

---

That settles the model implementation: given the same weights, psyphy builds
the same field. The next question is whether we can turn that field into the
thresholds the paper actually reports.

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
first place. So we have to compute the inverse numerically following
the procedure given in the paper:

1. Probe `n_theta` directions around each reference point.
2. Along each, evaluate `P(correct)` at `n_length` distances and keep the one
closest to 2/3. We thus have one boundary point per direction.
3. Fit an ellipse to those `n_theta` points. This step does have a closed-form solution and so can be done quickly.

Step 3 needs no optimizer, the ellipse fit is closed-form.

To compute this inverse using psyphy, we construct the `WPPMPredictivePosterior`
object with the ``threshold_pred`` argument set to ``True``, passing it the relevant arguments.

??? note "Why the ellipse fit is closed-form"

    A point at radius `r` in direction `u` satisfies $u^TΣ^{-1}u = 1/r^2$, which
    is **linear** in the three free entries of $Σ^{-1}$. So the fit is least
    squares over those three unknowns, followed by a single matrix inverse to
    recover $Σ$ itself. No iteration, and nothing that can fail to converge.

We run the inversion at the paper's own settings — 16 directions, 1,000
distances along each, 2,000 Monte Carlo samples per evaluation:

```python title="Compute settings"
--8<-- "docs/examples/wppm/hong2025_reproduction.py:threshold_settings"
```

The loading and the model are the same as in the recipe above; what is new here
is asking the predictive posterior for thresholds rather than probabilities:

```python title="Threshold inversion at every published reference point"
--8<-- "docs/examples/wppm/hong2025_reproduction.py:thresholds"
```



---

Both questions so far handed psyphy the paper's own weights, so neither has
asked it to *fit* anything. That is the next step, which is the expensive part.

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
    <p><em><span class="arithmatex">\(\Sigma_{\text{noise}}(x)\)</span> for subject 1 (CH): dashed gray is the field
    from the authors' published weights, red is our own MAP refit. This is the
    paper's supplementary Figure S3. The ellipses are a single color here rather
    than colored by reference stimulus — that convention is reserved for the
    threshold figures, so the two cannot be confused at a glance.
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

This is the figure at the top of the page, and this is where it comes from.

The two steps before it each held something fixed: the Figure 2B inversion used
the authors' *published* weights, so it tested our inversion with the optimizer
removed; the refit fit weights from the raw trials but only ever compared noise
fields. Joining them closes the loop:
*raw trials -> our weights -> our contours
-> the published figure*, with nothing published entering except the data:

Everything is identical to the inversion above — same model, same
`ThresholdConfig`, same call — except for where the weights come from:

```python title="Invert our own fitted weights"
--8<-- "docs/examples/wppm/hong2025_reproduction.py:end_to_end"
```

??? note "Drawing it: both fields on one axes"

    Both contour fields go onto one axes in a single
    [`plot_ellipses`](../../reference/viz.md) call — published dashed
    underneath, ours on top, each ellipse colored by its own reference
    stimulus.

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
    weights <em>we</em> fit to the raw trials. There are no published weights anywhere in
    this chain. Dashed gray is the authors' published inversion; colored solid is
    ours, each ellipse taking the color of its reference stimulus.</em></p>
</div>



---

## Is that close enough? The paper's own bootstrap interval

How close is close enough? The authors answered that themselves. They resampled
the trials 120 times, refit the model to each, and kept the 114 fits (95% of
120) that came out most like their original. The spread of those 114 contours
is their 95% confidence interval and we check
whether the threshold generated from psyphy's
fit is comprised by that confidence interval
in the figure below.





The figure below shows that our fit is indistinguishable from their run-to-run variaton at all 49 reference points and every direction tested, and in that
sense *psyphy's refit is indistinguishable from their fit*.

<div align="center">
    <img src="../plots/hong2025_bootstrap_envelope.png"
         alt="Our threshold contours against the paper's 95% bootstrap confidence interval"
         width="620"/>
    <p><em>Our end-to-end contours against the paper's own 95% bootstrap interval
    for subject 1 (CH). The gray band is the 114 retained bootstrap refits,
    dashed gray the published fit, colored solid ours.</em></p>
</div>

??? note "Plotting it: the band, then the two fits on top"

    `plot_ellipses` takes a whole stack of fields at once, so all 114 retained
    refits go on in a single call. The published fit and ours are drawn over
    them in the usual convention.

    ```python
    --8<-- "docs/examples/wppm/hong2025_reproduction.py:envelope_plot"
    ```

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
    | **Refit — full** | 1 GPU | **~8 min** | 6,000 trials, 1,500 steps, `mc=2000`, 3 restarts |
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
- **The same seed gives the same answer on the *same* machine, but not
  necessarily on a different one.** Re-running the inversion here is
  bit-identical: JAX's PRNG is deterministic given a key, so nothing changes
  between runs on the same machine. But what changes across machines is the floating-point arithmetic
  underneath: XLA reassociates or rewrite an expression, and a sum
  accumulated in a different order lands on a slightly different value
  ([JAX FAQ](https://docs.jax.dev/en/latest/faq.html#jit-changes-the-exact-numerics-of-outputs)).
  The exact check is unaffected, since it compares against a table rounded to 8
  decimals. The thresholds and the refit can differ in
  their low-order digits between a laptop and a GPU
- **Loss values are not comparable to the paper's.** psyphy's `Prior.log_prob`
  drops a constant, which the paper keeps (still  identical gradients but different numbers)


---

## See also

- [Full WPPM fit (simulated data)](full_wppm_fit_example.md) — same machinery with ground truth available.
- [Quick start](quick_start.md) — the minimal version.
- [Plotting ellipse fields](../viz/ellipse_plots.md) — `plot_ellipses` on its own, with synthetic data.
- `psyphy.data.published.hong2025` in [Data](../../reference/data.md); `WPPMPredictivePosterior` and `ThresholdConfig` in [Posterior](../../reference/posterior.md).
