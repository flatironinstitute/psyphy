# Fit your own data

This page is for readers who know the Wishart Process Psychophysical Model (WPPM) and want
to fit it to their own trials with psyphy. It shows how psyphy stores trials and the calls in
order, from a CSV file to thresholds and the checks we run on a fit. The model itself is
explained in the [WPPM tutorial](full_wppm_fit_example.md).

We use one observer's published trials from
[Hong et al.](https://doi.org/10.7554/eLife.108943.2): 12,000 oddity-task trials in a 2-D
color plane. The script reads the number of stimulus dimensions from the data directly, so the same
code would work on 3-D stimuli, for example.

The companion script to this tutorial is
[`fit_your_own_data.py`](https://github.com/flatironinstitute/psyphy/blob/main/docs/examples/wppm/fit_your_own_data.py).

The figures here come from `--mode full`, which require a GPU:

??? note "How to run the script and compute settings"

    ```bash
    pip install 'psyphy[viz]'
    python fit_your_own_data.py --mode quick   # smoke test, ca. 1 min on a laptop CPU
    python fit_your_own_data.py --mode full    # Hong et al.'s Monte Carlo settings, GPU
    ```

    The script downloads the trials from OSF the first time it runs (about 1 MB). We found no
    setting that converges within 10 min on a laptop CPU, so for your own data plan on a GPU.

## Data representation in psyphy

psyphy stores trials in a [`TrialData`](../../reference/data.md#psyphy.data.TrialData) container:

- `stimuli`, shape `(N, 2, d)`: $N$ trials, 2 _distinct_ stimuli per trial, $d$ stimulus dimensions.
  Slot 0 is the reference, slot 1 the comparison. Note that while, in total, the Hong et al paper shows 3 stimui, the oddity task shows the reference twice (a copy) and psyphy stores it once, hence psyphy only stores the number of _distinct_ stimuli per trial
- `responses`, shape `(N, 1)`: 1 for a correct response, 0 for an incorrect one

Every coordinate must lie in $[-1, 1]$, the domain of the Chebyshev basis. Rescale first, e.g.
`2 * (x - lo) / (hi - lo) - 1`.

!!! warning "No check for the wrong domain"
    Currently, psyphy's fit doesn't check this. It would return a wrong field with no error.

[`hong2025.fetch`](../../reference/data.md#psyphy.data.published.hong2025.fetch) downloads
Hong et al.'s CSV file from OSF. It has one row per trial and the columns `TrialType`, `xref`,
`x1` and `y`. We read it with numpy, then put the arrays into a `TrialData`:

```python title="Read the CSV"
--8<-- "docs/examples/wppm/fit_your_own_data.py:load_csv"
```

The `MOCS` rows are held-out trials and are trials at fixed comparison levels that Hong et al. didn't fit. We hold
them out for [check (c)](#checks-with-no-ground-truth). With your own data, hold out a random
subset instead, e.g. a fifth of the trials.

```python title="Build TrialData"
--8<-- "docs/examples/wppm/fit_your_own_data.py:trial_data"
```

If your file has the columns `ref`, `probe` and `response`, with each stimulus written as a
list (`"[0.1, -0.2]"`),
[`load_responses_csv`](../../reference/data.md#psyphy.data.io.load_responses_csv) returns
the `TrialData` in one call out of the box.

## Model

```python title="Build the model"
--8<-- "docs/examples/wppm/fit_your_own_data.py:model"
```

For the prior, we use the same that Hong et al used. For 3-D stimuli, nothing in this code would change: `d` is
3, and the weights `W` get shape `(5, 5, 5, 3, 4)` instead of `(5, 5, 2, 3)`. That is
Chebyshev degree 0 to 4 in each of the $d$ stimulus dimensions, then $d$, then $d + 1$
embedding dimensions.

## Fit

[`MAPOptimizer`](../../reference/inference.md#psyphy.inference.MAPOptimizer) starts from a
sample of the prior, picked by `seed`. We fit twice, from 2 prior samples (blue and orange in the figures below), for
[check (b)](#checks-with-no-ground-truth). The loss can have more than one local minimum, and
a fit stops in the one it reaches first. So we start from at least 2 points (seed 0 and 1) and keep the
better fit.

```python title="Fit from 2 prior samples"
--8<-- "docs/examples/wppm/fit_your_own_data.py:fit"
```

| mode | steps | Monte Carlo samples per trial: fit, predictions | time for both fits |
|---|---|---|---|
| `quick` | 300 | 20, 200 | 10 s, laptop CPU |
| `full` | 1,500 | 2,000, 2,000 | ca. 3 min, 1 GPU |

## Results

We look at the fits on a 7 x 7 grid of reference points:

```python title="Reference points"
--8<-- "docs/examples/wppm/fit_your_own_data.py:refs"
```

For thresholds we predict with more Monte Carlo samples than we fit with. This is because the
 predicted $P(correct)$ is then less noisy.

```python title="A model for predictions"
--8<-- "docs/examples/wppm/fit_your_own_data.py:eval_model"
```

[`WPPMPredictivePosterior`](../../reference/posterior.md#psyphy.posterior.WPPMPredictivePosterior)
with `threshold_pred=True` searches `n_theta` directions around each reference for the
comparison that is correct 66.7% of the time, then fits an ellipse through those points (an
ellipsoid in 3-D).

```python title="Thresholds"
--8<-- "docs/examples/wppm/fit_your_own_data.py:thresholds"
```

!!! note "More dimensions need more directions"
    In 2-D, 16 directions are enough, evenly spaced around the reference. From 3-D on, psyphy
    picks the directions at random, and with 16 the ellipsoid misses the 66.7% contour in some
    directions: P(correct) on it ranged from 0.48 to 0.97 in our check. With 64 it stayed
    between 0.5 and 0.7. So raise `n_theta` as $d$ grows (64 in 3-D costs ca. 4 times as
    much as 16). It needs at least $d(d+1)/2$.

We draw them with psyphy's
[`plot_ellipses`](../../reference/viz.md#psyphy.viz.ellipses.plot_ellipses) (2-D only; more
options in the [ellipse plotting tutorial](../viz/ellipse_plots.md)):

```python title="Plot"
--8<-- "docs/examples/wppm/fit_your_own_data.py:plot"
```

<div align="center">
  <img src="../plots/own_data_thresholds.png" width="440"/>
  <p><em>Threshold contours at 66.7% correct (<code>thresholds</code> above) (not the noise
  field!), on a 7 x 7 grid of reference points. Solid blue: fit from seed 0.
  Dashed orange: fit from seed 1. The 2 fits agree pretty much everywhere but in the lower left quadrant. The contours of seed 1 (orange) are larger and longer and differently angled. So here,
  the 2 fits found different fields. <a href="#checks-with-no-ground-truth">The loss curves </a> give insights as to why.</em></p>
</div>

## Checks with no ground truth

With your own data there's no true field to compare with. Instead, we can check these 3 things:

**(a) The learning curve has flattened.** The loss drops, then flattens into a noisy plateau.
A curve that still trends down needs more steps, and spikes mean the fit hasn't settled.

<div align="center">
  <img src="../plots/own_data_learning_curve.png" width="460"/>
  <p><em>The loss <code>MAPOptimizer</code> minimizes, at each step: the negative log
  posterior divided by the number of trials N, i.e. the negative log likelihood plus the
  negative log prior, per trial. One line per fit. Both drop within the first 50 steps
  and flatten near 0.5. Each spike is a step that jumped away from the plateau, and the
  loss comes back down after it. The last spike of seed 1 (orange), at ca. step 1,500, hasn't come
  back down by the final step, explaining potentially the misalignment in the lower left quadrant in the figure above.</em></p>
</div>

`MAPOptimizer` returns the parameters of the final step. A curve that ends on a spike
therefore returns a fit from inside the spike, as seed 1 does here. Run more steps, or fit
again from another seed. A flat curve alone doesn't show convergence either, since a fit can
sit on a plateau above the optimum. So we also run (b).

**(b) 2 fits from different prior samples give the same field.** We compare the thresholds
of the 2 fits point by point, as the relative Frobenius distance
$\lVert \Sigma_1 - \Sigma_0 \rVert_F / \lVert \Sigma_0 \rVert_F$:

```python title="Compare the 2 fits"
--8<-- "docs/examples/wppm/fit_your_own_data.py:two_fits"
```

For a converged fit this is close to 0 at every point. A value near 1 means the 2 fits
found different fields, as in the lower left of the threshold figure above.

**(c) The fit predicts trials it wasn't fitted on.** We sort the held-out trials into 10 bins
by predicted P(correct), and compare each bin's mean prediction with the proportion of
correct responses in it:

```python title="Predicted against observed, on held-out trials"
--8<-- "docs/examples/wppm/fit_your_own_data.py:held_out"
```

<div align="center">
  <img src="../plots/own_data_held_out.png" width="360"/>
  <p><em>Fit from seed 0, on the 6,000 held-out trials. One dot per bin of 600 trials,
  with 2 standard errors. Gray: perfect prediction. The dots follow the diagonal from a
  predicted P(correct) of 0.3 to 1, and the error bars of most bins cross it, so the fit
  predicts trials it wasn't fitted on pretty well! Only 2 bins, near 0.6 and 0.6, sit just above the gray perfect prediction baseline, meaning the model predicted too few correct responses.</em></p>
</div>

The dots should lie on the diagonal, within their error bars. Dots above it mean the fit
predicts too few correct responses and dots below too many.

!!! note "If you simulated your data"
    With a known field, compare the fit with it directly. Here we use Hong et al.'s
    published fit in place of a ground truth, loaded with
    [`load_reference_W`](../../reference/data.md#psyphy.data.published.hong2025.load_reference_W).
    [`WPPMCovarianceField`](../../reference/model.md#psyphy.model.WPPMCovarianceField) turns
    each set of weights into the covariance field $\Sigma(x)$, and we compare the 2 at the
    reference points:

    ```python title="Compare with a known field"
    --8<-- "docs/examples/wppm/fit_your_own_data.py:truth"
    ```

    For your simulation, pass your true `W` in place of `W_published`, in the shape
    `prior` gives it. `load_reference_W` reads 2-D weights only. If your simulation
    returns $\Sigma$ at the reference points instead, compare `covs` with those.

## Next steps

- [Sampling the posterior with NUTS](nuts_inference_example.md), for uncertainty on the
  thresholds instead of a single fit
- [Reproducing Hong et al. (2025)](hong2025_reproduction.md), the full refit of the paper
- questions and bugs: [GitHub issues](https://github.com/flatironinstitute/psyphy/issues)
