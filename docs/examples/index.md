# Examples

This section collects end-to-end, runnable examples.

## WPPM

- [Quick start](wppm/quick_start.md): one reference point, fitted in seconds on a CPU.
- [Fit your own data](wppm/fit_your_own_data.md): trials from a CSV file, the calls in order, and checks for a fit with no ground truth. Any number of stimulus dimensions.
- [Recovering Weber's law](wppm/weber_law.md): a 1-D fit that relearns Weber's law from simulated trials.
- [Full WPPM fit (end-to-end)](wppm/full_wppm_fit_example.md)
- [Reproducing a published fit (Hong et al. 2025)](wppm/hong2025_reproduction.md): fits real human data and compares against the authors' own published fit.
- [Sampling the posterior with NUTS](wppm/nuts_inference_example.md): full posterior
  inference, and credible bands on thresholds.
- [Checking MCMC samples with ArviZ](wppm/mcmc_diagnostics.md): the four diagnostics, and
  which sampler argument to change when one fails.
- [Plotting ellipse fields](viz/ellipse_plots.md): drawing covariance and threshold fields with `psyphy.viz`.


## Covariance field

- [Covariance Field Visualization](covariance_field/covariance_field.md)
