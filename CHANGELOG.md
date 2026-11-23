# Changelog

All notable changes to `psyphy` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). `psyphy` is
pre-1.0: while the version is `0.0.x`, **any release may change the public API**, so
breaking changes are called out below but do not force a minor-version bump.

## [Unreleased]

### Breaking

- **One covariance field class, `WPPMCovarianceField`, holds N parameter sets, and the sample axis is always first in every result.** The constructor takes a posterior, and
  `WPPMCovarianceField.from_params(model, params)` builds one from a ground truth or a
  published $W$. `from_posterior`, `sqrt_cov_batch` and the `CovarianceField` protocol are
  gone, and `psyphy.model` no longer exports `CovarianceField`.

  The sample axis is the leading axis of a result, with 1 entry per parameter set. An MCMC
  fit has 1 per sample. A MAP fit, a ground truth or a published $W$ is 1 parameter set,
  so its results now start with an axis of length 1. In the shapes below:

  - `x`: one stimulus location, shape `(d,)`
  - `X`: the stimulus locations at which we evaluate the field, shape `(n_points, d)`, or
    any batch `(..., d)` such as a grid `(n_rows, n_cols, d)`
  - `d`: stimulus dimensions, the model's `input_dim`
  - `m`: columns of $U(x)$, `input_dim + extra_embedding_dims`

  `field(x)` has shape `(1, d, d)`, `field(X)` has `(1, n_points, d, d)`, and
  `field.sqrt_cov(x)` has `(1, d, m)`. `field(x)[0]` is the covariance matrix at `x`.

  Code written for the old shapes can run on the new ones and give wrong numbers with no
  error, so check each result that used to be `(d, d)` or `(n_points, d, d)`. 2 cases:

  ```python
  jnp.trace(field(x))            # (d,): the first row of Σ(x), not its trace
  jnp.trace(field(x)[0])         # the trace

  # a 1-D model, X of shape (n_points, 1) and truth of shape (n_points,)
  field(X)[:, 0, 0] - truth      # (1, n_points): Σ at the first point minus every truth value
  field(X)[0, :, 0, 0] - truth   # (n_points,)
  ```

  `jax.vmap(field)(X)` evaluates the points one at a time and puts them first,
  `(n_points, 1, d, d)`. Pass `X` to `field` directly for the samples first, or write
  `jax.vmap(field, out_axes=1)(X)` inside code that has to `vmap` over the points.

  ```python
  # before
  field = WPPMCovarianceField(model, map_posterior.params)        # or .from_posterior(map_posterior)
  truth = WPPMCovarianceField(model, truth_params)
  prior = WPPMCovarianceField.from_prior(model, key)
  field(x)              # (d, d)
  field.sqrt_cov_batch(X)

  # now
  field = WPPMCovarianceField(map_posterior)                     # no key
  truth = WPPMCovarianceField.from_params(model, truth_params)
  prior = WPPMCovarianceField.from_prior(model, n_samples=1, key=key)
  field(x)[0]           # (d, d); field(x) is (1, d, d)
  field.sqrt_cov(X)[0]  # (n_points, d, m)
  ```

  `from_prior(model, n_samples=1, key=key)` splits the key, so it draws a different sample
  than the old `from_prior(model, key)`. For the old draw, use
  `from_params(model, model.init_params(key))`. `n_samples` and `key` are keyword-only in
  the constructor and in `from_prior`.

- **`cov()` and `cov_batch()` are gone from `WPPMCovarianceField`.** Both raised
  `DeprecationWarning` in v0.0.5, and `field(x)` replaces both.

  ```python
  # before
  field.cov(jnp.array([0.5, 0.3]))        # (2, 2), single point only
  field.cov_batch(jnp.zeros((10, 2)))     # (10, 2, 2), batch only

  # now
  field(jnp.array([0.5, 0.3]))[0]         # (2, 2)
  field(jnp.zeros((10, 2)))[0]            # (10, 2, 2)
  ```

  The numbers are unchanged: `cov()` forwarded to the same evaluation `field(x)` uses.

- **A Bernoulli task's `predict` returns `(log_p, log_1mp)`, not `(p_correct,)`.** These
  are log p(correct) and log p(incorrect) for one trial. `loglik` takes one of the 2 by the
  response and holds it in [1e-6, 1 - 1e-6], the bound `OddityTask.predict` applied to
  p(correct) before. So the log-likelihood is the same, up to float 32 rounding. `predict`
  itself no longer clips, and `WPPM.predict_prob` and `simulate` return `exp(log_p)`, which
  can be exactly 1.0.

  A task with a logit `z` returns `log_probs_from_logit(z)`, new in
  `psyphy.model.likelihood`:

  ```python
  # before
  def predict(self, params, stimuli, model, key=None):
      return (jnp.clip(jax.nn.sigmoid(z), 1e-6, 1 - 1e-6),)

  # now
  def predict(self, params, stimuli, model, key=None):
      return log_probs_from_logit(z)
  ```

  `predict` returns log-probabilities, not probabilities. For p(correct), take
  `jnp.exp(predict(...)[0])` or call `model.predict_prob`.

- **`model`, `samples` and `n_samples` on `WPPMCovarianceField` are read-only.**

  Before, `params` and `model` were plain attributes. After a batch call, `field(X)` kept
  returning the values from its first call, while `field(x)` used the reassigned ones.
  One object gave 2 different answers, and a loop that refits and assigns new parameters
  got stale results from every batch call.

  Now, assigning to one raises an `AttributeError`. `samples` returns a copy, so
  `field.samples["W"] = ...` changes nothing either. `from_params(model, params)` copies
  `params` too, so `params["W"] = ...` on the dict you passed in changes nothing.

  ```python
  # before
  field.params = new_params  # accepted, then field(X) returned the old values

  # now
  field.samples = new_samples  # raises AttributeError
  field = WPPMCovarianceField.from_params(model, new_params)  # build a new object instead
  ```

- **`Prior(basis_degree=None)` raises `ValueError`.** `basis_degree` must be an int of 0 or
  more. v0.0.5 accepted `None`.

  ```python
  # before
  Prior(basis_degree=None)  # built, then raised in sample_params

  # now
  Prior(basis_degree=4)  # None raises ValueError
  ```

- **`WPPM` no longer takes `input_dim`, `extra_dims`, `variance_scale` or `decay_rate`.**
  `Prior` takes them instead. `WPPM` stored `variance_scale` and `decay_rate` and never
  read them, and `extra_dims` fed only `embedding_dim`, so it disagreed with the parameter
  shape without an error whenever it differed from `Prior.extra_embedding_dims`.
  `WPPM.input_dim`, `.embedding_dim`, `.variance_scale` and `.decay_rate` are now
  read-only properties that read `self.prior`:

  ```python
  # before
  WPPM(prior, likelihood, input_dim=2, extra_dims=1, variance_scale=4e-3, decay_rate=0.4)

  # now
  Prior(input_dim=2, extra_embedding_dims=1, variance_scale=4e-3, decay_rate=0.4)
  WPPM(prior, likelihood)
  ```

  Passing any of the 4 names to `WPPM` raises `TypeError` and says to pass them to `Prior`.
  `diag_term` is unchanged.

- **`ThresholdConfig.target_pc` is now `target_prob_correct`.** The default stays 2/3.

- **`WPPMPredictivePosterior` takes a required `key`, and `cov_field` is gone.** The
  constructor splits the key once, into a key for drawing parameters and a key for the
  threshold sweep:

  ```python
  # before
  WPPMPredictivePosterior(posterior, X, n_samples=100)

  # now
  WPPMPredictivePosterior(posterior, X, n_samples=100, key=key)
  ```

  Before, both keys were fixed inside the class, so no key you chose could change the
  result. `cov_field(X)` also computed E[Σ(X) | data] under a fixed key. Use
  `WPPMCovarianceField(posterior).mean(x)`. `PredictivePosterior` no longer has a
  `cov_field` member, so a model with no covariance field can satisfy it.

- **`MAPPosterior.sample` requires `n` and `key`, and `key` can be passed positionally.**
  Before, `n` defaulted to 1 and `key` was keyword-only and optional. The signature now
  matches the `ParameterSource` protocol. `key` is also accepted positionally by
  `WPPM.log_likelihood_from_data`, `WPPM.log_posterior_from_data`, and `loglik`,
  `simulate` and `predict` on the task likelihoods. Calls that name it still work.

  ```python
  # before
  map_posterior.sample()            # one draw

  # now
  map_posterior.sample(1, key=key)  # the key goes unread for a point mass
  map_posterior.sample(1, key)      # also accepted
  ```

- **The model comes first in `MAPPosterior`.** It now matches `fit(model, data)` and
  `WPPMCovarianceField.from_params(model, params)`. `MCMCPosterior(model, samples)` takes
  the same order. Calls that name the arguments still work.

  ```python
  MAPPosterior(model, params)      # was MAPPosterior(params, model)
  MAPPosterior(params=params, model=model)  # unchanged
  ```

  The old order raises a `TypeError` when the object is built. Before, it built and failed
  at the first evaluation with `AttributeError: 'dict' object has no attribute 'input_dim'`.

### Added

- **`WPPMCovarianceField(posterior)` evaluates the covariance field for every sample.**
  `field(x)` returns Σ for each sample, `(N, d, d)` for one point or `(N, ..., d, d)` for a
  batch. `field.mean(x)` returns the mean matrix E[Σ(x)] over the samples, with no sample
  axis. `field.sqrt_cov(x)` returns the square root `U` for every sample, `(N, d, m)`, and
  `U U^T + diag_term * I` equals `field(x)[i]` for each `i`:

  ```python
  fields = WPPMCovarianceField(posterior)
  covs = fields.mean(ref_points)  # (n_points, d, d)
  U = fields.sqrt_cov(x)          # (N, d, m)
  ```

  To compute something per sample that psyphy has no method for, read the samples from
  `fields.samples`. `from_params` takes the model first, and `from_params(params, model)`
  raises `TypeError`.

- **`MAPPosterior` stores its 1 sample**, as `samples`, `n_chains` and `n_draws`, with
  chains first and then samples. So `WPPMCovarianceField(map_posterior)` takes a MAP fit
  with no `n_samples` and no key:

  ```python
  map_posterior.samples["W"].shape  # (1, 1, 3, 3, 2, 3): 1 chain, 1 sample, then W
  map_posterior.n_chains, map_posterior.n_draws  # (1, 1)

  WPPMCovarianceField(map_posterior)  # n_samples == 1
  ```

  A posterior with no `samples`, `n_chains` and `n_draws` still raises and asks for
  `n_samples` and a key.

- **`NUTSSampler` in `psyphy.inference`, `MCMCPosterior` in `psyphy.posterior`, and
  `covariance_to_arviz` in `psyphy.model`.** `NUTSSampler` runs the No-U-Turn Sampler
  (NUTS) from BlackJAX, a Markov chain Monte Carlo (MCMC) method. Its `fit` returns an
  `MCMCPosterior`, which holds the samples of every chain:

  ```python
  sampler = NUTSSampler(num_warmup=500, num_samples=1000, num_chains=4)
  posterior = sampler.fit(model, data, init_params=map_posterior.params)
  posterior.samples["W"].shape  # (4, 1000, 3, 3, 2, 3): 4 chains, 1000 samples each, then W
  fields = WPPMCovarianceField(posterior)
  ```

  `mean_params` is the mean of the samples, for example to start another fit. `MCMCPosterior`
  has no `params`. `sampler_stats` holds `acceptance_rate`, `diverging`, `energy` and
  `n_steps` per sample.

  `covariance_to_arviz(posterior, points)` returns Σ(x) at the points you choose, for every
  sample, as an ArviZ `DataTree`. Use it for convergence diagnostics such as R-hat and the
  effective sample size (ESS), not `posterior.to_arviz()`, which reads $W$.

  3 new extras:

  ```bash
  pip install 'psyphy[sampling]'     # BlackJAX, Python 3.11 or later
  pip install 'psyphy[diagnostics]'  # ArviZ 1.x, Python 3.12 or later
  pip install 'psyphy[progress]'     # tqdm; without it show_progress=True prints nothing
  ```

- **`MALASampler` in `psyphy.inference`**, an MCMC engine. It runs the
  Metropolis-adjusted Langevin algorithm from BlackJAX, with one gradient per step and a
  fixed step size. It returns an `MCMCPosterior` with the same layout as `NUTSSampler`, so
  `covariance_to_arviz` and `WPPMCovarianceField` take either.

  ```python
  sampler = MALASampler(num_warmup=2000, num_samples=2000, step_size=1e-4)
  posterior = sampler.fit(model, data, init_params=map_posterior.params)
  posterior.sampler_stats["acceptance_rate"].mean()  # aim for about 0.6
  ```

  `step_size` has no default. `sampler_stats` holds `acceptance_rate` and `is_accepted`,
  with no `diverging`, `energy` or `n_steps`.

- **`ParameterSource` in `psyphy.posterior`**, a one-method protocol for anything we can
  draw parameters from: `sample(n, *, key)` returning a PyTree with a leading axis of `n`.
  `Prior`, `MAPPosterior` and `MCMCPosterior` all satisfy it, and `ParameterPosterior` now
  extends it with `model`.

  A function that only calls `sample` on a `ParameterSource` works on a prior and on a
  fitted posterior. Chain structure and a density stay off it: `sample` pools the chains,
  and `MCMCPosterior` has no `log_prob`.

- **`Prior.sample(n, *, key)`**, the vectorized form of `sample_params`. Draw `i` equals
  `sample_params(jax.random.split(key, n)[i])`, so a prior ensemble matches the Python loop
  it replaces:

  ```python
  # before
  W = jnp.stack([prior.sample_params(k)["W"] for k in jr.split(key, 100)])

  # now
  prior.sample(100, key=key)["W"]     # (100, 3, 3, 2, 3)
  ```

  `sample_params(key)` is unchanged and returns 1 unbatched draw.

- **Credible bands on the contour radii**: `contour_radii`,
  `WPPMCovarianceField.credible_band` and `CredibleBand` in `psyphy.model`.
  `contour_radii` gives the radius of a covariance's one-standard-deviation ellipse
  (Mahalanobis-1 contour) per direction. For the oddity task that radius is the
  discrimination threshold. `credible_band` reduces those radii over the samples to a
  pointwise credible interval. With 1 sample, as after a MAP fit, it raises `ValueError`,
  because a band of zero width reads as certainty.

  ```python
  WPPMCovarianceField(map_posterior).credible_band(points)   # raises ValueError
  WPPMCovarianceField(mcmc_posterior).credible_band(points)  # the band
  ```

- **A page, *Fit your own data*** (`docs/examples/wppm/fit_your_own_data.md`), replaces
  `usage.md` in the nav. It reads trials from a CSV into a `TrialData`, fits a `WPPM` and
  checks the fit with no ground truth. Its script takes the number of stimulus dimensions
  from the data, so it fits 2-D and 3-D stimuli. The `ThresholdConfig.n_theta` docstring now
  says to raise it as the dimension grows: 16 is enough in 2-D, use 64 in 3-D.

## [0.0.5] - 2026-10-09

 Two new subpackages
(`psyphy.viz`, `psyphy.data.published`), a reproduction of a published figure, and two
breaking API changes in the model layer.

### Breaking

- **`TaskLikelihood.predict` signature and return type changed** (#139). Stimuli are now
  passed as one packed, slot-indexed array instead of separate arguments, and the return
  is a tuple of sufficient statistics instead of a bare scalar:

  ```python
  # before (0.0.4)
  def predict(self, params, ref, comparison, model, *, key=None) -> jnp.ndarray: ...
  #   returns p_correct

  # now (0.0.5)
  def predict(self, params, stimuli, model, *, key=None) -> tuple[jnp.ndarray, ...]: ...
  #   stimuli has shape (K, input_dim);  OddityTask reads stimuli[0], stimuli[1]
  #   returns (p_correct,) for Bernoulli tasks, (mu, sigma) for Gaussian task likelihoods
  ```

  **If you wrote a custom `TaskLikelihood`, it must be updated.** Unpack `stimuli`
  yourself and wrap the return in a tuple. Returning a bare array no longer works
  correctly: `BernoulliTaskLikelihood.loglik` indexes element 0 of the returned tuple, so
  a bare scalar silently collapses every trial onto the first trial's probability.

- **`TrialData` now stores stimuli as one array** (#136). `refs=` / `comparisons=` are
  replaced by a single `stimuli` array of shape `(N, K, d)`, with `responses` of shape
  `(N, R)`; `K` is the number of stimuli per trial, so tasks with more than two slots are
  now representable.

  ```python
  # before
  TrialData(refs=refs, comparisons=comparisons, responses=responses)
  # now
  TrialData(stimuli=jnp.stack([refs, comparisons], axis=1), responses=responses)
  ```

  A 1-D `responses` array of shape `(N,)` is still accepted and normalized to `(N, 1)`.
  Slots may optionally be named, e.g.,  `stimulus_names=("ref", "comp")` enables
  `data.stimulus("ref")` alongside positional `data.stimuli[:, 0, :]`.

  **For the oddity task, `K` is 2, not 3.** The observer is shown three stimuli, but only
  the two distinct means are stored; presenting the reference twice is encoded in
  `OddityTask`, which draws two samples from the reference distribution and one from the
  comparison. `K` counts stored stimuli, not presentations.

### Added

- **`psyphy.data.published`** loaders for published datasets, starting with
  `hong2025`: `fetch`, `fetch_calibration_matrix`, `load_calibration_matrix`,
  `load_trials`, `load_reference_W`, `load_sigma_table`, `build_paper_model`,
  `w2d_to_rgb`, `default_data_dir`. Datasets are downloaded on demand (resumable, cached
  outside the repo) rather than shipped with the package.
- **`psyphy.viz`** : `plot_ellipses`, plus `auto_scale` and `ellipse_segments`. The
  geometry is separated from the drawing layer so it is testable without a plotting
  backend, and `matplotlib` is imported lazily.
- **Threshold prediction as first-class API** ,  `WPPMPredictivePosterior(...,
  threshold_pred=True)` with a `ThresholdConfig`, recovering threshold contours by
  numerically inverting `P(correct)`. In this mode `X` is bare reference points of shape
  `(n_test, input_dim)` rather than paired stimuli.
- **1-D WPPM support** `input_dim=1` is now handled by the Chebyshev basis, the
  prior, and the covariance-field computation, alongside the existing 2-D and 3-D cases.
- **A distributional layer in the likelihood hierarchy**
  `BernoulliTaskLikelihood` and `GaussianTaskLikelihood` sit between `TaskLikelihood` and
  concrete tasks, each providing `loglik` and `simulate` so a new task only implements
  `predict`. `OddityTask` is now a `BernoulliTaskLikelihood`.
- **`reduction` on `MAPOptimizer`**  `"mean"` (new default) or `"sum"`, controlling how
  the per-trial objective is aggregated. This interacts with gradient clipping: under
  `"sum"` the gradient scales with the number of trials, so a learning rate tuned on one
  dataset size does not transfer. `"mean"` makes learning rates comparable to those
  reported by Hong et al 2025.
- **Tutorials** reproduction of Hong et al. (2025) Figure 2B (threshold contours) from the published data,
  recovery of Weber's law with a flexible WPPM, and ellipse-field plotting.
- **Tests**: `test_data_published_hong2025.py`, `test_viz.py`, `test_docs_recipe.py`
  (parses the code quoted in the docs and checks every call against the live API),
  `test_quick_start_recovery.py`, `test_map_optimizer_clipping.py`,
  `test_likelihood_logic.py`, `test_data_format.py`.

### Fixed

- **Response-shape broadcasting in the Bernoulli log-likelihood** (#140). With
  `responses` of shape `(N, 1)` and probabilities of shape `(N,)`, `jnp.where` broadcast
  to `(N, N)`, so the summed objective mixed every trial's response with every trial's
  probability and the gradient was wrong. Responses are now reduced to `(N,)` first.
- **Deprecated `jax.tree_map` replaced** with `jax.tree.map` (#71, thanks
  @rohansood10).

### Changed

- `requires-python` is `>=3.10`; CI runs lint, type checks, and tests on 3.10, 3.11, and
  3.12.
- Documentation pages now quote code from the runnable scripts beside them via snippet
  includes, so a page and its script cannot drift apart.

## [0.0.4] - 2026-03-31

Likelihood refactor. `loglik` and `simulate` became concrete methods on
`TaskLikelihood`, so a concrete task implements only `predict`; `OddityTask` lost its own
`loglik` and the duplicated vectorised Monte Carlo path (`likelihood.py` net −200 lines).
Documentation restructuring.

## [0.0.2] - 2026-03-27

First release published to PyPI. Full initial module set: `psyphy.model` (WPPM, priors,
noise models, the oddity task), `psyphy.inference` (MAP optimizer), `psyphy.posterior`,
 `psyphy.data`,  `psyphy.utils`.

[0.0.5]: https://github.com/flatironinstitute/psyphy/compare/v0.0.4...v0.0.5
[0.0.4]: https://github.com/flatironinstitute/psyphy/compare/v0.0.2...v0.0.4
[0.0.2]: https://github.com/flatironinstitute/psyphy/releases/tag/v0.0.2
