# Changelog

All notable changes to `psyphy` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). `psyphy` is
pre-1.0: while the version is `0.0.x`, **any release may change the public API**, so
breaking changes are called out below but do not force a minor-version bump.

## [Unreleased]

### Breaking

- **One covariance field class, `WPPMCovarianceField`, where the sample axis is always first in every result.** `WPPMCovarianceFieldSamples` is renamed `WPPMCovarianceField`. The single-set
  `WPPMCovarianceField`, the `CovarianceField` protocol, `sqrt_cov_batch` and
  `from_posterior` are gone, and `psyphy.model` no longer exports `CovarianceField` or
  `WPPMCovarianceFieldSamples`.

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
  `(n_points, 1, d, d)`. Pass `X` to `field` directly for the samples first.

  ```python
  # before
  field = WPPMCovarianceField(model, map_posterior.params)        # or .from_posterior(map_posterior)
  truth = WPPMCovarianceField(model, truth_params)
  prior = WPPMCovarianceField.from_prior(model, key)
  fields = WPPMCovarianceFieldSamples(nuts_posterior)
  field(x)              # (d, d)
  field.sqrt_cov_batch(X)

  # now
  field = WPPMCovarianceField(map_posterior)                     # no key
  truth = WPPMCovarianceField.from_params(model, truth_params)
  prior = WPPMCovarianceField.from_prior(model, n_samples=1, key=key)
  fields = WPPMCovarianceField(nuts_posterior)
  field(x)[0]           # (d, d); field(x) is (1, d, d)
  field.sqrt_cov(X)[0]  # (n_points, d, m)
  ```

  `from_prior(model, n_samples=1, key=key)` splits the key, so it draws a different sample
  than the old `from_prior(model, key)`. For the old draw, use
  `from_params(model, model.init_params(key))`.

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

- **The attributes of the covariance field classes are read-only.** These are `params` and
  `model` on `WPPMCovarianceField`, and `samples`, `n_samples` and `model` on
  `WPPMCovarianceFieldSamples`.

  Before, assigning to one was accepted without an error. After a batch call, `field(X)`,
  `fields(X)` and `fields.mean(X)` kept returning the old values, while `field(x)` and
  `fields.credible_band` used the new ones. One object gave 2 different answers, and
  `fields.n_samples` stayed at the old count. A loop that refits and assigns new samples
  got stale results from every batch call.

  Now, assigning to one raises an `AttributeError`. `params` and `samples` return a copy,
  so `fields.samples["W"] = ...` changes nothing either.
  `WPPMCovarianceField(model, params)` copies `params` too, so `params["W"] = ...` on the dict
  you passed in changes nothing.

  ```python
  # before
  fields.samples = new_samples  # accepted, then batch calls returned the old values

  # now
  fields.samples = new_samples  # raises AttributeError
  fields = WPPMCovarianceFieldSamples(new_posterior)  # build a new object instead
  ```

- **`Prior(basis_degree=None)` raises `ValueError`.** `basis_degree` must be an int of 0 or
  more. The old `None` was accepted. Instead, we now reject it
  at construction.

  ```python
  # before
  Prior(basis_degree=None)  # built, then raised in sample_params

  # now
  Prior(basis_degree=4)  # None raises ValueError
  ```

- **`WPPM` no longer takes `input_dim`, `extra_dims`, `variance_scale` or `decay_rate`.**
  They were duplicated on `Prior`, and nothing read 2 of the 4 copies on `WPPM`:
  `variance_scale` and `decay_rate` were stored and never used, and `extra_dims` fed only
  `embedding_dim`, silently disagreeing with the parameter shape when it differed from
  `Prior.extra_embedding_dims`. `WPPM.input_dim`, `.embedding_dim`, `.variance_scale` and
  `.decay_rate` are now read-only properties that read `self.prior`:

  ```python
  # before
  WPPM(prior, likelihood, input_dim=2, extra_dims=1, variance_scale=4e-3, decay_rate=0.4)

  # now
  Prior(input_dim=2, extra_embedding_dims=1, variance_scale=4e-3, decay_rate=0.4)
  WPPM(prior, likelihood)
  ```

  Passing any of the 4 names to `WPPM` now raises `TypeError` and says to pass them to
  `Prior` instead. `diag_term` is unchanged: it is a numerical-stability term in
  Sigma(x) = U(x) U(x)^T + diag_term * I, not a property of the prior distribution.

- **`ThresholdConfig.target_pc` is now `target_prob_correct` (cleanup only, no behavior change).**

  ```python
  # before
  ThresholdConfig(n_theta=16, n_length=1000, target_pc=2 / 3)

  # now
  ThresholdConfig(n_theta=16, n_length=1000, target_prob_correct=2 / 3)
  ```

  The default is unchanged at 2/3, the oddity task's criterion, so a
  caller relying on the default needs no edit.

- **`PosteriorMeanCovarianceField` and `contour_band` are gone, and their reductions are
  methods on the new `WPPMCovarianceFieldSamples`.** The draws are taken once, when the
  object is built, so the ensemble, the mean and the band describe one sample of the
  posterior:

  ```python
  # before
  mean_field = PosteriorMeanCovarianceField(posterior)
  covs = mean_field(ref_points)
  band = contour_band(posterior, ref_points, ci_prob=0.89, n_samples=400, key=key)

  # now
  fields = WPPMCovarianceFieldSamples(posterior)
  covs = fields.mean(ref_points)
  band = WPPMCovarianceFieldSamples(posterior, n_samples=400, key=key).credible_band(
      ref_points, ci_prob=0.89
  )
  ```

  `fields(ref_points)` gives Σ for every draw, `(n_samples, n_points, d, d)`, which
  replaces a Python loop over `posterior.sample`. `fields.samples` is the draws pytree, for
  a per-draw quantity psyphy does not ship. `WPPMCovarianceFieldSamples.from_prior(model,
  n_samples=..., key=...)` is the same object from a prior. Both take `n_samples` and `key`
  as keywords.

  `n_samples` and `key` move from the call to the construction, so one check covers every
  reduction. There is no `sqrt_cov`: an average of U Uᵀ is not U Uᵀ for any single U. Instad, use
  `jnp.linalg.cholesky(fields.mean(x))` to get the 'square root' of the mean matrix, or
  `WPPMCovarianceField` for one parameter set. The numbers are unchanged: the mean and the
  band reproduce what the removed entry points returned.

- **`WPPMPredictivePosterior` takes a required `key`, and `cov_field` is gone.**
  `mean` and `variance` are properties, so the constructor is the only place their
  randomness can enter. The key is split once into a key for drawing parameters and a key
  for the threshold sweep:

  ```python
  # before
  WPPMPredictivePosterior(posterior, X, n_samples=100)

  # now
  WPPMPredictivePosterior(posterior, X, n_samples=100, key=key)
  ```

  This removed 2 hardcoded keys, at `predictive_posterior.py:414` It computed `PosteriorMeanCovarianceField`'s quantity at
  a different default under its own key, so there were 2 public paths to
  E[Σ(X) | data] returning numbers up to a few percent apart. Use
  `WPPMCovarianceFieldSamples(posterior).mean(x)`. A model with no
  covariance field can now satisfy `PredictivePosterior`.


- **Drawing parameters now names its key** `MCMCPosterior.sample` and `MAPPosterior.sample`
  require both `n` and `key`! `MCMCPosterior.sample` used to fall back to
  `jax.random.PRNGKey(0)` and `MAPPosterior.sample` used to default `n=1`, so, the 3
  implementations now match the `ParameterSource` protocol they claim:

  ```python
  # before
  posterior.sample(100)             # seeded PRNGKey(0) somewhere you could not see
  map_posterior.sample()            # one draw

  # now
  posterior.sample(100, key=key)
  map_posterior.sample(1, key=key)  # the key goes unread for a point mass
  ```

### Changed

- **`WPPMCovarianceFieldSamples.credible_band` raises `ValueError` for 1 sample.** Before,
  it returned a band of zero width, with `lower`, `median` and `upper` equal. A plot of it
  read as certainty.

  ```python
  WPPMCovarianceFieldSamples(map_posterior).credible_band(points)  # raises ValueError
  WPPMCovarianceFieldSamples(mcmc_posterior).credible_band(points)  # unchanged
  ```

- **`WPPMCovarianceField.from_posterior` warns only for more than 1 stored sample.** Before,
  it warned for any posterior with an `n_draws`, which now includes `MAPPosterior`.

- **The model comes first in `MAPPosterior` and `MCMCPosterior`.** Both now match
  `WPPMCovarianceField(model, params)`, `from_prior(model, key)` and `fit(model, data)`.
  Calls that name the arguments still work.

  ```python
  MAPPosterior(model, params)      # was MAPPosterior(params, model)
  MCMCPosterior(model, samples)    # was MCMCPosterior(samples, model)
  MAPPosterior(params=params, model=model)  # unchanged
  ```

  The old order, and the swapped order in `WPPMCovarianceField(params, model)`, raises a
  `TypeError` when the object is built. Before, it built and failed at the first
  evaluation with `AttributeError: 'dict' object has no attribute 'input_dim'`.

- **`key` can be passed positionally** to `Prior.sample`, `ParameterSource.sample`,
  `MAPPosterior.sample`, `MCMCPosterior.sample`, `WPPM.log_likelihood_from_data`,
  `WPPM.log_posterior_from_data`, and `loglik`, `simulate` and `predict` on the task
  likelihoods. Calls that name it still work.

  ```python
  posterior.sample(100, key=key)  # unchanged
  posterior.sample(100, key)      # now also accepted
  ```

  `key` stays keyword-only in `WPPMCovarianceFieldSamples(...)` and `.from_prior`,
  `WPPMPredictivePosterior(...)` and both `rsample` methods. Pass `n_samples` and `key`
  by name to `WPPMCovarianceFieldSamples`.

- **The every-stored-sample path no longer subsamples.** `PosteriorMeanCovarianceField(posterior)`
  and `contour_band(posterior, points)` pool the stored samples in order instead of asking
  `posterior.sample` for a full-size draw. Both keep their current call shape and need no
  key, and the numbers are unchanged, because a full-size draw without replacement is a
  permutation (neither a mean nor a quantile depends on order).

  A key is now required when `n_samples` is given, since that genuinely subsamples:

  ```python
  PosteriorMeanCovarianceField(posterior)                             # unchanged
  PosteriorMeanCovarianceField(posterior, n_samples=100, key=key)     # key now required
  ```

  This also drops a `jax.random.choice` and a gather over every stored sample, run before
  only to shuffle something about to be averaged.

### Added

- **`MAPPosterior` stores its 1 sample**, as `samples`, `n_chains` and `n_draws`, in the
  layout `MCMCPosterior` uses. So `WPPMCovarianceFieldSamples(map_posterior)` takes a MAP
  fit with no `n_samples` and no key:

  ```python
  map_posterior.samples["W"].shape  # (1, 1, 3, 3, 2, 3): 1 chain, 1 sample, then W
  map_posterior.n_chains, map_posterior.n_draws  # (1, 1)

  # before
  WPPMCovarianceFieldSamples(map_posterior, n_samples=1, key=key)
  # now
  WPPMCovarianceFieldSamples(map_posterior)  # n_samples == 1
  ```

  A posterior with no `samples`, `n_chains` and `n_draws` still raises and asks for
  `n_samples` and a key.

- **`WPPMCovarianceFieldSamples.from_params(model, params)`** holds 1 parameter set, such
  as a ground truth or a published `W`, with no key. `params` has no leading sample axis.

  ```python
  fields = WPPMCovarianceFieldSamples.from_params(model, params)  # n_samples == 1
  fields(x)[0]  # (d, d), the same as model.local_covariance(params, x)
  ```

  A swapped call, `from_params(params, model)`, raises `TypeError`.

- **`MALASampler` in `psyphy.inference`**, a second MCMC engine. It runs the
  Metropolis-adjusted Langevin algorithm from BlackJAX, with one gradient per step and a
  fixed step size. It returns an `MCMCPosterior` with the same layout as `NUTSSampler`, so
  `covariance_to_arviz`, `WPPMCovarianceFieldSamples` and `plot_ellipses` take either.

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
  extends it with `params` and `model`.

  A function that only calls `sample` on a `ParameterSource` works on a prior and on a
  fitted posterior. Chain structure and a density stay off it: `sample` pools the chains, and `MCMCPosterior`
  has no `log_prob`.

- **`Prior.sample(n, *, key)`**, the vectorized form of `sample_params`. Draw `i` equals
  `sample_params(jax.random.split(key, n)[i])`, so a prior ensemble matches the Python loop
  it replaces:

  ```python
  # before
  W = jnp.stack([prior.sample_params(k)["W"] for k in jr.split(key, 100)])

  # now
  prior.sample(100, key=key)["W"]     # (100, 3, 3, 2, 3)
  ```

  `sample_params(key)` keeps its single-draw signature, because `model.init_params` is on
  the inference path and wants one unbatched draw rather than a batch of 1 to index.

- **`WPPMCovarianceFieldSamples.sqrt_cov(x)`** returns the square root `U` for every
  sample, shape `(N, d, m)` for one point or `(N, ..., d, m)` for a batch. `U U^T +
  diag_term * I` equals `fields(x)[i]` for each `i`:

  ```python
  fields = WPPMCovarianceFieldSamples(posterior)
  U = fields.sqrt_cov(x)   # (N, d, m)
  ```

  `isinstance(fields, CovarianceField)` is now `True`, since the check reads member names.
  Its outputs still carry the leading `N` axis.

### Breaking

- **`cov()` and `cov_batch()` are gone from the covariance fields.** Both methods have
  raised `DeprecationWarning` since the single entry point was introduced, and `field(x)` replaces
  both.

  ```python
  # before
  field.cov(jnp.array([0.5, 0.3]))        # (2, 2), single point only
  field.cov_batch(jnp.zeros((10, 2)))     # (10, 2, 2), batch only

  # now
  field(jnp.array([0.5, 0.3]))            # (2, 2)
  field(jnp.zeros((10, 2)))               # (10, 2, 2)
  ```

  The removal covers `WPPMCovarianceField`, `PosteriorMeanCovarianceField` and the
  `CovarianceField` `Protocol`, which now states `__call__` and `sqrt_cov`. The numbers are
  unchanged: `cov()` forwarded to the same evaluation `field(x)` uses. `sqrt_cov` and
  `sqrt_cov_batch` are untouched

### Added

- **Credible bands on the contour radii**: `contour_radii`,
  `WPPMCovarianceFieldSamples.credible_band` and `CredibleBand` in `psyphy.model`.
  `contour_radii` gives the radius of a covariance's
  one-standard-deviation ellipse (Mahalanobis-1 contour) per direction; for the oddity
  task that radius is the discrimination threshold. `credible_band` reduces those radii
  over the stored posterior samples to a pointwise credible interval.

  These names replace `threshold_radii`, `threshold_band` and `ThresholdBand`, which were
  never released: they compute a property of any covariance matrix, so the names no longer
  say "threshold". Nothing in `0.0.5` or earlier exposed them, so no installed version
  breaks. `threshold_pred` and `ThresholdConfig` in `psyphy.posterior` keep their names,
  because the contour they recover is defined by a target probability correct.

### Fixed

- **`WPPMCovarianceField.from_posterior` accepts any `ParameterPosterior`.** It read the
  private `posterior._model`, so a posterior class with `params`, `model` and `sample`
  passed `isinstance(posterior, ParameterPosterior)` and then raised `AttributeError`. It
  now reads `posterior.model`. Nothing changes for `MAPPosterior` or `MCMCPosterior`.
- **`OddityTask.predict` declared the wrong return type.** It was annotated
  `-> jnp.ndarray` while `TaskLikelihood.predict` specifies `-> tuple[jnp.ndarray, ...]`,
  and its helper `_simulate_trial_mc` was annotated the same way. Both already returned
  the one-tuple `(p_correct,)` that the base class documents, so **behavior is
  unchanged**; only the declarations were wrong.
- **`PredictivePosterior.mean` and `.variance` looked like they could return `None`.**
  The lazy cache they read is initialized to `None` and populated by
  `_ensure_computed()`; it is now annotated and the invariant is asserted
- 3 `list[np.array]` annotations in `psyphy.data.dataset` used the `np.array`
  *function* in type position instead of `np.ndarray`

### Internal

- **mypy now means 1 thing.** The commit hook, CI and local runs previously used three
  different mypy configurations that disagreed with each other. CI now runs the same
  pre-commit hook a developer runs, mypy is pinned to one version in both
  `.pre-commit-config.yaml` and the `dev` extra, and the hook checks the whole project
  rather than only staged files.
- **The commit hook can now see JAX.** It ran with `--no-site-packages`, under which every
  `jnp.ndarray` annotation in psyphy resolved to `Any`; array types were effectively
  unchecked by the only gate that blocked a commit.
- **The mypy CI job no longer carries `continue-on-error: true`.** It reported 19 errors
  on every run and failed nothing. Those are fixed and the gate is live.

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
