# Developer notes

Decisions in psyphy that are not obvious from the signature: what we were trying to achieve,
what we chose, and potentially at Trade-offs: Written for someone reading or extending the source (not
for someone using the library). The tutorials under
[Examples](examples/index.md) show how to *use* these pieces.


---

## The log density uses one frozen PRNG key

**Symbols:** [`NUTSSampler`](reference/inference.md#psyphy.inference.NUTSSampler)
(`logdensity_key_seed`), [`OddityTask`](reference/model.md#psyphy.model.OddityTask).
**Checked updated:** 2026-10-27

**What we wanted.** Gradient-based samplers assume the log density is a fixed function of
the parameters. BlackJAX's NUTS integrates a trajectory and then accepts or rejects it by
comparing energies along the way, which only means anything if the same parameters give the
same density every time.

**What psyphy does** `OddityTask` has no closed-form likelihood. It estimates
$P(\text{correct})$ by simulating `mc_samples` noisy internal representations per trial, so
two calls with different keys give slightly different answers. `NUTSSampler` freezes one key
(`logdensity_key_seed`) for the whole run, so the sampler sees a deterministic function.

**Trade-offs:** The sampler explores the posterior implied by that one noise realization,
not the exact posterior. Raising `mc_samples` shrinks the difference and costs proportionally
more compute. A deterministic surrogate likelihood would remove the compromise entirely, and
that is the main argument for building one.

---

## One covariance field object, built from any fit

**Symbols:** [`WPPMCovarianceField`](reference/model.md#psyphy.model.WPPMCovarianceField).
**Checked updated:** 2026-11-22

**What we wanted.** After any fit, 1 object that gives the covariance field. Many of
psyphy's users come from MATLAB, not software engineering, so `vmap`, batching and `jit`
stay inside psyphy.

**What psyphy does.** psyphy computes the covariance field $\Sigma(x)$ in 2 ways, and both
give the same result:

- the model: `model.local_covariance(params, x)` gives $\Sigma$ for 1 parameter set at 1
  stimulus point $x$
- the field object: `fields(X)` gives $\Sigma$ for every parameter set of a fit at every
  stimulus point in `X`, in 1 call

`WPPMCovarianceField` is a wrapper around the model. It holds the model and the fit's N
parameter sets, and calls `model.local_covariance` for each set and each point. The `vmap`s
and the averages over samples happen inside it, so a scientist doesn't write them.

Every result starts with the sample axis, which has 1 entry per parameter set:

- N = 1 after a MAP fit, for a ground truth or for a published $W$
- N = the number of samples after an MCMC fit

So for `X` of shape `(n_points, d)`, with `n_points` the stimulus points we evaluate at and
$d$ the stimulus dimensions, `fields(X)` has shape `(N, n_points, d, d)`. For example, that
would be
`(1, n_points, d, d)` after a MAP fit and `(2000, n_points, d, d)` after an MCMC fit with
2000 samples. Code downstream reads this shape and doesn't have to check
whether it got a MAP point estimate or MCMC samples.

We build the object from the posterior, `WPPMCovarianceField(post)`. Without a posterior
there are 2 factories, `from_params(model, params)` and `from_prior(model, n_samples=, key=)`.
What we read off it:

- `fields(X)`: $\Sigma$ for each sample at each point
- `mean(X)`: $\mathbb{E}[\Sigma(x)]$ at each point, averaged over the samples
- `credible_band(points)`: a credible band on the one-standard-deviation contour at each point
- `sqrt_cov(X)`: $U(x)$ for each sample at each point

The object takes its samples once, when it's built, so every figure made from 1 object uses
the same samples. There is no public constructor from a dict of samples. A source psyphy
doesn't ship needs `.model`, and either `.sample(n, *, key)` or `.samples`, `.n_chains` and
`.n_draws`.

**Trade-offs:** The object adds an interface and no new computation. Each method has a
model pathway that gives the same result but with more lines of code necessary to write and no convenient reduction methods such as .mean or .credible_band():

- `fields(X)`: `jax.vmap` of `model.local_covariance` over the samples and over the points
- `mean(X)`: the same, then a mean over the sample axis
- `credible_band(points)`: the same, then `contour_radii` and a quantile over the sample axis
- `sqrt_cov(X)`: `jax.vmap` of `model._compute_sqrt`. That method is private, so the field
  object is the only public way to get $U(x)$

Where the field object falls short:

- It keeps no results between calls. `fields(X)`, `mean(X)` and `credible_band(points)` each
  go through `model.local_covariance` for every sample at every point again, so calling
  `mean(X)` after `fields(X)` on the same `X` does that work twice.
- The sample axis stays when N = 1. After a MAP fit, `fields(x)` at 1 stimulus point is
  `(1, d, d)`, and we write `fields(x)[0]` for the matrix to get shape (d,d)
- `jax.vmap(fields)(X)` evaluates 1 point at a time and puts the points first,
  `(n_points, N, d, d)`; after a MAP fit `(n_points, 1, d, d)`. The values are the same,
  but code that reads axis 0 as the samples gets the points instead: `.mean(axis=0)`
  averages over the points with no error, and `[0]` picks the first point. Pass `X` to
  `fields` directly for `(N, n_points, d, d)`. Code that has to `vmap` over the points
  writes `jax.vmap(fields, out_axes=1)(X)`, which puts the samples first again.
- The samples are fixed when the object is built. For a mean over 500 of an MCMC fit's 2000
  samples, build a second object with `WPPMCovarianceField(post, n_samples=500, key=key)`.

We'd revisit the design if users pass fields to their own code as functions of $x$ and the
sample axis gets in the way.

---

## Diagnostics run on $\Sigma$, and take the posterior rather than samples

**Symbols:** [`covariance_to_arviz`](reference/model.md#psyphy.model.covariance_to_arviz).
**Last updated:** 2026-10-27

**What we wanted.** Convergence diagnostics that report real failures and not artifacts of
the parameterization.

**What psyphy does.** $W$ is identified only up to an orthogonal rotation of its embedding
axis, so two chains can describe the same covariance field from different orientations.
R-hat on $W$ calls that non-convergence. `covariance_to_arviz` evaluates $\Sigma$ at points
the caller chooses and hands ArviZ that instead, in an `xarray.DataTree` with dimensions
`(chain, draw, point, quantity)`. It takes the posterior object rather than samples from it,
because `posterior.sample(n)` pools the chains and R-hat needs to know which chain a sample
came from.

The sampler statistics ride along in ArviZ's `sample_stats` group, and each engine spells
their keys the way ArviZ reads them. Every engine reports `acceptance_rate`. The rest
depend on the engine:

- `NUTSSampler`: `diverging`, `energy`, `n_steps`
- `MALASampler`: `is_accepted`

A new engine that reports a divergence, an energy or a tree size has to use those
spellings, or `az.plot_energy` and the divergence ticks on a trace plot will not find the
statistics they draw. R-hat and effective sample size need only the draws, so they work
the same for every engine.

**Trade-offs:** The caller has to choose evaluation points, which is why `points` has no
default. The diagnostics then describe $\Sigma$ at those points rather than the parameters
themselves.

---

## MCMC engines share one chain loop, as functions

**Symbols:** [`NUTSSampler`](reference/inference.md#psyphy.inference.NUTSSampler),
[`MALASampler`](reference/inference.md#psyphy.inference.MALASampler).
**Last updated:** 2026-11-17

**What we wanted.** A second sampler without copying the 4 parts of a fit that have nothing
to do with the sampler.

**What psyphy does.** The private module `psyphy.inference.mcmc` holds those parts:

- the log density with a fixed Monte Carlo key
- prior preconditioning, sampling z = W / prior_std and returning W
- all chains vmapped, burn-in then sampling
- the draws wrapped in an `MCMCPosterior`

An engine builds its BlackJAX kernel and calls `_fit_chains` with it. `NUTSSampler` and
`MALASampler` each subclass `InferenceEngine` only. The engines tune in different ways
(window adaptation for NUTS, none for MALA), so each one tunes before it hands over the
kernel.

**Trade-offs:** There is no public base class to subclass for a new sampler. A new engine
goes into `psyphy.inference` next to the other two. We'd add a public base when someone
outside psyphy needs to write one.

---

## A parameter source is one method, and a posterior extends it

**Symbols:** [`ParameterSource`](reference/posterior.md#psyphy.posterior.ParameterSource),
[`ParameterPosterior`](reference/posterior.md#psyphy.posterior.ParameterPosterior),
[`Prior.sample`](reference/model.md#psyphy.model.Prior.sample).
**Last updated:** 2026-10-28

**What we wanted.** One type for "a thing we can draw parameters from", so that a prior
predictive check and a posterior predictive check are the same code called twice. Before it,
`Prior.sample_params(key)` returned one draw and `ParameterPosterior.sample(n, key=...)`
returned $n$, so nothing accepted both. Four docstrings in the source built prior draws in a
Python loop and wrapped the result in an `MCMCPosterior` to get something samplable.

**What psyphy does.** `ParameterSource` declares `sample(n, *, key)` and nothing else.
`ParameterPosterior` extends it with `model`, so the parameter source protocol is the
narrower of the two. `params` is not on either: `MAPPosterior.params` is the fitted parameter
set, and `MCMCPosterior.params` raises because the posterior holds samples (its
`mean_params` is the mean of them). The protocol stays asymmetric: `WPPM` holds its `Prior`,
so a prior carrying a model back would close a cycle. Drawing parameters is the overlap prior and posteriors share.

Three things stayed off the protocol:

- **Chain structure.** `sample` pools the chains, and R-hat needs to know which chain a
  draw came from, so chain-shaped draws remain an `MCMCPosterior` capability. See the
  diagnostics note above. We're flagging this here, as this is important for downstream ArviZ diagnostic tools.
- **A density.** `MCMCPosterior` has no `log_prob`, so a `Distribution`-style base class
  requiring one would be failed by the main implementation.
- **Weights.** No psyphy engine produces weighted draws (yet). An unweighted mean over
  importance-sampled draws would be wrong. So the day one
  arrives it needs its own type instead of a silent reinterpretation of this one.

The key is a required keyword argument. A default would put the seed behind a result
somewhere other than the call site, which is how two public paths to
$\mathbb{E}[\Sigma(x)]$ came to agree at exactly 100 stored draws and disagree by a few
percent at every other fit size.

**Trade-offs:** Three draw semantics hide behind one method name: a prior generates
independent draws, a `MAPPosterior` repeats one parameter set, an `MCMCPosterior` subsamples.
The protocol docstring has to state all three, or the ambiguity stays. Requiring the key also means a caller who does not care has to write
`key=jr.PRNGKey(0)` themselves, which is the intended tradeoff: the seed is visible in their
code.
