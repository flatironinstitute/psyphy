# Developer notes

Decisions in psyphy that are not obvious from the signature: what we were trying to achieve,
what we chose, and potentially at what it costs. Written for someone reading or extending the source (not
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

**What it costs.** The sampler explores the posterior implied by that one noise realization,
not the exact posterior. Raising `mc_samples` shrinks the difference and costs proportionally
more compute. A deterministic surrogate likelihood would remove the compromise entirely, and
that is the main argument for building one.

---

## The summaries of a fit are reductions on one draws object

**Symbols:**
[`WPPMCovarianceFieldSamples`](reference/model.md#psyphy.model.WPPMCovarianceFieldSamples),
[`WPPMCovarianceField`](reference/model.md#psyphy.model.WPPMCovarianceField).
**Checked updated:**  2026-10-28

**What we wanted.** The summaries of a sample-based posterior, without inviting the wrong
summary and without each one drawing its own samples.

**What psyphy does.** `WPPMCovarianceFieldSamples` holds N parameter draws and offers the
reductions over them as methods: the ensemble from `fields(x)`, $\mathbb{E}[\Sigma(x)]$
from `mean`, a credible band on the contour from `band`. The draws are taken once, at
construction, so a figure built from all three shows one sample of the posterior. The class
is separate from `WPPMCovarianceField` rather than a flag on it, because the two take
different inputs: one takes a parameter set, the other takes draws.

**What it costs.** Two names to learn instead of one, and a caller who wants the mean over
a different subsample builds a second object.
`WPPMCovarianceField.from_posterior` warns rather than raises when handed a sample-based
posterior, since $\Sigma(\mathbb{E}[W])$ is a legitimate object that somebody may want.
The class docstring has the identity showing why it is the wrong posterior mean.

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

The sampler statistics ride along in ArviZ's `sample_stats` group, and `NUTSSampler` spells
their keys the way ArviZ reads them:

- `diverging`
- `energy`
- `n_steps`

A new engine that fills `sampler_stats` has to use those spellings, or `az.plot_energy` and
the divergence ticks on a trace plot will not find the statistics they draw.

**What it costs.** The caller has to choose evaluation points, which is why `points` has no
default. The diagnostics then describe $\Sigma$ at those points rather than the parameters
themselves.

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
`ParameterPosterior` extends it with `params` and `model`, so the parameter source protocol is the
narrower of the two. The protocol stays asymmetric: a prior has no
point estimate, and `WPPM` holds its `Prior`, so a prior carrying a model back would close a
cycle. Drawing parameters is the overlap prior and posteriors share.

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

**What it costs.** Three draw semantics hide behind one method name: a prior generates
independent draws, a `MAPPosterior` repeats one parameter set, an `MCMCPosterior` subsamples.
The protocol docstring has to state all three, or the ambiguity stays. Requiring the key also means a caller who does not care has to write
`key=jr.PRNGKey(0)` themselves, which is the intended tradeoff: the seed is visible in their
code.
