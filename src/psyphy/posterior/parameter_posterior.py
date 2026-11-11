"""
parameter_posterior.py
---------------------

Protocols for things we can draw model parameters from.

2 of them, one nested in the other:
- ParameterSource: anything that hands back n parameter draws. A Prior is one,
  and so is every posterior
- ParameterPosterior: a source that ALSO carries a point estimate and the model
  it was fitted to, representing p(θ | data)

Design
------
Different inference engines produce different posterior representations:
- MAP: a point estimate
- MCMC: collection of samples

All implement a common protocol for polymorphic use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import jax


@runtime_checkable
class ParameterSource(Protocol):
    """
    Protocol for anything we can draw model parameters from.

    Satisfied by `Prior`, `MAPPosterior` and `MCMCPosterior`.

    Notes
    -----
    `sample` pools the chains. For chain-shaped draws, use `MCMCPosterior.samples`
    or :func:`psyphy.model.covariance_to_arviz`, both of which are MCMC-only.

    A source draws parameters and cannot score them: there is no `log_prob` here.
    """

    def sample(self, n: int, key: jax.Array) -> dict:
        """
        Draw n parameter vectors.

        Parameters
        ----------
        n : int
            Number of draws.
        key : jax.Array
            PRNG key for the draw.

        Returns
        -------
        dict
            Parameter PyTree with a leading axis of n, so for WPPM
            {"W": (n, *W_shape)}.

        Notes
        -----
        What a draw is depends on the source:

        - a `Prior` generates independent draws
        - a `MAPPosterior` repeats one parameter set, so the draws are identical
        - an `MCMCPosterior` subsamples its stored draws, without replacement
          while n fits

        The draws are unweighted, so reduce them with a plain mean or quantile.
        """
        ...


@runtime_checkable
class ParameterPosterior(ParameterSource, Protocol):
    """
    Protocol for posterior distributions over model parameters p(θ | data).

    Returned by InferenceEngine.fit(model, data).
    Used for diagnostics, parameter sampling, uncertainty.

    A :class:`ParameterSource` that also carries `params` and `model`.
    """

    @property
    def params(self) -> dict:
        """
        Point estimate or posterior mean parameters.

        Returns
        -------
        dict
            Parameter PyTree (e.g., {"W": jnp.ndarray})

        Notes
        -----
        - MAP: θ_MAP
        - MCMC: posterior mean of the parameter samples. This is a parameter
          average, not an average of model outputs: for WPPM, Σ(params) is not
          the posterior mean covariance (use WPPMCovarianceFieldSamples).
        """
        ...

    @property
    def model(self):
        """
        Associated generative model.

        Returns
        -------
        Model
            The WPPM or (other model) instance used for predictions.
        """
        ...

    # sample(n, *, key) comes from ParameterSource
