"""
mutual_information.py
---------------------

Mutual information (information gain) acquisition function.

Also known as BALD (Bayesian Active Learning by Disagreement).
Selects points that maximize information gain about model parameters.

References
----------
Houlsby, N., Huszár, F., Ghahramani, Z., & Lengyel, M. (2011).
Bayesian active learning for classification and preference learning.
arXiv preprint arXiv:1112.5745.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import jax.random as jr

if TYPE_CHECKING:
    from psyphy.posterior import ParameterPosterior


def mutual_information(
    param_posterior: ParameterPosterior,
    X: jnp.ndarray,
    probes: jnp.ndarray | None = None,
    n_samples: int = 100,
    key: Any = None,
) -> jnp.ndarray:
    """
    Mutual information between parameters and observations.

    Computes I(θ; y | X, data) = H[p(y | X, data)] - E_θ[H[p(y | θ, X)]]

    This measures how much we expect to learn about parameters θ
    from observing response y at location X.

    Parameters
    ----------
    param_posterior : ParameterPosterior
        Posterior over model parameters p(θ | data)
    X : jnp.ndarray, shape (n_candidates, input_dim)
        Candidate reference stimuli
    probes : jnp.ndarray, shape (n_candidates, input_dim) | None
        Candidate probe stimuli. Required for discrimination tasks.
    n_samples : int, default=100
        Number of posterior samples for MC approximation
    key : jax.random.KeyArray | None
        PRNG key for sampling

    Returns
    -------
    jnp.ndarray, shape (n_candidates,)
        Mutual information scores (higher = more informative)

    Examples
    --------
    >>> # Basic usage
    >>> param_post = model.posterior(kind="parameter")
    >>> X_candidates = jnp.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
    >>> probes = X_candidates + 0.1
    >>>
    >>> mi = mutual_information(param_post, X_candidates, probes, n_samples=200)
    >>> X_next = X_candidates[jnp.argmax(mi)]

    >>> # With optimization (discrete)
    >>> def acq_fn(X):
    ...     return mutual_information(param_post, X, probes=None, n_samples=100)
    >>> X_next, mi_val = optimize_acqf_discrete(acq_fn, candidates, q=1)

    Notes
    -----
    For psychophysics, mutual information is ideal for:
    - **Threshold estimation**: Find stimuli that maximally reduce
      uncertainty about perceptual thresholds
    - **Model selection**: Distinguish between competing perceptual models
    - **Efficient design**: Minimize trials needed for desired precision

    Computational Cost
    ------------------
    Requires n_samples posterior samples and forward passes through the model.
    For large candidate sets, use optimize_acqf_discrete() for efficiency.

    Mathematical Details
    --------------------
    Let θ ~ p(θ | data_observed) be the current parameter posterior.
    Let y be the hypothetical response at candidate X.

    Mutual information:
        I(θ; y | X) = H[p(y | X)] - E_θ[H[p(y | θ, X)]]

    where:
    - H[p(y | X)] is the predictive entropy (uncertainty before observing y)
    - E_θ[H[p(y | θ, X)]] is the expected conditional entropy (average
      uncertainty given a parameter sample)

    This is approximated via MC:
        p(y | X) ≈ (1/N) Σᵢ p(y | θᵢ, X)  where θᵢ ~ p(θ | data)

    BALD Interpretation
    -------------------
    BALD (Bayesian Active Learning by Disagreement) selects points where
    different parameter samples θᵢ **disagree** most about the prediction.
    High disagreement → high information gain.
    """
    if key is None:
        key = jr.PRNGKey(0)

    # Sample from parameter posterior
    # TODO: Currently not using samples - need model.predict_prob_from_params()
    _param_samples = param_posterior.sample(n_samples, key=key)

    # Get model from posterior (assuming WPPM structure)
    # TODO: Need model.predict_prob_from_params() method
    # model = param_posterior.model  # Not yet implemented

    # Compute predictive probabilities for each parameter sample
    # Shape: (n_samples, n_candidates)
    prob_correct = []

    for _ in range(n_samples):
        # Extract parameters for this sample
        # TODO: Use params_i to compute p(correct | θᵢ, X)
        # params_i = {k: v[i] for k, v in param_samples.items()}

        # Compute p(correct | θᵢ, X)
        if probes is not None:
            # Discrimination task: compute prob of correct response
            # This requires model-specific logic
            # For WPPM: prob_correct = model.predict_prob(params_i, X, probes)

            # Placeholder: use predictive posterior mean as proxy
            # TODO: Implement model.predict_prob_from_params(params, X, probes)
            prob_i = jnp.ones(X.shape[0]) * 0.5  # Stub
        else:
            # Threshold task
            prob_i = jnp.ones(X.shape[0]) * 0.5  # Stub

        prob_correct.append(prob_i)

    # Convert list to array
    prob_correct_array = jnp.array(prob_correct)  # Shape: (n_samples, n_candidates)

    # Compute predictive entropy: H[p(y | X)]
    # p(y=1 | X) ≈ mean over samples
    pred_prob_mean = jnp.mean(prob_correct_array, axis=0)  # Shape: (n_candidates,)
    pred_entropy = binary_entropy(pred_prob_mean)

    # Compute expected conditional entropy: E_θ[H[p(y | θ, X)]]
    conditional_entropies = binary_entropy(
        prob_correct_array
    )  # Shape: (n_samples, n_candidates)
    expected_cond_entropy = jnp.mean(conditional_entropies, axis=0)

    # Mutual information = H[p(y|X)] - E_θ[H[p(y|θ,X)]]
    mi = pred_entropy - expected_cond_entropy

    # Should be non-negative (numerical stability)
    mi = jnp.maximum(mi, 0.0)

    return mi


def binary_entropy(p: jnp.ndarray) -> jnp.ndarray:
    """
    Binary entropy H(p) = -p log(p) - (1-p) log(1-p).

    Parameters
    ----------
    p : jnp.ndarray
        Probabilities in [0, 1]

    Returns
    -------
    jnp.ndarray
        Entropy values
    """
    # Clip for numerical stability
    p = jnp.clip(p, 1e-10, 1 - 1e-10)

    entropy = -p * jnp.log(p) - (1 - p) * jnp.log(1 - p)

    return entropy
