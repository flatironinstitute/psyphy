"""
bootstrap.py
------------

Bootstrap resampling for frequentist confidence intervals.

Provides non-parametric uncertainty estimates via resampling. Useful for:
- Model diagnostics (is MAP estimate stable?)
- Note:assumes iid data

NOT recommended for:
- Acquisition functions (use Bayesian posterior.variance instead)
- Online learning (too expensive - requires N refits)
- Real-time decisions (computationally intensive)

Examples
--------
>>> # Bootstrap CIs for psychometric function
>>> from psyphy.utils.bootstrap import bootstrap_predictions
>>> model = WPPM(input_dim=2, ...)
>>> mean, lower, upper = bootstrap_predictions(
...     model,
...     X_train,
...     y_train,
...     X_test,
...     probes=probes_test,
...     n_bootstrap=100,
...     key=jr.PRNGKey(0),
... )

>>> # Bootstrap CI for threshold estimate
>>> from psyphy.utils.bootstrap import bootstrap_statistic
>>> def get_threshold(fitted_model):
...     return fitted_model.estimate_threshold(criterion=0.75)
>>> threshold, lower, upper = bootstrap_statistic(
...     model, X, y, get_threshold, n_bootstrap=200, key=jr.PRNGKey(0)
... )

References
----------
Efron, B., & Tibshirani, R. J. (1994). An introduction to the bootstrap.
CRC press.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import jax.numpy as jnp
import jax.random as jr

if TYPE_CHECKING:
    from psyphy.model import Model


def bootstrap_predictions(
    model: Model,
    X_train: jnp.ndarray,
    y_train: jnp.ndarray,
    X_test: jnp.ndarray,
    *,
    n_bootstrap: int = 100,
    probes: jnp.ndarray | None = None,
    confidence_level: float = 0.95,
    inference: str = "map",
    inference_config: dict[str, Any] | None = None,
    key: Any,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Bootstrap confidence intervals for model predictions.

    Resamples training data with replacement, refits model N times,
    and computes prediction quantiles at test points.

    Parameters
    ----------
    model : Model
        Unfitted model instance (will be cloned for each bootstrap sample)
    X_train : jnp.ndarray, shape (n_train, ...)
        Training stimuli
    y_train : jnp.ndarray, shape (n_train,)
        Training responses
    X_test : jnp.ndarray, shape (n_test, ...)
        Test points for predictions
    n_bootstrap : int, default=100
        Number of bootstrap samples.
        Typical values: 100 (quick), 1000 (publication quality)
    probes : jnp.ndarray, optional
        Test probes for discrimination tasks
    confidence_level : float, default=0.95
        Confidence level (e.g., 0.95 for 95% CI, 0.99 for 99% CI)
    inference : str, default="map"
        Inference method for each bootstrap fit
    inference_config : dict, optional
        Configuration for inference engine
    key : Any
        JAX random key for reproducibility

    Returns
    -------
    mean_estimate : jnp.ndarray, shape (n_test,)
        Average prediction across bootstrap samples
    ci_lower : jnp.ndarray, shape (n_test,)
        Lower confidence bound at each test point
    ci_upper : jnp.ndarray, shape (n_test,)
        Upper confidence bound at each test point


    """
    n_train = len(X_train)
    alpha = 1 - confidence_level

    # Store predictions from each bootstrap sample
    predictions = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        key, subkey = jr.split(key)
        indices = jr.randint(subkey, (n_train,), 0, n_train)
        X_boot = X_train[indices]
        y_boot = y_train[indices]

        # Create fresh model instance and fit
        model_boot = _clone_model(model)
        model_boot.fit(
            X_boot,
            y_boot,
            inference=inference,
            inference_config=inference_config,
        )

        # Get predictions
        posterior_boot = model_boot.posterior(X_test, probes=probes)
        predictions.append(posterior_boot.mean)  # type: ignore[attr-defined, union-attr]

    # Stack and compute quantiles
    predictions = jnp.stack(predictions, axis=0)  # (n_bootstrap, n_test)

    mean_estimate = jnp.mean(predictions, axis=0)
    ci_lower = jnp.percentile(predictions, 100 * alpha / 2, axis=0)
    ci_upper = jnp.percentile(predictions, 100 * (1 - alpha / 2), axis=0)

    return mean_estimate, ci_lower, ci_upper


def bootstrap_statistic(
    model: Model,
    X: jnp.ndarray,
    y: jnp.ndarray,
    statistic_fn: Callable[[Model], float | jnp.ndarray],
    *,
    n_bootstrap: int = 100,
    confidence_level: float = 0.95,
    inference: str = "map",
    inference_config: dict[str, Any] | None = None,
    key: Any,
) -> tuple[float | jnp.ndarray, float | jnp.ndarray, float | jnp.ndarray]:
    """
    Bootstrap confidence interval for any model-derived statistic.

    Resamples data, refits model, and computes statistic for each
    bootstrap sample. Returns point estimate and confidence interval.

    Parameters
    ----------
    model : Model
        Unfitted model instance
    X : jnp.ndarray, shape (n_trials, ...)
        Training stimuli
    y : jnp.ndarray, shape (n_trials,)
        Training responses
    statistic_fn : callable
        Function that takes a fitted Model and returns a scalar or array.
        Examples:
        - lambda m: m.estimate_threshold(criterion=0.75)
        - lambda m: m.posterior(X_test).mean
        - lambda m: jnp.linalg.norm(m._posterior.params["decay_rates"])
    n_bootstrap : int, default=100
        Number of bootstrap samples
    confidence_level : float, default=0.95
        Confidence level for interval
    inference : str, default="map"
        Inference method
    inference_config : dict, optional
        Inference configuration
    key : jr.KeyArray
        Random key

    Returns
    -------
    estimate : float or jnp.ndarray
        Point estimate (mean across bootstrap samples)
    ci_lower : float or jnp.ndarray
        Lower confidence bound
    ci_upper : float or jnp.ndarray
        Upper confidence bound

    Examples
    --------
    >>> # Bootstrap CI for threshold estimate
    >>> def get_threshold(fitted_model):
    ...     # Example threshold estimation
    ...     X_grid = jnp.linspace(-2, 2, 100)[:, None]
    ...     probes = X_grid + 0.1
    ...     posterior = fitted_model.posterior(X_grid, probes=probes)
    ...     probs = posterior.mean
    ...     idx = jnp.argmin(jnp.abs(probs - 0.75))
    ...     return X_grid[idx, 0]
    >>>
    >>> threshold, lower, upper = bootstrap_statistic(
    ...     model,
    ...     X,
    ...     y,
    ...     statistic_fn=get_threshold,
    ...     n_bootstrap=200,
    ...     key=jr.PRNGKey(0),
    ... )
    >>> print(f"Threshold: {threshold:.3f} [{lower:.3f}, {upper:.3f}]")

    >>> # Bootstrap CI for model parameter
    >>> def get_decay_rate(fitted_model):
    ...     return fitted_model._posterior.params["decay_rates"][0]
    >>>
    >>> ls, ls_lower, ls_upper = bootstrap_statistic(
    ...     model,
    ...     X,
    ...     y,
    ...     statistic_fn=get_decay_rate,
    ...     n_bootstrap=100,
    ...     key=jr.PRNGKey(42),
    ... )

    >>> # Compare two models
    >>> def test_accuracy(fitted_model):
    ...     posterior = fitted_model.posterior(X_test, probes=probes_test)
    ...     preds = (posterior.mean > 0.5).astype(int)
    ...     return jnp.mean(preds == y_test)
    >>>
    >>> acc1, l1, u1 = bootstrap_statistic(
    ...     model1, X_train, y_train, test_accuracy, n_bootstrap=100, key=jr.PRNGKey(0)
    ... )
    >>> acc2, l2, u2 = bootstrap_statistic(
    ...     model2, X_train, y_train, test_accuracy, n_bootstrap=100, key=jr.PRNGKey(1)
    ... )
    >>> # If CIs don't overlap, difference is statistically significant
    >>> print(f"Model 1: {acc1:.3f} [{l1:.3f}, {u1:.3f}]")
    >>> print(f"Model 2: {acc2:.3f} [{l2:.3f}, {u2:.3f}]")

    Notes
    -----
    This is a general-purpose function for any statistic you can compute
    from a fitted model. The statistic_fn should:
    - Take a fitted Model as input
    - Return a scalar or array (but shape must be consistent across samples)
    - Not modify the model

    For vector-valued statistics, confidence intervals are computed
    element-wise.
    """
    n_train = len(X)
    alpha = 1 - confidence_level

    statistics = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        key, subkey = jr.split(key)
        indices = jr.randint(subkey, (n_train,), 0, n_train)
        X_boot = X[indices]
        y_boot = y[indices]

        # Fit and compute statistic
        model_boot = _clone_model(model)
        model_boot.fit(
            X_boot,
            y_boot,
            inference=inference,
            inference_config=inference_config,
        )

        stat = statistic_fn(model_boot)
        statistics.append(stat)

    # Stack and compute quantiles
    statistics = jnp.stack(statistics, axis=0)

    estimate = jnp.mean(statistics, axis=0)
    ci_lower = jnp.percentile(statistics, 100 * alpha / 2, axis=0)
    ci_upper = jnp.percentile(statistics, 100 * (1 - alpha / 2), axis=0)

    return estimate, ci_lower, ci_upper


def bootstrap_compare_models(
    model1: Model,
    model2: Model,
    X_train: jnp.ndarray,
    y_train: jnp.ndarray,
    X_test: jnp.ndarray,
    y_test: jnp.ndarray,
    *,
    metric_fn: Callable[[jnp.ndarray, jnp.ndarray], float] | None = None,
    n_bootstrap: int = 100,
    confidence_level: float = 0.95,
    probes: jnp.ndarray | None = None,
    inference: str = "map",
    inference_config: dict[str, Any] | None = None,
    key: Any,
) -> tuple[float, float, float, bool]:
    """
    Bootstrap comparison of two models' predictive performance.

    Tests whether model1 performs significantly better/worse than model2
    by computing confidence intervals on the performance difference.

    Parameters
    ----------
    model1, model2 : Model
        Unfitted model instances to compare
    X_train, y_train : Training data
    X_test, y_test : Test data for evaluation
    metric_fn : callable, optional
        Function that takes (y_true, y_pred) and returns a scalar.
        Default: accuracy for binary classification
    n_bootstrap : int, default=100
        Number of bootstrap samples
    confidence_level : float, default=0.95
        Confidence level
    probes : optional
        Test probes for discrimination tasks
    inference : str
        Inference method
    inference_config : dict, optional
        Inference configuration
    key : jr.KeyArray
        Random key

    Returns
    -------
    diff_estimate : float
        Estimated difference in performance (model1 - model2)
        Positive = model1 is better
    ci_lower : float
        Lower bound on difference
    ci_upper : float
        Upper bound on difference
    is_significant : bool
        True if the difference is statistically significant
        (i.e., confidence interval excludes zero)

    Examples
    --------
    >>> # Compare two models on held-out data
    >>> from psyphy.utils.bootstrap import bootstrap_compare_models
    >>>
    >>> model1 = WPPM(input_dim=2, prior=Prior(input_dim=2, scale=0.5), ...)
    >>> model2 = WPPM(input_dim=2, prior=Prior(input_dim=2, scale=1.0), ...)
    >>>
    >>> diff, lower, upper, significant = bootstrap_compare_models(
    ...     model1,
    ...     model2,
    ...     X_train,
    ...     y_train,
    ...     X_test,
    ...     y_test,
    ...     n_bootstrap=200,
    ...     key=jr.PRNGKey(0),
    ... )
    >>>
    >>> if significant:
    ...     winner = "Model 1" if diff > 0 else "Model 2"
    ...     print(f"{winner} is significantly better")
    ...     print(f"Difference: {diff:.3f} [{lower:.3f}, {upper:.3f}]")
    >>> else:
    ...     print("No significant difference")

    >>> # Custom metric: mean squared error
    >>> def mse(y_true, y_pred):
    ...     return jnp.mean((y_true - y_pred) ** 2)
    >>>
    >>> diff, lower, upper, sig = bootstrap_compare_models(
    ...     model1,
    ...     model2,
    ...     X_train,
    ...     y_train,
    ...     X_test,
    ...     y_test,
    ...     metric=mse,  # Lower is better
    ...     n_bootstrap=100,
    ...     key=jr.PRNGKey(1),
    ... )

    Notes
    -----
    This function performs paired bootstrap comparison: for each
    bootstrap sample, both models are fit on the same resampled
    training data and evaluated on the same test data. This controls
    for data sampling variability.

    The null hypothesis is: "models have equal performance"
    We reject this if the CI on the difference excludes zero.
    """
    # Set default metric if not provided
    _metric: Callable[[Any, Any], float]
    if metric_fn is None:
        # Default: accuracy
        def default_metric(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> float:
            return float(jnp.mean(y_pred == y_true))

        _metric = default_metric
    else:
        _metric = metric_fn  # type: ignore[assignment]

    n_train = len(X_train)
    alpha = 1 - confidence_level

    differences = []

    for _ in range(n_bootstrap):
        # Resample training data (same sample for both models)
        key, subkey = jr.split(key)
        indices = jr.randint(subkey, (n_train,), 0, n_train)
        X_boot = X_train[indices]
        y_boot = y_train[indices]

        # Fit model 1
        m1_boot = _clone_model(model1)
        m1_boot.fit(
            X_boot,
            y_boot,
            inference=inference,
            inference_config=inference_config,
        )

        # Fit model 2
        m2_boot = _clone_model(model2)
        m2_boot.fit(
            X_boot,
            y_boot,
            inference=inference,
            inference_config=inference_config,
        )

        # Evaluate on test data
        post1 = m1_boot.posterior(X_test, probes=probes)
        post2 = m2_boot.posterior(X_test, probes=probes)

        # Convert to predictions (threshold at 0.5)
        y_pred1 = (post1.mean > 0.5).astype(int)  # type: ignore[attr-defined, union-attr]
        y_pred2 = (post2.mean > 0.5).astype(int)  # type: ignore[attr-defined, union-attr]

        # Compute metrics
        score1 = _metric(y_test, y_pred1)
        score2 = _metric(y_test, y_pred2)

        differences.append(score1 - score2)

    # Stack and compute statistics
    differences = jnp.array(differences)

    diff_estimate = float(jnp.mean(differences))
    ci_lower = float(jnp.percentile(differences, 100 * alpha / 2))
    ci_upper = float(jnp.percentile(differences, 100 * (1 - alpha / 2)))

    # Test significance: CI excludes zero?
    is_significant = bool((ci_lower > 0) or (ci_upper < 0))

    return diff_estimate, ci_lower, ci_upper, is_significant


def _clone_model(model: Model) -> Model:
    """
    Create a fresh copy of a model for bootstrap resampling.

    Parameters
    ----------
    model : Model
        Model to clone (should be unfitted or we'll create a fresh instance)

    Returns
    -------
    Model
        New instance with same configuration
    """
    # Get model class
    model_class = model.__class__

    # Try to get initialization kwargs
    # Most models should store these for cloning
    if hasattr(model, "_init_kwargs"):
        return model_class(**model._init_kwargs)  # type: ignore[attr-defined]

    # Fallback: try to reconstruct from attributes
    # This works for WPPM and similar models
    init_kwargs: dict[str, Any] = {}

    if hasattr(model, "input_dim"):
        init_kwargs["input_dim"] = model.input_dim  # type: ignore[attr-defined]
    if hasattr(model, "prior"):
        init_kwargs["prior"] = model.prior  # type: ignore[attr-defined]
    if hasattr(model, "task"):
        init_kwargs["task"] = model.task  # type: ignore[attr-defined]
    if hasattr(model, "noise"):
        init_kwargs["noise"] = model.noise  # type: ignore[attr-defined]

    return model_class(**init_kwargs)
