"""
Tests for the continuous touch task decision rule and core algorithm.

These tests verify that the full continuous touch implementation:
1. Uses the correct decision rule, and implements MC simulation or closed-form
calculation appropriately depending on the rule type.
2. Scales rules appropriately.
3. Produces correct behavior for edge cases (nonlinear rules, mutli-feature rules, etc)
4. Behaves as expected across different scenarios
5. MC simulation appropriately approximates known closed-form solutions.
"""

import jax.numpy as jnp
import jax.random as jr
import pytest

from psyphy.model import (
    WPPM,
    ContinuousTouchRule,
    ContinuousTouchTask,
    ContinuousTouchTaskConfig,
    GaussianNoise,
    Prior,
)


class TestContinuousTouchRule:
    """Test the continuous touch rule representation."""

    def test_linear_rule(self):
        """Tests that linear rules behave appropriately."""
        coeff = jnp.array([[1, 3], [1, 2]])
        offset = jnp.array([0, 1])
        rule = ContinuousTouchRule(linear_coeff=coeff, offset=offset)

        stim = jnp.array([[0.5, 0.5]])
        true_resp = (coeff @ stim.T).T + offset

        assert jnp.all(rule.apply_rule(stim) == true_resp)
        assert not rule.requires_simulation

    def test_nonlinear_rule(self):
        coeff = jnp.array([[1, 3], [1, 2]])
        offset = jnp.array([0, 1])

        def rule_func(x):
            return jnp.array([jnp.exp(x[0]), x[1] * x[0] * x[0]])

        rule = ContinuousTouchRule(
            nonlinear_func=rule_func, linear_coeff=coeff, offset=offset
        )

        stim = jnp.array([[0.5, 0.5]])
        true_resp = (
            coeff
            @ (jnp.array([jnp.exp(stim[0, 0]), stim[0, 1] * stim[0, 0] * stim[0, 0]])).T
        ).T + offset

        assert jnp.all(rule.apply_rule(stim) == true_resp)
        assert rule.requires_simulation

    def test_mc_when_required(self):
        """Tests that the rule will correctly identify whether to use a closed-form
        solution or MC simulation.
        Ensures that appropriate errors are thrown if the wrong solution
        is attempted."""

        coeff = jnp.array([[1, 3], [1, 2]])
        offset = jnp.array([0, 1])

        def rule_func(x):
            jnp.array([jnp.exp(x[0]), x[1]])

        closed_rule = ContinuousTouchRule(linear_coeff=coeff, offset=offset)
        mc_rule = ContinuousTouchRule(
            linear_coeff=coeff, offset=offset, nonlinear_func=rule_func
        )

        assert not closed_rule.requires_simulation
        assert mc_rule.requires_simulation

        with pytest.raises(NotImplementedError):
            mc_rule.get_rule_adjusted_sigma(jnp.eye(2, 2))


class TestContinuousTouchRulePredicts:
    """Test the rule-based prediction features for ContinuousTouchTask"""

    def test_mc_vs_closed_form(self):
        """
        For a rule with a closed-form solution, MC simulation should be capable
        of approximately recovering that solution.
        """
        # Create task object for testing.
        task = ContinuousTouchTask(config=ContinuousTouchTaskConfig(num_samples=5000))

        # Create  model
        model = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=2),
            likelihood=task,
            noise=GaussianNoise(sigma=0.01),  # Small noise
        )
        params = model.init_params(jr.PRNGKey(0))

        # Create data with identical stimulus and response
        stim = jnp.array([0.5, 0.5])

        # mu, sigma from task directly. Should use closed-form by default.
        mu_closed, sigma_closed = task.predict(
            params=params, stimuli=stim, model=model, key=jr.PRNGKey(42)
        )

        # artificially force MC simulation
        task.config.rule.requires_simulation = True
        mu_sim, sigma_sim = task.predict(
            params=params, stimuli=stim, model=model, key=jr.PRNGKey(42)
        )

        # MC simulation and closed-form calculation should produce approximately similar answers.
        assert jnp.all(abs(mu_sim - mu_closed) < 0.05)
        assert jnp.all(abs(sigma_sim - sigma_closed) < 0.05)

    def test_harder_rules(self):
        """
        For a more complicated linear rule with a closed-form solution,
        MC simulation should be capable of approximately recovering that solution.
        """
        # Create complicated linear rule
        coeff = jnp.array([[0.1, 3], [5, 2]])
        offset = jnp.array([3, 1.9])
        rule = ContinuousTouchRule(linear_coeff=coeff, offset=offset)

        # Create task object for testing.
        task = ContinuousTouchTask(
            config=ContinuousTouchTaskConfig(num_samples=5000, rule=rule)
        )

        # Create  model
        model = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=2),
            likelihood=task,
            noise=GaussianNoise(sigma=0.01),  # Small noise
        )
        params = model.init_params(jr.PRNGKey(0))

        # Create some data
        stim = jnp.array([0.5, 0.9])

        # mu, sigma from task directly. Should use closed-form by default.
        mu_closed, sigma_closed = task.predict(
            params=params, stimuli=stim, model=model, key=jr.PRNGKey(42)
        )

        # artificially force MC simulation
        task.config.rule.requires_simulation = True
        mu_sim, sigma_sim = task.predict(
            params=params, stimuli=stim, model=model, key=jr.PRNGKey(42)
        )

        # MC simulation and closed-form calculation should produce approximately similar answers.
        assert jnp.all(abs(mu_sim - mu_closed) < 0.05)
        assert jnp.all(abs(sigma_sim - sigma_closed) < 0.05)
