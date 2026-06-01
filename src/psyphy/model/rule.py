"""psyphy.model.rule

Rules for relating stimuli to responses.

For tasks which require flexible rules (e.g. ContinuousTouchTask), the relevant
rule objects live here.

Current Rule:
-----------------
ContinuousTouchRule: the rule is created using an offset, a linear coefficient,
and a function. The rule object represents the function which maps from stimulus
to response. It also records whether MC simulation is required or if there is
a closed form solution, and includes helper functions for dealing with rule
manipulations.

The public API is:

- ``ContinuousTouchRule(linear_coeff, offset, nonlinear func)``
    All fields are optional; if everything is blank, the rule will be an
    identity function.

- ``ContinuousTouchRule.requires_simulation``
    Report if MC simulation is necessary.

- ``ContinuousTouchRule.apply_rule``
    Applies the rule to a given stimulus to report the corresponding response.
"""

from typing import Callable

import jax.numpy as jnp


class ContinuousTouchRule:
    """
    This is the current representation of a rule implemented for the Continuous Touch Task.

    - Represents function which maps any given stimulus to corresponding response.
    - Represents whether MC simulation will be necessary for log likelihood calculation
    or if there is a closed-form solution.
    - Includes logic for calculating a closed-form solution if one is available.


    Attributes
    ----------
    linear_coeff : Any (array-like)
            shape = (stim_dim, 2)
        The linear transformation which relates a stim_dimensional stimulus to
        the final coordinate.
        Applied *after* any other function. For any linear rule, this can act
        as the application of the rule itself. For non-linear rules given via
        explicit function, can be left as the identity matrix.
    offset : Any (array-like)
            shape = (2,)
        Offsets to be applied to calculated coordinates to get the true final
        coordinate. If the offsets are included in the rule given via explicit
        function, offset can be left as 0 vector.
    nonlinear_func : function | None
        User-specified non-linear function to be applied to the features to get
        the final coordinates.
        If a linear function will suffice, please input it as linear_coeff & offset
    func : function
        The total function which maps from a stimulus to a touch response.
    requires_simulation : boolean
        Whether MC simulation will be required for log likelihood calculation.

    """

    def __init__(
        self,
        linear_coeff=None,  # array-like
        offset=None,  # array-like
        nonlinear_func: Callable | None = None,  # function
    ):
        # Coefficients for linear portion
        if linear_coeff is None:
            linear_coeff = jnp.array([[1, 0], [0, 1]])
        self.linear_coeff = jnp.asarray(linear_coeff)
        if offset is None:
            offset = jnp.array([0, 0])
        self.offset = jnp.squeeze(jnp.asarray(offset))

        # Check appropriate shapes:
        if self.offset.shape != (2,):
            raise ValueError(
                "Expected an offset with shape (2, ) to be applied "
                f"to 2D touch coordinates. Received {offset.shape}."
            )
        if self.linear_coeff.shape[0] != 2:
            raise ValueError(
                "Expected linear_coeff with shape (2, X) to map to"
                f"2D touch coordinates. Received ({self.linear_coeff.shape[0]}, X)"
            )

        self.nonlinear_func = nonlinear_func
        self._update_rule()

    def apply_rule(self, stimulus: jnp.array):
        """
        Applies the rule to a given stimulus. Returns the associated response.
        (Note that this also serves to get the adjusted mu for closed-form calculations.)

        Returns
        ----------
        Response (jnp.array)
        """
        stimulus = jnp.squeeze(stimulus)
        if jnp.ndim(stimulus) != 1:
            raise ValueError(
                "Expected a single stimulus in the form of a vector."
                f"Received {jnp.ndim(stimulus) - 1} unexpected additional dimensions."
            )
        return self.func(stimulus)

    def get_rule_adjusted_sigma(self, stim_sigma):
        """
        If the rule is a simple linear function, we can calculate the Σ_response
        based on the known Σ_stimulus and known linear rule.

        This holds for linear rules since linear transformations of Gaussians are
        still Gaussian, with known closed-form solutions to adjust mean and
        variance.

        For non-linear rules, the results are not guaranteed to be Gaussian, nor
        are they guaranteed to have a closed form solution at all.

        Returns
        ----------
        Σ_response (jnp.array)
        """
        if self.requires_simulation:
            raise NotImplementedError(
                "There is no closed-form solution currently"
                "implemented for this rule type."
            )

        return self.linear_coeff @ stim_sigma @ self.linear_coeff.T

    def _update_rule(self):
        """Generates & updates the complete rule function based on the current attributes.
        Updates whether or not MC simulation will be required for
        log-likelihood calculations under this rule."""
        if self.nonlinear_func is None:
            self.func = lambda x: (jnp.matmul(self.linear_coeff, x)).T + self.offset
            self.requires_simulation = False
            # If there is no nonlinear component, we can find a closed form solution!
        else:
            self.func = (
                lambda x: jnp.matmul(self.linear_coeff, self.nonlinear_func(x))
                + self.offset
            )
            self.requires_simulation = True
            # If there is a nonlinear component, we will require MC simulation.
