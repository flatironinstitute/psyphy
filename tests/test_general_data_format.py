import jax.numpy as jnp
import pytest

from psyphy.data.dataset import ResponseData


class TestResponses:
    """Test that all data structures accept appropriate continuous & > 1D responses"""

    def test_cts_responses(self):
        """1D continuous response values be accepted and manipulated appropriately."""

        # Create ResponseData
        data = ResponseData()
        stimuli = ([0.5, 0.5], [1, 0])

        # Add trials with non-binary responses:
        data.add_trial(input=stimuli, resp=1.0)
        data.add_trial(input=stimuli, resp=0.3)
        data.add_trial(input=stimuli, resp=0.82)

        # Check to ensure responses are not stored as binary:
        nb_resp = data.responses[0]
        assert nb_resp is not int, (
            "continuous response was incorrectly forced to type int."
        )

        # Convert to TrialData
        td_data = data.to_trial_data()

        # Check that TrialData still stores continuous responses:
        nb_resp = td_data.responses[1]
        assert nb_resp == 0.3, (
            "continuous response was not correctly preserved during ResponseData \
                --> TrialData conversion"
        )

        # Convert back to ResponseData & recheck
        r_data = ResponseData.from_trial_data(td_data)
        assert r_data.responses[2] == 0.82, (
            "continuous response was not correctly preserved during TrialData \
                --> ResponseData re-conversion"
        )

    def test_response_dimensions(self):
        """Test that responses can be > 1D"""

        # Create Response Data:
        stimuli = (jnp.array([0.5, 0.5]), jnp.array([1, 1]))

        # Create many different types of responses to add:
        responses = [
            1,
            0.5,
            jnp.array([0.5, 0.5]),
            jnp.array([1, 23, 0.1]),
            [1, 3],
            [1, 0.2, 0.9],
        ]

        # Add & test responses as both ResponseData and TrialData:
        for resp in responses:
            data = ResponseData()
            data.add_trial(stimuli, resp)
            assert data.responses == [resp], (
                f"The response value of {resp} was incorrectly saved as \
                    {data.responses} in ResponseData"
            )
            td_data = data.to_trial_data()
            assert (td_data.responses == jnp.asarray(resp)).all(), (
                f"The response value of {resp} was incorrectly saved as \
                    {td_data.responses} in TrialData"
            )


class TestInputs:
    """Test that all data structures correctly handle general response types.

    Should be able to handle stimuli of arbitrary dimension, and inputs with
    arbitrary stimulus length.

    Should NOT accept inputs with altered number of
    stimuli or altered stimulus dimensions once precedent has already been
    established for a given ResponseData instance.
    """

    def test_num_stim(self):
        """Test all data structures can handle when there are not exactly two stimuli"""
        response = 1
        stimulus_1 = jnp.array([1, 1])
        stimulus_2 = (jnp.array([0, 0.5]), jnp.array([1, 1]))
        stimulus_3 = (jnp.array([0, 0.5]), jnp.array([1, 1]), jnp.array([0, 6]))
        stimuli = [stimulus_1, stimulus_2, stimulus_3]
        for i, stim in enumerate(stimuli):
            data = ResponseData()
            data.add_trial(stim, response)
            data.add_trial(stim, response)
            with pytest.raises(ValueError):
                data.add_trial(stimuli[abs(i - 1)], response)
            assert len(stim) == data.stim_shape[0], (
                f"ResponseData was given {len(stim)} stimuli, but represented {data.stim_shape[0]}."
            )
            td_data = data.to_trial_data()
            assert len(stim) == td_data.inputs.shape[1], (
                f"TrialData was given {len(stim)} stimuli, but represented {data.stim_shape[0]}."
            )

    def test_stim_dem(self):
        """Test that all data structures correctly handle stimuli that are not 2D"""
        response = 1
        stim_1 = (jnp.array([0]), jnp.array([1]))
        stim_2 = (jnp.array([0, 0.5]), jnp.array([1, 1]))
        stim_3 = (jnp.array([0, 1, 0.3]), jnp.array([0.4, 0.4, 0.4]))
        stimuli = [stim_1, stim_2, stim_3]
        for i, stim in enumerate(stimuli):
            data = ResponseData()
            data.add_trial(stim, response)
            data.add_trial(stim, response)
            with pytest.raises(ValueError):
                data.add_trial(stimuli[abs(i - 1)], response)
            dim = stim[0].shape[0]
            assert dim == data.stim_shape[1], (
                f"ResponseData was given {dim}-dim stimuli, but represented \
                    {data.stim_shape[1]} dimensions."
            )
            td_data = data.to_trial_data()
            assert dim == td_data.inputs.shape[2], (
                f"TrialData was given {dim}-dim stimuli, but represented \
                    {data.stim_shape[0]} dimensions."
            )
