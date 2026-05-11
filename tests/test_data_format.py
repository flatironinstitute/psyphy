import jax.numpy as jnp
import numpy as np
import pytest

from psyphy.data.dataset import ResponseData, TrialData


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
            np.array([1, 23, 0.1]),
            [1, 3],
            [1, 0.2, 0.9],
        ]

        # Add & test responses as both ResponseData and TrialData:
        for resp in responses:
            data = ResponseData()
            data.add_trial(stimuli, resp)
            assert (data.responses[0] == np.asarray(resp)).all(), (
                f"The response value of {resp} was incorrectly saved as \
                    {data.responses} in ResponseData"
            )
            td_data = data.to_trial_data()
            assert (td_data.responses == jnp.asarray(resp)).all(), (
                f"The response value of {resp} was incorrectly saved as\
                {td_data.responses} in TrialData"
            )


class TestInputs:
    """Test that all data structures correctly handle general response types.

    Should be able to handle stimuli of arbitrary dimension and arbitrary K
    (number of stimuli per trial).

    Should NOT accept stimuli with altered number of
    stimuli or altered stimulus dimensions once precedent has already been
    established for a given ResponseData instance.
    """

    def test_num_stim(self):
        """Test all data structures can handle when there are not exactly two stimuli"""
        response = 1
        stimulus_1 = (jnp.array([1, 1]),)  # K=1: wrap in tuple so len() gives K, not d
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
            assert len(stim) == td_data.stimuli.shape[1], (
                f"TrialData was given {len(stim)} stimuli, but represented {data.stim_shape[0]}."
            )

    def test_stim_shape_none_before_first_trial(self):
        """stim_shape must be None on a fresh ResponseData, not raise AttributeError."""
        data = ResponseData()
        assert data.stim_shape is None

    def test_from_arrays_2d_input(self):
        """from_arrays must accept 2D X (n_trials, d) and treat it as K=1."""
        X = np.array([[0.1, 0.2], [0.3, 0.4]])  # shape (2, d) — no K axis
        y = np.array([[1], [0]])
        data = ResponseData.from_arrays(X, y)
        assert len(data) == 2
        assert data.stim_shape == (1, 2)  # K=1, d=2

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
            assert dim == td_data.stimuli.shape[2], (
                f"TrialData was given {dim}-dim stimuli, but represented \
                    {data.stim_shape[0]} dimensions."
            )


class TestShapeValidation:
    """TrialData must reject arrays that are not exactly (N, K, d) and (N, R)."""

    def test_4d_stimuli_raises(self):
        """A 4D stimuli array must be rejected — previously slipped through > 3 check."""
        stimuli_4d = jnp.ones((5, 2, 3, 4))  # one extra axis
        responses = jnp.ones((5, 1))
        with pytest.raises(ValueError, match="stimuli must be 3D"):
            TrialData(stimuli=stimuli_4d, responses=responses)

    def test_2d_stimuli_raises(self):
        """A 2D stimuli array (missing K axis) must be rejected."""
        stimuli_2d = jnp.ones((5, 3))
        responses = jnp.ones((5, 1))
        with pytest.raises(ValueError, match="stimuli must be 3D"):
            TrialData(stimuli=stimuli_2d, responses=responses)

    def test_3d_responses_raises(self):
        """A 3D responses array must be rejected — previously slipped through > 2 check."""
        stimuli = jnp.ones((5, 2, 3))
        responses_3d = jnp.ones((5, 1, 1))
        with pytest.raises(ValueError, match="responses must be 2D"):
            TrialData(stimuli=stimuli, responses=responses_3d)

    def test_valid_shapes_accepted(self):
        """Correct (N, K, d) / (N, R) shapes must not raise."""
        TrialData(stimuli=jnp.ones((5, 2, 3)), responses=jnp.ones((5, 1)))


class TestContext:
    """Ensure that the optional attribute context behaves as expected for all
    data structures.
    """

    def test_merge_with_1d_context(self):
        """merge() must not crash when context is 1D (scalar per trial).
        Previously raised IndexError via .shape[1] on a shape-() array."""
        stim = ([0.5, 0.5], [1.0, 0.0])
        a = ResponseData()
        a.add_trial(stim, 1, context=0.3)
        b = ResponseData()
        b.add_trial(stim, 0, context=0.7)
        a.merge(b)
        assert len(a) == 2

    def test_add_1Dcontext_to_ResponseData(self):
        """Add 1D contexts"""
        data = ResponseData()
        stimuli = ([0.5, 0.5], [1, 0])
        response = 1
        c1 = 2
        c2 = 9.2
        data.add_trial(input=stimuli, resp=response, context=c1)
        assert data.contexts == [c1]
        data.add_trial(input=stimuli, resp=response, context=c2)
        assert data.contexts == [c1, c2]

    def test_add_nDcontext_to_ResponseData(self):
        """Add n-dimensional contexts"""
        data = ResponseData()
        stimuli = ([0.5, 0.5], [1, 0])
        response = 1
        c1 = [0, 2, 0.1]
        c2 = [3, 1, 2]
        data.add_trial(input=stimuli, resp=response, context=c1)
        assert (data.contexts[0] == np.asarray(c1)).all()
        data.add_trial(input=stimuli, resp=response, context=c2)
        for i, c in enumerate([c1, c2]):
            assert (data.contexts[i] == c).all()

    def test_convert_data_with_context(self):
        """test conversions between ResponseData and TrialData for instances
        with context."""

        # Create ResponseData instance
        data = ResponseData()

        # Add trial with context to ResponseData
        stimuli = ([0.5, 0.5], [1, 0])
        response = 1
        context = [0, 2, 0.1]
        data.add_trial(input=stimuli, resp=response, context=context)

        # Convert to TrialData and check context
        td_data = data.to_trial_data()
        np.testing.assert_array_equal(td_data.context, jnp.asarray([context]))

        # Convert back to ResponseData and check context
        r_data = ResponseData.from_trial_data(td_data)
        np.testing.assert_allclose(r_data.contexts[0], context)


class TestStimulusNames:
    """TrialData.stimulus_names and the .stimulus() named accessor."""

    _stimuli = jnp.array(
        [[[0.3, -0.5], [0.4, -0.4]], [[0.1, 0.2], [0.5, 0.6]]]
    )  # (N=2, K=2, d=2)
    _responses = jnp.array([[1], [0]])  # (N=2, R=1)

    def test_stimulus_names_default_empty(self):
        """stimulus_names defaults to an empty tuple when not provided."""
        data = TrialData(stimuli=self._stimuli, responses=self._responses)
        assert data.stimulus_names == ()

    def test_stimulus_names_stored(self):
        """stimulus_names is stored and retrievable."""
        data = TrialData(
            stimuli=self._stimuli,
            responses=self._responses,
            stimulus_names=("ref", "comp"),
        )
        assert data.stimulus_names == ("ref", "comp")

    def test_stimulus_accessor_returns_correct_slice(self):
        """data.stimulus('ref') returns stimuli[:, 0, :] for slot 0."""
        data = TrialData(
            stimuli=self._stimuli,
            responses=self._responses,
            stimulus_names=("ref", "comp"),
        )
        np.testing.assert_array_equal(data.stimulus("ref"), self._stimuli[:, 0, :])
        np.testing.assert_array_equal(data.stimulus("comp"), self._stimuli[:, 1, :])

    def test_stimulus_accessor_unknown_name_raises(self):
        """data.stimulus() with an unrecognised name raises ValueError."""
        data = TrialData(
            stimuli=self._stimuli,
            responses=self._responses,
            stimulus_names=("ref", "comp"),
        )
        with pytest.raises(ValueError, match="unknown stimulus name"):
            data.stimulus("nonexistent")

    def test_stimulus_accessor_empty_names_raises(self):
        """data.stimulus() raises if stimulus_names was not set."""
        data = TrialData(stimuli=self._stimuli, responses=self._responses)
        with pytest.raises(ValueError, match="stimulus_names"):
            data.stimulus("ref")
