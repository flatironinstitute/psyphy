"""
tests/test_mala_sampler.py
--------------------------

Tests for MALASampler, the second MCMC engine on the chain loop in
``psyphy.inference.mcmc``.

The draws a MALA fit returns go through the same downstream path as NUTS
draws: ArviZ diagnostics on Σ, the covariance field over the samples, and the
ellipse plot. The end-to-end test runs that path, so a change to the shared
loop that breaks the layout of ``MCMCPosterior`` fails here as well as in the
NUTS tests.

Requires blackjax (``pip install 'psyphy[sampling]'``).
"""

from __future__ import annotations

import sys

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

blackjax = pytest.importorskip("blackjax", reason="blackjax not installed")

from psyphy.data.dataset import TrialData  # noqa: E402
from psyphy.inference import MALASampler  # noqa: E402
from psyphy.model import WPPM, GaussianNoise, OddityTask, Prior  # noqa: E402
from psyphy.model.likelihood import OddityTaskConfig  # noqa: E402
from psyphy.posterior.mcmc_posterior import MCMCPosterior  # noqa: E402

NUM_CHAINS = 2
NUM_SAMPLES = 10


@pytest.fixture(scope="module")
def tiny_model():
    """Minimal WPPM: degree=1, 2D, 1 extra dim -> W shape (2, 2, 2, 3)."""
    return WPPM(
        prior=Prior(input_dim=2, basis_degree=1, extra_embedding_dims=1),
        likelihood=OddityTask(config=OddityTaskConfig(num_samples=5)),
        noise=GaussianNoise(),
    )


@pytest.fixture(scope="module")
def tiny_data():
    """5 trials with refs and comparisons inside [-0.5, 0.5]."""
    refs = jr.uniform(jr.PRNGKey(0), (5, 2), minval=-0.5, maxval=0.5)
    comparisons = jnp.clip(refs + jr.normal(jr.PRNGKey(1), (5, 2)) * 0.05, -1.0, 1.0)
    return TrialData(
        stimuli=jnp.stack([refs, comparisons], axis=1),  # (5, K=2, 2)
        responses=jnp.ones(5, dtype=jnp.int32),
    )


def _sampler(**kwargs):
    return MALASampler(
        num_warmup=20,
        num_samples=NUM_SAMPLES,
        num_chains=NUM_CHAINS,
        step_size=kwargs.pop("step_size", 0.01),
        show_progress=False,
        **kwargs,
    )


@pytest.fixture(scope="module")
def posterior(tiny_model, tiny_data):
    """One MALA fit shared by the module."""
    return _sampler().fit(tiny_model, tiny_data, seed=3)


def test_import_error_without_blackjax(tiny_model, tiny_data, monkeypatch):
    """Without blackjax, fit names the extra to install."""
    monkeypatch.setitem(sys.modules, "blackjax", None)
    with pytest.raises(ImportError, match="MALASampler.*psyphy\\[sampling\\]"):
        _sampler().fit(tiny_model, tiny_data)


def test_step_size_is_required():
    """MALA does not tune itself, so there is no default step size."""
    with pytest.raises(TypeError, match="step_size"):
        MALASampler()


def test_nonpositive_step_size_raises():
    with pytest.raises(ValueError, match="step_size must be > 0"):
        MALASampler(step_size=0.0)


def test_returns_mcmc_posterior_with_nuts_layout(posterior):
    """Draws are (n_chains, n_draws, *W_shape), the layout every consumer reads."""
    assert isinstance(posterior, MCMCPosterior)
    assert posterior.n_chains == NUM_CHAINS
    assert posterior.n_draws == NUM_SAMPLES
    assert posterior.samples["W"].shape == (NUM_CHAINS, NUM_SAMPLES, 2, 2, 2, 3)
    assert bool(jnp.all(jnp.isfinite(posterior.samples["W"])))


def test_sampler_stats_keys_and_shapes(posterior):
    """MALA reports acceptance only: no divergences, energies or tree sizes."""
    assert set(posterior.sampler_stats) == {"acceptance_rate", "is_accepted"}
    for v in posterior.sampler_stats.values():
        assert v.shape == (NUM_CHAINS, NUM_SAMPLES)
    acc = np.asarray(posterior.sampler_stats["acceptance_rate"])
    assert np.all((acc >= 0) & (acc <= 1))


def test_same_seed_gives_same_draws(tiny_model, tiny_data, posterior):
    """seed is the only source of randomness in a fit."""
    again = _sampler().fit(tiny_model, tiny_data, seed=3)
    np.testing.assert_array_equal(posterior.samples["W"], again.samples["W"])


def test_downstream_path_end_to_end(posterior):
    """fit -> covariance_to_arviz -> WPPMCovarianceField -> plot_ellipses.

    The same calls the NUTS tutorial makes on a NUTS posterior.
    """
    pytest.importorskip("arviz", reason="arviz not installed")
    pytest.importorskip("matplotlib", reason="psyphy[viz] not installed")
    import matplotlib.pyplot as plt

    from psyphy.model import WPPMCovarianceField, covariance_to_arviz
    from psyphy.viz import plot_ellipses

    points = jnp.array([[-0.3, -0.3], [0.0, 0.0], [0.3, 0.3]])  # (3, 2)

    dt = covariance_to_arviz(posterior, points)
    sigma = dt["posterior"]["Sigma"]
    assert sigma.dims[:3] == ("chain", "draw", "point")
    assert sigma.shape[:3] == (NUM_CHAINS, NUM_SAMPLES, 3)
    assert "acceptance_rate" in dt["sample_stats"]

    mean_covs = WPPMCovarianceField(posterior).mean(points)  # (3, 2, 2)
    assert mean_covs.shape == (3, 2, 2)

    ax = plot_ellipses(np.asarray(points), np.asarray(mean_covs))
    assert len(ax.collections) == 1  # one field
    plt.close(ax.figure)
