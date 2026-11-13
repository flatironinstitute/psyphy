"""
test_covariance_arviz.py
------------------------

Tests for covariance_to_arviz: per-draw Σ(x) at user-chosen points, as an
ArviZ DataTree, for convergence diagnostics.

Why diagnostics on Σ and not W: W is only identified up to an orthogonal
rotation of its embedding axis (W -> W R leaves Σ unchanged), so chains can
sit at different orientations, disagree about W, and still agree about every
covariance ellipse. R-hat on W then reports a failure that does not exist.
Σ(x) is invariant, but it is a function of the stimulus x, so the caller must
choose the points at which to evaluate it.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

az = pytest.importorskip("arviz", reason="arviz not installed")

from psyphy.model import WPPM, Prior  # noqa: E402
from psyphy.model.covariance_field import (  # noqa: E402
    WPPMCovarianceField,
    covariance_to_arviz,
)
from psyphy.model.likelihood import OddityTask  # noqa: E402
from psyphy.model.noise import GaussianNoise  # noqa: E402
from psyphy.posterior.mcmc_posterior import MCMCPosterior  # noqa: E402
from psyphy.posterior.posterior import MAPPosterior  # noqa: E402


@pytest.fixture(autouse=True)
def _full_matmul_precision():
    """Full float32 matmuls: TF32 on Ampere+ GPUs would blur exact comparisons.

    See test_posterior_mean_covariance_field.py for the full reasoning.
    """
    with jax.default_matmul_precision("float32"):
        yield


@pytest.fixture
def model():
    """Small 2D WPPM: W has shape (3, 3, 2, 3) -- embedding_dim m = 3."""
    return WPPM(
        prior=Prior(
            input_dim=2,
            basis_degree=2,
            variance_scale=4e-3,
            decay_rate=0.4,
            extra_embedding_dims=1,
        ),
        likelihood=OddityTask(),
        noise=GaussianNoise(),
        diag_term=1e-4,
    )


@pytest.fixture
def points():
    """Four evaluation points, shape (4, 2)."""
    return jnp.array([[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]])


def _prior_draws(model, n_chains, n_draws, seed=0):
    keys = jr.split(jr.PRNGKey(seed), n_chains * n_draws)
    W = jnp.stack([model.init_params(k)["W"] for k in keys])
    return {"W": W.reshape(n_chains, n_draws, *W.shape[1:])}


def _random_orthogonal(key, m):
    Q, _ = jnp.linalg.qr(jr.normal(key, (m, m)))
    return Q


# ==============================================================================
# Shape and content
# ==============================================================================


def test_returns_sigma_with_chain_draw_point_quantity_dims(model, points):
    posterior = MCMCPosterior(model, _prior_draws(model, 2, 5))
    dt = covariance_to_arviz(posterior, points)

    sigma = dt["posterior"]["Sigma"]
    assert sigma.dims == ("chain", "draw", "point", "quantity")
    assert sigma.shape == (2, 5, 4, 3)
    assert list(sigma["quantity"].values) == ["log_var_0", "log_var_1", "corr_0_1"]


def test_values_match_per_draw_sigma(model, points):
    """Each entry is computed from that one draw's Σ at that one point."""
    samples = _prior_draws(model, 2, 3)
    posterior = MCMCPosterior(model, samples)
    sigma = covariance_to_arviz(posterior, points)["posterior"]["Sigma"].values

    c, s = 1, 2
    S = np.asarray(WPPMCovarianceField(model, {"W": samples["W"][c, s]})(points))
    expected = np.stack(
        [
            np.log(S[:, 0, 0]),
            np.log(S[:, 1, 1]),
            S[:, 0, 1] / np.sqrt(S[:, 0, 0] * S[:, 1, 1]),
        ],
        axis=-1,
    )
    np.testing.assert_allclose(sigma[c, s], expected, rtol=1e-5)


def test_evaluation_points_are_recorded(model, points):
    """The chosen points travel with the data, so results can be traced back."""
    posterior = MCMCPosterior(model, _prior_draws(model, 1, 2))
    sigma = covariance_to_arviz(posterior, points)["posterior"]["Sigma"]
    # One 1-D coordinate per stimulus axis along `point` (xarray cannot attach
    # a (point, axis) coordinate to a variable that has no `axis` dimension).
    np.testing.assert_allclose(sigma["location_0"].values, np.asarray(points[:, 0]))
    np.testing.assert_allclose(sigma["location_1"].values, np.asarray(points[:, 1]))


def test_sample_stats_are_carried_over(model, points):
    """Divergences etc. stay available, e.g. for the ticks on az.plot_trace."""
    stats = {"diverging": np.zeros((2, 3), dtype=bool)}
    posterior = MCMCPosterior(model, _prior_draws(model, 2, 3), sampler_stats=stats)
    dt = covariance_to_arviz(posterior, points)
    assert "diverging" in dt["sample_stats"]


# ==============================================================================
# The point of the helper: no false alarm from the rotation symmetry
# ==============================================================================


def test_rhat_on_sigma_ignores_embedding_rotation(model, points):
    """Chains at different rotations of the same W: R-hat on W fails, on Σ not.

    Both chains scatter around one W0 with small noise; chain 1 is rotated by
    an orthogonal R on the embedding axis. Every draw of chain 1 has the same
    Σ distribution as chain 0, but its W values sit elsewhere.
    """
    n_draws = 300
    W0 = model.init_params(jr.PRNGKey(4))["W"]
    noise = (
        0.05 * jnp.abs(W0).mean() * jr.normal(jr.PRNGKey(5), (2, n_draws, *W0.shape))
    )
    R = _random_orthogonal(jr.PRNGKey(6), W0.shape[-1])
    chain0 = W0 + noise[0]
    chain1 = (W0 + noise[1]) @ R
    posterior = MCMCPosterior(model, {"W": jnp.stack([chain0, chain1])})

    rhat_W = az.rhat(posterior.to_arviz())["W"].values
    rhat_sigma = az.rhat(covariance_to_arviz(posterior, points))["Sigma"].values

    assert np.max(rhat_W) > 1.5  # the false alarm on W
    assert np.max(rhat_sigma) < 1.05  # Σ: chains agree


# ==============================================================================
# Input checks
# ==============================================================================


def test_points_must_match_input_dim(model):
    posterior = MCMCPosterior(model, _prior_draws(model, 1, 2))
    with pytest.raises(ValueError, match="input_dim"):
        covariance_to_arviz(posterior, jnp.ones((4, 3)))


def test_points_must_be_a_batch(model):
    """A single point (d,) is rejected: always pass shape (n_points, d)."""
    posterior = MCMCPosterior(model, _prior_draws(model, 1, 2))
    with pytest.raises(ValueError, match="n_points"):
        covariance_to_arviz(posterior, jnp.array([0.0, 0.0]))


def test_rejects_posterior_without_chains(model, points):
    """MAP has no chains, so there is nothing to diagnose."""
    posterior = MAPPosterior(model, model.init_params(jr.PRNGKey(0)))
    with pytest.raises(TypeError, match="to_arviz"):
        covariance_to_arviz(posterior, points)


def test_exported_from_psyphy_model():
    from psyphy.model import covariance_to_arviz as exported

    assert exported is covariance_to_arviz
