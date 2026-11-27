"""
tests/test_covariance_field_samples.py
--------------------------------------

Tests for WPPMCovarianceField: one object holding N parameter draws, with
the reductions over them as methods.

The golden arrays below were captured from PosteriorMeanCovarianceField and
contour_band before those two moved onto this class, so they pin behavior
across the move. A failure here means the moved code computes something
different.

Draws are built by hand, as in test_contour_band.py, so the suite runs with no
sampling backend installed.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from psyphy.model import WPPM, GaussianNoise, OddityTask, Prior
from psyphy.model.covariance_field import WPPMCovarianceField
from psyphy.model.likelihood import OddityTaskConfig
from psyphy.posterior.mcmc_posterior import MCMCPosterior
from psyphy.posterior.posterior import MAPPosterior

# Σ entries here run to 0.12 and contour radii from 0.0072 to 0.57. Two summation
# orders over the same draws sit about 1e-6 apart at worst, so atol=1e-4 is 100 times
# the noise floor. Every real regression clears it, measured: averaging W before Σ
# moves the mean by 0.12, which is 1200x the tolerance; halving the draw count 40x;
# ci_kind="hdi" instead of "eti" 309x; and the tightest, ci_prob=0.90 instead of
# 0.89, still 20x.
ATOL = 1e-4

MEAN_ALL = [
    [
        [0.12208183109760284, 0.031568314880132675],
        [0.031568314880132675, 0.12228402495384216],
    ],
    [
        [0.09403368830680847, 0.0050445133820176125],
        [0.0050445133820176125, 0.0926375687122345],
    ],
]
# lower, median and upper at direction indices 0, 45 and 90 of the 181 the band returns
BAND_IDX = [0, 45, 90]
BAND_LOWER = [
    [0.011567057110369205, 0.008865531533956528, 0.011567061766982079],
    [0.04944310709834099, 0.025759821757674217, 0.04944310709834099],
]
BAND_MEDIAN = [
    [0.0995648205280304, 0.10730376839637756, 0.09956485778093338],
    [0.18460702896118164, 0.19025801122188568, 0.1846071481704712],
]
BAND_UPPER = [
    [0.459035187959671, 0.4504614770412445, 0.45903515815734863],
    [0.4132041335105896, 0.4352034330368042, 0.41320401430130005],
]


@pytest.fixture(autouse=True)
def _full_matmul_precision():
    """Full float32 matmuls: TF32 on Ampere+ GPUs would blur the golden comparisons."""
    with jax.default_matmul_precision("float32"):
        yield


@pytest.fixture
def model():
    """Minimal WPPM: degree=1, 2D, 1 extra dim -> W shape (2, 2, 2, 3)."""
    return WPPM(
        prior=Prior(input_dim=2, basis_degree=1, extra_embedding_dims=1),
        likelihood=OddityTask(config=OddityTaskConfig(num_samples=5)),
        noise=GaussianNoise(),
    )


@pytest.fixture
def posterior(model):
    """MCMCPosterior with hand-made draws: 2 chains x 25 draws = 50 stored.

    The draws come from arithmetic rather than jax.random, because the golden
    arrays above are literal values. ``jax.random.normal`` returns different
    numbers depending on ``jax_threefry_partitionable``, which is True in current
    JAX and False in the older JAX that CI resolves on Python 3.10, so a
    PRNG-built fixture pins the literals to one JAX version. This sawtooth is
    exact in float32 and identical everywhere.
    """
    n = 2 * 25 * 2 * 2 * 2 * 3
    idx = jnp.arange(n, dtype=jnp.float32)
    W = (((idx * 37.0) % 19.0) / 19.0 - 0.5).reshape(2, 25, 2, 2, 2, 3) * 0.6
    return MCMCPosterior(model, {"W": W})


@pytest.fixture
def X():
    """Two evaluation points, shape (n_points=2, d=2)."""
    return jnp.array([[-0.4, 0.2], [0.5, -0.3]])


def test_mean_over_every_stored_draw(posterior, X):
    """E[Σ(x)] over all 50 stored draws, in stored order."""
    fields = WPPMCovarianceField(posterior)
    got = fields.mean(X)  # (n_points=2, d=2, d=2)
    assert got.shape == (2, 2, 2)
    assert jnp.allclose(got, jnp.array(MEAN_ALL), rtol=0.0, atol=ATOL)


def test_mean_over_a_subsample(posterior, X, model):
    """E[Σ(x)] over 10 draws averages exactly the 10 draws the object holds.

    No golden array here: which 10 of the 50 a key picks is ``jr.choice``'s
    business and moves with the JAX version. What has to hold is that the mean
    uses the draws the object kept, whichever they are.
    """
    fields = WPPMCovarianceField(posterior, n_samples=10, key=jr.PRNGKey(3))
    assert fields.n_samples == 10

    by_hand = jnp.mean(
        jnp.stack([_sigma(model, W, X) for W in fields.samples["W"]]),
        axis=0,
    )
    assert jnp.allclose(fields.mean(X), by_hand, rtol=0.0, atol=ATOL)


def test_band_over_every_stored_draw(posterior, X):
    """The 89% band on the one-standard-deviation ellipse (Mahalanobis-1 contour), over all stored draws."""
    band = WPPMCovarianceField(posterior).credible_band(X, ci_prob=0.89)
    assert band.lower.shape == (2, 181)  # (n_points, n_directions)
    assert band.directions.shape == (181, 2)
    assert (band.ci_prob, band.ci_kind) == (0.89, "eti")
    assert jnp.allclose(
        band.lower[:, BAND_IDX], jnp.array(BAND_LOWER), rtol=0.0, atol=ATOL
    )
    assert jnp.allclose(
        band.median[:, BAND_IDX], jnp.array(BAND_MEDIAN), rtol=0.0, atol=ATOL
    )
    assert jnp.allclose(
        band.upper[:, BAND_IDX], jnp.array(BAND_UPPER), rtol=0.0, atol=ATOL
    )


# ==============================================================================
# The draws, and the reductions over them
# ==============================================================================


def test_call_gives_one_matrix_per_draw(posterior, X, model):
    """fields(x) is the ensemble: draw i equals the field built from draw i."""
    fields = WPPMCovarianceField(posterior)
    ensemble = fields(X)  # (n_samples, n_points, d, d)
    assert ensemble.shape == (50, 2, 2, 2)

    pooled = posterior.samples["W"].reshape(50, *posterior.samples["W"].shape[2:])
    for i in (0, 17, 49):
        one = _sigma(model, pooled[i], X)
        assert jnp.allclose(ensemble[i], one, rtol=0.0, atol=ATOL)


def test_mean_is_the_mean_of_the_draws(posterior, X):
    """mean() and the ensemble average are the same reduction."""
    fields = WPPMCovarianceField(posterior)
    assert jnp.allclose(fields.mean(X), fields(X).mean(axis=0), rtol=0.0, atol=ATOL)


def test_single_point_matches_the_batch_row(posterior, X):
    """Shape (d,) input gives the same matrices as the matching batch row."""
    fields = WPPMCovarianceField(posterior)
    assert fields(X[0]).shape == (50, 2, 2)
    assert fields.mean(X[0]).shape == (2, 2)
    assert jnp.allclose(fields(X[0]), fields(X)[:, 0], rtol=0.0, atol=ATOL)
    assert jnp.allclose(fields.mean(X[0]), fields.mean(X)[0], rtol=0.0, atol=ATOL)


def test_samples_carries_the_draws(posterior):
    """samples is public, so a per-draw quantity psyphy does not ship is reachable."""
    fields = WPPMCovarianceField(posterior)
    assert fields.samples["W"].shape[0] == fields.n_samples == 50
    # any per-draw function, mapped by the caller
    traces = jax.vmap(lambda W: jnp.sum(W**2))(fields.samples["W"])
    assert traces.shape == (50,)


def test_arbitrary_batch_dimensions(posterior, model):
    """A grid of points keeps its shape, with the draws on the leading axis."""
    grid = jnp.zeros((4, 3, 2))  # (4, 3, d=2)
    fields = WPPMCovarianceField(posterior, n_samples=5, key=jr.PRNGKey(0))
    assert fields(grid).shape == (5, 4, 3, 2, 2)
    assert fields.mean(grid).shape == (4, 3, 2, 2)


# ==============================================================================
# Construction
# ==============================================================================


def test_from_prior_takes_n_samples_and_key(model):
    """A prior stores no draws, so both are required."""
    fields = WPPMCovarianceField.from_prior(model, n_samples=7, key=jr.PRNGKey(0))
    assert fields.n_samples == 7
    assert fields.samples["W"].shape[0] == 7
    with pytest.raises(TypeError):
        WPPMCovarianceField.from_prior(model, n_samples=7)


def test_constructors_take_the_count_by_the_same_keyword(model, posterior):
    """from_prior and the constructor both spell it n_samples, keyword-only."""
    key = jr.PRNGKey(0)
    assert WPPMCovarianceField.from_prior(model, n_samples=4, key=key).n_samples == 4
    assert WPPMCovarianceField(posterior, n_samples=4, key=key).n_samples == 4
    with pytest.raises(TypeError):
        WPPMCovarianceField.from_prior(model, 4, key=key)
    with pytest.raises(TypeError):
        WPPMCovarianceField(posterior, 4, key=key)


def test_subsample_requires_a_key(posterior):
    """Picking 10 of 50 draws depends on a key, so we require one."""
    with pytest.raises(ValueError, match="key"):
        WPPMCovarianceField(posterior, n_samples=10)


def test_map_posterior_needs_no_n_samples(model):
    """A MAPPosterior stores its 1 sample, so the count is known."""
    point_mass = MAPPosterior(model, model.init_params(jr.PRNGKey(0)))
    assert WPPMCovarianceField(point_mass).n_samples == 1


def test_map_posterior_gives_sigma_of_theta_map(model, X):
    """For a MAP posterior every draw is θ_MAP, so the mean is Σ(θ_MAP)."""
    params = model.init_params(jr.PRNGKey(3))
    point_mass = MAPPosterior(model, params)
    fields = WPPMCovarianceField(point_mass, n_samples=3, key=jr.PRNGKey(0))
    expected = _sigma(model, params["W"], X)
    assert jnp.allclose(fields.mean(X), expected, rtol=1e-5, atol=1e-9)


def test_rejects_samples_without_parameters(model):
    """A posterior whose sample() returns an empty PyTree gives no field."""

    class Empty:
        def __init__(self, model):
            self.model = model

        def sample(self, n, *, key):
            return {}

    with pytest.raises(ValueError, match="no parameter arrays"):
        WPPMCovarianceField(Empty(model), n_samples=3, key=jr.PRNGKey(0))


def test_rejects_wrong_input_dim(posterior):
    fields = WPPMCovarianceField(posterior)
    with pytest.raises(ValueError, match="input_dim"):
        fields(jnp.ones((4, 3)))
    with pytest.raises(ValueError, match="input_dim"):
        fields.mean(jnp.ones((4, 3)))


def test_exported_from_psyphy_model():
    from psyphy.model import WPPMCovarianceField as Exported

    assert Exported is WPPMCovarianceField


# ==============================================================================
# The two effects that make Σ(E[W]) wrong
# ==============================================================================


def _prior_draws(model, n_chains, n_draws, seed=0):
    """Stack independent prior draws into MCMC layout (n_chains, n_draws, *W)."""
    keys = jr.split(jr.PRNGKey(seed), n_chains * n_draws)
    W = jnp.stack([model.init_params(k)["W"] for k in keys])
    return {"W": W.reshape(n_chains, n_draws, *W.shape[1:])}


def _random_orthogonal(key, m):
    """A random m x m orthogonal matrix (rotation or reflection), via QR."""
    Q, _ = jnp.linalg.qr(jr.normal(key, (m, m)))
    return Q


def _sigma(model, W, X):
    """Σ(X) for a single parameter draw W, with model.local_covariance: (n_points, d, d)."""
    return jax.vmap(lambda x: model.local_covariance({"W": W}, x))(X)


def test_jensen_gap_mean_sigma_dominates_sigma_of_mean_W(model, X):
    """E[Σ] - Σ(E[W]) = Cov(U) is positive semidefinite at every point."""
    samples = _prior_draws(model, n_chains=2, n_draws=10)
    posterior = MCMCPosterior(model, samples)

    mean_sigma = WPPMCovarianceField(posterior).mean(X)
    sigma_of_mean_W = _sigma(model, posterior.mean_params["W"], X)

    gap_eigs = jnp.linalg.eigvalsh(mean_sigma - sigma_of_mean_W)
    assert jnp.all(gap_eigs >= -1e-8)


def test_sign_flipped_draws_do_not_cancel(model, X):
    """W and -W give the same Σ. Averaging W gives 0 -> λI; averaging Σ must not."""
    W = model.init_params(jr.PRNGKey(1))["W"]
    samples = {"W": jnp.stack([W, -W])[None]}  # (1 chain, 2 draws, *W)
    posterior = MCMCPosterior(model, samples)

    # The trap this class replaces: E[W] = 0, so Σ(E[W]) is just λI.
    assert jnp.allclose(posterior.mean_params["W"], 0.0)
    lam_I = model.diag_term * jnp.eye(2)
    assert jnp.allclose(_sigma(model, posterior.mean_params["W"], X), lam_I)

    fields = WPPMCovarianceField(posterior)
    assert jnp.allclose(fields.mean(X), _sigma(model, W, X), rtol=1e-5, atol=1e-9)


def test_invariant_to_embedding_rotations(model, X):
    """Draws W R_k (R_k orthogonal on the embedding axis) all share Σ(W).

    This is the WPPM's non-identifiability: chains that settle at different
    orientations agree on Σ but not on W. The mean must equal Σ(W).
    """
    W = model.init_params(jr.PRNGKey(2))["W"]
    m = W.shape[-1]
    Rs = [_random_orthogonal(k, m) for k in jr.split(jr.PRNGKey(9), 6)]
    # W @ R acts on the last (embedding) axis: U(x) -> U(x) R
    rotated = jnp.stack([W @ R for R in Rs]).reshape(2, 3, *W.shape)
    posterior = MCMCPosterior(model, {"W": rotated})

    fields = WPPMCovarianceField(posterior)
    assert jnp.allclose(fields.mean(X), _sigma(model, W, X), rtol=1e-4, atol=1e-8)


# ==============================================================================
# The square root, per draw
# ==============================================================================


def test_sqrt_cov_is_the_models_square_root_per_draw(posterior, X, model):
    """sqrt_cov(x)[i] is model._compute_sqrt on draw i, shape (N, n_points, d, m)."""
    fields = WPPMCovarianceField(posterior)
    U = fields.sqrt_cov(X)  # (50, n_points=2, d=2, m=3)
    assert U.shape == (50, 2, 2, 3)

    pooled = posterior.samples["W"].reshape(50, *posterior.samples["W"].shape[2:])
    for i in (0, 17, 49):
        for j in range(2):
            ref = model._compute_sqrt({"W": pooled[i]}, X[j])
            assert jnp.allclose(U[i, j], ref, rtol=0.0, atol=ATOL)


def test_sqrt_cov_squares_to_the_field(posterior, X, model):
    """U U^T + diag_term * I equals fields(x)[i] for every draw."""
    fields = WPPMCovarianceField(posterior)
    U = fields.sqrt_cov(X)
    sigma = U @ jnp.swapaxes(U, -1, -2) + model.diag_term * jnp.eye(2)
    assert jnp.allclose(sigma, fields(X), rtol=0.0, atol=ATOL)


def test_sqrt_cov_single_point_and_batch_shapes(posterior, X):
    """Shape (d,) drops the points axis, and a grid keeps its shape."""
    fields = WPPMCovarianceField(posterior, n_samples=5, key=jr.PRNGKey(0))
    assert fields.sqrt_cov(X[0]).shape == (5, 2, 3)
    assert fields.sqrt_cov(jnp.zeros((4, 3, 2))).shape == (5, 4, 3, 2, 3)
    assert jnp.allclose(
        fields.sqrt_cov(X[0]), fields.sqrt_cov(X)[:, 0], rtol=0.0, atol=ATOL
    )


def test_sqrt_cov_rejects_wrong_input_dim(posterior):
    fields = WPPMCovarianceField(posterior)
    with pytest.raises(ValueError, match="input_dim"):
        fields.sqrt_cov(jnp.zeros((3,)))


def test_sqrt_cov_has_no_average(posterior, X):
    """An average of U U^T has no single U behind it, so the mean is not a root."""
    fields = WPPMCovarianceField(posterior)
    U = fields.sqrt_cov(X)
    # averaging the draws of U gives a different matrix than U U^T averaged
    from_mean_U = U.mean(axis=0) @ jnp.swapaxes(U.mean(axis=0), -1, -2)
    gap = jnp.abs(from_mean_U - (U @ jnp.swapaxes(U, -1, -2)).mean(axis=0)).max()
    assert gap > 1e-3


# ==============================================================================
# Read-only attributes (backlog C24, decision 0012)
# ==============================================================================


@pytest.mark.parametrize("name", ["samples", "n_samples", "model"])
def test_samples_attributes_refuse_assignment(posterior, model, name):
    """Assigning to a fixed attribute raises, so a refit loop can't be silently ignored."""
    fields = WPPMCovarianceField(posterior)
    with pytest.raises(AttributeError, match="Build a new one"):
        setattr(fields, name, getattr(fields, name))


def test_samples_attributes_still_read(posterior, model):
    """Reading each attribute returns the values the object was built with."""
    fields = WPPMCovarianceField(posterior)
    assert fields.n_samples == 50
    assert fields.model is model
    assert fields.samples["W"].shape == (50, 2, 2, 2, 3)  # (n_samples, *W_shape)
    assert jnp.array_equal(
        fields.samples["W"], posterior.samples["W"].reshape(50, 2, 2, 2, 3)
    )


def test_samples_item_assignment_changes_nothing(posterior, X):
    """Replacing an item of the returned dict can't reach the stored samples."""
    fields = WPPMCovarianceField(posterior)
    before = fields(X)  # (n_samples, n_points, d, d)
    mean_before = fields.mean(X)  # (n_points, d, d)
    fields.samples["W"] = jnp.zeros_like(fields.samples["W"])
    assert jnp.array_equal(fields(X), before)
    assert jnp.array_equal(fields.mean(X), mean_before)
    assert fields.samples["W"].any()
