import jax
import jax.numpy as jnp
import numpy as np
import pytest

from psyphy.model import WPPM, GaussianNoise, OddityTask, Prior


@pytest.mark.parametrize("diag_term", [1e-6, 1e-4, 1e-3, 1e-2])
def test_local_covariance_positive_definite(diag_term):
    input_dim = 2
    basis_degree = 2
    extra_dims = 1
    prior = Prior(
        input_dim=input_dim,
        basis_degree=basis_degree,
        extra_embedding_dims=extra_dims,
        variance_scale=0.2,
        decay_rate=0.5,
    )
    task = OddityTask()
    noise = GaussianNoise(sigma=1.0)
    model = WPPM(
        input_dim=input_dim,
        prior=prior,
        task=task,
        noise=noise,
        diag_term=diag_term,
    )
    key = jax.random.PRNGKey(0)
    for _ in range(5):
        params = model.init_params(key)
        # Test on a grid of points
        grid = np.linspace(-0.3, 0.3, 5)
        for x in np.stack(np.meshgrid(grid, grid), axis=-1).reshape(-1, 2):
            cov = np.array(model.local_covariance(params, jnp.array(x)))
            eigvals = np.linalg.eigvalsh(cov)
            assert np.all(eigvals > 0), (
                f"Covariance not PD at x={x}, eigvals={eigvals}, diag_term={diag_term}"
            )
