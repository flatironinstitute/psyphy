"""
laplace.py
----------

Laplace approximation to posterior.


Approximates posterior with a Gaussian:
    N(mean = MAP, covariance = H^-1 at MAP)

Provides posterior.sample(n, key=...) cheaply.
Useful for InfoGainPlacement when only MAP fit is available.


Currently a stub: `from_map` returns the MAP posterior and computes no Hessian.
"""

# import jax.scipy.linalg as jl
# from optax import GradientTransformation

from psyphy.inference.base import InferenceEngine
from psyphy.posterior.posterior import MAPPosterior


class LaplaceApproximation(InferenceEngine):
    """
    Laplace approximation around MAP estimate.

    Methods
    -------
    from_map(map_posterior) -> Posterior
        Construct a Gaussian approximation centered at MAP.
    """

    def from_map(self, map_posterior: MAPPosterior) -> MAPPosterior:
        """
        Return posterior approximation from MAP.

        Parameters
        ----------
        map_posterior : Posterior
            Posterior object from MAP optimization.

        Returns
        -------
        MAPPosterior
            Posterior distribution containing Laplace approximation.
        """
        # 1. First find MAP estimate
        # 2. Compute Hessian at MAP
        # 3. Invert Hessian to get covariance
        # 4. Return Gaussian posterior with MAP mean and covariance

        return map_posterior
