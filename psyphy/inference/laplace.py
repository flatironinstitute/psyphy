"""
laplace.py
----------

Laplace approximation to posterior.


Approximates posterior with a Gaussian:
    N(mean = MAP, covariance = H^-1 at MAP)

Provides posterior.sample() cheaply.
Useful for InfoGainPlacement when only MAP fit is available.


MVP implementation:
- Stub that just returns the MAP posterior.
- Future: compute covariance from Hessian at MAP params.
"""

from psyphy.posterior.posterior import Posterior


class LaplaceApproximation:
    """
    Laplace approximation around MAP estimate.

    Methods
    -------
    from_map(map_posterior) -> Posterior
        Construct a Gaussian approximation centered at MAP.
    """

    def from_map(self, map_posterior: Posterior) -> Posterior:
        """
        Return posterior approximation from MAP.

        Parameters
        ----------
        map_posterior : Posterior
            Posterior object from MAP optimization.

        Returns
        -------
        Posterior
            Same posterior object (MVP).
        """
        return map_posterior
