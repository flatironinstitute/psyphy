from __future__ import annotations

from dataclasses import dataclass

"""
Noise models for observer's internal representations.

Implements:
- GaussianNoise: default (ellipsoidal isoperformance contours)
- StudentTNoise: heavier-tailed observer variability
- Extensible for other families (e.g., skewed or anisotropic)

NoiseModel objects plug into WPPM to define trial likelihoods.
"""


@dataclass
class GaussianNoise:
    sigma: float = 1.0

    def log_prob(self, residual: float) -> float:
        _ = residual
        return -0.5


@dataclass
class StudentTNoise:
    df: float = 3.0
    scale: float = 1.0

    def log_prob(self, residual: float) -> float:
        _ = residual
        return -0.5
