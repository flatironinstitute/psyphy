"""
transforms.py
-------------

color space transformations.

functions (MVP stubs):
- to_model_space(rgb): map RGB stimulus values into model space
- to_rgb(model_coords): map from model space back to RGB

Future extensions:
- other color spaces
"""

from typing import Sequence, Union

import numpy as np

ArrayLike = Union[Sequence[float], np.ndarray]


def stimuli_to_model_space(rgb: ArrayLike) -> np.ndarray:
    """
    Map RGB stimulus values to model space coordinates.

    Parameters
    ----------
    rgb : array-like
        RGB values, shape (3,) or similar.

    Returns
    -------
    np.ndarray
        Model-space coordinates (MVP: identical to input).
    """
    return np.array(rgb)


def model_to_stimuli(model_coords: ArrayLike) -> np.ndarray:
    """
    Map from model space coordinates to RGB values.

    Parameters
    ----------
    model_coords : array-like
        Model space coordinates.

    Returns
    -------
    np.ndarray
        RGB values (MVP: identical to input).
    """
    return np.array(model_coords)
