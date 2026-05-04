"""
io.py
-----

I/O utilities for saving and loading psyphy data.

Supports:
- CSV for human-readable trial logs
- Pickle (.pkl) for Posterior and ResponseData checkpoints?

Notes
-----
- Data is stored in NumPy arrays (via ResponseData.to_numpy()).
- Convert to jax.numpy when passing into models.
"""

from __future__ import annotations

import ast
import csv
import pickle
from pathlib import Path
from typing import Union

import numpy as np

from .dataset import ResponseData, TrialData

PathLike = Union[str, Path]


def save_responses_csv(data: TrialData | ResponseData, path: PathLike) -> None:
    """
    Save ResponseData to a CSV file.

    Parameters
    ----------
    data : ResponseData
    path : str or Path
    """
    if isinstance(data, TrialData):
        inputs, resps = (
            np.asarray(data.inputs),
            np.asarray(data.responses),
        )
    else:
        inputs, resps = data.to_numpy()
    row_names = []
    for s in range(inputs.shape[1]):
        row_names.append("stimulus " + str(s))
    row_names.append("response")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row_names)
        for x, y in zip(inputs, resps):
            row = x.tolist()
            row.append(y.tolist())
            writer.writerow(row)


def load_responses_csv(path: PathLike) -> TrialData:
    """
    Load ResponseData from a CSV file.
    Currently catering to OddityTask data format.

    Parameters
    ----------
    path : str or Path
    - must be of expected format for OddityTask

    Returns
    -------
    ResponseData
    """
    data = ResponseData()
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ref = ast.literal_eval(row["ref"])
            probe = ast.literal_eval(row["probe"])
            resp = int(row["response"])
            data.add_trial((ref, probe), resp)
    return data.to_trial_data()


def save_posterior(posterior: object, path: PathLike) -> None:
    """
    Save a Posterior object to disk using pickle.
    """
    with open(path, "wb") as f:
        pickle.dump(posterior, f)


def load_posterior(path: PathLike) -> object:
    """
    Load a Posterior object from pickle.
    """
    with open(path, "rb") as f:
        return pickle.load(f)
