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

from .dataset import ResponseData

PathLike = Union[str, Path]


def save_responses_csv(data: ResponseData, path: PathLike) -> None:
    """
    Save ResponseData to a CSV file.

    Parameters
    ----------
    data : ResponseData
    path : str or Path
    """
    refs, probes, resps = data.to_numpy()
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ref", "probe", "response"])
        for r, p, y in zip(refs, probes, resps):
            writer.writerow([r.tolist(), p.tolist(), int(y)])


def load_responses_csv(path: PathLike) -> ResponseData:
    """
    Load ResponseData from a CSV file.

    Parameters
    ----------
    path : str or Path

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
            data.add_trial(ref, probe, resp)
    return data


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
