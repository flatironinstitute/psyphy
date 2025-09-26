"""
psyphy.data
==========

submodule for handling psychophysical experiment data.

Includes:
- dataset: ResponseData, TrialBatch, loaders
- transforms: color/model space conversions
- io: save/load datasets
"""

from .dataset import ResponseData, TrialBatch

__all__ = ["ResponseData", "TrialBatch"]
