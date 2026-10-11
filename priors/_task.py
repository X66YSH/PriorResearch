"""Minimal stand-in for FinTFM's `fintfm.prior.base.Task`, holding only the fields its financial prior sets."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Task:
    X: np.ndarray                  # (n_rows, n_features) float32, may contain NaN
    y: np.ndarray                  # (n_rows,) int class labels
    n_classes: int
    is_categorical: np.ndarray     # (n_features,) bool
    source: str = "unknown"
    period: np.ndarray | None = None
    n_horizons: int | None = None
