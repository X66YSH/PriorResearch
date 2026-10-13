"""Binary metrics: ranking (AUC) and calibration (ECE, Brier). p is the predicted probability of class 1."""

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Equal-width bins on p; weighted mean of |mean predicted p - observed rate of class 1| per bin."""
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        in_bin = bins == b
        if in_bin.any():
            ece += in_bin.mean() * abs(p[in_bin].mean() - y[in_bin].mean())
    return float(ece)


def binary_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "auc": float(roc_auc_score(y, p)),
        "ece": expected_calibration_error(y, p),
        "brier": float(brier_score_loss(y, p)),
    }
