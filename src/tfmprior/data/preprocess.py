"""Preprocessing shared by training and evaluation."""

import numpy as np


def impute_with_train_mean(X: np.ndarray, n_train: int) -> np.ndarray:
    """Fills NaN with the column mean of the first n_train rows (0 if that column has no value there).

    The model normalizes each column by its mean and std, so a single NaN would turn the whole column into NaN.
    Only the train rows are used so that test rows never leak into the fill value.
    Evaluation on real tables must call this same function.
    """
    X = X.copy()
    with np.errstate(all="ignore"):
        col_mean = np.nanmean(X[:n_train], axis=0)
    col_mean = np.nan_to_num(col_mean, nan=0.0)
    rows, cols = np.where(np.isnan(X))
    X[rows, cols] = col_mean[cols]
    return X
