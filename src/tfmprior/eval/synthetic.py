"""Evaluation on held-out synthetic tables: validation pools generated with a seed the training pools never use.

Each table is split in half (first half context, second half queries); only tables with both classes in each half
are used. Reports the mean AUC and the mean log loss (cross-entropy) over the first n_tables such tables. The log
loss is what a data mixing law is fitted to (RQ2)."""

import h5py
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

from tfmprior.data.preprocess import impute_with_train_mean


def load_validation_tables(path: str, n_tables: int) -> list:
    tables = []
    with h5py.File(path, "r") as f:
        for i in range(f["y"].shape[0]):
            X, y = f["X"][i, :, :f["num_features"][i]], f["y"][i].astype(np.int64)
            k = len(y) // 2
            if len(np.unique(y[:k])) == 2 and len(np.unique(y[k:])) == 2:
                tables.append((X, y))
            if len(tables) == n_tables:
                break
    if len(tables) < n_tables:
        raise ValueError(f"{path} has only {len(tables)} usable tables, {n_tables} requested")
    return tables


def validation_scores(classifier, tables: list) -> dict:
    aucs, losses = [], []
    for X, y in tables:
        k = len(y) // 2
        X = impute_with_train_mean(X, k)
        classifier.fit(X[:k], y[:k])
        p = classifier.predict_proba(X[k:])[:, 1]
        aucs.append(roc_auc_score(y[k:], p))
        losses.append(log_loss(y[k:], p, labels=[0, 1]))
    return {"auc": float(np.mean(aucs)), "log_loss": float(np.mean(losses)), "n_tables": len(tables)}
