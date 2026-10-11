"""How learnable are the tables each prior produces? A random forest is the yardstick.

For each prior, draw tables, fit a random forest on the first half of the rows and report its AUC on the second half.
A low score means many tables carry little signal; that is normal for priors (the original h5 dump also scores ~0.6).

Run from the repo root: python -m diagnostics.prior_learnability
"""

import random

import h5py
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from tabicl.prior import PriorDataset

from priors.batching import impute_with_train_mean
from priors.fintfm_financial import sample_financial_task

N_TABLES = 40


def rf_auc(X, y):
    k = len(y) // 2
    if len(np.unique(y[:k])) < 2 or len(np.unique(y[k:])) < 2:
        return None
    rf = RandomForestClassifier(100, random_state=0, n_jobs=-1).fit(X[:k], y[:k])
    return roc_auc_score(y[k:], rf.predict_proba(X[k:])[:, 1])


def report(name, sample_table):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    aucs = []
    for _ in range(N_TABLES):
        X, y = sample_table()
        auc = rf_auc(impute_with_train_mean(X, len(y) // 2), y)
        if auc is not None:
            aucs.append(auc)
    aucs = np.array(aucs)
    print(f"{name:12} RF AUC mean {aucs.mean():.3f} | median {np.median(aucs):.3f} | share of tables > 0.7: {(aucs > 0.7).mean():.2f}")


for prior_type in ["mlp_scm", "tree_scm", "mix_scm", "graph_scm"]:
    dataset = PriorDataset(batch_size=1, batch_size_per_gp=1, min_features=2, max_features=24, max_classes=2,
                           min_seq_len=256, max_seq_len=257, prior_type=prior_type, n_jobs=1)

    def sample_tabicl(dataset=dataset):
        X, y, d, _, _ = dataset.get_batch(1)
        return X[0, :, :int(d[0])].numpy(), y[0].numpy()

    report(prior_type, sample_tabicl)

rng = np.random.default_rng(0)


def sample_finance():
    task = sample_financial_task(rng, 256, max_features=24)
    return task.X, task.y


report("finance", sample_finance)

h5 = h5py.File("300k_150x5_2.h5", "r")
h5_index = iter(range(N_TABLES))


def sample_h5():
    i = next(h5_index)
    n_rows, n_features = int(h5["num_datapoints"][i]), int(h5["num_features"][i])
    return h5["X"][i, :n_rows, :n_features], h5["y"][i, :n_rows]


report("h5 dump", sample_h5)
