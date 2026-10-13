"""How learnable are a prior's tables? A random forest is the yardstick (EXPERIMENT.md 4.1).

For the first --n_tables tables of each pool: fit a random forest on the first half of the rows, report AUC on the
second half. Low scores are normal for priors (the original nanoTabPFN prior dump scored about 0.6).

    python scripts/diagnostics/prior_learnability.py artifacts/dumps/*_1k_*.h5
"""

import argparse
import os

import h5py
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from tfmprior.data.preprocess import impute_with_train_mean

parser = argparse.ArgumentParser()
parser.add_argument("pools", nargs="+")
parser.add_argument("--n_tables", type=int, default=40)
args = parser.parse_args()

for path in args.pools:
    aucs = []
    with h5py.File(path, "r") as f:
        for i in range(args.n_tables):
            k = int(f["num_features"][i])
            X, y = f["X"][i, :, :k], f["y"][i].astype(np.int64)
            half = len(y) // 2
            if len(np.unique(y[:half])) < 2 or len(np.unique(y[half:])) < 2:
                continue
            X = impute_with_train_mean(X, half)
            rf = RandomForestClassifier(100, random_state=0, n_jobs=-1).fit(X[:half], y[:half])
            aucs.append(roc_auc_score(y[half:], rf.predict_proba(X[half:])[:, 1]))
    aucs = np.array(aucs)
    print(f"{os.path.basename(path):45} RF AUC mean {aucs.mean():.3f} | median {np.median(aucs):.3f} | "
          f"share of tables > 0.7: {(aucs > 0.7).mean():.2f} | {len(aucs)} usable of {args.n_tables}")
