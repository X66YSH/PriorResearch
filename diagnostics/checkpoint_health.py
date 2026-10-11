"""Health check for trained checkpoints: did each run learn its priors, independent of the real-table suites?

Reports breast cancer AUC and mean AUC on 100 fresh tables from each prior (seed 999, never in the pools),
plus the EXPERIMENT.md 4.4 label on the generic tables. A run that is weak on both priors is a weak run,
whatever its r; this separates training noise from the effect of r.

Run from the repo root: python -m diagnostics.checkpoint_health checkpoints/lr1e-3/*.pth
"""

import argparse
import json
import os
import random

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from evaluate import load_checkpoint
from priors.fintfm_financial import sample_financial_task
from priors.mixture import GenericPrior
from train import eval, get_default_device

N_TABLES, N_ROWS, MAX_FEATURES = 100, 256, 24


def fresh_tables(prior: str):
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    rng = np.random.default_rng(999)
    generic = GenericPrior(N_ROWS, MAX_FEATURES, 2, "graph_scm")
    k, tables = N_ROWS // 2, []
    while len(tables) < N_TABLES:
        if prior == "generic":
            X, y = generic.sample()
        else:
            task = sample_financial_task(rng, N_ROWS, max_features=MAX_FEATURES)
            X, y = task.X, task.y
        if len(np.unique(y[:k])) == 2 and len(np.unique(y[k:])) == 2:
            tables.append((X, y))
    return tables


def mean_auc(classifier, tables):
    from priors.batching import impute_with_train_mean
    aucs = []
    for X, y in tables:
        k = len(y) // 2
        X = impute_with_train_mean(X, k)
        classifier.fit(X[:k], y[:k])
        aucs.append(roc_auc_score(y[k:], classifier.predict_proba(X[k:])[:, 1]))
    return float(np.mean(aucs))


parser = argparse.ArgumentParser()
parser.add_argument("checkpoints", nargs="+")
parser.add_argument("--out", default=None, help="json path; default: results/<checkpoint dir>/health.json")
args = parser.parse_args()

device = get_default_device()
tables = {prior: fresh_tables(prior) for prior in ["generic", "finance"]}
rows = {}
print(f"{'checkpoint':16}{'breast cancer':>15}{'generic prior':>15}{'finance prior':>15}{'label':>11}")
for path in sorted(args.checkpoints):
    name = os.path.splitext(os.path.basename(path))[0]
    classifier = load_checkpoint(path, device)
    bc = eval(classifier)["roc_auc"]
    generic, finance = mean_auc(classifier, tables["generic"]), mean_auc(classifier, tables["finance"])
    label = "healthy" if generic >= 0.65 and bc >= 0.9 else "collapsed" if generic < 0.55 or bc < 0.6 else "unclear"
    rows[name] = {"breast_cancer_auc": bc, "generic_prior_auc": generic, "finance_prior_auc": finance, "label": label}
    print(f"{name:16}{bc:>15.3f}{generic:>15.3f}{finance:>15.3f}{label:>11}")

out = args.out or os.path.join("results", os.path.basename(os.path.dirname(os.path.abspath(args.checkpoints[0]))), "health.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    json.dump(rows, f, indent=2)
print("saved", out)
