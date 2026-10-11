"""Does a smaller learning rate stop training from collapsing? (EXPERIMENT.md 4.4)

With 256 x 24 tables, batch 8 and lr 4e-3, 2 of 4 runs collapsed or inverted. lr 4e-3 was tuned for batch 32;
scaling it with the batch gives 1e-3. Trains on the pools (r = 0 by default) for each lr and seed, then labels
each run with the criteria of EXPERIMENT.md 4.4:
    healthy:   held-out prior AUC >= 0.65 and breast cancer AUC >= 0.9
    collapsed: held-out prior AUC < 0.55 or breast cancer AUC < 0.6
    unclear:   anything in between
Held-out tables are generated fresh (seed 999), so none of them is in the pools.

Run from the repo root:
    python -m diagnostics.lr_stability --lrs 4e-3 1e-3 --seeds 0 1 2 --steps 1000
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
from sklearn.metrics import log_loss, roc_auc_score

from model import NanoTabPFNClassifier, NanoTabPFNModel
from priors.batching import DumpMixedDataLoader
from priors.mixture import GenericPrior
from train import eval, get_default_device, set_randomness_seed, train

parser = argparse.ArgumentParser()
parser.add_argument("--lrs", type=float, nargs="+", default=[4e-3, 1e-3])
parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--r", type=float, default=0.0)
parser.add_argument("--generic_dump", default="prior_dumps/generic_graph_scm_100k_256x24_seed0.h5")
parser.add_argument("--finance_dump", default="prior_dumps/finance_100k_256x24_seed0.h5")
parser.add_argument("--out", default="results/lr_stability/summary.json")
args = parser.parse_args()


def held_out_tables(n_tables=100, n_rows=256, max_features=24):
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    prior = GenericPrior(n_rows, max_features, 2, "graph_scm")
    tables, k = [], n_rows // 2
    while len(tables) < n_tables:
        X, y = prior.sample()
        if len(np.unique(y[:k])) == 2 and len(np.unique(y[k:])) == 2:
            tables.append((X, y))
    return tables


def held_out_scores(classifier, tables):
    """Mean AUC and log loss on the held-out tables; first half of the rows as context."""
    aucs, losses = [], []
    for X, y in tables:
        k = len(y) // 2
        classifier.fit(X[:k], y[:k])
        p = classifier.predict_proba(X[k:])[:, 1]
        aucs.append(roc_auc_score(y[k:], p))
        losses.append(log_loss(y[k:], p, labels=[0, 1]))
    return float(np.mean(aucs)), float(np.mean(losses))


def label(prior_auc, bc_auc):
    if prior_auc >= 0.65 and bc_auc >= 0.9:
        return "healthy"
    if prior_auc < 0.55 or bc_auc < 0.6:
        return "collapsed"
    return "unclear"


device = get_default_device()
tables = held_out_tables()
runs = []
for lr in args.lrs:
    for seed in args.seeds:
        print(f"--- lr {lr:g} seed {seed}, r = {args.r}, {args.steps} steps ---", flush=True)
        start = time.time()
        set_randomness_seed(seed)
        model = NanoTabPFNModel(embedding_size=96, num_attention_heads=4, mlp_hidden_size=192, num_layers=3, num_outputs=2)
        prior = DumpMixedDataLoader(args.r, args.steps, args.batch_size, args.generic_dump, args.finance_dump, seed=seed)
        model, history = train(model, prior, lr=lr, device=device, steps_per_eval=250, eval_func=eval)
        classifier = NanoTabPFNClassifier(model, device)
        bc_auc = eval(classifier)["roc_auc"]
        prior_auc, prior_loss = held_out_scores(classifier, tables)
        run = {"lr": lr, "seed": seed, "breast_cancer_auc": bc_auc, "held_out_prior_auc": prior_auc,
               "held_out_prior_log_loss": prior_loss, "label": label(prior_auc, bc_auc),
               "breast_cancer_auc_during_training": [h[1]["roc_auc"] for h in history], "minutes": (time.time() - start) / 60}
        runs.append(run)
        print(f"RESULT lr {lr:g} seed {seed}: {run['label']} | held-out prior AUC {prior_auc:.3f}, log loss {prior_loss:.3f} "
              f"| breast cancer AUC {bc_auc:.3f} | {run['minutes']:.1f} min", flush=True)

os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w") as f:
    json.dump({"args": vars(args), "runs": runs}, f, indent=2)

print(f"\n{'lr':>8} {'seed':>5} {'label':>10} {'prior AUC':>10} {'log loss':>9} {'BC AUC':>7}   BC AUC every 250 steps")
for run in runs:
    trace = " ".join(f"{a:.2f}" for a in run["breast_cancer_auc_during_training"])
    print(f"{run['lr']:>8g} {run['seed']:>5} {run['label']:>10} {run['held_out_prior_auc']:>10.3f} "
          f"{run['held_out_prior_log_loss']:>9.3f} {run['breast_cancer_auc']:>7.3f}   {trace}")
print("saved", args.out)
