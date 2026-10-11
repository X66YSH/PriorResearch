"""Does the model learn stably from the generic prior at our table size? Compare with the original nanoTabPFN size.

Trains on the generic prior only (r = 0), two seeds per table size, and reports
breast cancer AUC (1 real dataset, noisy) and mean AUC on 100 held-out generic prior tables (in-distribution).

Results on a 16 GB Mac (MPS), generic = mix_scm, 600 steps x batch 8, breast cancer AUC:
    150 rows x 5 features:  0.947 / 0.939
    256 rows x 24 features: 0.226 / 0.877   -> unstable when undertrained, not a loader bug

Run from the repo root:
    python -m diagnostics.table_size --steps 600
    python -m diagnostics.table_size --steps 3000 --sizes 256x24 --steps_per_eval 500   (plan (a) in EXPERIMENT.md)
"""

import argparse

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from model import NanoTabPFNClassifier, NanoTabPFNModel
from priors.batching import MixedPriorDataLoader
from priors.mixture import GenericPrior
from train import eval, get_default_device, set_randomness_seed, train

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
parser.add_argument("--sizes", nargs="+", default=["150x5", "256x24"], help="rows x max features")
parser.add_argument("--generic", default="graph_scm")
parser.add_argument("--steps_per_eval", type=int, default=10**9)
args = parser.parse_args()


def held_out_prior_auc(classifier, n_rows, max_features, n_tables=100):
    """Mean AUC on fresh generic prior tables, first half of the rows as context, second half as queries."""
    np.random.seed(999)
    torch.manual_seed(999)
    prior = GenericPrior(n_rows, max_features, 2, args.generic)
    k = n_rows // 2
    aucs = []
    for _ in range(n_tables):
        X, y = prior.sample()
        if len(np.unique(y[:k])) < 2 or len(np.unique(y[k:])) < 2:
            continue
        classifier.fit(X[:k], y[:k])
        aucs.append(roc_auc_score(y[k:], classifier.predict_proba(X[k:])[:, 1]))
    return np.mean(aucs)


device = get_default_device()
for size in args.sizes:
    n_rows, max_features = map(int, size.split("x"))
    for seed in args.seeds:
        set_randomness_seed(seed)
        model = NanoTabPFNModel(embedding_size=96, num_attention_heads=4, mlp_hidden_size=192, num_layers=3, num_outputs=2)
        prior = MixedPriorDataLoader(r=0.0, num_steps=args.steps, batch_size=args.batch_size, n_rows=n_rows,
                                     max_features=max_features, seed=seed, generic_type=args.generic)
        print(f"--- {args.generic} {size} seed {seed}, {args.steps} steps ---", flush=True)
        model, _ = train(model, prior, lr=4e-3, device=device, steps_per_eval=args.steps_per_eval, eval_func=eval)
        classifier = NanoTabPFNClassifier(model, device)
        print(f"RESULT {args.generic} {size} seed {seed}: breast cancer AUC {eval(classifier)['roc_auc']:.3f} | "
              f"held-out prior tables AUC {held_out_prior_auc(classifier, n_rows, max_features):.3f}", flush=True)
