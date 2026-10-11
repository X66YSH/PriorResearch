"""Is the generic-prior model losing on real tables because of their format (heavy tails, skew)?

The generic prior produces columns within about +-4; the finance prior and real tables have heavy-tailed raw values.
This re-evaluates trained checkpoints after a per-column quantile transform to a normal distribution (fitted on the
train rows of each fold, as TabPFN-style preprocessing does), which removes scale and tail differences. If the
r = 0 model gains much more than the r = 1 model, the r = 1 advantage is largely about format, not finance.

Run from the repo root:
    python -m diagnostics.format_check checkpoints/lr1e-3/r0.00_seed0.pth checkpoints/lr1e-3/r1.00_seed0.pth
"""

import argparse
import json
import os

from sklearn.preprocessing import QuantileTransformer

from benchmark.datasets import SUITES, load_suite
from evaluate import evaluate, load_checkpoint
from train import get_default_device


class QuantileThen:
    """Quantile-transforms each column to a normal distribution using the train rows, then calls the classifier."""

    def __init__(self, classifier):
        self.classifier = classifier

    def fit(self, X, y):
        self.transform = QuantileTransformer(n_quantiles=min(100, len(X)), output_distribution="normal").fit(X)
        self.classifier.fit(self.transform.transform(X).astype("float32"), y)
        return self

    def predict_proba(self, X):
        return self.classifier.predict_proba(self.transform.transform(X).astype("float32"))


parser = argparse.ArgumentParser()
parser.add_argument("checkpoints", nargs="+")
parser.add_argument("--out", default="results/format_check.json")
args = parser.parse_args()

suites = {suite: load_suite(suite) for suite in SUITES}
device = get_default_device()
results = {}
print(f"{'model':34}{'preprocessing':>14}" + "".join(f"{s + ' auc':>14}{s + ' ece':>14}" for s in SUITES))
for path in args.checkpoints:
    name = os.path.splitext(os.path.basename(path))[0]
    for preprocessing in ["none", "quantile"]:
        classifier = load_checkpoint(path, device)
        if preprocessing == "quantile":
            classifier = QuantileThen(classifier)
        result = evaluate(classifier, suites)
        results[f"{name}/{preprocessing}"] = result
        print(f"{name:34}{preprocessing:>14}" + "".join(f"{result[s]['mean']['auc']:>14.4f}{result[s]['mean']['ece']:>14.4f}" for s in SUITES))

os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w") as f:
    json.dump(results, f, indent=2)
