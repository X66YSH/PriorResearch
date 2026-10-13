"""Does a model lose on real tables because of their format (heavy tails, skew)? (EXPERIMENT.md 4.6)

Re-evaluates checkpoints after a per-column quantile transform to a normal distribution, fitted on the train rows
of each fold, which removes scale and tail differences between the priors and real tables.

    python scripts/diagnostics/format_check.py runs/E1_endpoints/*/checkpoints/step_3000.pt
"""

import argparse

from sklearn.preprocessing import QuantileTransformer

from tfmprior.config import load_config
from tfmprior.eval.evaluate import evaluate_suites, load_classifier
from tfmprior.run import load_suites
from tfmprior.utils.device import get_default_device


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
args = parser.parse_args()

suites = load_suites(load_config())
device = get_default_device()
print(f"{'checkpoint':60}{'preprocessing':>14}" + "".join(f"{s + ' auc':>14}" for s in suites))
for path in args.checkpoints:
    for preprocessing in ["none", "quantile"]:
        classifier = load_classifier(path, device)
        if preprocessing == "quantile":
            classifier = QuantileThen(classifier)
        _, summary = evaluate_suites(classifier, suites)
        print(f"{path[-60:]:60}{preprocessing:>14}" + "".join(f"{summary[s]['mean']['auc']:>14.4f}" for s in suites))
