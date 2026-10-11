"""Evaluates checkpoints (and optional sklearn baselines) on the finance and general suites.

For each dataset and repeat: stratified 5-fold CV, NaN imputed per fold with train-row means,
out-of-fold probabilities pooled over the 5 folds, then AUC / ECE / Brier. Repeats are averaged per
dataset, datasets are averaged per suite. Results go to results/<name>.json.

Examples:
    python evaluate.py checkpoints/r0.00_seed0.pth
    python evaluate.py checkpoints/*.pth --baselines
"""

import argparse
import json
import os

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from benchmark.datasets import SUITES, load_suite
from benchmark.metrics import binary_metrics
from model import NanoTabPFNClassifier, NanoTabPFNModel
from priors.batching import impute_with_train_mean
from train import get_default_device


def load_checkpoint(path: str, device: str) -> NanoTabPFNClassifier:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    args = checkpoint["args"]
    model = NanoTabPFNModel(
        embedding_size=args["embedding_size"],
        num_attention_heads=4,
        mlp_hidden_size=2 * args["embedding_size"],
        num_layers=args["num_layers"],
        num_outputs=2,
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return NanoTabPFNClassifier(model, device)


def predict_out_of_fold(classifier, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.zeros(len(y))
    for train_idx, test_idx in StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(X, y):
        X_fold = impute_with_train_mean(np.concatenate([X[train_idx], X[test_idx]]), len(train_idx))
        classifier.fit(X_fold[:len(train_idx)], y[train_idx])
        p[test_idx] = classifier.predict_proba(X_fold[len(train_idx):])[:, 1]
    return p


def evaluate(classifier, suites: dict) -> dict:
    results = {}
    for suite, datasets in suites.items():
        per_dataset = {}
        for name, repeats in datasets.items():
            scores = [binary_metrics(y, predict_out_of_fold(classifier, X, y)) for X, y in repeats]
            per_dataset[name] = {k: float(np.mean([s[k] for s in scores])) for k in scores[0]}
        mean = {k: float(np.mean([d[k] for d in per_dataset.values()])) for k in ["auc", "ece", "brier"]}
        results[suite] = {"mean": mean, "datasets": per_dataset}
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="*")
    parser.add_argument("--baselines", action="store_true", help="also evaluate random forest and logistic regression")
    parser.add_argument("--out_dir", default="results")
    args = parser.parse_args()

    suites = {suite: load_suite(suite) for suite in SUITES}
    for suite, datasets in suites.items():
        print(f"{suite}: {len(datasets)} datasets: {', '.join(datasets)}")

    device = get_default_device()
    models = {os.path.splitext(os.path.basename(p))[0]: (lambda p=p: load_checkpoint(p, device)) for p in args.checkpoints}
    if args.baselines:
        models["baseline_random_forest"] = lambda: RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
        models["baseline_logistic_regression"] = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"\n{'model':32}" + "".join(f"{s + ' ' + m:>15}" for s in SUITES for m in ["auc", "ece", "brier"]))
    for name, make_classifier in models.items():
        results = evaluate(make_classifier(), suites)
        with open(os.path.join(args.out_dir, f"{name}.json"), "w") as f:
            json.dump(results, f, indent=2)
        print(f"{name:32}" + "".join(f"{results[s]['mean'][m]:>15.4f}" for s in SUITES for m in ["auc", "ece", "brier"]))
