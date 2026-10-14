"""Evaluation on the real-table suites: stratified 5-fold CV per subsample, NaN imputed per fold with train-row means,
out-of-fold probabilities pooled over the folds, then AUC / ECE / Brier per subsample (repeat).
A dataset's score is the mean over its repeats; a suite's score is the mean over its datasets.
"""

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tfmprior.data.preprocess import impute_with_train_mean
from tfmprior.eval.metrics import binary_metrics
from tfmprior.model.nanotabpfn import NanoTabPFNClassifier, NanoTabPFNModel

METRICS = ["auc", "ece", "brier"]


def build_model(model_config: dict) -> NanoTabPFNModel:
    return NanoTabPFNModel(
        embedding_size=model_config["embedding_size"],
        num_attention_heads=model_config["num_attention_heads"],
        mlp_hidden_size=model_config["mlp_hidden_size"],
        num_layers=model_config["num_layers"],
        num_outputs=model_config["num_outputs"],
    )


def load_classifier(path: str, device: str) -> NanoTabPFNClassifier:
    """Loads a notebook checkpoint or a checkpoint from an earlier experiment."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "config" in checkpoint:
        model_config = checkpoint["config"]["model"]
    else:  # pre-refactor: {"model", "args"}; 4 heads and mlp = 2 x embedding were fixed then
        args = checkpoint["args"]
        model_config = dict(embedding_size=args["embedding_size"], num_attention_heads=4,
                            mlp_hidden_size=2 * args["embedding_size"], num_layers=args["num_layers"], num_outputs=2)
    model = build_model(model_config)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return NanoTabPFNClassifier(model, device)


def baseline_classifiers() -> dict:
    return {
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1),
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    }


def predict_out_of_fold(classifier, X: np.ndarray, y: np.ndarray, n_folds: int = 5) -> np.ndarray:
    p = np.zeros(len(y))
    for train_idx, test_idx in StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0).split(X, y):
        X_fold = impute_with_train_mean(np.concatenate([X[train_idx], X[test_idx]]), len(train_idx))
        classifier.fit(X_fold[:len(train_idx)], y[train_idx])
        p[test_idx] = classifier.predict_proba(X_fold[len(train_idx):])[:, 1]
    return p


def evaluate_suites(classifier, suites: dict, n_folds: int = 5) -> tuple[list[dict], dict]:
    """Returns (rows, summary). rows: one dict per suite x dataset x repeat with every metric.
    summary: {suite: {"mean": {metric}, "datasets": {name: {metric}}}}, the same numbers as before the refactor."""
    rows, summary = [], {}
    for suite, datasets in suites.items():
        per_dataset = {}
        for name, repeats in datasets.items():
            scores = []
            for repeat, (X, y) in enumerate(repeats):
                score = binary_metrics(y, predict_out_of_fold(classifier, X, y, n_folds))
                scores.append(score)
                rows.append(dict(suite=suite, dataset=name, repeat=repeat, **score))
            per_dataset[name] = {k: float(np.mean([s[k] for s in scores])) for k in METRICS}
        mean = {k: float(np.mean([d[k] for d in per_dataset.values()])) for k in METRICS}
        summary[suite] = {"mean": mean, "datasets": per_dataset}
    return rows, summary
