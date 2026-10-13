"""Compares what the priors' tables look like: feature count, positive rate, missing values, value scale.

Reads the first --n_tables tables of each pool (no generation). Used to check that format_aug matches the finance
prior's format (Phase B) and to describe each prior in docs/EXPERIMENTS.md.

    python scripts/diagnostics/prior_stats.py artifacts/dumps/generic_graph_scm_100k_256x24_seed0.h5 artifacts/dumps/finance_100k_256x24_seed0.h5
"""

import argparse
import os

import h5py
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("pools", nargs="+")
parser.add_argument("--n_tables", type=int, default=2000)
args = parser.parse_args()


def table_stats(path: str, n: int) -> dict:
    with h5py.File(path, "r") as f:
        X, y, nf = f["X"][:n], f["y"][:n], f["num_features"][:n]
    stats = {"n_features": [], "positive_rate": [], "nan_fraction": [], "log10_abs_max": [], "abs_skew": []}
    for X_i, y_i, k in zip(X, y, nf):
        X_i = X_i[:, :k]
        stats["n_features"].append(k)
        stats["positive_rate"].append((y_i == 1).mean())
        stats["nan_fraction"].append(np.isnan(X_i).mean())
        stats["log10_abs_max"].append(np.log10(np.nanmax(np.abs(X_i)) + 1e-12))
        columns = [c[~np.isnan(c)] for c in X_i.T]
        skews = [abs(((c - c.mean()) ** 3).mean() / (c.std() ** 3 + 1e-12)) for c in columns if len(c) > 2 and c.std() > 0]
        stats["abs_skew"].append(np.median(skews) if skews else 0.0)
    return {k: np.asarray(v, dtype=float) for k, v in stats.items()}


rows = {os.path.basename(p): table_stats(p, args.n_tables) for p in args.pools}
print(f"first {args.n_tables} tables per pool; median [10th, 90th percentile] over tables\n")
print(f"{'':16}" + "".join(f"{name[:34]:>36}" for name in rows))
for key in ["n_features", "positive_rate", "nan_fraction", "log10_abs_max", "abs_skew"]:
    line = f"{key:16}"
    for stats in rows.values():
        v = stats[key]
        line += f"{f'{np.median(v):.3g} [{np.percentile(v, 10):.3g}, {np.percentile(v, 90):.3g}]':>36}"
    print(line)
print(f"{'positive < 5%':16}" + "".join(f"{(s['positive_rate'] < 0.05).mean():>36.2f}" for s in rows.values()))
