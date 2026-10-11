"""Sanity check for priors/mixture.py: draw tables at one r and compare what the two priors produce."""

import argparse
import time
from collections import defaultdict

import numpy as np

from priors.mixture import MixedPrior

parser = argparse.ArgumentParser()
parser.add_argument("--r", type=float, default=0.3)
parser.add_argument("--n_tables", type=int, default=500)
args = parser.parse_args()

prior = MixedPrior(r=args.r)
stats = defaultdict(lambda: defaultdict(list))
for _ in range(args.n_tables):
    start = time.time()
    X, y, source = prior.sample()
    s = stats[source]
    s["seconds"].append(time.time() - start)
    s["n_rows"].append(X.shape[0])
    s["n_features"].append(X.shape[1])
    s["nan_frac"].append(np.isnan(X).mean())
    s["n_classes"].append(len(np.unique(y)))
    s["pos_rate"].append((y == 1).mean())
    s["abs_max"].append(np.nanmax(np.abs(X)))

n_finance = len(stats["finance"]["n_rows"])
print(f"r = {args.r}: {n_finance}/{args.n_tables} tables from finance = {n_finance / args.n_tables:.3f}\n")
print(f"{'':12}{'generic':>30}{'finance':>30}")
for key in ["n_rows", "n_features", "n_classes", "pos_rate", "nan_frac", "abs_max", "seconds"]:
    row = f"{key:12}"
    for source in ["generic", "finance"]:
        v = np.array(stats[source][key])
        row += f"{f'{np.median(v):.3g} [{v.min():.3g}, {v.max():.3g}]':>30}"
    print(row)
print("\ncells: median [min, max] over tables")
for source in ["generic", "finance"]:
    pos = np.array(stats[source]["pos_rate"])
    print(f"{source}: share of tables with positive rate < 5%: {(pos < 0.05).mean():.2f}")
