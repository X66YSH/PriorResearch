"""Plots the RQ1 tradeoff curves from results/r{r}_seed{seed}.json: one panel per metric, x = finance share r.

Each line is one suite (finance, general): mean over seeds, band = +-1 std over seeds.
Dashed lines are the random forest baseline on the same suite, for reference.

Example: python plot_curves.py --results_dir results --out figures/rq1_curves.png
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

COLORS = {"finance": "#2a78d6", "general": "#eb6834"}  # categorical slots 1 and 2, validated for CVD
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
METRICS = [("auc", "AUC (higher is better)"), ("ece", "ECE (lower is better)"), ("brier", "Brier (lower is better)")]

parser = argparse.ArgumentParser()
parser.add_argument("--results_dir", default="results")
parser.add_argument("--out", default="figures/rq1_curves.png")
args = parser.parse_args()

# scores[suite][metric][r] -> list over seeds
scores = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
for path in glob.glob(os.path.join(args.results_dir, "r*_seed*.json")):
    r = float(re.match(r"r([\d.]+)_seed\d+", os.path.basename(path)).group(1))
    results = json.load(open(path))
    for suite in COLORS:
        for metric, _ in METRICS:
            scores[suite][metric][r].append(results[suite]["mean"][metric])
baseline_path = os.path.join(args.results_dir, "baseline_random_forest.json")
baseline = json.load(open(baseline_path)) if os.path.exists(baseline_path) else None

fig, axes = plt.subplots(1, len(METRICS), figsize=(13, 4), facecolor=SURFACE)
for ax, (metric, title) in zip(axes, METRICS):
    ax.set_facecolor(SURFACE)
    for suite, color in COLORS.items():
        rs = sorted(scores[suite][metric])
        mean = np.array([np.mean(scores[suite][metric][r]) for r in rs])
        std = np.array([np.std(scores[suite][metric][r]) for r in rs])
        ax.fill_between(rs, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)
        ax.plot(rs, mean, color=color, linewidth=2, marker="o", markersize=8, markeredgecolor=SURFACE,
                markeredgewidth=2, label=f"nanoTabPFN on {suite} tables")
        ax.annotate(suite, (rs[-1], mean[-1]), xytext=(8, 0), textcoords="offset points", va="center",
                    color=INK, fontsize=9)
        if baseline is not None:
            ax.axhline(baseline[suite]["mean"][metric], color=color, linewidth=1, linestyle=(0, (4, 3)), alpha=0.8,
                       label=f"random forest on {suite} tables")
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel("finance prior share r", color=MUTED)
    ax.set_xticks(sorted({r for suite in COLORS for r in scores[suite][metric]}))
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(-0.05, 1.15)  # room for the direct labels at r = 1

n_seeds = max(len(v) for suite in COLORS for v in scores[suite]["auc"].values())
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=9, labelcolor=INK)
fig.text(0.01, 0.01, f"mean over {n_seeds} seed(s), band = +-1 std; dashed = random forest", color=MUTED, fontsize=8)
fig.tight_layout(rect=(0, 0.03, 1, 0.92))
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
fig.savefig(args.out, dpi=200, facecolor=SURFACE)
print("saved", args.out)

for suite in COLORS:
    print(f"\n{suite}")
    for r in sorted(scores[suite]["auc"]):
        print(f"  r={r:.2f}  " + "  ".join(f"{m} {np.mean(scores[suite][m][r]):.4f}" for m, _ in METRICS))
