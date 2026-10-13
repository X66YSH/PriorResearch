"""Evaluates runs on the real-table suites, or the reference baselines.

    python scripts/evaluate.py runs/E1_endpoints/r0_seed0_generic-graph_scm      # every checkpoint of a run
    python scripts/evaluate.py runs/E1_endpoints/*  --steps 3000                  # several runs, one step
    python scripts/evaluate.py --baselines                                        # random forest, logistic regression

Baselines are written like a run: runs/baselines/<name>/eval/final/{metrics.json, per_dataset.csv}.
"""

import argparse
import os

import yaml

from tfmprior.config import load_config
from tfmprior.eval.evaluate import baseline_classifiers, evaluate_suites
from tfmprior.run import evaluate_run, load_suites, resolve_device, write_eval

parser = argparse.ArgumentParser()
parser.add_argument("run_dirs", nargs="*")
parser.add_argument("--steps", type=int, nargs="*", help="default: every checkpoint")
parser.add_argument("--baselines", action="store_true")
parser.add_argument("--config", default=None, help="config for the baselines' suites (default: configs/base.yaml)")
args = parser.parse_args()

for run_dir in args.run_dirs:
    with open(os.path.join(run_dir, "config.yaml")) as f:
        config = yaml.safe_load(f)
    print(f"--- {run_dir}", flush=True)
    evaluate_run(run_dir, config, resolve_device(config), steps=args.steps, log=lambda m: print(m, flush=True))

if args.baselines:
    config = load_config(args.config)
    suites = load_suites(config)
    for name, classifier in baseline_classifiers().items():
        rows, summary = evaluate_suites(classifier, suites, config["eval"]["n_folds"])
        out_dir = os.path.join(config["output"]["runs_dir"], "baselines", name, "eval", "final")
        write_eval(out_dir, rows, summary, None, None)
        print(name, " | ".join(f"{s} auc {summary[s]['mean']['auc']:.4f}" for s in summary), flush=True)
