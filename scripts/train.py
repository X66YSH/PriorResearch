"""Trains (and evaluates) the runs of one experiment config, one after another.

    python scripts/train.py --config configs/experiments/E1_endpoints.yaml
    python scripts/train.py --config configs/experiments/E1_endpoints.yaml train.seed=0 train.steps=10   # overrides
    python scripts/train.py --config configs/experiments/E1_endpoints.yaml --list                       # only list runs

An existing run directory is an error unless --resume (skip DONE runs, restart unfinished ones from scratch)
or --overwrite. scripts/sweep.py runs the same runs in parallel on one GPU.
"""

import argparse
import os
import sys

from tfmprior.config import expand_sweep, load_config
from tfmprior.run import run_dir_of, train_run

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, help="experiment file in configs/experiments/")
parser.add_argument("overrides", nargs="*", help="key=value, e.g. train.steps=10")
group = parser.add_mutually_exclusive_group()
group.add_argument("--resume", action="store_true")
group.add_argument("--overwrite", action="store_true")
parser.add_argument("--list", action="store_true", help="print the run directories and exit")
args = parser.parse_args()

runs = expand_sweep(load_config(args.config, args.overrides))
if args.list:
    for run in runs:
        print(run_dir_of(run))
    sys.exit(0)
mode = "resume" if args.resume else "overwrite" if args.overwrite else "new"
for i, run in enumerate(runs, 1):
    print(f"[{i}/{len(runs)}] {run_dir_of(run)}", flush=True)
    train_run(run, mode, log=lambda msg: print(msg, flush=True))
print("all runs finished:", os.path.join(runs[0]["output"]["runs_dir"], runs[0]["experiment"]["name"]))
