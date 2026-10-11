"""Runs the RQ1 grid: one training run per (r, seed), then evaluates every checkpoint on both suites.

Finished runs are skipped, so the grid can be stopped and restarted; a checkpoint trained with other
settings (steps, batch size, generic prior) stops the grid instead of being reused. Each run logs to logs/.

Checkpoints go to --out_dir, results to results/<basename of --out_dir>/.

Example (pilot): python run_grid.py --steps 3000 --batch_size 8 --lr 1e-3 --out_dir checkpoints/lr1e-3
"""

import argparse
import os
import subprocess
import sys

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--rs", type=float, nargs="+", default=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
parser.add_argument("--steps", type=int, required=True)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--lr", type=float, default=4e-3)
parser.add_argument("--out_dir", default="checkpoints")
parser.add_argument("--generic", default="graph_scm")
args = parser.parse_args()

os.makedirs("logs", exist_ok=True)
checkpoints = []
for seed in args.seeds:  # seed in the outer loop: a partial grid still covers every r
    for r in args.rs:
        name = f"r{r:.2f}_seed{seed}"
        path = os.path.join(args.out_dir, f"{name}.pth")
        checkpoints.append(path)
        if os.path.exists(path):
            saved = torch.load(path, map_location="cpu", weights_only=False)["args"]
            wanted = {"steps": args.steps, "batch_size": args.batch_size, "generic": args.generic, "lr": args.lr,
                      "on_the_fly": False}
            if any(saved.get(k) != v for k, v in wanted.items()):
                sys.exit(f"{path} was trained with different settings ({ {k: saved.get(k) for k in wanted} }), "
                         f"move it away before rerunning")
            print(f"skip {name}: checkpoint exists with the same settings")
            continue
        print(f"train {name} ...", flush=True)
        with open(os.path.join("logs", f"train_{os.path.basename(os.path.normpath(args.out_dir))}_{name}.log"), "w") as log:
            subprocess.run([sys.executable, "-W", "ignore", "-u", "train_mixed.py", "--r", str(r), "--seed", str(seed),
                            "--steps", str(args.steps), "--batch_size", str(args.batch_size), "--generic", args.generic,
                            "--lr", str(args.lr), "--out_dir", args.out_dir,
                            "--steps_per_eval", "500"], stdout=log, stderr=subprocess.STDOUT, check=True)

subprocess.run([sys.executable, "-W", "ignore", "evaluate.py", *checkpoints, "--baselines",
                "--out_dir", os.path.join("results", os.path.basename(os.path.normpath(args.out_dir)))], check=True)
