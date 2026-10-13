"""Checks the current code against the golden answers in golden/ (recorded with the pre-refactor code, A0).

    python scripts/verify_golden.py --device mps            # all four checks; train and eval compare with golden/*_mps.json
    python scripts/verify_golden.py --device cpu --part train

Pass criteria: pools and loader identical; training loss sequence and final parameters bit-identical on the same
device; evaluation metrics within 1e-6.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys

import h5py
import numpy as np

from tfmprior.config import load_config
from tfmprior.data.dump import content_checksum
from tfmprior.data.loader import PoolMixtureLoader
from tfmprior.eval.evaluate import baseline_classifiers, evaluate_suites, load_classifier
from tfmprior.run import load_suites
from tfmprior.utils.io import REPO_ROOT, file_sha256, state_dict_sha256

GOLDEN = os.path.join(REPO_ROOT, "golden")
CONFIG = load_config()
POOLS_DIR = CONFIG["data"]["pools_dir"]
LEGACY_CHECKPOINT = "legacy_results/checkpoints/bs32_lr0.004_steps3000/r0.00_seed0.pth"


def golden(name):
    with open(os.path.join(GOLDEN, f"{name}.json")) as f:
        return json.load(f)


def check(label, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f": {detail}" if detail else ""), flush=True)
    return ok


def verify_pools():
    ok = True
    for filename, want in golden("pools").items():
        path = os.path.join(POOLS_DIR, filename)
        with h5py.File(path, "r") as f:
            first = [hashlib.sha256(f["X"][i].tobytes() + f["y"][i].tobytes() + f["num_features"][i].tobytes()).hexdigest()
                     for i in range(5)]
            n = int(f["X"].shape[0])
        same = (file_sha256(path) == want["file_sha256"], content_checksum(path) == want["content_checksum"],
                first == want["first5_sha256"], n == want["n_tables"])
        ok &= check(f"pool {filename}", all(same), "file sha256 / content / first 5 tables / size: " + " ".join(map(str, same)))
    return ok


def verify_loader():
    want = golden("loader")
    s = want["settings"]
    draws = []

    class Recording(PoolMixtureLoader):
        def _next_index(self, name):
            index = super()._next_index(name)
            draws.append(["finance" if name == "fintfm" else "generic", index])
            return index

    pools = {name: os.path.join(POOLS_DIR, f) for name, f in CONFIG["data"]["pools"].items()}
    loader = Recording({"graph_scm": 1 - s["r"], "fintfm": s["r"]}, pools, s["batches"], s["batch_size"],
                       CONFIG["data"]["split_sizes"], s["seed"])
    splits = [batch["train_test_split_index"] for batch in loader]
    ok = check("loader: source and pool index of all 640 tables", draws == want["draws"],
               f"{sum(a == b for a, b in zip(draws, want['draws']))}/{len(want['draws'])} identical")
    return check("loader: train/test split per batch", splits == want["split_index_per_batch"]) and ok


def verify_eval(device):
    want = golden(f"eval_{device}")
    suites = load_suites(CONFIG)
    ok = True
    for key, classifier in [("model", load_classifier(LEGACY_CHECKPOINT, device)),
                            ("random_forest", baseline_classifiers()["random_forest"])]:
        rows, _ = evaluate_suites(classifier, suites)
        got = {f"{r['suite']}/{r['dataset']}/{r['repeat']}": r for r in rows}
        diffs = [abs(got[k][m] - v[m]) for k, v in want[key].items() for m in ("auc", "ece", "brier")]
        ok &= check(f"eval {key}: {len(want[key])} dataset x repeat entries, 3 metrics",
                    set(got) == set(want[key]) and max(diffs) < 1e-6, f"max abs difference {max(diffs):.2e}")
    return ok


def verify_train(device):
    want = golden(f"train_{device}")
    cmd = [sys.executable, "scripts/train.py", "--config", "configs/experiments/golden.yaml", "--overwrite",
           f"train.device={device}", "output.runs_dir=runs_golden_check"]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    run_dir = os.path.join(REPO_ROOT, "runs_golden_check", "golden", "r0.3_seed0_generic-graph_scm_domain-fintfm")
    with open(os.path.join(run_dir, "train_log.csv")) as f:
        losses = [float(line.split(",")[1]) for line in f.read().splitlines()[1:]]
    import torch
    state = torch.load(os.path.join(run_dir, "checkpoints", "step_200.pt"), map_location="cpu", weights_only=False)["model"]
    summary = json.load(open(os.path.join(run_dir, "train_summary.json")))
    same = sum(a == b for a, b in zip(losses, want["losses"]))
    ok = check(f"train ({device}): loss of all {len(want['losses'])} steps bit-identical",
               losses == want["losses"], f"{same}/{len(want['losses'])} identical")
    ok &= check(f"train ({device}): final parameters", state_dict_sha256(state) == want["param_sha256"])
    return check(f"train ({device}): finance tables", summary["tables_per_prior"]["fintfm"] == want["finance_tables"],
                 f"{summary['tables_per_prior']['fintfm']} vs {want['finance_tables']}") and ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="mps")
    parser.add_argument("--part", choices=["all", "pools", "loader", "eval", "train"], default="all")
    args = parser.parse_args()
    results = []
    if args.part in ("all", "pools"):
        results.append(verify_pools())
    if args.part in ("all", "loader"):
        results.append(verify_loader())
    if args.part in ("all", "eval"):
        results.append(verify_eval(args.device))
    if args.part in ("all", "train"):
        results.append(verify_train(args.device))
    print("ALL PASS" if all(results) else "SOME CHECKS FAILED")
    sys.exit(0 if all(results) else 1)
