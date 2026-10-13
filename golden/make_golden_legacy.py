"""Records the golden answers with the pre-refactor code (A0). Run from the repo root at commit 6332ed5:

    python golden/make_golden_legacy.py --device mps --part all

It only calls the existing modules (train.py, evaluate.py, priors/, benchmark/) and never changes them.
Outputs, all small JSON files in golden/:
    train_<device>.json       per-step training loss and the final parameter hash (r = 0.3, seed 0, batch 32, lr 4e-3, 200 steps)
    eval_<device>.json        every dataset x repeat x metric for one checkpoint, plus the random forest baseline
    pools.json                sha256 of each pool file, content checksum, and hashes of the first 5 tables
    loader.json               source and pool index of every table in the first 20 batches (r = 0.3, seed 0, batch 32)
"""

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.getcwd())  # the legacy modules live at the repo root

import h5py
import numpy as np
import torch

GENERIC = "prior_dumps/generic_graph_scm_100k_256x24_seed0.h5"
FINANCE = "prior_dumps/finance_100k_256x24_seed0.h5"
HELDOUT = ["prior_dumps/generic_graph_scm_1k_256x24_seed1.h5", "prior_dumps/finance_1k_256x24_seed1.h5"]
EVAL_CHECKPOINT = "checkpoints/bs32_lr0.004_steps3000/r0.00_seed0.pth"
TRAIN = dict(r=0.3, seed=0, batch_size=32, lr=4e-3, steps=200)
LOADER = dict(r=0.3, seed=0, batch_size=32, batches=20)


def state_hash(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def golden_train(device):
    import train as legacy_train
    from model import NanoTabPFNModel
    from priors.batching import DumpMixedDataLoader

    losses = []
    base = torch.nn.CrossEntropyLoss

    class RecordingLoss(base):  # train.py builds nn.CrossEntropyLoss(); record each step's loss without touching train.py
        def forward(self, input, target):
            loss = super().forward(input, target)
            losses.append(loss.item())
            return loss

    torch.nn.CrossEntropyLoss = RecordingLoss
    try:
        legacy_train.set_randomness_seed(TRAIN["seed"])  # same order as train_mixed.py: seed, model, loader
        model = NanoTabPFNModel(embedding_size=96, num_attention_heads=4, mlp_hidden_size=192, num_layers=3, num_outputs=2)
        prior = DumpMixedDataLoader(TRAIN["r"], TRAIN["steps"], TRAIN["batch_size"], GENERIC, FINANCE, seed=TRAIN["seed"])
        model, _ = legacy_train.train(model, prior, lr=TRAIN["lr"], device=device, steps_per_eval=10**9, eval_func=None)
    finally:
        torch.nn.CrossEntropyLoss = base
    return {"settings": TRAIN, "device": device, "losses": losses, "param_sha256": state_hash(model),
            "finance_tables": prior.n_finance, "tables": prior.n_tables}


def golden_eval(device):
    from sklearn.ensemble import RandomForestClassifier

    from benchmark.datasets import SUITES, load_suite
    from benchmark.metrics import binary_metrics
    from evaluate import load_checkpoint, predict_out_of_fold

    out = {"checkpoint": EVAL_CHECKPOINT, "device": device, "model": {}, "random_forest": {}}
    for key, make in [("model", lambda: load_checkpoint(EVAL_CHECKPOINT, device)),
                      ("random_forest", lambda: RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1))]:
        classifier = make()
        for suite in SUITES:
            for name, repeats in load_suite(suite).items():
                for i, (X, y) in enumerate(repeats):
                    out[key][f"{suite}/{name}/{i}"] = binary_metrics(y, predict_out_of_fold(classifier, X, y))
    return out


def golden_pools():
    from make_prior_dump import content_checksum

    out = {}
    for path in [GENERIC, FINANCE, *HELDOUT]:
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 24), b""):
                digest.update(block)
        with h5py.File(path, "r") as f:
            first = [hashlib.sha256(f["X"][i].tobytes() + f["y"][i].tobytes() + f["num_features"][i].tobytes()).hexdigest()
                     for i in range(5)]
            n = int(f["X"].shape[0])
        out[os.path.basename(path)] = {"file_sha256": digest.hexdigest(), "content_checksum": content_checksum(path),
                                       "n_tables": n, "first5_sha256": first}
    return out


def golden_loader():
    from priors.batching import DumpMixedDataLoader

    draws = []

    class Recording(DumpMixedDataLoader):
        def _next_index(self, source):
            index = super()._next_index(source)
            draws.append([source, index])
            return index

    loader = Recording(LOADER["r"], LOADER["batches"], LOADER["batch_size"], GENERIC, FINANCE, seed=LOADER["seed"])
    splits = [batch["train_test_split_index"] for batch in loader]
    return {"settings": LOADER, "draws": draws, "split_index_per_batch": splits}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="mps")
    parser.add_argument("--part", choices=["all", "train", "eval", "pools", "loader"], default="all")
    parser.add_argument("--suffix", default="", help="appended to train/eval file names, e.g. _run2 for a determinism check")
    args = parser.parse_args()
    os.makedirs("golden", exist_ok=True)

    def save(name, obj):
        with open(f"golden/{name}.json", "w") as f:
            json.dump(obj, f, indent=1)
        print("saved", f"golden/{name}.json", flush=True)

    if args.part in ("all", "pools"):
        save("pools", golden_pools())
    if args.part in ("all", "loader"):
        save("loader", golden_loader())
    if args.part in ("all", "eval"):
        save(f"eval_{args.device}{args.suffix}", golden_eval(args.device))
    if args.part in ("all", "train"):
        save(f"train_{args.device}{args.suffix}", golden_train(args.device))
