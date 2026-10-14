"""One run = one directory under runs/<experiment>/<run name>/:

    config.json, git_commit.txt, env.txt   resolved config and provenance
    train_log.csv                          step, loss, share of non-base tables in the batch, seconds
    checkpoints/step_<N>.pt                schedule-free evaluation weights + config
    eval/step_<N>/synthetic.json           held-out synthetic tables + breast cancer, written when the checkpoint is saved
    eval/step_<N>/metrics.json             suites summary + synthetic + health label
    eval/step_<N>/per_dataset.csv          suite, dataset, repeat, auc, ece, brier
    DONE                                   written last; a run without it is unfinished
"""

import csv
import glob
import json
import os
import re
import time

import torch

from tfmprior.data.loader import PoolMixtureLoader
from tfmprior.eval.benchmarks import load_suite
from tfmprior.eval.evaluate import build_model, evaluate_suites, load_classifier
from tfmprior.eval.health import breast_cancer_scores, health_label
from tfmprior.eval.synthetic import load_validation_tables, validation_scores
from tfmprior.model.nanotabpfn import NanoTabPFNClassifier
from tfmprior.train.loop import train
from tfmprior.utils.device import get_default_device
from tfmprior.utils.io import prepare_run_dir, write_json, write_provenance
from tfmprior.utils.seed import set_seed


def resolve_device(config: dict) -> str:
    device = config["train"].get("device", "auto")
    return get_default_device() if device == "auto" else device


def run_dir_of(config: dict) -> str:
    return os.path.join(config["output"]["runs_dir"], config["experiment"]["name"], config["run"]["name"])


def pool_paths(config: dict, which: dict) -> dict:
    return {name: os.path.join(config["data"]["pools_dir"], filename) for name, filename in which.items()}


def build_loader(config: dict) -> PoolMixtureLoader:
    data, tr = config["data"], config["train"]
    missing = [name for name in data["mixture"] if name not in data["pools"]]
    if missing:
        raise KeyError(f"no pool configured for {missing} (data.pools)")
    return PoolMixtureLoader(data["mixture"], pool_paths(config, data["pools"]), tr["steps"], tr["batch_size"],
                             data["split_sizes"], tr["seed"])


def load_suites(config: dict) -> dict:
    ev = config["eval"]
    suites = {}
    for suite in ev["suites"]:
        datasets = load_suite(suite, ev["benchmark_cache"])
        if ev.get("max_datasets_per_suite"):
            datasets = dict(list(datasets.items())[:ev["max_datasets_per_suite"]])
        suites[suite] = datasets
    return suites


def load_validation(config: dict) -> dict:
    validation = config["eval"]["validation"]
    paths = pool_paths(config, validation["pools"])
    return {name: load_validation_tables(path, validation["n_tables"]) for name, path in paths.items()}


def synthetic_scores(classifier, validation: dict) -> dict:
    return {"validation": {name: validation_scores(classifier, tables) for name, tables in validation.items()},
            "breast_cancer": breast_cancer_scores(classifier)}


def write_eval(out_dir: str, rows: list, summary: dict, synthetic: dict | None, config: dict | None) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    metrics = {"suites": summary}
    if synthetic is not None:
        base = next(iter(config["data"]["mixture"]))
        metrics.update(synthetic)
        metrics["health"] = health_label(synthetic["validation"][base]["auc"], synthetic["breast_cancer"]["roc_auc"],
                                         config["eval"]["health"])
    write_json(os.path.join(out_dir, "metrics.json"), metrics)
    with open(os.path.join(out_dir, "per_dataset.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["suite", "dataset", "repeat", "auc", "ece", "brier"])
        writer.writeheader()
        writer.writerows(rows)
    return metrics


def checkpoint_steps(run_dir: str) -> list[int]:
    return sorted(int(re.search(r"step_(\d+)\.pt$", p).group(1)) for p in glob.glob(os.path.join(run_dir, "checkpoints", "step_*.pt")))


def evaluate_run(run_dir: str, config: dict, device: str, steps=None, suites=None, log=print) -> None:
    """Real-suite evaluation (and synthetic, if not done at save time) for each checkpoint of a run."""
    suites = suites if suites is not None else load_suites(config)
    validation = None
    for step in steps or checkpoint_steps(run_dir):
        out_dir = os.path.join(run_dir, "eval", f"step_{step}")
        synthetic_path = os.path.join(out_dir, "synthetic.json")
        classifier = load_classifier(os.path.join(run_dir, "checkpoints", f"step_{step}.pt"), device)
        if os.path.exists(synthetic_path):
            with open(synthetic_path) as f:
                synthetic = json.load(f)
        else:
            validation = validation or load_validation(config)
            synthetic = synthetic_scores(classifier, validation)
            os.makedirs(out_dir, exist_ok=True)
            write_json(synthetic_path, synthetic)
        rows, summary = evaluate_suites(classifier, suites, config["eval"]["n_folds"])
        metrics = write_eval(out_dir, rows, summary, synthetic, config)
        log(f"step {step}: health {metrics['health']} | " + " | ".join(
            f"{s} auc {summary[s]['mean']['auc']:.4f} ece {summary[s]['mean']['ece']:.4f}" for s in summary))


def train_run(config: dict, mode: str = "new", log=print) -> str | None:
    """Trains (and by default evaluates) one run. Returns its directory, or None if it was already DONE."""
    tr = config["train"]
    bad = [s for s in tr["save_at_steps"] if not 1 <= s <= tr["steps"]]
    if bad or not tr["save_at_steps"]:
        raise ValueError(f"train.save_at_steps must be non-empty steps within 1..{tr['steps']}, got {tr['save_at_steps']}")
    run_dir = run_dir_of(config)
    if not prepare_run_dir(run_dir, mode):
        log(f"skip, DONE: {run_dir}")
        return None
    device = resolve_device(config)
    write_provenance(run_dir, config, device)
    log(f"--- {run_dir} on {device} ---")

    set_seed(tr["seed"])
    model = build_model(config["model"])
    loader = build_loader(config)
    validation = load_validation(config)
    base = next(iter(config["data"]["mixture"]))
    os.makedirs(os.path.join(run_dir, "checkpoints"))
    log_file = open(os.path.join(run_dir, "train_log.csv"), "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["step", "loss", "non_base_share", "seconds"])
    start = time.time()

    def on_step(step, loss, batch, seconds):
        log_writer.writerow([step, repr(loss), sum(s != base for s in batch["sources"]) / len(batch["sources"]), f"{seconds:.4f}"])
        if step % 100 == 0 or step == tr["steps"]:
            log_file.flush()
            log(f"step {step}/{tr['steps']} | loss {loss:.4f} | {time.time() - start:.0f}s")

    def on_save(step, model):
        torch.save({"model": model.state_dict(), "config": config, "step": step},
                   os.path.join(run_dir, "checkpoints", f"step_{step}.pt"))
        synthetic = synthetic_scores(NanoTabPFNClassifier(model, device), validation)
        out_dir = os.path.join(run_dir, "eval", f"step_{step}")
        os.makedirs(out_dir, exist_ok=True)
        write_json(os.path.join(out_dir, "synthetic.json"), synthetic)
        log(f"saved step {step}: " + " | ".join(f"{n} auc {v['auc']:.3f} loss {v['log_loss']:.3f}"
                                                 for n, v in synthetic["validation"].items())
            + f" | breast cancer {synthetic['breast_cancer']['roc_auc']:.3f}")

    train(model, loader, tr["lr"], device, tr["clip_grad_norm"], on_step=on_step,
          save_at_steps=tr["save_at_steps"], on_save=on_save)
    log_file.close()
    write_json(os.path.join(run_dir, "train_summary.json"),
               {"tables_per_prior": loader.counts, "required_tables": loader.required, "seconds": time.time() - start})

    if config["eval"].get("after_train", True):
        evaluate_run(run_dir, config, device, log=log)
    with open(os.path.join(run_dir, "DONE"), "w") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S\n"))
    return run_dir
