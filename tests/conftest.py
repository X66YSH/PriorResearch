import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_pool(prior: str, n_tables: int, seed: int, out_dir: str, workers: int = 2) -> str:
    """Generates a small pool with scripts/make_dump.py in a subprocess (it fixes PYTHONHASHSEED itself)."""
    subprocess.run([sys.executable, "scripts/make_dump.py", "--prior", prior, "--n_tables", str(n_tables),
                    "--seed", str(seed), "--workers", str(workers), "--out_dir", out_dir],
                   cwd=REPO_ROOT, check=True, capture_output=True)
    from tfmprior.priors.registry import pool_filename
    return os.path.join(out_dir, pool_filename(prior, n_tables, 256, 24, seed))


@pytest.fixture(scope="session")
def tiny_pools(tmp_path_factory):
    """60-table training pools (seed 0) and validation pools (seed 1) for graph_scm and fintfm."""
    out_dir = str(tmp_path_factory.mktemp("pools"))
    return {
        "dir": out_dir,
        "train": {p: make_pool(p, 60, 0, out_dir) for p in ["graph_scm", "fintfm"]},
        "validation": {p: make_pool(p, 60, 1, out_dir) for p in ["graph_scm", "fintfm"]},
    }


def tiny_config(tiny_pools, runs_dir: str, **train_overrides) -> dict:
    """A resolved run config: 2 layers, batch 4, 10 steps, CPU, one dataset per suite."""
    from tfmprior.config import expand_sweep, load_config

    overrides = [
        f"data.pools_dir={tiny_pools['dir']}",
        "data.pools={graph_scm: " + os.path.basename(tiny_pools["train"]["graph_scm"])
        + ", fintfm: " + os.path.basename(tiny_pools["train"]["fintfm"]) + "}",
        "eval.validation.pools={graph_scm: " + os.path.basename(tiny_pools["validation"]["graph_scm"])
        + ", fintfm: " + os.path.basename(tiny_pools["validation"]["fintfm"]) + "}",
        "eval.validation.n_tables=2",
        "eval.max_datasets_per_suite=1",
        f"eval.benchmark_cache={os.path.join(REPO_ROOT, 'artifacts', 'benchmark_cache')}",
        "data.mixture={graph_scm: 0.7, fintfm: 0.3}",
        "model.num_layers=2", "model.embedding_size=16", "model.mlp_hidden_size=32",
        "train.batch_size=4", "train.steps=10", "train.save_at_steps=[5, 10]", "train.device=cpu",
        f"output.runs_dir={runs_dir}", "experiment.name=smoke",
    ] + [f"train.{k}={v}" for k, v in train_overrides.items()]
    return expand_sweep(load_config(None, overrides))[0]
