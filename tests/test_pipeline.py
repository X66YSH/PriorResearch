"""Smoke test and the key properties of the training pipeline (Phase A, A7)."""

import json
import os

import numpy as np
import pytest
import torch

from conftest import REPO_ROOT, make_pool, tiny_config
from tfmprior.data.dump import content_checksum
from tfmprior.data.loader import PoolMixtureLoader, PoolTooSmallError, required_tables
from tfmprior.eval.evaluate import build_model
from tfmprior.run import build_loader, train_run
from tfmprior.train.loop import train
from tfmprior.utils.seed import set_seed

SPLITS = [80, 96, 112, 128, 144, 160, 176, 192, 208, 224]
HAS_BENCHMARK_CACHE = os.path.isdir(os.path.join(REPO_ROOT, "artifacts", "benchmark_cache"))


@pytest.mark.skipif(not HAS_BENCHMARK_CACHE, reason="needs artifacts/benchmark_cache")
def test_smoke_train_save_evaluate(tiny_pools, tmp_path):
    """1. A tiny config trains, saves, evaluates and leaves a complete run directory."""
    config = tiny_config(tiny_pools, str(tmp_path))
    run_dir = train_run(config, log=lambda msg: None)
    for name in ["config.yaml", "git_commit.txt", "env.txt", "train_log.csv", "train_summary.json", "DONE",
                 "checkpoints/step_5.pt", "checkpoints/step_10.pt", "eval/step_10/metrics.json",
                 "eval/step_10/per_dataset.csv", "eval/step_5/synthetic.json"]:
        assert os.path.exists(os.path.join(run_dir, name)), name
    with open(os.path.join(run_dir, "train_log.csv")) as f:
        assert len(f.read().splitlines()) == 1 + 10
    metrics = json.load(open(os.path.join(run_dir, "eval", "step_10", "metrics.json")))
    assert set(metrics["suites"]) == {"finance", "general"}
    assert metrics["health"] in {"healthy", "unclear", "collapsed"}
    assert train_run(config, mode="resume", log=lambda msg: None) is None  # DONE runs are skipped
    with pytest.raises(RuntimeError):
        train_run(config, mode="new", log=lambda msg: None)  # never overwrite by accident


def test_pool_is_deterministic(tmp_path):
    """2. The same seed gives the same pool, whatever the number of worker processes."""
    for prior in ["fintfm", "graph_scm"]:
        a = make_pool(prior, 12, 3, str(tmp_path / "a"), workers=1)
        b = make_pool(prior, 12, 3, str(tmp_path / "b"), workers=2)
        assert content_checksum(a) == content_checksum(b), prior


def test_mixture_share():
    """3. With r = 0.3, the share of finance tables among 2000 draws is within 0.3 +- 0.03."""
    counts = required_tables({"graph_scm": 0.7, "fintfm": 0.3}, num_steps=500, batch_size=4, seed=0)
    assert sum(counts.values()) == 2000
    assert abs(counts["fintfm"] / 2000 - 0.3) <= 0.03


def test_loader_sources_match_plan(tiny_pools):
    """The loader takes exactly the planned number of tables from each pool."""
    loader = PoolMixtureLoader({"graph_scm": 0.7, "fintfm": 0.3}, tiny_pools["train"], 10, 4, SPLITS, seed=0)
    for batch in loader:
        assert batch["x"].shape == (4, 256, 24) and not torch.isnan(batch["x"]).any()
    assert loader.counts == loader.required


def test_r0_never_reads_finance_pool(tiny_pools):
    """4. With r = 0 the finance pool is never opened (its path does not even exist here)."""
    pools = {"graph_scm": tiny_pools["train"]["graph_scm"], "fintfm": "/does/not/exist.h5"}
    loader = PoolMixtureLoader({"graph_scm": 1.0, "fintfm": 0.0}, pools, 10, 4, SPLITS, seed=0)
    assert "fintfm" not in loader.paths
    assert all(set(batch["sources"]) == {"graph_scm"} for batch in loader)


def test_saved_weights_are_eval_weights_and_saving_does_not_change_training(tiny_pools):
    """5. A checkpoint holds the schedule-free evaluation weights, and saving leaves the trajectory unchanged."""
    def run(save_at):
        config = tiny_config(tiny_pools, "unused")
        set_seed(0)
        model = build_model(config["model"])
        losses, saved = [], {}
        train(model, build_loader(config), 4e-3, "cpu", on_step=lambda s, loss, b, t: losses.append(loss),
              save_at_steps=save_at, on_save=lambda s, m: saved.update({s: {k: v.clone() for k, v in m.state_dict().items()}}))
        return losses, saved, {k: v.clone() for k, v in model.state_dict().items()}

    losses_plain, _, final_plain = run(save_at=[])
    losses_saving, saved, final_saving = run(save_at=[3, 7, 10])
    assert losses_plain == losses_saving  # bit-identical trajectory
    assert all(torch.equal(final_plain[k], final_saving[k]) for k in final_plain)
    # the step-10 checkpoint equals the final weights train() returns, which are the eval weights (optimizer.eval())
    assert all(torch.equal(saved[10][k], final_saving[k]) for k in final_saving)
    # and the step-3 checkpoint differs from the training weights at step 3 (it is not the train-mode y)
    config = tiny_config(tiny_pools, "unused", steps=3)
    set_seed(0)
    model = build_model(config["model"])
    import schedulefree  # noqa: F401
    from tfmprior.train import loop
    captured = {}

    def capture_train_mode(step, loss, batch, seconds):
        if step == 3:
            captured.update({k: v.clone() for k, v in model.state_dict().items()})
    loop.train(model, build_loader(config), 4e-3, "cpu", on_step=capture_train_mode)
    assert any(not torch.equal(captured[k], saved[3][k]) for k in captured)


def test_pool_too_small_is_refused(tiny_pools):
    """6. A run that needs more tables than a pool holds fails before training."""
    with pytest.raises(PoolTooSmallError):
        PoolMixtureLoader({"graph_scm": 0.7, "fintfm": 0.3}, tiny_pools["train"], num_steps=100, batch_size=4,
                          split_sizes=SPLITS, seed=0)


def test_common_random_numbers_across_r():
    """Finance tables for r = 0.3 are the first ones r = 0.5 would take (same u stream)."""
    u = np.random.default_rng([0, 0]).random(1000)
    from tfmprior.data.loader import assign_sources
    low = assign_sources(u, {"graph_scm": 0.7, "fintfm": 0.3}) == 1
    high = assign_sources(u, {"graph_scm": 0.5, "fintfm": 0.5}) == 1
    assert np.all(high[low])  # every table that is finance at 0.3 is finance at 0.5
