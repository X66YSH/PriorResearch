"""Config loading, overrides, sweep expansion and run names."""

import os

from tfmprior.config import expand_sweep, load_config

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_base_defaults_match_the_batch32_runs():
    config = load_config()
    assert config["train"]["batch_size"] == 32 and config["train"]["lr"] == 0.004
    assert config["data"]["split_sizes"] == list(range(80, 225, 16))
    assert list(config["data"]["mixture"]) == ["graph_scm"]


def test_override_and_sweep(tmp_path):
    experiment = tmp_path / "exp.yaml"
    experiment.write_text("experiment: {name: t}\nsweep:\n  data.mixture: [{graph_scm: 1.0}, {graph_scm: 0.7, fintfm: 0.3}]\n"
                          "  train.seed: [0, 1]\n  train.lr: [0.004, 0.001]\n")
    runs = expand_sweep(load_config(str(experiment), ["train.steps=7", "train.lr=0.002"]))
    assert len(runs) == 4  # lr is fixed by the command line, so only mixture x seed remain
    assert all(r["train"]["steps"] == 7 and r["train"]["lr"] == 0.002 for r in runs)
    names = [r["run"]["name"] for r in runs]
    assert names == ["r0_seed0_generic-graph_scm", "r0_seed1_generic-graph_scm",
                     "r0.3_seed0_generic-graph_scm_domain-fintfm", "r0.3_seed1_generic-graph_scm_domain-fintfm"]


def test_mixture_override_replaces_not_merges(tmp_path):
    runs = expand_sweep(load_config(None, ["data.mixture={fintfm: 1.0}"]))
    assert runs[0]["data"]["mixture"] == {"fintfm": 1.0}


def test_other_swept_keys_appear_in_the_run_name(tmp_path):
    experiment = tmp_path / "exp.yaml"
    experiment.write_text("sweep:\n  train.lr: [0.004, 0.001]\n")
    assert [r["run"]["name"] for r in expand_sweep(load_config(str(experiment)))] == [
        "r0_seed0_generic-graph_scm_lr0.004", "r0_seed0_generic-graph_scm_lr0.001"]
