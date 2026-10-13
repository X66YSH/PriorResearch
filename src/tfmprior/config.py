"""Configs: configs/base.yaml, overlaid by an experiment file, overlaid by command-line key=value overrides.

An experiment file may hold a `sweep` block: {dotted key: [values]}. expand_sweep turns it into one config per
combination (cartesian product, in the listed order). Values given on the command line override the sweep.

Run names come from the mixture, the seed and any other swept key, e.g.
    r0.3_seed1_generic-graph_scm_domain-fintfm    (two priors: r = share of the second)
    r0_seed0_generic-graph_scm                    (one prior)
    mix-graph_scm0.4-format_aug0.3-fintfm0.3_seed2 (three or more)
"""

import copy
import itertools
import os

import yaml

from tfmprior.utils.io import REPO_ROOT

BASE_CONFIG = os.path.join(REPO_ROOT, "configs", "base.yaml")


def deep_merge(base: dict, update: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict) and key != "mixture":
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)  # a mixture is replaced as a whole, never merged entry by entry
    return out


def set_dotted(config: dict, key: str, value) -> None:
    node = config
    *parents, last = key.split(".")
    for part in parents:
        node = node.setdefault(part, {})
    node[last] = value


def get_dotted(config: dict, key: str):
    node = config
    for part in key.split("."):
        node = node[part]
    return node


def parse_overrides(items: list[str]) -> dict:
    """['train.steps=10', 'data.mixture={graph_scm: 1.0}'] -> {dotted key: parsed YAML value}."""
    out = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"override {item!r} is not key=value")
        key, value = item.split("=", 1)
        out[key.strip()] = yaml.safe_load(value)
    return out


def load_config(experiment_path: str | None = None, overrides: list[str] | None = None, base_path: str = BASE_CONFIG) -> dict:
    with open(base_path) as f:
        config = yaml.safe_load(f)
    if experiment_path:
        with open(experiment_path) as f:
            config = deep_merge(config, yaml.safe_load(f) or {})
    config["overrides"] = parse_overrides(overrides or [])
    return config


def expand_sweep(config: dict) -> list[dict]:
    """One fully resolved config per run (sweep combinations, then the command-line overrides on top)."""
    sweep = config.get("sweep") or {}
    overrides = config.get("overrides") or {}
    keys = [k for k in sweep if k not in overrides]
    runs = []
    for values in itertools.product(*(sweep[k] for k in keys)):
        run = {k: v for k, v in config.items() if k not in ("sweep", "overrides")}
        run = copy.deepcopy(run)
        for key, value in zip(keys, values):
            set_dotted(run, key, copy.deepcopy(value))
        for key, value in overrides.items():
            set_dotted(run, key, copy.deepcopy(value))
        run["run"] = {"swept": {k: v for k, v in zip(keys, values)}, "overrides": overrides}
        run["run"]["name"] = run_name(run, swept_keys=keys)
        runs.append(run)
    return runs


def _number(x) -> str:
    return f"{x:g}"


def run_name(config: dict, swept_keys=()) -> str:
    mixture = config["data"]["mixture"]
    names = list(mixture)
    if len(names) == 1:
        name = f"r0_seed{config['train']['seed']}_generic-{names[0]}"
    elif len(names) == 2:
        name = f"r{_number(mixture[names[1]])}_seed{config['train']['seed']}_generic-{names[0]}_domain-{names[1]}"
    else:
        name = "mix-" + "-".join(f"{n}{_number(s)}" for n, s in mixture.items()) + f"_seed{config['train']['seed']}"
    for key in swept_keys:  # other swept settings, e.g. _lr0.001
        if key in ("data.mixture", "train.seed"):
            continue
        short = key.split(".")[-1]
        value = get_dotted(config, key)
        name += f"_{short}{_number(value) if isinstance(value, (int, float)) else value}"
    return name
