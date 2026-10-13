"""Run directories, config and provenance files, hashes."""

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class RunExistsError(RuntimeError):
    pass


def prepare_run_dir(path: str, mode: str = "new") -> bool:
    """Creates the run directory. Returns False when the run is already DONE and mode is 'resume' (nothing to do).

    mode 'new': an existing directory is an error, so results are never overwritten by accident.
    mode 'resume': a DONE run is skipped; an unfinished one is cleared and run again from the start.
    mode 'overwrite': any existing directory is cleared.
    """
    if os.path.exists(path):
        if mode == "new":
            raise RunExistsError(f"{path} exists; pass --resume to skip finished runs or --overwrite to replace it")
        if mode == "resume" and os.path.exists(os.path.join(path, "DONE")):
            return False
        shutil.rmtree(path)
    os.makedirs(path)
    return True


def write_yaml(path: str, obj) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def write_json(path: str, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def git_info() -> str:
    def git(*args):
        return subprocess.run(["git", "-C", REPO_ROOT, *args], capture_output=True, text=True).stdout.strip()

    commit = git("rev-parse", "HEAD") or "unknown"
    dirty = git("status", "--porcelain", "--untracked-files=no")
    return f"commit {commit}\ndirty {bool(dirty)}\n" + (f"{dirty}\n" if dirty else "")


def env_info(device: str) -> str:
    import torch

    def v(package):
        try:
            return version(package)
        except PackageNotFoundError:
            return "not installed"

    lines = [f"python {sys.version.split()[0]}", f"platform {platform.platform()}", f"device {device}"]
    if device == "cuda":
        lines.append(f"gpu {torch.cuda.get_device_name(0)}")
    lines += [f"{p} {v(p)}" for p in ["torch", "numpy", "scikit-learn", "tabicl", "schedulefree", "h5py"]]
    return "\n".join(lines) + "\n"


def write_provenance(run_dir: str, config: dict, device: str) -> None:
    write_yaml(os.path.join(run_dir, "config.yaml"), config)
    with open(os.path.join(run_dir, "git_commit.txt"), "w") as f:
        f.write(git_info())
    with open(os.path.join(run_dir, "env.txt"), "w") as f:
        f.write(env_info(device))


def state_dict_sha256(state_dict: dict) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()
