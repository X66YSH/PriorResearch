"""Random seeds. set_seed matches the original nanoTabPFN set_randomness_seed, so runs reproduce bit for bit."""

import os
import random
import sys

import numpy as np
import torch

HASH_SEED = "0"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_fixed_hash_seed() -> None:
    """Restarts the current script once with PYTHONHASHSEED fixed.

    TabICL's prior output depends on Python's per-process string hash seed, so generating tables from it
    (make_dump.py, some diagnostics) is reproducible only with a fixed hash seed. The hash seed can only be set
    before the interpreter starts. Training and evaluation read pre-generated pools and do not need this.
    """
    if os.environ.get("PYTHONHASHSEED") != HASH_SEED:
        os.execve(sys.executable, sys.orig_argv, {**os.environ, "PYTHONHASHSEED": HASH_SEED})
