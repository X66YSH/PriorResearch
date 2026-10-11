"""Fixes PYTHONHASHSEED before anything uses the priors.

TabICL's prior output depends on Python's per-process string hash seed, so the same `seed` gave
different tables in different processes (checked 2026-10-11: two different data checksums across
three processes, identical once PYTHONHASHSEED was fixed). The hash seed can only be set before the
interpreter starts, so a script that imports this package is restarted once with it fixed.
"""

import os
import sys
import warnings

HASH_SEED = "0"

if os.environ.get("PYTHONHASHSEED") != HASH_SEED:
    if "ipykernel" in sys.modules:  # restarting a notebook kernel would kill it
        warnings.warn(f"PYTHONHASHSEED is not {HASH_SEED}: tables generated on the fly from the generic prior "
                      f"(GenericPrior, MixedPriorDataLoader, make_prior_dump.py) will not be reproducible. "
                      f"Reading the pre-generated pools in prior_dumps/ is not affected.")
    else:
        os.execve(sys.executable, sys.orig_argv, {**os.environ, "PYTHONHASHSEED": HASH_SEED})
