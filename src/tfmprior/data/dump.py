"""Pre-generates a pool of tables from one prior into an HDF5 file.

Tables are generated in chunks of CHUNK tables; chunk c is seeded with (seed, c), so the file content does not
depend on the number of worker processes. Missing values stay NaN: they are imputed at training time, once the
train/test split of the batch is known. Generating from TabICL needs a fixed PYTHONHASHSEED
(utils.seed.ensure_fixed_hash_seed), or start Python with PYTHONHASHSEED=0.
"""

import hashlib
import os
import time
from multiprocessing import Pool

import h5py
import numpy as np
import torch

from tfmprior.priors.registry import get_prior, pool_filename

CHUNK = 1000


def generate_chunk(job):
    name, chunk, n_tables, n_rows, max_features, seed = job
    torch.set_num_threads(1)
    X = np.zeros((n_tables, n_rows, max_features), dtype=np.float32)
    y = np.zeros((n_tables, n_rows), dtype=np.int8)
    num_features = np.zeros(n_tables, dtype=np.int32)
    sample = get_prior(name).make_sampler(seed, chunk, n_rows, max_features)
    for i in range(n_tables):
        X_i, y_i = sample()
        X[i, :, :X_i.shape[1]] = X_i
        y[i] = y_i
        num_features[i] = X_i.shape[1]
    return chunk, X, y, num_features


def make_dump(name: str, n_tables: int, seed: int, n_rows: int = 256, max_features: int = 24, workers: int = 8,
              out_dir: str = "artifacts/dumps", log=print) -> str:
    """Writes the pool and returns its path; an existing pool is left as it is."""
    path = os.path.join(out_dir, pool_filename(name, n_tables, n_rows, max_features, seed))
    if os.path.exists(path):
        log(f"exists, not regenerated: {path}")
        return path
    os.makedirs(out_dir, exist_ok=True)
    n_chunks = -(-n_tables // CHUNK)
    jobs = [(name, c, min(CHUNK, n_tables - c * CHUNK), n_rows, max_features, seed) for c in range(n_chunks)]

    start = time.time()
    with h5py.File(path + ".tmp", "w") as f:
        X_out = f.create_dataset("X", (n_tables, n_rows, max_features), dtype="float32",
                                 chunks=(1, n_rows, max_features), compression="lzf")
        y_out = f.create_dataset("y", (n_tables, n_rows), dtype="int8", chunks=(1, n_rows), compression="lzf")
        nf_out = f.create_dataset("num_features", (n_tables,), dtype="int32")
        f.attrs.update({"prior": name, "n_rows": n_rows, "max_features": max_features, "max_classes": 2,
                        "seed": seed, "chunk": CHUNK, "nan": "missing values kept as NaN, impute at train time",
                        **get_prior(name).provenance()})
        with Pool(min(workers, n_chunks)) as pool:
            for done, (chunk, X, y, nf) in enumerate(pool.imap_unordered(generate_chunk, jobs), 1):
                lo = chunk * CHUNK
                X_out[lo:lo + len(X)] = X
                y_out[lo:lo + len(y)] = y
                nf_out[lo:lo + len(nf)] = nf
                log(f"{done}/{n_chunks} chunks, {time.time() - start:.0f}s")
    os.replace(path + ".tmp", path)  # only a complete file gets the final name
    log(f"saved {path} ({os.path.getsize(path) / 2**30:.2f} GB) in {time.time() - start:.0f}s")
    return path


def content_checksum(path: str) -> str:
    """md5 over the tables themselves (X, y, num_features), so HDF5 file metadata does not matter."""
    digest = hashlib.md5()
    with h5py.File(path, "r") as f:
        for lo in range(0, f["X"].shape[0], CHUNK):
            digest.update(f["X"][lo:lo + CHUNK].tobytes())
        digest.update(f["y"][:].tobytes())
        digest.update(f["num_features"][:].tobytes())
    return digest.hexdigest()[:12]


def pool_size(path: str) -> int:
    with h5py.File(path, "r") as f:
        return int(f["X"].shape[0])
