"""Pre-generates a pool of tables from one prior and stores it in an HDF5 file ("prior dump").

Training then reads tables from the pools instead of generating them on the fly: faster, exactly
reproducible, and every r draws from the same two pools.

Tables are generated in chunks of CHUNK tables; chunk c is seeded with (seed, c), so the file is
identical whatever the number of worker processes. Missing values stay NaN: they are imputed at
training time, once the train/test split of the batch is known.

Examples:
    python make_prior_dump.py --prior generic --n_tables 100000
    python make_prior_dump.py --prior finance --n_tables 100000
"""

import argparse
import os
import random
import time
from multiprocessing import Pool

import h5py
import numpy as np
import torch

from priors.fintfm_financial import sample_financial_task
from priors.mixture import GenericPrior

CHUNK = 1000
FINTFM_COMMIT = "7c297d02a29c44d16a579d67e7b648cf61a631ad"


def generate_chunk(job):
    prior, generic_type, chunk, n_tables, n_rows, max_features, seed = job
    torch.set_num_threads(1)
    X = np.zeros((n_tables, n_rows, max_features), dtype=np.float32)
    y = np.zeros((n_tables, n_rows), dtype=np.int8)
    num_features = np.zeros(n_tables, dtype=np.int32)
    if prior == "generic":
        chunk_seed = seed * 1_000_003 + chunk
        random.seed(chunk_seed)
        np.random.seed(chunk_seed)
        torch.manual_seed(chunk_seed)
        generator = GenericPrior(n_rows, max_features, 2, generic_type)
        sample = generator.sample
    else:
        rng = np.random.default_rng([seed, chunk])
        sample = lambda: (lambda t: (t.X, t.y))(sample_financial_task(rng, n_rows, max_features=max_features))
    for i in range(n_tables):
        X_i, y_i = sample()
        X[i, :, :X_i.shape[1]] = X_i
        y[i] = y_i
        num_features[i] = X_i.shape[1]
    return chunk, X, y, num_features


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior", choices=["generic", "finance"], required=True)
    parser.add_argument("--generic", default="graph_scm", help="TabICL prior type when --prior generic")
    parser.add_argument("--n_tables", type=int, default=100_000)
    parser.add_argument("--n_rows", type=int, default=256)
    parser.add_argument("--max_features", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out_dir", default="prior_dumps")
    args = parser.parse_args()

    name = f"generic_{args.generic}" if args.prior == "generic" else "finance"
    path = os.path.join(args.out_dir, f"{name}_{args.n_tables // 1000}k_{args.n_rows}x{args.max_features}_seed{args.seed}.h5")
    os.makedirs(args.out_dir, exist_ok=True)
    n_chunks = -(-args.n_tables // CHUNK)
    jobs = [(args.prior, args.generic, c, min(CHUNK, args.n_tables - c * CHUNK), args.n_rows, args.max_features, args.seed)
            for c in range(n_chunks)]

    start = time.time()
    with h5py.File(path + ".tmp", "w") as f:
        X_out = f.create_dataset("X", (args.n_tables, args.n_rows, args.max_features), dtype="float32",
                                 chunks=(1, args.n_rows, args.max_features), compression="lzf")
        y_out = f.create_dataset("y", (args.n_tables, args.n_rows), dtype="int8", chunks=(1, args.n_rows), compression="lzf")
        nf_out = f.create_dataset("num_features", (args.n_tables,), dtype="int32")
        f.attrs.update({"prior": name, "n_rows": args.n_rows, "max_features": args.max_features, "max_classes": 2,
                        "seed": args.seed, "chunk": CHUNK, "nan": "missing values kept as NaN, impute at train time"})
        if args.prior == "generic":
            from importlib.metadata import version
            f.attrs["tabicl_version"] = version("tabicl")
        else:
            f.attrs["fintfm_commit"] = FINTFM_COMMIT
        with Pool(args.workers) as pool:
            for done, (chunk, X, y, nf) in enumerate(pool.imap_unordered(generate_chunk, jobs), 1):
                lo = chunk * CHUNK
                X_out[lo:lo + len(X)] = X
                y_out[lo:lo + len(y)] = y
                nf_out[lo:lo + len(nf)] = nf
                print(f"{done}/{n_chunks} chunks, {time.time() - start:.0f}s", flush=True)
    os.replace(path + ".tmp", path)  # only a complete file gets the final name
    print(f"saved {path} ({os.path.getsize(path) / 2**30:.2f} GB) in {time.time() - start:.0f}s")
