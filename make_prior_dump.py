"""Pre-generates a pool of tables from one prior and stores it in an HDF5 file ("prior dump").

Training then reads tables from the pools instead of generating them on the fly: faster, exactly
reproducible, and every r draws from the same two pools.

Tables are generated in chunks of CHUNK tables; chunk c is seeded with (seed, c), so the file is
identical whatever the number of worker processes. Missing values stay NaN: they are imputed at
training time, once the train/test split of the batch is known.

Examples:
    python make_prior_dump.py --all          # the four pools the experiments use, same as on the Mac
    python make_prior_dump.py --prior generic --n_tables 100000
    python make_prior_dump.py --prior finance --n_tables 100000

Each file's content checksum is printed at the end and compared with the Mac pools (MAC_CHECKSUMS).
Pools regenerated on another machine can differ if torch or numpy draw differently there; experiments
stay comparable with each other as long as they all use the same pools.
"""

import argparse
import hashlib
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

# (prior, n_tables, seed): the training pools (seed 0) and the held-out pools for health checks (seed 1)
ALL_POOLS = [("finance", 100_000, 0), ("generic", 100_000, 0), ("finance", 1000, 1), ("generic", 1000, 1)]
# content checksums of the pools generated on the Mac on 2026-10-11
MAC_CHECKSUMS = {
    "finance_100k_256x24_seed0.h5": "b869733eb779",
    "generic_graph_scm_100k_256x24_seed0.h5": "0f3423218e96",
    "finance_1k_256x24_seed1.h5": "eb3e6b2272a2",
    "generic_graph_scm_1k_256x24_seed1.h5": "011ea0e594e1",
}


def content_checksum(path: str) -> str:
    """md5 over the tables themselves (X, y, num_features), so HDF5 file metadata does not matter."""
    digest = hashlib.md5()
    with h5py.File(path, "r") as f:
        for lo in range(0, f["X"].shape[0], CHUNK):
            digest.update(f["X"][lo:lo + CHUNK].tobytes())
        digest.update(f["y"][:].tobytes())
        digest.update(f["num_features"][:].tobytes())
    return digest.hexdigest()[:12]


def checksum_note(path: str) -> str:
    checksum, mac = content_checksum(path), MAC_CHECKSUMS.get(os.path.basename(path))
    match = "no Mac reference" if mac is None else "identical to the Mac pool" if checksum == mac else f"differs from the Mac pool ({mac})"
    return f"{os.path.basename(path)}: content checksum {checksum}, {match}"


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



def make_dump(prior, n_tables, seed, generic="graph_scm", n_rows=256, max_features=24, workers=8, out_dir="prior_dumps"):
    name = f"generic_{generic}" if prior == "generic" else "finance"
    path = os.path.join(out_dir, f"{name}_{n_tables // 1000}k_{n_rows}x{max_features}_seed{seed}.h5")
    if os.path.exists(path):
        print(f"exists, not regenerated: {checksum_note(path)}")
        return path
    os.makedirs(out_dir, exist_ok=True)
    n_chunks = -(-n_tables // CHUNK)
    jobs = [(prior, generic, c, min(CHUNK, n_tables - c * CHUNK), n_rows, max_features, seed) for c in range(n_chunks)]

    start = time.time()
    with h5py.File(path + ".tmp", "w") as f:
        X_out = f.create_dataset("X", (n_tables, n_rows, max_features), dtype="float32",
                                 chunks=(1, n_rows, max_features), compression="lzf")
        y_out = f.create_dataset("y", (n_tables, n_rows), dtype="int8", chunks=(1, n_rows), compression="lzf")
        nf_out = f.create_dataset("num_features", (n_tables,), dtype="int32")
        f.attrs.update({"prior": name, "n_rows": n_rows, "max_features": max_features, "max_classes": 2,
                        "seed": seed, "chunk": CHUNK, "nan": "missing values kept as NaN, impute at train time"})
        if prior == "generic":
            from importlib.metadata import version
            f.attrs["tabicl_version"] = version("tabicl")
        else:
            f.attrs["fintfm_commit"] = FINTFM_COMMIT
        with Pool(min(workers, n_chunks)) as pool:
            for done, (chunk, X, y, nf) in enumerate(pool.imap_unordered(generate_chunk, jobs), 1):
                lo = chunk * CHUNK
                X_out[lo:lo + len(X)] = X
                y_out[lo:lo + len(y)] = y
                nf_out[lo:lo + len(nf)] = nf
                print(f"{done}/{n_chunks} chunks, {time.time() - start:.0f}s", flush=True)
    os.replace(path + ".tmp", path)  # only a complete file gets the final name
    print(f"saved {path} ({os.path.getsize(path) / 2**30:.2f} GB) in {time.time() - start:.0f}s")
    print(checksum_note(path))
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="make the four pools in ALL_POOLS")
    parser.add_argument("--prior", choices=["generic", "finance"])
    parser.add_argument("--generic", default="graph_scm", help="TabICL prior type when --prior generic")
    parser.add_argument("--n_tables", type=int, default=100_000)
    parser.add_argument("--n_rows", type=int, default=256)
    parser.add_argument("--max_features", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out_dir", default="prior_dumps")
    args = parser.parse_args()

    if args.all:
        for prior, n_tables, seed in ALL_POOLS:
            make_dump(prior, n_tables, seed, workers=args.workers, out_dir=args.out_dir)
    elif args.prior:
        make_dump(args.prior, args.n_tables, args.seed, args.generic, args.n_rows, args.max_features, args.workers, args.out_dir)
    else:
        parser.error("give --all or --prior")
