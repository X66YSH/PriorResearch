"""Generates prior pools and checks them against artifacts/checksums.json (content checksums, kept in git).

    python scripts/make_dump.py --all --workers 8                         # every pool listed in artifacts/checksums.json
    python scripts/make_dump.py --prior fintfm --n_tables 400000 --seed 0 --record   # a new pool; --record adds it
    python scripts/make_dump.py --verify                                  # checksum the pools that exist

Pools generated on another machine can differ if torch or numpy draw differently there; runs stay comparable with
each other as long as they all use the same pools, but then they cannot be compared number by number with runs
on the reference pools. The checksum result says which case applies.
"""

import argparse
import json
import os

from tfmprior.data.dump import content_checksum, make_dump
from tfmprior.priors.registry import pool_filename
from tfmprior.utils.io import REPO_ROOT
from tfmprior.utils.seed import ensure_fixed_hash_seed

CHECKSUMS = os.path.join(REPO_ROOT, "artifacts", "checksums.json")


def load_checksums() -> dict:
    with open(CHECKSUMS) as f:
        return json.load(f)


def note(path: str, expected: str | None) -> str:
    checksum = content_checksum(path)
    if expected is None:
        status = "no reference checksum"
    elif checksum == expected:
        status = "identical to the reference pool"
    else:
        status = f"DIFFERS from the reference pool ({expected})"
    return f"{os.path.basename(path)}: content checksum {checksum}, {status}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="make every pool in artifacts/checksums.json")
    parser.add_argument("--verify", action="store_true", help="only checksum the pools in artifacts/checksums.json that exist")
    parser.add_argument("--prior")
    parser.add_argument("--n_tables", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_rows", type=int, default=256)
    parser.add_argument("--max_features", type=int, default=24)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out_dir", default="artifacts/dumps")
    parser.add_argument("--record", action="store_true", help="add the new pool's checksum to artifacts/checksums.json")
    args = parser.parse_args()
    log = lambda msg: print(msg, flush=True)  # noqa: E731

    if args.all or args.verify:
        for filename, spec in load_checksums()["pools"].items():
            path = os.path.join(args.out_dir, filename)
            if args.all:
                make_dump(spec["prior"], spec["n_tables"], spec["seed"], spec["n_rows"], spec["max_features"],
                          args.workers, args.out_dir, log)
            if os.path.exists(path):
                log(note(path, spec["content_checksum"]))
            else:
                log(f"{filename}: missing")
    elif args.prior:
        if not args.n_tables:
            parser.error("--n_tables is required with --prior")
        path = make_dump(args.prior, args.n_tables, args.seed, args.n_rows, args.max_features, args.workers, args.out_dir, log)
        checksums = load_checksums()
        filename = pool_filename(args.prior, args.n_tables, args.n_rows, args.max_features, args.seed)
        log(note(path, checksums["pools"].get(filename, {}).get("content_checksum")))
        if args.record:
            checksums["pools"][filename] = {"prior": args.prior, "n_tables": args.n_tables, "seed": args.seed,
                                            "n_rows": args.n_rows, "max_features": args.max_features,
                                            "content_checksum": content_checksum(path)}
            with open(CHECKSUMS, "w") as f:
                json.dump(checksums, f, indent=2)
            log(f"recorded {filename} in {CHECKSUMS}")
    else:
        parser.error("give --all, --verify or --prior")


if __name__ == "__main__":  # worker processes re-import this file; only the parent may run main()
    ensure_fixed_hash_seed()  # before anything generates tables from TabICL
    main()
