"""Downloads the pre-generated prior pools from the GitHub release into artifacts/dumps/ and checks their sha256.

The pools (2.4 GB) are too large for git (GitHub refuses files over 100 MB), so they are assets of the release
pools-v1 of github.com/X66YSH/PriorResearch. Uses only the Python standard library; no token is needed while the
repo is public (set GITHUB_TOKEN if it becomes private).

    python scripts/download_pools.py               # every pool listed in the release's SHA256SUMS
    python scripts/download_pools.py --only 1k      # only files whose name contains "1k"

Afterwards: python scripts/make_dump.py --verify   (content checksums against artifacts/checksums.json)
"""

import argparse
import hashlib
import os
import shutil
import sys
import urllib.request

REPO = "X66YSH/PriorResearch"


def download(url: str, path: str) -> None:
    headers = {"Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}"} if os.environ.get("GITHUB_TOKEN") else {}
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers)) as r, open(path + ".part", "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 24)
    os.replace(path + ".part", path)  # a broken download never gets the final name


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", default="pools-v1")
    parser.add_argument("--out_dir", default="artifacts/dumps")
    parser.add_argument("--only", default="", help="download only files whose name contains this text")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    base = f"https://github.com/{REPO}/releases/download/{args.release}"

    sums_path = os.path.join(args.out_dir, "SHA256SUMS")
    download(f"{base}/SHA256SUMS", sums_path)
    expected = {name: digest for digest, name in (line.split() for line in open(sums_path) if line.strip())}

    ok = True
    for name, digest in expected.items():
        if args.only and args.only not in name:
            continue
        path = os.path.join(args.out_dir, name)
        if not (os.path.exists(path) and sha256(path) == digest):
            print(f"downloading {name} ...", flush=True)
            download(f"{base}/{name}", path)
        match = sha256(path) == digest
        ok &= match
        print(f"{'OK      ' if match else 'MISMATCH'} {name}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
