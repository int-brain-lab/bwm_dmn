#!/usr/bin/env python3
"""Install the intermediate data release, so every figure can be rebuilt without the raw data.

The release (a folder, e.g. downloaded from the shared drive) contains MANIFEST.sha256 and
  data/  -> copied into $DMN_DATA (default ~/dmn): data stacks, shared caches, BWM decoding
            tables, compact RRR selectivity
  repo/  -> copied into this repository: outputs of the few steps that need raw IBL data
            (reaction-time split, wheel analysis, Fig. 1 event timing) or are version
            sensitive (the Rastermap order of Fig. 5)
Checksums are verified first. Then: pipeline/run_all.sh --from 3

Usage: python pipeline/install_intermediate.py /path/to/release [--link]
  --link   link instead of copy (hard links, or symbolic links across file systems)
"""

import argparse
import hashlib
import os
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def sha256(p, chunk=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("release", type=Path)
    ap.add_argument("--link", action="store_true")
    a = ap.parse_args()
    data = Path(os.environ.get("DMN_DATA", Path.home() / "dmn"))
    entries = [l.split(None, 1) for l in (a.release / "MANIFEST.sha256").read_text().splitlines() if l.strip()]
    for digest, rel in entries:
        rel = rel.strip().removeprefix("./")
        src = a.release / rel
        if sha256(src) != digest:
            raise SystemExit(f"checksum mismatch: {rel}")
        top, sub = rel.split("/", 1)
        dst = (data if top == "data" else REPO) / sub
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            dst.unlink()
        if a.link:
            try:
                os.link(src, dst)
            except OSError:  # different file system: symbolic link
                dst.symlink_to(src.resolve())
        else:
            shutil.copy2(src, dst)
        print(f"{rel} -> {dst}")
    print(f"installed {len(entries)} files; data in {data}. Next: pipeline/run_all.sh --from 3")


if __name__ == "__main__":
    main()
