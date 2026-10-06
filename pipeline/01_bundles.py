#!/usr/bin/env python3
"""Step 1: download the BWM data and compute one PETH bundle per insertion.

For the 515 insertions in insertions.csv, dmn_bwm.get_all_PETHs_parallel writes
$DMN_DATA/concat/<eid>_<probe>.npy (per-trial PETHs for the 21 conditions).
Existing bundles are kept (delete one to recompute it). Needs ONE/Alyx access;
downloads go to $ONE_CACHE_DIR (default ~/Downloads/ONE). Takes several hours.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import dmn_bwm as d  # noqa: E402


def insertions():
    with open(HERE / "insertions.csv") as fh:
        return [(row["eid"], row["probe"], row["pid"]) for row in csv.DictReader(fh)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    t0 = time.time()
    todo = [x for x in insertions() if not (d.DMN_BASE / "concat" / f"{x[0]}_{x[1]}.npy").exists()]
    print(f"{len(todo)} of {len(insertions())} bundles to compute", flush=True)
    if not todo:
        return
    d.use_online_one()
    res = d.get_all_PETHs_parallel(eids_plus=todo, vers="concat", n_workers=args.workers, bad_eids=[])
    # Retry failures with few workers (parallel downloads into the same session
    # folder can leave a probe's spike sorting empty).
    for workers in (2, 1):
        failed = [(e, p, q) for (e, p, q, _) in res["failures"]]
        if not failed:
            break
        print(f"retrying {len(failed)} insertions with {workers} worker(s)", flush=True)
        res = d.get_all_PETHs_parallel(eids_plus=failed, vers="concat", n_workers=workers, bad_eids=[])
    for f in res["failures"]:
        print("FAILED", f, flush=True)
    print(f"bundles done in {(time.time() - t0) / 60:.1f} min", flush=True)
    if res["failures"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
