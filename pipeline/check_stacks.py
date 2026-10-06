#!/usr/bin/env python3
"""Check a (re)built set of stacks against the published build.

Compares neuron and insertion counts, the trial split and the stored arrays with
the values of the manuscript build (2026-10-02). Exit status 1 on any mismatch.
"""

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import dmn_bwm as d  # noqa: E402

EXPECTED = {"concat_cvFalse.npy": 54_719, "concat_cvTrue.npy": 54_569}
N_INSERTIONS = 515


def main():
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("ok   " if cond else "FAIL ") + msg)
        ok &= bool(cond)

    with open(HERE / "insertions.csv") as fh:
        pids = {row["pid"] for row in csv.DictReader(fh)}
    check(len(pids) == N_INSERTIONS, f"insertions.csv lists {len(pids)} insertions")
    n_bundles = len(list((d.DMN_BASE / "concat").glob("*_probe*.npy")))
    check(n_bundles == N_INSERTIONS, f"{n_bundles} per-insertion bundles in concat/")
    for name, n in EXPECTED.items():
        r = np.load(d.DMN_BASE / name, allow_pickle=True).flat[0]
        n_neurons = len(r["uuids"])
        check(n_neurons == n, f"{name}: {n_neurons:,} neurons (published build {n:,})")
        got = set(np.asarray(r["pid"]).astype(str))
        check(got == pids, f"{name}: {len(got)} insertions, same as insertions.csv: {got == pids}")
        if name == "concat_cvTrue.npy":
            check(r.get("cv_split") == "oddeven", f"{name}: trial split {r.get('cv_split')!r}")
            for k in ("isort", "rm_labels", "X", "X_odd", "X_even"):
                check(k in r, f"{name}: has {k}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
