#!/usr/bin/env python3
"""Does a Rastermap block's diagonal replicate on held-out trials?

For neurons in a block of clusters (fit: results/rastermap_<set>.npz) and one PETH
column: peak time on odd trials (fit data) vs even trials (held out). Statistic:
Spearman rho between the two, with a null from shuffling the even-trial peaks
across neurons of the block; plus the median absolute peak-time difference, and
the same for the block's mean |odd - even| against the shuffle.
Feature vectors as in the fit with --rezscore (z-score within the fit columns;
the peak time is unaffected by z-scoring).
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path.home() / "dmn"
RES = Path(__file__).resolve().parent / "results"
C_SEC = 480.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", default="fit7z")
    ap.add_argument("--block", nargs=2, type=int, action="append", required=True,
                    metavar=("FIRST", "LAST"), help="cluster range (inclusive); repeatable")
    ap.add_argument("--column", action="append", required=True, help="PETH type, one per block")
    ap.add_argument("--n-shuffle", type=int, default=2000)
    args = ap.parse_args()

    r = np.load(ROOT / "concat_cvTrue.npy", allow_pickle=True).flat[0]
    f = np.load(RES / f"rastermap_{args.set}.npz", allow_pickle=True)
    labels = f["labels"]
    start = np.cumsum([0] + list(r["len"].values()))
    names = list(r["len"])
    rng = np.random.default_rng(0)
    lines = [f"fit {args.set}: peak time, odd vs even trials, within block and column"]
    for (lo, hi), col in zip(args.block, args.column):
        sel = np.flatnonzero((labels >= lo) & (labels <= hi))
        i = names.index(col)
        sl = slice(start[i], start[i + 1])
        po = np.asarray(r["X_odd"][sel, sl]).argmax(1) / C_SEC * 1000
        pe = np.asarray(r["X_even"][sel, sl]).argmax(1) / C_SEC * 1000
        rho = spearmanr(po, pe).statistic
        null = np.array([spearmanr(po, rng.permutation(pe)).statistic for _ in range(args.n_shuffle)])
        mad = np.median(np.abs(po - pe))
        mad_null = np.median([np.median(np.abs(po - rng.permutation(pe))) for _ in range(500)])
        p = (np.sum(null >= rho) + 1) / (args.n_shuffle + 1)
        lines.append(f"  clusters {lo}-{hi}, {col}: n={sel.size:,} | rho={rho:.2f} "
                     f"(shuffle 95th pct {np.percentile(null, 95):.2f}, p={p:.1e}) | "
                     f"median |odd-even| {mad:.0f} ms (shuffle {mad_null:.0f} ms)")
    out = "\n".join(lines)
    print(out)
    (RES / f"peak_time_replication_{args.set}.txt").write_text(out + "\n")


if __name__ == "__main__":
    main()
