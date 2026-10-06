#!/usr/bin/env python3
"""Rastermap fit on well-sampled PETH types only (odd trials).

The full fit (all 21 types) lines neurons up by noise peaks in rarely sampled
conditions (block change, discordant: median ~12 trials per half), producing
training-only diagonals. Here Rastermap (Methods parameters) is fitted on the odd
trials of the nine stimulus- and movement-aligned types with >= 10 trials per half
in >= 98% of insertions; all 21 types are then displayed on the even trials in
that order, so the excluded columns are a held-out test of the ordering.
Output: results/rastermap_fit9.npz (isort, labels numbered top to bottom, types).
"""

import argparse
import os
from pathlib import Path

import numpy as np
from rastermap import Rastermap

ROOT = Path(os.environ.get("DMN_DATA", Path.home() / "dmn"))
RES = Path(__file__).resolve().parent / "results"
RES.mkdir(exist_ok=True)
TYPE_SETS = {
    # >= 10 trials per half in >= 98% of insertions
    "fit9": ["stimLbLcL", "stimRbRcR", "mistake_s", "motor_init", "sLbLchoiceL",
             "sRbRchoiceR", "mistake_m", "choiceL", "choiceR"],
    # >= 10 trials per half in every insertion (mistake types held out as well)
    "fit7": ["stimLbLcL", "stimRbRcR", "motor_init", "sLbLchoiceL", "sRbRchoiceR",
             "choiceL", "choiceR"],
}
FIT_TYPES = TYPE_SETS["fit9"]


def zrows(X):
    """Row z-score (rows with ~no variance become 0)."""
    sd = X.std(1, keepdims=True)
    return np.where(sd > 1e-6, (X - X.mean(1, keepdims=True)) / np.maximum(sd, 1e-6), 0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", default="fit9", choices=TYPE_SETS)
    ap.add_argument("--rezscore", action="store_true",
                    help="z-score each neuron within the selected columns only (feature "
                         "vector = these PETH types), instead of using the 21-type z-scores")
    args = ap.parse_args()
    FIT_TYPES = TYPE_SETS[args.set]
    tag = args.set + ("z" if args.rezscore else "")
    r = np.load(ROOT / "concat_cvTrue.npy", allow_pickle=True).flat[0]
    if r.get("cv_split") != "oddeven":
        raise RuntimeError("CV stack is not the odd/even split")
    start = np.cumsum([0] + list(r["len"].values()))
    names = list(r["len"])
    cols = np.concatenate([np.arange(start[names.index(t)], start[names.index(t) + 1])
                           for t in FIT_TYPES])
    X = np.asarray(r["X_odd"], np.float32)[:, cols]
    if args.rezscore:
        X = zrows(X)
    # Neurons (near-)silent in all fit columns cannot be normalized by Rastermap;
    # they are not fitted and are appended at the bottom with label -1.
    active = X.std(1) >= 1e-2
    idx = np.flatnonzero(active)
    model = Rastermap(n_PCs=200, n_clusters=100, locality=0.75, grid_upsample=0,
                      time_lag_window=5, bin_size=1).fit(X[active])
    sub = np.asarray(model.isort, int)
    raw = np.asarray(model.embedding_clust, int).reshape(-1)
    # Renumber clusters 0..99 in order of appearance along the sorting.
    seen = list(dict.fromkeys(raw[sub]))
    sub_labels = np.vectorize({c: k for k, c in enumerate(seen)}.get)(raw)
    labels = np.full(len(X), -1, int)
    labels[idx] = sub_labels
    isort = np.r_[idx[sub], np.flatnonzero(~active)]
    blocks = np.count_nonzero(np.diff(labels[isort])) + 1
    print(f"{(~active).sum()} neurons silent in the fit columns (label -1, at the bottom)")
    RES.mkdir(exist_ok=True)
    np.savez(RES / f"rastermap_{tag}.npz", isort=isort, labels=labels, uuids=np.asarray(r["uuids"]),
             fit_types=np.array(FIT_TYPES), n_fit_bins=cols.size)
    print(f"fit on {len(FIT_TYPES)} types ({cols.size} bins), {len(isort):,} neurons, "
          f"{len(seen)} clusters, {blocks} contiguous blocks -> {RES / f'rastermap_{tag}.npz'}")


if __name__ == "__main__":
    main()
