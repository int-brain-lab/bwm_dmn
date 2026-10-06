#!/usr/bin/env python3
"""Are the stimulus-window "sequences" stimulus-locked latency diversity (a
"tornado") or timed relative to movement?

Per neuron (all 515 insertions, per-trial PETH bundles ~/dmn/concat/):
- trials: the four correct stimulus-aligned trial types (stimLbLcL, stimLbRcL,
  stimRbRcR, stimRbLcR; window 0-150 ms after stimulus onset) and the matching
  movement-aligned types (sLbLchoiceL, ...; window 150-0 ms before first movement);
- odd/even split within each trial type, as in the Methods (odd = 1st, 3rd, ...);
- odd trials: peak latency of the trial-type-averaged response (selection only);
  reliability = corr(odd, even) of the four concatenated per-type means;
- even trials: split at the session's median reaction time (first movement -
  stimulus onset) into fast and slow; peak latency on each, in both alignments.

Stimulus-locked: delta_stim = peak_slow - peak_fast ~ 0 (stimulus frame).
Movement-locked: delta_move ~ 0 (movement frame), delta_stim ~ RT difference.
Results: results/neurons.csv, results/peths.npz, results/summary.txt.
"""

import argparse
from multiprocessing import Pool
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("DMN_DATA", Path.home() / "dmn"))
HERE = Path(__file__).resolve().parent
RES = HERE / "results"
TRIALS = Path.home() / "Downloads/ONE/bwm_tables/trials.pqt"
STIM = ["stimLbLcL", "stimLbRcL", "stimRbRcR", "stimRbLcR"]
MOVE = ["sLbLchoiceL", "sLbRchoiceL", "sRbRchoiceR", "sRbLchoiceR"]
C_SEC = 480.0  # strided bins per second (dmn_bwm.c_sec)
EDGE = 3  # peaks within EDGE bins of a window edge are not interior

_T = None


def rts_for(eid):
    t = _T[_T.eid == eid].reset_index(drop=True)
    return (t.firstMovement_times - t.stimOn_times).to_numpy()


def per_type(D, names, rt):
    """For each trial type: (trials, neurons, bins) array, RT per trial, odd mask."""
    out = []
    tn = list(D["trial_names"])
    for name in names:
        X = np.asarray(D["ws"][tn.index(name)], dtype=np.float32)
        ti = np.asarray(D["trial_meta"][name]["trial_index"])
        odd = np.zeros(len(ti), bool)
        odd[0::2] = True
        out.append((X, rt[ti], odd))
    return out


def peak(p):
    return int(np.argmax(p))


def analyse(path):
    D = np.load(path, allow_pickle=True).flat[0]
    eid = path.name.split("_")[0]
    rt = rts_for(eid)
    stim, move = per_type(D, STIM, rt), per_type(D, MOVE, rt)
    if min(len(x[1]) for x in stim + move) < 4:
        return None
    even_rt = np.concatenate([r[~o] for _, r, o in stim])
    med = np.nanmedian(even_rt)

    def means(blocks, mask_fn):
        """Trial-type-averaged mean over trials selected by mask_fn(rt, odd)."""
        ms = []
        for X, r, o in blocks:
            m = mask_fn(r, o)
            if m.sum() == 0:
                return None
            ms.append(X[m].mean(0))
        return np.mean(ms, 0), np.concatenate(ms, 1)

    odd = means(stim, lambda r, o: o)
    even = means(stim, lambda r, o: ~o)
    sf = means(stim, lambda r, o: ~o & (r < med))
    ss = means(stim, lambda r, o: ~o & (r >= med))
    mf = means(move, lambda r, o: ~o & (r < med))
    ms = means(move, lambda r, o: ~o & (r >= med))
    if any(x is None for x in (odd, even, sf, ss, mf, ms)):
        return None
    a, b = odd[1], even[1]
    a0, b0 = a - a.mean(1, keepdims=True), b - b.mean(1, keepdims=True)
    rel = (a0 * b0).sum(1) / np.sqrt((a0 ** 2).sum(1) * (b0 ** 2).sum(1) + 1e-12)
    rows = []
    for i, u in enumerate(np.asarray(D["uuids"]).astype(str)):
        rows.append(dict(
            uuid=u, eid=eid, reliability=rel[i], rate=float(a[i].mean()),
            peak_odd=peak(odd[0][i]) / C_SEC,
            stim_fast=peak(sf[0][i]) / C_SEC, stim_slow=peak(ss[0][i]) / C_SEC,
            move_fast=peak(mf[0][i]) / C_SEC - 0.15, move_slow=peak(ms[0][i]) / C_SEC - 0.15,
            rt_fast=float(np.nanmedian(even_rt[even_rt < med])),
            rt_slow=float(np.nanmedian(even_rt[even_rt >= med])),
            interior=EDGE <= peak(odd[0][i]) < odd[0].shape[1] - EDGE))
    peths = dict(stim_fast=sf[0], stim_slow=ss[0], move_fast=mf[0], move_slow=ms[0], odd=odd[0])
    return pd.DataFrame(rows), peths


def init():
    global _T
    _T = pd.read_parquet(TRIALS, columns=["eid", "stimOn_times", "firstMovement_times"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    RES.mkdir(exist_ok=True)
    files = sorted((ROOT / "concat").glob("*.npy"))
    with Pool(args.workers, initializer=init) as pool:
        out = [o for o in pool.imap(analyse, files, chunksize=2) if o is not None]
    df = pd.concat([o[0] for o in out], ignore_index=True)
    peths = {k: np.concatenate([o[1][k] for o in out]) for k in out[0][1]}

    cv = np.load(ROOT / "concat_cvTrue.npy", allow_pickle=True).flat[0]
    pos = np.empty(len(cv["isort"]), int)
    pos[np.asarray(cv["isort"])] = np.arange(len(cv["isort"]))
    info = pd.DataFrame(dict(uuid=np.asarray(cv["uuids"]).astype(str),
                             rm_cluster=np.asarray(cv["rm_labels"]), rm_position=pos))
    df = df.merge(info, on="uuid", how="left")
    df.to_csv(RES / "neurons.csv", index=False)
    np.savez_compressed(RES / "peths.npz", uuid=df.uuid.to_numpy(), **peths)
    print(f"{len(df):,} neurons from {df.eid.nunique()} sessions -> {RES}")


if __name__ == "__main__":
    main()
