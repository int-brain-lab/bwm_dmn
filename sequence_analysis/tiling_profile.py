#!/usr/bin/env python3
"""Where along the Fig. 5b order is there reproducible latency tiling?

Order: upsampled Rastermap fit on odd trials (results/rastermap_upsample10.npz).
In sliding bands of BAND rows (step STEP): Spearman rho between each neuron's peak
latency on odd and on even trials (mean over the six stimulus-aligned types, 0-150 ms
window), plus the band's median odd/even reliability over the whole feature vector.
Shuffle baseline: rho with even peaks permuted within the band (95th percentile).
Output: results/tiling_profile.png/.csv.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path.home() / "dmn"
RES = Path(__file__).resolve().parent / "results"
STIM = ["block_change_s", "stimLbLcL", "stimLbRcL", "stimRbRcR", "stimRbLcR", "mistake_s"]
BAND, STEP = 1000, 250
GROUPS = {"dense tiling": (11800, 16299, "#ff7f0e"), "sparse cells": (29600, 35599, "#1f77b4")}


def rowcorr(a, b):
    a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1) + 1e-12)


def main():
    r = np.load(ROOT / "concat_cvTrue.npy", allow_pickle=True).flat[0]
    order = np.load(RES / "rastermap_upsample10.npz", allow_pickle=True)["isort"]
    start = np.cumsum([0] + list(r["len"].values())); names = list(r["len"])
    Xo, Xe = np.asarray(r["X_odd"], np.float32)[order], np.asarray(r["X_even"], np.float32)[order]
    so = np.mean([Xo[:, start[names.index(t)]:start[names.index(t) + 1]] for t in STIM], 0)
    se = np.mean([Xe[:, start[names.index(t)]:start[names.index(t) + 1]] for t in STIM], 0)
    po, pe = so.argmax(1), se.argmax(1)
    rel = rowcorr(Xo, Xe)
    rng = np.random.default_rng(0)
    rows = []
    for lo in range(0, len(order) - BAND + 1, STEP):
        sl = slice(lo, lo + BAND)
        rho = spearmanr(po[sl], pe[sl]).statistic
        null = [spearmanr(po[sl], rng.permutation(pe[sl])).statistic for _ in range(100)]
        rows.append(dict(center=lo + BAND / 2, rho=rho, null95=np.percentile(null, 95),
                         reliability=np.median(rel[sl])))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "tiling_profile.csv", index=False)

    fig, axs = plt.subplots(1, 2, figsize=(5.5, 6), sharey=True)
    for ax, key, lab in ((axs[0], "rho", "stimulus-window peak latency,\nodd vs even (Spearman ρ)"),
                         (axs[1], "reliability", "median odd/even\nreliability (whole vector)")):
        ax.plot(df[key], df.center, color="k", lw=0.8)
        if key == "rho":
            ax.plot(df.null95, df.center, color="0.6", lw=0.6, ls="--", label="shuffle 95%")
            ax.legend(frameon=False, fontsize=6)
        for name, (lo, hi, col) in GROUPS.items():
            ax.axhspan(lo, hi, color=col, alpha=0.25, lw=0)
        ax.set_xlabel(lab, fontsize=7); ax.tick_params(labelsize=6)
        ax.spines[["top", "right"]].set_visible(False)
    axs[0].set_ylim(len(order), 0); axs[0].set_ylabel("cell index (order of Fig. 5b)", fontsize=7)
    for name, (lo, hi, col) in GROUPS.items():
        axs[1].text(axs[1].get_xlim()[1], (lo + hi) / 2, " " + name, color=col, fontsize=6, va="center")
    fig.tight_layout()
    fig.savefig(RES / "tiling_profile.png", dpi=220)
    for name, (lo, hi, _) in GROUPS.items():
        g = df[(df.center >= lo) & (df.center <= hi)]
        print(f"{name}: band rho {g.rho.mean():.2f} (shuffle 95% {g.null95.mean():.2f}), reliability {g.reliability.mean():.2f}")
    top = df.sort_values("rho", ascending=False).head(8)
    print("bands with the strongest latency replication (center row, rho):",
          [(int(c), round(v, 2)) for c, v in zip(top.center, top.rho)])
    print(f"Saved {RES / 'tiling_profile.png'}")


if __name__ == "__main__":
    main()
