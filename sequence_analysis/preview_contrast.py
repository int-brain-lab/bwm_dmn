#!/usr/bin/env python3
"""Preview display mappings that make sparse sequence patterns more visible.

Same data and order as Fig. 3 a/b (odd-trial Rastermap fit; X_odd train, X_even
test). Display-only, non-linear mappings per neuron (row):
  linear : z-scored response clipped to [0, 1.5] (as the Methods / current panels)
  robust : (x - row median) / (row 99th percentile - row median), clipped to [0, 1],
           then gamma 0.6 (boosts faint but consistent peaks of sparse neurons)
Full rasters and a zoom on Rastermap clusters ZOOM.
"""

import matplotlib.pyplot as plt
import numpy as np

from seq_common import OUT, PREV, use_private_base

ROW_BLOCK = 5
ZOOM = (55, 75)


def robust(X, gamma=0.6):
    med = np.median(X, axis=1, keepdims=True)
    hi = np.percentile(X, 99, axis=1, keepdims=True)
    return np.clip((X - med) / (hi - med + 1e-6), 0, 1) ** gamma


def linear(X):
    return np.clip(X, 0, 1.5) / 1.5


def blocks(M, k):
    n = M.shape[0] // k * k
    return M[:n].reshape(-1, k, M.shape[1]).mean(1)


def main():
    d = use_private_base()
    r = d.regional_group("rm", vers="concat", cv=True)
    order = np.asarray(r["isort"])
    clusters = np.asarray(r["acs"])[order]
    data = {"train (odd)": np.asarray(r["concat_z_train"], np.float32)[order],
            "test (even)": np.asarray(r["concat_z"], np.float32)[order]}
    zoom = np.flatnonzero((clusters >= ZOOM[0]) & (clusters <= ZOOM[1]))
    bounds = np.r_[0, np.cumsum(list(r["len"].values()))]

    for tag, rows, k in (("full", slice(None), ROW_BLOCK), ("zoom", zoom, 1)):
        fig, axs = plt.subplots(2, 2, figsize=(12, 11), sharex=True, sharey=True)
        for j, (name, X) in enumerate(data.items()):
            for i, (mname, fn) in enumerate((("linear", linear), ("robust + gamma 0.6", robust))):
                ax = axs[i, j]
                M = blocks(fn(X[rows]), k)
                ax.imshow(1 - M, cmap="gray", vmin=0, vmax=1, aspect="auto",
                          interpolation="antialiased", extent=(0, X.shape[1], M.shape[0] * k, 0))
                for b in bounds[1:-1]:
                    ax.axvline(b, color="tab:red", lw=0.3, alpha=0.6)
                cl = clusters[rows]
                for y in np.flatnonzero(np.diff(cl)) + 1:
                    ax.axhline(y, color="tab:blue", lw=0.3, alpha=0.6)
                if tag == "zoom":
                    edges = np.r_[0, np.flatnonzero(np.diff(cl)) + 1, len(cl)]
                    for lo, hi in zip(edges[:-1], edges[1:]):
                        ax.text(X.shape[1] * 1.005, (lo + hi) / 2, str(cl[lo]), fontsize=6, va="center")
                ax.set_title(f"{name}, {mname}", fontsize=9)
                ax.set_xticks([])
        title = ("all neurons" if tag == "full" else
                 f"Rastermap clusters {ZOOM[0]}-{ZOOM[1]} ({len(zoom):,} neurons, no row averaging)")
        fig.suptitle(f"{title}; red: PETH-type boundaries, blue: cluster boundaries", fontsize=10)
        fig.tight_layout()
        out = PREV / f"preview_contrast_{tag}.png"
        fig.savefig(out, dpi=170)
        plt.close(fig)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
