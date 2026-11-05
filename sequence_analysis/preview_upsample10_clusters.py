#!/usr/bin/env python3
"""Fig. 5a, b at large size with the 100 Rastermap cluster numbers of the
grid_upsample=10 fit (results/rastermap_upsample10.npz, preview_upsample.py) on the
y axis, to choose groups by cluster number. Each number sits at its cluster's median
row; thin lines are midway between consecutive cluster medians. Display as Fig. 5a, b.
Saves rastermap_previews/upsample10_clusters.png/.pdf.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import blended_transform_factory

from seq_common import PREV, OUT, dmn_bwm, use_private_base
from seq_rasters import MM, configure_style

ROW_BLOCK = 5


def thresholded(X, q=80, gamma=0.5):
    med = np.median(X, 1, keepdims=True)
    hi = np.percentile(X, 99, 1, keepdims=True)
    R = np.clip((X - med) / (hi - med + 1e-6), 0, 1)
    return np.where(R >= np.percentile(R, q, 1, keepdims=True), R, 0) ** gamma


def image(M):
    n = M.shape[0] // ROW_BLOCK * ROW_BLOCK
    return 1 - M[:n].reshape(-1, ROW_BLOCK, M.shape[1]).mean(1)


def main():
    configure_style()
    d = use_private_base()
    r = d.regional_group("rm", vers="concat", cv=True)
    f = np.load(OUT / "results/rastermap_upsample10.npz", allow_pickle=True)
    order, labels = np.asarray(f["isort"]), np.asarray(f["labels"])
    L = labels[order]
    med = np.array([np.median(np.flatnonzero(L == c)) for c in range(labels.max() + 1)])
    cut = (np.sort(med)[1:] + np.sort(med)[:-1]) / 2
    n_bins = sum(r["len"].values())
    fig = plt.figure(figsize=(183 * MM, 290 * MM))
    for k, (key, title) in enumerate((("concat_z_train", "train: odd trials (Rastermap fit, grid_upsample=10)"),
                                      ("concat_z", "test: even trials, same order"))):
        ax = fig.add_axes([0.06 + k * 0.48, 0.02, 0.40, 0.9])
        ax.imshow(image(thresholded(np.asarray(r[key], np.float32)[order])), cmap="gray", vmin=0, vmax=1,
                  aspect="auto", interpolation="antialiased", rasterized=True,
                  extent=(0, n_bins, order.size, 0))
        for y in cut:
            ax.axhline(y, color="tab:red", lw=0.15, alpha=0.6)
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        start = 0
        for seg, n in r["len"].items():
            ax.text(start + n / 2, 1.003, r["peth_dict"][seg], transform=trans, rotation=65, ha="left",
                    va="bottom", rotation_mode="anchor", fontsize=4.3, clip_on=False)
            start += n
            if start < n_bins:
                ax.axvline(start, color="0.5", lw=0.3, ls=(0, (2, 2)))
        ax.set_xlim(0, n_bins); ax.set_ylim(order.size, 0); ax.set_xticks([])
        ax.set_yticks(med, [str(c) for c in range(med.size)], fontsize=3.6)
        ax.tick_params(axis="y", length=1, pad=1)
        sec = ax.secondary_yaxis("right")
        sec.set_yticks([t * 1000 for t in range(0, 55, 5)], [f"{t}k" for t in range(0, 55, 5)], fontsize=4)
        ax.set_ylabel("Rastermap cluster (grid_upsample=10 fit), at its median row", fontsize=5)
        for side in ("top", "bottom"):
            ax.spines[side].set_visible(False)
        ax.set_title(title, fontsize=6, pad=30)
    out = PREV / "upsample10_clusters"
    fig.savefig(out.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), dpi=600, facecolor="white")
    print(f"Saved {out}.png/.pdf")


if __name__ == "__main__":
    main()
