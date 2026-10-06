#!/usr/bin/env python3
"""Preview: Rastermap with grid_upsample=10 (otherwise Methods parameters), fit on the
odd-trial feature vectors (X_odd, all 21 PETH types), order applied to the even
trials (X_even). Display as the canonical preview (z clipped to [0, 1.5], grey on
white), no cluster boundaries. Saves rastermap_previews/preview_upsample10.png and
results/rastermap_upsample10.npz (isort).
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import blended_transform_factory
from rastermap import Rastermap

from seq_common import PREV, OUT, dmn_bwm, use_private_base
from seq_rasters import MM, configure_style, label, scale_bar

ROW_BLOCK = 5


def image(X):
    M = np.clip(X, 0, 1.5) / 1.5
    n = M.shape[0] // ROW_BLOCK * ROW_BLOCK
    return 1 - M[:n].reshape(-1, ROW_BLOCK, M.shape[1]).mean(1)


def main():
    configure_style()
    d = use_private_base()
    r = d.regional_group("rm", vers="concat", cv=True)
    Xo, Xe = np.asarray(r["X_odd"], np.float32), np.asarray(r["X_even"], np.float32)
    model = Rastermap(n_PCs=200, n_clusters=100, locality=0.75, grid_upsample=10,
                      time_lag_window=5, bin_size=1).fit(Xo)
    order = np.asarray(model.isort)
    np.savez(OUT / "results/rastermap_upsample10.npz", isort=order, uuids=np.asarray(r["uuids"]))

    fig = plt.figure(figsize=(183 * MM, 120 * MM))
    n_bins = Xo.shape[1]
    for k, (X, title) in enumerate(((Xo, "train: odd trials, sorted by their Rastermap fit (grid_upsample=10)"),
                                    (Xe, "test: even trials, same order (cross-validated)"))):
        ax = fig.add_axes([0.07 + k * 0.48, 0.05, 0.42, 0.78])
        ax.imshow(image(X[order]), cmap="gray", vmin=0, vmax=1, aspect="auto",
                  interpolation="antialiased", rasterized=True, extent=(0, n_bins, len(order), 0))
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        start = 0
        for seg, n in r["len"].items():
            ax.text(start + n / 2, 1.01, r["peth_dict"][seg], transform=trans, rotation=65, ha="left",
                    va="bottom", rotation_mode="anchor", fontsize=4.3, clip_on=False)
            start += n
            if start < n_bins:
                ax.axvline(start, color="0.5", lw=0.35, ls=(0, (2, 2)))
        ax.set_xlim(0, n_bins); ax.set_xticks([])
        ax.set_yticks([t * 1000 for t in (0, 10, 20, 30, 40, 50)], ["0", "10", "20", "30", "40", "50"])
        if k == 0:
            ax.set_ylabel(r"cell index ($\times\,10^3$)")
        for side in ("top", "right", "bottom"):
            ax.spines[side].set_visible(False)
        ax.set_title(title, fontsize=6, pad=34)
        scale_bar(ax, 0.2 * dmn_bwm.c_sec, "200 ms")
        label(fig, ax, "ab"[k], dy=0.09)
    out = PREV / "preview_upsample10.png"
    fig.savefig(out, dpi=300, facecolor="white")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
