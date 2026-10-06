#!/usr/bin/env python3
"""Rastermap images for Fig. 3: train (odd trials) and cross-validated test
(even trials), both in the order of the single Rastermap fit on the odd trials.

The CV stack (concat_cvTrue.npy) holds the odd/even trial averages and the
canonical fit (isort, rm_labels; n_PCs=200, n_clusters=100, locality=0.75,
grid_upsample=0, time_lag_window=5). dmn_bwm.plot_rastermap(mapping="rm", cv=True)
draws them with the Methods' display (grey on white, vmin=0, vmax=1.5); the RGBA
image it draws is captured, ROW_BLOCK neurons averaged, saved as PNG.
"""

import matplotlib.pyplot as plt
import numpy as np

from seq_common import OUT, PREV, use_private_base

ROW_BLOCK = 5
PANELS = {"train_odd": "concat_z_train", "test_even": "concat_z"}  # = X_odd, X_even


def capture(d, feat):
    d.plot_rastermap(vers="concat", feat=feat, mapping="rm", sort_method="rastermap",
                     bg=False, img_only=True, cv=True, zsc=True, vmax=1.5, bounds=False,
                     rerun=False, clsfig=False)
    fig = plt.gcf()
    rgba = np.asarray(fig.axes[0].images[0].get_array(), dtype=np.float32)
    plt.close(fig)
    n = rgba.shape[0] // ROW_BLOCK * ROW_BLOCK
    return np.clip(rgba[:n].reshape(-1, ROW_BLOCK, *rgba.shape[1:]).mean(1)[..., :3], 0, 1)


def main():
    d = use_private_base()
    for name, feat in PANELS.items():
        path = PREV / f"raster_{name}.png"
        plt.imsave(path, capture(d, feat))
        print(f"Saved {path}")
    r = d.regional_group("rm", vers="concat", cv=True)
    if r.get("cv_split") != "oddeven":
        raise RuntimeError("CV stack is not the odd/even split")
    clusters = np.asarray(r["acs"])[np.asarray(r["isort"])]
    np.save(PREV / "raster_clusters.npy", clusters)
    print(f"{clusters.size:,} neurons, {len(np.unique(clusters))} Rastermap clusters")


if __name__ == "__main__":
    main()
