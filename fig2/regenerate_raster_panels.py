#!/usr/bin/env python3
"""Render the raster image of Fig. 2a (neurons sorted by k-means cluster) with
dmn_bwm.plot_rastermap(sort_method="acs").

The RGBA image that plot_rastermap draws is captured unchanged, averaged over
blocks of ROW_BLOCK neurons for a manageable file size, and saved as PNG
together with the row metadata (cluster order and boundaries) for assembly.
"""

import matplotlib

matplotlib.use("Agg")  # plot_rastermap calls plt.ion(); never open windows
import matplotlib.pyplot as plt
import numpy as np

from fig2_common import OUT, use_private_base

ROW_BLOCK = 5


def capture_raster(d, cv, **kwargs):
    d.plot_rastermap(vers="concat", feat="concat_z", img_only=True, cv=cv,
                     zsc=True, vmax=2, bounds=False, rerun=False, clsfig=False,
                     **kwargs)
    fig = plt.gcf()
    rgba = np.asarray(fig.axes[0].images[0].get_array(), dtype=np.float32)
    plt.close(fig)
    return rgba


def save_rows(rgba, path):
    n = rgba.shape[0] // ROW_BLOCK * ROW_BLOCK
    head = rgba[:n].reshape(-1, ROW_BLOCK, *rgba.shape[1:]).mean(axis=1)
    tail = rgba[n:].mean(axis=0, keepdims=True) if n < rgba.shape[0] else rgba[:0]
    image = np.clip(np.concatenate([head, tail])[..., :3], 0, 1)
    plt.imsave(path, image)
    print(f"Saved {path} {image.shape}")


def main():
    d = use_private_base()

    # Panel a: all 54,719 neurons (all trials) sorted by k-means cluster,
    # grey on white (cluster boundaries are drawn in make_figure2.py).
    save_rows(capture_raster(d, False, mapping="kmeans", sort_method="acs", nclus=25,
                             bg=False),  # grey on white, no cluster background colours
              OUT / "panel_d_kmeans_raster.png")
    rk = d.regional_group("kmeans", vers="concat", cv=False, nclus=25)
    clusters_d = np.asarray(rk["acs"])[np.argsort(rk["acs"], kind="stable")]

    np.savez(OUT / "raster_rows.npz", clusters_d=clusters_d)
    print(f"Saved {OUT / 'raster_rows.npz'}")


if __name__ == "__main__":
    main()
