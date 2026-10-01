#!/usr/bin/env python3
"""Render the raster images of Fig. 2d and 2f with dmn_bwm.plot_rastermap.

The RGBA image that plot_rastermap draws is captured unchanged, averaged over
blocks of ROW_BLOCK neurons for a manageable file size, and saved as PNG
together with the row metadata (cluster order and boundaries) for assembly.
"""

import matplotlib

matplotlib.use("Agg")  # plot_rastermap calls plt.ion(); never open windows
import matplotlib.pyplot as plt
import numpy as np

from fig2_common import OUT, F_BG, F_BG_BRIGHT, use_private_base

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

    # d: all 54,719 neurons (no trial split, as panels a-c) sorted by k-means
    # cluster, row background = cluster colour.
    save_rows(capture_raster(d, False, mapping="kmeans", sort_method="acs", nclus=25,
                             bg=True, bg_bright=0.99),
              OUT / "panel_d_kmeans_raster.png")
    rk = d.regional_group("kmeans", vers="concat", cv=False, nclus=25)
    clusters_d = np.asarray(rk["acs"])[np.argsort(rk["acs"], kind="stable")]

    # f: held-out half of 53,021 neurons in Rastermap order (fitted on the
    # training half) on a light background; 100 Rastermap clusters.
    save_rows(capture_raster(d, True, mapping="rm", sort_method="rastermap",
                             bg=F_BG, bg_bright=F_BG_BRIGHT),
              OUT / "panel_f_rastermap_raster.png")
    rr = d.regional_group("rm", vers="concat", cv=True)
    clusters_f = np.asarray(rr["acs"])[rr["isort"]]

    np.savez(OUT / "raster_rows.npz", clusters_d=clusters_d, clusters_f=clusters_f)
    print(f"Saved {OUT / 'raster_rows.npz'}")


if __name__ == "__main__":
    main()
