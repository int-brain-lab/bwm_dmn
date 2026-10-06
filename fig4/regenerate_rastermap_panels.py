#!/usr/bin/env python3
"""Render the two Rastermap images of Fig. 4h with dmn_bwm.plot_rastermap.

left:  responses reconstructed from the real coefficients (syn_control=True)
right: synthetic responses B @ V (syn_control=False)
--sort rastermap (manuscript): each is sorted by Rastermap fitted on the displayed
responses; the RGBA image plot_rastermap draws is captured.
--sort kmeans: canonical k-means sorting (Fig. 2a, 25 clusters fit on the real
responses). Left: each row is its own neuron's cluster (ties in stack order).
Right: synthetic neurons have no identity, so each is assigned to the nearest of
the same 25 centroids (KMeans.predict). Same display as plot_rastermap
(1 - clip(z, 0, 2) / 2, grey on white).
Both are averaged over blocks of ROW_BLOCK neurons and saved as PNG.
"""

import matplotlib

matplotlib.use("Agg")  # plot_rastermap calls plt.ion(); never open windows
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from compute_synthetic import load
from fig4_common import OUT, kmeans_canonical, nearest_centroid, use_private_base

ROW_BLOCK = 5


def capture(d, syn_control, nclus):
    d.plot_rastermap(vers="concat", feat="concat_z", mapping="kmeans", nclus=nclus,
                     nclus_s=nclus, synthetic=True, syn_control=syn_control,
                     sort_method="rastermap", bg=False, img_only=True, cv=False,
                     zsc=True, vmax=2, bounds=False, rerun=False, clsfig=False)
    fig = plt.gcf()
    rgba = np.asarray(fig.axes[0].images[0].get_array(), dtype=np.float32)
    plt.close(fig)
    n = rgba.shape[0] // ROW_BLOCK * ROW_BLOCK
    return np.clip(rgba[:n].reshape(-1, ROW_BLOCK, *rgba.shape[1:]).mean(axis=1)[..., :3], 0, 1)


def block_rows(rgb):
    n = rgb.shape[0] // ROW_BLOCK * ROW_BLOCK
    return np.clip(rgb[:n].reshape(-1, ROW_BLOCK, *rgb.shape[1:]).mean(axis=1), 0, 1)


def capture_kmeans(d, syn_control, nclus, labels, centroids):
    r = load(syn_control, nclus)
    X = np.asarray(r["concat_zs"], dtype=np.float32)
    if syn_control:  # row i of the reconstruction is stack neuron C_rows[i]
        rows = np.asarray(r["C_rows"], dtype=int)
        order = np.lexsort((rows, labels[rows]))
    else:
        order = np.argsort(nearest_centroid(X, centroids), kind="stable")
    gray = 1 - np.clip(X[order], 0, 2) / 2
    return np.repeat(block_rows(gray)[..., None], 3, axis=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nclus", type=int, default=100, help="k-means basis size")
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--sort", choices=("rastermap", "kmeans"), default="rastermap")
    args = parser.parse_args()
    d = use_private_base()
    if args.sort == "kmeans":
        labels, centroids = kmeans_canonical(d)
    for control, name in ((True, "real_reconstruction"), (False, "synthetic")):
        path = args.out_dir / f"panel_h_{args.sort}_{name}.png"
        if args.sort == "kmeans":
            image = capture_kmeans(d, control, args.nclus, labels, centroids)
        else:
            image = capture(d, control, args.nclus)
        plt.imsave(path, image)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
