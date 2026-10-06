#!/usr/bin/env python3
"""Fig. 3a (candidate): neurons sorted by k-means cluster (all trials, 54,719
neurons), each row's background coloured by the neuron's Beryl region.

Uses dmn_bwm.plot_rastermap(mapping="kmeans", sort_method="acs", bg=True), whose
row backgrounds come from r["cols"]; within this call r["cols"] is replaced by
each neuron's Beryl colour (dmn_bwm.pal). The captured image is saved as PNG
(blocks of ROW_BLOCK neurons averaged) together with the sorted cluster labels,
plus a preview with axes.
"""

import contextlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import blended_transform_factory

from fig3_common import OUT, use_private_base

ROW_BLOCK = 5


@contextlib.contextmanager
def beryl_row_colours(d):
    original = d.regional_group

    def regional_group(mapping, *args, **kwargs):
        r = original(mapping, *args, **kwargs)
        if mapping == "kmeans":
            r = dict(r, cols=np.array([d.pal[b] if b in d.pal else (0.6, 0.6, 0.6, 1.0)
                                       for b in np.asarray(r["Beryl"])]))
        return r

    d.regional_group = regional_group
    try:
        yield
    finally:
        d.regional_group = original


def main():
    d = use_private_base()
    with beryl_row_colours(d):
        d.plot_rastermap(vers="concat", feat="concat_z", mapping="kmeans", nclus=25,
                         sort_method="acs", bg=True, bg_bright=0.99, img_only=True,
                         cv=False, zsc=True, vmax=2, bounds=False, rerun=False, clsfig=False)
        fig = plt.gcf()
        rgba = np.asarray(fig.axes[0].images[0].get_array(), dtype=np.float32)
        plt.close(fig)
    r = d.regional_group("kmeans", vers="concat", cv=False, nclus=25)
    clusters = np.asarray(r["acs"])[np.argsort(r["acs"], kind="stable")]

    n = rgba.shape[0] // ROW_BLOCK * ROW_BLOCK
    image = np.clip(rgba[:n].reshape(-1, ROW_BLOCK, *rgba.shape[1:]).mean(1)[..., :3], 0, 1)
    plt.imsave(OUT / "panel_a_kmeans_beryl_background.png", image)
    np.save(OUT / "panel_a_kmeans_clusters.npy", clusters)

    # Preview with axes, cluster boundaries and numbers.
    fig, ax = plt.subplots(figsize=(4, 5.5))
    ax.imshow(image, aspect="auto", interpolation="antialiased",
              extent=(0, rgba.shape[1], len(clusters), 0))
    edges = np.flatnonzero(np.diff(clusters)) + 1
    for y in edges:
        ax.axhline(y, color="k", lw=0.4)
    bounds = np.r_[0, edges, len(clusters)]
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        ax.text(1.01, (lo + hi) / 2, str(int(clusters[lo]) + 1), transform=trans,
                fontsize=6, va="center")
    start = 0
    for seg, length in r["len"].items():
        start += length
        ax.axvline(start, color="0.4", lw=0.3, ls=":")
    ax.set_xticks([]); ax.set_ylabel("neurons (sorted by k-means cluster)")
    ax.set_title("Fig. 3a candidate: k-means order, Beryl background", fontsize=8)
    fig.savefig(OUT / "panel_a_kmeans_preview.png", dpi=200, bbox_inches="tight")
    print("Saved panel_a_kmeans_beryl_background.png and panel_a_kmeans_preview.png")


if __name__ == "__main__":
    main()
