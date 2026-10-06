#!/usr/bin/env python3
"""Fig. 3d-e point clouds rendered from the data (all trials, cv=False, as in a-c).

d: UMAP of the feature vectors and 3D anatomical positions, coloured by Beryl region.
e: the same, coloured by the 25 k-means clusters.
Uses dmn_bwm.plot_dim_reduction / plot_xyz and writes panel_{d,e}_{umap,xyz}.png
into this folder; regenerate_from_data.py --pointclouds data places them (the
default, --pointclouds source, uses the images extracted from the earlier
published figure, source_alternate_*.jpg).
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fig3_common import OUT, use_private_base

DPI = 600


def save(fig, name):
    fig.savefig(OUT / name, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {OUT / name}")


def umap(d, mapping, name):
    fig, ax = plt.subplots(figsize=(8, 5.2))
    d.plot_dim_reduction(algo="umap_z", mapping=mapping, vers="concat", nclus=25,
                         nclus_rm=100, cv=False, ax=ax, ds=0.5, exa=False)
    ax.set_axis_off()
    ax.set_title("")
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()
    fig.subplots_adjust(0, 0, 1, 1)
    save(fig, name)


def xyz(d, mapping, name):
    fig = plt.figure(figsize=(7.2, 6.2))
    ax = fig.add_subplot(111, projection="3d")
    d.plot_xyz(mapping=mapping, vers="concat", nclus=25, nclus_rm=100,
               cv=False, ax=ax, axoff=True, exa=False)
    ax.set_title("")
    fig.subplots_adjust(0, 0, 1, 1)
    save(fig, name)


def main():
    d = use_private_base()
    umap(d, "Beryl", "panel_d_umap.png")
    xyz(d, "Beryl", "panel_d_xyz.png")
    umap(d, "kmeans", "panel_e_umap.png")
    xyz(d, "kmeans", "panel_e_xyz.png")


if __name__ == "__main__":
    main()
