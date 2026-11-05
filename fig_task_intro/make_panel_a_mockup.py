#!/usr/bin/env python3
"""Fig. 1a mockup: regional specialization (hypothesis) vs. what we observe.

Sagittal brain outline (iblatlas) with the six example regions of Fig. 3c at the
sagittal projection (AP, DV) of their neurons' centroid. Each region is a disc of N_DOTS
neurons coloured by k-means cluster (25 clusters, all trials, the colours of
Figs. 2-3):
  left  "specialized": every neuron has the region's most over-represented cluster
        (relative to all neurons; distinct across regions)
  right "observed": neurons sampled from the region's actual cluster proportions
Writes elements/a_mockup_diversity.svg/.png.
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from iblatlas.atlas import AllenAtlas
from iblatlas.regions import BrainRegions

ROOT = Path(os.environ.get("DMN_DATA", Path.home() / "dmn"))
OUT = Path(__file__).resolve().parent / "elements"
MM = 1 / 25.4
REGIONS = ["MOp", "CP", "CA1", "ZI", "MRN", "CUL4 5"]
N_DOTS, RADIUS = 70, 0.62e-3  # dots per region, disc radius (m)
NUDGE = {"MRN": (-0.2e-3, -0.45e-3)}  # (AP, DV) shift (m) so MRN does not touch CA1


def cluster_colors(n=25):
    """dmn_bwm's k-means colours: tab20 by cluster index, cluster index 1 pink."""
    cmap = mpl.colormaps["tab20"]
    cols = [cmap(i % 20) for i in range(n)]
    cols[1] = mpl.colors.to_rgba("pink")
    return np.array(cols)


def disc_points(rng, n, r):
    """n roughly evenly spread points in a disc (sunflower layout, jittered)."""
    i = np.arange(n) + 0.5
    rad = r * np.sqrt(i / n)
    th = i * np.pi * (3 - np.sqrt(5))
    p = np.c_[rad * np.cos(th), rad * np.sin(th)]
    return p + rng.normal(0, r * 0.03, p.shape)


def main():
    OUT.mkdir(exist_ok=True)
    br = BrainRegions()
    r = np.load(ROOT / "concat_cvFalse.npy", allow_pickle=True).flat[0]
    km = np.load(ROOT / "kmeans_concat_cvFalse_n25_nclusrm100_zsc1.npy", allow_pickle=True).flat[0]
    labels = np.asarray(km["kmeans_labels"], int)
    beryl = np.array(br.id2acronym(r["ids"], mapping="Beryl"))
    xyz = np.asarray(r["xyz"], float)
    cols = cluster_colors()

    ba = AllenAtlas(25)
    inside = (ba.label > 0).any(1)            # (AP, DV) sagittal silhouette
    ys, zs = ba.bc.yscale, ba.bc.zscale       # AP, DV (m)

    # hypothesis colours: each region's most over-represented cluster, distinct
    overall = np.bincount(labels, minlength=25) / labels.size
    taken, special = set(), {}
    for reg in REGIONS:
        frac = np.bincount(labels[beryl == reg], minlength=25) / (beryl == reg).sum()
        for c in np.argsort(-(frac / overall)):
            if c not in taken:
                special[reg] = c; taken.add(c); break

    rng = np.random.default_rng(0)
    fig, axs = plt.subplots(1, 2, figsize=(80 * MM, 34 * MM))
    titles = ("specialized regions?", "observed: every region\ncontains all response types")
    for k, ax in enumerate(axs):
        ax.contour(ys, zs, inside.T.astype(float), levels=[0.5], colors="0.25", linewidths=0.5)
        for reg in REGIONS:
            m = beryl == reg
            c = np.nanmean(xyz[m], 0)[1:] + np.array(NUDGE.get(reg, (0, 0)))
            counts = np.bincount(labels[m], minlength=25)
            if k == 0:
                lab = np.full(N_DOTS, special[reg])
            else:
                lab = rng.choice(25, N_DOTS, p=counts / counts.sum())
            p = c + disc_points(rng, N_DOTS, RADIUS)
            ax.scatter(p[:, 0], p[:, 1], s=2.2, c=cols[lab], linewidths=0, zorder=3)
            ax.text(c[0], c[1] - RADIUS * 1.35, reg.replace("CUL4 5", "CUL4,5"),
                    ha="center", va="top", fontsize=5, zorder=4)
        ax.set_aspect("equal")
        ax.set_xlim(ys.max(), ys.min())   # anterior to the left
        ax.set_ylim(zs.min(), zs.max())
        ax.set_axis_off()
        ax.set_title(titles[k], fontsize=6, pad=2)
    fig.subplots_adjust(0.01, 0.01, 0.99, 0.84, wspace=0.05)
    for ext, kw in (("svg", {}), ("png", {"dpi": 400})):
        fig.savefig(OUT / f"a_mockup_diversity.{ext}", facecolor="white", **kw)
    print(f"Saved {OUT / 'a_mockup_diversity'}.svg/.png")


if __name__ == "__main__":
    mpl.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                         "svg.fonttype": "none"})
    main()
