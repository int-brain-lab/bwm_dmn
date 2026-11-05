#!/usr/bin/env python3
"""Fig. 1 draft panel e: "Random or structured mixed selectivity?" (frames Fig. 4).

Top: one neuron's feature vector as a weighted mix of two response types, the real
mean feature vectors (concat_z, all trials, concat_cvFalse.npy) of k-means clusters 21
(stimulus, blue) and 5 (movement, green), the same two clusters and colours as the
callouts of panel d; the mix is w1 * stimulus + w2 * movement.
Bottom: schematic of each neuron's two weights (one dot per neuron, illustrative
points, not data): random mixing = weights drawn independently (shapeless cloud);
structured mixing = recurring combinations (clumps). Fig. 4 tests this with 100
response components (real coefficients vs independently resampled ones).
Writes elements/panel_mixing.svg/.pdf/.png at final size.
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from make_panel_a import cluster_colors

ROOT = Path(os.environ.get("DMN_DATA", Path.home() / "dmn"))
EL = Path(__file__).resolve().parent / "elements"
MM = 1 / 25.4
W, H = 46.0, 36.0  # final size (mm)
TITLE = "Random or structured mixed selectivity?"
STIM, MOVE = 21, 5  # k-means clusters (1-based, as Fig. 2)
W1, W2 = 0.7, 0.45  # weights of the example neuron


def main():
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans"],
                         "font.size": 5.5, "axes.linewidth": 0.5, "svg.fonttype": "none",
                         "pdf.fonttype": 42, "mathtext.fontset": "custom",
                         "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic"})
    r = np.load(ROOT / "concat_cvFalse.npy", allow_pickle=True).flat[0]
    labels = np.asarray(np.load(ROOT / "kmeans_concat_cvFalse_n25_nclusrm100_zsc1.npy",
                                allow_pickle=True).flat[0]["kmeans_labels"], int)
    X = np.asarray(r["concat_z"], np.float32)
    cols = cluster_colors()

    def mean_vec(c):
        v = X[labels == c - 1].mean(0)
        v = v[: v.size // 4 * 4].reshape(-1, 4).mean(1)
        return (v - v.min()) / (v.max() - v.min())

    s, m = mean_vec(STIM), mean_vec(MOVE)
    mix = W1 * s + W2 * m
    cs, cm = cols[STIM - 1], cols[MOVE - 1]

    fig = plt.figure(figsize=(W * MM, H * MM))
    fig.text(0.5, 0.975, TITLE, ha="center", va="top", fontsize=6)
    # top: w1 * stimulus + w2 * movement = neuron
    y0, h = 0.6, 0.22
    for x, v, c, lab in ((0.02, s, cs, "stimulus"), (0.36, m, cm, "movement"), (0.7, mix, "k", "neuron")):
        ax = fig.add_axes([x + 0.06, y0, 0.22, h])
        ax.plot(v, color=c, lw=0.5)
        ax.set_axis_off()
        ax.set_ylim(-0.05, 1.05)
        ax.text(0.5, -0.12, lab, transform=ax.transAxes, ha="center", va="top", fontsize=5, color=c)
    # plain Unicode (math-text glyphs do not survive the SVG -> Inkscape export)
    for x, t in ((0.02, "w\u2081 \u00d7"), (0.32, "+ w\u2082 \u00d7"), (0.695, "=")):
        fig.text(x, y0 + h / 2, t, ha="left", va="center", fontsize=6)
    # bottom: weights, one dot per neuron (schematic)
    rng = np.random.default_rng(3)
    rand = rng.uniform(0.05, 0.95, (70, 2))
    centres = np.array([[0.8, 0.15], [0.2, 0.8], [0.7, 0.7]])
    struct = np.concatenate([c + 0.05 * rng.standard_normal((23, 2)) for c in centres])
    for k, (pts, lab) in enumerate(((rand, "random"), (struct, "structured"))):
        ax = fig.add_axes([0.14 + 0.47 * k, 0.1, 0.32, 0.3])
        ax.scatter(pts[:, 0], pts[:, 1], s=1.2, color="0.25", lw=0)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("w\u2081", labelpad=1, color=cs); ax.set_ylabel("w\u2082", labelpad=1, color=cm)
        ax.set_title(lab, fontsize=5.5, pad=1.5)
    EL.mkdir(exist_ok=True)
    for ext, kw in (("svg", {}), ("pdf", {}), ("png", {"dpi": 600})):
        fig.savefig(EL / f"panel_mixing.{ext}", facecolor="white", **kw)
    print(f"Saved {EL / 'panel_mixing'}.svg/.pdf/.png ({W:.0f} x {H:.0f} mm)")


if __name__ == "__main__":
    main()
