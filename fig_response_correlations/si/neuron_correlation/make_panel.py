#!/usr/bin/env python3
"""Panel: cross-validated neuron x neuron correlation matrix (results/xcorr.npz),
Rastermap order of Fig. 4a, b. Left: all 54,569 neurons (means over 27 x 27 neuron
blocks). Right: the dense-tiling neurons at full resolution (exact entries).
Writes neuron_correlation.pdf/.png (183 mm wide).
"""

import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "sequence_analysis"))
import latency_tiling_groups as G  # noqa: E402

MM = 1 / 25.4
VMAX = 0.3
COL = dict(dense="#e8a33d", sparse="#3d7fc1")


def style():
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "Arial"],
                         "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans",
                         "font.size": 6, "axes.linewidth": 0.4, "xtick.major.width": 0.4,
                         "ytick.major.width": 0.4, "xtick.major.size": 2, "ytick.major.size": 2,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})


def group_bar(ax, rows, colour, scale, x0, w, label=None):
    """Mark the row ranges of a group (rows in matrix row units / scale) at x0..x0+w."""
    r = np.sort(rows)
    br = np.flatnonzero(np.diff(r) > 1)
    starts, ends = np.r_[r[0], r[br + 1]], np.r_[r[br], r[-1]]
    for s, e in zip(starts, ends):
        ax.add_patch(plt.Rectangle((x0, s / scale), w, (e - s + 1) / scale, color=colour,
                                   clip_on=False, lw=0))


def draw(fig, rect_a, rect_b, rect_ca, rect_cb, la="a", lb="b",
         title_a="all 54,569 neurons, odd vs even trials",
         title_b="dense tiling, clusters 21–28", compact=False):
    """Draw panel a (all neurons, block means) and b (dense tiling, exact) into fig."""
    R = np.load(HERE / "results" / "xcorr.npz")
    M, N, L = R["mean"], int(R["N"]), R["labels"]
    med, cut = G.cluster_bounds(L)
    dense_rows, sparse_rows = R["dense_rows"], R["sparse_rows"]
    ax = fig.add_axes(rect_a)
    im = ax.imshow(M, cmap="RdBu_r", vmin=-VMAX, vmax=VMAX, interpolation="none",
                   extent=(0, N, N, 0))
    ticks = list(range(0, 100, 10))
    ax.set_yticks(med[ticks]); ax.set_yticklabels(ticks)
    ax.set_xticks(med[ticks]); ax.set_xticklabels(ticks)
    ax.set_ylabel("neuron (Rastermap cluster)")
    ax.set_xlabel("neuron (Rastermap cluster)")
    ax.set_title(title_a, pad=4)
    if compact:
        ticks = list(range(0, 100, 10))
        ax.set_yticks(med[ticks]); ax.set_yticklabels(ticks)
        ax.set_xticks(med[ticks]); ax.set_xticklabels(ticks)
    for g, rows in (("dense", dense_rows), ("sparse", sparse_rows)):
        group_bar(ax, rows, COL[g], 1, N * (1.015 if g == "dense" else 1.035), N * 0.015)
    if not compact:
      ax.text(N * 1.0225, -N * 0.01, "dense", color=COL["dense"], rotation=90, ha="center",
            va="bottom", fontsize=4.5)
      ax.text(N * 1.0425, -N * 0.01, "sparse", color=COL["sparse"], rotation=90, ha="center",
            va="bottom", fontsize=4.5)
    d0, d1 = dense_rows.min(), dense_rows.max()
    ax.add_patch(plt.Rectangle((d0, d0), d1 - d0, d1 - d0, fill=False, ec="k", lw=0.5))
    if compact:
        ax.text(-0.17, 1.06, la, transform=ax.transAxes, fontsize=7, fontweight="bold", va="top")
    else:
        ax.text(-0.16, 1.04, la, transform=ax.transAxes, fontsize=8, fontweight="bold")

    ax2 = fig.add_axes(rect_b)
    D = R["dense"]
    n = len(D)
    # exact 3 x 3 block means: ~1,400 px over the 66 mm panel (~540 dpi), keeps the PDF small
    f = 3
    m = n // f
    Ds = D[:m * f, :m * f].reshape(m, f, m, f).mean((1, 3))
    ax2.imshow(Ds, cmap="RdBu_r", vmin=-0.8, vmax=0.8, interpolation="none",
               extent=(0, m * f, m * f, 0))
    Ld = L[dense_rows]
    for c in np.unique(Ld)[1:]:
        p = np.flatnonzero(Ld == c)[0]
        ax2.axhline(p, color="0.3", lw=0.25); ax2.axvline(p, color="0.3", lw=0.25)
    mids = [np.median(np.flatnonzero(Ld == c)) for c in np.unique(Ld)]
    ax2.set_yticks(mids); ax2.set_yticklabels(np.unique(Ld), fontsize=5)
    ax2.set_xticks([0, n]); ax2.set_xticklabels([0, f"{n:,}"])
    ax2.set_xlabel("neuron (full resolution)" if compact else "dense-tiling neuron (full resolution)")
    ax2.set_ylabel("cluster")
    ax2.set_title(title_b, pad=4)
    if compact:
        ax2.text(-0.15, 1.06, lb, transform=ax2.transAxes, fontsize=7, fontweight="bold", va="top")
    else:
        ax2.text(-0.2, 1.04, lb, transform=ax2.transAxes, fontsize=8, fontweight="bold")
    for rect, v, lab in ((rect_ca, VMAX, f"$r$ ({la}, block mean)"), (rect_cb, 0.8, f"$r$ ({lb})")):
        cax = fig.add_axes(rect)
        cb = fig.colorbar(mpl.cm.ScalarMappable(mpl.colors.Normalize(-v, v), "RdBu_r"), cax=cax,
                          orientation="horizontal", ticks=[-v, 0, v])
        cb.set_label(lab, fontsize=5, labelpad=1); cb.outline.set_linewidth(0.3)
        cb.ax.tick_params(labelsize=5, pad=1)


def main():
    style()
    fig = plt.figure(figsize=(183 * MM, 92 * MM))
    draw(fig, [0.07, 0.1, 0.44, 0.8], [0.6, 0.30, 0.3, 0.3 * 183 / 92],
         [0.6, 0.14, 0.12, 0.018], [0.78, 0.14, 0.12, 0.018])
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"neuron_correlation.{ext}", dpi=400)
    print("saved")


if __name__ == "__main__":
    main()
