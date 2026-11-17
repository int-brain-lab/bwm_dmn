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
VMAX = 0.4  # shared by panels a and b
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


# task phases of the feature vector (Table 1 order): pre-stimulus (rest, blocks, quiescence),
# stimulus-aligned, before movement onset, after movement onset, feedback
PHASES = (("pre", 4), ("stim", 6), ("pre-mv", 7), ("mv", 2), ("fb", 2))
PHASE_FULL = ("pre-stimulus", "stimulus", "pre-movement", "movement", "feedback")


def phase_edges(lens):
    """Fractions (0-1) of the feature vector at the phase boundaries."""
    cum = np.r_[0, np.cumsum(lens)] / np.sum(lens)
    idx = np.r_[0, np.cumsum([n for _, n in PHASES])]
    return cum[idx]


def draw(fig, rect_a, rect_b, rect_ca, rect_cb, la="a", lb="b",
         title_a="all 54,569 neurons, odd vs even trials",
         title_b="dense tiling, clusters 21–28", compact=False, rect_tr=None, rect_tt=None):
    """Draw panel a (all neurons, block means) and b (dense tiling, exact) into fig."""
    R = np.load(HERE / "results" / "xcorr.npz")
    M, N, L = R["mean"], int(R["N"]), R["labels"]
    med, cut = G.cluster_bounds(L)
    dense_rows, sparse_rows = R["dense_rows"], R["sparse_rows"]
    ax = fig.add_axes(rect_a)
    # exact mean correlation per pair of Rastermap clusters (compute_cluster_means.py);
    # each cell spans its clusters' rows, so axes stay in the neuron index of b
    CM = np.load(HERE / "results" / "xcorr_clusters.npz")
    edges = CM["cut"].astype(float)
    ax.pcolormesh(edges, edges, CM["mean"], cmap="RdBu_r", vmin=-VMAX, vmax=VMAX,
                  rasterized=True, linewidth=0)
    ax.set_xlim(0, N); ax.set_ylim(N, 0); ax.set_aspect("equal")
    ticks = list(range(0, 100, 10))
    ax.set_yticks(med[ticks]); ax.set_yticklabels(ticks)
    ax.set_xticks(med[ticks]); ax.set_xticklabels(ticks)
    ax.set_ylabel("Rastermap cluster")
    ax.set_xlabel("Rastermap cluster")
    ax.set_title(title_a, pad=17)
    for g, rows in (("dense", dense_rows), ("sparse", sparse_rows)):
        group_bar(ax, rows, COL[g], 1, N * (1.015 if g == "dense" else 1.035), N * 0.015)
    if not compact:
      ax.text(N * 1.0225, -N * 0.01, "dense", color=COL["dense"], rotation=90, ha="center",
            va="bottom", fontsize=4.5)
      ax.text(N * 1.0425, -N * 0.01, "sparse", color=COL["sparse"], rotation=90, ha="center",
            va="bottom", fontsize=4.5)
    d0, d1 = dense_rows.min(), dense_rows.max()
    WB = np.load(HERE / "results" / "window.npz")  # pick_window.py
    r0, c0, Wn = int(WB["r0"]), int(WB["c0"]), int(WB["W"])
    r1, c1 = r0 + Wn - 1, c0 + Wn - 1
    ax.add_patch(plt.Rectangle((c0, r0), Wn, Wn, fill=False, ec="k", lw=0.6))
    ym = r0 + Wn / 2
    # line from the box to a label "b" at the left y axis
    ax.plot([0.035 * N, c0], [ym, ym], color="k", lw=0.4)
    ax.text(0.01 * N, ym, lb, fontsize=6, fontweight="bold", va="center", ha="left")
    # block boundaries: rows on the left y axis, columns on the top x axis (fanned, leaders)
    for y, dy, va in ((r0, -0.025 * N, "bottom"), (r1, 0.025 * N, "top")):
        ax.annotate(f"{y:,}", xy=(0, y), xytext=(-0.105 * N, y + dy), ha="right", va=va, fontsize=4.5,
                    annotation_clip=False, arrowprops=dict(arrowstyle="-", lw=0.4, color="k", shrinkA=0, shrinkB=0))
    for x, dx, ha in ((c0, -0.03 * N, "right"), (c1, 0.03 * N, "left")):
        ax.annotate(f"{x:,}", xy=(x, 0), xytext=(x + dx, -0.035 * N), ha=ha, va="bottom", fontsize=4.5,
                    annotation_clip=False, arrowprops=dict(arrowstyle="-", lw=0.4, color="k", shrinkA=0, shrinkB=0))
    ax.yaxis.labelpad = 16
    if compact:
        ax.text(-0.17, 1.105, la, transform=ax.transAxes, fontsize=7, fontweight="bold", va="top")
    else:
        ax.text(-0.16, 1.04, la, transform=ax.transAxes, fontsize=8, fontweight="bold")

    ax2 = fig.add_axes(rect_b)
    B = WB["B"]  # exact entries, one pixel per neuron pair
    ax2.imshow(B, cmap="RdBu_r", vmin=-VMAX, vmax=VMAX, interpolation="none",
               extent=(c0, c0 + Wn, r0 + Wn, r0), aspect="auto")
    ax2.set_xticks([c0, c1]); ax2.set_xticklabels([f"{c0:,}", f"{c1:,}"])
    ax2.yaxis.tick_right(); ax2.yaxis.set_label_position("right")
    ax2.set_yticks([r0, r1]); ax2.set_yticklabels([f"{r0:,}", f"{r1:,}"])
    ax2.set_xlabel("neuron (column)", labelpad=1); ax2.set_ylabel("neuron (row)", labelpad=1)
    ax2.spines["right"].set_visible(True); ax2.spines["left"].set_visible(False)
    er, ec = r0 + WB["ex_r"], c0 + WB["ex_c"]
    for y in er:
        ax2.axhline(y + 0.5, color="0.15", lw=0.3, alpha=0.5, ls=(0, (2, 2)))
    for x in ec:
        ax2.axvline(x + 0.5, color="0.15", lw=0.3, alpha=0.5, ls=(0, (2, 2)))
    TCOL = "0.15"
    if rect_tr is not None:   # row examples, left, evenly spaced slots with leaders to their rows
        axt = fig.add_axes(rect_tr); axt.set_xlim(0, 1); axt.set_ylim(r0 + Wn, r0); axt.axis("off")
        x_img = (rect_b[0] - rect_tr[0]) / rect_tr[2]
        slots = r0 + (np.arange(len(er)) + 0.5) * Wn / len(er)
        h = 0.6 * Wn / len(er)
        x = np.linspace(0, 0.8, WB["ex_rvec"].shape[1])
        pe = 0.8 * phase_edges(WB["lens"])
        for k in (1, 3):   # shaded: stimulus and after movement onset
            axt.axvspan(pe[k], pe[k + 1], color="0.9", lw=0, zorder=0)
        for k, (name, _) in enumerate(PHASES):
            axt.text((pe[k] + pe[k + 1]) / 2, r0 - 0.01 * Wn, name, ha="center", va="bottom", fontsize=3.5,
                     color="0.35")
        for v, y, ys in zip(WB["ex_rvec"], er, slots):
            z = (v - v.min()) / (np.ptp(v) + 1e-12)
            axt.plot(x, ys + h / 2 - h * z, color=TCOL, lw=0.3)
            axt.plot([0.83, 0.92, x_img], [ys, y + 0.5, y + 0.5], color="0.4", lw=0.35, clip_on=False)
    if rect_tt is not None:   # column examples, top, evenly spaced slots with leaders to their columns
        axu = fig.add_axes(rect_tt); axu.set_xlim(c0, c0 + Wn); axu.set_ylim(0, 1); axu.axis("off")
        y_img = -(rect_tt[1] - (rect_b[1] + rect_b[3])) / rect_tt[3]
        n_ = len(ec); slot_w = Wn / n_
        fw, fh = fig.get_size_inches()
        # vertical traces (time running downwards), same physical length as the left traces
        Lr = (0.8 * rect_tr[2] * fw) / (rect_tt[3] * fh) if rect_tr is not None else 0.85
        pe = 0.98 - Lr * phase_edges(WB["lens"])
        for k in (1, 3):
            axu.axhspan(pe[k + 1], pe[k], color="0.9", lw=0, zorder=0)
        for k, name in enumerate(PHASE_FULL):  # written out: there is room left of the top traces
            axu.text(c0 - 0.01 * Wn, (pe[k] + pe[k + 1]) / 2, name, ha="right", va="center", fontsize=4.5,
                     color="0.35")
        for k, (v, xc) in enumerate(zip(WB["ex_cvec"], ec)):
            xm = c0 + (k + 0.5) * slot_w
            yy = np.linspace(0.98, 0.98 - Lr, len(v))
            z = (v - v.min()) / (np.ptp(v) + 1e-12)
            axu.plot(xm - 0.3 * slot_w + 0.6 * slot_w * z, yy, color=TCOL, lw=0.3)
            ybot = 0.98 - Lr - 0.02
            axu.plot([xm, xm, xc + 0.5, xc + 0.5], [ybot, ybot - 0.03, ybot - 0.08, y_img], color="0.4",
                     lw=0.35, clip_on=False)
    ax2.set_title(title_b, pad=4)
    if compact and rect_tr is not None:  # letter left of the example traces, level with a
        fig.text(rect_tr[0] - 0.012, rect_a[1] + rect_a[3] + 0.045, lb, fontsize=7, fontweight="bold", va="top")
    elif compact:
        ax2.text(-0.15, 1.06, lb, transform=ax2.transAxes, fontsize=7, fontweight="bold", va="top")
    else:
        ax2.text(-0.2, 1.04, lb, transform=ax2.transAxes, fontsize=8, fontweight="bold")
    for rect, v, lab in ((rect_ca, VMAX, "$r$ (cluster mean)"), (rect_cb, VMAX, "$r$")):
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
