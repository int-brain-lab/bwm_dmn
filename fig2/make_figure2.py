#!/usr/bin/env python3
"""Assemble Fig. 2 (functional response structure) from dmn_bwm functions.

Rastermap-free (all panels use k-means on all trials):
a  single-cell raster sorted by k-means cluster, from regenerate_raster_panels.py
   (dmn_bwm.plot_rastermap with sort_method="acs", i.e. no Rastermap ordering)
b  neurons selected by dmn_bwm.plot_fig2e_clean_examples, stacked top-down
c  dmn_bwm.plot_cluster_mean_PETHs (25 k-means cluster means)
d,e  the same cluster means for selected clusters and trial segments

Run regenerate_raster_panels.py first (or pass --rasters).
"""

import argparse
import contextlib
import subprocess
import sys

import matplotlib as mpl

mpl.use("Agg")  # dmn_bwm calls plt.show(); keep it non-blocking
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import blended_transform_factory

from fig2_common import (OUT, B_CLUSTERS, B_SEGMENTS, C_CLUSTERS, C_SEGMENTS,
                         dmn_bwm, use_private_base)

MM = 1 / 25.4
SEG_FONTSIZE = 4.3  # PETH-type labels: same size and tilt in every panel
SEG_ROTATION = 65


def configure_style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 5, "axes.labelsize": 5.5, "xtick.labelsize": 5,
        "ytick.labelsize": 5, "axes.linewidth": 0.5, "xtick.major.width": 0.5,
        "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2,
        "pdf.fonttype": 42, "svg.fonttype": "none",
        # math text (segment labels) in the same sans-serif as fig3/fig4
        "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans",
        "mathtext.it": "Liberation Sans:italic", "mathtext.bf": "Liberation Sans:bold",
    })


def label(fig, ax, letter, dx=-0.03, dy=0.0):
    pos = ax.get_position()
    fig.text(pos.x0 + dx, pos.y1 + dy, letter, fontsize=8, fontweight="bold",
             ha="left", va="bottom")

def scale_bar(ax, length, text, y=-0.03, lw=1.2):
    """Horizontal scale bar at the bottom right of ax: length in data x-units,
    y in axes fraction (below the axes when negative)."""
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    x1 = ax.get_xlim()[1]
    ax.plot([x1 - length, x1], [y, y], color="k", lw=lw, transform=trans,
            clip_on=False, solid_capstyle="butt")
    ax.text(x1 - length / 2, y - 0.012, text, ha="center", va="top", fontsize=5,
            transform=trans)


def segment_labels(ax, r, x_of_bin, y=1.01, segments=None, rotation=SEG_ROTATION):
    """Rotated trial-segment names above an axes (x in data, y in axes units)."""
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    start = 0
    for seg, n in r["len"].items():
        if segments is None or seg in segments:
            ax.text(x_of_bin(start + n / 2), y, r["peth_dict"][seg], transform=trans,
                    rotation=rotation, ha="left", va="bottom", fontsize=SEG_FONTSIZE,
                    rotation_mode="anchor", clip_on=False)
        start += n


def segment_lines(ax, r, x_of_bin, **kw):
    start = 0
    style = dict(color="0.35", lw=0.3, ls=(0, (2, 2)), zorder=4) | kw
    for n in list(r["len"].values())[:-1]:
        start += n
        ax.axvline(x_of_bin(start), **style)


def cluster_mean(r, cluster):
    idx = np.flatnonzero(r["acs"] == cluster - 1)
    return r["concat_z"][idx].mean(axis=0), r["cols"][idx[0]]


def segment_bins(r, segments):
    """Concatenated bin indices of the named segments, plus their lengths."""
    starts = np.cumsum([0] + list(r["len"].values()))
    names = list(r["len"])
    bins = [np.arange(starts[names.index(s)], starts[names.index(s) + 1]) for s in segments]
    return np.concatenate(bins), [len(b) for b in bins]


def panel_prototypes(fig, d, r, rect):
    n = len(np.unique(r["acs"]))
    top = rect[1] + rect[3]
    h = rect[3] / n
    axes = [fig.add_axes([rect[0], top - (k + 1) * h, rect[2], h]) for k in range(n)]
    d.plot_cluster_mean_PETHs(r, "kmeans", "concat_z", axx=axes, alone=False, cv=False)
    for k, ax in enumerate(axes):
        for t in list(ax.texts):
            t.remove()
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.lines[0].set_linewidth(0.6)
        for line in ax.lines[1:]:
            line.set(color="0.35", lw=0.3, ls=(0, (1, 1.5)))
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(bottom=False, labelbottom=False)
        ax.patch.set_alpha(0)
        ax.text(1.01, 0.5, str(k + 1), transform=ax.transAxes, fontsize=4.5,
                ha="left", va="center")
    segment_labels(axes[0], r, lambda b: b / d.c_sec, y=1.3)
    # 200 ms scale bar under the last trace.
    scale_bar(axes[-1], 0.2, "200 ms", y=-0.25)  # x in seconds
    axes[0].set_title("Mean cluster feature vector", fontsize=6, pad=30)
    top, bottom = axes[0].get_position(), axes[-1].get_position()
    fig.text(top.x1 + 0.028, (top.y1 + bottom.y0) / 2, "k-means cluster", rotation=90,
             ha="center", va="center", fontsize=5)
    return axes[0]


def panel_zoom_events(fig, r, rect):
    ax = fig.add_axes(rect)
    bins, lens = segment_bins(r, B_SEGMENTS)
    x = np.arange(bins.size)
    bounds = np.cumsum(lens)
    ends = {}
    for cluster, name in B_CLUSTERS.items():
        y, col = cluster_mean(r, cluster)
        y = y[bins]
        y = (y - y.min()) / (y.max() - y.min())  # shapes, not amplitudes
        ax.plot(x, y, color=col, lw=1.0)
        ends[cluster] = y[-1]
        seg = {"stim": 0, "integ": 0, "mvt init": 1}.get(name, 2)
        lo, hi = ([0] + list(bounds))[seg], bounds[seg]
        peak = lo + int(np.argmax(y[lo:hi]))
        dx = {21: -6, 9: 12, 12: -14, 14: 10}.get(cluster, 0)
        dy = {14: -0.22}.get(cluster, 0.02)
        ax.text(peak + dx, y[peak] + dy, name, color=col, fontsize=5, ha="center",
                va="bottom", fontweight="bold")
    # Right-edge cluster numbers, pushed apart where trace ends coincide.
    placed = []
    for cluster, y_end in sorted(ends.items(), key=lambda kv: kv[1]):
        y_txt = max([y_end] + [p + 0.09 for p in placed])
        placed.append(y_txt)
        ax.text(x[-1] + 2, y_txt, str(cluster), fontsize=4.5, va="center", ha="left")
    for b in bounds[:-1]:
        ax.axvline(b, color="0.35", lw=0.3, ls=(0, (2, 2)))
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for s, lo, n in zip(B_SEGMENTS, [0] + list(bounds[:-1]), lens):
        ax.text(lo + n / 2, 1.12, r["peth_dict"][s], transform=trans, rotation=SEG_ROTATION,
                ha="left", va="bottom", rotation_mode="anchor", fontsize=SEG_FONTSIZE,
                clip_on=False)
    ax.set_xlim(0, x[-1])
    ax.set_axis_off()
    scale_bar(ax, 0.1 * dmn_bwm.c_sec, "100 ms", y=-0.05)  # x in bins; segments are 150 ms
    return ax


def panel_zoom_states(fig, r, rect):
    ax = fig.add_axes(rect)
    bins, lens = segment_bins(r, C_SEGMENTS)
    x = np.arange(bins.size)
    bounds = np.cumsum(lens)
    offset, previous = 0.0, None
    for cluster, name in C_CLUSTERS.items():
        y, col = cluster_mean(r, cluster)
        y = y[bins]
        if previous is not None:
            offset = float(np.min(previous) - np.max(y)) - 0.35
        yy = y + offset
        ax.plot(x, yy, color=col, lw=1.0)
        ax.text(2, yy[:60].max() + 0.04, name, color=col, fontsize=5, ha="left",
                va="bottom", fontweight="bold")
        ax.text(x[-1] + 3, yy[-1], str(cluster), fontsize=4.5, va="center", ha="left")
        previous = yy
    for b in bounds[:-1]:
        ax.axvline(b, color="0.35", lw=0.3, ls=(0, (2, 2)))
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for s, lo, n in zip(C_SEGMENTS, [0] + list(bounds[:-1]), lens):
        ax.text(lo + n / 2, 1.01, r["peth_dict"][s], transform=trans, rotation=SEG_ROTATION,
                ha="left", va="bottom", rotation_mode="anchor", fontsize=SEG_FONTSIZE,
                clip_on=False)
    ax.set_xlim(0, x[-1])
    ax.set_axis_off()
    scale_bar(ax, 0.2 * dmn_bwm.c_sec, "200 ms", y=-0.04)  # x in bins
    return ax


def raster_axes(fig, r, rect, image, n_rows, yticks):
    ax = fig.add_axes(rect)
    n_bins = sum(r["len"].values())
    ax.imshow(image, aspect="auto", interpolation="antialiased", rasterized=True,
              extent=(0, n_bins, n_rows, 0))
    segment_lines(ax, r, lambda b: b, color="0.5", lw=0.35)
    segment_labels(ax, r, lambda b: b)
    ax.set_xlim(0, n_bins)
    ax.set_xticks([])
    ax.set_yticks([t * 1000 for t in yticks], [str(t) for t in yticks])
    ax.set_ylabel(r"cell index ($\times\,10^3$)")
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    return ax


def panel_raster_by_cluster(fig, r, rect, rows):
    clusters = rows["clusters_d"]
    ax = raster_axes(fig, r, rect, plt.imread(OUT / "panel_d_kmeans_raster.png"),
                     clusters.size, [1, 10, 20, 30, 40, 50])
    edges = np.flatnonzero(np.diff(clusters)) + 1
    for y in edges:  # cluster boundaries
        ax.axhline(y, color="k", lw=0.3, zorder=5)
    edges = np.r_[0, edges, clusters.size]
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    acs = np.asarray(r["acs"])
    for lo, hi in zip(edges[:-1], edges[1:]):
        cl = int(clusters[lo])
        ax.text(1.01, (lo + hi) / 2, str(cl + 1), transform=trans, fontsize=4.3,
                ha="left", va="center", color=r["cols"][np.flatnonzero(acs == cl)[0]])
    ax.text(1.075, 0.5, "k-means cluster", transform=ax.transAxes, rotation=90,
            ha="center", va="center", fontsize=5)
    scale_bar(ax, 0.2 * dmn_bwm.c_sec, "200 ms", y=-0.012)  # x in bins
    ax.set_title("single-cell response vectors by cluster", fontsize=6, pad=32)
    return ax


@contextlib.contextmanager
def full_trial_clusters(d):
    """Give each neuron of the CV k-means result its panel a-d cluster.

    plot_fig2e_clean_examples needs the trial-split responses (cv=True), but the
    cv=True and cv=False clusterings number (and partly assign) clusters
    differently. Within this context dmn_bwm.regional_group("kmeans", cv=True)
    returns the CV responses with 'acs'/'cols' taken from the cv=False
    clustering, matched per neuron by UUID, so panel e's numbers are panel a's.
    """
    original = d.regional_group
    full = original("kmeans", vers="concat", cv=False, nclus=25)
    row = {u: i for i, u in enumerate(np.asarray(full["uuids"]).astype(str))}

    def regional_group(mapping, *args, **kwargs):
        r = original(mapping, *args, **kwargs)
        if mapping == "kmeans" and kwargs.get("cv", True):
            idx = np.array([row[u] for u in np.asarray(r["uuids"]).astype(str)])
            r = dict(r, acs=np.asarray(full["acs"])[idx], cols=np.asarray(full["cols"])[idx])
        return r

    d.regional_group = regional_group
    try:
        yield regional_group
    finally:
        d.regional_group = original


def panel_examples(fig, d, rect):
    """Two reliable example neurons per cluster, stacked with cluster 1 at the top
    (as in a and d); cluster numbers on the left, Beryl regions on the right.

    Neurons are chosen by dmn_bwm.plot_fig2e_clean_examples on the odd/even trial
    split (even-trial trace ranked by correlation with the odd-trial cluster mean,
    reliability between the two), with panel a's clusters and without the
    Lempel-Ziv filter. As in panels a-d, the traces shown are the all-trial
    feature vectors (cv=False) of those neurons. The drawing mirrors that function
    (gaps, segment labels) in top-down order.
    """
    with full_trial_clusters(d) as regional_group:
        fig_e, selection = d.plot_fig2e_clean_examples(min_max_lz=None, save_formats=())
        rcv = regional_group("kmeans", vers="concat", cv=True, nclus=25)
    plt.close(fig_e)
    selection.to_csv(OUT / "panel_e_selection.csv", index=False)
    idx = selection.array_index.to_numpy()
    rall = d.regional_group("kmeans", vers="concat", cv=False, nclus=25)
    row = {u: i for i, u in enumerate(np.asarray(rall["uuids"]).astype(str))}
    traces = np.asarray(rall["concat_z"][[row[u] for u in selection.uuid.astype(str)]], dtype=float)
    amp = np.nanmedian(np.nanpercentile(traces, 95, axis=1) - np.nanpercentile(traces, 5, axis=1))
    trace_gap, cluster_gap = max(0.05, 0.18 * amp), 0.30 * amp  # room for labels

    ax = fig.add_axes(rect)
    x = np.arange(traces.shape[1]) / d.c_sec
    xpad = 0.012 * (x[-1] - x[0])
    clusters = selection.cluster.to_numpy()

    def stack(min_sep):
        """Offsets: each trace just below the previous one (plus gaps), and labels
        at least min_sep apart (data units)."""
        cursors, previous = [], None
        for k, y in enumerate(traces):
            cursor = 0.0
            if previous is not None:
                cursor = float(np.nanmin(previous - y)) - trace_gap
                if clusters[k] != clusters[k - 1]:
                    cursor -= cluster_gap
                cursor = min(cursor, cursors[-1] - min_sep)
            cursors.append(cursor)
            previous = y + cursor
        return np.asarray(cursors)

    # Labels are 4 pt; keep consecutive label centres >= 5 pt apart on the page.
    height_pt = rect[3] * fig.get_figheight() * 72
    min_sep = 0.0
    for _ in range(6):  # the span depends on min_sep; iterate to a fixed point
        cursors = stack(min_sep)
        span = (traces + cursors[:, None]).max() - (traces + cursors[:, None]).min()
        min_sep = 5.0 / height_pt * span * 1.12  # + margin for the axis padding

    plotted = []
    for k, (y, cluster, region) in enumerate(zip(traces, clusters, selection.Beryl)):
        cursor = cursors[k]
        yy = y + cursor
        ax.plot(x, yy, color="black", lw=0.4, alpha=0.95)
        ax.text(x[0] - xpad, cursor, str(cluster + 1), ha="right", va="center",
                fontsize=4, color=rcv["cols"][idx[k]], clip_on=False)
        if k == 0:
            ax.text(-0.075, 0.5, "k-means cluster", transform=ax.transAxes, rotation=90,
                    ha="center", va="center", fontsize=5)
        ax.text(x[-1] + xpad, cursor, region, ha="left", va="center", fontsize=4,
                color=d.pal[region] if region in d.pal else "black", clip_on=False)
        plotted.append(yy)

    plotted = np.asarray(plotted)
    y_min, y_max = float(np.nanmin(plotted)), float(np.nanmax(plotted))
    y_span = y_max - y_min
    y_top = y_max + 0.015 * y_span
    # Segment labels sit just above the axes top, at the same level as in panel a;
    # boundaries stop at the axes top, below the labels.
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    start = 0
    for seg, n in rcv["len"].items():
        ax.vlines((start + n) / d.c_sec, y_min - 0.01 * y_span, y_top,
                  color="0.55", lw=0.35, zorder=0)
        ax.text((start + n / 2) / d.c_sec, 1.01, rcv["peth_dict"][seg], rotation=SEG_ROTATION,
                ha="left", va="bottom", rotation_mode="anchor", fontsize=SEG_FONTSIZE,
                transform=trans, clip_on=False)
        start += n
    ax.set_xlim(x[0] - 0.01 * (x[-1] - x[0]), x[-1])
    ax.set_ylim(y_min - 0.01 * y_span, y_top)
    ax.set_axis_off()
    scale_bar(ax, 0.2, "200 ms", y=-0.012)  # x in seconds
    return ax


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rasters", action="store_true",
                        help="re-render the raster image of panel a first")
    args = parser.parse_args()
    if args.rasters or not (OUT / "raster_rows.npz").exists():
        subprocess.run([sys.executable, str(OUT / "regenerate_raster_panels.py")], check=True)

    configure_style()
    d = use_private_base()
    # a-d use the clustering of all trials (54,719 neurons); e selects neurons
    # with the odd/even split and shows their all-trial feature vectors.
    r = d.regional_group("kmeans", vers="concat", cv=False, nclus=25)
    rows = np.load(OUT / "raster_rows.npz")

    fig = plt.figure(figsize=(183 * MM, 140 * MM))
    # Rastermap-free layout, read left to right: a raster by cluster, b example
    # neurons, c prototypes with the zooms d and e below.
    axa = panel_raster_by_cluster(fig, r, [0.055, 0.04, 0.255, 0.78], rows)
    axb = panel_examples(fig, d, [0.385, 0.04, 0.215, 0.78])  # same extent as a
    axc = panel_prototypes(fig, d, r, [0.70, 0.55, 0.25, 0.30])
    axd = panel_zoom_events(fig, r, [0.70, 0.285, 0.255, 0.135])
    axe = panel_zoom_states(fig, r, [0.70, 0.04, 0.255, 0.165])
    for ax, letter, dy in [(axa, "a", 0.075), (axb, "b", 0.075), (axc, "c", 0.085),
                           (axd, "d", 0.065), (axe, "e", 0.05)]:
        label(fig, ax, letter, dy=dy)

    stem = OUT / "functional_response_structure"
    for ext, dpi in (("pdf", 600), ("svg", 600), ("png", 300)):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=dpi, facecolor="white")
    plt.close(fig)
    subprocess.run(["gs", "-q", "-dNOPAUSE", "-dBATCH", "-dSAFER", "-sDEVICE=pdfwrite",
                    "-dCompatibilityLevel=1.5", "-dPDFSETTINGS=/printer",
                    f"-sOutputFile={stem}_printer.pdf", str(stem.with_suffix(".pdf"))],
                   check=True)
    print(f"Saved {stem}.pdf/.svg/.png and {stem.name}_printer.pdf")


if __name__ == "__main__":
    main()
