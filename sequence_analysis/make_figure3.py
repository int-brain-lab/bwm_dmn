#!/usr/bin/env python3
"""Assemble Fig. 3 (neural sequences). Work in progress.

Top row: Rastermap of the odd-trial averages (train, sorted by its own fit) and of
the even-trial averages (test) sorted by the same order, i.e. cross-validated.
Run regenerate_rasters.py first (or pass --rasters).
"""

import argparse
import subprocess
import sys

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import blended_transform_factory

from seq_common import OUT, PREV, dmn_bwm, use_private_base

MM = 1 / 25.4
SEG_FONTSIZE = 4.3  # same PETH-label style as Fig. 2
SEG_ROTATION = 65


def configure_style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 5, "axes.labelsize": 5.5, "xtick.labelsize": 5, "ytick.labelsize": 5,
        "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2, "ytick.major.size": 2, "pdf.fonttype": 42, "svg.fonttype": "none",
        "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans",
        "mathtext.it": "Liberation Sans:italic", "mathtext.bf": "Liberation Sans:bold",
    })


def label(fig, ax, letter, dx=-0.035, dy=0.0):
    pos = ax.get_position()
    fig.text(pos.x0 + dx, pos.y1 + dy, letter, fontsize=8, fontweight="bold",
             ha="left", va="bottom")


def scale_bar(ax, length, text, y=-0.012):
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    x1 = ax.get_xlim()[1]
    ax.plot([x1 - length, x1], [y, y], color="k", lw=1.2, transform=trans,
            clip_on=False, solid_capstyle="butt")
    ax.text(x1 - length / 2, y - 0.012, text, ha="center", va="top", fontsize=5, transform=trans)


def raster_panel(fig, rect, image, r, clusters, title, show_ylabel=True):
    ax = fig.add_axes(rect)
    n_bins = sum(r["len"].values())
    ax.imshow(image, aspect="auto", interpolation="antialiased", rasterized=True,
              extent=(0, n_bins, clusters.size, 0))
    edges = np.flatnonzero(np.diff(clusters)) + 1
    for y in edges:
        ax.axhline(y, color="k", lw=0.2, zorder=5)
    # PETH-type segments: dotted boundaries and rotated labels.
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    start = 0
    for seg, n in r["len"].items():
        ax.text(start + n / 2, 1.01, r["peth_dict"][seg], transform=trans, rotation=SEG_ROTATION,
                ha="left", va="bottom", rotation_mode="anchor", fontsize=SEG_FONTSIZE,
                clip_on=False)
        start += n
        if start < n_bins:
            ax.axvline(start, color="0.5", lw=0.35, ls=(0, (2, 2)), zorder=6)
    ax.set_xlim(0, n_bins)
    ax.set_xticks([])
    ticks = [0, 10, 20, 30, 40, 50]
    ax.set_yticks([t * 1000 for t in ticks], [str(t) for t in ticks])
    if show_ylabel:
        ax.set_ylabel(r"cell index ($\times\,10^3$)")
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    # Right axis: Rastermap cluster numbers (0-99, top to bottom), every 5th.
    bounds = np.r_[0, edges, clusters.size]
    ids, mids = clusters[bounds[:-1]], (bounds[:-1] + bounds[1:]) / 2
    keep = ids % 5 == 0
    right = ax.twinx()
    right.set_ylim(ax.get_ylim())
    right.set_yticks(mids[keep], [str(int(i)) for i in ids[keep]])
    right.tick_params(axis="y", labelsize=3.5, length=1.2, width=0.4, pad=1)
    right.set_ylabel("Rastermap cluster", fontsize=5, labelpad=2)
    for side in ("top", "left", "bottom"):
        right.spines[side].set_visible(False)
    right.spines["right"].set_linewidth(0.5)
    ax.set_title(title, fontsize=6, pad=34)
    scale_bar(ax, 0.2 * dmn_bwm.c_sec, "200 ms")
    return ax


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rasters", action="store_true", help="re-render the raster images first")
    args = parser.parse_args()
    if args.rasters or not (PREV / "raster_clusters.npy").exists():
        subprocess.run([sys.executable, str(PREV / "regenerate_rasters.py")], check=True)

    configure_style()
    d = use_private_base()
    r = d.regional_group("rm", vers="concat", cv=True)
    clusters = np.load(PREV / "raster_clusters.npy")

    fig = plt.figure(figsize=(183 * MM, 120 * MM))
    ax_a = raster_panel(fig, [0.06, 0.05, 0.40, 0.78], plt.imread(PREV / "raster_train_odd.png"),
                        r, clusters, "train: odd trials, sorted by their Rastermap fit")
    ax_b = raster_panel(fig, [0.555, 0.05, 0.40, 0.78], plt.imread(PREV / "raster_test_even.png"),
                        r, clusters, "test: even trials, same order (cross-validated)",
                        show_ylabel=False)
    for ax, letter in ((ax_a, "a"), (ax_b, "b")):
        label(fig, ax, letter, dy=0.09)

    stem = PREV / "contextual_neural_sequences"
    for ext, dpi in (("pdf", 600), ("png", 300)):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=dpi, facecolor="white")
    plt.close(fig)
    subprocess.run(["gs", "-q", "-dNOPAUSE", "-dBATCH", "-dSAFER", "-sDEVICE=pdfwrite",
                    "-dCompatibilityLevel=1.5", "-dPDFSETTINGS=/printer",
                    f"-sOutputFile={stem}_printer.pdf", str(stem.with_suffix(".pdf"))], check=True)
    print(f"Saved {stem}.pdf/.png and {stem.name}_printer.pdf")


if __name__ == "__main__":
    main()
