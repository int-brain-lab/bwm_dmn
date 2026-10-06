#!/usr/bin/env python3
"""Fig. 5 (draft): brain-wide stimulus-locked latency tiling, from dense to sparse
cells, validated on held-out trials.

a, b  Rastermap fit on odd trials (Methods parameters, grid_upsample=10, all 21
      PETH types; ../sequence_analysis/preview_upsample.py); a: odd trials (train),
      b: even trials (test), same order.
      Display: per neuron, (x - median) / (99th pct - median) clipped to [0, 1],
      values below the neuron's 80th percentile set to white, gamma 0.5 (identical
      for a and b; display only).
      Brackets (row ranges in this order): sparse cells, rows 29,600-35,600 (the
      former "sequence cells"); dense latency tiling, rows 11,800-16,300.
c-j   ../sequence_analysis/sparse_tornado.py (selection and evaluation on disjoint
      trial types).
k-m   ../sequence_analysis/degrade_tornado.py: dense tiling cells + noise matched to the
      sparse cells' reliability, own Rastermap fit (odd), shown on odd and even.
Run degrade_tornado.py (and rt_split_latency.py) in ../sequence_analysis first.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEQ_DIR = HERE.parent / "sequence_analysis"
sys.path.insert(0, str(SEQ_DIR))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

from seq_common import use_private_base  # noqa: E402
import sparse_tornado  # noqa: E402

_spec = importlib.util.spec_from_file_location("seq_rasters", SEQ_DIR / "make_figure3.py")
seq_rasters = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(seq_rasters)

MM = 1 / 25.4
ROW_BLOCK = 5
# Groups = row ranges in the order of a, b (inclusive), as in sparse_tornado.py
BRACKETS = {"sparse cells": ((29600, 35599), "#1f77b4"), "dense tiling": ((11800, 16299), "#ff7f0e")}
STIM6 = ["block_change_s", "stimLbLcL", "stimLbRcL", "stimRbRcR", "stimRbLcR", "mistake_s"]


def thresholded(X, q=80, gamma=0.5):
    """Per-row robust scaling to [0, 1]; values below the row's q-th percentile -> 0;
    gamma < 1 darkens the kept (faint but above-threshold) values."""
    med = np.median(X, 1, keepdims=True)
    hi = np.percentile(X, 99, 1, keepdims=True)
    R = np.clip((X - med) / (hi - med + 1e-6), 0, 1)
    return np.where(R >= np.percentile(R, q, 1, keepdims=True), R, 0) ** gamma


def to_image(M):
    n = M.shape[0] // ROW_BLOCK * ROW_BLOCK
    M = M[:n].reshape(-1, ROW_BLOCK, M.shape[1]).mean(1)
    return np.repeat((1 - M)[..., None], 3, axis=2)


def bracket(ax, lo, hi, color, text):
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    ax.plot([1.02, 1.02], [lo, hi], color=color, lw=2.2, transform=trans,
            clip_on=False, solid_capstyle="butt")
    ax.text(1.045, (lo + hi) / 2, text, color=color, rotation=90, ha="left",
            va="center", fontsize=5, transform=trans)


def raster(fig, rect, img, r, n_rows, title, show_ylabel):
    """Raster with PETH-type labels, dotted segment boundaries, 200 ms bar."""
    ax = fig.add_axes(rect)
    n_bins = sum(r["len"].values())
    ax.imshow(img, aspect="auto", interpolation="antialiased", rasterized=True,
              extent=(0, n_bins, n_rows, 0))
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    start = 0
    for seg, n in r["len"].items():
        ax.text(start + n / 2, 1.01, r["peth_dict"][seg], transform=trans,
                rotation=seq_rasters.SEG_ROTATION, ha="left", va="bottom", rotation_mode="anchor",
                fontsize=seq_rasters.SEG_FONTSIZE, clip_on=False)
        start += n
        if start < n_bins:
            ax.axvline(start, color="0.5", lw=0.35, ls=(0, (2, 2)), zorder=6)
    ax.set_xlim(0, n_bins); ax.set_xticks([])
    ax.set_yticks([t * 1000 for t in (0, 10, 20, 30, 40, 50)], ["0", "10", "20", "30", "40", "50"])
    if show_ylabel:
        ax.set_ylabel(r"cell index ($\times\,10^3$)")
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_title(title, fontsize=6, pad=34)
    seq_rasters.scale_bar(ax, 0.2 * seq_rasters.dmn_bwm.c_sec, "200 ms")
    return ax


def main():
    seq_rasters.configure_style()
    d = use_private_base()
    r = d.regional_group("rm", vers="concat", cv=True)
    order = sparse_tornado.fit_order(r)  # grid_upsample=10 fit on odd trials

    fig = plt.figure(figsize=(183 * MM, 180 * MM))
    for k, (key, title) in enumerate((("concat_z_train", "train: odd trials, sorted by their Rastermap fit"),
                                      ("concat_z", "test: even trials, same order (cross-validated)"))):
        img = to_image(thresholded(np.asarray(r[key], np.float32)[order]))
        ax = raster(fig, [0.06 + k * 0.48, 0.615, 0.39, 0.31], img, r, order.size, title,
                    show_ylabel=(k == 0))
        for text, ((lo, hi), color) in BRACKETS.items():
            bracket(ax, lo, hi, color, text)
        seq_rasters.label(fig, ax, "ab"[k], dy=0.04)

    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1.1, 1], hspace=1.0, wspace=0.55,
                          left=0.07, right=0.98, top=0.515, bottom=0.04)
    cells = [gs[0, 0], gs[0, 1], gs[0, 2], gs[0, 3], gs[1, 0], gs[1, 1], gs[1, 2], gs[1, 3]]
    sparse_tornado.main(fig=fig, cells=cells, letters="cdefghij")

    D = np.load(SEQ_DIR / "results/degrade_tornado.npz", allow_pickle=True)
    for k, (key, half) in enumerate((("noisy_odd", "odd (train)"), ("noisy_even", "even (test)"))):
        ax = fig.add_subplot(gs[2, k])
        ax.imshow(to_image(thresholded(D[key]))[..., 0], cmap="gray", vmin=0, vmax=1,
                  aspect="auto", interpolation="antialiased", rasterized=True)
        for b in range(1, 6):
            ax.axvline(72 * b, color="0.5", lw=0.3, ls=":")
        ax.set_xticks([72 * b + 36 for b in range(6)], [seq_rasters.dmn_bwm.peth_dictm[t] for t in STIM6], rotation=40, fontsize=4.3)
        ax.set_yticks([])
        ax.set_title(f"dense tiling + noise, own fit\n{half}", fontsize=5.5)
        ax.text(-0.22, 1.06, "kl"[k], transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")
    ax = fig.add_subplot(gs[2, 2:4])
    names = ["sparse cells (rows 29.6–35.6k)", "dense tiling + noise", "dense tiling (rows 11.8–16.3k)"]
    colors = ["#1f77b4", "0.45", "#ff7f0e"]
    metrics = ["median reliability", "fraction r ≥ 0.3", "xval lag-0 corr", "peak-time ρ"]
    S = D["stats"]
    for j in range(3):
        ax.bar(np.arange(4) + 0.27 * (j - 1), S[j], width=0.26, color=colors[j], label=names[j])
    ax.set_xticks(range(4), metrics, fontsize=5)
    ax.set_title("noise alone turns dense tiling into sparse-cell statistics", fontsize=6)
    ax.legend(frameon=False, fontsize=4.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(-0.1, 1.06, "m", transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")

    stem = HERE / "contextual_neural_sequences"
    fig.savefig(stem.with_suffix(".pdf"), dpi=600, facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white")
    plt.close(fig)
    subprocess.run(["gs", "-q", "-dNOPAUSE", "-dBATCH", "-dSAFER", "-sDEVICE=pdfwrite",
                    "-dCompatibilityLevel=1.5", "-dPDFSETTINGS=/printer",
                    f"-sOutputFile={stem}_printer.pdf", str(stem.with_suffix(".pdf"))], check=True)
    print(f"Saved {stem}.pdf/.png and {stem.name}_printer.pdf")


if __name__ == "__main__":
    main()
