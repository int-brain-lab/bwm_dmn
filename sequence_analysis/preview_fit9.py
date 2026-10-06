#!/usr/bin/env python3
"""Preview: train (odd) and test (even) rasters in the order of the Rastermap fit
on the nine well-sampled PETH types (../sequence_analysis/results/rastermap_fit9.npz).
Fit columns are marked by a black bar above the raster; all others are held out
from the fit. Display as in the Methods (z-score clipped to [0, 1.5], grey on white).
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import blended_transform_factory

from seq_common import OUT, PREV, ROOT, use_private_base
from fit_rastermap_subset import zrows
from make_figure3 import MM, configure_style, label, raster_panel

FIT = ROOT / "sequence_analysis/results/rastermap_fit9.npz"
ROW_BLOCK = 5


def image(X, order):
    M = np.clip(X[order], 0, 1.5) / 1.5
    n = M.shape[0] // ROW_BLOCK * ROW_BLOCK
    M = M[:n].reshape(-1, ROW_BLOCK, M.shape[1]).mean(1)
    return np.repeat((1 - M)[..., None], 3, axis=2)


def mark_fit_columns(ax, r, fit_types):
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    start = 0
    for seg, n in r["len"].items():
        if seg in fit_types:
            ax.plot([start + 2, start + n - 2], [1.004, 1.004], color="k", lw=1.6,
                    transform=trans, clip_on=False, solid_capstyle="butt")
        start += n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fit-only", action="store_true", help="show only the fit columns")
    ap.add_argument("--set", default="fit9", help="results/rastermap_<set>.npz")
    ap.add_argument("--rezscore", action="store_true",
                    help="z-score each neuron within the shown columns (with --fit-only)")
    args = ap.parse_args()
    fit_only, tag, rez = args.fit_only, args.set, args.rezscore
    configure_style()
    d = use_private_base()
    r = d.regional_group("rm", vers="concat", cv=True)
    f = np.load(ROOT / f"sequence_analysis/results/rastermap_{tag}.npz", allow_pickle=True)
    if not np.array_equal(f["uuids"], np.asarray(r["uuids"])):
        raise RuntimeError("fit and stack neurons differ")
    order, labels, fit_types = f["isort"], f["labels"], set(f["fit_types"])
    clusters = labels[order]
    cols = slice(None)
    if fit_only:  # keep only the fit columns, in their order in the feature vector
        start = np.cumsum([0] + list(r["len"].values()))
        names = list(r["len"])
        keep = [t for t in names if t in fit_types]
        cols = np.concatenate([np.arange(start[names.index(t)], start[names.index(t) + 1])
                               for t in keep])
        r = dict(r, len={t: r["len"][t] for t in keep})
    fig = plt.figure(figsize=(183 * MM, 120 * MM))
    axes = []
    for k, (key, title) in enumerate((("concat_z_train", ("train: odd trials (fit columns only)" if fit_only else f"train: odd trials, fit on {len(fit_types)} well-sampled types (bars)")),
                                      ("concat_z", "test: even trials, same order (cross-validated)"))):
        ax = raster_panel(fig, [0.06 + k * 0.495, 0.05, 0.40, 0.78],
                          image(zrows(np.asarray(r[key], np.float32)[:, cols]) if rez else
                                np.asarray(r[key], np.float32)[:, cols], order), r, clusters, title,
                          show_ylabel=(k == 0))
        if not fit_only:
            mark_fit_columns(ax, r, fit_types)
        axes.append(ax)
    for ax, s in zip(axes, "ab"):
        label(fig, ax, s, dy=0.09)
    out = PREV / (f"preview_{tag}_fitcolumns.png" if fit_only else f"preview_{tag}.png")
    fig.savefig(out, dpi=300, facecolor="white")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
