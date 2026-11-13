#!/usr/bin/env python3
"""SI figure: neuron x neuron correlations and tests of random mixing.

Panels a, b: cross-validated neuron x neuron correlation matrix of the feature vectors
(odd vs even trials, Rastermap order), drawn by ../../neuron_correlation/make_panel.py
from its results/xcorr.npz (compute_xcorr.py).
Panels c-e from the full analysis in the parent folder (selectivity_test.py,
cache/results.json and cache/arrays.npz; drawing helpers from make_figure.py):
  c     ePAIRS angle distributions in all 1872 bins (../full_space_epairs.py --save-angles,
        cache/full_space_angles.npz): real vs covariance-matched Gaussian, PC-marginal
        null and a 25-cluster mixture (positive control); inset: the same in the top 10 PCs
  d     silhouette vs k in the top 10 PCs: no preferred number of clusters
  e     ePAIRS effect sizes (AUC) for response profiles (all 1872 bins from
        ../full_space_epairs.py, cache/full_space_epairs.json; top 10 PCs), each next to a 25-cluster
        mixture control fitted in the same space (open symbols), and for signed RRR
        task-variable selectivity (../rrr_epairs.py, cache/rrr_epairs.json; the S2 features),
        against four nulls: Gaussian, PC-marginal, Gaussian copula and
        ICA-marginal (the last two from ../extra_nulls.py, cache/extra_nulls.json)
Writes response_correlations.pdf/.png in this folder (183 mm wide).
"""

import json
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import make_figure as MF  # noqa: E402
sys.path.insert(0, str(HERE.parents[1] / "neuron_correlation"))
import make_panel as XC  # noqa: E402

MM = 1 / 25.4
NULLS = ("gauss", "pcmarg")
LAB = dict(gauss="Gaussian (cov.-matched)", pcmarg="PC-marginal",
           copula="Gaussian copula", icamarg="ICA-marginal")
NULLS_E = ("gauss", "pcmarg", "copula", "icamarg")
COL_E = dict(gauss="#2a78d6", pcmarg="#eb6834", copula="#7b4fc9", icamarg="#1baf7a")


def main():
    MF.style()
    R = json.loads((HERE.parent / "cache" / "results.json").read_text())
    A = np.load(HERE.parent / "cache" / "arrays.npz")
    W, H = 183, 150
    fig = plt.figure(figsize=(W * MM, H * MM))

    def R_(x, top, w, h):  # mm, from the top-left corner -> figure fraction
        return [x / W, 1 - (top + h) / H, w / W, h / H]

    s1, top1 = 66, 8   # row 1: square panels a, b
    XC.draw(fig, R_(14, top1, s1, s1), R_(108, top1, s1, s1),
            R_(14 + s1 * 0.2, top1 + s1 + 10, s1 * 0.6, 1.3), R_(108 + s1 * 0.2, top1 + s1 + 10, s1 * 0.6, 1.3),
            title_a="all neurons, odd vs even trials", title_b="dense tiling (21–28)", compact=True)

    top2, h2 = 101, 34  # row 2: c, d, e
    # c
    ax = fig.add_axes(R_(14, top2, 40, h2))
    # main: all 1872 bins (../full_space_epairs.py --save-angles -> cache/full_space_angles.npz)
    FA = np.load(HERE.parent / "cache" / "full_space_angles.npz")
    FS = json.loads((HERE.parent / "cache" / "full_space_epairs.json").read_text())["epairs"]
    bins = np.linspace(10, 80, 90)
    MF.hist(ax, FA["real"], MF.COL["real"], "real", bins=bins, lw=1)
    for nn in NULLS:
        MF.hist(ax, FA[nn], MF.COL[nn], LAB[nn], bins=bins)
    MF.hist(ax, FA["mix25"], MF.COL["mix"], "25-cluster mixture", ls="--", bins=bins)
    e = FS["ALL|xprobe|all"]
    ax.text(0.02, 0.98, f"AUC {e['gauss']['5']['auc']:.2f} / {e['pcmarg']['5']['auc']:.2f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=4)
    ax.set_xlabel("mean angle to the 5 most similar neurons (deg)")
    ax.set_ylabel("density")
    ax.set_title("response profiles, all time bins", pad=2)
    MF.ticks3(ax)  # colours as in the legend of d
    # inset: top 10 PCs (cache/arrays.npz, selectivity_test.py), the space of panel d
    base = "PC10|xprobe|all"
    ins = ax.inset_axes([0.05, 0.42, 0.36, 0.36])
    b2 = np.linspace(0, 50, 60)
    MF.hist(ins, A[f"ang_{base}_real"], MF.COL["real"], None, bins=b2, lw=0.7)
    for nn in NULLS:
        MF.hist(ins, A[f"ang_{base}_{nn}"], MF.COL[nn], None, bins=b2, lw=0.6)
    MF.hist(ins, A["ang_PC10mix25|all|all_real"], MF.COL["mix"], None, ls="--", bins=b2, lw=0.6)
    ep10 = R["epairs"][base]
    ins.set_title(f"top 10 PCs\nAUC {ep10['gauss']['5']['auc']:.2f} / {ep10['pcmarg']['5']['auc']:.2f}",
                  fontsize=3.8, pad=1.5, linespacing=1.2)
    ins.set_xticks([0, 25, 50]); ins.set_yticks([])
    ins.tick_params(labelsize=3.8, length=1.2, pad=0.8, width=0.3)
    for sp in ("top", "right", "left"):
        ins.spines[sp].set_visible(False)
    ins.spines["bottom"].set_linewidth(0.3)
    MF.letter(ax, "c")

    # d
    km = R["kmeans"]
    ks = np.array(km["PC10"]["ks"])
    ax = fig.add_axes(R_(68, top2, 34, h2))
    ax.plot(ks, km["PC10"]["real_sil"], color="k", marker="o", ms=1.5, lw=1, label="real")
    for nn in NULLS:
        ax.plot(ks, km["PC10"][nn]["sil_mean"], color=MF.COL[nn], label=LAB[nn])
    ax.plot(ks, km["PC10mix25"]["real_sil"], color=MF.COL["mix"], ls="--", label="25-cluster mixture")
    ax.set_xscale("log")
    ax.set_xticks([2, 10, 50]); ax.set_xticklabels(["2", "10", "50"])
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_xlabel("number of k-means clusters")
    ax.set_ylabel("silhouette (top 10 PCs)")
    ax.set_title("no preferred cluster number", pad=2)
    MF.ticks3(ax, x=False)
    ax.legend(loc="upper right", handlelength=1.2, fontsize=4, frameon=False)
    MF.letter(ax, "d")

    # e: response profiles (left) and RRR task-variable selectivity (right, rrr_epairs.py)
    ax = fig.add_axes(R_(118, top2, 63, h2))
    ep = R["epairs"]
    EX = json.loads((HERE.parent / "cache" / "extra_nulls.json").read_text())["epairs"]
    RR = json.loads((HERE.parent / "cache" / "rrr_epairs.json").read_text())["epairs"]
    # each data column is followed by the mixture control fitted in the same space
    rows = [("ALL|xprobe|all", "all bins"), ("ALLmix25|all|all", "mixture"),
            ("PC10|xprobe|all", "10 PCs"), ("PC10mix25|all|all", "mixture"),
            ("RRRsigned|xsession|all", "signed"),
            ("RRRsignedmix25|all|all", "mixture")]

    def src(key, nn):
        if key.startswith("RRR"):
            return RR
        if key.startswith("ALL"):
            return FS
        return ep if nn in NULLS else EX

    for i, (key, _) in enumerate(rows):
        mix = "mix" in key
        for nn, dx in zip(NULLS_E, (-0.21, -0.07, 0.07, 0.21)):
            ax.plot(i + dx, src(key, nn)[key][nn]["5"]["auc"], "o", ms=2.5,
                    mfc="white" if mix else COL_E[nn], mec=COL_E[nn], mew=0.6,
                    label=LAB[nn] if i == 0 else None)
    for x in (1.5, 3.5):
        ax.axvline(x, color="0.7", lw=0.4, ls=":")
    ax.text(1.5, 0.765, "response profiles", ha="center", va="bottom", fontsize=5)
    ax.text(4.5, 0.765, "task-variable\nselectivity (RRR)", ha="center", va="bottom", fontsize=5)
    ax.axhline(0.5, color="0.6", lw=0.4)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r[1] for r in rows], fontsize=4.5)
    for lab, (key, _) in zip(ax.get_xticklabels(), rows):
        if "mix" in key:
            lab.set_color("0.45"); lab.set_fontstyle("italic")
    ax.set_ylim(0.45, 0.75); ax.set_yticks([0.5, 0.6, 0.7])
    ax.set_ylabel("excess similarity of the\nmost similar neurons, AUC", linespacing=1.3)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.0), handlelength=0.8, ncol=2,
              fontsize=4, columnspacing=0.8, frameon=False)
    MF.letter(ax, "e", x=-0.16)

    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"response_correlations.{ext}", dpi=400)
    print("saved", HERE / "response_correlations.pdf")


if __name__ == "__main__":
    main()
