#!/usr/bin/env python3
"""Figure: structured vs random mixed selectivity (reads ./cache from
selectivity_test.py). Output: selectivity_test.pdf/.png in this folder."""

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

HERE = Path(__file__).resolve().parent
MM = 1 / 25.4
COL = dict(real="k", gauss="#2a78d6", pcmarg="#eb6834", ellip="#1baf7a",
           axismarg="#eda100", mix="#e87ba4")
LAB = dict(real="real", gauss="Gaussian (cov.-matched)", pcmarg="PC-marginal",
           ellip="elliptical", axismarg="axis-marginal (Fig. 4 null)",
           mix="25-cluster mixture (pos. ctrl)")
SEL_SHORT = dict(block_pre="block pre", block_stim="block stim", block_move="block move",
                 side_stim="side stim", side_move="side move", choice_post="choice post",
                 congr_stim="congr. stim", congr_move="congr. move",
                 error_stim="error stim", error_move="error move", outcome_fb="outcome fb")


def style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 5, "axes.labelsize": 5.5, "axes.titlesize": 5.5,
        "xtick.labelsize": 5, "ytick.labelsize": 5, "axes.linewidth": 0.5,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2,
        "ytick.major.size": 2, "legend.fontsize": 4.5, "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "lines.linewidth": 0.8,
    })


def ticks3(ax, x=True, y=True):
    if x:
        ax.xaxis.set_major_locator(MaxNLocator(3))
    if y:
        ax.yaxis.set_major_locator(MaxNLocator(3))


def letter(ax, s, x=-0.28):
    ax.text(x, 1.12, s, transform=ax.transAxes, fontsize=7, fontweight="bold",
            va="top", ha="left")


def horder(M):
    d = np.clip(1 - np.nan_to_num(M), 0, None)
    np.fill_diagonal(d, 0)
    return hierarchy.leaves_list(hierarchy.linkage(squareform(d, checks=False), "average"))


SHORT = dict(real="real", gauss="Gaussian", pcmarg="PC-marg.", ellip="elliptical",
             axismarg="axis-marg.", mix="mixture ctrl")


def hist(ax, x, color, label, lw=0.8, ls="-", bins=None):
    h, e = np.histogram(x, bins=bins, density=True)
    ax.plot(0.5 * (e[1:] + e[:-1]), h, color=color, lw=lw, ls=ls, label=label)


def main():
    style()
    R = json.loads((HERE / "cache" / "results.json").read_text())
    A = np.load(HERE / "cache" / "arrays.npz")
    fig = plt.figure(figsize=(183 * MM, 130 * MM))
    gs = fig.add_gridspec(3, 4, left=0.06, right=0.985, top=0.95, bottom=0.07,
                          wspace=0.55, hspace=0.75)

    # ---- a, b: coefficient correlations, k-means basis vs Loewdin basis
    cc = R["coef_corr"]
    for i, (nm, title) in enumerate((("V", "k-means means V"), ("W", "orthonormalised W"))):
        ax = fig.add_subplot(gs[0, i])
        M = A[f"corrC_{nm}"]
        o = horder(A["corrC_V"])
        im = ax.imshow(M[np.ix_(o, o)], cmap="coolwarm", vmin=-1, vmax=1, interpolation="none")
        ax.set_title(f"corr(C), {title}", pad=2)
        ax.set_xlabel("coefficient")
        ax.set_ylabel("coefficient")
        ax.set_xticks([0, 99])
        ax.set_yticks([0, 99])
        ax.set_xticklabels([1, 100])
        ax.set_yticklabels([1, 100])
        d = cc[nm]
        ax.text(1.02, 0.98, f"r(white noise)\n= {d['r_corrC_vs_whitenoise']:.2f}\n"
                f"PC1 {100 * d['pc1_var_frac']:.0f}%", transform=ax.transAxes,
                fontsize=4.5, va="top")
        letter(ax, "ab"[i])
    cax = fig.add_axes([0.20, 0.70, 0.05, 0.007])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[-1, 0, 1])
    cb.outline.set_linewidth(0.3)

    # ---- c: mean |offdiag| per basis and null
    ax = fig.add_subplot(gs[0, 2])
    keys = [("real", "mean_abs_offdiag_real"), ("white noise", "mean_abs_offdiag_whitenoise"),
            ("axis-marg.", "mean_abs_offdiag_margnull"), ("Gaussian", "mean_abs_offdiag_gauss")]
    cols = ["k", "0.6", COL["axismarg"], COL["gauss"]]
    w = 0.2
    for j, ((lab, k), c) in enumerate(zip(keys, cols)):
        ax.bar(np.arange(2) + (j - 1.5) * w, [cc["V"][k], cc["W"][k]], w * 0.9, color=c, label=lab)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["V", "W"])
    for x in (0, 1):
        ax.text(x + 0.5 * w, 0.01, "~0", ha="center", fontsize=4, color=COL["axismarg"])
    ax.set_ylabel("mean |off-diag. corr|")
    ax.legend(loc="upper right", handlelength=0.8)
    ticks3(ax, x=False)
    letter(ax, "c")

    # ---- d: PCA spectrum
    ax = fig.add_subplot(gs[0, 3])
    ev = A["pca_ev"]
    cum = np.cumsum(ev) / ev.sum()
    ax.plot(np.arange(1, 201), cum[:200], color="k")
    for d in (10, 20):
        ax.axvline(d, color="0.6", lw=0.5, ls=":")
        ax.text(d, 0.05, f" {100 * cum[d - 1]:.0f}%", fontsize=4.5, color="0.4")
    ax.set_xscale("log")
    ax.set_xlabel("# PCs of X")
    ax.set_ylabel("cum. variance")
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xticks([1, 10, 100])
    ax.set_xticklabels(["1", "10", "100"])
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    letter(ax, "d")

    # ---- e, f: kNN angle distributions
    def angle_panel(ax, base, nulls, title, mix=None, bins=None):
        hist(ax, A[f"ang_{base}_real"], COL["real"], SHORT["real"], bins=bins, lw=1)
        for nn in nulls:
            hist(ax, A[f"ang_{base}_{nn}"], COL[nn], SHORT[nn], bins=bins)
        if mix is not None:
            hist(ax, A[f"ang_{mix}_real"], COL["mix"], SHORT["mix"], ls="--", bins=bins)
        ax.set_xlabel("mean angle to 5 NN (deg)")
        ax.set_ylabel("density")
        ax.set_title(title, pad=2)
        ticks3(ax)

    ax = fig.add_subplot(gs[1, 0])
    angle_panel(ax, "PC10|xprobe|all", ["gauss", "pcmarg"], "top 10 PCs",
                mix="PC10mix25|all|all", bins=np.linspace(0, 50, 80))
    e = R["epairs"]["PC10|xprobe|all"]
    ax.text(0.98, 0.98, f"AUC {e['gauss']['5']['auc']:.2f} / {e['pcmarg']['5']['auc']:.2f}\n"
            f"$\\Delta$ {e['gauss']['5']['delta_deg']:.1f}$\\degree$ / "
            f"{e['pcmarg']['5']['delta_deg']:.1f}$\\degree$",
            transform=ax.transAxes, ha="right", va="top", fontsize=4.5)
    ax.legend(loc="center right", bbox_to_anchor=(1.06, 0.4), handlelength=1.2, fontsize=4)
    letter(ax, "e")

    ax = fig.add_subplot(gs[1, 1])
    angle_panel(ax, "W100|xprobe|all", ["gauss", "pcmarg", "axismarg"],
                "100 orthonormalised coefficients", bins=np.linspace(15, 80, 80))
    e = R["epairs"]["W100|xprobe|all"]
    ax.text(0.02, 0.98, f"AUC vs Gauss {e['gauss']['5']['auc']:.2f}\n"
            f"vs axis-marg. {e['axismarg']['5']['auc']:.2f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=4.5)
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.5), handlelength=1.2, fontsize=4)
    letter(ax, "f")

    # ---- g: gap statistic
    ax = fig.add_subplot(gs[1, 2])
    km = R["kmeans"]
    ks = np.array(km["PC10"]["ks"])
    for nn in ("gauss", "pcmarg", "ellip"):
        ax.plot(ks, km["PC10"][nn]["gap"], color=COL[nn], marker="o", ms=1.5, label=LAB[nn])
        ax.plot(ks, km["PC20"][nn]["gap"], color=COL[nn], ls=":", lw=0.7)
    ax.plot(ks, km["PC10mix25"]["gauss"]["gap"], color=COL["mix"], ls="--",
            label="mixture vs Gaussian")
    ax.axhline(0, color="0.6", lw=0.4)
    ax.set_xscale("log")
    ax.set_xticks([2, 10, 50])
    ax.set_xticklabels(["2", "10", "50"])
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_xlabel("k (k-means)")
    ax.set_ylabel("gap = log W$_{null}$ - log W$_{real}$")
    ax.set_title("top 10 PCs (dotted: 20)", pad=2)
    ticks3(ax, x=False)
    ax.legend(loc="upper left", handlelength=1.2, fontsize=4)
    letter(ax, "g")

    # ---- h: silhouette
    ax = fig.add_subplot(gs[1, 3])
    ax.plot(ks, km["PC10"]["real_sil"], color="k", marker="o", ms=1.5, lw=1, label="real")
    for nn in ("gauss", "pcmarg", "ellip"):
        ax.plot(ks, km["PC10"][nn]["sil_mean"], color=COL[nn], label=LAB[nn])
    ax.plot(ks, km["PC10mix25"]["real_sil"], color=COL["mix"], ls="--", label="mixture")
    ax.set_xscale("log")
    ax.set_xticks([2, 10, 50])
    ax.set_xticklabels(["2", "10", "50"])
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_xlabel("k (k-means)")
    ax.set_ylabel("silhouette")
    ax.set_title("top 10 PCs", pad=2)
    ticks3(ax, x=False)
    ax.legend(loc="upper right", handlelength=1.2, fontsize=4)
    letter(ax, "h")

    # ---- i: selectivity correlations (lower: observed; upper: predicted noise)
    ax = fig.add_subplot(gs[2, 0])
    names = R["sel_names"]
    C = A["corrS"].copy()
    P = A["corrS_noise_pred"]
    iu = np.triu_indices(len(names), 1)
    C[iu] = P[iu]
    np.fill_diagonal(C, np.nan)
    ax.imshow(C, cmap="coolwarm", vmin=-1, vmax=1, interpolation="none")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels([SEL_SHORT[n] for n in names], rotation=90, fontsize=4)
    ax.set_yticklabels([SEL_SHORT[n] for n in names], fontsize=4)
    ax.tick_params(length=1)
    for i in range(len(names)):
        for j in range(len(names)):
            if i != j and abs(C[i, j]) >= 0.15:
                ax.text(j, i, f"{C[i, j]:.2f}".replace("0.", "."), ha="center", va="center",
                        fontsize=3, color="w" if abs(C[i, j]) > 0.5 else "k")
    ax.set_title("selectivity corr. (upper: noise pred.)", pad=2)
    letter(ax, "i")

    # ---- j: selectivity ePAIRS (all neurons, and top 20 %)
    ax = fig.add_subplot(gs[2, 1])
    bins = np.linspace(0, 50, 70)
    for base, ls in (("SEL|xprobe|all", "-"), ("SEL|xprobe|top20", "--")):
        hist(ax, A[f"ang_{base}_real"], "k", "real" if ls == "-" else None, ls=ls, bins=bins, lw=1)
        for nn in ("gauss", "pcmarg"):
            hist(ax, A[f"ang_{base}_{nn}"], COL[nn], SHORT[nn] if ls == "-" else None,
                 ls=ls, bins=bins)
    ax.set_xlabel("mean angle to 5 NN (deg)")
    ax.set_ylabel("density")
    ax.set_title("11 task selectivities", pad=2)
    ax.text(0.98, 0.98, "solid: all neurons\ndashed: top 20% |s|", transform=ax.transAxes,
            ha="right", va="top", fontsize=4.5)
    ax.legend(loc="center right", bbox_to_anchor=(1.04, 0.45), handlelength=1.2, fontsize=4)
    ticks3(ax)
    letter(ax, "j")

    # ---- k: AUC summary
    ax = fig.add_subplot(gs[2, 2:])
    ep = R["epairs"]
    rows = [("PC10|xprobe|all", "10 PCs"), ("PC10|xprobe|top20", "10 PCs, top 20%"),
            ("PC20|xprobe|all", "20 PCs"), ("W100|xprobe|all", "100 W coef."),
            ("PC10mix25|all|all", "mixture ctrl (10 PCs)"),
            ("SEL|xprobe|all", "selectivity"), ("SEL|xprobe|top20", "selectivity, top 20%"),
            ("SELmix25|all|all", "mixture ctrl (sel.)")]
    off = dict(gauss=-0.2, pcmarg=0, ellip=0.2)
    for i, (key, lab) in enumerate(rows):
        for nn, dx in off.items():
            ax.plot(i + dx, ep[key][nn]["5"]["auc"], "o", ms=2.5, color=COL[nn],
                    label=LAB[nn] if i == 0 else None)
    ax.axhline(0.5, color="0.6", lw=0.4)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r[1] for r in rows], rotation=25, ha="right", fontsize=4.5)
    ax.set_ylabel("AUC  P(angle$_{real}$ < angle$_{null}$)")
    ax.set_ylim(0.45, 0.8)
    ax.set_yticks([0.5, 0.6, 0.7])
    ax.legend(loc="upper right", handlelength=0.8, ncol=3)
    ax.set_title("ePAIRS effect size (5 NN, cross-probe neighbours)", pad=2)
    letter(ax, "k", x=-0.12)

    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"selectivity_test.{ext}", dpi=400)
    print("saved", HERE / "selectivity_test.pdf")


if __name__ == "__main__":
    main()
