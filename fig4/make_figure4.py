#!/usr/bin/env python3
"""Assemble Fig. 4 (structured, non-random selectivity) from the dmn_bwm analyses.

Data: dmn_bwm.regional_group(mapping="kmeans", synthetic=True, cv=False,
nclus=100, nclus_s=100) -- real coefficients C onto a 100-cluster k-means basis
and marginal-matched i.i.d. synthetic coefficients B (compute_synthetic.py).
The panel computations follow dmn_bwm.plot_fig4_assembly (the earlier name of
this figure); the layout follows the manuscript. Panel h comes from
regenerate_rastermap_panels.py. --nclus 40 --out-dir si --stem mixed_selectivity_k40
makes Fig. S10 (see si/make_figure_s10.py).

Panels j and k group each row of C by its own Beryl region and real-data
k-means cluster (dmn_bwm.synthetic_row_labels). The manuscript version paired
C's rows with misaligned labels; check_row_alignment.py compares the two.

--sort kmeans replaces every Rastermap neuron order (a, c, e, g, h) by the
canonical k-means sorting of Fig. 2a (25 clusters fit on the real responses,
neurons sorted by cluster): real neurons by their own cluster, synthetic neurons
(no identity) by the nearest of the same 25 centroids. Output stem
structured_mixed_selectivity_kmeans_sort. The alpha (column) order is unchanged.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA

from compute_synthetic import load
from fig4_common import OUT, kmeans_canonical, nearest_centroid, use_private_base

MM = 1 / 25.4
N_SUBSET = 2000
MIN_GROUP_N = 20
CMAP = "coolwarm"


def configure_style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 5, "axes.labelsize": 5.5, "axes.titlesize": 5.5,
        "xtick.labelsize": 4.5, "ytick.labelsize": 4.5, "axes.linewidth": 0.5,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2,
        "ytick.major.size": 2, "legend.fontsize": 4.5, "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })


def horder(mat):
    """Average-linkage leaf order on 1 - correlation (as in plot_fig4_assembly)."""
    mat = np.nan_to_num(np.asarray(mat, float), nan=0.0, posinf=0.0, neginf=0.0)
    dist = np.clip(1.0 - mat, 0.0, None)
    np.fill_diagonal(dist, 0.0)
    return hierarchy.leaves_list(hierarchy.linkage(squareform(dist, checks=False), method="average"))


def normalized_emd(a, b):
    both = np.concatenate([a, b])
    span = np.nanmax(both) - np.nanmin(both)
    return wasserstein_distance(a, b) / span if span > 0 else 0.0


def se_per_group(score, labels, exclude=("root", "void")):
    out = []
    for lab in np.unique(labels):
        if str(lab) in exclude:
            continue
        v = score[labels == lab]
        v = v[np.isfinite(v)]
        if v.size >= MIN_GROUP_N:
            out.append(np.std(v, ddof=1) / np.sqrt(v.size))
    return np.asarray(out)


def kmeans_row_orders(r):
    """Canonical k-means order of C's rows (own cluster, ties in stack order) and of
    B's rows (nearest of the same centroids, applied to the synthetic responses)."""
    labels, centroids = kmeans_canonical(use_private_base())
    rows = np.asarray(r["C_rows"], dtype=int)
    syn = nearest_centroid(np.asarray(r["concat_zs"], dtype=np.float32), centroids)
    return np.lexsort((rows, labels[rows])), np.argsort(syn, kind="stable")


def compute(nclus, sort="rastermap"):
    r = load(nclus=nclus)
    C = np.asarray(r["C"], float)[:, :nclus]
    B = np.asarray(r["B"], float)[:, :nclus]
    if sort == "kmeans":  # row order only; every statistic is order-invariant
        ord_rows_C, ord_rows_B = kmeans_row_orders(r)
    else:
        ord_rows_C = ord_rows_B = np.arange(C.shape[0])
    corr_C, corr_B = np.corrcoef(C, rowvar=False), np.corrcoef(B, rowvar=False)
    ord_C = horder(corr_C)
    idx_n = np.linspace(0, C.shape[0] - 1, N_SUBSET).round().astype(int)
    pca = PCA(n_components=1).fit(C)
    pc_real, pc_syn = pca.transform(C)[:, 0], pca.transform(B)[:, 0]

    beryl, clusters = use_private_base().synthetic_row_labels(r)
    rng = np.random.default_rng(0)
    Cs, Bs = C[ord_rows_C], B[ord_rows_B]
    return dict(
        C=Cs[:, ord_C], B=Bs[:, ord_C], C_raw=C, B_raw=B,
        corr_C=corr_C[np.ix_(ord_C, ord_C)], corr_B=corr_B[np.ix_(ord_C, ord_C)],
        # c, g: 2000 evenly spaced neurons, keeping panel a's / e's order.
        corrN_C=np.corrcoef(Cs[idx_n]), corrN_B=np.corrcoef(Bs[idx_n]),
        pc_real=pc_real, pc_syn=pc_syn,
        se_reg=se_per_group(pc_real, beryl),
        se_rand=se_per_group(pc_real, rng.permutation(beryl)),
        se_km=se_per_group(pc_real, clusters),
    )


def despine(ax):
    ax.spines[["top", "right"]].set_visible(False)


def ticks3(ax):
    ax.xaxis.set_major_locator(MaxNLocator(nbins=2, min_n_ticks=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=2, min_n_ticks=3))


def heat(ax, M, title, xlabel, ylabel, vmin, vmax, aspect="auto", interpolation="nearest"):
    im = ax.imshow(M, vmin=vmin, vmax=vmax, cmap=CMAP, origin="lower", aspect=aspect,
                   interpolation=interpolation, rasterized=True)
    ax.set_title(title, pad=2)
    ax.set_xlabel(xlabel, labelpad=1)
    ax.set_ylabel(ylabel, labelpad=1)
    ticks3(ax)
    return im


def marginals(fig, rect, C, B):
    """d: 15 evenly spaced alphas, synthetic (black) over real (green) histograms."""
    alphas = np.linspace(0, C.shape[1] - 1, 15).round().astype(int)
    x0, y0, w, h = rect
    cw, rh = w / 3, h / 5
    for p, a in enumerate(alphas):
        row, col = p % 5, p // 5
        ax = fig.add_axes([x0 + col * cw, y0 + h - (row + 1) * rh, cw * 0.92, rh * 0.86])
        c, b = C[:, a], B[:, a]
        edges = np.linspace(min(c.min(), b.min()), max(c.max(), b.max()), 201)
        ax.stairs(np.histogram(b, edges, density=True)[0], edges, color="black", lw=1.4, label="synth")
        ax.stairs(np.histogram(c, edges, density=True)[0], edges, color="lime", lw=0.7,
                  ls=":", label="real")
        ax.text(0.03, 0.82, f"α={a}", transform=ax.transAxes, fontsize=3.8)
        ax.axis("off")
        if p == 0:
            first = ax
    first.legend(frameon=False, loc="lower left", bbox_to_anchor=(1.0, 1.0),
                 handlelength=1.5, borderaxespad=0)
    return first


def hist_pair(ax, a, b, labels, colors, xlabel, ylabel, title, bins, density=False):
    for v, lab, col in zip((a, b), labels, colors):
        ax.hist(v, bins=bins, density=density, histtype="step", lw=1, label=lab, color=col)
    ax.set_xlabel(xlabel, labelpad=1)
    ax.set_ylabel(ylabel, labelpad=1)
    ax.set_title(title, pad=2)
    ax.legend(frameon=False, loc="upper right", handlelength=1.5)
    despine(ax)
    ticks3(ax)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rasters", action="store_true", help="re-render panel h images first")
    parser.add_argument("--nclus", type=int, default=100, help="k-means basis size")
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--stem", default=None)
    parser.add_argument("--sort", choices=("rastermap", "kmeans"), default="rastermap",
                        help="neuron order of a, c, e, g, h")
    args = parser.parse_args()
    if args.stem is None:
        args.stem = "structured_mixed_selectivity" + ("_kmeans_sort" if args.sort == "kmeans" else "")
    out = args.out_dir.resolve()
    if args.rasters or not (out / f"panel_h_{args.sort}_synthetic.png").exists():
        subprocess.run([sys.executable, str(OUT / "regenerate_rastermap_panels.py"),
                        "--nclus", str(args.nclus), "--out-dir", str(out), "--sort", args.sort],
                       check=True)

    configure_style()
    v = compute(args.nclus, args.sort)
    fig = plt.figure(figsize=(183 * MM, 107 * MM))
    A = lambda rect: fig.add_axes(rect)  # noqa: E731
    letters = []

    vC = np.percentile(v["C"], [2, 98])
    vB = np.percentile(v["B"], [2, 98])
    vlim = float(max(np.nanmax(np.abs(m)) for m in
                     (v["corr_C"], v["corr_B"], v["corrN_C"], v["corrN_B"])))

    ax = A([0.05, 0.445, 0.14, 0.5]); letters.append((ax, "a"))
    # ~55k neuron rows: antialiased display averages rows, as in the manuscript.
    im = heat(ax, v["C"], "C (real)", "α", "neuron", *vC, interpolation="antialiased")
    cax = ax.inset_axes([0.05, 0.05, 0.05, 0.2])
    cb = fig.colorbar(im, cax=cax, ticks=[int(vC[0] / 100) * 100, int(vC[1] / 100) * 100])
    cb.ax.tick_params(labelsize=4, length=1.5, pad=1)
    cb.outline.set_linewidth(0.3)

    ax = A([0.235, 0.745, 0.1, 0.2]); letters.append((ax, "b"))
    im = heat(ax, v["corr_C"], "corr(C)", "α", "α", -vlim, vlim, aspect="equal")
    cax = ax.inset_axes([1.08, 0.0, 0.06, 0.3])
    cb = fig.colorbar(im, cax=cax, ticks=[-1, 1], format="%.1f")
    cb.ax.tick_params(labelsize=4, length=1.5, pad=1)
    cb.outline.set_linewidth(0.3)
    ax = A([0.235, 0.445, 0.1, 0.2]); letters.append((ax, "c"))
    heat(ax, v["corrN_C"], "corr(C)", "neuron", "neuron", -vlim, vlim, aspect="equal")

    first = marginals(fig, [0.4, 0.43, 0.2, 0.5], v["C_raw"], v["B_raw"])
    letters.append((first, "d"))

    ax = A([0.645, 0.445, 0.13, 0.5]); letters.append((ax, "e"))
    heat(ax, v["B"], "B (synthetic)", "α", "neuron", *vB, interpolation="antialiased")
    ax = A([0.855, 0.745, 0.1, 0.2]); letters.append((ax, "f"))
    heat(ax, v["corr_B"], "corr(B)", "α", "α", -vlim, vlim, aspect="equal")
    ax = A([0.855, 0.445, 0.1, 0.2]); letters.append((ax, "g"))
    heat(ax, v["corrN_B"], "corr(B)", "neuron", "neuron", -vlim, vlim, aspect="equal")

    for k, name in enumerate(("real_reconstruction", "synthetic")):
        ax = A([0.05 + k * 0.125, 0.04, 0.105, 0.3])
        ax.imshow(plt.imread(out / f"panel_h_{args.sort}_{name}.png"), aspect="auto",
                  interpolation="antialiased", rasterized=True)
        ax.axis("off")
        if k == 0:
            letters.append((ax, "h"))

    ax = A([0.4, 0.085, 0.13, 0.25]); letters.append((ax, "i"))
    hist_pair(ax, v["pc_real"], v["pc_syn"], ("real", "synth"), ("black", "orange"),
              "PC0 score", "dens", f"EMD={normalized_emd(v['pc_real'], v['pc_syn']):.3g}",
              bins=60, density=True)
    ax.legend(frameon=False, loc="upper left", handlelength=1.5)

    both = np.concatenate([v["se_reg"], v["se_rand"]])
    ax = A([0.6, 0.085, 0.155, 0.25]); letters.append((ax, "j"))
    hist_pair(ax, v["se_reg"], v["se_rand"], ("real", "random"), ("tab:blue", "red"),
              "SE(PC0) per group", "count",
              f"Beryl regions, EMD={normalized_emd(v['se_reg'], v['se_rand']):.3g}",
              bins=np.linspace(both.min(), both.max(), 30))
    both = np.concatenate([v["se_reg"], v["se_km"]])
    ax = A([0.83, 0.085, 0.155, 0.25]); letters.append((ax, "k"))
    hist_pair(ax, v["se_reg"], v["se_km"], ("Beryl", "KMeans"), ("tab:blue", "green"),
              "SE(PC0) per group", "count",
              f"Beryl vs KMeans, EMD={normalized_emd(v['se_reg'], v['se_km']):.3g}",
              bins=np.linspace(both.min(), both.max(), 30))

    for ax, letter in letters:
        pos = ax.get_position()
        fig.text(pos.x0 - 0.035, pos.y1 + 0.012, letter, fontsize=8, fontweight="bold",
                 ha="left", va="bottom")

    stem = out / args.stem
    (out / f"{args.stem}_values.txt").write_text(
        f"nclus={args.nclus}\n"
        f"i  PC0 real vs synth EMD = {normalized_emd(v['pc_real'], v['pc_syn']):.3g}\n"
        f"j  Beryl vs random EMD = {normalized_emd(v['se_reg'], v['se_rand']):.3g}\n"
        f"k  Beryl vs KMeans EMD = {normalized_emd(v['se_reg'], v['se_km']):.3g}\n")
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
