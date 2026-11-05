#!/usr/bin/env python3
"""Fig. S6 (rebuilt 2026-10-12): the 100 basis vectors of Fig. 4, in Fig. 4's column order.

Basis V and coefficients C are read from Fig. 4's own synthetic-analysis cache
(compute_synthetic.load(nclus=100): V = mean z-scored feature vector of each of the 100
k-means basis clusters, C = X V^T). Columns are ordered exactly as in Fig. 4
(make_structured_mixed_selectivity.horder: average-linkage leaf order on 1 - corr(C)), so row k here is the
basis vector of column (alpha) k in Fig. 4a, b, d-g. The earlier S6 plotted the means of
a separate k-means run (dmn_bwm.plot_sorted_cluster_lines), whose cluster numbering need
not match Fig. 4's basis; its corr(C) panel duplicated Fig. 4b and is dropped.
Writes basis_vectors_k100.pdf/.png and basis_vectors_k100_printer.pdf (183 mm wide).
"""

import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from compute_synthetic import load  # noqa: E402
from make_structured_mixed_selectivity import horder  # noqa: E402
from response_correlations_common import use_private_base  # noqa: E402

MM = 1 / 25.4
NCLUS = 100


def main():
    d = use_private_base()
    r = load(nclus=NCLUS)
    V = np.asarray(r["V"], float)[:NCLUS]
    C = np.asarray(r["C"], float)[:, :NCLUS]
    corr = np.nan_to_num(np.corrcoef(C, rowvar=False))
    order = horder(corr)  # Fig. 4 column order
    dist = np.clip(1 - corr, 0, None); np.fill_diagonal(dist, 0)
    Z = hierarchy.linkage(squareform(dist, checks=False), method="average")
    assert np.array_equal(hierarchy.leaves_list(Z), order)

    lens = list(d.regional_group("kmeans", vers="concat", cv=False, nclus=25)["len"].items())
    peth = d.peth_dictm if hasattr(d, "peth_dictm") else {k: k for k, _ in lens}
    n_bins = V.shape[1]
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans"],
                         "font.size": 5, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(183 * MM, 230 * MM))
    gs = fig.add_gridspec(NCLUS, 2, width_ratios=[1, 7], wspace=0.01, hspace=0,
                          left=0.02, right=0.96, top=0.93, bottom=0.01)
    axd = fig.add_subplot(gs[:, 0])
    hierarchy.dendrogram(Z, orientation="left", ax=axd, no_labels=True, color_threshold=0,
                         above_threshold_color="k", link_color_func=lambda _: "k")
    axd.set_ylim(10 * NCLUS, 0); axd.axis("off")
    for line in axd.get_lines():
        line.set_linewidth(0.3)
    cmap = plt.get_cmap("tab10")
    for k, a in enumerate(order):
        ax = fig.add_subplot(gs[k, 1])
        ax.plot(V[a], color=cmap(k % 10), lw=0.5)
        ax.set_xlim(0, n_bins); ax.axis("off")
        ax.text(1.005, 0.5, str(k), transform=ax.transAxes, fontsize=3.8, va="center")
        h = 0
        for name, n in lens:
            h += n
            if h < n_bins:
                ax.axvline(h, color="0.6", lw=0.2, ls=(0, (2, 2)))
            if k == 0:
                ax.text(h - n / 2, 1.6, peth.get(name, name), rotation=60, fontsize=4.2,
                        ha="left", va="bottom", transform=ax.get_xaxis_transform())
    fig.text(0.985, 0.5, "coefficient index $\\alpha$ (as in Fig. 4)", rotation=270, va="center",
             fontsize=5)
    stem = HERE / "basis_vectors_k100"
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white")
    subprocess.run(["gs", "-q", "-dNOPAUSE", "-dBATCH", "-dSAFER", "-sDEVICE=pdfwrite",
                    "-dCompatibilityLevel=1.5", "-dPDFSETTINGS=/printer",
                    f"-sOutputFile={stem}_printer.pdf", str(stem.with_suffix(".pdf"))], check=True)
    print(f"Saved {stem}.pdf/.png and {stem.name}_printer.pdf")


if __name__ == "__main__":
    main()
