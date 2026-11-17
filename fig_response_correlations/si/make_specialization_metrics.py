#!/usr/bin/env python3
"""Fig. S5 (rebuilt 2026-10-12): specialization metrics, plus former Fig. 4j.

Panels a-c are the existing Inkscape panels of figures/specialization_metric_comparison.pdf
(dmn_bwm.scat_dec_clus), reordered: a = old b (clustering vs decoding specialization
across cluster numbers), b = old a (clustering specialization vs Harris hierarchy),
c unchanged (decoding specialization vs hierarchy). Old panel letters are masked and
redrawn. Panel d is former Fig. 4j, recomputed with make_structured_mixed_selectivity.compute (100-cluster
basis, row-aligned Beryl labels): standard error of the first-PC coefficient score
within Beryl regions (>= 20 neurons) versus within size-matched random groups, with a
shuffle test on the mean SE (N_PERM region-label permutations).

Panels d and e merged 2026-11-16; since 2026-11-16 (later) panel d is make_regional_diversity_fullspace.py
(full feature space, no projection); panel_d() below is the superseded PC1 version, kept for reference.
which shrinks with group size) of the same score for Beryl regions, size-matched random
groups and the 100 k-means groups. k-means groups minimise this spread by construction,
so they are a lower reference, not a test.

Writes specialization_metric_comparison_d.pdf (panel d), _e.pdf (panel e), specialization_metric_comparison.pdf
(assembled with tectonic) and specialization_metric_comparison_values.txt in this folder.
"""

import subprocess
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from response_correlations_common import use_private_base  # noqa: E402
from make_structured_mixed_selectivity import MIN_GROUP_N, compute, normalized_emd, se_per_group  # noqa: E402
from compute_synthetic import load  # noqa: E402

MM = 1 / 25.4
N_PERM = 1000
OLD = Path.home() / "dmn/manuscript/IBL_supersession_paper_revision/figures/specialization_metric_comparison.pdf"
SRC = HERE / "specialization_metric_comparison_abc_source.pdf"  # frozen copy of the old a-c

TEX = r"""\documentclass{standalone}
\usepackage{graphicx,tikz}
\usepackage{fontspec}\setmainfont{Liberation Sans}
\newcommand{\src}{SRCPATH}
\newcommand{\pan}[2]{\includegraphics[trim=#1 0 #2 0,clip]{\src}}
\begin{document}
\begin{tikzpicture}[inner sep=0]
% old b (x 120-308 pt) -> a
\node[anchor=north west] (A) at (0,0) {\pan{120bp}{110bp}};
\fill[white] (147.5bp-120bp,-4.0bp) rectangle (156.5bp-120bp,-11.7bp);
% old a (x 0-120 pt) -> b
\node[anchor=north west] (B) at (190bp,0) {\pan{0bp}{298bp}};
\fill[white] (190bp+6.5bp,-8.2bp) rectangle (190bp+12.7bp,-15.4bp);
% old c (x 308-418 pt)
\node[anchor=north west] (C) at (312bp,0) {\pan{308bp}{0bp}};
\fill[white] (312bp+6.8bp,-10.3bp) rectangle (312bp+12.0bp,-17.5bp);
% new d
\node[anchor=north west] (D) at (428bp,0) {\includegraphics[height=129bp]{DPATH}};
\foreach \x/\l in {2bp/a, 192bp/b, 314bp/c, 428bp/d}
  \node[anchor=north west,font=\bfseries\large] at (\x,-1bp) {\l};
\end{tikzpicture}
\end{document}
"""


def sd_per_group(score, labels, exclude=("root", "void")):
    out = []
    for lab in np.unique(labels):
        if str(lab) in exclude:
            continue
        x = score[labels == lab]
        x = x[np.isfinite(x)]
        if x.size >= MIN_GROUP_N:
            out.append(np.std(x, ddof=1))
    return np.asarray(out)


def functional_labels(beryl, clusters, rng):
    """Relabel neurons so that groups have exactly the Beryl region sizes but are
    functionally homogeneous: neurons are ordered by k-means cluster (clusters in random
    order, neurons in random order within a cluster) and cut into contiguous blocks of
    the region sizes (regions in random order)."""
    _, inv = np.unique(clusters, return_inverse=True)
    key = rng.permutation(inv.max() + 1)[inv]  # random cluster order
    order = np.lexsort((rng.random(len(clusters)), key))
    labs, counts = np.unique(beryl, return_counts=True)
    perm = rng.permutation(len(labs))
    out = np.empty_like(beryl)
    out[order] = np.repeat(labs[perm], counts[perm])
    return out


def panel_d():
    """Single panel replacing former d (SE) and e (s.d.): within-group s.d. of the PC1
    coefficient score for Beryl regions, random groups and k-means-based functional groups,
    the latter two with exactly the regions' sizes."""
    v = compute(100, "rastermap")
    r = load(nclus=100)
    beryl, clusters = use_private_base().synthetic_row_labels(r)
    beryl, clusters = np.asarray(beryl), np.asarray(clusters)
    pc = v["pc_real"]
    rng = np.random.default_rng(1)
    reg = sd_per_group(pc, beryl)
    rand_all = [sd_per_group(pc, rng.permutation(beryl)) for _ in range(N_PERM)]
    null = np.array([x.mean() for x in rand_all])
    rand = np.concatenate(rand_all)
    p = (1 + (null <= reg.mean()).sum()) / (1 + N_PERM)  # one-sided: regions less diverse
    N_FUNC = 20
    func_all = [sd_per_group(pc, functional_labels(beryl, clusters, rng)) for _ in range(N_FUNC)]
    func = np.concatenate(func_all)
    fnull = np.array([x.mean() for x in func_all])
    pos = (reg.mean() - func.mean()) / (rand.mean() - func.mean())

    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans"],
                         "font.size": 5.5, "pdf.fonttype": 42, "axes.linewidth": 0.6})
    fig, ax = plt.subplots(figsize=(58 * MM, 129 / 72 * 25.4 * MM))
    bins = np.linspace(0, max(reg.max(), rand.max(), func.max()) * 1.02, 40)
    for x, lab, col, w in ((func, "functional groups", "green", 1 / N_FUNC),
                           (reg, f"Beryl regions ({len(reg)})", "tab:blue", 1),
                           (rand, "random groups", "red", 1 / N_PERM)):
        ax.hist(x, bins=bins, histtype="step", lw=1, color=col, label=lab, weights=np.full(len(x), w))
    ax.set_xlabel("s.d. of PC1 coefficient score per group")
    ax.set_ylabel("groups")
    ax.spines[["top", "right"]].set_visible(False)
    ax.locator_params(nbins=3)
    ax.legend(frameon=False, loc="upper left", fontsize=5)
    ax.text(0.98, 0.62, f"mean s.d.\nfunctional {func.mean():.0f}\nregions {reg.mean():.0f}\n"
            f"random {rand.mean():.0f}\nregions vs random:\n$p$ = {p:.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=5)
    fig.subplots_adjust(left=0.2, right=0.97, top=0.9, bottom=0.2)
    out = HERE / "specialization_metric_comparison_d.pdf"
    fig.savefig(out)
    plt.close(fig)
    txt = (f"d  within-group s.d. of the PC1 score, all groups with the sizes of the {len(reg)} Beryl "
           f"regions (>= {MIN_GROUP_N} neurons): regions mean {reg.mean():.1f}; random groups "
           f"{null.mean():.1f} +- {null.std():.1f} ({N_PERM} allocations; regions "
           f"{100 * (1 - reg.mean() / null.mean()):.1f}% lower, one-sided p = {p:.4f}); functional "
           f"groups (k-means-ordered blocks) {fnull.mean():.1f} +- {fnull.std():.1f} ({N_FUNC} "
           f"allocations); regions at {100 * pos:.0f}% of the way from functional to random\n")
    (HERE / "specialization_metric_comparison_values.txt").write_text(txt)
    print(txt)
    return out


def main():
    if not SRC.exists():
        SRC.write_bytes(OLD.read_bytes())
    import make_regional_diversity_fullspace as fs
    fs.main()
    d = HERE / "regional_diversity_fullspace.pdf"
    tex = HERE / "specialization_metric_comparison.tex"
    tex.write_text(TEX.replace("SRCPATH", str(SRC)).replace("DPATH", str(d)))
    subprocess.run(["tectonic", "-X", "compile", str(tex), "--outdir", str(HERE)], check=True,
                   capture_output=True)
    print(f"Saved {HERE / 'specialization_metric_comparison.pdf'}")


if __name__ == "__main__":
    main()
