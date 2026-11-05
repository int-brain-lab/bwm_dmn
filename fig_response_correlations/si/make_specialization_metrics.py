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

Panel e (2026-11-04; former Fig. 4k as a reference scale): within-group s.d. (not s.e.,
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
% new d, e
\node[anchor=north west] (D) at (428bp,0) {\includegraphics[height=129bp]{DPATH}};
\node[anchor=north west] (E) at (596bp,0) {\includegraphics[height=129bp]{EPATH}};
\foreach \x/\l in {2bp/a, 192bp/b, 314bp/c, 428bp/d, 596bp/e}
  \node[anchor=north west,font=\bfseries\large] at (\x,-1bp) {\l};
\end{tikzpicture}
\end{document}
"""


def panel_d():
    v = compute(100, "rastermap")
    r = load(nclus=100)
    beryl, _ = use_private_base().synthetic_row_labels(r)
    rng = np.random.default_rng(1)
    obs = v["se_reg"].mean()
    null = np.array([se_per_group(v["pc_real"], rng.permutation(beryl)).mean() for _ in range(N_PERM)])
    p = (1 + (null <= obs).sum()) / (1 + N_PERM)  # one-sided: regions less diverse than random
    emd = normalized_emd(v["se_reg"], v["se_rand"])

    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans"],
                         "font.size": 5.5, "pdf.fonttype": 42, "axes.linewidth": 0.6})
    fig, ax = plt.subplots(figsize=(58 * MM, 129 / 72 * 25.4 * MM))
    both = np.concatenate([v["se_reg"], v["se_rand"]])
    bins = np.linspace(both.min(), both.max(), 30)
    for x, lab, col in ((v["se_reg"], "Beryl regions", "tab:blue"), (v["se_rand"], "random groups", "red")):
        ax.hist(x, bins=bins, histtype="step", lw=1, color=col, label=f"{lab} ({len(x)})")
    ax.set_xlabel("SE of PC1 coefficient score per group")
    ax.set_ylabel("count")
    ax.spines[["top", "right"]].set_visible(False)
    ax.locator_params(nbins=3)
    ax.legend(frameon=False, loc="upper right", fontsize=5)
    ax.text(0.97, 0.5, f"EMD = {emd:.3f}\nmean SE vs shuffle:\n$p$ = {p:.3f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=5)
    fig.subplots_adjust(left=0.2, right=0.97, top=0.9, bottom=0.2)
    out = HERE / "specialization_metric_comparison_d.pdf"
    fig.savefig(out)
    plt.close(fig)
    txt = (f"d  regions (>= 20 neurons) {len(v['se_reg'])}; mean SE real {obs:.1f}, shuffle "
           f"{null.mean():.1f} +- {null.std():.1f}; one-sided p (regions less diverse) = {p:.3f}; "
           f"EMD real vs one random allocation = {emd:.4f}\n")
    (HERE / "specialization_metric_comparison_values.txt").write_text(txt)
    print(txt)
    return out, v, beryl


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


def panel_e(v, beryl):
    r = load(nclus=100)
    _, clusters = use_private_base().synthetic_row_labels(r)
    pc = v["pc_real"]
    rng = np.random.default_rng(1)
    reg = sd_per_group(pc, beryl)
    rand = np.concatenate([sd_per_group(pc, rng.permutation(beryl)) for _ in range(20)])
    km = sd_per_group(pc, clusters)
    pos = (reg.mean() - km.mean()) / (rand.mean() - km.mean())
    fig, ax = plt.subplots(figsize=(58 * MM, 129 / 72 * 25.4 * MM))
    bins = np.linspace(0, max(reg.max(), rand.max()) * 1.02, 36)
    for x, lab, col, w in ((km, f"k-means groups ({len(km)})", "green", None),
                           (reg, f"Beryl regions ({len(reg)})", "tab:blue", None),
                           (rand, "random groups (20 draws)", "red", 1 / 20)):
        ax.hist(x, bins=bins, histtype="step", lw=1, color=col, label=lab,
                weights=None if w is None else np.full(len(x), w))
    ax.set_xlabel("s.d. of PC1 coefficient score per group")
    ax.set_ylabel("count")
    ax.spines[["top", "right"]].set_visible(False)
    ax.locator_params(nbins=3)
    ax.legend(frameon=False, loc="upper left", fontsize=5)
    ax.text(0.33, 0.62, f"mean s.d.\nk-means {km.mean():.0f}\nregions {reg.mean():.0f}\n"
            f"random {rand.mean():.0f}", transform=ax.transAxes, ha="left", va="top", fontsize=5)
    fig.subplots_adjust(left=0.2, right=0.97, top=0.9, bottom=0.2)
    out = HERE / "specialization_metric_comparison_e.pdf"
    fig.savefig(out)
    plt.close(fig)
    txt = (f"e  within-group s.d. of the PC1 score: k-means groups {len(km)} mean {km.mean():.1f}; "
           f"Beryl regions {len(reg)} mean {reg.mean():.1f}; random groups (20 allocations) mean "
           f"{rand.mean():.1f}; regions at {100 * pos:.0f}% of the way from k-means to random\n")
    with open(HERE / "specialization_metric_comparison_values.txt", "a") as fh:
        fh.write(txt)
    print(txt)
    return out


def main():
    if not SRC.exists():
        SRC.write_bytes(OLD.read_bytes())
    d, v, beryl = panel_d()
    e = panel_e(v, beryl)
    tex = HERE / "specialization_metric_comparison.tex"
    tex.write_text(TEX.replace("SRCPATH", str(SRC)).replace("DPATH", str(d)).replace("EPATH", str(e)))
    subprocess.run(["tectonic", "-X", "compile", str(tex), "--outdir", str(HERE)], check=True,
                   capture_output=True)
    print(f"Saved {HERE / 'specialization_metric_comparison.pdf'}")


if __name__ == "__main__":
    main()
