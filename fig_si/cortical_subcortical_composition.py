#!/usr/bin/env python3
"""Cortical--subcortical separation in functional-cluster composition space (old Fig. S11,
dmn_bwm.si_quantify_cortical_sub), reproduced on the all-trials data of Figs. 2-3 and checked.

Data and groups as cortex_rate_analysis.load(): concat_cvFalse.npy with its 25-cluster
k-means labels; cortical = Cosmos Isocortex, OLF, HPF, CTXsp; subcortical = CNU, TH, HY,
MB, HB, CB; neurons with Beryl or Cosmos root/void removed. Per Beryl region with >= NMIN
neurons: composition vector p_r = fraction of the region's neurons in each cluster.

  a  cell-count-weighted Fisher criterion D (centroid distance / pooled RMS deviation)
     vs 1,000 region-label shuffles (group sizes kept)
  b  PERMANOVA pseudo-F on Euclidean distances between composition vectors (skbio,
     unweighted), with its shuffle null
  d  PCA of the composition vectors (rows weighted by sqrt(n) for the fit; projection of
     the unweighted vectors after subtracting the weighted mean, so that the axes are
     centred -- the original centred on the weighted rows, shifting both groups)
  c  specialization 1 - H(p)/log K per region, cortical vs subcortical (Mann-Whitney U;
     the original reported only a normalised EMD)
(Rate-matched D, previously panel e, removed 2026-10-12; see cortex_rate_analysis.py.)
Matches dmn_bwm.si_quantify_cortical_sub(cv=False) for a and b (D 0.618, F 10.22).
Writes cortical_subcortical_composition.pdf/.png, results/cortical_subcortical_composition.txt.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy, mannwhitneyu
from skbio.stats.distance import DistanceMatrix, permanova

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cortex_rate_analysis as CR  # noqa: E402
import make_cortex_rate as MCR  # noqa: E402

N_PERM = 1000
SEED = 0
MM = 1 / 25.4
C_CTX, C_SUB = MCR.C_CTX, MCR.C_SUB


def main():
    rng = np.random.default_rng(SEED)
    br, df, _ = CR.load()
    regs, P, N, ctx = CR.composition(df)
    n = len(regs)
    # a  Fisher D
    D = CR.fisher_d(P, N, ctx)
    D_null = np.array([CR.fisher_d(P, N, rng.permutation(ctx)) for _ in range(N_PERM)])
    p_D = (1 + (D_null >= D).sum()) / (1 + N_PERM)
    # b  PERMANOVA
    dist = squareform(pdist(P))
    grp = np.where(ctx, "cortical", "subcortical")
    res = permanova(DistanceMatrix(dist, ids=list(regs)), pd.Series(grp, index=list(regs), name="group"), column="group",
                    permutations=N_PERM)
    F, p_F = float(res["test statistic"]), float(res["p-value"])
    R2 = F / (F + n - 2)
    D2 = dist ** 2

    def pseudo_f(g):
        sst = D2.sum() / (2 * n)
        ssw = sum(D2[np.ix_(g == k, g == k)].sum() / (2 * (g == k).sum()) for k in (True, False))
        return (sst - ssw) / (ssw / (n - 2))
    F_null = np.array([pseudo_f(rng.permutation(ctx)) for _ in range(N_PERM)])
    # c  PCA (weighted fit, centred projection)
    w = np.sqrt(N)[:, None]
    mu = (P * N[:, None]).sum(0) / N.sum()
    U, S, Vt = np.linalg.svd((P - mu) * w, full_matrices=False)
    var = S ** 2 / (S ** 2).sum()
    PC = (P - mu) @ Vt[:2].T
    # d  specialization
    spec = np.array([1 - entropy(p) / np.log(P.shape[1]) for p in P])
    p_spec = mannwhitneyu(spec[ctx], spec[~ctx]).pvalue

    lines = [f"regions (>= {CR.NMIN} neurons): {n} ({ctx.sum()} cortical, {(~ctx).sum()} subcortical); "
             f"neurons {len(df):,}",
             f"Fisher D = {D:.3f} (shuffle {D_null.mean():.3f} +- {D_null.std():.3f}; p = {p_D:.3f})",
             f"PERMANOVA pseudo-F = {F:.2f}, R2 = {R2:.3f}, p = {p_F:.3f}",
             f"PCA: PC1 {100 * var[0]:.1f}%, PC2 {100 * var[1]:.1f}%; centroids cortical "
             f"({PC[ctx, 0].mean():.3f}, {PC[ctx, 1].mean():.3f}), subcortical "
             f"({PC[~ctx, 0].mean():.3f}, {PC[~ctx, 1].mean():.3f})",
             f"specialization: cortical median {np.median(spec[ctx]):.3f} (mean {spec[ctx].mean():.3f}), "
             f"subcortical median {np.median(spec[~ctx]):.3f} (mean {spec[~ctx].mean():.3f}); "
             f"Mann-Whitney p = {p_spec:.2f}"]
    (HERE / "results" / "cortical_subcortical_composition.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # figure (Beryl colours first: importing dmn_bwm resets rcParams)
    cols = CR_colors(regs)
    MCR.style()
    fig = plt.figure(figsize=(183 * MM, 120 * MM))
    g = fig.add_gridspec(2, 2, wspace=0.3, hspace=0.45, left=0.09, right=0.98, top=0.95, bottom=0.08)
    for k, (null, obs, lab, txt) in enumerate((
            (D_null, D, "Fisher criterion $D$", f"$D={D:.2f}$\n$p={p_D:.3f}$"),
            (F_null, F, "PERMANOVA pseudo-$F$", f"$F={F:.1f}$, $R^2={R2:.3f}$\n$p={p_F:.3f}$"))):
        ax = fig.add_subplot(g[0, k])
        ax.hist(null, bins=40, color="0.8", edgecolor="none")
        ax.axvline(obs, color="k", lw=1)
        ax.text(0.9, 0.6, txt, transform=ax.transAxes, ha="right", va="top", fontsize=5)
        ax.set_xlabel(lab); ax.set_ylabel("shuffles")
        ax.locator_params(nbins=3)
        MCR.letter(ax, "ab"[k], -0.15)
    ax = fig.add_subplot(g[1, 0])
    bins = np.linspace(0, spec.max() * 1.02, 25)
    for m, c, lab in ((ctx, C_CTX, "cortical"), (~ctx, C_SUB, "subcortical")):
        ax.hist(spec[m], bins=bins, density=True, histtype="step", color=c, lw=1,
                label=f"{lab} ({m.sum()})")
    ax.text(0.97, 0.6, f"Mann–Whitney\n$p={p_spec:.2f}$", transform=ax.transAxes, ha="right", fontsize=5)
    ax.legend(frameon=False, fontsize=4.8, loc="upper right")
    ax.set_xlabel("specialization $1-H/\\log K$"); ax.set_ylabel("density"); ax.locator_params(nbins=3)
    MCR.letter(ax, "c", -0.15)
    ax = fig.add_subplot(g[1, 1])
    for m, mk, c in ((ctx, "o", C_CTX), (~ctx, "s", C_SUB)):
        ax.scatter(PC[m, 0], PC[m, 1], c=cols[m], marker=mk, s=8, edgecolors="none", zorder=3)
        cov = np.cov(PC[m, 0], PC[m, 1]); vals, vecs = np.linalg.eigh(cov)
        ang = np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1]))
        ax.add_patch(Ellipse(PC[m].mean(0), 2 * np.sqrt(vals[-1]), 2 * np.sqrt(vals[0]), angle=ang,
                             fill=False, ls="--", lw=0.8, color=c, zorder=4))
    for i in np.argsort(N)[::-1][:10]:
        ax.text(PC[i, 0], PC[i, 1] + 0.01, regs[i], fontsize=4, ha="center", color=cols[i])
    ax.set_xlabel(f"PC1 ({100 * var[0]:.0f}%)"); ax.set_ylabel(f"PC2 ({100 * var[1]:.0f}%)")
    ax.locator_params(nbins=3)
    ax.text(0.02, 0.97, "circles cortical, squares subcortical;\ndashed, 1 s.d.", transform=ax.transAxes,
            va="top", fontsize=4.5, color="0.3")
    MCR.letter(ax, "d", -0.15)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(HERE / f"cortical_subcortical_composition.{ext}", facecolor="white", **kw)


def CR_colors(regs):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fig_anatomy"))
    import regenerate_from_data as fig_anatomy
    return fig_anatomy.region_colors(None, np.asarray(regs), "Beryl")


if __name__ == "__main__":
    main()
