#!/usr/bin/env python3
"""Compact k-robustness figure for the regional distribution (candidate SI; R3 comment 5).

Uses cache/k_robustness.npy and cache/kmeans_k*.npy (compute_k_robustness.py). Per k:
a  types present per region (fraction of k), real vs size-matched shuffle;
b  fraction of region x type cells within two-fold of the brain-wide share, real vs shuffle;
c  specialization S = 1 - H/log k, real vs shuffle;
d  Spearman correlation of regions' S at each k with S at k = 25.
Regions: Beryl, >= 50 neurons. Shuffle: neurons permuted across regions, sizes kept.
"""
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
MM = 1 / 25.4
N_SHUF = 100


def within2(comp, share):
    e = comp / share
    return ((e >= 0.5) & (e <= 2)).mean()


def main():
    o = np.load(HERE / "cache" / "k_robustness.npy", allow_pickle=True).flat[0]
    ks, sizes = o["ks"], o["sizes"]
    rng = np.random.default_rng(1)
    rows = []
    for k in ks:
        v = o[k]
        lab = np.load(HERE / "cache" / f"kmeans_k{k}.npy")
        # pool = all neurons of kept regions; rebuild via composition counts
        counts = np.rint(v["comp"] * sizes[:, None]).astype(int)
        pool = np.repeat(np.tile(np.arange(k), len(sizes)), counts.ravel())
        nul_w2 = []
        for _ in range(N_SHUF):
            perm = rng.permutation(pool)
            g = np.split(perm, np.cumsum(sizes)[:-1])
            comp = np.array([np.bincount(x, minlength=k) / len(x) for x in g])
            nul_w2.append(within2(comp, v["share"]))
        rows.append(dict(k=k, pres=v["present"], npres=np.median(v["null_present"], 0),
                         w2=within2(v["comp"], v["share"]), nw2=np.array(nul_w2),
                         spec=v["spec"], nspec=np.median(v["null_spec"], 0)))
    s25 = rows[ks.index(25)]["spec"]
    rho = [spearmanr(r["spec"], s25)[0] for r in rows]

    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans"],
                         "font.size": 6, "pdf.fonttype": 42, "axes.linewidth": .5,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "xtick.major.size": 2, "ytick.major.size": 2})
    fig, axs = plt.subplots(1, 4, figsize=(183 * MM, 45 * MM), constrained_layout=True)
    x = np.array(ks)

    def band(ax, vals, col, lab):
        q = np.array([np.percentile(v, [25, 50, 75]) for v in vals])
        ax.fill_between(x, q[:, 0], q[:, 2], color=col, alpha=.25, lw=0)
        ax.plot(x, q[:, 1], "o-", color=col, ms=2.5, lw=1, label=lab)

    a, b, c, d = axs
    band(a, [r["pres"] for r in rows], "k", "regions")
    band(a, [r["npres"] for r in rows], "tab:red", "shuffled")
    a.set(ylabel="types present per region\n(fraction of k)", ylim=(0, 1.05))
    a.legend(frameon=False, loc="lower left")
    b.plot(x, [r["w2"] for r in rows], "o-", color="k", ms=2.5, lw=1)
    q = np.array([np.percentile(r["nw2"], [2.5, 50, 97.5]) for r in rows])
    b.fill_between(x, q[:, 0], q[:, 2], color="tab:red", alpha=.25, lw=0)
    b.plot(x, q[:, 1], "o-", color="tab:red", ms=2.5, lw=1)
    b.set(ylabel="region x type cells within\ntwo-fold of brain-wide share", ylim=(0, 1.05))
    band(c, [r["spec"] for r in rows], "k", "regions")
    band(c, [r["nspec"] for r in rows], "tab:red", "shuffled")
    c.set(ylabel="specialization S")
    d.plot(x, rho, "o-", color="k", ms=2.5, lw=1)
    d.set(ylabel="Spearman ρ of regions' S\nwith S at k = 25", ylim=(0, 1.05))
    for ax, l in zip(axs, "abcd"):
        ax.set_xscale("log"); ax.set_xticks([10, 25, 100]); ax.set_xticklabels(["10", "25", "100"])
        ax.minorticks_off(); ax.set_xlabel("number of k-means clusters k")
        ax.axvline(25, color=".7", lw=.5, ls=":")
        ax.text(-.3, 1.05, l, transform=ax.transAxes, fontweight="bold", fontsize=8, va="top")
    fig.savefig(HERE / "k_robustness.pdf"); fig.savefig(HERE / "k_robustness.png", dpi=300)
    lines = [f"regions {len(sizes)} (Beryl, >=50 neurons)"]
    for r, p in zip(rows, rho):
        lines.append(f"k={r['k']}: types present median {np.median(r['pres'])*r['k']:.0f}/{r['k']} "
                     f"(shuffle {np.median(r['npres'])*r['k']:.0f}); within two-fold {r['w2']:.3f} "
                     f"(shuffle {np.median(r['nw2']):.3f}); median S {np.median(r['spec']):.3f} "
                     f"(shuffle {np.median(r['nspec']):.3f}); rho with k=25 {p:.2f}")
    (HERE / "k_robustness_values.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
