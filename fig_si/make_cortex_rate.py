#!/usr/bin/env python3
"""Revised old Fig. S11: the cortical / subcortical divide is mainly firing rate.

Reads results/cortex_rate.npz (cortex_rate_analysis.py); writes cortex_rate.pdf/.png
and cortex_rate_printer.pdf (183 mm wide). Panels:
  a  rate per Cosmos area (all neurons)       b  median rate per Beryl region, ranked
  c  per-neuron cortical vs subcortical decoding (insertion-grouped CV AUC)
  d  old S11 composition separation (Fisher D): all, same-size random, rate-matched,
     and rate composition instead of response-type composition
  e  k-means clusters (Fig. 3a): cortical fraction vs median rate
  f  Fig. 5 Rastermap clusters (= Fig. 5m): same, sparse / dense clusters marked
  g  sparse and dense cells in cortex vs subcortex, all and rate-matched
panel_region_scatter (Beryl regions: median rate vs number of neurons) is drawn in Fig. 5n.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sequence_analysis"))
import latency_tiling_groups  # noqa: E402

MM = 1 / 25.4
W, H = 183, 138
C_CTX, C_SUB = "#2e8b3d", "#c2409f"
C_SPARSE, C_DENSE = "#1f77b4", "#ff7f0e"


def style():
    matplotlib.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 6, "axes.labelsize": 6, "xtick.labelsize": 5, "ytick.labelsize": 5,
        "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2, "ytick.major.size": 2, "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans",
        "mathtext.it": "Liberation Sans:italic"})


def cosmos_colors(names):
    from iblatlas.regions import BrainRegions
    br = BrainRegions()
    col = br.get(br.acronym2id(np.asarray(names), mapping="Cosmos")).rgb / 255
    col[list(names).index("CB")] = (0.85, 0.75, 0.25)  # readable instead of pale yellow
    return dict(zip(names, col))


def letter(ax, s, x=-0.18, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")


def log_hz_axis(ax, which="x"):
    ticks = [1, 10, 100]
    (ax.set_xticks if which == "x" else ax.set_yticks)(np.log10(ticks))
    (ax.set_xticklabels if which == "x" else ax.set_yticklabels)([str(t) for t in ticks])


def panel_a(ax, R, cc):
    names = list(R["cosmos"])
    data = list(R["cosmos_lograte"])
    pos = np.arange(len(names))[::-1]
    parts = ax.violinplot(data, positions=pos, vert=False, widths=0.85, showextrema=False)
    for b, n in zip(parts["bodies"], names):
        b.set_facecolor(cc[n]); b.set_edgecolor("none"); b.set_alpha(0.9)
    for p, d in zip(pos, data):
        q = np.quantile(d, [0.25, 0.5, 0.75])
        ax.plot(q[[0, 2]], [p, p], color="k", lw=0.8)
        ax.plot(q[1], p, "o", ms=2.2, color="white", mec="k", mew=0.4)
    ax.set_yticks(pos); ax.set_yticklabels(names)
    ax.set_xlim(-1.5, 2.3); log_hz_axis(ax)
    ax.set_xlabel("firing rate (Hz)")
    n_ctx = 4
    for lo, hi, txt, c in ((pos[n_ctx - 1], pos[0], "cortical", C_CTX), (pos[-1], pos[n_ctx], "subcortical", C_SUB)):
        ax.plot([2.25, 2.25], [lo - 0.35, hi + 0.35], color=c, lw=1.5, clip_on=False)
        ax.text(2.33, (lo + hi) / 2, txt, color=c, rotation=270, va="center", ha="left", clip_on=False)


def panel_b(ax, R, cc):
    med, names, cos = np.log10(R["reg_med"]), R["reg_names"], R["reg_cosmos"]
    x = np.arange(med.size)
    lo = np.log10(0.5)
    ax.bar(x, med - lo, bottom=lo, width=1.0, color=[cc[c] for c in cos], linewidth=0)
    ax.text(0.03, 0.04, "lowest: " + ", ".join(names[:5]), transform=ax.transAxes, fontsize=4.8,
            color="0.15", va="bottom", bbox=dict(fc="white", ec="none", pad=0.6))
    ax.set_xlim(-1, med.size); ax.set_ylim(lo, np.log10(60))
    ax.set_yticks(np.log10([1, 10])); ax.set_yticklabels(["1", "10"])
    ax.set_xticks([0, med.size - 1]); ax.set_xticklabels(["1", str(med.size)])
    ax.set_xlabel("Beryl region, ranked"); ax.set_ylabel("median rate (Hz)")
    # cortical share along the ranking (running window)
    ctx = R["reg_ctx"].astype(float)
    k = 21
    run = np.convolve(ctx, np.ones(k) / k, mode="valid")
    ax2 = ax.inset_axes([0.12, 0.6, 0.34, 0.3])
    ax2.plot(np.arange(run.size) + k // 2, run, color=C_CTX, lw=0.9)
    ax2.axhline(ctx.mean(), color="0.5", lw=0.4, ls=":")
    ax2.set_xlim(0, med.size); ax2.set_ylim(0, 1)
    ax2.set_xticks([]); ax2.set_yticks([0, 1])
    ax2.set_ylabel("cortical\nfraction", fontsize=5, labelpad=1)
    ax2.tick_params(labelsize=4.5)


def panel_region_scatter(ax, R, colors):
    """One dot per Beryl region, Beryl colour: median rate vs number of neurons (Fig. 5n)."""
    med, n, names = np.log10(R["reg_med"]), np.log10(R["reg_n"]), R["reg_names"]
    ax.scatter(med, n, s=6, c=colors, edgecolors="none", zorder=3)
    lab = set(np.argsort(med)[:4]) | set(np.argsort(med)[-4:]) | set(np.argsort(n)[-4:])
    for i in lab:
        ax.text(med[i], n[i] + 0.06, names[i], fontsize=4.2, ha="center", va="bottom", color=colors[i])
    ax.set_xlim(np.log10(0.7), np.log10(60)); log_hz_axis(ax)
    ax.set_ylim(np.log10(15), np.log10(6000))
    ax.set_yticks(np.log10([10 ** 2, 10 ** 3])); ax.set_yticklabels(["100", "1,000"])
    ax.set_xlabel("median firing rate (Hz)"); ax.set_ylabel("neurons per region")


def beryl_colors(names):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fig_anatomy"))
    import regenerate_from_data as fig_anatomy  # Beryl colours of Fig. 3 (dmn_bwm.pal)
    return fig_anatomy.region_colors(None, names, "Beryl")


def panel_c(ax, R):
    names, auc = list(R["dec_names"]), R["dec_auc"]
    order = ["rate", "reliability", "latency", "k-means cluster", "response shape", "shape + rate",
             "rate, rate-matched", "k-means cluster, rate-matched", "response shape, rate-matched"]
    lab = ["rate", "reliability", "peak latency", "k-means cluster", "response shape", "shape + rate",
           "rate", "k-means cluster", "response shape"]
    v = np.array([auc[names.index(o)] for o in order])
    y = np.array([10, 9, 8, 7, 6, 5, 2.8, 1.8, 0.8])
    cols = ["k" if o.startswith("rate") or o == "shape + rate" else "0.55" for o in order]
    ax.barh(y, v - 0.5, left=0.5, color=cols, height=0.7)
    for yi, vi in zip(y, v):
        ax.text(max(vi, 0.5) + 0.003, yi, f"{vi:.2f}", va="center", fontsize=4.5)
    ax.set_yticks(y); ax.set_yticklabels(lab)
    ax.text(-0.03, 3.6, "rate-matched neurons:", fontsize=5, color="0.3", va="bottom", ha="right",
            style="italic", transform=ax.get_yaxis_transform())
    ax.set_xlim(0.47, 0.68); ax.set_xticks([0.5, 0.6])
    ax.axvline(0.5, color="0.4", lw=0.4, ls=":")
    ax.set_xlabel("cortical vs subcortical, AUC")


def panel_d(ax, R):
    D = [(R["d_all"], R["null_all"][None]), (R["d_rand"], R["null_rand"]),
         (R["d_match"], R["null_match"]), (R["d_rate"], R["null_rate"][None])]
    lab = ["response types,\nall neurons", "response types,\nrandom subsample", "response types,\nrate-matched",
           "rate bins,\nall neurons"]
    cols = ["0.55", "0.55", "0.55", "k"]
    for i, (d, null) in enumerate(D):
        lo, hi = np.quantile(null, [0.025, 0.975])
        ax.bar(i, hi - lo, bottom=lo, width=0.55, color="0.88", linewidth=0)
        ax.plot([i - 0.27, i + 0.27], [np.median(null)] * 2, color="0.6", lw=0.6)
        d = np.atleast_1d(d)
        ax.plot(np.full(d.size, i) + np.linspace(-0.12, 0.12, d.size) * (d.size > 1), d, "o",
                ms=2.2 if d.size == 1 else 1.2, color=cols[i], mec="none")
    ax.set_xticks(range(4)); ax.set_xticklabels(lab, fontsize=4.5)
    ax.set_ylim(0, 1.15); ax.yaxis.set_major_locator(MaxNLocator(2))
    ax.set_ylabel("cortical vs subcortical\nseparation (Fisher $D$)")


def scatter_clusters(ax, med, ctx, n, base, labels=None, colors=None, size=(3, 25)):
    s = size[0] + size[1] * n / n.max()
    ax.scatter(np.log10(med), ctx, s=s, c=colors if colors is not None else "0.4",
               edgecolors="none", alpha=0.9, zorder=3)
    ax.axhline(base, color="0.5", lw=0.4, ls=":")
    if labels is not None:
        for x, yv, t in zip(np.log10(med), ctx, labels):
            ax.text(x + 0.03, yv + 0.006, t, fontsize=4.2, color="0.25", zorder=4)
    ax.set_xlim(np.log10(1.2), np.log10(60)); log_hz_axis(ax)
    ax.set_ylim(0, 0.8); ax.set_yticks([0, 0.4, 0.8])
    ax.set_xlabel("median firing rate (Hz)"); ax.set_ylabel("cortical fraction")


def panel_g(ax, R):
    w = 0.36
    for j, (g, c) in enumerate((("sparse", C_SPARSE), ("dense", C_DENSE))):
        F = R[f"{g}_frac"]  # rows all / rate-matched, cols ctx / sub
        for i in range(2):
            x = 2.4 * j + i
            ax.bar(x - w / 2, 100 * F[i, 0], w, color=c)
            ax.bar(x + w / 2, 100 * F[i, 1], w, color=c, alpha=0.4)
            ax.text(x, 100 * F[i].max() + 0.8, "all" if i == 0 else "rate-\nmatched",
                    ha="center", va="bottom", fontsize=4.5)
    ax.set_xticks([0.5, 2.9]); ax.set_xticklabels(["sparse tiling", "dense tiling"])
    ax.set_ylabel("% of neurons"); ax.set_ylim(0, 36); ax.yaxis.set_major_locator(MaxNLocator(2))
    ax.bar(0, 0, color="0.3", label="cortical"); ax.bar(0, 0, color="0.3", alpha=0.4, label="subcortical")
    ax.legend(frameon=False, fontsize=4.8, loc="upper right", handlelength=1, borderaxespad=0)


def main():
    R = np.load(HERE / "results" / "cortex_rate.npz", allow_pickle=True)
    style()
    cc = cosmos_colors(list(R["cosmos"]))
    fig = plt.figure(figsize=(W * MM, H * MM))
    gs = GridSpec(3, 6, figure=fig, left=0.09, right=0.975, top=0.965, bottom=0.07,
                  hspace=0.6, wspace=1.7, height_ratios=[1.1, 1, 1])
    ax = fig.add_subplot(gs[0, :2]); panel_a(ax, R, cc); letter(ax, "a", -0.27)
    ax = fig.add_subplot(gs[0, 2:]); panel_b(ax, R, cc); letter(ax, "b", -0.1)
    ax = fig.add_subplot(gs[1, 1:3]); panel_c(ax, R); letter(ax, "c", -0.82)
    ax = fig.add_subplot(gs[1, 3:]); panel_d(ax, R); letter(ax, "d", -0.2)
    ax = fig.add_subplot(gs[2, :2])
    scatter_clusters(ax, R["km_med"], R["km_ctx"], R["km_n"], R["base_ctx"],
                     labels=[str(k) for k in range(len(R["km_med"]))]); letter(ax, "e", -0.27)
    ax = fig.add_subplot(gs[2, 2:4])
    cl = R["rm_cluster"].astype(int)
    col = np.where(np.isin(cl, latency_tiling_groups.SPARSE), C_SPARSE,
                   np.where(np.isin(cl, latency_tiling_groups.DENSE), C_DENSE, "0.6"))
    scatter_clusters(ax, R["rm_med"], R["rm_ctx"], R["rm_n"], R["base_ctx"], colors=col)
    ax.text(0.98, 0.98, "sparse tiling", color=C_SPARSE, transform=ax.transAxes, ha="right", va="top", fontsize=5)
    ax.text(0.98, 0.88, "dense tiling", color=C_DENSE, transform=ax.transAxes, ha="right", va="top", fontsize=5)
    letter(ax, "f", -0.27)
    ax = fig.add_subplot(gs[2, 4:]); panel_g(ax, R); letter(ax, "g", -0.27)
    for ext, kw in (("pdf", {}), ("png", dict(dpi=300))):
        fig.savefig(HERE / f"cortex_rate.{ext}", facecolor="white", **kw)
    print(f"saved {HERE / 'cortex_rate.pdf'} ({W} x {H} mm)")


if __name__ == "__main__":
    main()
