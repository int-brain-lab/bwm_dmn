#!/usr/bin/env python3
"""Reproduce the Fig. 3 tests of whether "sequence" neurons differ from stimulus/
integrator neurons, on the odd/even data, with cross-validated versions.

Groups (Rastermap fit on odd trials of 7 well-sampled types, z-scored within them;
results/rastermap_fit7z.npz):
  sequence  : clusters 25-30 and 46-51 (sparse train-trial diagonals)
  stim/integ: clusters 90-99 (graded-latency "tornado")
Columns: the six stimulus-aligned trial types (0-150 ms after stimulus); four of
them (change_b, both discordant, mistake) were not used for the fit.
Data: X_odd / X_even of the CV stack (feature vectors z-scored over all 21 types).

Analyses (all on even trials unless stated; "xval" = odd bins vs even bins, which
share no spikes and so are free of the overlap of the strided 12.5 ms bins):
  a  rasters, odd and even, in the fit order
  b  within-trial-type time-time correlation, even x even and odd x even
  c  correlation vs time lag (mean over the six types)
  d  mean z-scored response over neurons
  e  mean firing rate
  f  cross-trial-type correlation of the cell x time matrices, odd type A vs even type B
  g  peak time odd vs even, per neuron
  h  Cosmos composition vs all neurons
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from iblatlas.regions import BrainRegions
from matplotlib.transforms import blended_transform_factory
from scipy.stats import spearmanr

ROOT = Path.home() / "dmn"
HERE = Path(__file__).resolve().parent
RES = HERE / "results"
C_SEC = 480.0
GROUPS = {"sequence": [(25, 30), (46, 51)], "stim/integ": [(90, 99)]}  # defaults (fit7z)
COLORS = {"sequence": "#1f77b4", "stim/integ": "#ff7f0e", "rate-matched": "0.45"}
STIM = ["block_change_s", "stimLbLcL", "stimLbRcL", "stimRbRcR", "stimRbLcR", "mistake_s"]
MM = 1 / 25.4
PETH_LABELS = {"block_change_s": r"$\mathrm{change_b, s}$", "stimLbLcL": r"$\mathrm{L_sL_cL_b, s}$",
               "stimLbRcL": r"$\mathrm{L_sL_cR_b, s}$", "stimRbRcR": r"$\mathrm{R_sR_cR_b, s}$",
               "stimRbLcR": r"$\mathrm{R_sR_cL_b, s}$", "mistake_s": r"$\mathrm{mistake, s}$"}


def configure_style():
    matplotlib.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 5.5, "axes.labelsize": 5.5, "axes.titlesize": 6, "xtick.labelsize": 5,
        "ytick.labelsize": 5, "axes.linewidth": 0.5, "xtick.major.width": 0.5,
        "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2,
        "legend.fontsize": 5, "pdf.fonttype": 42,
        "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans",
        "mathtext.it": "Liberation Sans:italic", "mathtext.bf": "Liberation Sans:bold",
    })


def colcorr(A, B):
    """Correlation (across neurons) between every column of A and every column of B."""
    A = (A - A.mean(0)) / (A.std(0) + 1e-9)
    B = (B - B.mean(0)) / (B.std(0) + 1e-9)
    return A.T @ B / A.shape[0]


def matcorr(A, B):
    a, b = A.ravel() - A.mean(), B.ravel() - B.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def lag_profile(C, max_lag=40):
    return np.array([np.mean(np.diagonal(C, k)) for k in range(max_lag)])


def parse_blocks(text):
    return [tuple(int(x) for x in b.split("-")) for b in text.split(",")]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fit", default="fit7z",
                    help="'canonical' (odd-trial fit stored in the CV stack, all 21 types) "
                         "or a results/rastermap_<fit>.npz tag")
    ap.add_argument("--seq", default="25-30,46-51", help="sequence clusters, e.g. 60-69")
    ap.add_argument("--stim", default="90-99", help="stim/integ clusters, e.g. 17-30")
    ap.add_argument("--tag", default="", help="suffix for output files")
    args = ap.parse_args()
    global GROUPS
    GROUPS = {"sequence": parse_blocks(args.seq), "stim/integ": parse_blocks(args.stim)}
    configure_style()
    r = np.load(ROOT / "concat_cvTrue.npy", allow_pickle=True).flat[0]
    if args.fit == "canonical":
        labels, order = np.asarray(r["rm_labels"]), np.asarray(r["isort"])
    else:
        fit = np.load(RES / f"rastermap_{args.fit}.npz", allow_pickle=True)
        labels, order = fit["labels"], fit["isort"]
    pos = np.empty_like(order); pos[order] = np.arange(order.size)
    start = np.cumsum([0] + list(r["len"].values())); names = list(r["len"])
    seg = {t: slice(start[names.index(t)], start[names.index(t) + 1]) for t in STIM}
    Xo, Xe = np.asarray(r["X_odd"], np.float32), np.asarray(r["X_even"], np.float32)
    cosmos = np.asarray(BrainRegions().id2acronym(np.asarray(r["ids"]), mapping="Cosmos"))
    fr = np.asarray(r["fr"], float)
    lab = {t: PETH_LABELS[t] for t in STIM}

    G = {}
    for g, blocks in GROUPS.items():
        m = np.zeros(labels.size, bool)
        for lo, hi in blocks:
            m |= (labels >= lo) & (labels <= hi)
        idx = np.flatnonzero(m)
        G[g] = idx[np.argsort(pos[idx])]
    # Rate-matched control: neurons outside both groups, sampled to match the
    # sequence group's firing-rate distribution (deciles).
    rng = np.random.default_rng(0)
    used = np.zeros(labels.size, bool); used[np.concatenate(list(G.values()))] = True
    pool = np.flatnonzero(~used & (labels >= 0))
    edges = np.quantile(fr[G["sequence"]], np.linspace(0, 1, 11))
    ctrl = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        n_need = np.sum((fr[G["sequence"]] >= lo) & (fr[G["sequence"]] <= hi))
        cand = pool[(fr[pool] >= lo) & (fr[pool] <= hi)]
        ctrl.append(rng.choice(cand, min(n_need, cand.size), replace=False))
    ctrl = np.unique(np.concatenate(ctrl))
    G["rate-matched"] = ctrl[np.argsort(pos[ctrl])]
    GROUPS["rate-matched"] = []
    out, lines = {}, []
    for g, idx in G.items():
        within_ee = np.mean([colcorr(Xe[idx, seg[t]], Xe[idx, seg[t]]) for t in STIM], 0)
        within_oe = np.mean([colcorr(Xo[idx, seg[t]], Xe[idx, seg[t]]) for t in STIM], 0)
        cross = np.array([[matcorr(Xo[idx, seg[a]], Xe[idx, seg[b]]) for b in STIM] for a in STIM])
        po = np.concatenate([Xo[idx, seg[t]].argmax(1) for t in STIM[1:2]]) / C_SEC * 1000
        pe = np.concatenate([Xe[idx, seg[t]].argmax(1) for t in STIM[1:2]]) / C_SEC * 1000
        out[g] = dict(idx=idx, ee=within_ee, oe=within_oe, cross=cross, po=po, pe=pe,
                      mean_e=np.mean([Xe[idx, seg[t]].mean(0) for t in STIM], 0),
                      mean_per=np.concatenate([Xe[idx, seg[t]].mean(0) for t in STIM]),
                      fr=fr[idx], cosmos=cosmos[idx])
        rho = spearmanr(po, pe).statistic
        lines.append(f"{g}: n={idx.size:,}, mean rate median {np.median(fr[idx]):.3f} (stack units), "
                     f"peak time odd vs even rho={rho:.2f} (L_sL_cL_b,s), "
                     f"xval lag-0 corr {within_oe[np.arange(72), np.arange(72)].mean():.2f}, "
                     f"mean cross-type xval corr (off-diagonal) {cross[~np.eye(6, dtype=bool)].mean():.2f}, "
                     f"same-type xval {np.diag(cross).mean():.2f}")
    lines.append(f"all neurons: mean rate median {np.median(fr):.3f} (stack units)")
    lines.insert(0, f"fit {args.fit}; sequence clusters {args.seq}; stim/integ clusters {args.stim}; "
                    f"rate-matched control from all other clusters")
    (RES / f"sequence_vs_stim{args.tag}.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # ---------------- figure ----------------
    fig = plt.figure(figsize=(183 * MM, 200 * MM))
    gs = fig.add_gridspec(4, 4, height_ratios=[1.35, 0.85, 0.85, 0.85], hspace=0.65, wspace=0.45,
                          left=0.07, right=0.98, top=0.93, bottom=0.05)
    letters = iter("abcdefghijkl")

    def letter(ax, dx=-0.18, dy=1.08):
        ax.text(dx, dy, next(letters), transform=ax.transAxes, fontsize=8, fontweight="bold",
                va="bottom")

    nb = 72 * len(STIM)
    for k, (g, half, X) in enumerate([("sequence", "odd (train)", Xo), ("sequence", "even (test)", Xe),
                                      ("stim/integ", "odd (train)", Xo), ("stim/integ", "even (test)", Xe)]):
        ax = fig.add_subplot(gs[0, k])
        idx = out[g]["idx"]
        M = np.clip(np.concatenate([X[idx, seg[t]] for t in STIM], 1), 0, 1.5) / 1.5
        ax.imshow(1 - M, cmap="gray", vmin=0, vmax=1, aspect="auto", interpolation="antialiased",
                  rasterized=True)
        for b in range(1, len(STIM)):
            ax.axvline(72 * b - 0.5, color="0.5", lw=0.3, ls=":")
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        for b, t in enumerate(STIM):
            ax.text(72 * b + 36, 1.01, lab[t], transform=trans, rotation=65, ha="left",
                    va="bottom", rotation_mode="anchor", fontsize=4.3)
        ax.set_xticks([]); ax.set_yticks([0, len(idx) - 1], ["0", f"{len(idx):,}"])
        ax.set_title(f"{g}, {half}", fontsize=6, pad=24, color=COLORS[g])
        if k == 0:
            letter(ax, dy=1.18); ax.set_ylabel("neurons (fit order)")
        if k == 2:
            letter(ax, dy=1.18)

    # b: time-time correlation within trial type (mean over six types)
    vmax = 0.5
    for k, (g, key, ttl) in enumerate([("sequence", "ee", "even × even"), ("sequence", "oe", "odd × even (xval)"),
                                      ("stim/integ", "ee", "even × even"), ("stim/integ", "oe", "odd × even (xval)")]):
        ax = fig.add_subplot(gs[1, k])
        C = out[g][key].copy()
        im = ax.imshow(C, cmap="RdBu_r", vmin=-vmax, vmax=vmax, extent=(0, 150, 150, 0))
        ax.set_title(f"{g}: {ttl}", fontsize=5.5, color=COLORS[g])
        ax.set_xticks([0, 75, 150]); ax.set_yticks([0, 75, 150])
        ax.set_xlabel("time (ms)"); ax.set_ylabel("time (ms)") if k == 0 else None
        if k == 0:
            letter(ax)
    cax = ax.inset_axes([1.05, 0.0, 0.05, 0.5])
    fig.colorbar(im, cax=cax, ticks=[-vmax, 0, vmax]).ax.tick_params(labelsize=4.5)

    # c: lag profiles
    ax = fig.add_subplot(gs[2, 0]); letter(ax)
    lags = np.arange(40) / C_SEC * 1000
    for g in GROUPS:
        ax.plot(lags, lag_profile(out[g]["ee"]), color=COLORS[g], lw=0.8, label=f"{g}, even×even")
        ax.plot(lags, lag_profile(out[g]["oe"]), color=COLORS[g], lw=0.8, ls="--", label=f"{g}, xval")
    ax.axvline(12.5, color="0.5", lw=0.4, ls=":")
    ax.text(13, 0.9, "bin width", fontsize=4.5, color="0.4")
    ax.set_xlabel("time lag (ms)"); ax.set_ylabel("mean correlation")
    ax.legend(frameon=False, fontsize=4.2); ax.spines[["top", "right"]].set_visible(False)

    # d: mean z-scored response
    ax = fig.add_subplot(gs[2, 1:3]); letter(ax, dx=-0.08)
    for g in GROUPS:
        ax.plot(out[g]["mean_per"], color=COLORS[g], lw=0.8, label=g)
    for b in range(1, len(STIM)):
        ax.axvline(72 * b, color="0.6", lw=0.3, ls=":")
    ax.set_xticks([72 * b + 36 for b in range(len(STIM))], [lab[t] for t in STIM], rotation=30, fontsize=4.3)
    ax.set_ylabel("mean z-scored rate (even)"); ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    # e: firing rate
    ax = fig.add_subplot(gs[2, 3]); letter(ax)
    data = [out[g]["fr"] for g in GROUPS] + [fr]
    ax.boxplot(data, showfliers=False, widths=0.5, medianprops=dict(color="k", lw=0.8),
               boxprops=dict(lw=0.5), whiskerprops=dict(lw=0.5), capprops=dict(lw=0.5))
    ax.set_xticks(range(1, len(GROUPS) + 2), ["sequence", "stim/\ninteg", "rate-\nmatched", "all"], fontsize=4.5)
    ax.set_ylabel("mean rate (stack units)"); ax.spines[["top", "right"]].set_visible(False)

    # f: cross-trial-type xval correlation
    for k, g in enumerate(["sequence", "stim/integ"]):
        ax = fig.add_subplot(gs[3, k])
        im = ax.imshow(out[g]["cross"], cmap="viridis", vmin=0, vmax=0.6)
        ax.set_xticks(range(6), [lab[t] for t in STIM], rotation=70, fontsize=4)
        ax.set_yticks(range(6), [lab[t] for t in STIM], fontsize=4)
        ax.set_title(f"{g}: odd type × even type", fontsize=5.5, color=COLORS[g])
        if k == 0:
            letter(ax, dx=-0.45)
    cax = ax.inset_axes([1.05, 0.0, 0.06, 0.6])
    fig.colorbar(im, cax=cax, ticks=[0, 0.3, 0.6]).ax.tick_params(labelsize=4.5)

    # g: peak-time replication
    ax = fig.add_subplot(gs[3, 2]); letter(ax)
    for g in GROUPS:
        jit = np.random.default_rng(0).uniform(-1, 1, (2, out[g]["po"].size))
        ax.scatter(out[g]["po"] + jit[0], out[g]["pe"] + jit[1], s=0.4, color=COLORS[g],
                   alpha=0.25, lw=0, rasterized=True)
        rho = spearmanr(out[g]["po"], out[g]["pe"]).statistic
        ax.text(0.02, 0.98 - 0.08 * list(GROUPS).index(g), f"{g}: ρ = {rho:.2f}",
                transform=ax.transAxes, color=COLORS[g], va="top", fontsize=5)
    ax.set_xlabel(f"peak time, odd ({lab['stimLbLcL']}, ms)"); ax.set_ylabel("peak time, even (ms)")
    ax.set_xticks([0, 75, 150]); ax.set_yticks([0, 75, 150])
    ax.spines[["top", "right"]].set_visible(False)

    # h: Cosmos composition vs all neurons
    ax = fig.add_subplot(gs[3, 3]); letter(ax)
    regs = ["Isocortex", "OLF", "HPF", "CTXsp", "CNU", "TH", "HY", "MB", "HB", "CB"]
    base = np.array([np.mean(cosmos == c) for c in regs])
    for j, g in enumerate(GROUPS):
        frac = np.array([np.mean(out[g]["cosmos"] == c) for c in regs])
        ax.barh(np.arange(len(regs)) + 0.27 * (j - 1), np.log2((frac + 1e-4) / (base + 1e-4)),
                height=0.26, color=COLORS[g], label=g)
    ax.axvline(0, color="k", lw=0.4)
    ax.set_yticks(range(len(regs)), regs, fontsize=4.5); ax.invert_yaxis()
    ax.set_xlabel("log2(fraction / fraction of all neurons)")
    ax.legend(frameon=False, fontsize=4.5); ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"{args.fit} fit: sequence = clusters {args.seq}, stim/integ = {args.stim}, "
                 f"grey = rate-matched control", fontsize=6.5, y=0.995)
    fig.savefig(RES / f"sequence_vs_stim{args.tag}.png", dpi=250, facecolor="white")
    fig.savefig(RES / f"sequence_vs_stim{args.tag}.pdf", facecolor="white")
    print(f"Saved {RES / f'sequence_vs_stim{args.tag}.png'}")


if __name__ == "__main__":
    main()
