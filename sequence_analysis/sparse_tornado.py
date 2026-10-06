#!/usr/bin/env python3
"""Are "sequence" cells just sparse versions of the stimulus-locked "tornado" cells?

Groups on the canonical odd-trial Rastermap fit (CV stack): sequence = clusters
60-69, tornado (stim/integ) = clusters 21-28; all other neurons as background.

To avoid circularity, selection and evaluation use disjoint trials:
  selection  : the two concordant stimulus-aligned types (L_sL_cL_b,s, R_sR_cR_b,s),
               odd vs even trials -> reliability, odd-trial peak latency
  evaluation : the four other stimulus-aligned types (change_b, both discordant,
               mistake), even trials -> responses shown in the rasters, peak times
RT test: results/neurons.csv from rt_split_latency.py (stimulus/movement frames).

Panels
  a  odd/even reliability vs mean rate, all neurons (binned medians per group)
  b  peak-time replication (odd sel. vs even eval.) per rate bin, per group
  c  reliable sequence cells (r >= R_MIN), held-out types, sorted by selection peak
  d  same for reliable tornado cells
  e  held-out peak time vs selection peak time, reliable cells of both groups
  f  RT split (fast vs slow even trials): stimulus- vs movement-frame peak shift
  g  latency distribution of reliable cells
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(os.environ.get("DMN_DATA", Path.home() / "dmn"))
RES = Path(__file__).resolve().parent / "results"
RES.mkdir(exist_ok=True)
C_SEC = 480.0
# Groups = row ranges in the upsampled Rastermap order (Fig. 5a/b), inclusive
SEQ, TOR = (29600, 35599), (11800, 16299)
SEL = ["stimLbLcL", "stimRbRcR"]
EVAL = ["block_change_s", "stimLbRcL", "stimRbLcR", "mistake_s"]
R_MIN = 0.3
COL = {"sequence": "#1f77b4", "tornado": "#ff7f0e", "other": "0.6"}
NAME = {"sequence": "sparse cells (rows 29.6–35.6k)", "tornado": "dense tiling (rows 11.8–16.3k)", "other": "other"}
MM = 1 / 25.4


def rowcorr(a, b):
    a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1) + 1e-12)


def fit_order(r):
    """Row order of Fig. 5a/b: Rastermap with grid_upsample=10 fit on odd trials
    (preview_upsample.py -> results/rastermap_upsample10.npz)."""
    f = np.load(RES / "rastermap_upsample10.npz", allow_pickle=True)
    if not np.array_equal(f["uuids"], np.asarray(r["uuids"])):
        raise RuntimeError("upsample10 fit and CV stack neurons differ")
    return np.asarray(f["isort"])


def main(fig=None, cells=None, letters=None):
    """Standalone, or draw the 8 panels into `cells` (8 subplot specs) of `fig`."""
    embedded = fig is not None
    matplotlib.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
                                "font.size": 5.5, "axes.titlesize": 6, "axes.linewidth": 0.5,
                                "xtick.labelsize": 5, "ytick.labelsize": 5, "pdf.fonttype": 42,
                                "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans",
                                "mathtext.it": "Liberation Sans:italic"})
    r = np.load(ROOT / "concat_cvTrue.npy", allow_pickle=True).flat[0]
    order = fit_order(r)
    lab = np.empty_like(order); lab[order] = np.arange(order.size)  # row of each neuron
    rm_pos = np.empty_like(order); rm_pos[order] = np.arange(order.size)  # row in panels a, b
    start = np.cumsum([0] + list(r["len"].values())); names = list(r["len"])
    seg = lambda t: slice(start[names.index(t)], start[names.index(t) + 1])
    Xo, Xe = np.asarray(r["X_odd"], np.float32), np.asarray(r["X_even"], np.float32)
    fr = np.asarray(r["fr"], float)
    group = np.full(lab.size, "other", object)
    group[(lab >= SEQ[0]) & (lab <= SEQ[1])] = "sequence"
    group[(lab >= TOR[0]) & (lab <= TOR[1])] = "tornado"

    # selection (concordant types): mean over the two types
    so = np.mean([Xo[:, seg(t)] for t in SEL], 0); se = np.mean([Xe[:, seg(t)] for t in SEL], 0)
    rel = rowcorr(so, se)
    peak_sel = so.argmax(1) / C_SEC * 1000
    # evaluation (held-out types, even trials)
    ev = np.mean([Xe[:, seg(t)] for t in EVAL], 0)
    peak_eval = ev.argmax(1) / C_SEC * 1000
    interior = (peak_sel >= 6) & (peak_sel <= 144)

    lines = []
    bins = np.quantile(fr, np.linspace(0, 1, 9))
    rb = np.clip(np.digitize(fr, bins[1:-1]), 0, 7)
    rep = {}
    for g in ("sequence", "tornado", "other"):
        rep[g] = []
        for b in range(8):
            m = (group == g) & (rb == b) & interior
            rep[g].append(spearmanr(peak_sel[m], peak_eval[m]).statistic if m.sum() >= 80 else np.nan)
        m = (group == g)
        lines.append(f"{g}: n={m.sum():,}, median rate {np.median(fr[m]):.3f}, median reliability "
                     f"{np.median(rel[m]):.2f}, reliable (r>={R_MIN}) {np.mean(rel[m] >= R_MIN):.0%}")
    lines.append("peak-time replication (held-out types) per rate octile, sequence / tornado / other:")
    for b in range(8):
        lines.append(f"  rate {bins[b]:.3f}-{bins[b+1]:.3f}: " + " / ".join(
            "  -  " if np.isnan(rep[g][b]) else f"{rep[g][b]:.2f}" for g in ("sequence", "tornado", "other")))

    rel_seq = np.flatnonzero((group == "sequence") & (rel >= R_MIN) & interior)
    rel_tor = np.flatnonzero((group == "tornado") & (rel >= R_MIN) & interior)
    for g, idx in (("sequence", rel_seq), ("tornado", rel_tor)):
        rho = spearmanr(peak_sel[idx], peak_eval[idx]).statistic
        lines.append(f"reliable {g}: n={idx.size:,}, held-out peak-time rho={rho:.2f}, "
                     f"median |sel-eval| {np.median(np.abs(peak_sel[idx] - peak_eval[idx])):.0f} ms")

    rt = pd.read_csv(RES / "neurons.csv")[["uuid", "stim_fast", "stim_slow", "move_fast", "move_slow",
                                           "rt_fast", "rt_slow"]]
    uu = pd.DataFrame(dict(uuid=np.asarray(r["uuids"]).astype(str), i=np.arange(lab.size)))
    rt = rt.merge(uu, on="uuid")
    rt["d_stim"] = (rt.stim_slow - rt.stim_fast) * 1000
    rt["d_move"] = (rt.move_slow - rt.move_fast) * 1000
    shifts = {}
    for g, idx in (("sequence", rel_seq), ("tornado", rel_tor)):
        # RT test is informative only for peaks well inside the 150 ms window
        idx = idx[(peak_sel[idx] >= 20) & (peak_sel[idx] <= 100)]
        sub = rt[rt.i.isin(idx)]
        shifts[g] = (sub.d_stim.to_numpy(), sub.d_move.to_numpy())
        lines.append(f"RT split, reliable {g}, peaks 20-100 ms (n={len(sub)}): stimulus-frame shift median {sub.d_stim.median():.1f} ms, "
                     f"movement-frame {sub.d_move.median():.1f} ms (RT gap {1000*np.median(sub.rt_slow-sub.rt_fast):.0f} ms)")
    # Population test (robust for sparse cells): correlation of fast vs slow response
    # patterns (neurons x time, z-scored per neuron) in each alignment frame.
    P = np.load(RES / "peths.npz", allow_pickle=True)
    prow = {u: k for k, u in enumerate(P["uuid"].astype(str))}

    def zr(M):
        return (M - M.mean(1, keepdims=True)) / (M.std(1, keepdims=True) + 1e-9)

    def pattern_corr(rows, a, b):
        A, B = zr(P[a][rows]), zr(P[b][rows])
        return float(np.mean(np.sum(A * B, 1) / A.shape[1]))

    pop = {}
    for g, idx in (("sequence", rel_seq), ("tornado", rel_tor)):
        rows = np.array([prow[u] for u in np.asarray(r["uuids"]).astype(str)[idx] if u in prow])
        ok = np.isfinite(P["stim_fast"][rows]).all(1) & np.isfinite(P["move_fast"][rows]).all(1)
        rows = rows[ok]
        pop[g] = (pattern_corr(rows, "stim_fast", "stim_slow"), pattern_corr(rows, "move_fast", "move_slow"))
        lines.append(f"fast vs slow pattern correlation, reliable {g} (n={rows.size}): stimulus frame "
                     f"{pop[g][0]:.2f}, movement frame {pop[g][1]:.2f}")
    (RES / "sparse_tornado.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # ---------------- figure ----------------
    if not embedded:
        fig = plt.figure(figsize=(183 * MM, 150 * MM))
        gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.15], hspace=0.5, wspace=0.5,
                              left=0.07, right=0.98, top=0.92, bottom=0.07)
        cells = [gs[0, 0], gs[0, 1], gs[0, 2], gs[0, 3], gs[1, 0], gs[1, 1], gs[1, 2], gs[1, 3]]
    cell = iter(cells)
    L = iter(letters or "abcdefgh")

    def letter(ax, dx=-0.2):
        ax.text(dx, 1.1, next(L), transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")

    ax = fig.add_subplot(next(cell)); letter(ax)
    centers = np.sqrt(bins[:-1] * bins[1:] + 1e-12)  # rate-octile centres (panel d)
    hb = np.linspace(-1, 1, 51)
    for g in ("other", "tornado", "sequence"):
        ax.hist(rel[group == g], bins=hb, density=True, histtype="step", lw=1,
                color=COL[g] if g != "other" else "k", label=NAME[g])
        ax.axvline(np.median(rel[group == g]), color=COL[g] if g != "other" else "k", lw=0.5, ls=":")
    ax.set_xlabel("odd/even reliability (Pearson r)"); ax.set_ylabel("density")
    ax.set_xticks([-1, 0, 1])
    ax.set_title("reliability of each neuron", fontsize=6)
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none", framealpha=0.85, fontsize=4.2, handlelength=1.2, borderpad=0.2); ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(next(cell)); letter(ax)
    for g in ("other", "tornado", "sequence"):
        ax.plot(centers, rep[g], color=COL[g] if g != "other" else "k", lw=1, marker="o", ms=2, label=NAME[g])
    ax.set_xscale("log"); ax.set_xlabel("mean rate (stack units)")
    ax.set_ylabel("peak-time ρ (held-out types)")
    ax.set_title("timing replication at matched rate", fontsize=6)
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none", framealpha=0.85, fontsize=4.2, handlelength=1.2, borderpad=0.2); ax.spines[["top", "right"]].set_visible(False)

    STIM6 = ["block_change_s", "stimLbLcL", "stimLbRcL", "stimRbRcR", "stimRbLcR", "mistake_s"]
    from seq_common import dmn_bwm
    STIM6_LAB = [dmn_bwm.peth_dictm[t] for t in STIM6]  # full PETH names, as in panel b

    def zoom_display(X, q=80, gamma=0.5):
        """Display of panel b: per neuron over the full feature vector, (x - median) /
        (99th pct - median) in [0, 1], values below the 80th percentile -> 0, gamma."""
        med = np.median(X, 1, keepdims=True)
        hi = np.percentile(X, 99, 1, keepdims=True)
        R = np.clip((X - med) / (hi - med + 1e-6), 0, 1)
        return np.where(R >= np.percentile(R, q, 1, keepdims=True), R, 0) ** gamma

    def raster(ax, cl, title, color):
        """Zoom of panel b: all neurons of Rastermap clusters cl, in b's order (fit on
        odd trials), even trials, the six stimulus-aligned types, no row averaging."""
        idx = np.flatnonzero((lab >= cl[0]) & (lab <= cl[1]))
        idx = idx[np.argsort(rm_pos[idx], kind="stable")]
        D = zoom_display(Xe[idx])
        M = np.concatenate([D[:, seg(t)] for t in STIM6], 1)
        ax.imshow(1 - M, cmap="gray", aspect="auto", interpolation="antialiased", vmin=0, vmax=1,
                  rasterized=True)
        for b in range(1, len(STIM6)):
            ax.axvline(72 * b - 0.5, color="0.5", lw=0.3, ls=":")
        ax.set_xticks([72 * b + 36 for b in range(len(STIM6))], STIM6_LAB, rotation=40, fontsize=4.3)
        ax.set_yticks([0, len(idx) - 1], ["0", f"{len(idx):,}"])
        ax.set_title(title, fontsize=6, color=color)

    ax = fig.add_subplot(next(cell)); letter(ax)
    raster(ax, SEQ, "sparse cells, zoom of b\n(rows 29.6–35.6k, even trials)", COL["sequence"])
    ax.set_ylabel("Rastermap order (fit on odd trials)")
    ax = fig.add_subplot(next(cell)); letter(ax)
    raster(ax, TOR, "dense tiling, zoom of b\n(rows 11.8–16.3k, even trials)", COL["tornado"])

    ax = fig.add_subplot(next(cell)); letter(ax)
    for g, idx in (("tornado", rel_tor), ("sequence", rel_seq)):
        jit = np.random.default_rng(1).uniform(-1, 1, (2, idx.size))
        ax.scatter(peak_sel[idx] + jit[0], peak_eval[idx] + jit[1], s=1.2, lw=0, color=COL[g],
                   alpha=0.4, rasterized=True, label=f"{NAME[g]} (ρ={spearmanr(peak_sel[idx], peak_eval[idx]).statistic:.2f})")
    ax.plot([0, 150], [0, 150], color="k", lw=0.4, ls=":")
    ax.set_xlabel("peak, selection types (odd, ms)"); ax.set_ylabel("peak, held-out types (even, ms)")
    ax.set_title("same latency in held-out trials", fontsize=6)
    ax.legend(loc="upper left", markerscale=3, frameon=True, facecolor="white", edgecolor="none", framealpha=0.85, fontsize=4.2, handlelength=1.2, borderpad=0.2); ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(next(cell)); letter(ax)
    for k, g in enumerate(("sequence", "tornado")):
        ax.bar([3 * k, 3 * k + 1], pop[g], color=COL[g], alpha=1)
        ax.bar([3 * k + 1], [pop[g][1]], color="white", edgecolor=COL[g], hatch="////", lw=0.6)
    ax.set_xticks([0, 1, 3, 4], ["sparse\nstim", "sparse\nmove", "dense\nstim", "dense\nmove"], fontsize=4.5)
    ax.set_ylabel("fast vs slow RT pattern correlation")
    ax.set_title("aligned to stimulus or movement?", fontsize=6)
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(next(cell)); letter(ax)
    bb = np.linspace(0, 150, 31)
    for g, idx in (("tornado", rel_tor), ("sequence", rel_seq)):
        ax.hist(peak_sel[idx], bins=bb, density=True, histtype="step", lw=1, color=COL[g], label=NAME[g])
    ax.set_xlabel("peak latency after stimulus (ms)"); ax.set_ylabel("density")
    ax.set_title("latency distributions overlap", fontsize=6)
    ax.legend(loc="upper center", frameon=True, facecolor="white", edgecolor="none", framealpha=0.85, fontsize=4.2, handlelength=1.2, borderpad=0.2); ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(next(cell)); letter(ax)
    frac = [np.mean(rel[group == g] >= R_MIN) for g in ("sequence", "tornado", "other")]
    ax.bar([0, 1, 2], frac, color=[COL["sequence"], COL["tornado"], COL["other"]])
    ax.set_xticks([0, 1, 2], ["sparse", "dense", "other"])
    ax.set_ylabel(f"fraction reliable (r ≥ {R_MIN})")
    ax.set_title("sparse cells: rarely reliable", fontsize=6)
    ax.spines[["top", "right"]].set_visible(False)

    if embedded:
        return
    fig.suptitle("The 'sequence' cluster (canonical 60–69) is a pool of low-reliability sparse cells; "
                 "its reproducible part is stimulus-locked latency tiling, as in dense latency tiling (21–28)", fontsize=6.5)
    fig.savefig(RES / "sparse_tornado.png", dpi=250, facecolor="white")
    fig.savefig(RES / "sparse_tornado.pdf", facecolor="white")
    print(f"Saved {RES / 'sparse_tornado.png'}")


if __name__ == "__main__":
    main()
