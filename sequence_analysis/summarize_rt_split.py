#!/usr/bin/env python3
"""Summary and figure for rt_split_latency.py.

Neurons: reliable (odd vs even r >= REL), mean rate >= MIN_RATE, odd-trial peak in
the interior of the 150 ms stimulus window. For each latency bin (odd-trial peak):
median shift of the peak between slow and fast even trials, in the stimulus frame
(0 if stimulus-locked) and the movement frame (0 if movement-locked).
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = __import__("pathlib").Path(__file__).resolve().parent
RES = HERE / "results"
RES.mkdir(exist_ok=True)
REL, MIN_RATE = 0.5, 0.5
BINS = np.array([0, 25, 50, 75, 100, 125, 150]) / 1000


def boot_median(x, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    m = np.median(rng.choice(x, (n, x.size)), axis=1)
    return np.percentile(m, [2.5, 97.5])


def main():
    df = pd.read_csv(RES / "neurons.csv")
    P = np.load(RES / "peths.npz", allow_pickle=True)
    sel = (df.reliability >= REL) & (df.rate >= MIN_RATE) & df.interior
    d = df[sel].copy()
    d["d_stim"] = (d.stim_slow - d.stim_fast) * 1000
    d["d_move"] = (d.move_slow - d.move_fast) * 1000
    rt_gap = float(np.median(d.rt_slow - d.rt_fast) * 1000)

    lines = [f"neurons analysed: {len(df):,}; selected (r>={REL}, rate>={MIN_RATE} Hz, interior peak): {len(d):,}",
             f"median RT fast {1000 * d.rt_fast.median():.0f} ms, slow {1000 * d.rt_slow.median():.0f} ms "
             f"(median per-session gap {rt_gap:.0f} ms)", "",
             "odd-trial peak (ms) |   n  | stim-frame shift slow-fast (ms) [95% CI] | move-frame shift (ms) [95% CI]"]
    rows = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        g = d[(d.peak_odd >= lo) & (d.peak_odd < hi)]
        if len(g) < 20:
            continue
        cs, cm = boot_median(g.d_stim.to_numpy()), boot_median(g.d_move.to_numpy())
        ps, pm = wilcoxon(g.d_stim).pvalue, wilcoxon(g.d_move).pvalue
        rows.append((1000 * (lo + hi) / 2, g.d_stim.median(), cs, g.d_move.median(), cm, len(g)))
        lines.append(f"  {1000*lo:3.0f}-{1000*hi:3.0f}          | {len(g):5d} | {g.d_stim.median():6.1f} "
                     f"[{cs[0]:.1f}, {cs[1]:.1f}] p={ps:.1e} | {g.d_move.median():6.1f} "
                     f"[{cm[0]:.1f}, {cm[1]:.1f}] p={pm:.1e}")
    (RES / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # Figure: tornado plots (even trials, fast vs slow), sorted by odd-trial peak.
    idx = np.flatnonzero(sel.to_numpy())
    order = idx[np.argsort(df.peak_odd.to_numpy()[idx], kind="stable")]

    def norm(M):
        M = M[order]
        lo, hi = M.min(1, keepdims=True), M.max(1, keepdims=True)
        return (M - lo) / (hi - lo + 1e-9)

    fig = plt.figure(figsize=(7.2, 5.2))
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 0.25, 1, 1], height_ratios=[1, 0.62],
                          hspace=0.45, wspace=0.12, left=0.07, right=0.98, top=0.9, bottom=0.09)
    panels = [("stim_fast", "stimulus-aligned, fast", (0, 150)),
              ("stim_slow", "stimulus-aligned, slow", (0, 150)),
              ("move_fast", "movement-aligned, fast", (-150, 0)),
              ("move_slow", "movement-aligned, slow", (-150, 0))]
    for k, (key, title, ext) in enumerate(panels):
        ax = fig.add_subplot(gs[0, k if k < 2 else k + 1])
        ax.imshow(norm(P[key]), aspect="auto", cmap="gray_r", interpolation="antialiased",
                  extent=(*ext, len(order), 0), rasterized=True)
        ax.set_title(title, fontsize=7)
        ax.set_xticks([ext[0], np.mean(ext), ext[1]]); ax.tick_params(labelsize=6)
        ax.set_xlabel("time from " + ("stimulus" if k < 2 else "movement") + " (ms)", fontsize=6)
        if k == 0:
            ax.set_ylabel(f"{len(order):,} neurons, sorted by odd-trial peak", fontsize=6)
        else:
            ax.set_yticks([])
    ax = fig.add_subplot(gs[1, 0:2])
    x = np.array([r[0] for r in rows])
    for j, col, lab in ((1, "tab:blue", "stimulus frame"), (3, "tab:red", "movement frame")):
        y = np.array([r[j] for r in rows]); ci = np.array([r[j + 1] for r in rows])
        ax.errorbar(x, y, yerr=[y - ci[:, 0], ci[:, 1] - y], color=col, marker="o", ms=3,
                    lw=1, capsize=2, label=lab)
    ax.axhline(0, color="0.5", lw=0.5, ls=":")
    ax.axhline(rt_gap, color="0.5", lw=0.5, ls="--")
    ax.text(x[-1], rt_gap, " RT gap", fontsize=6, va="center", color="0.4")
    ax.set_xlabel("odd-trial peak latency after stimulus (ms)", fontsize=6)
    ax.set_ylabel("peak shift, slow - fast (ms)", fontsize=6)
    ax.tick_params(labelsize=6); ax.legend(frameon=False, fontsize=6)
    ax.spines[["top", "right"]].set_visible(False)
    ax = fig.add_subplot(gs[1, 3:5])
    ax.hist(1000 * df.peak_odd[sel], bins=36, color="0.4")
    ax.set_xlabel("odd-trial peak latency after stimulus (ms)", fontsize=6)
    ax.set_ylabel("neurons", fontsize=6); ax.tick_params(labelsize=6)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Stimulus-window responses: fixed latency (stimulus-locked) or timed to movement?",
                 fontsize=8)
    fig.savefig(RES / "rt_split_latency.png", dpi=250)
    fig.savefig(RES / "rt_split_latency.pdf")
    print(f"Saved {RES / 'rt_split_latency.png'}")


if __name__ == "__main__":
    main()
