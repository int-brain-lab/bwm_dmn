#!/usr/bin/env python3
"""Fig. 1 draft, panels b and c (the earlier panels e and f), from the data.

Mean over all 54,719 neurons (all trials, concat_cvFalse.npy) of the z-scored
feature vectors (concat_z).
  b  the grand average of each trial type, overlaid, on a real time axis as in
     dmn_bwm.plot_ave_PETHs: stimulus onset at 0, movement and feedback windows at
     the session-averaged movement and feedback times (cache/event_timing.npy,
     compute_event_timing.py); the rest window is moved next to quiescence (axis
     break). A selection of PETH types is labelled at the trace ends (LABELS).
  c  the grand-average feature vector: all 21 PETH types concatenated, labelled
     as in Figs. 2 and 5
Writes elements/panel_b.svg/.pdf/.png and panel_c.svg/.pdf/.png at final size.
"""

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import blended_transform_factory

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import make_timeline as tl  # noqa: E402

ROOT = Path(os.environ.get("DMN_DATA", Path.home() / "dmn"))
EL = HERE / "elements"
MM = 1 / 25.4
C_SEC = 480.0
H = 32.4                       # row height (mm), as panel a (data)
WC, HC = 58.0, 35.856           # panel c width, height (mm): row 2, as panel d
GROUPS = [  # (window name, timeline segment, window start (s), PETH types)
    ("rest", tl.REST, -1.15, ["inter_trial"]),
    ("quiescence", tl.STIM, -0.4, ["blockL", "blockR", "quiescence"]),
    ("stim", tl.STIM, 0.0, ["block_change_s", "stimLbLcL", "stimLbRcL", "stimRbRcR", "stimRbLcR", "mistake_s"]),
    ("choice", tl.MOVE, -0.15, ["motor_init", "block_change_m", "sLbLchoiceL", "sLbRchoiceL",
                                "sRbRchoiceR", "sRbLchoiceR", "mistake_m"]),
    ("move", tl.MOVE, 0.0, ["choiceL", "choiceR"]),
    ("feedback", tl.FDBK, 0.0, ["fback1", "fback0"]),
]
EVENTS = [("stim", tl.STIM), ("mvt", tl.MOVE), ("fdbk", tl.FDBK)]


def style():
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans"],
                         "font.size": 5.5, "axes.linewidth": 0.5, "xtick.major.width": 0.5,
                         "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2,
                         "svg.fonttype": "none", "pdf.fonttype": 42, "mathtext.fontset": "custom",
                         "mathtext.rm": "Arial", "mathtext.it": "Arial:italic"})


def save(fig, name):
    for ext, kw in (("svg", {}), ("pdf", {}), ("png", {"dpi": 600})):
        fig.savefig(EL / f"{name}.{ext}", facecolor="white", **kw)
    plt.close(fig)


WINDOWS = {  # PETH type -> (alignment event, window start, window end) in s, as dmn_bwm.concat_PETHs
    "inter_trial": ("stim", -1.15, -1.0), "blockL": ("stim", -0.4, -0.1), "blockR": ("stim", -0.4, -0.1),
    "quiescence": ("stim", -0.4, -0.1),
    **{k: ("stim", 0.0, 0.15) for k in ["block_change_s", "stimLbLcL", "stimLbRcL", "stimRbRcR",
                                        "stimRbLcR", "mistake_s"]},
    **{k: ("move", -0.15, 0.0) for k in ["motor_init", "block_change_m", "sLbLchoiceL", "sLbRchoiceL",
                                         "sRbRchoiceR", "sRbLchoiceR", "mistake_m"]},
    "choiceL": ("move", 0.0, 0.15), "choiceR": ("move", 0.0, 0.15),
    "fback1": ("fdbk", 0.0, 0.3), "fback0": ("fdbk", 0.0, 0.3),
}
REST_SHIFT = 0.55   # s: rest window drawn at [-0.60, -0.45] instead of [-1.15, -1.0] (axis break)
# labelled types; for the stimulus and choice windows, the highest and lowest trace end
LABELS = ["inter_trial", "blockL", "blockR", "quiescence", "STIM_EXTREMES", "MOVE_EXTREMES",
          "choiceL", "choiceR", "fback1", "fback0"]
WB_REAL = 115.0     # panel b width (mm): rest of row 1 next to panel a


def repel(y, gap):
    """Shift sorted label positions apart by at least gap, keeping their mean."""
    order = np.argsort(y)
    ys = np.asarray(y, float)[order].copy()
    for _ in range(50):
        moved = False
        for k in range(1, ys.size):
            if ys[k] - ys[k - 1] < gap:
                d = (gap - (ys[k] - ys[k - 1])) / 2
                ys[k - 1] -= d; ys[k] += d; moved = True
        if not moved:
            break
    out = np.empty_like(ys); out[order] = ys
    return out


def panel_b(seg, labels):
    timing = np.load(HERE / "cache" / "event_timing.npy", allow_pickle=True).flat[0]
    rt, fb = (float(v) for v in timing["mean"])
    anchor = {"stim": 0.0, "move": rt, "fdbk": rt + fb}
    print(f"event timing ({timing['n_sessions']} sessions): movement {rt * 1000:.0f} ms after the "
          f"stimulus, feedback {fb * 1000:.0f} ms after movement")
    fig = plt.figure(figsize=(WB_REAL * MM, H * MM))
    ax = fig.add_axes([10 / WB_REAL, 0.06, 1 - 26 / WB_REAL, 0.9])
    ends = {}
    for ty, (ev, t0, t1) in WINDOWS.items():
        y = seg[ty]
        t = anchor[ev] + t0 + (np.arange(y.size) + 0.5) / C_SEC
        if ty == "inter_trial":
            t = t + REST_SHIFT
        ax.plot(t, y, color="k", lw=0.45, alpha=0.8, solid_capstyle="round")
        ends[ty] = (t[-1], y[-1])
    # labels: chosen types and the extremes of the crowded windows
    chosen = []
    for key in LABELS:
        if key in ("STIM_EXTREMES", "MOVE_EXTREMES"):
            ev = "stim" if key == "STIM_EXTREMES" else "move"
            grp = [k for k, w in WINDOWS.items() if w[0] == ev and w[1] == (0.0 if ev == "stim" else -0.15)]
            grp.sort(key=lambda k: ends[k][1])
            chosen += [grp[0], grp[-1]]
        else:
            chosen.append(key)
    span = np.ptp(np.concatenate([seg[k] for k in WINDOWS]))
    crowded = [k for k in chosen if WINDOWS[k] in (("stim", 0.0, 0.15), ("move", -0.15, 0.0))]
    # stimulus and choice windows overlap in time (movement 253 ms after the stimulus):
    # their labels go to the empty upper left, with thin leader lines to the trace ends
    # the stimulus-window labels; the choice-window labels go to the empty bottom centre
    # (between movement and feedback onset), so the two groups of leader lines do not cross
    top = max(seg[k].max() for k in WINDOWS)
    bottom = min(seg[k].min() for k in WINDOWS)
    stim_l = [k for k in crowded if WINDOWS[k][0] == "stim"]
    move_l = [k for k in crowded if WINDOWS[k][0] == "move"]
    for j, k in enumerate(sorted(stim_l, key=lambda k: -ends[k][1])):
        xl, yl = -0.13, top - (0.42 + j * 0.085) * span   # left of and below the vertical onset label
        ax.text(xl, yl, labels[k], fontsize=4.3, va="center", ha="right")
        ax.plot([xl + 0.004, ends[k][0]], [yl, ends[k][1]], color="0.55", lw=0.25, zorder=0)
    for j, k in enumerate(sorted(move_l, key=lambda k: -ends[k][1])):
        xl, yl = anchor["move"] + 0.12, bottom + (0.2 - j * 0.085) * span
        ax.text(xl, yl, labels[k], fontsize=4.3, va="center", ha="left")
        ax.plot([xl - 0.004, ends[k][0]], [yl, ends[k][1]], color="0.55", lw=0.25, zorder=0)
    by_x = {}
    for k in chosen:
        if k not in crowded:
            by_x.setdefault(round(ends[k][0], 3), []).append(k)
    for x_end, ks in by_x.items():
        ys = repel([ends[k][1] for k in ks], 0.085 * span)
        for k, yl in zip(ks, ys):
            name = "rest" if k == "inter_trial" else labels[k]
            ax.text(x_end + 0.008, yl, name, fontsize=4.3, va="center", ha="left")
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for name, ev in (("stimulus onset", "stim"), ("movement onset", "move"), ("feedback onset", "fdbk")):
        ax.axvline(anchor[ev], color="0.6", lw=0.4, ls=(0, (2, 1.5)), zorder=0)
        ax.text(anchor[ev] - 0.006, 1.0, name, transform=trans, ha="right", va="top",
                rotation=90, fontsize=4.5, color="0.35")
    ylo = ax.get_ylim()[0]
    # axis break between rest and quiescence, 100 ms scale bar
    xb = -1.0 + REST_SHIFT + 0.025
    for dx in (-0.006, 0.006):
        ax.plot([xb + dx - 0.006, xb + dx + 0.006], [ylo - 0.015 * span, ylo + 0.015 * span],
                color="k", lw=0.5, clip_on=False)
    x1 = anchor["fdbk"] + 0.3
    ax.plot([x1 - 0.1, x1], [ylo, ylo], color="k", lw=0.8, solid_capstyle="butt")
    ax.text(x1 - 0.05, ylo + 0.01 * span, "100 ms", ha="center", va="bottom", fontsize=5)
    ax.set_xlim(-1.15 + REST_SHIFT - 0.02, x1 + 0.02)
    ax.set_xticks([])
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=2, min_n_ticks=3))
    ax.set_ylabel("mean response (z)", labelpad=1)
    save(fig, "panel_b")


def main():
    import dmn_bwm  # noqa: F401  (PETH labels; imported first, since it sets its own rcParams)
    style()
    r = np.load(ROOT / "concat_cvFalse.npy", allow_pickle=True).flat[0]
    X = np.asarray(r["concat_z"], np.float32)
    names = list(r["len"])
    start = np.cumsum([0] + list(r["len"].values()))
    mean = X.mean(0)
    seg = {n: mean[start[i]:start[i + 1]] for i, n in enumerate(names)}
    print(f"{X.shape[0]:,} neurons")

    panel_b(seg, dmn_bwm.peth_dictm)

    # c: grand-average feature vector (21 PETH types)
    fig = plt.figure(figsize=(WC * MM, HC * MM))
    ax = fig.add_axes([0.15, 0.08, 0.83, 0.6])
    ax.plot(np.arange(mean.size), mean, color="k", lw=0.5)
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for i, n in enumerate(names):
        ax.text((start[i] + start[i + 1]) / 2, 1.02, dmn_bwm.peth_dictm[n], transform=trans,
                rotation=65, ha="left", va="bottom", rotation_mode="anchor", fontsize=4.3)
        if i:
            ax.axvline(start[i], color="0.6", lw=0.35, ls=(0, (2, 2)), zorder=0)
    ax.set_xlim(0, mean.size)
    ax.set_xticks([])
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=2, min_n_ticks=3))
    ax.set_ylabel("mean response (z)", labelpad=1)
    ax.plot([mean.size - 0.2 * C_SEC, mean.size], [ax.get_ylim()[0]] * 2, color="k", lw=0.8,
            solid_capstyle="butt")
    ax.text(mean.size, ax.get_ylim()[0], "200 ms", ha="right", va="top", fontsize=5)
    save(fig, "panel_c")
    print(f"Saved {EL}/panel_b.* and panel_c.*")


if __name__ == "__main__":
    main()
