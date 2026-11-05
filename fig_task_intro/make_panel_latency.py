#!/usr/bin/env python3
"""Fig. 1 draft, latency tiling in one PETH window: L_sL_cL_b,s (0-150 ms after the
onset of a left stimulus in a left block, correct choice; dmn_bwm 'stimLbLcL'), the
stimulus-locked tiling that Fig. 5 validates. (Until 2026-10-12: window m, motor_init.)

All trials, all neurons (concat_cvFalse.npy, 54,719 neurons), no trial split:
  left   heatmap of all neurons sorted by the time of their peak in this window,
         each row scaled to [0, 1]
  right  EXAMPLES neurons with the cleanest single peak at evenly spaced latencies:
         among neurons in the top quarter by firing rate (FR_Q; smooth responses)
         with their peak within 2 bins of the target, the one best matching a Gaussian
         bump at the target (sigma SIGMA bins); each scaled to [0, 1], coloured by
         latency
Neurons without any spike in the window (constant response; 1,744, median rate 0.1 Hz)
are left out of the sort. Note: sorting by each neuron's own peak gives a diagonal even
for noise; the
cross-validated test of the latency tiling is in Fig. 5.
Writes elements/panel_latency.svg/.pdf/.png at final size.
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator

ROOT = Path(os.environ.get("DMN_DATA", Path.home() / "dmn"))
EL = Path(__file__).resolve().parent / "elements"
MM = 1 / 25.4
C_SEC = 480.0               # bins per second in the feature vectors
FR_Q = 0.75                 # example candidates: top quarter by firing rate (smooth responses)
TITLE = "Brain-wide latency tiling: everywhere? along the hierarchy?"
WINDOW = "stimLbLcL"
EXAMPLES = 6
SIGMA = 6.0                 # bins (12.5 ms) of the template bump
W, H = 62.0, 36.0           # final size (mm)


def rowcorr(a, b):
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1) + 1e-12)


def unit(X):
    lo, hi = X.min(1, keepdims=True), X.max(1, keepdims=True)
    return (X - lo) / (hi - lo + 1e-12)


def main():
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans"],
                         "font.size": 5.5, "axes.linewidth": 0.5, "xtick.major.width": 0.5,
                         "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    r = np.load(ROOT / "concat_cvFalse.npy", allow_pickle=True).flat[0]
    names = list(r["len"])
    start = np.cumsum([0] + list(r["len"].values()))
    k = names.index(WINDOW)
    Xa = np.asarray(r["concat_z"], np.float32)[:, start[k]:start[k + 1]]
    nb = Xa.shape[1]
    t = (np.arange(nb) + 0.5) / C_SEC                  # s, relative to stimulus onset
    peak = Xa.argmax(1)
    # neurons without a single spike in this window (constant row: no peak; argmax would
    # put them first and the [0, 1] scaling makes them blank) are left out
    active = np.ptp(Xa, 1) > 1e-6
    idx = np.flatnonzero(active)[np.argsort(peak[active], kind="stable")]
    print(f"{idx.size:,} neurons (all trials), sorted by peak time; "
          f"{(~active).sum():,} without a spike in the window left out")

    # examples: strong peaks that best match a single Gaussian bump at each target latency
    bins = np.arange(nb)
    fr = np.asarray(r["fr"], float)
    strong = fr >= np.quantile(fr, FR_Q)
    targets = np.linspace(12, nb - 10, EXAMPLES).round().astype(int)
    U = unit(Xa)
    ex = []
    for b in targets:
        cand = np.flatnonzero(strong & (np.abs(peak - b) <= 2))
        bump = np.exp(-0.5 * ((bins - b) / SIGMA) ** 2)[None]
        ex.append(cand[np.argmax(rowcorr(U[cand], bump))])

    fig = plt.figure(figsize=(W * MM, H * MM))
    fig.text(0.5, 0.975, TITLE, ha="center", va="top", fontsize=6)
    ax = fig.add_axes([0.2, 0.17, 0.24, 0.66])
    ax.imshow(U[idx], aspect="auto", cmap="gray_r", interpolation="antialiased",
              extent=(t[0] - 0.5 / C_SEC, t[-1] + 0.5 / C_SEC, idx.size, 0), rasterized=True)
    ax.set_yticks([0, idx.size], ["0", f"{idx.size:,}"])
    ax.set_ylabel("neurons (sorted by peak time)", labelpad=0)
    ax.set_title("all neurons", fontsize=5.5, pad=2)
    ax2 = fig.add_axes([0.6, 0.17, 0.38, 0.66])
    cmap = mpl.colormaps["viridis"]
    for j, i in enumerate(ex):
        ax2.plot(t, unit(Xa[i:i + 1])[0], color=cmap(j / (EXAMPLES - 1) * 0.9), lw=0.8)
    ax2.set_ylim(-0.03, 1.08)
    ax2.set_yticks([0, 1])
    ax2.set_ylabel("response (scaled)", labelpad=1)
    ax2.set_title("example neurons", fontsize=5.5, pad=2)
    for a in (ax, ax2):
        a.xaxis.set_major_locator(FixedLocator([0, 0.05, 0.1, 0.15]))
        a.set_xticklabels(["0", "", "", "0.15"])
        a.set_xlabel("time from stimulus onset (s)", labelpad=1)
        a.spines[["top", "right"]].set_visible(False)
    EL.mkdir(exist_ok=True)
    for ext, kw in (("svg", {}), ("pdf", {}), ("png", {"dpi": 600})):
        fig.savefig(EL / f"panel_latency.{ext}", facecolor="white", **kw)
    print(f"Saved {EL / 'panel_latency'}.svg/.pdf/.png; examples at "
          + ", ".join(f"{1000 * t[peak[i]]:.0f} ms" for i in ex))


if __name__ == "__main__":
    main()
