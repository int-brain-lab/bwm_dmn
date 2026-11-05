#!/usr/bin/env python3
"""Fig. S3c (added 2026-10-13): overlap of the stimulus- and movement-aligned PETH windows.

Reads results/trial_metrics.pqt (written by analyze_wheel_movement_onset.py) and keeps the
trials that enter the PETHs (BWM load_trials_and_mask defaults: 0.08 s <= reaction time
<= 2 s, choice made). Reaction time = firstMovement_times - stimOn_times. The stimulus
window is 0-150 ms after stimulus onset and the pre-movement window 150 ms before movement
onset (Table 1), so the two overlap whenever the reaction time is below 300 ms, and
movement starts inside the stimulus window below 150 ms.

Writes results/panel_c_window_overlap.pdf, results/window_overlap.txt and assembles
results/movement_onset_subthreshold_wheel.pdf from the frozen a,b source
(results/movement_onset_subthreshold_wheel_ab_source.pdf) and panel c with tectonic.
"""
import subprocess
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
W = 0.15  # PETH window length (s)
MM = 1 / 25.4

TEX = r"""\documentclass{standalone}
\usepackage{graphicx}
\begin{document}
\includegraphics[height=46mm]{AB}\hspace{1mm}\includegraphics[height=46mm]{C}
\end{document}
"""


def main():
    d = pd.read_parquet(RES / "trial_metrics.pqt")
    rt = d.movement_latency_s
    d = d[(rt >= 0.08) & (rt <= 2.0) & d.choice.isin([-1, 1])].copy()
    d["rt"] = d.movement_latency_s
    d["ov"] = np.clip(np.minimum(W, d.rt) - np.maximum(0, d.rt - W), 0, None) / W
    r = d.rt.values
    zones = dict(inside=(r < W).mean(), overlap=((r >= W) & (r < 2 * W)).mean(), separate=(r >= 2 * W).mean())
    s = d.groupby("session").agg(lt300=("rt", lambda x: (x < 2 * W).mean()),
                                 lt150=("rt", lambda x: (x < W).mean()), ov=("ov", "mean"))
    txt = (f"trials {len(d)}, sessions {d.session.nunique()}, subjects {d.subject.nunique()}\n"
           f"reaction time median {np.median(r)*1e3:.0f} ms, mean {r.mean()*1e3:.0f} ms\n"
           f"pooled: RT<150 ms {zones['inside']:.3f}; 150-300 ms {zones['overlap']:.3f}; >=300 ms {zones['separate']:.3f}\n"
           f"pooled mean overlap of stimulus and pre-movement windows: {d.ov.mean():.3f} of 150 ms\n"
           f"per session median [IQR]: RT<300 {s.lt300.median():.3f} [{s.lt300.quantile(.25):.3f}-{s.lt300.quantile(.75):.3f}]; "
           f"RT<150 {s.lt150.median():.3f} [{s.lt150.quantile(.25):.3f}-{s.lt150.quantile(.75):.3f}]; "
           f"overlap {s.ov.median():.3f} [{s.ov.quantile(.25):.3f}-{s.ov.quantile(.75):.3f}]\n")
    (RES / "window_overlap.txt").write_text(txt)
    print(txt)

    mpl.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
                         "font.size": 6, "axes.labelsize": 6.5, "xtick.labelsize": 5.5, "ytick.labelsize": 5.5,
                         "pdf.fonttype": 42, "axes.linewidth": .5, "xtick.major.width": .5,
                         "ytick.major.width": .5, "xtick.major.size": 2, "ytick.major.size": 2,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(52 * MM, 46 * MM), constrained_layout=True)
    cols = ["#c44e52", "#dd8452", ".8"]
    edges = [(0, W), (W, 2 * W), (2 * W, 0.8)]
    labs = [f"movement in stimulus\nwindow {zones['inside']*100:.0f}%",
            f"windows overlap\n{zones['overlap']*100:.0f}%",
            f"separate\n{zones['separate']*100:.0f}%"]
    bins = np.arange(0, 0.8 + 1e-9, 0.01)
    h, _ = np.histogram(r, bins)
    h = h / len(r) * 100
    for (lo, hi), c in zip(edges, cols):
        sel = (bins[:-1] >= lo - 1e-9) & (bins[:-1] < hi - 1e-9)
        ax.bar(bins[:-1][sel] * 1e3, h[sel], width=10, align="edge", color=c, lw=0)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, lw=0, label=l.replace("\n", " ")) for c, l in zip(cols, labs)],
              frameon=False, fontsize=5, loc="upper right", handlelength=1, handleheight=.8,
              labelspacing=.3, borderaxespad=0)
    ytop = h.max() * 1.05
    ax.axvline(80, color="k", ls=":", lw=.5)
    ax.set(xlim=(0, 800), ylim=(0, ytop * 1.1), xlabel="Reaction time (ms)\n(stimulus to movement onset)",
           ylabel="Trials (%)", xticks=[0, 300, 600])
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=2))
    ax.text(.98, .55, f"median {np.median(r)*1e3:.0f} ms\n{(r > 0.8).mean()*100:.0f}% > 800 ms",
            transform=ax.transAxes, ha="right", va="top", fontsize=5)
    ax.text(-.2, 1.06, "c", transform=ax.transAxes, fontweight="bold", fontsize=8, va="top")
    c = RES / "panel_c_window_overlap.pdf"
    fig.savefig(c); fig.savefig(c.with_suffix(".png"), dpi=300)
    plt.close(fig)

    tex = RES / "movement_onset_subthreshold_wheel.tex"
    tex.write_text(TEX.replace("AB", str(RES / "movement_onset_subthreshold_wheel_ab_source.pdf")).replace("C", str(c)))
    subprocess.run(["tectonic", str(tex)], check=True, capture_output=True, cwd=RES)
    subprocess.run(["pdftoppm", "-png", "-r", "300", "-singlefile", str(RES / "movement_onset_subthreshold_wheel.pdf"),
                    str(RES / "movement_onset_subthreshold_wheel")], check=True)
    print("saved", RES / "movement_onset_subthreshold_wheel.pdf")


if __name__ == "__main__":
    main()
