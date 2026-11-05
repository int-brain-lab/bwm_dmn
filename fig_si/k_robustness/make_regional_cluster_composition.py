#!/usr/bin/env python3
"""Fig. S4 (2026-10-13): a = existing per-region composition grid (frozen copy,
regional_cluster_composition_k25_source.pdf), b = number of response types present per
region vs number of k-means clusters k (cache/k_robustness.npy, compute_k_robustness.py).

Writes regional_cluster_composition_panel_b.pdf and regional_cluster_composition.pdf (tectonic, 183 mm wide).
"""
import subprocess
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
MM = 1 / 25.4
TEX = r"""\documentclass{standalone}
\usepackage{graphicx,tikz}
\usepackage{fontspec}\setmainfont{Liberation Sans}
\begin{document}
\begin{tikzpicture}[inner sep=0]
\node[anchor=north west] at (0,0) {\includegraphics[width=183mm]{GRID}};
\node[anchor=north west] at (0,-97mm) {\includegraphics{PANELB}};
\node[anchor=north west,font=\bfseries\large] at (-1mm,1mm) {a};
\node[anchor=north west,font=\bfseries\large] at (-1mm,-96mm) {b};
\end{tikzpicture}
\end{document}
"""


def panel_b():
    o = np.load(HERE / "cache" / "k_robustness.npy", allow_pickle=True).flat[0]
    ks = np.array(o["ks"])
    real = [np.percentile(o[k]["present"] * k, [25, 50, 75]) for k in ks]
    null = [np.percentile(np.median(o[k]["null_present"], 0) * k, [25, 50, 75]) for k in ks]
    real, null = np.array(real), np.array(null)
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans"],
                         "font.size": 6, "pdf.fonttype": 42, "axes.linewidth": .5,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "xtick.major.size": 2, "ytick.major.size": 2})
    fig, ax = plt.subplots(figsize=(75 * MM, 45 * MM))
    ax.plot([0, 105], [0, 105], color=".6", lw=.5, ls=":")
    for q, col, lab in ((null, "tab:red", "regions of the same size,\nneurons shuffled"), (real, "k", "Beryl regions")):
        ax.fill_between(ks, q[:, 0], q[:, 2], color=col, alpha=.2, lw=0)
        ax.plot(ks, q[:, 1], "o-", color=col, ms=2.5, lw=1, label=lab)
    for k, m in zip(ks, real[:, 1]):
        if k in (10, 25, 100):
            ax.annotate(f"{m:.0f}/{k}", (k, m), xytext=(-24, -10) if k == 100 else (4, -9),
                        textcoords="offset points", fontsize=5.5)
    ax.text(88, 92, "all k types", color=".5", fontsize=5, ha="right", va="bottom")
    ax.set(xlim=(0, 105), ylim=(0, 105), xticks=[0, 50, 100], yticks=[0, 50, 100],
           xlabel="number of k-means clusters k", ylabel="response types present\nper region")
    ax.legend(frameon=False, loc="upper left", fontsize=5.5)
    fig.subplots_adjust(left=0.2, right=0.97, top=0.95, bottom=0.2)
    out = HERE / "regional_cluster_composition_panel_b.pdf"
    fig.savefig(out); fig.savefig(out.with_suffix(".png"), dpi=300)
    return out


def main():
    b = panel_b()
    tex = HERE / "regional_cluster_composition.tex"
    tex.write_text(TEX.replace("GRID", str(HERE / "regional_cluster_composition_k25_source.pdf")).replace("PANELB", str(b)))
    subprocess.run(["tectonic", str(tex)], check=True, capture_output=True, cwd=HERE)
    subprocess.run(["pdftoppm", "-png", "-r", "200", "-singlefile", str(HERE / "regional_cluster_composition.pdf"),
                    str(HERE / "regional_cluster_composition")], check=True)
    print("saved", HERE / "regional_cluster_composition.pdf")


if __name__ == "__main__":
    main()
