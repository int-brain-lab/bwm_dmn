#!/usr/bin/env python3
"""Build the seven-panel SI Lorenzo figure with a native LaTeX caption."""

import subprocess
import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
FIGS = HERE.parents[1] / "figs"
TECTONIC = Path(os.environ.get("TECTONIC", "tectonic"))


def main():
    tex = rf"""\documentclass[10pt,a4paper]{{article}}
\usepackage[a4paper,landscape,margin=10mm]{{geometry}}
\usepackage{{graphicx,caption,microtype}}
\pagestyle{{empty}}
\captionsetup{{font=small,labelfont=bf,justification=justified,singlelinecheck=false}}
\begin{{document}}
\begin{{figure}}[p]
\centering
\includegraphics[width=\textwidth,height=0.74\textheight,keepaspectratio]{{{(FIGS / 'si_lorenzo_panels.pdf').as_posix()}}}
\caption{{\textbf{{a}} RRR selectivity similarity uses each neuron's time-integrated absolute coefficient for every task variable as features; neurons are averaged within each region, features are standardized across regions, and cosine similarity is calculated between regional vectors.
\textbf{{b}} PETH selectivity similarity uses every PETH time bin as features; neurons are averaged within each region and cosine similarity is calculated between regional vectors. The horizontal axis uses the same Posani-style anatomical connectivity score as in \textbf{{a}}.
\textbf{{c}} Restricting to 67 regions with at least 50 UUID-matched neurons (10,477 included), regional DMN PETH geometry covaries with rank-5 RRR time-summed selectivity geometry.
\textbf{{d}} Rank-5 RRR silhouette $z$-score versus Posani hierarchy position for 24 regions passing RRR selectivity, neuron-count, and session-dominance criteria.
\textbf{{e}} Same analysis as \textbf{{d}}, but using the rank-3 ($\lambda=25$) RRR model selected automatically by three-fold cross-validation (mean CV $R^2$ 0.068 versus 0.066 for rank 5); neuron inclusion uses the rank-3 $\Delta R^2>0.015$, leaving 26 regions.
\textbf{{f}} PETH silhouette $z$-score versus the same hierarchy position for 34 regions passing the PETH neuron-count and session-dominance criteria.
\textbf{{g}} Across-region comparison of RRR mean absolute selectivity (Posani et al., Fig.~2c; authors' released coefficients) with BWM regional decoding scores, restricted to the task variables present in both analyses; RRR wheel selectivity is compared with both wheel-speed and wheel-velocity decoding. Each point is one of the 31 cortical regions with $\geq30$ RRR neurons and $\geq3$ regional decoding scores; both quantities are $z$-scored across these regions. Spearman $P$ values are uncorrected; after Bonferroni correction for five comparisons, feedback and wheel velocity remain significant (uncorrected $P<0.01$).
Dashed lines show the one-sided Bonferroni-corrected clustering criterion; its legend appears only in \textbf{{d}}. RRR has fewer regions than PETH because RRR first requires neuron-level prediction improvement over the null ($\Delta R^2>0.015$, evaluated for each rank); PETH uses all available cortical DMN neurons before both pipelines apply the $\geq50$-neuron and session-dominance criteria. Region-pair $P$ values use two-sided 10,000-permutation region-label QAP tests because pairs sharing regions are not independent.}}
\end{{figure}}
\end{{document}}
"""
    tex_path = FIGS / "si_lorenzo.tex"
    tex_path.write_text(tex)
    subprocess.run([str(TECTONIC), "--outdir", str(FIGS), tex_path.name],
                   cwd=FIGS, check=True)
    print(f"Saved {tex_path} and {FIGS / 'si_lorenzo.pdf'}")


if __name__ == "__main__":
    main()
