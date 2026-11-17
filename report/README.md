# Manuscript (LaTeX)

`main.tex` (+ `supp_figs.tex`, eLife class, bibliographies) and the figure PDFs in
`figures/` it includes. `build.sh` compiles it; `build.sh --figures` first rebuilds every
figure that has code from the data and copies it into `figures/`. `../FIGURES.md` maps each
figure to its folder and script.

Fig. S2 (`rrr_peth_selectivity_comparison.pdf`) is drawn from saved result tables
(`../lorenzo/manuscript_comparison_figure/`; full recomputation needs the RRR fit of Posani et
al.), Fig. S3 (`movement_onset_subthreshold_wheel.pdf`) from the wheel analysis outputs (raw wheel
objects in the ONE cache, or the intermediate data release); `structure_tree.pdf` (Fig. S7) is static.

Figures of earlier versions that are no longer in the manuscript (functional networks,
Rastermap categories, mixed-selectivity basis analysis and others) are left out of the
LaTeX with `\iffalse` and are not included here; they remain in the repository history.
