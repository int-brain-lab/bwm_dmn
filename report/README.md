# Manuscript (LaTeX)

`main.tex` (+ `supp_figs.tex`, eLife class, bibliographies) and the figure PDFs in
`figures/` it includes. `build.sh` compiles it; `build.sh --figures` first rebuilds every
figure that has code from the data and copies it into `figures/`. `../FIGURES.md` maps each
figure to its folder and script.

Included as PDFs only:
- `rrr_peth_selectivity_comparison.pdf` (Fig. S2): needs the RRR fit of Posani et al. (see
  `../DATA_REQUIREMENTS.md`);
- `movement_onset_subthreshold_wheel.pdf` (Fig. S3): rebuilt by
  `../wheel_movement_onset_review/analyze_wheel_movement_onset.py`, which needs the wheel
  objects in the ONE cache;
- `structure_tree.pdf` (Fig. S8): static.

Figures of earlier versions that are no longer in the manuscript (functional networks,
Rastermap categories, mixed-selectivity basis analysis and others) are left out of the
LaTeX with `\iffalse` and are not included here; they remain in the repository history.
