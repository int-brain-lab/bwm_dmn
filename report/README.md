# Manuscript (LaTeX)

`main.tex` (+ `supp_figs.tex`, eLife class, bibliographies) and the figure PDFs
in `figures/` it includes. `figures/svg_figures/` holds the SVG sources of
supplementary figures (and of Fig. 1) that do not yet have a figure folder.

Main figures with a folder in this repository (rebuilt from data):

| Manuscript file | Folder |
|---|---|
| `figures/functional_response_structure.pdf` (Fig. 2) | `../fig2` |
| `figures/anatomy_function_correspondence.pdf` (Fig. 3) | `../fig3` |
| `figures/structured_mixed_selectivity.pdf` (Fig. 4) | `../fig4` (`make_figure4.py --sort kmeans`) |
| `figures/contextual_neural_sequences.pdf` (Fig. 5) | `../fig5` (with `../sequence_analysis`) |
| `figures/mixed_selectivity_k40.pdf` (Fig. S10) | `../fig4/si` |

Supplementary figures with code in this repository: Fig. S10
(`../fig4/si`) and the movement-onset / sub-threshold wheel figure
(`movement_onset_subthreshold_wheel.pdf`, `../si/movement_onset_wheel`; needs the
wheel objects in the ONE cache).

Included as PDFs only (no code here yet): Fig. 1 (Illustrator), the Posani et al.
RRR comparison (`rrr_peth_selectivity_comparison.pdf`; depends on the authors'
RRR coefficients), and `umap_clustering_resolution`, `rastermap_cross_validation`,
`rastermap_functional_categories`, `sequence_shuffle_control`,
`regional_cluster_composition_k25`, `specialization_metric_comparison`,
`prototype_vectors_k100`, `phase_specific_functional_networks`,
`network_stability_controls`, `cortical_subcortical_specialization`,
`structure_tree`. The three Rastermap SI figures were made with the earlier
first-half/second-half trial split and need rebuilding with the odd/even stack.

Sections and figures were reordered on 2026-10-06 (old Fig. 3 -> 5, 4 -> 3,
5 -> 4).

## Build

```bash
report/build.sh             # compile -> report/main.pdf (figures as in report/figures)
report/build.sh --figures   # rebuild figs 2-5 and S10 from data first
```

Requires `tectonic` (the script works around its lookup of the "Latin Modern
Mono" font on a temporary copy) and, for `--figures`, the analysis environment
(e.g. `iblenv`), Ghostscript, Inkscape and the data in `$DMN_DATA` (default
`~/dmn`; see `../DATA_REQUIREMENTS.md`).
