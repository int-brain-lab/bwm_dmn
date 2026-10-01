# Manuscript (LaTeX)

`main.tex` (+ `supp_figs.tex`, eLife class, bibliographies) and the figure PDFs
in `figures/` it includes. `figures/svg_figures/` holds the SVG sources of
supplementary figures (and of Fig. 1) that do not yet have a figure folder.

Main figures with a folder in this repository (rebuilt from data):

| Manuscript file | Folder |
|---|---|
| `figures/functional_response_structure.pdf` (Fig. 2) | `../fig2` |
| `figures/anatomy_function_correspondence.pdf` (Fig. 4) | `../fig4` |
| `figures/structured_mixed_selectivity.pdf` (Fig. 5) | `../fig5` |
| `figures/mixed_selectivity_k40.pdf` (Fig. S10) | `../fig5/si` |

Fig. 1 (`computational_architecture_and_task.pdf`) and Fig. 3
(`contextual_neural_sequences.pdf`) and the other SI figures are included as
PDFs for now.

## Build

```bash
report/build.sh             # compile -> report/main.pdf
report/build.sh --figures   # rebuild figs 2, 4, 5 and S10 from data first
```

Requires `tectonic` (the script works around its lookup of the "Latin Modern
Mono" font on a temporary copy) and, for `--figures`, the analysis environment
(e.g. `iblenv`), Ghostscript, Inkscape and the data in `$DMN_DATA` (default
`~/dmn`; see `../DATA_REQUIREMENTS.md`).
