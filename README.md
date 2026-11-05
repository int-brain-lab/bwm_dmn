# Every brain region contains nearly every task response type

Analysis code and manuscript for:

> Schartner, M., Liu, A., International Brain Laboratory, & Fiete, I. Every brain region contains
> nearly every task response type. Earlier version: *Broadcast, structured, and
> sequence-dominated: brain-wide computation at cellular resolution in mice*, bioRxiv 2025
> ([10.1101/2025.07.30.667641](https://www.biorxiv.org/content/10.1101/2025.07.30.667641v2)).

Built on the IBL Brain-Wide Map dataset: https://www.nature.com/articles/s41586-025-09235-0
Data access: https://docs.internationalbrainlab.org/notebooks_external/data_structure.html

## Rebuilding the manuscript

From the IBL data up (each step skips what already exists):

```bash
conda env create -f environment.yml && conda activate bwm_dmn
export DMN_DATA=~/dmn            # data, stacks and caches (default ~/dmn)
pipeline/run_all.sh              # download -> stacks -> caches -> check -> figures -> report/main.pdf
pipeline/run_all.sh --from 5     # only figures + manuscript, from existing stacks
report/build.sh                  # only compile, with the figure PDFs in report/figures
```

- `pipeline/`: `01_bundles.py` (ONE download, per-insertion PETH bundles for the
  515 insertions in `insertions.csv`; hours), `02_stacks.py` (all-trial and odd/even CV
  stacks with the canonical Rastermap fit), `03_caches.py` (25-cluster k-means,
  region/cluster counts), `check_stacks.py` (neuron counts etc. against the published build).
- Inputs the code cannot download (BWM decoding tables, the RRR fit for Fig. S2 and
  Fig. S4e) are listed in `DATA_REQUIREMENTS.md`.

## Layout

Figure folders are named by content, not by manuscript number. **`FIGURES.md`** maps every
main and supplementary figure to its folder, script and manuscript PDF.

| Folder | Figures |
|---|---|
| `fig_task_intro/` | Fig. 1 (task, recordings, feature vectors) |
| `fig_response_types/` | Fig. 2 (k-means response types) |
| `fig_anatomy/` | Fig. 3 (response types across regions) |
| `fig_latency_tiling/` (+ `si/`) | Fig. 4 (stimulus-locked latency tiling), Fig. S1 |
| `fig_response_correlations/si/` | Fig. S4 (neuron × neuron correlations, tests of random mixing), Fig. S6 |
| `fig_si/` | Fig. S5 (`k_robustness/`), Fig. S7, analyses feeding Fig. 4 and S1 |
| `wheel_movement_onset_review/` | Fig. S3 (needs the wheel objects in the ONE cache) |
| `sequence_analysis/` | latency-tiling analyses used by Fig. 4, S1 and S4 |
| `report/` | the LaTeX manuscript and the figure PDFs it includes |
| `auxiliary_analyses/` | analyses not used in the manuscript |

Each figure folder rebuilds its figure from the data stacks with `dmn_bwm.py` functions and
keeps its caches in its own `cache/`; see each folder's README. Figure style defaults:
`FIGURE_STYLE.md`. Fig. S4 and parts of other figures use PyTorch on a CUDA GPU (validated
against NumPy / scikit-learn).

## Requirements

Python 3.10+ (`environment.yml`): `numpy`, `scipy`, `pandas`, `matplotlib`, `scikit-learn`,
`torch`, `umap-learn`, `rastermap`, `iblatlas`, `brainbox`, `one.api`. Ghostscript, Inkscape
(Fig. 1) and tectonic (manuscript).
