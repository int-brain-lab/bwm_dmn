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
# without the raw data: install the intermediate data release, then build from step 3
python pipeline/install_intermediate.py /path/to/release && pipeline/run_all.sh --from 3
report/build.sh                  # only compile, with the figure PDFs in report/figures
```

- `pipeline/`: `01_bundles.py` (ONE download, per-insertion PETH bundles for the
  515 insertions in `insertions.csv`; hours), `02_stacks.py` (all-trial and odd/even CV
  stacks with the canonical Rastermap fit), `03_caches.py` (25-cluster k-means,
  region/cluster counts), `check_stacks.py` (neuron counts etc. against the published build).
- Inputs the code cannot download (BWM decoding tables, the RRR fit for Fig. S2 and
  Fig. 4e) are listed in `DATA_REQUIREMENTS.md`; all of them, in compact form, are in the
  intermediate data release (`pipeline/install_intermediate.py`, ~2.8 GB, checksummed).

## Layout

Figure folders are named by content, not by manuscript number. **`FIGURES.md`** maps every
main and supplementary figure to its folder, script and manuscript PDF.

| Folder | Figures |
|---|---|
| `fig_task_intro/` | Fig. 1 (task, recordings, feature vectors) |
| `fig_response_types/` | Fig. 2 (k-means response types) |
| `fig_anatomy/` | Fig. 3 (response types across regions) |
| `fig_response_correlations/si/` | Fig. 4 (correlations between neurons, tests beyond pairwise correlations), Fig. S5 |
| `fig_latency_tiling/` (+ `si/`) | Fig. 5 (stimulus-locked latency tiling), Fig. S1 |
| `lorenzo/manuscript_comparison_figure/` | Fig. S2 (comparison with the RRR of Posani et al.) |
| `wheel_movement_onset_review/` | Fig. S3 (needs the wheel objects in the ONE cache, or the release) |
| `fig_si/` | Fig. S4 (`k_robustness/`), Fig. S6, analyses feeding Fig. 5 and S1 |
| `sequence_analysis/` | latency-tiling analyses used by Figs. 4, 5 and S1 |
| `report/` | the LaTeX manuscript and the figure PDFs it includes |
| `auxiliary_analyses/` | analyses not used in the manuscript |

Each figure folder rebuilds its figure from the data stacks with `dmn_bwm.py` functions and
keeps its caches in its own `cache/`; see each folder's README. Figure style defaults:
`FIGURE_STYLE.md`. Fig. 4 and parts of other figures use PyTorch on a CUDA GPU (validated
against NumPy / scikit-learn).

## Requirements

Python 3.10+ (`environment.yml`): `numpy`, `scipy`, `pandas`, `matplotlib`, `scikit-learn`,
`torch`, `umap-learn`, `rastermap`, `iblatlas`, `brainbox`, `one.api`. Ghostscript, Inkscape
(Fig. 1) and tectonic (manuscript).
