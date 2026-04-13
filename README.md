
# Spatially Distributed and Regionally Unbound Cellular-Resolution Brain-Wide Processing Loops in Mice

This repository contains the main analysis code supporting the publication:

Schartner, Michael, et al. "Spatially distributed and regionally unbound cellular resolution brain-wide processing loops in mice." bioRxiv (2025): 2025-07.
https://www.biorxiv.org/content/10.1101/2025.07.30.667641v1

The script `dmn_bwm.py` implements data processing, dimensionality reduction, anatomical mapping, and specialization analyses for the IBL Brain-Wide Map dataset (paper about the data and basic correlates: https://www.nature.com/articles/s41586-025-09235-0).
Data accessible here: https://docs.internationalbrainlab.org/notebooks_external/data_structure.html  

---

## Main-figure functions

This README is limited to the functions needed for the main paper figures.

### 1) Build per-insertion PETH data and stack to a supersession
- `get_all_PETHs_parallel(...)`  
  Computes/saves per-insertion PETH bundles.
- `stack_concat(vers='concat', cv=False/True, ...)`  
  Builds the concatenated feature matrix (`concat`, `concat_z`) and embeddings/sorting metadata.
- `concat_PETHs(...)`  
  Defines the 21 PETH conditions used in the main analysis (stimulus-, movement-, and feedback-aligned windows).

### 2) Functional clustering/embedding and rastermap figures
- `plot_dim_reduction(mapping='kmeans', algo='umap_z', ...)`  
  UMAP colored by k-means clusters; with `exa_kmeans=True` also shows cluster-average feature vectors.
- `plot_rastermap(...)`  
  Rastermap-sorted population plot (including CV usage from odd/even trial split produced by `stack_concat(cv=True)`).
- `plot_example_neurons(...)`  
  Single-cell example feature vectors per cluster.

### 3) Function–anatomy comparison and specialization
- `plot_dim_reduction(mapping='Beryl', ...)` and `plot_xyz(mapping='kmeans', ...)`  
  2D functional embedding vs 3D anatomical location views.
- `plot_cluster_profile(...)`  
  Per-cluster regional composition (pie/polar views).
- `clus_freqs(...)`  
  Region-wise functional composition / specialization summaries.
- `plot_three_swansons(...)`  
  Swanson maps for specialization-related summaries.
- `scat_dec_clus(...)`  
  Correlation between decoding-based and cluster-based specialization metrics.

---

## Requirements
- Python 3.10+  
- Dependencies: `numpy`, `scipy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `umap-learn`, `rastermap`, `iblatlas`, `brainbox`, `one.api`.  

---

## Usage
Typical workflow:
1. **Per-insertion PETHs:** `get_all_PETHs_parallel(...)`  
2. **Supersession stack:** `stack_concat(vers='concat', cv=False)` and `stack_concat(vers='concat', cv=True)`  
3. **Main figure generation:** use the functions listed above (`plot_dim_reduction`, `plot_rastermap`, `plot_example_neurons`, `plot_xyz`, `plot_cluster_profile`, `clus_freqs`, `plot_three_swansons`, `scat_dec_clus`).

Outputs are written under the local ONE cache `dmn/` tree (notably `dmn/res/`, `dmn/imgs/`, and `dmn/figs/`).
