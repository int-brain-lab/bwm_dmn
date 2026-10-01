# Figure 2: functional response structure

Recreates the manuscript's Fig. 2 ("Brain-wide activation across trial types
and trial intervals reveals functionally interpretable features ...") from the
local DMN data, using `~/dmn/dmn_bwm.py` functions.

## Run (iblenv)

```bash
cd ~/dmn/fig2
/home/mic/miniforge3/envs/iblenv/bin/python make_figure2.py --rasters  # full rebuild
/home/mic/miniforge3/envs/iblenv/bin/python make_figure2.py            # reuse raster PNGs
```

Outputs: `functional_response_structure.pdf/.svg/.png` and the
Ghostscript-compressed `functional_response_structure_printer.pdf`.

## Files

- `fig2_common.py`: makes `fig2/cache` dmn_bwm's base folder (`DMN_BASE`). The
  existing `~/dmn/*.npy` inputs are symlinked there, so every new cache and every
  figure dmn_bwm saves lands in `fig2/cache`, and nothing outside `fig2` is written.
  Also holds the clusters and trial segments chosen for panels b and c.
- `regenerate_raster_panels.py`: panels d and f. It runs `plot_rastermap`,
  captures the RGBA image it draws, averages blocks of 5 neurons, and saves
  `panel_d_kmeans_raster.png`, `panel_f_rastermap_raster.png` and
  `raster_rows.npz` (cluster order/boundaries).
- `make_figure2.py`: assembles a–f at 183 mm width.

## Panel provenance

| Panel | dmn_bwm function | Data |
|---|---|---|
| a | `plot_cluster_mean_PETHs` (restyled) | 25 k-means clusters, `cv=False` (54,719 neurons) |
| b | cluster means, min–max normalized in the window | clusters 21, 9, 12, 14, 5; segments L_sL_cL_b,s / L_sL_cL_b,m / L_move |
| c | cluster means, vertically offset | clusters 16, 22, 2, 11; segments L_b … mistake,s |
| d | `plot_rastermap(mapping="kmeans", sort_method="acs")` | `cv=False`, cluster-coloured rows |
| e | neurons selected by `plot_fig2e_clean_examples` (`min_max_lz=None`), drawn top-down (cluster 1 at top; cluster numbers left, regions right) | `cv=True` trial halves, labelled with each neuron's panel-a (`cv=False`) cluster via UUID; 2 reliable neurons per cluster |
| f | `plot_rastermap(mapping="rm", sort_method="rastermap")` | `cv=True` (53,021 neurons, held-out half), 100 Rastermap clusters |

The `cv=False` clustering is the one that matches the manuscript's cluster
numbering (e.g. cluster 21 = stimulus, cluster 16 = largest).

Panel e needs the trial halves, which only exist in the `cv=True` data, but the
`cv=True` and `cv=False` clusterings number clusters differently and agree
for only 43% of neurons. `full_trial_clusters` in `make_figure2.py` therefore
gives every `cv=True` neuron its `cv=False` cluster (same UUID), so the numbers
in e are panel a's clusters. Selection: reliability (half vs half r >= 0.2),
firing rate 0.1-100 (stored units), no Lempel-Ziv filter; ranked by
held-out trace vs training-half cluster mean, two different regions per
cluster. The chosen traces correlate with panel a's cluster means at median
r = 0.90 (min 0.53). Note that the `cv=False` cluster assignment itself used
all trials. The selection is written to `panel_e_selection.csv`.

## Known differences from the manuscript version

- **f:** no Rastermap cache for the manuscript's ordering existed locally, so
  `fig2/cache/rm_concat_cvTrue_nclusrm100_zsc1.npy` was computed afresh. The
  main blocks (sequence whorl at the top, feedback block at the bottom) recur,
  but block positions differ (e.g. the dark "rest" block is at ~35k rather
  than ~50k).
- **e:** the manuscript's panel e labelled neurons with the `cv=True` cluster
  numbers (which do not correspond to panel a) and used an LZ <= 0.6 filter;
  both are changed here, so the example neurons differ from the manuscript.
- **Caption:** the manuscript caption describes a panel g (UMAP) and says f's
  clusters are numbered on the right. Neither is in the manuscript figure, so
  neither is drawn here.
- Fonts and sizes follow `~/dmn/FIGURE_STYLE.md` (Arial, 5–6 pt, 183 mm)
  rather than the Illustrator-assembled original.
