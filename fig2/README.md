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

Rastermap-free since 2026-10-02 (the Rastermap panel, previously f, was removed;
re-lettered to read left to right: old d → a, e → b, a → c, b → d, c → e).
Panel b keeps consecutive labels >= 5 pt apart.

| Panel | dmn_bwm function | Data |
|---|---|---|
| a | `plot_rastermap(mapping="kmeans", sort_method="acs")`: neurons sorted by k-means cluster, no Rastermap ordering | all trials (`cv=False`, 54,719 neurons), grey on white with cluster boundary lines (no background colours) |
| b | neurons selected by `plot_fig2e_clean_examples` (`min_max_lz=None`) on the odd/even trial split, shown as their all-trial feature vectors; top-down, cluster numbers left, regions right | panel-a clusters via UUID |
| c | `plot_cluster_mean_PETHs` (restyled) | 25 k-means clusters, all trials |
| d | cluster means, min–max normalized in the window | clusters 21, 9, 12, 14, 5; segments L_sL_cL_b,s / L_sL_cL_b,m / L_move |
| e | cluster means, vertically offset | clusters 16, 22, 2, 11; segments L_b … mistake,s |

The `cv=False` clustering is the one that matches the manuscript's cluster
numbering (e.g. cluster 21 = stimulus, cluster 16 = largest).

## Changes from the original manuscript version

- **Rastermap panel removed** (was f); its Rastermap analyses belong with the
  odd/even CV stack (Fig. 3, Fig. 4a–c, SI). The old `rm_*` cache in
  `fig2/cache` is no longer used.
- **e:** the original labelled neurons with the `cv=True` cluster numbers (which
  do not correspond to panel b's clusters) and used an LZ <= 0.6 filter; both
  changed, so the example neurons differ.
- **Caption:** the original described a UMAP panel g that the figure did not
  contain (removed from the caption on 2026-10-01).
- Fonts and sizes follow `~/dmn/FIGURE_STYLE.md` (Arial, 5–6 pt, 183 mm).
