# Auxiliary analyses

Earlier IBL analysis modules kept for reference and for the helpers `dmn_bwm.py`
imports from them (`dmn_bwm.py` adds this folder to `sys.path`).

- `granger.py`: region-to-region Granger causality (spectral_connectivity) and
  structural connectivity; `dmn_bwm` uses `get_volume`, `get_centroids`,
  `get_res`, `get_structural` and `get_ari`.
- `state_space_bwm.py`: Brain-Wide Map state-space (trajectory) analysis;
  `dmn_bwm` uses `get_cmap_bwm` and `pre_post`.
- `cell_corr.py`: cell-to-cell activity correlations per session (not used by
  `dmn_bwm` or the manuscript figures).
