# Data requirements

All data, caches and generated figures live in one folder, `$DMN_DATA`
(default `~/dmn`). ONE/Alyx downloads go to `$ONE_CACHE_DIR` (default
`~/Downloads/ONE`). The code reads both variables (`dmn_bwm.py`, the figure
folders, `sequence_analysis/`, `pipeline/`).

`pipeline/run_all.sh` builds everything below in order and skips what exists;
`pipeline/check_stacks.py` compares a build with the published one.

## 1. Downloaded automatically (ONE/Alyx access required)

For the 515 Brain-Wide Map insertions in `pipeline/insertions.csv`,
`dmn_bwm.concat_PETHs` loads trials (`stimOn_times`, `firstMovement_times`,
`feedback_times`, `intervals_1`, `probabilityLeft`, `contrastLeft`,
`contrastRight`, `choice`, `feedbackType`, plus the BWM trial-quality mask),
spike times and clusters, and good-unit metadata (`cluster_id`, `atlas_id`,
`x`, `y`, `z`, `channels`, `axial_um`, `lateral_um`, `uuids`). The BWM trials
aggregate table uses the `2024_Q2_IBL_et_al_BWM` release tag (patched in
`dmn_bwm._bwm_trials_tag_fix`).

## 2. Built by the pipeline (in `$DMN_DATA`)

| File | Step | Content |
|---|---|---|
| `concat/<eid>_<probe>.npy` (515) | `01_bundles.py` | per-insertion, per-trial PETHs for the 21 conditions |
| `concat_cvFalse.npy` | `02_stacks.py` | all trials per condition, z-scored; 54,719 neurons; all-trial UMAP. Used by the k-means analyses (Figs. 2–4) |
| `concat_cvTrue.npy` | `02_stacks.py` | odd/even split per condition (Methods): `X_odd` (= `concat_z_train`, Rastermap fit), `X_even` (= `concat_z`, display), `X` (all trials, same neurons); one canonical Rastermap fit on odd trials (`isort`, `rm_labels`); 54,569 neurons (those with enough trials in both halves). Used by everything involving Rastermap (Fig. 4, SI) |
| `kmeans_concat_cvFalse_n25_nclusrm100_zsc1.npy` | `03_caches.py` | the canonical 25-cluster k-means (all trials) of Figs. 2–4 |
| `counts/cf_Beryl_…npy`, `counts/cf_dec_…npy` | `03_caches.py` | region × cluster counts and decoding-based counts (Fig. 3f–j) |
| `alleninfo.npy` | automatic | Allen region colours (from `iblatlas`) |

Both stacks are needed. The CV stack holds only the 54,569 neurons with enough
trials in each half; its `X` is the all-trial vectors of those neurons, copied
from `concat_cvFalse.npy`. The k-means clustering (and its cluster numbers, used
in the text) is fit on all 54,719 neurons of the all-trial stack.

Every figure folder writes its own caches to its `cache/` (with the inputs
above symlinked in), and `sequence_analysis/` to `sequence_analysis/cache/` and
`results/`.

## 3. External inputs (not downloadable by the code)

- `bwm_decoding/{stimside,choice,feedback,wheel-speed,wheel-velocity}_stage2.pqt`:
  Brain-Wide Map decoding results (International Brain Laboratory, 2025), read
  by `dmn_bwm.get_dec_bwm` for the decoding-based specialization (Fig. 3h, j).
  Copy them from the BWM paper's released data.
- The Harris et al. (2019) cortico-thalamic hierarchy is included in the code
  (`dmn_bwm.harris_hierarchy`, `fig_anatomy/regenerate_from_data.py`).
- The reduced-rank regression (RRR) fit of Posani et al. for Fig. S2 and the
  task-variable half of Fig. S4e (`fig_response_correlations/si/mixed_selectivity/rrr_epairs.py`):
  `$DMN_DATA/lorenzo/sillouette_score/reproduction_audit/RRRGD_rank5_l275_full.pt`,
  `rank5_l275_cv_r2.npz` and the per-session features in
  `$DMN_DATA/lorenzo/brainwide-RRR-encoding-model/trained_RRR_model_paper178/session_features/`,
  fitted with the authors' code (Methods, Reduced-rank regression). Fig. S2 is included as a PDF.
- Fig. 3d,e use, by default, point-cloud images extracted from the earlier published
  figure (`fig_anatomy/source_alternate_*.jpg`); `--pointclouds data` renders them from the data.
- Fig. S8 (anatomical hierarchy) is static. `FIGURES.md` lists the script of every other figure.

## Optional

`load_atlas_data()` (ephys-atlas features, `ephys=True` stacks) needs the
`ephys_atlas` package and downloads release `2024_W50`; the manuscript figures
do not use it.
