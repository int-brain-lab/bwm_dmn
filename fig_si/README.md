# Supplementary figures

One SI figure at a time; each has an analysis script (results in `results/`), a plotting
script, `<name>_caption.tex`, and a one-page `<name>_with_caption.pdf` (tectonic +
`elife.cls` copy, as in `fig_latency_tiling/`).

## Revised old Fig. S11: cortical vs subcortical = mainly firing rate (2026-10-10)

Old S11 (`dmn_bwm.si_quantify_cortical_sub`; `figures/cortical_subcortical_specialization.pdf`)
showed that cortical and subcortical regions differ in their k-means response-type
composition (Fisher D, PERMANOVA, PCA, specialization). The revision asks how much of that
divide is firing rate.

- `cortex_rate_analysis.py` -> `results/cortex_rate.npz`, `results/cortex_rate.txt`
  (all numbers in the caption). Data: `~/dmn/concat_cvFalse.npy` + its 25-cluster k-means
  (as Fig. 3a), joined by uuid to the Fig. 5 CV stack (reliability, latency, Rastermap
  clusters, sparse / dense groups via `sequence_analysis/latency_tiling_groups.py`). ~10 s.
- `make_cortex_rate.py` -> `cortex_rate.pdf/.png` (183 x 138 mm), panels a–g (the region scatter, median rate vs neurons per region, moved to Fig. 5n).
- `cortex_rate_caption.tex`; `cortex_rate_with_caption.tex` -> `cortex_rate_with_caption.pdf` (numbered S11 for now).

Main results (neurons with Beryl or Cosmos root/void removed, 47,378 neurons): subcortical
neurons fire ~2x faster (6.9 vs 3.5 Hz); regional median rate separates cortex from
subcortex with AUC 0.87 and is unrelated to the number of neurons per region (Fig. 5n); per
neuron, rate is the best single predictor (AUC 0.64; response shape 0.56, k-means cluster
0.59, shape + rate 0.66); the old S11 composition separation (D 0.62) drops to 0.50 after
rate matching but stays significant; the same statistic on rate bins gives D 1.01. Sparse
cells (Fig. 5) are 1.64x more frequent in cortex, 1.29x after rate matching; dense tiling
is not enriched. Panel f is also Fig. 5m.

## Old Fig. S11 reproduced and checked: cortical--subcortical separation in composition space (2026-10-12)

Old S11 = `dmn_bwm.si_quantify_cortical_sub` (resubmission: D 0.577, F 7.32, R2 0.036, PCA
44/13.5%, EMD 0.041; cv=True, first/second-half stack). Rerun 2026-10-12: cv=True gives
D 0.666, F 11.3 (cache rebuilt since); cv=False (all trials, as Figs. 2-3) D 0.618, F 10.22.
Checks: (1) the PCA panel projected unweighted rows without subtracting the weighted mean,
so PC1 (41-44%) mostly reflected sqrt(n) row weighting -- fixed (PC1 24%, PC2 18%, cortical
and subcortical shifted along PC1); (2) specialization had no test and does not differ
(Mann-Whitney p = 0.80; the old text said subcortical regions are less specialized);
(3) a rate-matched Fisher D panel (0.50, still above all shuffles) was added and then removed again at the user's request.

- `cortical_subcortical_composition.py` -> `cortical_subcortical_composition.pdf/.png`
  (183 x 120 mm, a-d), `results/cortical_subcortical_composition.txt`; reuses
  `cortex_rate_analysis.py` (data, Fisher D).
- `cortical_subcortical_composition_caption.tex`; in the manuscript as Fig. S10
  (label `fig:cortical_subcortical_specialization`), cited for Fig. 3a, d.
