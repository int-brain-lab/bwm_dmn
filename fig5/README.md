# Figure 5: structured, non-random selectivity

Regenerates the manuscript's Fig. 5 ("Structured, non-random selectivity at
the cellular level with random distributions at the regional level") from the
local DMN data. All computation goes through `~/dmn/dmn_bwm.py`, and all caches
and outputs stay in this folder (`fig5_common.py` makes `fig5/cache` dmn_bwm's
base folder, as in `fig2`).

## Run (iblenv)

```bash
cd ~/dmn/fig5
/home/mic/miniforge3/envs/iblenv/bin/python compute_synthetic.py      # once, ~2 min
/home/mic/miniforge3/envs/iblenv/bin/python make_figure5.py --rasters  # full figure
/home/mic/miniforge3/envs/iblenv/bin/python check_row_alignment.py     # j-k regression check
```

Outputs: `structured_mixed_selectivity.pdf/.svg/.png` and
`structured_mixed_selectivity_printer.pdf` (Ghostscript /printer).

## Files

- `compute_synthetic.py`: `regional_group(mapping="kmeans", synthetic=True,
  cv=False, nclus=100, nclus_s=100)`, with and without `syn_control`. It makes a
  100-cluster k-means basis V (random_state=0), real coefficients C = X V^T,
  i.i.d. synthetic coefficients B drawn from C's 200-bin marginals (seed 0),
  and the Rastermap orders of the synthetic and reconstructed responses.
- `regenerate_rastermap_panels.py`: panel h via `plot_rastermap(synthetic=True)`.
  Left: responses reconstructed from real C (`syn_control=True`). Right:
  synthetic B V.
- `si/`: Fig. S10, the same figure with a 40-cluster basis (`si/make_figure_s10.py`,
  which runs these scripts with `--nclus 40 --out-dir si`).
- `make_figure5.py`: panels a–k. The computations follow
  `dmn_bwm.plot_fig4_assembly` (this figure's earlier name); the layout follows
  the manuscript.

## Panels

a C, alpha columns in hierarchical order of corr(C) (used for all alpha axes),
neuron rows in Rastermap order. b corr(C). c neuron-by-neuron corr of C for
2000 evenly spaced neurons, kept in panel a's order (as the caption says). d 15
evenly spaced alpha marginals, synthetic (black) and real (green). e–g the same
as a–c for B. h Rastermap images. i PC0 (fit on C) of real and synthetic
neurons. j SE(PC0) per Beryl region (>= 20 neurons) vs the same labels shuffled.
k SE(PC0) per Beryl region vs per k-means cluster. EMD is the Wasserstein
distance divided by the pooled range.

## Row alignment of panels j–k (fixed 2026-09-30)

In `regional_group(synthetic=True)`, C's rows are X in Rastermap order
(`C = X[isort] @ V.T`). `r['Beryl']` is in stack order, and `r['acs']` is a
k-means fit on the synthetic responses. The manuscript version (via
`plot_fig4_assembly`) grouped C's PC0 by these, so each score was paired with
another neuron's region and with a synthetic-data cluster.

Fix in `~/dmn/dmn_bwm.py` (the change is saved as `dmn_bwm_alignment_fix.patch`;
`patch -R` undoes it):
- the synthetic analysis stores the row order as `r['C_rows']` (also in new
  caches; caches written earlier fall back to the stack isort they were built with);
- new `synthetic_row_labels(r)` returns each row's own Beryl region and
  real-data k-means basis cluster;
- `plot_fig4_assembly` and `plot_coeff_entropy_flatness_real_vs_synth` use it.

`check_row_alignment.py` verifies `C == X[C_rows] @ V.T` and prints both pairings:

| | manuscript pairing | fixed |
|---|---|---|
| i PC0 real vs synth, EMD | 0.14 | 0.14 |
| j Beryl vs random, EMD | 0.036 | 0.028 |
| k Beryl vs KMeans, EMD | 0.216 | 0.337 |
| k median SE(PC0), KMeans groups | 106 | 34 |

The figure uses the fixed pairing. The old pairing reproduces the manuscript's
published values exactly, which confirms the rebuild otherwise matches. Panel
j's EMD also depends on the random shuffle (seed 0), by about ±0.01. SE
scales with group size (k-means groups average ~550 neurons; Beryl regions
vary widely).
