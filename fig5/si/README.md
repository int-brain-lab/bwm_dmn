# Figure S10: mixed selectivity with a 40-cluster basis

Same analysis and layout as Fig. 5 (`../make_figure5.py`), with a 40-cluster
k-means basis instead of 100. Panels j–k use the row-aligned labels
(`dmn_bwm.synthetic_row_labels`; see `../README.md`).

## Run (iblenv)

```bash
cd ~/dmn/fig5
/home/mic/miniforge3/envs/iblenv/bin/python si/make_figure_s10.py   # ~2.5 min from scratch
```

It computes the 40-cluster synthetic analyses (caches in `fig5/cache`), runs
the row-alignment check, and writes into this folder:
`mixed_selectivity_k40.pdf/.svg/.png`, `mixed_selectivity_k40_printer.pdf`
(the manuscript file), the panel h images, and `mixed_selectivity_k40_values.txt`.

## Values

| | published S10 | this rebuild |
|---|---|---|
| i PC0 real vs synth, EMD | 0.126 | 0.126 |
| j Beryl vs random, EMD | 0.036 | 0.0276 |
| k Beryl vs KMeans, EMD | 0.282 | 0.351 |

With the old (misaligned) pairing, the rebuild gives exactly the published
0.036 and 0.282, so the only difference is the label fix.
