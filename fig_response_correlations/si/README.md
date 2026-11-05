# Figure S10: mixed selectivity with a 40-cluster basis

Same analysis and layout as Fig. 4 (`../make_structured_mixed_selectivity.py`), with a 40-cluster
k-means basis instead of 100. Panels j–k use the row-aligned labels
(`dmn_bwm.synthetic_row_labels`; see `../README.md`).

## Run (iblenv)

```bash
cd ~/dmn/fig_response_correlations
/home/mic/miniforge3/envs/iblenv/bin/python si/make_mixed_selectivity_k40.py   # ~2.5 min from scratch
```

It computes the 40-cluster synthetic analyses (caches in `fig_response_correlations/cache`), runs
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

## Fig. S6: basis vectors (rebuilt 2026-10-12)

`make_basis_vectors.py` -> `basis_vectors_k100.pdf/.png/_printer.pdf`: the 100 basis vectors V
of Fig. 4 (from Fig. 4's own synthetic-analysis cache), in Fig. 4's column order
(`make_structured_mixed_selectivity.horder` of corr(C)) with its dendrogram; row k = coefficient alpha = k. The
earlier S6 (`dmn_bwm.plot_sorted_cluster_lines`) plotted the means of a separate k-means run,
whose numbering need not match Fig. 4's basis (rows matched only approximately); its corr(C)
panel duplicated Fig. 4b and was dropped.
