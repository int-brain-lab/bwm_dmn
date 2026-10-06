# Figure 5 (Figure 3 before the 2026-10-06 section reorder): stimulus-locked latency tiling, from dense to sparse cells (draft)

```bash
cd ~/dmn/sequence_analysis
python preview_upsample.py   # order for a, b (once)
python rt_split_latency.py && python degrade_tornado.py   # inputs for panels h, k-m
cd ~/dmn/fig5
python make_figure5.py   # contextual_neural_sequences.pdf/.png/_printer.pdf
tectonic figure_with_caption.tex   # one A4 page: figure + figure_caption.tex
```

| | content | source |
|---|---|---|
| a, b | Rastermap fit on odd trials (Methods parameters, `grid_upsample=10`, order in `../sequence_analysis/results/rastermap_upsample10.npz`); a odd (train), b even (test), same order. Display (identical for a, b): per neuron (x − median)/(99th pct − median) in [0, 1], values below the neuron's 80th percentile white, gamma 0.5. Brackets (row ranges in this order): dense latency tiling rows 11,800–16,300, sparse cells rows 29,600–35,600 (former "sequence cells") | `../sequence_analysis/preview_upsample.py` (order), `make_figure5.py` |
| c–j | reliability vs rate, timing replication at matched rate, held-out rasters of reliable cells, held-out latency, RT-frame test, latency distributions, fraction reliable; selection on the two concordant stimulus types, evaluation on the four others | `../sequence_analysis/sparse_tornado.py` |
| k–m | dense tiling cells + noise matched to the sparse cells' reliability (scaled by 1/sqrt(trials) per PETH type), own fit; statistics vs real sparse and dense tiling cells | `../sequence_analysis/degrade_tornado.py` |

Interpretation and caveats: `../sequence_analysis/SUMMARY.md`.
