# Figure 5 (Figure 3 before the 2026-10-06 section reorder): stimulus-locked latency tiling, from dense to sparse cells (draft)

```bash
cd ~/dmn/sequence_analysis
python rt_split_latency.py && python degrade_tornado.py   # inputs for si panels d-g
# and ../fig_si: python cortex_rate_analysis.py   # input for panels i, j
python ari_controls.py   # panel f, si panel a (old Fig. 3b, d/e/h/j redone)
cd ~/dmn/fig_latency_tiling
python make_latency_tiling.py   # latency_tiling.pdf/.png/_printer.pdf
tectonic figure_with_caption.tex   # one A4 page: printer-compressed figure + figure_caption.tex
python si/make_latency_tiling_controls.py   # si/latency_tiling_controls.pdf/.png/_printer.pdf
cd si && tectonic latency_tiling_controls_with_caption.tex   # one A4 page
```

Since 2026-10-12 the controls (old panels e, g, i, j, o–q) are in `si/`: Figure 5—figure
supplement 1, panels a–g (`si/README.md`).

| | content | source |
|---|---|---|
| groups | sparse cells = clusters 0–7, 44–54, 61–69 (13,275 neurons); dense tiling = clusters 21–28 (4,251); clusters of the Rastermap fit stored in the CV stack (`concat_cvTrue.npy`: `isort`, `rm_labels`; Methods parameters, `grid_upsample=0`, fit on odd trials), contiguous blocks numbered along the order (since 2026-10-12; before: a separate `grid_upsample=10` fit with row-span clusters) | `../sequence_analysis/latency_tiling_groups.py` |
| a, b | Rastermap fit on odd trials (the stack's fit, `grid_upsample=0`); a odd (train), b even (test), same order; cluster boundaries, every 5th cluster number, group bars. Display: per neuron (x − median)/(99th pct − median) in [0, 1], values below the neuron's 80th percentile white, gamma 0.5 | `make_latency_tiling.py` |
| c, d, e | dense / sparse cells (even trials); reliability on the two concordant stimulus types | `../sequence_analysis/sparse_tornado.py` |
| f | cross-validated time–time correlation (odd bin × even bin), within and across trial types, dense vs sparse | `../sequence_analysis/ari_controls.py` |
| g, h | Beryl pies as Fig. 3f (count in group / count in stack, regions ≥ 50 neurons, canonical order, 10 largest labelled) | `make_latency_tiling.py` (`group_pies`) |
| i | Rastermap clusters of a: cortical fraction vs median rate (sparse / dense clusters coloured, top-left cluster 68 annotated); same as Fig. S11f | `../fig_si/cortex_rate_analysis.py`, drawn by `make_latency_tiling.py` (`cortex_rate_panel`) |
| j | Beryl regions (≥ 20 neurons): median firing rate vs number of neurons, Beryl colours (moved from Fig. S11) | `../fig_si/cortex_rate_analysis.py`, drawn by `make_latency_tiling.py` (`region_rate_panel`) |

Interpretation and caveats: `../sequence_analysis/SUMMARY.md`.
