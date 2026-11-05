# Figure 5—figure supplement 1: controls for the latency tiling (since 2026-10-12)

Panels moved out of Fig. 5 (old letters in brackets). Same data, groups and helpers as
`../make_latency_tiling.py`.

| | content | source |
|---|---|---|
| a–c (old o–q) | dense tiling + noise matched to the sparse cells' reliability, own fit, odd / even; statistics | `../../sequence_analysis/degrade_tornado.py` |
| d (old g) | held-out peak-latency replication per rate octile | `../../sequence_analysis/sparse_tornado.py` |
| e, f | regional held-out peak latency vs Harris et al. 2019 hierarchy score: e stimulus window, f PETH type m | `../../fig_si/latency_hierarchy_analysis.py` (run first; `draw_panel`) |
| g (restored 2026-10-13) | stimulus- vs movement-locking: fast vs slow reaction-time pattern correlation | `../../sequence_analysis/sparse_tornado.py` (`rt_split_latency.py` first) |

Left out on 2026-10-12 (still computed): neuron-shuffle control, held-out latency of reliable
cells (`ari_controls.py`, `sparse_tornado.py`).

`sparse_cortex_test.py` (2026-10-13): cortical enrichment of sparse tiling (chi-square,
rate-stratified Cochran-Mantel-Haenszel, insertion-clustered logistic regression) and the
k-means clusters of dense / sparse tiling; numbers in `sparse_cortex_test.txt` (Results and
Methods of the latency section).

```bash
python make_latency_tiling_controls.py   # latency_tiling_controls.pdf/.png/_printer.pdf (183 x 118 mm)
tectonic latency_tiling_controls_with_caption.tex   # one A4 page, printer-compressed figure + caption
```
