# Figure 1 (intro figure): elements for the redesign

**Draft:** `task_intro.svg/.pdf/.png` (183 × 75 mm; panel f, the analysis pipeline, dropped 2026-10-08; the mixed-selectivity schematic (old e, `make_panel_mixing.py`) dropped 2026-10-12, see `fig_response_correlations/si/mixed_selectivity/REPORT.md`), `make_task_intro.py`:

| Row | Panel | Content | Element (script) |
|---|---|---|---|
| 1 | a | data: task, recordings, timeline | `elements/panel_d.svg` (`make_panel_d.py`, `make_timeline.py`) |
| 1 | b | (wide, 115 mm, since 2026-10-12) grand average per trial type on a real time axis, as `dmn_bwm.plot_ave_PETHs`: movement and feedback windows at the session-averaged event times (movement 253 ms after the stimulus, feedback 451 ms after movement; 345 sessions, `compute_event_timing.py` → `cache/event_timing.npy`); rest moved next to quiescence (axis break); a selection of PETH types labelled | `elements/panel_b.svg` (`make_panels_bc.py`) |
| 2 | c | grand-average feature vector (21 PETH types; 58 mm, moved to row 2 on 2026-10-12) | `elements/panel_c.svg` (`make_panels_bc.py`) |
| 2 | d | How homogeneous is task-related neural response diversity? | `elements/panel_a.svg` (`make_panel_a.py`) |
| 2 | e | "Brain-wide latency tiling: everywhere? along the hierarchy?" (f until 2026-10-12): 52,975 neurons (the 1,744 without a spike in the window left out) sorted by peak time in L_sL_cL_b,s (0–150 ms after stimulus onset; window m until 2026-10-12), no trial split; six example neurons | `elements/panel_latency.svg` (`make_panel_latency.py`) |

`computational_architecture_and_task.svg` is an SVG export of the current
Illustrator figure (the manuscript uses
`figures/computational_architecture_and_task.pdf`). New panel elements are in
`elements/`; the final assembly is still to be decided.

| Element | File | Source |
|---|---|---|
| **a** assembled panel: task specialized regions? (left) vs. how mixed is each region? (right), final size 59 × 36 mm, vector; on the left brain two ringed dots are linked to example feature vectors under it (cluster means of a stimulus and a movement type, no cluster numbers) | `panel_a.svg/.pdf/.png` (raster versions: `a_specialized.png`, `a_observed.png`, `a_preview.png`) | `make_panel_a.py`: the abstract brain of the earlier panel a (`a_brain_outline.svg`, outline and sulci only) split into 12 regions; stippled "neurons", one colour per region (left) or sampled from the real 25-cluster proportions of the 12 most densely recorded Beryl regions (right) |
| **d** assembled panel (task, recordings, timeline), final size 62 × 32 mm | `panel_d.svg/.pdf/.png` | `make_panel_d.py`, positions as in the earlier panel d |
| d, timeline of the PETH windows (to scale, vector, final size) | `d_timeline.svg` | `make_timeline.py` (windows as in `dmn_bwm.concat_PETHs`) |
| d, task cartoon (two mice, screens) | `d_task_cartoon.png` (1022 × 576) | largest copy found: embedded in `~/dmn/figs/overleaf/fig1/single_panels/fig1_d.svg` |
| d, Neuropixels probe tip micrograph | `d_probe_tip.png` (256 × 418) | same source |
| d, 3D rendering of the recorded insertions | `d_insertions_brain.png` (444 × 506, upright) | same source (`d_insertions_brain_lowres.png`), panel letter "c" and "15 mice" painted out |
| alternatives, not used | `a_mockup_diversity.*` (regions as discs on a sagittal atlas outline), `d_insertions_dorsal.*` (insertions redrawn from data, dorsal view) | `make_panel_a_mockup.py`, `make_insertions.py` |

No higher-resolution originals of the raster elements were found on this
machine (searched `~/dmn`, `~/Dropbox`, the old manuscript figures and the whole
disk); the manuscript PDF embeds smaller copies of the same images.

```bash
python make_timeline.py && python make_panel_d.py   # data panel (Inkscape for the PDF/PNG export)
python make_panel_a.py                               # brains (iblenv; Inkscape; data from $DMN_DATA)
python make_panels_bc.py && python make_panel_latency.py   # iblenv; data from $DMN_DATA
python make_task_intro.py                               # assemble task_intro
gs -q -dNOPAUSE -dBATCH -dSAFER -sDEVICE=pdfwrite -dCompatibilityLevel=1.5 -dPDFSETTINGS=/printer \
   -sOutputFile=task_intro_printer.pdf task_intro.pdf
tectonic figure_with_caption.tex                     # one A4 page: printer PDF + figure_caption.tex
```
