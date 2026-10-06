# Figure 3 (anatomy/function correspondence; Figure 4 before the 2026-10-06 section reorder)

In the compressed-printer manuscript this figure is Figure 4, on page 13 of
`IBL_supersession_paper_resubmission_compressed_printer.pdf`. It is the
anatomy/function correspondence figure with panels **a--j**, beginning with
Rastermap-sorted response vectors and ending with anatomical hierarchy versus
specialization.

Figure numbering changed during revision. The exact assembled source for this
printer figure is:

`IBL_supersession_paper_resubmission/figures/anatomy_function_correspondence.pdf`

It was assembled in Adobe Illustrator from analysis panels, rather than made by
one Python function. `regenerate_fig3.py` therefore preserves that publication
source as the PDF master and creates SVG and 600-dpi PNG review copies. It also
checks that the regenerated PDF is byte-identical to the source.

## Analysis provenance

The underlying analyses are in `~/dmn/dmn_bwm.py` and the historical scripts in
`~/Dropbox/scripts/IBL/`:

- **a--c (Rastermap-free, all trials, 54,719 neurons; since 2026-10-02):**
  a `regenerate_panel_a_kmeans.py`: `plot_rastermap(mapping="kmeans",
  sort_method="acs", cv=False)`, rows sorted by k-means cluster (25), row
  background = each neuron's Beryl colour. b `regenerate_panel_b.py`: the same
  neurons in canonical Beryl order (`mapping="Beryl", sort_method="acs", cv=False`),
  Beryl background (root/void neurons have no Beryl colour and appear black).
  c `regenerate_panel_c.py`: all neurons of CP, MRN, ZI, MOp, CA1 and CUL4,5, each
  region sorted by k-means cluster, cluster-colour background, every bin.
  The earlier Rastermap version of a (`regenerate_panel_a.py`) is no longer used.
- **d--e:** The four point-cloud images are extracted without alteration from
  `IBL_supersession_paper_resubmission/figures/anatomy_function_correspondence_alternate.pdf`.
  Both pairs show UMAP on the left and anatomical xyz on the right, with UMAP
  axes and an xyz orientation triad overlaid during assembly.
- **f:** drawn directly in `regenerate_from_data.py` as vector polar pies
  (`cluster_region_fractions`, `draw_cluster_pies`). Wedges are the same as in
  `dmn_bwm.plot_cluster_profile(norm_reg_count=True, canonical_order=True)`:
  root/void removed, each region's count in the cluster divided by its total
  count, canonical Beryl order. The 5 largest wedges are labelled radially,
  centred on the wedge's middle angle. Label font size is linear in wedge
  fraction on one scale shared by all pies (smallest labelled wedge 3.5 pt,
  largest wedge 7 pt). Overlaps (tested on the rotated text rectangles) are
  resolved only by moving labels outward along their own radius
  (`separate_labels`). Pie radius is 1/`PIE_RMAX` of its cell. Colours are the
  Allen atlas colours of the Beryl regions (258 regions, 63 distinct colours),
  so the 5 labelled wedges get white outlines to stay visible inside
  same-coloured blocks. `check_labels_centred` stops the build if any rendered
  label's centre is more than 1 degree off its wedge's middle angle.
  `regenerate_panel_f.py` (the earlier 800-dpi PNG version) is no longer used.
- **d–e:** the xyz tripods carry a vertical "anatomical space" label.
- **g:** `clus_freqs` plots the original eight example regions (PA, PAA, MOB,
  MEA, MRN, SCm, PRNr, PGRN) in two compact columns with in-panel labels and
  regional cluster-specialization scores.
- **h:** regional specialization distributions, directly below g.
- **i:** portrait Swanson flatmaps produced with `plot_swanson_vector`, tightly
  paired above j. The left map has a matching raw-specialization colorbar and
  a key for regions without data. Panel f fills the full lower-left block.
- **j:** specialization versus the Harris anatomical hierarchy; the current
  comparison helper starts at `plot_specialization_comparison` in the analysis
  script.

The main inputs are `concat_cvFalse.npy`, the 25-cluster k-means cache, Beryl
region assignments, BWM decoding tables, and the Harris hierarchy table.

## Private cache

`fig3_common.py` makes `fig3/cache` dmn_bwm's base folder (`DMN_BASE` and
`pth_dmn`), with `~/dmn/*.npy` and `counts/` symlinked in. Every cache and
intermediate figure (e.g. the panel a/b Rastermap SVGs) is written there, and
nothing outside `fig3` is written.

## Run (iblenv)

```bash
cd fig3
python regenerate_from_data.py   # panels a-c via regenerate_panel_{a,b,c}.py, then a-j + printer PDF
```

`anatomy_function_correspondence_panels.pdf/.svg/.png` is the figure;
`anatomy_function_correspondence_printer.pdf` is the compressed manuscript
file (`report/figures/anatomy_function_correspondence.pdf`). Panels d–e use the
four point-cloud images `source_alternate_*.jpg` (pixels unchanged within the
displayed crops); all other panels are computed from the data. Beryl colours
use `dmn_bwm.py`'s palette (Allen atlas colours, with a readable cerebellar
olive). Data are read from `$DMN_DATA` (default `~/dmn`).
