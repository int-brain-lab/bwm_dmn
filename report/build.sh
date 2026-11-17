#!/usr/bin/env bash
# Compile the manuscript into report/main.pdf.
#   report/build.sh            compile with the figures in report/figures
#   report/build.sh --figures  first rebuild every figure that has code (see ../FIGURES.md)
#                              from the data ($DMN_DATA, default ~/dmn) and copy it into
#                              report/figures
# FIG3_POINTCLOUDS=data renders Fig. 3d-e from the data (default: the images
# extracted from the earlier published figure, as in the manuscript).
# Needs: python env with dmn_bwm's dependencies (environment.yml; torch with CUDA for
# Fig. 4c-e), Ghostscript, Inkscape (Fig. 1), tectonic.
# Steps that need raw IBL data (ONE) are skipped when their outputs exist, e.g. after
# pipeline/install_intermediate.py: the reaction-time split (Fig. S1g), the wheel analysis
# (Fig. S3) and the Fig. 1 event timing. The Rastermap order of Fig. 5 is likewise reused
# when present (it can differ between Rastermap versions). Fig. S2 is drawn from its saved
# result tables (full recomputation: lorenzo/manuscript_comparison_figure/README.md);
# Fig. S7 (anatomical hierarchy) is static.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(dirname "$HERE")
PY=${PYTHON:-python}
TECTONIC=${TECTONIC:-tectonic}
FIG=$HERE/figures

if [[ "${1:-}" == "--figures" ]]; then
  run() { (cd "$REPO/$1" && shift && "$PY" "$@"); }
  have() { for f in "$@"; do [[ -e "$REPO/$f" ]] || return 1; done; }
  # analyses that feed several figures
  run fig_si cortex_rate_analysis.py
  run fig_si latency_hierarchy_analysis.py
  have sequence_analysis/results/rastermap_upsample10.npz || run sequence_analysis preview_upsample.py
  have sequence_analysis/results/neurons.csv sequence_analysis/results/peths.npz || run sequence_analysis rt_split_latency.py
  run sequence_analysis degrade_tornado.py
  have fig_task_intro/cache/event_timing.npy || run fig_task_intro compute_event_timing.py
  # Fig. 1
  run fig_task_intro make_timeline.py
  run fig_task_intro make_panel_d.py
  run fig_task_intro make_panel_a.py
  run fig_task_intro make_panels_bc.py
  run fig_task_intro make_panel_latency.py
  run fig_task_intro make_task_intro.py
  gs -q -dNOPAUSE -dBATCH -dSAFER -sDEVICE=pdfwrite -dCompatibilityLevel=1.5 -dPDFSETTINGS=/printer \
     -sOutputFile="$FIG/task_data_intro.pdf" "$REPO/fig_task_intro/task_intro.pdf"
  # Figs. 2, 3
  run fig_response_types make_response_types.py --rasters
  run fig_anatomy regenerate_from_data.py --pointclouds "${FIG3_POINTCLOUDS:-source}"
  # Fig. 4 (correlations between neurons; tests beyond pairwise correlations)
  run fig_response_correlations compute_synthetic.py
  run fig_response_correlations/si/neuron_correlation compute_xcorr.py            # a, b
  run fig_response_correlations/si/neuron_correlation compute_cluster_means.py    # a
  run fig_response_correlations/si/neuron_correlation pick_window.py              # b
  run fig_response_correlations/si/neuron_correlation group_anatomy.py            # Discussion numbers
  run fig_response_correlations/si/mixed_selectivity selectivity_test.py          # c-e, top 10 PCs
  run fig_response_correlations/si/mixed_selectivity extra_nulls.py
  run fig_response_correlations/si/mixed_selectivity full_space_epairs.py         # e, all bins
  have fig_response_correlations/si/mixed_selectivity/cache/full_space_angles.npz || \
    run fig_response_correlations/si/mixed_selectivity full_space_epairs.py --save-angles   # d
  run fig_response_correlations/si/mixed_selectivity rrr_epairs.py                # e, RRR
  run fig_response_correlations/si/mixed_selectivity extra_panels_cd.py           # c, d
  run fig_response_correlations/si/mixed_selectivity knn_sweep.py                 # Methods numbers
  run fig_response_correlations/si/mixed_selectivity/si_figure make_response_correlations.py
  # Fig. 5
  run fig_latency_tiling make_latency_tiling.py
  # SI
  run fig_latency_tiling/si make_latency_tiling_controls.py                       # S1
  run lorenzo/manuscript_comparison_figure compose_si_lorenzo_native.py           # S2
  have wheel_movement_onset_review/results/trial_metrics.pqt || \
    run wheel_movement_onset_review analyze_wheel_movement_onset.py
  run wheel_movement_onset_review make_panel_overlap.py                           # S3
  run fig_si/k_robustness compute_k_robustness.py
  run fig_si/k_robustness make_regional_cluster_composition.py                    # S4
  run fig_response_correlations/si make_specialization_metrics.py                 # S5
  run fig_si cortical_subcortical_composition.py                                  # S6
  cp "$REPO/fig_response_types/functional_response_structure_printer.pdf" "$FIG/functional_response_structure.pdf"
  cp "$REPO/fig_anatomy/anatomy_function_correspondence_printer.pdf" "$FIG/anatomy_function_correspondence.pdf"
  cp "$REPO/fig_response_correlations/si/mixed_selectivity/si_figure/response_correlations.pdf" "$FIG/response_correlations.pdf"
  cp "$REPO/fig_latency_tiling/latency_tiling_printer.pdf" "$FIG/latency_tiling.pdf"
  cp "$REPO/fig_latency_tiling/si/latency_tiling_controls_printer.pdf" "$FIG/latency_tiling_controls.pdf"
  cp "$REPO/figs/si_lorenzo_panels.pdf" "$FIG/rrr_peth_selectivity_comparison.pdf"
  cp "$REPO/wheel_movement_onset_review/results/movement_onset_subthreshold_wheel.pdf" "$FIG/movement_onset_subthreshold_wheel.pdf"
  cp "$REPO/fig_si/k_robustness/regional_cluster_composition.pdf" "$FIG/regional_cluster_composition.pdf"
  cp "$REPO/fig_response_correlations/si/specialization_metric_comparison.pdf" "$FIG/specialization_metric_comparison.pdf"
  cp "$REPO/fig_si/cortical_subcortical_composition.pdf" "$FIG/cortical_subcortical_composition.pdf"
fi

# tectonic (XeTeX) cannot find "Latin Modern Mono" by name; compile a copy whose
# class loads the same font by file name. The sources stay unchanged.
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
cp -r "$HERE"/. "$TMP"
sed -i 's/\\setmonofont\[Scale=MatchUppercase\]{Latin Modern Mono}/\\setmonofont[Scale=MatchUppercase,Extension=.otf]{lmmono10-regular}/' "$TMP/elife.cls"
(cd "$TMP" && "$TECTONIC" --keep-logs main.tex)
cp "$TMP/main.pdf" "$HERE/main.pdf"
echo "Wrote $HERE/main.pdf"
