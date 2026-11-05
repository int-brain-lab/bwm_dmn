#!/usr/bin/env bash
# Compile the manuscript into report/main.pdf.
#   report/build.sh            compile with the figures in report/figures
#   report/build.sh --figures  first rebuild every figure that has code (see ../FIGURES.md)
#                              from the data ($DMN_DATA, default ~/dmn) and copy it into
#                              report/figures
# FIG3_POINTCLOUDS=data renders Fig. 3d-e from the data (default: the images
# extracted from the earlier published figure, as in the manuscript).
# Needs: python env with dmn_bwm's dependencies (environment.yml; torch with CUDA for
# Fig. S4), Ghostscript, Inkscape (Fig. 1), tectonic.
# Not rebuilt here: Fig. S2 (needs the RRR fit of Posani et al.), Fig. S3 (needs the wheel
# objects in the ONE cache; ../wheel_movement_onset_review), Fig. S4e right half (RRR fit),
# Fig. S8 (static).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(dirname "$HERE")
PY=${PYTHON:-python}
TECTONIC=${TECTONIC:-tectonic}
FIG=$HERE/figures

if [[ "${1:-}" == "--figures" ]]; then
  run() { (cd "$REPO/$1" && shift && "$PY" "$@"); }
  # analyses that feed several figures
  run fig_si cortex_rate_analysis.py
  run fig_si latency_hierarchy_analysis.py
  run sequence_analysis preview_upsample.py
  run sequence_analysis rt_split_latency.py
  run sequence_analysis degrade_tornado.py
  # Fig. 1
  run fig_task_intro make_timeline.py
  run fig_task_intro make_panel_d.py
  run fig_task_intro make_panel_a.py
  run fig_task_intro make_panels_bc.py
  run fig_task_intro make_panel_latency.py
  run fig_task_intro make_task_intro.py
  gs -q -dNOPAUSE -dBATCH -dSAFER -sDEVICE=pdfwrite -dCompatibilityLevel=1.5 -dPDFSETTINGS=/printer \
     -sOutputFile="$FIG/task_data_intro.pdf" "$REPO/fig_task_intro/task_intro.pdf"
  # Figs. 2-4
  run fig_response_types make_response_types.py --rasters
  run fig_anatomy regenerate_from_data.py --pointclouds "${FIG3_POINTCLOUDS:-source}"
  run fig_latency_tiling make_latency_tiling.py
  # SI
  run fig_latency_tiling/si make_latency_tiling_controls.py                       # S1
  run fig_response_correlations compute_synthetic.py
  run fig_response_correlations/si/neuron_correlation compute_xcorr.py            # S4a,b
  run fig_response_correlations/si/mixed_selectivity selectivity_test.py          # S4c-e
  run fig_response_correlations/si/mixed_selectivity extra_nulls.py
  run fig_response_correlations/si/mixed_selectivity/si_figure make_response_correlations.py
  run fig_si/k_robustness compute_k_robustness.py
  run fig_si/k_robustness make_regional_cluster_composition.py                    # S5
  run fig_response_correlations/si make_specialization_metrics.py                 # S6
  run fig_si cortical_subcortical_composition.py                                  # S7
  cp "$REPO/fig_response_types/functional_response_structure_printer.pdf" "$FIG/functional_response_structure.pdf"
  cp "$REPO/fig_anatomy/anatomy_function_correspondence_printer.pdf" "$FIG/anatomy_function_correspondence.pdf"
  cp "$REPO/fig_latency_tiling/latency_tiling_printer.pdf" "$FIG/latency_tiling.pdf"
  cp "$REPO/fig_latency_tiling/si/latency_tiling_controls_printer.pdf" "$FIG/latency_tiling_controls.pdf"
  cp "$REPO/fig_response_correlations/si/mixed_selectivity/si_figure/response_correlations.pdf" "$FIG/response_correlations.pdf"
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
