#!/usr/bin/env bash
# Compile the manuscript into report/main.pdf.
#   report/build.sh            compile with the figures in report/figures
#   report/build.sh --figures  first rebuild fig2-fig5 (+ Fig. S10) from data
#                              ($DMN_DATA, default ~/dmn) and copy them into report/figures
# FIG3_POINTCLOUDS=data renders Fig. 3d-e from the data (default: the images
# extracted from the earlier published figure, as in the manuscript).
# Needs: python env with dmn_bwm's dependencies (environment.yml), Ghostscript, tectonic.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(dirname "$HERE")
PY=${PYTHON:-python}
TECTONIC=${TECTONIC:-tectonic}

if [[ "${1:-}" == "--figures" ]]; then
  (cd "$REPO/fig2" && "$PY" make_figure2.py --rasters)
  (cd "$REPO/fig3" && "$PY" regenerate_from_data.py --pointclouds "${FIG3_POINTCLOUDS:-source}")
  (cd "$REPO/fig4" && "$PY" compute_synthetic.py && "$PY" make_figure4.py --rasters --sort kmeans \
     && "$PY" si/make_figure_s10.py)
  (cd "$REPO/sequence_analysis" && "$PY" preview_upsample.py && "$PY" rt_split_latency.py \
     && "$PY" degrade_tornado.py)
  (cd "$REPO/fig5" && "$PY" make_figure5.py)
  cp "$REPO/fig2/functional_response_structure_printer.pdf" "$HERE/figures/functional_response_structure.pdf"
  cp "$REPO/fig3/anatomy_function_correspondence_printer.pdf" "$HERE/figures/anatomy_function_correspondence.pdf"
  cp "$REPO/fig4/structured_mixed_selectivity_kmeans_sort_printer.pdf" "$HERE/figures/structured_mixed_selectivity.pdf"
  cp "$REPO/fig4/si/mixed_selectivity_k40_printer.pdf" "$HERE/figures/mixed_selectivity_k40.pdf"
  cp "$REPO/fig5/contextual_neural_sequences_printer.pdf" "$HERE/figures/contextual_neural_sequences.pdf"
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
