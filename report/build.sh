#!/usr/bin/env bash
# Compile the manuscript into report/main.pdf.
#   report/build.sh            compile with the figures in report/figures
#   report/build.sh --figures  first rebuild fig2, fig4, fig5 (+ Fig. S10) from data
#                              ($DMN_DATA, default ~/dmn) and copy them into report/figures
# Needs: python env with dmn_bwm's dependencies, Ghostscript, tectonic.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(dirname "$HERE")
PY=${PYTHON:-python}
TECTONIC=${TECTONIC:-tectonic}

if [[ "${1:-}" == "--figures" ]]; then
  (cd "$REPO/fig2" && "$PY" make_figure2.py --rasters)
  (cd "$REPO/fig4" && "$PY" regenerate_from_data.py)
  (cd "$REPO/fig5" && "$PY" compute_synthetic.py && "$PY" make_figure5.py --rasters \
     && "$PY" si/make_figure_s10.py)
  cp "$REPO/fig2/functional_response_structure_printer.pdf" "$HERE/figures/functional_response_structure.pdf"
  cp "$REPO/fig4/anatomy_function_correspondence_printer.pdf" "$HERE/figures/anatomy_function_correspondence.pdf"
  cp "$REPO/fig5/structured_mixed_selectivity_printer.pdf" "$HERE/figures/structured_mixed_selectivity.pdf"
  cp "$REPO/fig5/si/mixed_selectivity_k40_printer.pdf" "$HERE/figures/mixed_selectivity_k40.pdf"
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
