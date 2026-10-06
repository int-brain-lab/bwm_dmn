#!/usr/bin/env bash
# Build the manuscript from the IBL data up (repository layout: run from anywhere).
#   pipeline/run_all.sh            all steps; each skips what already exists
#   pipeline/run_all.sh --from N   start at step N (1-5)
# Steps: 1 download + per-insertion PETH bundles (hours, needs ONE/Alyx access)
#        2 stacks (all-trial and odd/even CV, with the canonical Rastermap fit)
#        3 shared caches (25-cluster k-means, region/cluster counts)
#        4 check the stacks against the published build
#        5 all figures with code (Figs. 2-5, S10) + compile -> report/main.pdf
# Data and caches: $DMN_DATA (default ~/dmn); see DATA_REQUIREMENTS.md.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(dirname "$HERE")
PY=${PYTHON:-python}
FROM=1
if [[ "${1:-}" == "--from" ]]; then FROM=$2; fi

step() { echo; echo "=== step $1: $2"; }
if (( FROM <= 1 )); then step 1 "PETH bundles"; "$PY" "$HERE/01_bundles.py"; fi
if (( FROM <= 2 )); then step 2 "stacks"; "$PY" "$HERE/02_stacks.py"; fi
if (( FROM <= 3 )); then step 3 "caches"; "$PY" "$HERE/03_caches.py"; fi
if (( FROM <= 4 )); then step 4 "check"; "$PY" "$HERE/check_stacks.py"; fi
if (( FROM <= 5 )); then
  step 5 "figures and manuscript"
  # Fig. 3d-e point clouds rendered from the data (not the extracted images).
  FIG3_POINTCLOUDS=data PYTHON="$PY" "$REPO/report/build.sh" --figures
fi
