#!/usr/bin/env python3
"""Fig. S10: Fig. 4 with a 40-cluster k-means basis instead of 100.

Runs the fig_response_correlations scripts with --nclus 40 --sort kmeans (as the manuscript Fig. 4) and writes everything (panel h images,
figure, printer PDF, panel values) into this folder. Caches go to fig_response_correlations/cache.
"""

import subprocess
import sys
from pathlib import Path

SI = Path(__file__).resolve().parent
PARENT = SI.parent
NCLUS = 40


def run(script, *args):
    subprocess.run([sys.executable, str(PARENT / script), *map(str, args)], check=True, cwd=PARENT)


if __name__ == "__main__":
    run("compute_synthetic.py", "--nclus", NCLUS)
    run("check_row_alignment.py", "--nclus", NCLUS)
    run("make_structured_mixed_selectivity.py", "--rasters", "--nclus", NCLUS, "--out-dir", SI,
        "--stem", "mixed_selectivity_k40", "--sort", "kmeans")
    print((SI / "mixed_selectivity_k40_values.txt").read_text())
