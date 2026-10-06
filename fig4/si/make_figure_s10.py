#!/usr/bin/env python3
"""Fig. S10: Fig. 4 with a 40-cluster k-means basis instead of 100.

Runs the fig4 scripts with --nclus 40 --sort kmeans (as the manuscript Fig. 4) and writes everything (panel h images,
figure, printer PDF, panel values) into this folder. Caches go to fig4/cache.
"""

import subprocess
import sys
from pathlib import Path

SI = Path(__file__).resolve().parent
FIG5 = SI.parent
NCLUS = 40


def run(script, *args):
    subprocess.run([sys.executable, str(FIG5 / script), *map(str, args)], check=True, cwd=FIG5)


if __name__ == "__main__":
    run("compute_synthetic.py", "--nclus", NCLUS)
    run("check_row_alignment.py", "--nclus", NCLUS)
    run("make_figure4.py", "--rasters", "--nclus", NCLUS, "--out-dir", SI,
        "--stem", "mixed_selectivity_k40", "--sort", "kmeans")
    print((SI / "mixed_selectivity_k40_values.txt").read_text())
