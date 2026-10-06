"""Shared setup for the Figure 2 scripts.

``dmn_bwm`` reads its inputs from, and writes every cache and figure to,
``dmn_bwm.DMN_BASE``. ``use_private_base`` points that at ``fig2/cache``, where
the existing ~/dmn caches are symlinked, so nothing outside ``fig2`` is written.
"""

from pathlib import Path
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")

OUT = Path(__file__).resolve().parent
CACHE = OUT / "cache"
# Code (dmn_bwm.py) lives in the folder above; data in $DMN_DATA (default ~/dmn).
sys.path.insert(0, str(OUT.parent))
ROOT = Path(os.environ.get("DMN_DATA", Path.home() / "dmn"))

import dmn_bwm  # noqa: E402

# Cluster numbers as printed in the manuscript (1-based; r['acs'] is 0-based).
B_CLUSTERS = {21: "stim", 9: "integ", 12: "mvt init", 14: "mvt", 5: "mvt"}
B_SEGMENTS = ["stimLbLcL", "sLbLchoiceL", "choiceL"]
C_CLUSTERS = {16: "change", 22: "mistake", 2: "R block", 11: "L block"}
C_SEGMENTS = ["blockL", "blockR", "quiescence", "block_change_s", "stimLbLcL",
              "stimLbRcL", "stimRbRcR", "stimRbLcR", "mistake_s"]


def use_private_base():
    """Symlink ~/dmn inputs into fig2/cache and make it dmn_bwm's base folder."""
    (CACHE / "figs").mkdir(parents=True, exist_ok=True)
    for source in ROOT.glob("*.npy"):
        link = CACHE / source.name
        if not link.exists():
            link.symlink_to(source)
    dmn_bwm.DMN_BASE = CACHE
    return dmn_bwm
