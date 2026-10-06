"""Shared setup for the Figure 3 scripts.

``dmn_bwm`` reads inputs from, and writes caches and figures to, ``DMN_BASE``
(and ``pth_dmn`` for the cluster-count caches). ``use_private_base`` points both
at ``fig3/cache``, where the ~/dmn inputs (*.npy and counts/) are symlinked, so
nothing outside ``fig3`` is written.
"""

from pathlib import Path
import os
import sys

import matplotlib

matplotlib.use("Agg")  # dmn_bwm calls plt.show()/plt.ion(); never open windows

OUT = Path(__file__).resolve().parent
CACHE = OUT / "cache"
# Code (dmn_bwm.py) lives in the folder above; data in $DMN_DATA (default ~/dmn).
sys.path.insert(0, str(OUT.parent))
ROOT = Path(os.environ.get("DMN_DATA", Path.home() / "dmn"))

import dmn_bwm  # noqa: E402


def use_private_base():
    """Symlink ~/dmn inputs into fig3/cache and make it dmn_bwm's base folder."""
    (CACHE / "figs").mkdir(parents=True, exist_ok=True)
    for source in [*ROOT.glob("*.npy"), ROOT / "counts"]:
        link = CACHE / source.name
        if not link.exists():
            link.symlink_to(source)
    dmn_bwm.DMN_BASE = CACHE
    dmn_bwm.pth_dmn = CACHE
    return dmn_bwm
