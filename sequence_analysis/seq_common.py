"""Shared setup for the sequence-analysis scripts (Rastermap exploration).

``dmn_bwm`` reads its inputs from, and writes every cache and figure to,
``dmn_bwm.DMN_BASE``. ``use_private_base`` points that at ``sequence_analysis/cache``, where
the existing ~/dmn caches are symlinked, so nothing outside ``sequence_analysis`` is written.
"""

from pathlib import Path
import os
import sys

import matplotlib

matplotlib.use("Agg")  # dmn_bwm calls plt.show()/plt.ion(); never open windows

OUT = Path(__file__).resolve().parent
CACHE = OUT / "cache"
PREV = OUT / "rastermap_previews"  # images and draft figures
# Code (dmn_bwm.py) lives in the folder above; data in $DMN_DATA (default ~/dmn).
sys.path.insert(0, str(OUT.parent))
ROOT = Path(os.environ.get("DMN_DATA", Path.home() / "dmn"))

import dmn_bwm  # noqa: E402

def use_private_base():
    """Symlink ~/dmn inputs into sequence_analysis/cache and make it dmn_bwm's base folder."""
    (CACHE / "figs").mkdir(parents=True, exist_ok=True)
    for source in ROOT.glob("*.npy"):
        link = CACHE / source.name
        if not link.exists():
            link.symlink_to(source)
    dmn_bwm.DMN_BASE = CACHE
    return dmn_bwm
