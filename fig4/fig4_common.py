"""Shared setup for the Figure 4 scripts.

``dmn_bwm`` reads its inputs from, and writes every cache and figure to,
``dmn_bwm.DMN_BASE``. ``use_private_base`` points that at ``fig4/cache``, where
the existing ~/dmn caches are symlinked, so nothing outside ``fig4`` is written.
"""

from pathlib import Path
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")  # dmn_bwm calls plt.show()/plt.ion(); never open windows

OUT = Path(__file__).resolve().parent
CACHE = OUT / "cache"
# Code (dmn_bwm.py) lives in the folder above; data in $DMN_DATA (default ~/dmn).
sys.path.insert(0, str(OUT.parent))
ROOT = Path(os.environ.get("DMN_DATA", Path.home() / "dmn"))

import dmn_bwm  # noqa: E402


def use_private_base():
    """Symlink ~/dmn inputs into fig4/cache and make it dmn_bwm's base folder."""
    (CACHE / "figs").mkdir(parents=True, exist_ok=True)
    for source in ROOT.glob("*.npy"):
        link = CACHE / source.name
        if not link.exists():
            link.symlink_to(source)
    dmn_bwm.DMN_BASE = CACHE
    return dmn_bwm


KMEANS_NCLUS = 25  # the canonical k-means sorting of Figs. 2a and 4a


def kmeans_canonical(d):
    """Canonical k-means sorting (Fig. 2a): 25 clusters fit on all trials (concat_z,
    cv=False, random_state=0), neurons stably sorted by cluster label.

    Returns (labels in stack order, centroids). The centroids are the cluster means,
    so nearest-centroid assignment of new vectors equals KMeans.predict."""
    rk = d.regional_group("kmeans", vers="concat", cv=False, nclus=KMEANS_NCLUS)
    labels = np.asarray(rk["acs"], dtype=int)
    X = np.asarray(rk["concat_z"], dtype=np.float32)
    centroids = np.stack([X[labels == k].mean(0) for k in range(KMEANS_NCLUS)])
    return labels, centroids


def nearest_centroid(X, centroids, chunk=5000):
    """Label of the nearest centroid (squared Euclidean) for each row of X."""
    c2 = (centroids ** 2).sum(1)
    out = np.empty(X.shape[0], dtype=int)
    for s in range(0, X.shape[0], chunk):
        x = np.asarray(X[s:s + chunk], dtype=np.float32)
        out[s:s + chunk] = np.argmin(c2[None] - 2 * x @ centroids.T, axis=1)
    return out
