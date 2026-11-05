"""Fig. 5 neuron groups, defined by cluster numbers of the Rastermap fit of Fig. 5a, b:
the fit stored in the CV stack (concat_cvTrue.npy: isort, rm_labels), Methods parameters
(n_PCs=200, n_clusters=100, locality=0.75, grid_upsample=0, time_lag_window=5,
bin_size=1), fitted on the odd-trial feature vectors (concat_z_train). With
grid_upsample=0 every cluster is one contiguous block of rows, numbered 0-99 along the
order (rastermap_previews/upsample0_clusters.png).

  sparse tiling : clusters 0-7, 44-54, 61-69 (sharp diagonals on odd trials, each in one
                 rarely sampled trial type; ~2 Hz, low reliability)
  dense tiling : clusters 21-28 (graded stimulus-locked latencies, 28 earliest to 21 latest)

(Until 2026-10-12 a separate grid_upsample=10 fit was used, rastermap_upsample10.npz.)
"""

from pathlib import Path
import os

import numpy as np

ROOT = Path(os.environ.get("DMN_DATA", Path.home() / "dmn"))
SPARSE = [*range(0, 8), *range(44, 55), *range(61, 70)]
DENSE = list(range(21, 29))
SPARSE_TEXT = "clusters 0–7, 44–54, 61–69"
DENSE_TEXT = "clusters 21–28"


def load_fit(r):
    """Row order (isort) and per-neuron cluster labels of the Fig. 5a, b fit."""
    f = np.load(ROOT / "concat_cvTrue.npy", allow_pickle=True).flat[0]
    if not np.array_equal(np.asarray(f["uuids"]), np.asarray(r["uuids"])):
        raise RuntimeError("Rastermap fit and CV stack neurons differ")
    order, labels = np.asarray(f["isort"]), np.asarray(f["rm_labels"])
    if np.any(np.diff(labels[order]) < 0):
        raise RuntimeError("clusters are not contiguous along the order (grid_upsample != 0?)")
    return order, labels


def cluster_bounds(L):
    """Row span of each cluster in the a, b order (L: label per row): median rows and
    block boundaries (rows cut[c] <= row < cut[c + 1])."""
    med = np.array([np.median(np.flatnonzero(L == c)) for c in range(L.max() + 1)])
    cut = np.r_[0, np.flatnonzero(np.diff(L)) + 1, L.size].astype(float)
    return med, cut


def row_clusters(r):
    """Per neuron (stack order): its Rastermap cluster."""
    return load_fit(r)[1]


def group_indices(r):
    """{'dense': idx, 'sparse': idx}: neurons of the group's clusters, sorted by row in
    Fig. 5a, b."""
    order, labels = load_fit(r)
    L = labels[order]
    return {g: order[np.isin(L, cl)] for g, cl in (("dense", DENSE), ("sparse", SPARSE))}
