#!/usr/bin/env python3
"""Neuron order for Fig. 4a, b: 100 k-means clusters (2026-11-20, replaces the Rastermap order).

Run (iblenv):  python kmeans_order.py   (writes results/kmeans100_order.npz; ~1 min)

So that Rastermap appears only in the latency-tiling figure, Fig. 4a, b order neurons by
k-means instead. KMeans(100, random_state=0), as for the 25 clusters of Figs. 2-3, fitted on
the z-scored odd-trial feature vectors (concat_z_train), so that the displayed odd-vs-even
correlations stay cross-validated as before (the Rastermap order was also fitted on odd
trials). Clusters are ordered by average-linkage hierarchical clustering of their odd-trial
centroids (correlation distance) with optimal leaf ordering, and renumbered 0-99 along that
order. Within a cluster, neurons are sorted by corr(next centroid) - corr(previous centroid),
so that each cluster runs from its previous to its next neighbour.

get_order(f) -> (order, labels): stack indices in display order, and the (renumbered)
cluster label of each neuron in stack order.
"""

import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "kmeans100_order.npz"
K = 100


def load_stack():
    return np.load(Path(os.environ.get("DMN_DATA", Path.home() / "dmn")) / "concat_cvTrue.npy",
                   allow_pickle=True).flat[0]


def _unit(X):
    X = X - X.mean(1, keepdims=True)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1
    return X / n


def compute(f):
    from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering, leaves_list
    from scipy.spatial.distance import pdist
    from sklearn.cluster import KMeans

    X = np.asarray(f["concat_z_train"], dtype=float)  # z-scored odd-trial feature vectors
    km = KMeans(n_clusters=K, random_state=0).fit(X)
    lab0 = km.labels_.astype(int)
    cen = np.stack([X[lab0 == k].mean(0) for k in range(K)])
    d = pdist(cen, metric="correlation")
    Z = optimal_leaf_ordering(linkage(d, method="average"), d)
    leaf = leaves_list(Z)                                  # old cluster ids in display order
    new = np.empty(K, int); new[leaf] = np.arange(K)
    labels = new[lab0]
    cen = cen[leaf]
    U, Cu = _unit(X), _unit(cen)
    order = []
    for k in range(K):
        idx = np.flatnonzero(labels == k)
        nxt = U[idx] @ Cu[min(k + 1, K - 1)] if k < K - 1 else np.zeros(idx.size)
        prv = U[idx] @ Cu[max(k - 1, 0)] if k > 0 else np.zeros(idx.size)
        order.append(idx[np.argsort(nxt - prv, kind="stable")])
    order = np.concatenate(order)
    assert np.array_equal(np.sort(order), np.arange(len(labels)))
    assert np.all(np.diff(labels[order]) >= 0)
    np.savez(OUT, order=order, labels=labels, centroids=cen.astype(np.float32),
             counts=np.bincount(labels, minlength=K))
    return order, labels


def get_order(f=None):
    if OUT.exists():
        R = np.load(OUT)
        return R["order"], R["labels"]
    return compute(load_stack() if f is None else f)


if __name__ == "__main__":
    o, l = compute(load_stack())
    c = np.bincount(l, minlength=K)
    print(f"{len(o)} neurons, {K} clusters; size median {int(np.median(c))}, range {c.min()}-{c.max()}")
