#!/usr/bin/env python3
"""Cluster x cluster mean of the cross-validated correlation matrix (Fig. 4a, 2026-11-17).

Same matrix as compute_xcorr.py: S[i, j] = (C[i, j] + C[j, i]) / 2 with C[i, j] the Pearson
correlation of neuron i's odd-trial and neuron j's even-trial feature vector. For k-means
clusters A, B (kmeans_order.py; Rastermap clusters before 2026-11-20) the mean over all pairs i in A, j in B is exact via cluster-mean unit vectors:
mean_AB = (mo_A . me_B + me_A . mo_B) / 2. Within a cluster, self-pairs (i = j, the
neuron's odd/even reliability) are excluded. Checked against brute force for a few
cluster pairs. Output: results/xcorr_clusters.npz (mean, counts, cut).
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from compute_xcorr import unit_rows  # noqa: E402
import kmeans_order  # noqa: E402

DEV = torch.device("cuda")


def main():
    f = np.load(Path(os.environ.get("DMN_DATA", Path.home() / "dmn")) / "concat_cvTrue.npy",
                allow_pickle=True).flat[0]
    order, labels = kmeans_order.get_order(f)
    L = labels[order]
    Zo, _ = unit_rows(np.asarray(f["X_odd"])[order])
    Ze, _ = unit_rows(np.asarray(f["X_even"])[order])
    ref = np.load(HERE / "results" / "xcorr.npz")
    assert np.array_equal(ref["labels"], L), "row order differs from xcorr.npz"

    K = L.max() + 1
    o = torch.as_tensor(Zo, dtype=torch.float64, device=DEV)
    e = torch.as_tensor(Ze, dtype=torch.float64, device=DEV)
    lab = torch.as_tensor(L, device=DEV)
    n = torch.bincount(lab, minlength=K).double()
    so = torch.zeros(K, o.shape[1], dtype=torch.float64, device=DEV).index_add_(0, lab, o)
    se = torch.zeros_like(so).index_add_(0, lab, e)
    tot = 0.5 * (so @ se.T + se @ so.T)                      # sum of S over all pairs
    diag = torch.zeros(K, dtype=torch.float64, device=DEV).index_add_(0, lab, (o * e).sum(1))
    npair = torch.outer(n, n)
    tot.diagonal().sub_(diag)
    npair.diagonal().sub_(n)
    M = (tot / npair).cpu().numpy()

    # brute-force check on a few cluster pairs
    for a, b in ((0, 0), (3, 50), (22, 24), (80, 90)):
        ia, ib = np.flatnonzero(L == a), np.flatnonzero(L == b)
        C = Zo[ia].astype(np.float64) @ Ze[ib].T.astype(np.float64)
        Ct = Ze[ia].astype(np.float64) @ Zo[ib].T.astype(np.float64)
        S = 0.5 * (C + Ct)
        if a == b:
            S = S[~np.eye(len(ia), dtype=bool)]
        assert abs(S.mean() - M[a, b]) < 1e-6, (a, b, S.mean(), M[a, b])
    cut = np.r_[0, np.flatnonzero(np.diff(L)) + 1, L.size]
    np.savez(HERE / "results" / "xcorr_clusters.npz", mean=M, counts=n.cpu().numpy(), cut=cut)
    print(f"{K} clusters; mean r range [{M.min():.3f}, {M.max():.3f}]; diag median {np.median(np.diag(M)):.3f}")


if __name__ == "__main__":
    main()
