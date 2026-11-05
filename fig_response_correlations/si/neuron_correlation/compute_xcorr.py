#!/usr/bin/env python3
"""Cross-validated neuron x neuron correlation matrix of the feature vectors.

Run (iblenv):  python compute_xcorr.py

C[i, j] = Pearson correlation between neuron i's odd-trial feature vector and neuron
j's even-trial feature vector (1872 bins; ~/dmn/concat_cvTrue.npy: X_odd, X_even;
54,569 neurons). No spikes are shared between the two, so the diagonal is each neuron's
odd/even reliability, and off-diagonal structure is what replicates across trial halves.
The matrix is symmetrised, S = (C + C^T) / 2, and ordered by the Rastermap fit on odd
trials (isort, as in Fig. 4a, b).

Full matrix (3.0e9 entries) is computed blockwise on the GPU and reduced to a
B x B grid of BIN x BIN neuron blocks: mean (exact, also = Gram of block-mean unit
vectors, used as a check), s.d. and the max |r| within each block. Also stored: the
exact sub-matrix of every BIN-th neuron, the exact matrices of the dense- and
sparse-tiling groups at full resolution, and the off-diagonal histogram.
Output: results/xcorr.npz
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
(HERE / "results").mkdir(exist_ok=True)
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "sequence_analysis"))
import latency_tiling_groups  # noqa: E402

DEV = torch.device("cuda")
BIN = 27  # 54,569 / 27 -> 2,022 blocks


def unit_rows(X):
    X = X.astype(np.float64)
    X = X - X.mean(1, keepdims=True)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    ok = n[:, 0] > 0
    X[ok] /= n[ok]
    return X.astype(np.float32), ok


def main():
    t0 = time.time()
    f = np.load(Path(os.environ.get("DMN_DATA", Path.home() / "dmn")) / "concat_cvTrue.npy", allow_pickle=True).flat[0]
    order = np.asarray(f["isort"])
    labels = np.asarray(f["rm_labels"])
    Zo, oko = unit_rows(np.asarray(f["X_odd"])[order])
    Ze, oke = unit_rows(np.asarray(f["X_even"])[order])
    L = labels[order]
    N = len(order)
    print(f"N={N}, zero-variance odd {np.sum(~oko)}, even {np.sum(~oke)}", flush=True)

    nb = int(np.ceil(N / BIN))
    o = torch.as_tensor(Zo, device=DEV)
    e = torch.as_tensor(Ze, device=DEV)
    blk = torch.as_tensor(np.arange(N) // BIN, device=DEV)
    cnt = torch.bincount(blk, minlength=nb).double()
    S1 = torch.zeros(nb, nb, dtype=torch.float64, device=DEV)
    S2 = torch.zeros_like(S1)
    MX = torch.zeros(nb, nb, dtype=torch.float32, device=DEV)
    hist_edges = torch.linspace(-1, 1, 401, device=DEV)
    H = torch.zeros(400, dtype=torch.float64, device=DEV)
    diag = np.empty(N, np.float32)
    CH = BIN * 40  # row chunk (whole blocks)
    for s in range(0, N, CH):
        t_ = min(N, s + CH)
        C = o[s:t_] @ e.T                                    # odd rows x even cols
        Ct = (e[s:t_] @ o.T)                                 # transpose part
        Sm = 0.5 * (C + Ct)                                  # symmetrised rows s:t_
        r = torch.arange(t_ - s, device=DEV)
        diag[s:t_] = Sm[r, r + s].cpu().numpy()
        rb = blk[s:t_]
        # reduce over columns into blocks: (rows x nb) via index_add on transposed
        colsum = torch.zeros(t_ - s, nb, dtype=torch.float64, device=DEV).index_add_(
            1, blk, Sm.double())
        colsq = torch.zeros_like(colsum).index_add_(1, blk, Sm.double() ** 2)
        S1.index_add_(0, rb, colsum)
        S2.index_add_(0, rb, colsq)
        # block max |r|: per column block via scatter_reduce
        A = Sm.abs()
        cm = torch.zeros(t_ - s, nb, device=DEV).scatter_reduce_(
            1, blk.expand(t_ - s, -1), A, reduce="amax", include_self=True)
        MX.scatter_reduce_(0, rb[:, None].expand(-1, nb), cm, reduce="amax", include_self=True)
        Sm[r, r + s] = 2.0  # exclude diagonal from the histogram
        H += torch.histc(Sm[Sm <= 1].float(), bins=400, min=-1, max=1).double()
        del C, Ct, Sm, A, colsum, colsq, cm
        print(f"  rows {t_}/{N}  {time.time() - t0:.0f}s", flush=True)
    cc = torch.outer(cnt, cnt)
    mean = (S1 / cc).cpu().numpy()
    sd = torch.sqrt((S2 / cc - (S1 / cc) ** 2).clamp_min(0)).cpu().numpy()
    MX = MX.cpu().numpy()

    # check: block mean == Gram of block-mean vectors (symmetrised)
    mo = torch.zeros(nb, o.shape[1], dtype=torch.float64, device=DEV).index_add_(0, blk, o.double())
    me = torch.zeros_like(mo).index_add_(0, blk, e.double())
    mo, me = mo / cnt[:, None], me / cnt[:, None]
    G = 0.5 * (mo @ me.T + me @ mo.T)
    err = float((G.cpu().numpy() - mean).__abs__().max())
    print("max |block mean - Gram of block means| =", err)
    assert err < 1e-5

    # exact sub-matrix (every BIN-th neuron) and group matrices at full resolution
    sub = np.arange(0, N, BIN)
    def exact(idx):
        i = torch.as_tensor(idx, device=DEV)
        M = o[i] @ e[i].T
        return (0.5 * (M + M.T)).cpu().numpy()
    gi = latency_tiling_groups.group_indices(f)      # stack indices, in a/b row order
    pos = np.empty(N, int); pos[order] = np.arange(N)
    groups = {g: np.sort(pos[idx]) for g, idx in gi.items()}  # row positions
    # validate against numpy on a random subset
    rng = np.random.default_rng(0)
    q = np.sort(rng.choice(N, 300, replace=False))
    ref = np.array([[np.corrcoef(np.asarray(f["X_odd"])[order[a]], np.asarray(f["X_even"])[order[b]])[0, 1]
                     for b in q[:40]] for a in q[:40]])
    M = exact(q[:40])
    ref_s = 0.5 * (ref + ref.T)
    okm = np.isfinite(ref_s)
    verr = float(np.abs(M[okm] - ref_s[okm]).max())
    print("validation vs numpy corrcoef: max err", verr)
    assert verr < 1e-4

    np.savez_compressed(
        HERE / "results" / "xcorr.npz", mean=mean.astype(np.float32), sd=sd.astype(np.float32),
        maxabs=MX, bin=BIN, N=N, labels=L, diag=diag, sub_idx=sub,
        sub=exact(sub).astype(np.float32),
        dense_rows=groups["dense"], sparse_rows=groups["sparse"],
        dense=exact(groups["dense"]).astype(np.float32),
        hist=H.cpu().numpy(), hist_edges=hist_edges.cpu().numpy(),
        fr=np.asarray(f["fr"])[order])
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
