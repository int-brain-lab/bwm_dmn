#!/usr/bin/env python3
"""Window and example neurons for Fig. 4b (2026-11-17, replaces pick_examples.py).

Scans the cross-validated correlation matrix (as compute_xcorr.py: S = (C + C^T)/2,
C[i, j] = corr(odd_i, even_j), k-means order of Fig. 4a, kmeans_order.py) for a W x W block (rows and
columns on a STRIDE grid, on or off the diagonal) that contains both strongly positive
and strongly negative pairs: score = |TL - TR - BL + BR| / 4 of the 2 x 2 quadrant means (a checkerboard:
row halves and column halves with opposite correlation), among blocks with at least
5% of pairs r > T and 5% r < -T. The block is
stored at single-pair resolution. Examples: 5 reliable rows (left) and 5 reliable columns
(top) chosen greedily so that their 5 x 5 entries span strong positive, strong negative
and near-zero correlations: start with the most positive and the most negative pair,
then add neurons that are near-zero with the chosen ones, then fill by spread in position.
Output: results/window.npz
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

W, STRIDE, T, R_MIN, NEX, MIN_SEP = 500, 250, 0.3, 0.5, 5, 60
DEV = torch.device("cuda")


def main():
    f = np.load(Path(os.environ.get("DMN_DATA", Path.home() / "dmn")) / "concat_cvTrue.npy",
                allow_pickle=True).flat[0]
    order, _ = kmeans_order.get_order(f)
    Zo, _ = unit_rows(np.asarray(f["X_odd"])[order])
    Ze, _ = unit_rows(np.asarray(f["X_even"])[order])
    N = len(order)
    o, e = torch.as_tensor(Zo, device=DEV), torch.as_tensor(Ze, device=DEV)
    starts = np.arange(0, N - W + 1, STRIDE)
    cs = torch.as_tensor(starts, device=DEV)
    best = (-1, None)
    for r0 in starts:
        S = 0.5 * (o[r0:r0 + W] @ e.T + e[r0:r0 + W] @ o.T)          # W x N
        z = torch.zeros(1, device=DEV)
        P = torch.cumsum(torch.cat([z, (S > T).float().sum(0)]), 0)
        Q = torch.cumsum(torch.cat([z, (S < -T).float().sum(0)]), 0)
        fp, fn = (P[cs + W] - P[cs]) / W ** 2, (Q[cs + W] - Q[cs]) / W ** 2
        # checkerboard: interaction of the 2 x 2 quadrant means (row halves x column halves)
        h = W // 2
        Ct = torch.cumsum(torch.cat([z, S[:h].sum(0)]), 0); Cb = torch.cumsum(torch.cat([z, S[h:].sum(0)]), 0)
        tl, tr = (Ct[cs + h] - Ct[cs]) / h ** 2, (Ct[cs + W] - Ct[cs + h]) / h ** 2
        bl, br = (Cb[cs + h] - Cb[cs]) / h ** 2, (Cb[cs + W] - Cb[cs + h]) / h ** 2
        inter = (tl - tr - bl + br).abs() / 4
        sd_r, sd_c = inter, (tl - tr - bl + br) / 4   # stored for the printout
        sc = torch.where((fp >= 0.05) & (fn >= 0.05), inter, torch.zeros_like(fp))
        k = int(sc.argmax())
        if float(sc[k]) > best[0]:
            best = (float(sc[k]), (int(r0), int(starts[k]), float(fp[k]), float(fn[k]),
                                   float(sd_r[k]), float(sd_c[k])))
    sc, (r0, c0, fp, fn, sdr, sdc) = best
    if sc <= 0:
        # k-means order (2026-11-20): neighbouring clusters are similar, so no grid block passes.
        # Use the row cluster whose mean correlation flips sign most strongly across the
        # boundary of two adjacent column clusters (results/xcorr_clusters.npz); rows centred
        # on that cluster, columns centred on the boundary.
        CM = np.load(HERE / "results" / "xcorr_clusters.npz")
        M, n, cut = CM["mean"], CM["counts"], CM["cut"].astype(int)
        cand = []
        for r in range(len(M)):
            for q in range(len(M) - 1):
                a_, b_ = M[r, q], M[r, q + 1]
                s_ = min(a_, -b_) if a_ > 0 else min(-a_, b_)
                if n[q] >= W // 2 and n[q + 1] >= W // 2:
                    cand.append((s_, r, q))
        s_, rc, q = max(cand)
        r0 = int(np.clip((cut[rc] + cut[rc + 1]) // 2 - W // 2, 0, N - W))
        c0 = int(cut[q + 1] - W // 2)
        Sb = 0.5 * (o[r0:r0 + W] @ e[c0:c0 + W].T + e[r0:r0 + W] @ o[c0:c0 + W].T)
        fp, fn = float((Sb > T).float().mean()), float((Sb < -T).float().mean())
        sdc = float(M[rc, q] - M[rc, q + 1])
        print(f"cluster-guided window: row cluster {rc}, columns across clusters {q}|{q + 1} "
              f"(mean r {M[rc, q]:+.2f} / {M[rc, q + 1]:+.2f})")
    print(f"best block rows {r0}-{r0 + W - 1}, cols {c0}-{c0 + W - 1}: frac r>{T} {fp:.3f}, r<-{T} {fn:.3f}, "
          f"checkerboard interaction {sdc:+.3f}")

    ri, ci = np.arange(r0, r0 + W), np.arange(c0, c0 + W)
    B = 0.5 * (o[r0:r0 + W] @ e[c0:c0 + W].T + e[r0:r0 + W] @ o[c0:c0 + W].T)
    B = B.cpu().numpy()
    rel = (o * e).sum(1).cpu().numpy()  # odd/even reliability per row
    okr, okc = np.flatnonzero(rel[ri] >= R_MIN), np.flatnonzero(rel[ci] >= R_MIN)
    Bs = B[np.ix_(okr, okc)]
    rows, cols = [], []

    def far(lst, x):
        return all(abs(x - y) >= MIN_SEP for y in lst)

    for pick in (np.argmax, np.argmin):                  # strongest +, strongest -
        a, b = np.unravel_index(pick(Bs), Bs.shape)
        if far(rows, okr[a]) and far(cols, okc[b]):
            rows.append(okr[a]); cols.append(okc[b])
    # near-zero pair, then fill with neurons spread over the block
    near_r = np.array([not far(rows, x) for x in okr]); near_c = np.array([not far(cols, x) for x in okc])
    a, b = np.unravel_index(np.argmin(np.abs(Bs) + 10 * (near_r[:, None] | near_c[None, :])), Bs.shape)
    rows.append(okr[a]); cols.append(okc[b])
    for lst, ok in ((rows, okr), (cols, okc)):
        for q in np.linspace(0, W - 1, 12):
            if len(lst) >= NEX:
                break
            cand = ok[np.argmin(np.abs(ok - q))]
            if far(lst, cand):
                lst.append(cand)
    rows, cols = np.sort(rows[:NEX]), np.sort(cols[:NEX])
    E = B[np.ix_(rows, cols)]
    print("example rows", ri[rows], "cols", ci[cols]); print(np.round(E, 2))
    Xe = np.asarray(f["X_even"])
    np.savez(HERE / "results" / "window.npz", r0=r0, c0=c0, W=W, B=B, frac_pos=fp, frac_neg=fn, T=T,
             ex_r=rows, ex_c=cols, ex_rvec=Xe[order[ri[rows]]], ex_cvec=Xe[order[ci[cols]]],
             ex_E=E, ex_rrel=rel[ri[rows]], ex_crel=rel[ci[cols]],
             lens=np.array(list(f["len"].values())))


if __name__ == "__main__":
    main()
