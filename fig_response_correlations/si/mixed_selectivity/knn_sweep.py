#!/usr/bin/env python3
"""Neighbourhood-size sweep for the ePAIRS test of Fig. 4c,e (2026-11-17).

Same test as full_space_epairs.py (neighbours from other probe insertions, AUC on
20,000 subsampled neurons, synthetic angles pooled over 3 draws), for k = 1, 5, 20,
100 and 500 most similar neurons, in the full feature vectors and in their top 10
principal components, against the four covariance-matched nulls, plus the
25-cluster mixture control fitted in each space. k = 5 must reproduce the cached
AUCs of full_space_epairs.py / selectivity_test.py.

Run (iblenv, GPU): python knn_sweep.py  -> cache/knn_sweep.json
"""

import json
import time

import numpy as np
import torch

import full_space_epairs as FS
import selectivity_test as ST

KS = (1, 5, 20, 100, 500)
N_DRAW = 3


def sweep(Y, groups, seed=7):
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=FS.DEV).manual_seed(seed)
    real = ST.knn_angles(Y.to(torch.float32).cpu().numpy(), groups, ks=KS)
    out = {}
    for nn, fn in FS.NULLS.items():
        pooled = {k: [] for k in KS}
        for _ in range(N_DRAW):
            Yn = fn(Y, gen).to(torch.float32).cpu().numpy()
            gn = None if groups is None else groups[rng.permutation(len(Yn))]
            an = ST.knn_angles(Yn, gn, ks=KS)
            for k in KS:
                pooled[k].append(an[k])
        out[nn] = {k: float(ST.auc_less(real[k], np.concatenate(pooled[k]))) for k in KS}
        torch.cuda.empty_cache()
    FS._COP.clear(); FS._ICA.clear()
    return out


def main():
    t0 = time.time()
    D = ST.load_data()
    grp = D["pid_codes"]
    Xt = torch.as_tensor(D["X"], dtype=FS.F64, device=FS.DEV)
    Y10 = torch.as_tensor(ST.pca_torch(Xt.to(torch.float32).cpu().numpy(), 10)["scores"],
                          dtype=FS.F64, device=FS.DEV)
    R = dict(ks=list(KS), n_draw=N_DRAW)
    R["ALL"] = sweep(Xt, grp); print("ALL", R["ALL"], f"{time.time() - t0:.0f}s", flush=True)
    R["PC10"] = sweep(Y10, grp); print("PC10", R["PC10"], f"{time.time() - t0:.0f}s", flush=True)
    gen = torch.Generator(device=FS.DEV).manual_seed(11)
    Xm = FS.mixture_control(Xt, 25, gen)
    del Xt
    torch.cuda.empty_cache()
    R["ALLmix25"] = sweep(Xm, None); print("ALLmix25", R["ALLmix25"], f"{time.time() - t0:.0f}s", flush=True)
    del Xm
    Ym = FS.mixture_control(Y10, 25, gen)
    R["PC10mix25"] = sweep(Ym, None); print("PC10mix25", R["PC10mix25"], f"{time.time() - t0:.0f}s", flush=True)
    R["runtime_s"] = time.time() - t0
    (ST.CACHE / "knn_sweep.json").write_text(json.dumps(R, indent=1))
    print("done")


if __name__ == "__main__":
    main()
