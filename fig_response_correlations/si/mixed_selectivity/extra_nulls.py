#!/usr/bin/env python3
"""Two nulls closer to the data than the Gaussian / PC-marginal ones, for panel e
of the SI mixing figure (ePAIRS AUC).

Run (iblenv):  python extra_nulls.py [--rerun] [--n-null 10]

* copula: Gaussian copula. Every coordinate keeps its exact empirical marginal
  (rank-mapped), and the coordinates keep their full rank-correlation matrix.
  This is the old Fig. 4 null (independent marginals per coefficient) plus the
  data's pairwise correlations.
* icamarg: ICA-marginal. FastICA unmixes the (whitened) data into maximally
  non-Gaussian, uncorrelated sources; each source's scores are permuted
  independently and mixed back. Keeps the covariance and every source's sparse,
  heavy-tailed marginal: neurons that combine independent sparse features at
  random ("random mixed selectivity" with non-Gaussian components). Excess
  clumping against this null needs dependencies between features.

Same datasets, probe exclusion, subsets, k = 5 neighbours and seeds as
selectivity_test.py; results -> cache/extra_nulls.json.
"""

import argparse
import json
import time
from collections import OrderedDict

import numpy as np
from scipy.stats import norm, rankdata
from sklearn.decomposition import FastICA

import selectivity_test as ST


def null_copula(Y, rng):
    N, d = Y.shape
    Z = norm.ppf((rankdata(Y, axis=0) - 0.5) / N)
    R = np.corrcoef(Z.T)
    L = np.linalg.cholesky(R + 1e-10 * np.eye(d))
    G = rng.standard_normal((N, d)) @ L.T
    srt = np.sort(Y, axis=0)
    rk = np.argsort(np.argsort(G, axis=0), axis=0)
    return np.take_along_axis(srt, rk, axis=0)


_ICA = {}


def null_icamarg(Y, rng):
    key = (id(Y), Y.shape)
    if key not in _ICA:
        mu = Y.mean(0)
        ica = FastICA(whiten="unit-variance", max_iter=2000, tol=1e-4, random_state=0)
        S = ica.fit_transform(Y - mu)
        _ICA[key] = (mu, S, ica.mixing_)
        # sanity: reconstruction
        assert np.allclose(S @ ica.mixing_.T, Y - mu, atol=1e-6 * np.abs(Y).max() * 10)
    mu, S, A = _ICA[key]
    P = np.stack([rng.permutation(S[:, j]) for j in range(S.shape[1])], 1)
    return mu + P @ A.T


def main(rerun=False, n_null=10):
    fout = ST.CACHE / "extra_nulls.json"
    if fout.exists() and not rerun:
        print("cached:", fout)
        return
    t0 = time.time()
    D = ST.load_data()
    X, V, lens, grp = D["X"], D["V"], D["lens"], D["pid_codes"]
    pca = ST.pca_torch(X, 100)
    gev, gvec = np.linalg.eigh(V @ V.T)
    W = gvec @ np.diag(gev ** -0.5) @ gvec.T @ V
    CW = X.astype(np.float64) @ W.T
    S = ST.selectivity(X, lens)
    ds = OrderedDict([("PC10", pca["scores"][:, :10]), ("PC20", pca["scores"][:, :20]),
                      ("W100", CW), ("SEL", S)])
    rng_pc = np.random.default_rng(11)  # same draws as selectivity_test.py
    ds["PC10mix25"] = ST.mixture_control(ds["PC10"], 25, rng_pc)
    ds["SELmix25"] = ST.mixture_control(S / S.std(0), 25, rng_pc) * S.std(0)
    del X

    # check: the old (Fig. 4) null vs copula on the coefficient correlations
    rng = np.random.default_rng(0)
    Yc = null_copula(CW, rng)
    off = ~np.eye(CW.shape[1], dtype=bool)
    cr, cn = np.corrcoef(CW.T)[off], np.corrcoef(Yc.T)[off]
    R = dict(n_null=n_null, copula_W100_corr_match=float(np.corrcoef(cr, cn)[0, 1]),
             copula_W100_mean_abs_real=float(np.abs(cr).mean()),
             copula_W100_mean_abs_null=float(np.abs(cn).mean()))
    print(R)

    keys = [("PC10", "xprobe", "all"), ("PC20", "xprobe", "all"), ("W100", "xprobe", "all"),
            ("PC10mix25", "all", "all"), ("SEL", "xprobe", "all"), ("SEL", "xprobe", "top20"),
            ("SELmix25", "all", "all")]
    nulls = OrderedDict(copula=null_copula, icamarg=null_icamarg)
    ep = {}
    for name, excl, sub in keys:
        key = f"{name}|{excl}|{sub}"
        print("ePAIRS", key, f"{time.time() - t0:.0f}s", flush=True)
        res = ST.epairs(ds[name], grp if excl == "xprobe" else None, nulls, n_null, seed=7,
                        subset=ST.top_norm if sub == "top20" else None)
        ep[key] = ST.epairs_summary(res)
        print({nn: round(ep[key][nn][5]["auc"], 3) for nn in nulls}, flush=True)
    R["epairs"] = ep
    R["runtime_s"] = time.time() - t0
    fout.write_text(json.dumps(R, indent=1))
    print("done", fout)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--n-null", type=int, default=10)
    a = ap.parse_args()
    main(a.rerun, a.n_null)
