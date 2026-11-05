#!/usr/bin/env python3
"""ePAIRS test of random mixing on RRR task-variable selectivity (Posani et al. features).

Run (iblenv):  python rrr_epairs.py [--rerun] [--n-null 10]

Features are the ones of SI Fig. S2 (lorenzo/manuscript_comparison_figure): the local
rank-5, lambda-75 reduced-rank regression fit of single-trial activity on 8 task
variables (block, stimulus side, contrast, choice, outcome, wheel, whisker motion,
lick; -0.2..0.8 s around stimulus onset, 100 bins). Per neuron and variable the
coefficient time course is U[n, v, :] @ V; the feature is its absolute value summed
over time (as in S2) or, as a check, its signed sum. Neurons: three-fold cross-validated
R^2 minus the trial-average null R^2 > 0.015 (as in S2), all regions.

Same test as selectivity_test.py / extra_nulls.py: mean angle to the 5 nearest
neighbours, neighbours from other sessions only (session labels permuted for nulls);
nulls Gaussian, PC-marginal, Gaussian copula, ICA-marginal (all covariance-matched);
positive control: 25-cluster Gaussian mixture. Features are z-scored per variable
before the test (the 8 variables have different units).
Results -> cache/rrr_epairs.json.
"""

import argparse
import json
import os
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

import selectivity_test as ST
from extra_nulls import null_copula, null_icamarg

LORENZO = Path(os.environ.get("DMN_DATA", Path.home() / "dmn")) / "lorenzo"  # RRR fit (not in the repository)
AUDIT = LORENZO / "sillouette_score" / "reproduction_audit"
SESS = LORENZO / "brainwide-RRR-encoding-model" / "trained_RRR_model_paper178" / "session_features"
VARS = ["block", "stimulus side", "contrast", "choice", "outcome", "wheel", "whisker motion", "lick"]
DELTA = 0.015


def load_rrr():
    ck = torch.load(AUDIT / "RRRGD_rank5_l275_full.pt", map_location="cpu",
                    weights_only=False, mmap=True)["RRRGD_model"]
    state = ck["model"]
    cv = np.load(AUDIT / "rank5_l275_cv_r2.npz")
    V = state["V"].double()                       # (rank, time)
    A, S, sess, acr, n_all = [], [], [], [], 0
    for i, eid in enumerate(ck["eids"]):
        with np.load(SESS / f"{eid}.npz") as old:
            keep = (cv[eid] - old["null_r2"]) > DELTA
            a = old["acronym"].astype(str)
        U = state[eid + "_U"].double()             # (neurons, variables, rank)
        n_all += len(U)
        beta = torch.einsum("nvr,rt->nvt", U, V)   # coefficient time courses
        A.append(beta.abs().sum(2).numpy()[keep])
        S.append(beta.sum(2).numpy()[keep])
        sess.append(np.full(keep.sum(), i))
        acr.append(a[keep])
    assert A[0].shape[1] == len(VARS)
    return dict(abs=np.concatenate(A), signed=np.concatenate(S), sess=np.concatenate(sess),
                acronym=np.concatenate(acr), n_all=n_all, n_sessions=len(ck["eids"]))


def main(rerun=False, n_null=10):
    fout = ST.CACHE / "rrr_epairs.json"
    if fout.exists() and not rerun:
        print("cached:", fout)
        return
    t0 = time.time()
    D = load_rrr()
    grp = D["sess"].astype(np.int64)
    R = dict(n_neurons=int(len(grp)), n_all=int(D["n_all"]), n_sessions=D["n_sessions"],
             n_regions=int(len(np.unique(D["acronym"]))), delta_r2=DELTA, variables=VARS,
             n_null=n_null)
    print(R, flush=True)
    rng = np.random.default_rng(11)
    ds = OrderedDict()
    for kind in ("abs", "signed"):
        Y = D[kind]
        Y = (Y - Y.mean(0)) / Y.std(0)
        ds[f"RRR{kind}"] = Y
        ds[f"RRR{kind}mix25"] = ST.mixture_control(Y, 25, rng)
    nulls = OrderedDict(gauss=ST.null_gauss, pcmarg=ST.null_pcmarg,
                        copula=null_copula, icamarg=null_icamarg)
    ep = {}
    for name, Y in ds.items():
        g = None if "mix" in name else grp
        for sub in ("all", "top20"):
            if "mix" in name and sub == "top20":
                continue
            key = f"{name}|{'xsession' if g is not None else 'all'}|{sub}"
            res = ST.epairs(Y, g, nulls, n_null, seed=7,
                            subset=ST.top_norm if sub == "top20" else None)
            ep[key] = ST.epairs_summary(res)
            print(key, {nn: round(ep[key][nn][5]["auc"], 3) for nn in nulls},
                  f"{time.time() - t0:.0f}s", flush=True)
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
