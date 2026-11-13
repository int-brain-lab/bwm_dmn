#!/usr/bin/env python3
"""ePAIRS test (panel e of the SI response-correlation figure) on the full feature
vectors: all 1872 time bins, no dimensionality reduction.

Run (iblenv, GPU):  python full_space_epairs.py [--rerun] [--n-null 10]

Same test as selectivity_test.py / extra_nulls.py (mean angle to the 5 nearest
neighbours, neighbours from other probe insertions only, AUC on 20,000 subsampled
neurons, synthetic angles pooled over 3 draws), against the same four
covariance-matched nulls and a 25-cluster mixture positive control fitted in the
full space. Because each neuron's feature vector is z-scored over its bins, the
cosine between two neurons' full feature vectors is their Pearson correlation.

The numpy/sklearn nulls do not scale to 1872 dimensions (the covariance is
singular because every row is z-scored, so Cholesky fails; sklearn FastICA with
1872 components takes hours), so the nulls are re-implemented in torch:
  * covariance samplers use the eigendecomposition (handles the null space);
  * permutations, the copula's ranks and sorts run on the GPU;
  * FastICA: symmetric (parallel) FastICA with the logcosh contrast, as in
    sklearn, on PCA-whitened data restricted to the non-null eigenvalues.
Validation: the torch nulls are run on the top 10 PCs and must reproduce the
cached AUCs of selectivity_test.py / extra_nulls.py (results["validation_PC10"]);
torch FastICA is compared with sklearn FastICA on the top 10 PCs.
Results -> cache/full_space_epairs.json.
"""

import argparse
import json
import time
from collections import OrderedDict

import numpy as np
import torch

import selectivity_test as ST

DEV = ST.DEV
F64 = torch.float64


def _cov_sqrt(Yc):
    """Symmetric square root factor L (d x d) with L L^T = cov(Yc), via eigh."""
    cov = Yc.T @ Yc / (Yc.shape[0] - 1)
    ev, evec = torch.linalg.eigh(cov)
    return evec * ev.clamp_min(0).sqrt()[None], ev, evec


def null_gauss(Y, gen):
    mu = Y.mean(0, keepdim=True)
    L, _, _ = _cov_sqrt(Y - mu)
    Z = torch.randn(Y.shape, dtype=F64, device=DEV, generator=gen)
    return mu + Z @ L.T


def _permute_cols(P, gen):
    idx = torch.rand(P.shape, device=DEV, generator=gen).argsort(0)
    return P.gather(0, idx)


def null_pcmarg(Y, gen):
    mu = Y.mean(0, keepdim=True)
    Yc = Y - mu
    _, _, evec = _cov_sqrt(Yc)
    P = _permute_cols(Yc @ evec, gen)
    return mu + P @ evec.T


_COP = {}
CHUNK = 128  # columns per chunk for ranks/sorts (GPU memory)


def null_copula(Y, gen):
    """Gaussian copula, computed in column chunks to fit 1872 dimensions on the GPU."""
    N, d = Y.shape
    key = (Y.data_ptr(), tuple(Y.shape))
    if key not in _COP:
        Z = torch.empty((N, d), dtype=torch.float32, device=DEV)
        srt = torch.empty((N, d), dtype=torch.float32, device=DEV)
        for c in range(0, d, CHUNK):
            y = Y[:, c:c + CHUNK]
            rk = y.argsort(0).argsort(0).to(F64)  # continuous data: ties negligible
            z = torch.special.ndtri((rk + 0.5) / N)
            Z[:, c:c + CHUNK] = ((z - z.mean(0)) / z.std(0)).to(torch.float32)
            srt[:, c:c + CHUNK] = y.sort(0).values.to(torch.float32)
            del rk, z
        R = (Z.T.to(F64) @ Z.to(F64)) / (N - 1) if d <= 64 else _gram64(Z) / (N - 1)
        del Z
        ev, evec = torch.linalg.eigh(R)
        _COP[key] = (srt, evec * ev.clamp_min(0).sqrt()[None])
        torch.cuda.empty_cache()
    srt, L = _COP[key]
    G = (torch.randn((N, d), dtype=F64, device=DEV, generator=gen) @ L.T).to(torch.float32)
    out = torch.empty((N, d), dtype=torch.float32, device=DEV)
    for c in range(0, d, CHUNK):
        rk = G[:, c:c + CHUNK].argsort(0).argsort(0)
        out[:, c:c + CHUNK] = srt[:, c:c + CHUNK].gather(0, rk)
        del rk
    del G
    return out


def _gram64(Z, rows=8192):
    """Z^T Z in float64, accumulated over row blocks."""
    out = torch.zeros((Z.shape[1], Z.shape[1]), dtype=F64, device=DEV)
    for s in range(0, Z.shape[0], rows):
        z = Z[s:s + rows].to(F64)
        out += z.T @ z
    return out


def fastica_torch(Yc, max_iter=1000, tol=1e-4, seed=0, rel_eig=1e-8):
    """Symmetric FastICA (logcosh, alpha=1) on PCA-whitened data. Returns sources S
    (N x r, unit variance) and mixing A (d x r) with S @ A.T == Yc, plus n_iter."""
    N = Yc.shape[0]
    cov = Yc.T @ Yc / (N - 1)
    ev, evec = torch.linalg.eigh(cov)
    keep = ev > rel_eig * ev.max()
    ev, evec = ev[keep], evec[:, keep]
    K = evec / ev.sqrt()[None]                      # whitening (d x r)
    Xw = (Yc @ K).to(torch.float32)                 # N x r, identity covariance
    r = Xw.shape[1]
    g = torch.Generator(device=DEV).manual_seed(seed)
    W = torch.randn((r, r), dtype=F64, device=DEV, generator=g)

    def sym_decorr(W):
        s, u = torch.linalg.eigh(W @ W.T)
        return (u * s.clamp_min(1e-30).rsqrt()[None]) @ u.T @ W

    W = sym_decorr(W)
    for it in range(max_iter):
        Wf = W.to(torch.float32)
        G = torch.tanh(Xw @ Wf.T)                   # N x r
        gp = (1 - G ** 2).mean(0).to(F64)
        W1 = (G.T @ Xw).to(F64) / N - gp[:, None] * W
        W1 = sym_decorr(W1)
        lim = (torch.abs(torch.abs((W1 * W).sum(1)) - 1)).max().item()
        W = W1
        if lim < tol:
            break
    S = Xw.to(F64) @ W.T                            # unit variance sources
    A = evec * ev.sqrt()[None] @ W.T                # d x r: Yc = S @ A.T
    return S, A, it + 1, lim


_ICA = {}


def null_icamarg(Y, gen):
    key = (Y.data_ptr(), tuple(Y.shape))
    if key not in _ICA:
        mu = Y.mean(0, keepdim=True)
        S, A, nit, lim = fastica_torch(Y - mu)
        err = ((S @ A.T + mu - Y).abs().max() / Y.abs().max()).item()
        _ICA[key] = (mu, S, A)
        print(f"  FastICA: {S.shape[1]} sources, {nit} iterations, conv {lim:.1e}, "
              f"rel. reconstruction error {err:.1e}", flush=True)
        ICA_INFO[key] = dict(n_sources=int(S.shape[1]), n_iter=nit, conv=lim, recon_err=err)
    mu, S, A = _ICA[key]
    return mu + _permute_cols(S, gen) @ A.T


ICA_INFO = {}
NULLS = OrderedDict(gauss=null_gauss, pcmarg=null_pcmarg, copula=null_copula,
                    icamarg=null_icamarg)


def mixture_control(Y, k, gen, seed=0):
    """As ST.mixture_control (k-means means + pooled within-cluster covariance),
    sampled with the eigendecomposition so a singular covariance is fine."""
    _, lab = ST.kmeans_torch(Y.to(torch.float32).cpu().numpy(), k, seed=seed)
    lab = torch.as_tensor(lab, device=DEV)
    M = torch.zeros((k, Y.shape[1]), dtype=F64, device=DEV).index_add_(0, lab, Y)
    M = M / torch.bincount(lab, minlength=k).to(F64)[:, None]
    L, _, _ = _cov_sqrt(Y - M[lab])
    perm = lab[torch.randperm(len(lab), device=DEV, generator=gen)]
    return M[perm] + torch.randn(Y.shape, dtype=F64, device=DEV, generator=gen) @ L.T


def epairs(Y, groups, n_null, seed=7, nulls=NULLS):
    """Same procedure as ST.epairs, nulls generated on the GPU."""
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=DEV).manual_seed(seed)
    real = ST.knn_angles(Y.to(torch.float32).cpu().numpy(), groups)
    res = dict(real=real, nulls={})
    for nn, fn in nulls.items():
        meds = {k: [] for k in ST.KNN_KS}
        pooled = {k: [] for k in ST.KNN_KS}
        for b in range(n_null):
            Yn = fn(Y, gen).to(torch.float32).cpu().numpy()
            gn = None if groups is None else groups[rng.permutation(len(Yn))]
            an = ST.knn_angles(Yn, gn)
            del Yn
            for k in ST.KNN_KS:
                meds[k].append(float(np.median(an[k])))
                if b < 3:
                    pooled[k].append(an[k])
        res["nulls"][nn] = dict(medians=meds,
                                pooled={k: np.concatenate(v) for k, v in pooled.items()})
        torch.cuda.empty_cache()
    return ST.epairs_summary(res)


def session_codes(pid):
    """Session (eid) code per neuron from the pipeline's insertion table."""
    import csv
    from pathlib import Path
    tab = Path(__file__).resolve().parents[3] / "pipeline" / "insertions.csv"
    p2e = {r["pid"]: r["eid"] for r in csv.DictReader(open(tab))}
    eid = np.array([p2e[str(p)] for p in pid])
    return np.unique(eid, return_inverse=True)[1].astype(np.int64)


def validate_ica(Y10):
    """torch vs sklearn FastICA on the top 10 PCs: best-match |corr| of sources."""
    from sklearn.decomposition import FastICA
    Yc = Y10 - Y10.mean(0)
    Ss = FastICA(whiten="unit-variance", max_iter=2000, tol=1e-4,
                 random_state=0).fit_transform(Yc)
    St, _, nit, _ = fastica_torch(torch.as_tensor(Yc, dtype=F64, device=DEV))
    C = np.abs(np.corrcoef(Ss.T, St.cpu().numpy().T)[:10, 10:])
    from scipy.optimize import linear_sum_assignment
    r, c = linear_sum_assignment(-C)
    return dict(min_matched_abs_corr=float(C[r, c].min()),
                mean_matched_abs_corr=float(C[r, c].mean()), torch_iter=nit)


def main(rerun=False, n_null=10):
    fout = ST.CACHE / "full_space_epairs.json"
    if fout.exists() and not rerun:
        print("cached:", fout)
        return
    t0 = time.time()
    D = ST.load_data()
    X, grp = D["X"], D["pid_codes"]
    Xt = torch.as_tensor(X, dtype=F64, device=DEV)
    del X
    ev = torch.linalg.eigvalsh(torch.cov(Xt.T)).flip(0).cpu().numpy()
    R = dict(n_neurons=int(Xt.shape[0]), n_bins=int(Xt.shape[1]), n_null=n_null,
             rank=int((ev > 1e-8 * ev[0]).sum()),
             var_frac_top10=float(ev[:10].sum() / ev.sum()))
    rn = Xt.norm(dim=1)
    R["row_norm_cv"] = float((rn.std() / rn.mean()).item())  # z-scored rows: ~0
    print(R, flush=True)

    # --- validation: torch nulls on the top 10 PCs vs the cached numpy/sklearn AUCs
    pca = ST.pca_torch(Xt.to(torch.float32).cpu().numpy(), 10)
    Y10 = pca["scores"]
    R["validation_ica_PC10"] = validate_ica(Y10)
    print("ICA validation", R["validation_ica_PC10"], flush=True)
    old = json.loads((ST.CACHE / "results.json").read_text())["epairs"]["PC10|xprobe|all"]
    old_x = json.loads((ST.CACHE / "extra_nulls.json").read_text())["epairs"]["PC10|xprobe|all"]
    ref = {nn: (old if nn in old else old_x)[nn]["5"]["auc"] for nn in NULLS}
    val = epairs(torch.as_tensor(Y10, dtype=F64, device=DEV), grp, n_null)
    R["validation_PC10"] = {nn: dict(torch=val[nn][5]["auc"], cached=ref[nn]) for nn in NULLS}
    _COP.clear(); _ICA.clear()
    print("PC10 validation", R["validation_PC10"], f"{time.time() - t0:.0f}s", flush=True)

    # --- full space and its mixture control
    ep = {}
    ep["ALL|xprobe|all"] = epairs(Xt, grp, n_null)
    print("ALL", {nn: round(ep["ALL|xprobe|all"][nn][5]["auc"], 3) for nn in NULLS},
          f"{time.time() - t0:.0f}s", flush=True)
    # control: neighbours from other sessions only (two probes of one session share trials)
    ses = session_codes(D["pid"])
    R["n_sessions"] = int(len(np.unique(ses)))
    ep["ALL|xsession|all"] = epairs(Xt, ses, n_null)
    print("ALL xsession", {nn: round(ep["ALL|xsession|all"][nn][5]["auc"], 3) for nn in NULLS},
          f"{time.time() - t0:.0f}s", flush=True)
    _COP.clear(); _ICA.clear()
    ep["PC10|xsession|all"] = epairs(torch.as_tensor(Y10, dtype=F64, device=DEV), ses, n_null)
    print("PC10 xsession", {nn: round(ep["PC10|xsession|all"][nn][5]["auc"], 3) for nn in NULLS},
          f"{time.time() - t0:.0f}s", flush=True)
    _COP.clear(); _ICA.clear()
    torch.cuda.empty_cache()
    gen = torch.Generator(device=DEV).manual_seed(11)
    Xm = mixture_control(Xt, 25, gen)
    del Xt
    torch.cuda.empty_cache()
    ep["ALLmix25|all|all"] = epairs(Xm, None, n_null)
    print("ALLmix25", {nn: round(ep["ALLmix25|all|all"][nn][5]["auc"], 3) for nn in NULLS},
          f"{time.time() - t0:.0f}s", flush=True)
    R["epairs"] = ep
    R["ica"] = {str(k): v for k, v in ICA_INFO.items()}
    R["runtime_s"] = time.time() - t0
    fout.write_text(json.dumps(R, indent=1))
    print("done", fout)


def save_angles(n_draw=3):
    """Angle distributions (5 most similar neurons) for panel c in all bins: real
    (other insertions only), Gaussian and PC-marginal nulls (n_draw draws pooled,
    as for the AUCs) and the 25-cluster mixture fitted in the same space (same seeds
    as main()). -> cache/full_space_angles.npz"""
    D = ST.load_data()
    grp = D["pid_codes"]
    Xt = torch.as_tensor(D["X"], dtype=F64, device=DEV)
    del D
    out = {"real": ST.knn_angles(Xt.to(torch.float32).cpu().numpy(), grp)[5]}
    rng = np.random.default_rng(7)
    gen = torch.Generator(device=DEV).manual_seed(7)
    for nn in ("gauss", "pcmarg"):
        a = []
        for b in range(n_draw):
            Yn = NULLS[nn](Xt, gen).to(torch.float32).cpu().numpy()
            a.append(ST.knn_angles(Yn, grp[rng.permutation(len(Yn))])[5])
        out[nn] = np.concatenate(a)
        print(nn, f"median {np.median(out[nn]):.1f}", flush=True)
    gen = torch.Generator(device=DEV).manual_seed(11)
    Xm = mixture_control(Xt, 25, gen)
    out["mix25"] = ST.knn_angles(Xm.to(torch.float32).cpu().numpy(), None)[5]
    print({k: round(float(np.median(v)), 1) for k, v in out.items()})
    np.savez_compressed(ST.CACHE / "full_space_angles.npz", **out)
    print("saved", ST.CACHE / "full_space_angles.npz")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--n-null", type=int, default=10)
    ap.add_argument("--save-angles", action="store_true",
                    help="only save the panel c angle distributions")
    a = ap.parse_args()
    if a.save_angles:
        save_angles()
    else:
        main(a.rerun, a.n_null)
