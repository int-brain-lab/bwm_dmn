#!/usr/bin/env python3
"""Structured vs random mixed selectivity: tests beyond second order.

Run (iblenv):  python selectivity_test.py [--rerun] [--n-null 10]

Data: compute_synthetic.load(nclus=100) (fig_response_correlations) -> X = r['concat_z'] (54,719
neurons x 1872 bins, stack order), the 100-cluster k-means basis V, probe ids.
Nothing outside this folder is written; results go to ./cache.

Part 1 (orthonormal bases)
  * C_V = X V^T (current Fig. 4), C_W = X W^T with W = (V V^T)^(-1/2) V
    (Loewdin / symmetric orthonormalisation of the k-means means), and PCA
    scores of X. Coefficient correlations vs basis Gram / white-noise
    projection, vs the marginal-independent null and a covariance-matched
    Gaussian null.
  * Beyond second order: ePAIRS-style k-nearest-neighbour angle test and
    k-means inertia (gap statistic) / silhouette in the top 10 and 20 PCs,
    real vs covariance-matched Gaussian and vs a "PC-marginal" null
    (each PC score permuted independently: same covariance AND same 1-D
    marginals along the PCs, higher-order dependencies destroyed).
Part 2 (task-variable selectivity)
  * Contrasts of condition-averaged z-scored responses (window means) ->
    11 selectivity indices per neuron, their correlations, ePAIRS + k-means
    vs the same nulls.
GPU (torch) is used for PCA, kNN angles, k-means and silhouettes; validated
against numpy/sklearn on subsets (validation block in results.json).
"""

import argparse
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RC_DIR = HERE.parents[1]  # ~/dmn/fig_response_correlations
sys.path.insert(0, str(RC_DIR))
CACHE = HERE / "cache"
CACHE.mkdir(exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

KNN_KS = (1, 5, 20)
KM_KS = (2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50)
SIL_N = 10000
TOP_FRAC = 0.2  # strongly selective subset (top 20 % by vector norm)

# --------------------------------------------------------------------------
# data


def load_data():
    import os
    cwd = os.getcwd()
    os.chdir(RC_DIR)
    try:
        from compute_synthetic import load
        r = load(nclus=100)
    finally:
        os.chdir(cwd)
    X = np.asarray(r["concat_z"], dtype=np.float32)
    V = np.asarray(r["V"], dtype=np.float64)
    lens = OrderedDict((k, int(v)) for k, v in r["len"].items())
    pid = np.asarray(r["pid"])
    _, pid_codes = np.unique(pid, return_inverse=True)
    # sanity: V/C as in the Fig. 4 cache
    rows = np.asarray(r["C_rows"], dtype=int)
    sub = rows[:500]
    assert np.allclose(X[sub].astype(np.float64) @ V.T, r["C"][:500], atol=1e-3)
    return dict(X=X, V=V, lens=lens, pid_codes=pid_codes.astype(np.int64), pid=pid,
                beryl=np.asarray(r["Beryl"]), fr=np.asarray(r["fr"], dtype=float))


def segments(lens):
    out, s = OrderedDict(), 0
    for k, n in lens.items():
        out[k] = (s, s + n)
        s += n
    return out


# --------------------------------------------------------------------------
# task-variable selectivity


# name -> (dict window: weight, epoch label, description)
SEL_DEFS = OrderedDict([
    ("block_pre", ({"blockL": 1, "blockR": -1}, "pre-stim",
                   "L-block minus R-block, -0.4..-0.1 s before stim (all valid trials)")),
    ("block_stim", ({"stimLbLcL": .5, "stimRbLcR": .5, "stimLbRcL": -.5, "stimRbRcR": -.5},
                    "stim", "block L-R at stim onset, correct trials, side-balanced")),
    ("block_move", ({"sLbLchoiceL": .5, "sRbLchoiceR": .5, "sLbRchoiceL": -.5, "sRbRchoiceR": -.5},
                    "pre-move", "block L-R before movement, correct trials, side-balanced")),
    ("side_stim", ({"stimLbLcL": .5, "stimLbRcL": .5, "stimRbRcR": -.5, "stimRbLcR": -.5},
                   "stim", "stim L = choice L vs R, correct trials (stimulus and choice yoked)")),
    ("side_move", ({"sLbLchoiceL": .5, "sLbRchoiceL": .5, "sRbRchoiceR": -.5, "sRbLchoiceR": -.5},
                   "pre-move", "stim L = choice L vs R before movement, correct trials (yoked)")),
    ("choice_post", ({"choiceL": 1, "choiceR": -1}, "post-move",
                     "choice L vs R after movement onset, all trials (incl. errors)")),
    ("congr_stim", ({"stimLbLcL": .5, "stimRbRcR": .5, "stimLbRcL": -.5, "stimRbLcR": -.5},
                    "stim", "prior-congruent vs incongruent stimulus (block x side), correct")),
    ("congr_move", ({"sLbLchoiceL": .5, "sRbRchoiceR": .5, "sLbRchoiceL": -.5, "sRbLchoiceR": -.5},
                    "pre-move", "prior-congruent vs incongruent, before movement, correct")),
    ("error_stim", ({"mistake_s": 1, "stimLbLcL": -.25, "stimLbRcL": -.25, "stimRbRcR": -.25,
                     "stimRbLcR": -.25}, "stim", "incorrect minus correct at stim onset")),
    ("error_move", ({"mistake_m": 1, "sLbLchoiceL": -.25, "sLbRchoiceL": -.25,
                     "sRbRchoiceR": -.25, "sRbLchoiceR": -.25}, "pre-move",
                    "incorrect minus correct before movement")),
    ("outcome_fb", ({"fback1": 1, "fback0": -1}, "feedback",
                    "correct (reward) minus incorrect (noise burst) feedback, 0..0.3 s")),
])


def selectivity(X, lens):
    """Window-mean contrasts of the z-scored PETHs (z units of each neuron's
    concatenated PETH; the direction of each neuron's vector is invariant to
    its z-scoring)."""
    seg = segments(lens)
    means = {k: X[:, a:b].mean(1) for k, (a, b) in seg.items()}
    S = np.stack([sum(w * means[k] for k, w in d.items()) for d, _, _ in SEL_DEFS.values()], 1)
    return S.astype(np.float64)


NOISE_LAG = 7  # 7 x 2 ms > 12.5 ms bin: x[t] and x[t+7] use disjoint spikes


def noise_model(X, lens):
    """Approximate sampling-noise covariance of the selectivity indices.

    Per neuron and window, the trial-averaging noise variance of one 12.5 ms
    bin is estimated from lag-7 differences, sigma^2 ~ mean((x[t+7]-x[t])^2)/2
    (an upper bound: slow signal changes add to it), and the window mean has
    ~len/6 independent bins. Contrasts within an epoch share windows, so their
    noise covariance is sum_w a_w b_w sigma_w^2 / n_eff,w. Noise shared across
    epochs through common trials (slow drifts, engagement) is not captured."""
    seg = segments(lens)
    nv = {}
    for k, (a, b) in seg.items():
        x = X[:, a:b].astype(np.float64)
        d = x[:, NOISE_LAG:] - x[:, :-NOISE_LAG]
        nv[k] = (d ** 2).mean(1) / 2 / ((b - a) / 6.0)
    names = list(SEL_DEFS)
    P = len(names)
    Sig = np.zeros((P, P))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            wa, wb = SEL_DEFS[a][0], SEL_DEFS[b][0]
            Sig[i, j] = sum(wa[w] * wb[w] * nv[w].mean() for w in wa if w in wb)
    return Sig


# --------------------------------------------------------------------------
# torch helpers


def t(x, dtype=torch.float32):
    return torch.as_tensor(np.asarray(x), dtype=dtype, device=DEV)


def pca_torch(X, ncomp=100):
    """PCA via the covariance eigendecomposition (float64, GPU)."""
    Xt = t(X, torch.float64)
    mu = Xt.mean(0, keepdim=True)
    Xc = Xt - mu
    cov = Xc.T @ Xc / (Xt.shape[0] - 1)
    ev, evec = torch.linalg.eigh(cov)
    ev, evec = ev.flip(0), evec.flip(1)
    scores = (Xc @ evec[:, :ncomp]).cpu().numpy()
    out = dict(ev=ev.cpu().numpy(), comps=evec[:, :ncomp].T.cpu().numpy(), scores=scores,
               mean=mu.cpu().numpy()[0])
    del Xt, Xc
    torch.cuda.empty_cache()
    return out


def knn_angles(Y, groups=None, ks=KNN_KS, chunk=2048):
    """Mean angle (deg) from each row to its k nearest neighbours in angle
    (largest cosine), for each k. groups: rows with the same code are never
    neighbours (e.g. same probe insertion)."""
    Yt = t(Y)
    Yt = Yt / Yt.norm(dim=1, keepdim=True).clamp_min(1e-12)
    N = Yt.shape[0]
    kmax = max(ks)
    g = t(groups, torch.int64) if groups is not None else None
    out = {k: np.empty(N, np.float32) for k in ks}
    ar = torch.arange(N, device=DEV)
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        sim = Yt[s:e] @ Yt.T
        sim[torch.arange(e - s, device=DEV), ar[s:e]] = -2.0  # self
        if g is not None:
            sim[g[s:e, None] == g[None, :]] = -2.0
        top = sim.topk(kmax, dim=1).values.clamp(-1, 1)
        ang = torch.rad2deg(torch.arccos(top))
        for k in ks:
            out[k][s:e] = ang[:, :k].mean(1).cpu().numpy()
    del Yt
    return out


def kmeans_torch(Y, k, n_init=4, iters=150, seed=0):
    """Lloyd k-means with k-means++ init; returns (inertia, labels) of best init."""
    Yt = t(Y)
    N = Yt.shape[0]
    y2 = (Yt ** 2).sum(1)
    best = (np.inf, None)
    for i in range(n_init):
        gen = torch.Generator(device=DEV).manual_seed(seed * 1000 + i)
        idx = torch.randint(N, (1,), device=DEV, generator=gen)
        C = Yt[idx]
        d2 = ((Yt - C) ** 2).sum(1)
        for _ in range(1, k):
            j = torch.multinomial(d2 / d2.sum(), 1, generator=gen)
            C = torch.cat([C, Yt[j]], 0)
            d2 = torch.minimum(d2, ((Yt - Yt[j]) ** 2).sum(1))
        lab_old = None
        for _ in range(iters):
            D = y2[:, None] - 2 * Yt @ C.T + (C ** 2).sum(1)[None]
            dmin, lab = D.min(1)
            if lab_old is not None and torch.equal(lab, lab_old):
                break
            lab_old = lab
            cnt = torch.bincount(lab, minlength=k).to(Yt.dtype)
            Cn = torch.zeros_like(C).index_add_(0, lab, Yt)
            empty = cnt == 0
            Cn = Cn / cnt.clamp_min(1)[:, None]
            if empty.any():  # re-seed empty clusters at the worst-fit points
                far = dmin.topk(int(empty.sum())).indices
                Cn[empty] = Yt[far]
            C = Cn
        D = y2[:, None] - 2 * Yt @ C.T + (C ** 2).sum(1)[None]
        dmin, lab = D.clamp_min(0).min(1)
        inertia = float(dmin.sum())
        if inertia < best[0]:
            best = (inertia, lab.cpu().numpy())
    return best


def silhouette_torch(Y, labels, k):
    Yt = t(Y)
    lab = t(labels, torch.int64)
    D = torch.cdist(Yt, Yt)
    oh = torch.nn.functional.one_hot(lab, k).to(D.dtype)
    cnt = oh.sum(0)
    S = D @ oh  # sum of distances to each cluster
    own = S.gather(1, lab[:, None])[:, 0]
    n_own = cnt[lab]
    a = own / (n_own - 1).clamp_min(1)
    M = S / cnt.clamp_min(1)[None]
    M[torch.arange(len(lab), device=DEV), lab] = float("inf")
    M[:, cnt == 0] = float("inf")
    b = M.min(1).values
    s = (b - a) / torch.maximum(a, b)
    s[n_own <= 1] = 0
    return float(s.mean())


# --------------------------------------------------------------------------
# nulls


def null_gauss(Y, rng):
    """Gaussian with the same mean and covariance as Y."""
    mu = Y.mean(0)
    L = np.linalg.cholesky(np.cov(Y.T) + 1e-12 * np.eye(Y.shape[1]))
    Z = rng.standard_normal(Y.shape)
    return mu + Z @ L.T


def null_pcmarg(Y, rng):
    """Rotate to the PCs of Y, permute each PC score independently, rotate
    back: preserves the covariance (up to sampling) and each PC's marginal."""
    mu = Y.mean(0)
    Yc = Y - mu
    _, _, Vt = np.linalg.svd(Yc, full_matrices=False)
    P = Yc @ Vt.T
    P = np.stack([rng.permutation(P[:, j]) for j in range(P.shape[1])], 1)
    return mu + P @ Vt


def null_axismarg(Y, rng):
    """Each coordinate permuted independently (the Fig. 4 marginal null)."""
    return np.stack([rng.permutation(Y[:, j]) for j in range(Y.shape[1])], 1)


def null_ellip(Y, rng):
    """Elliptical null: Gaussian directions (covariance-matched) with the real
    Mahalanobis radii (permuted). Same covariance and radial distribution as Y,
    no angular structure beyond the ellipse; controls k-means/gap for heavy
    tails / radial structure (angles are those of the Gaussian null)."""
    mu = Y.mean(0)
    L = np.linalg.cholesky(np.cov(Y.T) + 1e-12 * np.eye(Y.shape[1]))
    rad = np.linalg.norm(np.linalg.solve(L, (Y - mu).T), axis=0)
    Z = rng.standard_normal(Y.shape)
    Z *= (rng.permutation(rad) / np.linalg.norm(Z, axis=1))[:, None]
    return mu + Z @ L.T


def mixture_control(Y, k, rng, seed=0):
    """Positive control: a k-component Gaussian mixture fitted by k-means to Y
    (cluster means + pooled within-cluster covariance); total covariance ~ Y's."""
    _, lab = kmeans_torch(Y, k, seed=seed)
    M = np.stack([Y[lab == c].mean(0) for c in range(k)])
    Rw = Y - M[lab]
    L = np.linalg.cholesky(np.cov(Rw.T) + 1e-12 * np.eye(Y.shape[1]))
    return M[rng.permutation(lab)] + rng.standard_normal(Y.shape) @ L.T


NULLS = OrderedDict(gauss=null_gauss, pcmarg=null_pcmarg, ellip=null_ellip)


def top_norm(Y, frac=TOP_FRAC):
    """Rows with the largest norm (distance from 0 = no selectivity)."""
    n = np.linalg.norm(Y, axis=1)
    return np.argsort(n)[::-1][: int(round(frac * len(n)))]


# --------------------------------------------------------------------------
# summaries


def auc_less(a, b, n=20000, seed=0):
    """P(a < b) + 0.5 P(a == b) on random subsamples (Mann-Whitney AUC)."""
    rng = np.random.default_rng(seed)
    a = rng.choice(a, min(n, len(a)), replace=False)
    b = rng.choice(b, min(n, len(b)), replace=False)
    allv = np.concatenate([a, b])
    from scipy.stats import rankdata
    rk = rankdata(allv)
    ra = rk[: len(a)].sum()
    U = ra - len(a) * (len(a) + 1) / 2
    return 1 - U / (len(a) * len(b))


def epairs(Y, groups, nulls, n_null, seed, subset=None, keep_null=True):
    """kNN angle test of Y vs each null. subset: callable returning row indices
    (applied to real and null alike)."""
    rng = np.random.default_rng(seed)
    sel = subset(Y) if subset else np.arange(len(Y))
    real = knn_angles(Y[sel], None if groups is None else groups[sel])
    res = dict(real={k: v for k, v in real.items()}, nulls={})
    for nn, fn in nulls.items():
        meds = {k: [] for k in KNN_KS}
        pooled = {k: [] for k in KNN_KS}
        for b in range(n_null):
            Yn = fn(Y, rng)
            seln = subset(Yn) if subset else np.arange(len(Yn))
            gn = None if groups is None else groups[sel][rng.permutation(len(seln))]
            an = knn_angles(Yn[seln], gn)
            for k in KNN_KS:
                meds[k].append(float(np.median(an[k])))
                if b < 3 and keep_null:
                    pooled[k].append(an[k])
        res["nulls"][nn] = dict(medians=meds, pooled={k: np.concatenate(v) if v else None
                                                      for k, v in pooled.items()})
    return res


def epairs_summary(res):
    out = {}
    for nn, d in res["nulls"].items():
        out[nn] = {}
        for k in KNN_KS:
            mr = float(np.median(res["real"][k]))
            mn = np.array(d["medians"][k])
            entry = dict(median_real=mr, median_null_mean=float(mn.mean()),
                         median_null_sd=float(mn.std(ddof=1)) if len(mn) > 1 else 0.0,
                         delta_deg=float(mn.mean() - mr),
                         rel_delta=float((mn.mean() - mr) / mn.mean()),
                         z=float((mn.mean() - mr) / max(mn.std(ddof=1), 1e-9)) if len(mn) > 1 else np.nan,
                         p_emp=float((np.sum(mn <= mr) + 1) / (len(mn) + 1)))
            if d["pooled"][k] is not None:
                entry["auc"] = float(auc_less(res["real"][k], d["pooled"][k]))
            out[nn][k] = entry
    return out


def kmeans_curves(Y, nulls, n_null, seed, ks=KM_KS):
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(Y), min(SIL_N, len(Y)), replace=False)
    out = dict(ks=list(ks), real=dict(logW=[], sil=[]))
    for k in ks:
        w, lab = kmeans_torch(Y, k, seed=seed)
        out["real"]["logW"].append(np.log(w))
        out["real"]["sil"].append(silhouette_torch(Y[sub], lab[sub], k))
    for nn, fn in nulls.items():
        L = np.zeros((n_null, len(ks)))
        Sl = np.zeros((n_null, len(ks)))
        for b in range(n_null):
            Yn = fn(Y, rng)
            for i, k in enumerate(ks):
                w, lab = kmeans_torch(Yn, k, seed=seed + b + 1)
                L[b, i] = np.log(w)
                Sl[b, i] = silhouette_torch(Yn[sub], lab[sub], k)
        out[nn] = dict(logW=L, sil=Sl)
        out[nn]["gap"] = L.mean(0) - np.array(out["real"]["logW"])
        out[nn]["gap_sd"] = L.std(0, ddof=1) * np.sqrt(1 + 1 / n_null)
    return out


def offdiag(M):
    return M[~np.eye(len(M), dtype=bool)]


def corr_rows(A):
    return np.corrcoef(A)


# --------------------------------------------------------------------------
# validation (torch vs numpy / sklearn on subsets)


def validate(X, pcs, rng):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import NearestNeighbors
    out = {}
    sub = rng.choice(len(X), 3000, replace=False)
    Xs = X[sub].astype(np.float64)
    ev_np = np.linalg.eigvalsh(np.cov(Xs.T))[::-1][:20]
    ev_t = pca_torch(Xs, 20)["ev"][:20]
    out["pca_eig_maxrel_err"] = float(np.max(np.abs(ev_np - ev_t) / ev_np))
    Y = pcs[sub, :10]
    nn = NearestNeighbors(n_neighbors=6, metric="cosine").fit(Y)
    dist, _ = nn.kneighbors(Y)
    ang_sk = np.degrees(np.arccos(np.clip(1 - dist[:, 1:], -1, 1))).mean(1)
    ang_t = knn_angles(Y, ks=(5,))[5]
    out["knn5_angle_max_abs_err_deg"] = float(np.max(np.abs(ang_sk - ang_t)))
    km = KMeans(8, n_init=4, random_state=0).fit(Y)
    w_t, lab_t = kmeans_torch(Y, 8)
    out["kmeans8_inertia_sklearn"] = float(km.inertia_)
    out["kmeans8_inertia_torch"] = float(w_t)
    out["silhouette_sklearn"] = float(silhouette_score(Y, lab_t))
    out["silhouette_torch"] = silhouette_torch(Y, lab_t, 8)
    return out


# --------------------------------------------------------------------------


def main(rerun=False, n_null=10):
    fres = CACHE / "results.json"
    farr = CACHE / "arrays.npz"
    if fres.exists() and farr.exists() and not rerun:
        print("cached:", fres)
        return
    t0 = time.time()
    D = load_data()
    X, V, lens, grp = D["X"], D["V"], D["lens"], D["pid_codes"]
    N = len(X)
    print(f"N={N}, T={X.shape[1]}, probes={grp.max() + 1}, device={DEV}")
    R = dict(N=int(N), n_probes=int(grp.max() + 1), n_null=n_null)
    A = {}

    # ---------- bases
    pca = pca_torch(X, 100)
    ev = pca["ev"]
    R["pca_var_frac"] = {str(d): float(ev[:d].sum() / ev.sum()) for d in (1, 2, 5, 10, 20, 50, 100)}
    A["pca_ev"] = ev
    G = V @ V.T
    gev, gvec = np.linalg.eigh(G)
    R["V_gram_cond"] = float(gev.max() / gev.min())
    W = gvec @ np.diag(gev ** -0.5) @ gvec.T @ V  # Loewdin orthonormalisation
    assert np.allclose(W @ W.T, np.eye(len(W)), atol=1e-6)
    R["loewdin_row_cos_to_V_median"] = float(np.median(
        np.sum(W * V, 1) / np.linalg.norm(V, axis=1)))
    X64 = X.astype(np.float64)
    CV = X64 @ V.T
    CW = X64 @ W.T
    del X64

    # ---------- part 1a: coefficient correlations
    rng = np.random.default_rng(0)
    corr = {}
    for name, B, C in (("V", V, CV), ("W", W, CW)):
        cC = np.corrcoef(C.T)
        cB = corr_rows(B)                          # Pearson between basis rows
        gram = B @ B.T
        cos = gram / np.sqrt(np.outer(np.diag(gram), np.diag(gram)))  # white-noise proj.
        cM = np.corrcoef(null_axismarg(C, rng).T)  # marginal-independent null
        cG = np.corrcoef(null_gauss(C, rng).T)     # covariance-matched Gaussian
        evC = np.linalg.eigvalsh(np.cov(C.T))[::-1]
        corr[name] = dict(
            r_corrC_vs_corrBasis=float(np.corrcoef(offdiag(cC), offdiag(cB))[0, 1]),
            r_corrC_vs_whitenoise=float(np.corrcoef(offdiag(cC), offdiag(cos))[0, 1]),
            mean_abs_offdiag_real=float(np.abs(offdiag(cC)).mean()),
            mean_abs_offdiag_whitenoise=float(np.abs(offdiag(cos)).mean()),
            mean_abs_offdiag_margnull=float(np.abs(offdiag(cM)).mean()),
            mean_abs_offdiag_gauss=float(np.abs(offdiag(cG)).mean()),
            r_corrC_vs_gauss=float(np.corrcoef(offdiag(cC), offdiag(cG))[0, 1]),
            pc1_var_frac=float(evC[0] / evC.sum()),
        )
        A[f"corrC_{name}"] = cC
        A[f"corrWN_{name}"] = cos
        A[f"corrMarg_{name}"] = cM
    R["coef_corr"] = corr
    print("coef corr", json.dumps(corr, indent=1))

    # ---------- part 2: selectivity
    S = selectivity(X, lens)
    A["S"] = S
    R["sel_names"] = list(SEL_DEFS)
    R["sel_desc"] = {k: v[2] for k, v in SEL_DEFS.items()}
    cS = np.corrcoef(S.T)
    A["corrS"] = cS
    R["sel_corr"] = {f"{a}|{b}": float(cS[i, j]) for i, a in enumerate(SEL_DEFS)
                     for j, b in enumerate(SEL_DEFS) if j > i}
    covS = np.cov(S.T)
    SigN = noise_model(X, lens)
    A["noise_cov"] = SigN
    sd = np.sqrt(np.diag(covS))
    R["sel_noise_frac"] = dict(zip(SEL_DEFS, (np.diag(SigN) / np.diag(covS)).tolist()))
    pn = SigN / np.outer(sd, sd)
    A["corrS_noise_pred"] = pn
    R["sel_noise_pred_corr"] = {f"{a}|{b}": float(pn[i, j]) for i, a in enumerate(SEL_DEFS)
                                for j, b in enumerate(SEL_DEFS) if j > i and abs(pn[i, j]) > 1e-9}
    evS = np.linalg.eigvalsh(covS)[::-1]
    R["sel_pca_var_frac"] = (evS / evS.sum()).tolist()
    R["sel_participation_ratio"] = float(evS.sum() ** 2 / (evS ** 2).sum())
    from scipy.stats import kurtosis
    R["sel_excess_kurtosis"] = dict(zip(SEL_DEFS, kurtosis(S, axis=0).tolist()))
    # yoking expectation for choice_post vs side_move (linear stim a, choice b, error rate e)
    yok = {}
    for e in (0.15, 0.2, 0.25):
        for ratio in (0.5, 1.0, 2.0):  # sd(a)/sd(b)
            sa2, sb2 = ratio ** 2, 1.0
            c = (sa2 * (1 - 2 * e) + sb2) / np.sqrt((sa2 + sb2) * (sa2 * (1 - 2 * e) ** 2 + sb2))
            yok[f"e{e}_sa/sb{ratio}"] = float(c)
    R["yoking_expected_corr_side_vs_choiceAllTrials"] = yok

    # ---------- datasets for higher-order tests
    datasets = OrderedDict([
        ("PC10", pca["scores"][:, :10]),
        ("PC20", pca["scores"][:, :20]),
        ("W100", CW),
        ("SEL", S),
    ])
    # positive controls: k-means-fitted Gaussian mixtures (25 components)
    rng_pc = np.random.default_rng(11)
    datasets["PC10mix25"] = mixture_control(datasets["PC10"], 25, rng_pc)
    datasets["SELmix25"] = mixture_control(S / S.std(0), 25, rng_pc) * S.std(0)
    from scipy.stats import kurtosis as _kurt
    R["pc_excess_kurtosis"] = _kurt(pca["scores"][:, :20], axis=0).tolist()
    R["dataset_dims"] = {k: int(v.shape[1]) for k, v in datasets.items()}

    # ---------- validation
    R["validation"] = validate(X, pca["scores"], np.random.default_rng(1))
    print("validation", R["validation"])
    del X
    torch.cuda.empty_cache()

    # ---------- ePAIRS
    ep = {}
    for name, Y in datasets.items():
        nulls = OrderedDict(NULLS)
        if name == "W100":
            nulls["axismarg"] = null_axismarg
        for excl in ("all", "xprobe"):
            g = grp if excl == "xprobe" else None
            for sub in ("all", "top20"):
                if sub == "top20" and name not in ("SEL", "PC10"):
                    continue
                if name.endswith("mix25") and excl == "xprobe":
                    continue
                key = f"{name}|{excl}|{sub}"
                print("ePAIRS", key, f"{time.time() - t0:.0f}s")
                res = epairs(Y, g, nulls, n_null, seed=7,
                             subset=top_norm if sub == "top20" else None)
                ep[key] = epairs_summary(res)
                if True:  # keep every 5-NN angle distribution for plotting
                    A[f"ang_{key}_real"] = res["real"][5]
                    for nn in nulls:
                        A[f"ang_{key}_{nn}"] = res["nulls"][nn]["pooled"][5]
    R["epairs"] = ep

    # ---------- firing-rate strata (each stratum gets its own fitted nulls):
    # does heterogeneous SNR across neurons explain the angle concentration?
    q = np.quantile(D["fr"], [0, .2, .4, .6, .8, 1])
    strata = np.clip(np.searchsorted(q, D["fr"], side="right") - 1, 0, 4)
    R["fr_quintile_edges"] = q.tolist()
    frs = {}
    for name in ("PC10", "SEL"):
        for qi in range(5):
            idx = np.where(strata == qi)[0]
            res = epairs(datasets[name][idx], grp[idx], NULLS, 5, seed=17 + qi)
            frs[f"{name}|q{qi}"] = epairs_summary(res)
    R["epairs_fr_strata"] = frs

    # ---------- k-means inertia / gap / silhouette
    km = {}
    for name in ("PC10", "PC20", "SEL", "PC10mix25"):
        print("kmeans", name, f"{time.time() - t0:.0f}s")
        Y = datasets[name]
        if name in ("SEL",):  # unit-variance columns so no index dominates the metric
            Y = Y / Y.std(0)
        out = kmeans_curves(Y, NULLS, n_null, seed=3)
        km[name] = out
        A[f"km_{name}_real_logW"] = np.array(out["real"]["logW"])
        A[f"km_{name}_real_sil"] = np.array(out["real"]["sil"])
        for nn in NULLS:
            A[f"km_{name}_{nn}_logW"] = out[nn]["logW"]
            A[f"km_{name}_{nn}_sil"] = out[nn]["sil"]
    R["kmeans"] = {
        name: dict(ks=out["ks"], real_logW=list(map(float, out["real"]["logW"])),
                   real_sil=list(map(float, out["real"]["sil"])),
                   **{nn: dict(gap=out[nn]["gap"].tolist(), gap_sd=out[nn]["gap_sd"].tolist(),
                               sil_mean=out[nn]["sil"].mean(0).tolist(),
                               sil_sd=out[nn]["sil"].std(0, ddof=1).tolist(),
                               inertia_ratio_real_over_null=np.exp(
                                   np.array(out["real"]["logW"]) - out[nn]["logW"].mean(0)).tolist())
                      for nn in NULLS})
        for name, out in km.items()}

    R["runtime_s"] = time.time() - t0
    np.savez_compressed(farr, **A)
    fres.write_text(json.dumps(R, indent=1))
    print(f"done in {R['runtime_s']:.0f}s ->", fres)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--n-null", type=int, default=10)
    a = ap.parse_args()
    main(a.rerun, a.n_null)
