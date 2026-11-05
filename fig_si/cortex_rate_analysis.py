#!/usr/bin/env python3
"""Cortical vs subcortical neurons: is the divide mainly firing rate? (revised old Fig. S11)

Data: ~/dmn/concat_cvFalse.npy (all trials, 54,719 neurons, as Fig. 3a) with its k-means
labels (25 clusters), joined by uuid to the Fig. 5 CV stack (concat_cvTrue.npy): odd/even reliability and
odd-trial peak latency on the concordant stimulus types (as Fig. 5f), Rastermap clusters
and sparse / dense groups (latency_tiling_groups).

Cortical = Cosmos Isocortex, OLF, HPF, CTXsp; subcortical = CNU, TH, HY, MB, HB, CB
(as old S11); neurons with Beryl or Cosmos root/void removed. Rate = mean of the feature vector (spikes per 12.5 ms
bin, over all PETH windows) x 80 = Hz.

Analyses (results/cortex_rate.npz, results/cortex_rate.txt):
  1  rate per Cosmos area and per Beryl region (>= NMIN neurons); AUC cortical vs sub.
  2  per-neuron decoding of cortical vs subcortical (logistic regression, 5-fold CV
     grouped by insertion so test probes are never seen): log rate; reliability; peak
     latency; k-means cluster; response shape (PCA of the z-scored feature vector);
     shape + rate; and shape / cluster on rate-matched neurons.
  3  composition separation of old S11 (cell-weighted Fisher D of 25-cluster regional
     composition vectors vs region-label shuffles), all neurons vs rate-matched
     neurons (cortical and subcortical subsampled to the same log-rate histogram), vs
     random subsamples of the same size, and on a 25-bin rate composition instead.
  4  per k-means cluster and per Fig. 5 Rastermap cluster: cortical fraction vs rate;
     sparse-tiling fraction (Fig. 5) cortical vs subcortical, all and rate-matched.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline

HERE = Path(__file__).resolve().parent
DMN = Path(os.environ.get("DMN_DATA", Path.home() / "dmn"))  # data
SEQ = Path(__file__).resolve().parents[1] / "sequence_analysis"  # code
sys.path.insert(0, str(SEQ))
import latency_tiling_groups  # noqa: E402
import sparse_tornado as st  # noqa: E402  (SEL, C_SEC, rowcorr: Fig. 5f definitions)

RES = HERE / "results"
RES.mkdir(exist_ok=True)
CTX = ["Isocortex", "OLF", "HPF", "CTXsp"]
SUB = ["CNU", "TH", "HY", "MB", "HB", "CB"]
NMIN = 20
HZ = 1 / 0.0125
RATE_BINS = np.linspace(-3, 2.7, 39)  # log10 Hz, for rate matching
N_MATCH = 20  # rate-matched resamples
N_MATCH_DEC = 5  # rate-matched resamples for decoding
N_PERM = 1000
SEED = 0


def load():
    from iblatlas.regions import BrainRegions
    br = BrainRegions()
    r = np.load(DMN / "concat_cvFalse.npy", allow_pickle=True).flat[0]
    km = np.load(DMN / "kmeans_concat_cvFalse_n25_nclusrm100_zsc1.npy", allow_pickle=True).flat[0]
    ids = np.asarray(r["ids"])
    df = pd.DataFrame(dict(
        uuid=np.asarray(r["uuids"]).astype(str), pid=np.asarray(r["pid"]).astype(str),
        cosmos=br.id2acronym(ids, mapping="Cosmos"), beryl=br.id2acronym(ids, mapping="Beryl"),
        rate=np.asarray(r["fr"], float) * HZ, kmeans=np.asarray(km["kmeans_labels"], int)))
    Z = np.asarray(r["concat_z"], np.float32)

    # Fig. 5 stack: reliability, latency, Rastermap cluster, sparse / dense groups
    rc = np.load(DMN / "concat_cvTrue.npy", allow_pickle=True).flat[0]
    labels = latency_tiling_groups.row_clusters(rc)  # cluster = row span in Fig. 5a, b
    grp = latency_tiling_groups.group_indices(rc)
    start = np.cumsum([0] + list(rc["len"].values())); names = list(rc["len"])
    seg = lambda t: slice(start[names.index(t)], start[names.index(t) + 1])
    so = np.mean([rc["X_odd"][:, seg(t)] for t in st.SEL], 0)
    se = np.mean([rc["X_even"][:, seg(t)] for t in st.SEL], 0)
    f5 = pd.DataFrame(dict(uuid=np.asarray(rc["uuids"]).astype(str), reliability=st.rowcorr(so, se),
                           latency=so.argmax(1) / st.C_SEC * 1000, rm_cluster=labels,
                           group="other"))
    f5.loc[grp["sparse"], "group"] = "sparse"
    f5.loc[grp["dense"], "group"] = "dense"
    df = df.merge(f5, on="uuid", how="left")

    keep = (df["cosmos"].isin(CTX + SUB).to_numpy() & (df["rate"] > 0).to_numpy()
            & ~df["beryl"].isin(["root", "void"]).to_numpy())
    df, Z = df[keep].reset_index(drop=True), Z[keep]
    df["ctx"] = df["cosmos"].isin(CTX)
    df["lograte"] = np.log10(df["rate"])
    return br, df, Z


def rate_match(lograte, y, rng):
    """Indices with equal cortical and subcortical counts in every log-rate bin."""
    b = np.digitize(lograte, RATE_BINS)
    out = []
    for k in np.unique(b):
        c, s = np.flatnonzero((b == k) & y), np.flatnonzero((b == k) & ~y)
        m = min(c.size, s.size)
        if m:
            out += [rng.choice(c, m, replace=False), rng.choice(s, m, replace=False)]
    return np.sort(np.concatenate(out))


def cv_auc(X, y, groups, categorical=False):
    """Insertion-grouped 5-fold CV AUC of a balanced logistic regression."""
    pre = (OneHotEncoder(handle_unknown="ignore", sparse_output=False) if categorical
           else StandardScaler())
    clf = make_pipeline(pre, LogisticRegression(max_iter=2000, class_weight="balanced"))
    p = np.empty(y.size)
    for tr, te in GroupKFold(5).split(X, y, groups):
        p[te] = clf.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return roc_auc_score(y, p)


def composition(df, nclus=25, col="kmeans"):
    regs, P, N, ctx = [], [], [], []
    for reg, d in df.groupby("beryl"):
        if len(d) < NMIN:
            continue
        regs.append(reg)
        P.append(np.bincount(d[col], minlength=nclus) / len(d))
        N.append(len(d)); ctx.append(bool(d["ctx"].iloc[0]))
    return np.array(regs), np.array(P), np.array(N, float), np.array(ctx)


def fisher_d(P, N, mc):
    ms = ~mc
    wc, ws = N[mc], N[ms]
    mu_c = (P[mc] * wc[:, None]).sum(0) / wc.sum()
    mu_s = (P[ms] * ws[:, None]).sum(0) / ws.sum()
    s2c = (wc * ((P[mc] - mu_c) ** 2).sum(1)).sum() / wc.sum()
    s2s = (ws * ((P[ms] - mu_s) ** 2).sum(1)).sum() / ws.sum()
    return np.linalg.norm(mu_c - mu_s) / np.sqrt((s2c + s2s) / 2)


def fisher_test(df, rng, n_perm=N_PERM, col="kmeans"):
    _, P, N, ctx = composition(df, col=col)
    d = fisher_d(P, N, ctx)
    null = np.array([fisher_d(P, N, rng.permutation(ctx)) for _ in range(n_perm)])
    return d, null, int(ctx.sum()), int((~ctx).sum())


def main():
    rng = np.random.default_rng(SEED)
    br, df, Z = load()
    y, groups = df["ctx"].to_numpy(), df["pid"].to_numpy()
    out, lines = {}, []
    lines.append(f"neurons: {len(df):,} ({y.sum():,} cortical, {(~y).sum():,} subcortical), "
                 f"{df['pid'].nunique()} insertions")

    # 1  rate per Cosmos area and Beryl region
    lr = df["lograte"].to_numpy()
    for c in CTX + SUB:
        v = df.loc[df["cosmos"] == c, "rate"]
        lines.append(f"rate {c}: median {v.median():.2f} Hz (IQR {v.quantile(.25):.2f}-"
                     f"{v.quantile(.75):.2f}), n={len(v):,}")
    out["cosmos"] = np.array(CTX + SUB)
    out["cosmos_lograte"] = np.array([df.loc[df["cosmos"] == c, "lograte"].to_numpy()
                                      for c in CTX + SUB], dtype=object)
    med_c, med_s = df.loc[y, "rate"].median(), df.loc[~y, "rate"].median()
    auc_rate_raw = roc_auc_score(~y, lr)  # higher rate -> subcortical
    lines.append(f"median rate cortical {med_c:.2f} Hz, subcortical {med_s:.2f} Hz "
                 f"(ratio {med_s / med_c:.2f}); AUC(rate, sub vs ctx) = {auc_rate_raw:.3f}")
    reg = df.groupby("beryl").agg(n=("rate", "size"), med=("rate", "median"),
                                  ctx=("ctx", "first"), cosmos=("cosmos", "first"))
    reg = reg[reg["n"] >= NMIN].sort_values("med")
    auc_reg = roc_auc_score(~reg["ctx"].to_numpy(), reg["med"].to_numpy())
    lines.append(f"Beryl regions (>={NMIN} neurons): {len(reg)} ({reg['ctx'].sum()} cortical); "
                 f"AUC(regional median rate, sub vs ctx) = {auc_reg:.3f}")
    lines.append("  10 lowest-rate regions: " + ", ".join(
        f"{a} {m:.1f}" for a, m in zip(reg.index[:10], reg["med"][:10])))
    lines.append("  10 highest-rate regions: " + ", ".join(
        f"{a} {m:.1f}" for a, m in zip(reg.index[-10:], reg["med"][-10:])))
    out["reg_names"], out["reg_med"] = reg.index.to_numpy(), reg["med"].to_numpy()
    out["reg_ctx"], out["reg_cosmos"] = reg["ctx"].to_numpy(), reg["cosmos"].to_numpy()
    out["reg_n"] = reg["n"].to_numpy()

    # 2  per-neuron decoding, insertion-grouped CV
    shape = PCA(30, random_state=SEED).fit_transform(Z)
    has5 = df["reliability"].notna().to_numpy()
    feats = {
        "rate": (lr[:, None], False, None),
        "reliability": (df["reliability"].to_numpy()[:, None], False, has5),
        "latency": (df["latency"].to_numpy()[:, None], False, has5),
        "k-means cluster": (df["kmeans"].to_numpy()[:, None], True, None),
        "response shape": (shape, False, None),
        "shape + rate": (np.c_[shape, lr], False, None),
    }
    names, aucs = [], []
    for name, (X, cat, m) in feats.items():
        m = np.ones(y.size, bool) if m is None else m
        a = cv_auc(X[m], y[m], groups[m], cat)
        names.append(name); aucs.append(a)
        lines.append(f"decode ctx vs sub from {name}: AUC {a:.3f} (n={m.sum():,})")
    # rate-matched: shape and cluster on N_MATCH resamples
    rm = {"response shape": [], "k-means cluster": [], "rate": []}
    for k in range(N_MATCH_DEC):
        idx = rate_match(lr, y, rng)
        rm["rate"].append(cv_auc(lr[idx, None], y[idx], groups[idx]))
        rm["response shape"].append(cv_auc(shape[idx], y[idx], groups[idx]))
        rm["k-means cluster"].append(cv_auc(df["kmeans"].to_numpy()[idx, None], y[idx],
                                            groups[idx], True))
    lines.append(f"rate-matched sample: n={idx.size:,} per resample")
    for name, v in rm.items():
        names.append(f"{name}, rate-matched"); aucs.append(np.mean(v))
        lines.append(f"decode ctx vs sub from {name}, rate-matched: AUC {np.mean(v):.3f} "
                     f"(range {np.min(v):.3f}-{np.max(v):.3f}, {N_MATCH_DEC} resamples)")
    out["dec_names"], out["dec_auc"] = np.array(names), np.array(aucs)

    # 3  old S11 composition separation, all vs rate-matched
    d_all, null_all, nc_, ns_ = fisher_test(df, rng)
    p_all = (1 + (null_all >= d_all).sum()) / (1 + N_PERM)
    lines.append(f"Fisher D, all neurons: {d_all:.3f} (null {null_all.mean():.3f} +- "
                 f"{null_all.std():.3f}; z {(d_all - null_all.mean()) / null_all.std():.1f}; "
                 f"p={p_all:.3f}); regions {nc_} ctx / {ns_} sub")
    dm, nm = [], []
    for k in range(N_MATCH):
        idx = rate_match(lr, y, rng)
        d, null, a, b = fisher_test(df.iloc[idx], rng, n_perm=200)
        dm.append(d); nm.append(null)
    dm, nm = np.array(dm), np.array(nm)
    zm = (dm - nm.mean(1)) / nm.std(1)
    pm = (1 + (nm >= dm[:, None]).sum(1)) / 201
    lines.append(f"Fisher D, rate-matched ({N_MATCH} resamples, 200 shuffles each): "
                 f"{dm.mean():.3f} (range {dm.min():.3f}-{dm.max():.3f}); null "
                 f"{nm.mean():.3f}; z {zm.mean():.1f} (range {zm.min():.1f}-{zm.max():.1f}); "
                 f"p median {np.median(pm):.3f}, max {pm.max():.3f}; regions {a} ctx / {b} sub")
    # controls: random subsamples of the rate-matched size (not rate-matched), and the
    # same statistic on a 25-bin rate composition (rate quantiles over all neurons)
    dr, nr = [], []
    for k in range(N_MATCH):
        idx = np.sort(rng.choice(y.size, idx.size, replace=False))
        d, null, _, _ = fisher_test(df.iloc[idx], rng, n_perm=200)
        dr.append(d); nr.append(null)
    dr, nr = np.array(dr), np.array(nr)
    zr = (dr - nr.mean(1)) / nr.std(1)
    lines.append(f"Fisher D, random subsample of the same size ({N_MATCH} resamples): "
                 f"{dr.mean():.3f}; z {zr.mean():.1f} (range {zr.min():.1f}-{zr.max():.1f})")
    df["ratebin"] = np.clip(np.searchsorted(np.quantile(lr, np.linspace(0, 1, 26)[1:-1]), lr), 0, 24)
    d_rate, null_rate, _, _ = fisher_test(df, rng, col="ratebin")
    z_all = (d_all - null_all.mean()) / null_all.std()
    z_rate = (d_rate - null_rate.mean()) / null_rate.std()
    lines.append(f"Fisher D on 25 rate-quantile bins (rate composition), all neurons: {d_rate:.3f} "
                 f"(null {null_rate.mean():.3f}; z {z_rate:.1f})")
    out.update(d_all=d_all, null_all=null_all, d_match=dm, null_match=nm, d_rand=dr,
               null_rand=nr, d_rate=d_rate, null_rate=null_rate,
               fisher_z=np.array([z_all, z_rate, zr.mean(), zm.mean()]),
               fisher_z_sd=np.array([0, 0, zr.std(), zm.std()]))

    # 4  clusters: cortical fraction vs rate; sparse tiling
    base = y.mean()
    km = df.groupby("kmeans").agg(ctx=("ctx", "mean"), med=("rate", "median"), n=("ctx", "size"))
    rho_km = pd.Series(km["ctx"]).corr(np.log10(km["med"]), method="spearman")
    lines.append(f"k-means clusters: Spearman(cortical fraction, log median rate) = {rho_km:.2f}")
    lines.append("  " + ", ".join(f"{k}: {100 * c:.0f}% ctx {m:.1f} Hz" for k, (c, m) in
                                  km[["ctx", "med"]].iterrows()))
    out["km_ctx"], out["km_med"], out["km_n"] = km["ctx"].to_numpy(), km["med"].to_numpy(), km["n"].to_numpy()
    d5 = df[has5]
    rmc = d5.groupby("rm_cluster").agg(ctx=("ctx", "mean"), med=("rate", "median"), n=("ctx", "size"))
    rho_rm = pd.Series(rmc["ctx"]).corr(np.log10(rmc["med"]), method="spearman")
    lines.append(f"Fig. 5 Rastermap clusters: Spearman(cortical fraction, log median rate) = {rho_rm:.2f}")
    out["rm_cluster"] = rmc.index.to_numpy(); out["rm_ctx"] = rmc["ctx"].to_numpy()
    out["rm_med"] = rmc["med"].to_numpy(); out["rm_n"] = rmc["n"].to_numpy()
    out["base_ctx"] = base
    y5, lr5, g5 = d5["ctx"].to_numpy(), d5["lograte"].to_numpy(), d5["group"].to_numpy()
    for g in ("sparse", "dense"):
        fc, fs = np.mean(g5[y5] == g), np.mean(g5[~y5] == g)
        mm = []
        for k in range(N_MATCH):
            idx = rate_match(lr5, y5, rng)
            mm.append([np.mean(g5[idx][y5[idx]] == g), np.mean(g5[idx][~y5[idx]] == g)])
        mm = np.array(mm)
        out[f"{g}_frac"] = np.array([[fc, fs], mm.mean(0)])
        lines.append(f"{g} cells: cortical {100 * fc:.1f}%, subcortical {100 * fs:.1f}% of neurons "
                     f"(ratio {fc / fs:.2f}); rate-matched {100 * mm[:, 0].mean():.1f}% vs "
                     f"{100 * mm[:, 1].mean():.1f}% (ratio {mm[:, 0].mean() / mm[:, 1].mean():.2f})")
        out[f"{g}_cosmos"] = np.array([np.mean(g5[d5["cosmos"].to_numpy() == c] == g)
                                       for c in CTX + SUB])
    out["cosmos_has5"] = np.array([np.sum(d5["cosmos"] == c) for c in CTX + SUB])

    np.savez(RES / "cortex_rate.npz", **out, allow_pickle=True)
    (RES / "cortex_rate.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
