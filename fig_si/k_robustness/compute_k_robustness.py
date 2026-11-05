#!/usr/bin/env python3
"""k-robustness of the regional distribution of response types (R3 comment 5).

For k in KS, fits k-means (sklearn, random_state=0, as dmn_bwm.regional_group) on the
z-scored feature vectors (concat_cvFalse.npy, 54,719 neurons) and, per Beryl region with
>= NMIN neurons, computes:
  - types present: number of clusters with >= 1 neuron, as a fraction of k;
  - specialization S = 1 - H(p)/log k (as Fig. 3);
  - composition enrichment p_rk / p_k (region share / brain-wide share).
Null: neurons shuffled across regions (region sizes kept), N_SHUF times, which gives the
number of types expected from sampling alone. Labels cached per k in cache/.
"""
from pathlib import Path

import numpy as np
from iblatlas.regions import BrainRegions
from sklearn.cluster import KMeans

HERE = Path(__file__).resolve().parent
DATA = Path.home() / "dmn"
KS = [10, 15, 25, 40, 60, 100]
NMIN = 50
N_SHUF = 100


def labels_for(k, X):
    f = HERE / "cache" / f"kmeans_k{k}.npy"
    if f.exists():
        return np.load(f)
    lab = KMeans(n_clusters=k, random_state=0).fit_predict(X).astype(int)
    np.save(f, lab)
    return lab


def per_region(lab, regs, k):
    present, spec, comp = [], [], []
    for g in regs:
        c = np.bincount(lab[g], minlength=k)
        p = c / c.sum()
        nz = p[p > 0]
        present.append((c > 0).sum() / k)
        spec.append(1 + (nz * np.log(nz)).sum() / np.log(k))
        comp.append(p)
    return np.array(present), np.array(spec), np.array(comp)


def main():
    r = np.load(DATA / "concat_cvFalse.npy", allow_pickle=True).flat[0]
    X = np.asarray(r["concat_z"], dtype=float)
    acs = BrainRegions().id2acronym(r["ids"], mapping="Beryl")
    names, counts = np.unique(acs, return_counts=True)
    keep = [n for n, c in zip(names, counts) if c >= NMIN and n not in ("root", "void")]
    regs = [np.flatnonzero(acs == n) for n in keep]
    sizes = np.array([len(g) for g in regs])
    pool = np.concatenate(regs)
    rng = np.random.default_rng(0)

    ref = np.load(DATA / "kmeans_concat_cvFalse_n25_nclusrm100_zsc1.npy", allow_pickle=True).flat[0]
    out = {"ks": KS, "regions": keep, "sizes": sizes}
    for k in KS:
        lab = labels_for(k, X)
        if k == 25:
            from sklearn.metrics import adjusted_rand_score
            out["ari_k25_vs_paper"] = adjusted_rand_score(ref["kmeans_labels"], lab)
            print("ARI k=25 vs paper labels:", out["ari_k25_vs_paper"])
        pres, spec, comp = per_region(lab, regs, k)
        share = np.bincount(lab[pool], minlength=k) / len(pool)
        nul_pres, nul_spec = [], []
        for _ in range(N_SHUF):
            perm = rng.permutation(lab[pool])
            groups = np.split(np.arange(len(pool)), np.cumsum(sizes)[:-1])
            pp, ss, _ = per_region(perm, groups, k)
            nul_pres.append(pp); nul_spec.append(ss)
        out[k] = dict(present=pres, spec=spec, comp=comp, share=share,
                      null_present=np.array(nul_pres), null_spec=np.array(nul_spec))
        print(f"k={k}: median types present {np.median(pres)*k:.0f}/{k} "
              f"(null {np.median(np.median(nul_pres,1))*k:.1f}); median S {np.median(spec):.3f} "
              f"(null {np.median(nul_spec):.3f})")
    np.save(HERE / "cache" / "k_robustness.npy", out, allow_pickle=True)


if __name__ == "__main__":
    main()
