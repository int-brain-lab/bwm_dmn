#!/usr/bin/env python3
"""Gaussian-copula and ICA-marginal populations for Fig. 4c,d (2026-11-17).

These two stricter nulls were added after panels c and d were made (extra_nulls.py,
full_space_epairs.py) and only their AUCs (panel e) were computed. This adds:
  d  k-means silhouette vs k in the top 10 PCs (same procedure as selectivity_test.py:
     kmeans_curves, ks 2-50, 10 draws, seed 3, silhouette on 10,000 neurons)
  c  angle to the 5 most similar neurons (other insertions only), 3 draws pooled,
     in all 1872 bins (main histogram) and in the top 10 PCs (inset)
Nulls are the GPU implementations of full_space_epairs.py (validated there against the
numpy/sklearn ones on the top 10 PCs).
Run (iblenv, GPU): python extra_panels_cd.py -> cache/extra_panels_cd.json, .npz
"""

import json
import time

import numpy as np
import torch

import full_space_epairs as FS
import selectivity_test as ST

NEW = ("copula", "icamarg")


def wrap(nn, gen):
    """numpy-in/numpy-out null for ST.kmeans_curves (its rng argument is unused)."""
    def f(Y, rng):
        return FS.NULLS[nn](torch.as_tensor(Y, dtype=FS.F64, device=FS.DEV), gen).to(
            torch.float32).cpu().numpy()
    return f


def angles(Y, grp, n_draw=3, seed=7):
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=FS.DEV).manual_seed(seed)
    out = {}
    for nn in NEW:
        a = []
        for _ in range(n_draw):
            Yn = FS.NULLS[nn](Y, gen).to(torch.float32).cpu().numpy()
            a.append(ST.knn_angles(Yn, grp[rng.permutation(len(Yn))], ks=(5,))[5])
        out[nn] = np.concatenate(a)
        FS._COP.clear(); FS._ICA.clear()
        torch.cuda.empty_cache()
    return out


def main():
    t0 = time.time()
    D = ST.load_data()
    grp = D["pid_codes"]
    Xt = torch.as_tensor(D["X"], dtype=FS.F64, device=FS.DEV)
    Y10 = ST.pca_torch(D["X"], 100)["scores"][:, :10]  # as selectivity_test.py
    del D

    # d: silhouettes, top 10 PCs
    gen = torch.Generator(device=FS.DEV).manual_seed(3)
    km = ST.kmeans_curves(Y10, {nn: wrap(nn, gen) for nn in NEW}, 10, seed=3)
    FS._COP.clear(); FS._ICA.clear()
    old = json.loads((ST.CACHE / "results.json").read_text())["kmeans"]["PC10"]
    check = float(np.max(np.abs(np.array(km["real"]["sil"]) - np.array(old["real_sil"]))))
    print("real silhouette vs cached, max |diff|", check, f"{time.time() - t0:.0f}s", flush=True)
    R = dict(ks=km["ks"], real_sil_check_max_abs_diff=check,
             **{nn: dict(sil_mean=km[nn]["sil"].mean(0).tolist(),
                         sil_sd=km[nn]["sil"].std(0, ddof=1).tolist()) for nn in NEW})

    # c: angles, all bins and top 10 PCs
    A = {f"ALL_{k}": v for k, v in angles(Xt, grp).items()}
    print("ALL angles", {k: round(float(np.median(v)), 2) for k, v in A.items()}, f"{time.time() - t0:.0f}s", flush=True)
    del Xt
    torch.cuda.empty_cache()
    A.update({f"PC10_{k}": v for k, v in
              angles(torch.as_tensor(Y10, dtype=FS.F64, device=FS.DEV), grp).items()})
    # check: AUC of the cached real angles against the new draws vs the panel-e AUCs
    real = dict(ALL=np.load(ST.CACHE / "full_space_angles.npz")["real"],
                PC10=np.load(ST.CACHE / "arrays.npz")["ang_PC10|xprobe|all_real"])
    ref = dict(ALL=json.loads((ST.CACHE / "full_space_epairs.json").read_text())["epairs"]["ALL|xprobe|all"],
               PC10=json.loads((ST.CACHE / "extra_nulls.json").read_text())["epairs"]["PC10|xprobe|all"])
    R["auc_check"] = {f"{sp}_{nn}": dict(new=float(ST.auc_less(real[sp], A[f"{sp}_{nn}"])),
                                         panel_e=ref[sp][nn]["5"]["auc"])
                      for sp in ("ALL", "PC10") for nn in NEW}
    print("AUC check", R["auc_check"], flush=True)
    R["runtime_s"] = time.time() - t0
    np.savez_compressed(ST.CACHE / "extra_panels_cd.npz", **A)
    (ST.CACHE / "extra_panels_cd.json").write_text(json.dumps(R, indent=1))
    print("done", {k: round(float(np.median(v)), 2) for k, v in A.items()}, f"{R['runtime_s']:.0f}s")


if __name__ == "__main__":
    main()
