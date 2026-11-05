#!/usr/bin/env python3
"""Three analyses of the old Fig. 3 (Ari's panels), redone on the Fig. 5 groups.

Groups = cluster sets of the Fig. 5a, b Rastermap fit (latency_tiling_groups.py): dense tiling
clusters 21-28, sparse tiling clusters 0-7, 44, 49-53, 61-69.
Six stimulus-aligned PETH types (72 bins = 150 ms each).

1  Neuron-shuffle control (old Fig. 3b): dense tiling cells, neuron identity permuted
   independently per PETH type (the same permutation for odd and even trials of that
   type, so every response is kept), Rastermap (Methods parameters) fitted on the
   shuffled odd trials, shown on odd and even trials. Statistic: Spearman rho of
   peak times between different types (odd type a vs even type b, a != b) and
   within a type (odd vs even), real vs shuffled.
2  Cross-validated time-time correlation (old Fig. 3d, e, h, j): population-vector
   correlation between odd-trial bin t1 and even-trial bin t2 (rows z-scored over the
   six types), so no spikes are shared between the correlated bins (the 12.5 ms bins
   are strided by 2.1 ms and overlap within a half). Within a type (averaged over the
   six) and across types (odd a vs even b, a != b, averaged over the 30 pairs).
3  Anatomy (old Fig. 3k): Cosmos composition of each group relative to a rate-matched
   null (1,000 draws from all other neurons, matched by rate decile).
Output: results/ari_controls.npz/.txt; draw_* functions are used by ../fig_latency_tiling/make_latency_tiling.py.
"""

import os
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

import latency_tiling_groups

ROOT = Path(os.environ.get("DMN_DATA", Path.home() / "dmn"))
RES = Path(__file__).resolve().parent / "results"
RES.mkdir(exist_ok=True)
STIM6 = ["block_change_s", "stimLbLcL", "stimLbRcL", "stimRbRcR", "stimRbLcR", "mistake_s"]
COSMOS = ["Isocortex", "OLF", "HPF", "CTXsp", "CNU", "TH", "HY", "MB", "HB", "CB"]
COL = {"sparse": "#1f77b4", "dense": "#ff7f0e"}
N_NULL = 1000


def zrows(X):
    return (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-9)


def colcorr(A, B):
    A = (A - A.mean(0)) / (A.std(0) + 1e-9); B = (B - B.mean(0)) / (B.std(0) + 1e-9)
    return A.T @ B / A.shape[0]


def display(X, q=80, gamma=0.5):
    """Display of Fig. 5b, per neuron over its full feature vector."""
    med = np.median(X, 1, keepdims=True)
    hi = np.percentile(X, 99, 1, keepdims=True)
    R = np.clip((X - med) / (hi - med + 1e-6), 0, 1)
    return np.where(R >= np.percentile(R, q, 1, keepdims=True), R, 0) ** gamma


def peak_rhos(Po, Pe):
    """Po, Pe: neurons x types peak bins. Mean rho within types and across types."""
    k = Po.shape[1]
    within = np.mean([spearmanr(Po[:, a], Pe[:, a]).statistic for a in range(k)])
    across = np.mean([spearmanr(Po[:, a], Pe[:, b]).statistic for a in range(k) for b in range(k) if a != b])
    return within, across


def main():
    from iblatlas.regions import BrainRegions
    from rastermap import Rastermap

    r = np.load(ROOT / "concat_cvTrue.npy", allow_pickle=True).flat[0]
    start = np.cumsum([0] + list(r["len"].values())); names = list(r["len"])
    seg = {t: slice(start[names.index(t)], start[names.index(t) + 1]) for t in names}
    Xo, Xe = np.asarray(r["X_odd"], np.float32), np.asarray(r["X_even"], np.float32)
    grp = latency_tiling_groups.group_indices(r)  # each sorted by row in Fig. 5a, b
    out, lines = {}, []

    # 1  neuron-shuffle control on dense tiling
    idx = grp["dense"]
    Do, De = display(Xo[idx]), display(Xe[idx])
    rng = np.random.default_rng(0)
    so, se, ro, re_ = [], [], [], []
    for t in STIM6:
        p = rng.permutation(idx.size)
        so.append(Xo[idx][p, seg[t]]); se.append(Xe[idx][p, seg[t]])
        ro.append(Do[p, seg[t]]); re_.append(De[p, seg[t]])
    So, Se = np.concatenate(so, 1), np.concatenate(se, 1)
    model = Rastermap(n_PCs=200, n_clusters=30, locality=0.75, grid_upsample=0,
                      time_lag_window=5, bin_size=1).fit(zrows(So))
    isort = np.asarray(model.isort)
    out["shuffle_odd"] = np.concatenate(ro, 1)[isort]
    out["shuffle_even"] = np.concatenate(re_, 1)[isort]
    real = peak_rhos(np.stack([Xo[idx, seg[t]].argmax(1) for t in STIM6], 1),
                     np.stack([Xe[idx, seg[t]].argmax(1) for t in STIM6], 1))
    shuf = peak_rhos(np.stack([s.argmax(1) for s in so], 1), np.stack([s.argmax(1) for s in se], 1))
    out["shuffle_rho"] = np.array([real, shuf])  # rows real/shuffled, cols within/across
    lines += [f"neuron-shuffle control, dense tiling (n={idx.size:,}): peak-time rho within type "
              f"(odd vs even) real {real[0]:.2f}, shuffled {shuf[0]:.2f}; across types "
              f"(odd a vs even b) real {real[1]:.2f}, shuffled {shuf[1]:.2f}"]

    # 2  cross-validated time-time correlation
    for g, idx in grp.items():
        Zo = zrows(np.concatenate([Xo[idx, seg[t]] for t in STIM6], 1))
        Ze = zrows(np.concatenate([Xe[idx, seg[t]] for t in STIM6], 1))
        blk = [slice(72 * k, 72 * (k + 1)) for k in range(6)]
        within = np.mean([colcorr(Zo[:, b], Ze[:, b]) for b in blk], 0)
        across = np.mean([colcorr(Zo[:, a], Ze[:, b]) for a in blk for b in blk if a != b], 0)
        out[f"tt_{g}_within"], out[f"tt_{g}_across"] = within, across
        lines.append(f"xval time-time, {g}: mean diagonal within types {np.diag(within).mean():.2f}, "
                     f"across types {np.diag(across).mean():.2f}")

    # 3  anatomy vs rate-matched null
    cosmos = np.asarray(BrainRegions().id2acronym(np.asarray(r["ids"]), mapping="Cosmos"))
    fr = np.asarray(r["fr"], float)
    rng = np.random.default_rng(1)
    for g, idx in grp.items():
        pool = np.setdiff1d(np.arange(fr.size), idx)
        edges = np.quantile(fr[idx], np.linspace(0, 1, 11))
        bins_g = np.clip(np.digitize(fr[idx], edges[1:-1]), 0, 9)
        bins_p = np.digitize(fr[pool], edges[1:-1]); okp = (fr[pool] >= edges[0]) & (fr[pool] <= edges[-1])
        need = np.bincount(bins_g, minlength=10)
        cands = [pool[okp & (bins_p == b)] for b in range(10)]
        obs = np.array([np.mean(cosmos[idx] == c) for c in COSMOS])
        null = np.empty((N_NULL, len(COSMOS)))
        for k in range(N_NULL):
            draw = np.concatenate([rng.choice(cands[b], need[b], replace=need[b] > cands[b].size)
                                   for b in range(10)])
            null[k] = [np.mean(cosmos[draw] == c) for c in COSMOS]
        base = np.array([np.mean(cosmos == c) for c in COSMOS])
        out[f"anat_{g}_obs"], out[f"anat_{g}_null"], out["anat_base"] = obs, null, base
        p = np.minimum(1, 2 * np.minimum((null >= obs).mean(0), (null <= obs).mean(0)) + 1 / N_NULL)
        lines.append(f"anatomy, {g} (n={idx.size:,}): " + ", ".join(
            f"{c} {100 * o:.1f}% (rate-matched {100 * n:.1f}%, all {100 * b:.1f}%, p={pp:.3f})"
            for c, o, n, b, pp in zip(COSMOS, obs, null.mean(0), base, p)))
    out["cosmos"] = np.array(COSMOS)
    np.savez_compressed(RES / "ari_controls.npz", **out)
    (RES / "ari_controls.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# ---------------- drawing (used by ../fig_latency_tiling/make_latency_tiling.py) ----------------
def _stim_axis(ax, labels):
    for b in range(1, 6):
        ax.axvline(72 * b - 0.5, color="0.5", lw=0.3, ls=":")
    ax.set_xticks([72 * b + 36 for b in range(6)], labels, rotation=40, fontsize=4.3)


def draw_shuffle(ax, D, key, labels, title):
    M = D[key]
    ax.imshow(1 - M, cmap="gray", aspect="auto", interpolation="antialiased", vmin=0, vmax=1,
              rasterized=True)
    _stim_axis(ax, labels)
    ax.set_yticks([0, M.shape[0] - 1], ["0", f"{M.shape[0]:,}"])
    ax.set_title(title, fontsize=6, color="0.35")


def draw_timetime(fig, spec, D, letter):
    """2 x 2 heatmaps: rows dense / sparse, columns within / across types."""
    sub = spec.subgridspec(2, 3, width_ratios=[1, 1, 0.07], wspace=0.12, hspace=0.18)
    ext = (0, 150, 150, 0)
    for i, g in enumerate(("dense", "sparse")):
        for j, kind in enumerate(("within", "across")):
            ax = fig.add_subplot(sub[i, j])
            im = ax.imshow(D[f"tt_{g}_{kind}"], cmap="RdBu_r", vmin=-0.6, vmax=0.6, extent=ext,
                           interpolation="nearest", rasterized=True)
            ax.set_xticks([0, 150] if i else []); ax.set_yticks([0, 150] if not j else [])
            ax.tick_params(labelsize=4.5, length=1.5, pad=1)
            if i == 0:
                ax.set_title(f"{kind} type", fontsize=5, pad=2)
            if j == 0:
                ax.set_ylabel(f"{g}\nodd (ms)", fontsize=5, color=COL[g], labelpad=0)
            if i == 1:
                ax.set_xlabel("even (ms)", fontsize=5, labelpad=0)
            if i == 0 and j == 0:
                ax.text(-0.8, 1.45, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")
                ax.text(1.05, 1.45, "time–time, odd vs even", transform=ax.transAxes, ha="center",
                        va="bottom", fontsize=6)
    cax = fig.add_subplot(sub[:, 2])
    cb = fig.colorbar(im, cax=cax, ticks=[-0.6, 0, 0.6]); cb.ax.tick_params(labelsize=4.5, length=1.5, pad=1)
    cb.outline.set_linewidth(0.3); cb.set_label("r", fontsize=5, labelpad=0)


def draw_anatomy(ax, D):
    """Horizontal: one row per Cosmos area, log2(observed / rate-matched)."""
    C = list(D["cosmos"]); y = np.arange(len(C))
    for k, g in enumerate(("dense", "sparse")):
        obs, null = D[f"anat_{g}_obs"], D[f"anat_{g}_null"]
        mu = null.mean(0)
        lo, hi = np.percentile(null, [2.5, 97.5], 0)
        dy = 0.18 * (2 * k - 1)
        ax.hlines(y + dy, np.log2(lo / mu), np.log2(hi / mu), color="0.75", lw=2)
        ax.plot(np.log2(obs / mu), y + dy, "o", color=COL[g], ms=2.5,
                label={"dense": "dense tiling", "sparse": "sparse tiling"}[g])
    ax.axvline(0, color="k", lw=0.4)
    ax.set_yticks(y, C, fontsize=5); ax.invert_yaxis()
    ax.set_xlabel("log$_2$(observed / rate-matched)")
    ax.set_title("where the cells are", fontsize=6, pad=10)
    ax.legend(frameon=False, fontsize=4.5, loc="lower center", bbox_to_anchor=(0.5, 0.99), ncol=2,
              handletextpad=0.1, borderpad=0.1, columnspacing=0.8)
    ax.spines[["top", "right"]].set_visible(False)


if __name__ == "__main__":
    main()
