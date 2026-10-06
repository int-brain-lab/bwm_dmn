#!/usr/bin/env python3
"""Does noise alone turn stimulus-locked "tornado" cells into "sequence" cells?

Tornado cells (canonical clusters 21-28) get independent, temporally smoothed
Gaussian noise added to their odd and even feature vectors (6-bin boxcar, like the
strided 12.5 ms bins), scaled so their median odd/even reliability over the
stimulus window matches that of the sequence cells (canonical 60-69). Rastermap
(Methods parameters) is fitted on the noisy odd vectors; the same cross-validated
statistics as in sequence_vs_stim.py are computed for
  real sequence cells, noisy tornado cells, original tornado cells.
Output: results/degrade_tornado.png/.txt.
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rastermap import Rastermap
from scipy.ndimage import uniform_filter1d
from scipy.stats import spearmanr

ROOT = Path(os.environ.get("DMN_DATA", Path.home() / "dmn"))
RES = Path(__file__).resolve().parent / "results"
# Groups = row ranges in the upsampled Rastermap order (Fig. 5a/b), inclusive
SEQ, TOR = (29600, 35599), (11800, 16299)
STIM = ["block_change_s", "stimLbLcL", "stimLbRcL", "stimRbRcR", "stimRbLcR", "mistake_s"]
C_SEC = 480.0
MM = 1 / 25.4
TRIALS_PER_HALF = {  # median trials per half over the 515 insertions (odd/even stack)
    "inter_trial": 178, "blockL": 94, "blockR": 90, "quiescence": 217, "block_change_s": 12,
    "stimLbLcL": 64, "stimLbRcL": 12, "stimRbRcR": 58, "stimRbLcR": 12, "mistake_s": 30,
    "motor_init": 217, "block_change_m": 12, "sLbLchoiceL": 67, "sLbRchoiceL": 14,
    "sRbRchoiceR": 62, "sRbLchoiceR": 14, "mistake_m": 30, "choiceL": 109, "choiceR": 104,
    "fback1": 185, "fback0": 30}


def zrows(X):
    return (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-9)


def rowcorr(a, b):
    a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1) + 1e-12)


def colcorr(A, B):
    A = (A - A.mean(0)) / (A.std(0) + 1e-9); B = (B - B.mean(0)) / (B.std(0) + 1e-9)
    return A.T @ B / A.shape[0]


def stats(Xo, Xe, seg):
    """Cross-validated statistics over the six stimulus-aligned types."""
    so = np.concatenate([Xo[:, seg[t]] for t in STIM], 1)
    se = np.concatenate([Xe[:, seg[t]] for t in STIM], 1)
    rel = rowcorr(so, se)
    lag0 = np.mean([np.diag(colcorr(Xo[:, seg[t]], Xe[:, seg[t]])).mean() for t in STIM])
    po, pe = Xo[:, seg["stimLbLcL"]].argmax(1), Xe[:, seg["stimLbLcL"]].argmax(1)
    return dict(rel=np.median(rel), frac=np.mean(rel >= 0.3), lag0=lag0,
                rho=spearmanr(po, pe).statistic)


def smooth_noise(rng, shape):
    return uniform_filter1d(rng.standard_normal(shape).astype(np.float32), 6, axis=1)


def fit_order(r):
    """Row order of Fig. 5a/b: Rastermap with grid_upsample=10 fit on odd trials
    (preview_upsample.py -> results/rastermap_upsample10.npz)."""
    f = np.load(RES / "rastermap_upsample10.npz", allow_pickle=True)
    if not np.array_equal(f["uuids"], np.asarray(r["uuids"])):
        raise RuntimeError("upsample10 fit and CV stack neurons differ")
    return np.asarray(f["isort"])


def main():
    r = np.load(ROOT / "concat_cvTrue.npy", allow_pickle=True).flat[0]
    order = fit_order(r)
    lab = np.empty_like(order); lab[order] = np.arange(order.size)  # row of each neuron
    pos = np.empty_like(order); pos[order] = np.arange(order.size)
    start = np.cumsum([0] + list(r["len"].values())); names = list(r["len"])
    seg = {t: slice(start[names.index(t)], start[names.index(t) + 1]) for t in names}
    Xo, Xe = np.asarray(r["X_odd"], np.float32), np.asarray(r["X_even"], np.float32)
    seq = np.flatnonzero((lab >= SEQ[0]) & (lab <= SEQ[1])); seq = seq[np.argsort(pos[seq])]
    tor = np.flatnonzero((lab >= TOR[0]) & (lab <= TOR[1])); tor = tor[np.argsort(pos[tor])]
    target = stats(Xo[seq], Xe[seq], seg)["rel"]

    # Per-neuron noise: map the tornado reliability distribution onto the sequence
    # cells' distribution quantile by quantile (rank-preserving), by vectorized
    # bisection on each neuron's noise scale.
    def rel_rows(A, B):
        so = np.concatenate([A[:, seg[t]] for t in STIM], 1)
        se = np.concatenate([B[:, seg[t]] for t in STIM], 1)
        return rowcorr(so, se)
    so_seq = rel_rows(Xo[seq], Xe[seq])
    rel_tor = rel_rows(Xo[tor], Xe[tor])
    ranks = np.argsort(np.argsort(rel_tor)) / (tor.size - 1)
    target_i = np.quantile(so_seq, ranks)
    rng = np.random.default_rng(0)
    No, Ne = smooth_noise(rng, Xo[tor].shape), smooth_noise(rng, Xe[tor].shape)
    No /= No.std(); Ne /= Ne.std()
    # Noise SD per PETH type ~ 1/sqrt(trials per half) (median over insertions), so
    # rarely sampled conditions are noisier, as in the data.
    col_scale = np.ones(No.shape[1], np.float32)
    for t, n in TRIALS_PER_HALF.items():
        col_scale[seg[t]] = np.sqrt(64.0 / n)
    No *= col_scale; Ne *= col_scale
    lo, hi = np.zeros(tor.size, np.float32), np.full(tor.size, 50, np.float32)
    for _ in range(25):
        s_i = (lo + hi) / 2
        rel = rel_rows(zrows(Xo[tor] + s_i[:, None] * No), zrows(Xe[tor] + s_i[:, None] * Ne))
        up = rel > target_i
        lo, hi = np.where(up, s_i, lo), np.where(up, hi, s_i)
    s_i = (lo + hi) / 2
    s = float(np.median(s_i))
    Do, De = zrows(Xo[tor] + s_i[:, None] * No), zrows(Xe[tor] + s_i[:, None] * Ne)
    model = Rastermap(n_PCs=200, n_clusters=30, locality=0.75, grid_upsample=0,
                      time_lag_window=5, bin_size=1).fit(Do)
    dsort = np.asarray(model.isort)

    S = {"sparse cells (rows 29.6-35.6k)": stats(Xo[seq], Xe[seq], seg),
         "dense tiling + noise": stats(Do, De, seg),
         "dense tiling (rows 11.8-16.3k)": stats(Xo[tor], Xe[tor], seg)}
    lines = [f"per-neuron noise (median scale {s:.2f} x feature SD) maps the tornado reliability "
             f"distribution onto the sequence cells' (median {target:.2f})",
             "group | median reliability | fraction r>=0.3 | xval lag-0 corr | peak-time rho"]
    for k, v in S.items():
        lines.append(f"{k:26s} | {v['rel']:.2f} | {v['frac']:.0%} | {v['lag0']:.2f} | {v['rho']:.2f}")
    (RES / "degrade_tornado.txt").write_text("\n".join(lines) + "\n")
    cols = np.concatenate([np.arange(seg[t].start, seg[t].stop) for t in STIM])
    np.savez_compressed(RES / "degrade_tornado.npz", noisy_odd=Do[dsort][:, cols],
                        noisy_even=De[dsort][:, cols], stim_types=np.array(STIM),
                        stats_names=np.array(list(S)),
                        stats=np.array([[v["rel"], v["frac"], v["lag0"], v["rho"]] for v in S.values()]))
    print("\n".join(lines))

    matplotlib.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Liberation Sans"],
                                "font.size": 5.5, "pdf.fonttype": 42, "mathtext.fontset": "custom",
                                "mathtext.rm": "Liberation Sans"})
    fig, axs = plt.subplots(2, 3, figsize=(183 * MM, 130 * MM), gridspec_kw=dict(hspace=0.35, wspace=0.25))
    panels = [(Xo[seq], Xe[seq], "real sequence cells (canonical 60–69)"),
              (Do[dsort], De[dsort], "tornado cells + trial-count-scaled noise, own fit"),
              (Xo[tor], Xe[tor], "original tornado cells (21–28)")]
    for j, (A, B, title) in enumerate(panels):
        for i, (M, half) in enumerate(((A, "odd (train)"), (B, "even (test)"))):
            ax = axs[i, j]
            Y = np.clip(np.concatenate([M[:, seg[t]] for t in STIM], 1), 0, 1.5) / 1.5
            n = Y.shape[0] // 5 * 5
            ax.imshow(1 - Y[:n].reshape(-1, 5, Y.shape[1]).mean(1), cmap="gray", vmin=0, vmax=1,
                      aspect="auto", interpolation="antialiased", rasterized=True)
            for b in range(1, len(STIM)):
                ax.axvline(72 * b, color="0.5", lw=0.3, ls=":")
            ax.set_xticks([72 * b + 36 for b in range(len(STIM))],
                          [r"$\mathrm{change_b}$", r"$\mathrm{L_sL_cL_b}$", r"$\mathrm{L_sL_cR_b}$",
                           r"$\mathrm{R_sR_cR_b}$", r"$\mathrm{R_sR_cL_b}$", "mistake"],
                          rotation=40, fontsize=4.3)
            ax.set_yticks([])
            k = list(S)[j]
            ax.set_title(f"{title}\n{half}" + (f"   (rel {S[k]['rel']:.2f}, ρ {S[k]['rho']:.2f})" if i else ""),
                         fontsize=5.5)
    fig.savefig(RES / "degrade_tornado.png", dpi=250, facecolor="white")
    print(f"Saved {RES / 'degrade_tornado.png'}")


if __name__ == "__main__":
    main()
