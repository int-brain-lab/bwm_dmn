#!/usr/bin/env python3
"""Regional diversity in the full feature space (no projection), 2026-11-16.

Each group's spread is its multivariate standard deviation in the z-scored supersession
space (concat_z, 54,719 neurons x 1,872 bins): sqrt of the summed per-bin variance
(ddof=1), i.e. the RMS distance of its neurons to the group centroid. It is reported
relative to the same quantity for all neurons together (1 = as diverse as the whole brain).

Groups:
  - Beryl regions with >= MIN_N neurons (root/void excluded),
  - random groups: region labels permuted across all neurons (N_PERM allocations; same
    sizes as the regions),
  - functional groups: k-means of all neurons with k = number of regions (dmn_bwm
    pipeline, sklearn KMeans, random_state=0), clusters with >= MIN_N neurons.

Writes regional_diversity_fullspace.pdf, _values.txt and _data.npz in this folder.
"""

import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from response_correlations_common import use_private_base  # noqa: E402

MIN_N = 20
N_PERM = 1000
MM = 1 / 25.4
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def group_sd(X, labels, sq_norms, ngroups):
    """Multivariate s.d. per group label 0..ngroups-1 (labels < 0 ignored). X on device."""
    keep = labels >= 0
    lab = labels[keep]
    n = torch.bincount(lab, minlength=ngroups).double()
    S = torch.zeros(ngroups, X.shape[1], dtype=torch.float64, device=X.device)
    S.index_add_(0, lab, X[keep])
    Q = torch.zeros(ngroups, dtype=torch.float64, device=X.device).index_add_(0, lab, sq_norms[keep])
    ss = Q - (S ** 2).sum(1) / n  # sum of squared distances to centroid
    return torch.sqrt(ss / (n - 1)), n


def sd_reference(X, labels, ngroups):
    """Direct (slow) computation for validation."""
    out = []
    for g in range(ngroups):
        x = X[labels == g]
        out.append(np.sqrt(((x - x.mean(0)) ** 2).sum(1).sum() / (len(x) - 1)))
    return np.array(out)


def main():
    d = use_private_base()
    r = d.regional_group("kmeans", vers="concat", cv=False, nclus=25)
    X_np = np.asarray(r["concat_z"], dtype=np.float64)
    beryl = np.asarray(r["Beryl"]).astype(str)
    N = len(beryl)

    labs, counts = np.unique(beryl, return_counts=True)
    ok = (counts >= MIN_N) & ~np.isin(labs, ["root", "void"])
    regions = labs[ok]
    K = len(regions)
    reg_idx = {lab: i for i, lab in enumerate(regions)}
    reg_lab = np.array([reg_idx.get(b, -1) for b in beryl])  # -1: small region / root / void
    print(f"{N} neurons, {K} regions with >= {MIN_N} neurons ({(reg_lab >= 0).sum()} neurons)")

    rk = d.regional_group("kmeans", vers="concat", cv=False, nclus=K)
    km_lab = np.asarray(rk["acs"], dtype=int)
    assert np.array_equal(np.asarray(rk["Beryl"]).astype(str), beryl), "row order differs"
    _, km_lab = np.unique(km_lab, return_inverse=True)

    X = torch.tensor(X_np, device=DEV)
    sq = (X ** 2).sum(1)
    total_sd = float(torch.sqrt(((X - X.mean(0)) ** 2).sum() / (N - 1)))

    reg_sd, reg_n = group_sd(X, torch.tensor(reg_lab, device=DEV), sq, K)
    km_sd, km_n = group_sd(X, torch.tensor(km_lab, device=DEV), sq, km_lab.max() + 1)
    km_keep = km_n >= MIN_N
    km_sd, km_n = km_sd[km_keep], km_n[km_keep]

    # validation of the grouped formula against a direct computation
    chk = sd_reference(X_np, reg_lab, 5)
    assert np.allclose(chk, reg_sd[:5].cpu().numpy(), rtol=1e-8), (chk, reg_sd[:5])

    rng = np.random.default_rng(1)
    rand_means, rand_all = [], []
    for _ in range(N_PERM):
        s, _ = group_sd(X, torch.tensor(rng.permutation(reg_lab), device=DEV), sq, K)
        rand_means.append(float(s.mean()))
        rand_all.append(s.cpu().numpy())
    rand_means = np.array(rand_means)

    reg = reg_sd.cpu().numpy() / total_sd
    km = km_sd.cpu().numpy() / total_sd
    rand = np.concatenate(rand_all) / total_sd
    rm = rand_means / total_sd
    p = (1 + (rm <= reg.mean()).sum()) / (1 + N_PERM)
    pos = (reg.mean() - km.mean()) / (rm.mean() - km.mean())
    # neuron-weighted variant (each neuron counts once)
    regw = np.average(reg, weights=reg_n.cpu().numpy())
    kmw = np.average(km, weights=km_n.cpu().numpy())

    np.savez(HERE / "regional_diversity_fullspace_data.npz", regions=regions, reg=reg,
             reg_n=reg_n.cpu().numpy(), km=km, km_n=km_n.cpu().numpy(), rand_means=rm,
             total_sd=total_sd)

    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans"],
                         "font.size": 5.5, "pdf.fonttype": 42, "axes.linewidth": 0.6})
    fig, ax = plt.subplots(figsize=(58 * MM, 129 / 72 * 25.4 * MM))
    lo = min(km.min(), reg.min(), rand.min())
    bins = np.linspace(lo * 0.98, 1.02 * max(reg.max(), rand.max(), km.max()), 40)
    for x, lab, col, w in ((km, f"k-means groups ({len(km)})", "green", 1),
                           (reg, f"anatomical regions ({len(reg)})", "tab:blue", 1),
                           (rand, "random groups", "red", 1 / N_PERM)):
        ax.hist(x, bins=bins, histtype="step", lw=1, color=col, label=lab, weights=np.full(len(x), w))
    ax.set_xlabel("within-group s.d. (fraction of all neurons)")
    ax.set_ylabel("groups")
    ax.spines[["top", "right"]].set_visible(False)
    ax.locator_params(nbins=3)
    ax.legend(frameon=False, loc="upper left", fontsize=5)
    fig.subplots_adjust(left=0.2, right=0.97, top=0.9, bottom=0.2)
    fig.savefig(HERE / "regional_diversity_fullspace.pdf")
    plt.close(fig)

    txt = (f"Full-space within-group s.d. (z-scored feature vectors, {X_np.shape[1]} bins), relative to all "
           f"neurons (total s.d. {total_sd:.2f}).\n"
           f"Beryl regions (>= {MIN_N}): {K}, mean {reg.mean():.4f} (median {np.median(reg):.4f}; "
           f"neuron-weighted {regw:.4f})\n"
           f"random groups, region sizes, {N_PERM} perms: mean {rm.mean():.4f} +- {rm.std():.4f}; "
           f"regions {100 * (1 - reg.mean() / rm.mean()):.2f}% lower, one-sided p = {p:.4f}\n"
           f"k-means k={K} on all neurons: {len(km)} clusters >= {MIN_N}, mean {km.mean():.4f} "
           f"(neuron-weighted {kmw:.4f}); sizes median {np.median(km_n.cpu().numpy()):.0f} "
           f"[{km_n.min():.0f}, {km_n.max():.0f}] vs regions median {np.median(reg_n.cpu().numpy()):.0f} "
           f"[{reg_n.min():.0f}, {reg_n.max():.0f}]\n"
           f"regions at {100 * pos:.0f}% of the way from k-means to random\n")
    (HERE / "regional_diversity_fullspace_values.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
