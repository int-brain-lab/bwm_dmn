#!/usr/bin/env python3
"""Fig. 5 (draft): brain-wide stimulus-locked latency tiling, from dense to sparse
cells, validated on held-out trials. Controls are in si/make_latency_tiling_controls.py.

Panel order (story): a, b Rastermap on odd (train) and even (test) trials, with
the 100 cluster boundaries and the two groups (latency_tiling_groups.py: sparse tiling
clusters 0-7, 44-54, 61-69; dense tiling clusters 21-28); c, d what replicates: dense tiling and sparse tiling (even trials);
e reliability; f cross-validated time-time correlation; g, h where the cells are
(Beryl pies as in Fig. 3f); i cortical fraction vs rate per cluster (sparse tiling:
low rate, cortical); j median rate vs size per Beryl region.

a, b  Rastermap fit on odd trials stored in the CV stack (Methods parameters,
      grid_upsample=0, all 21 PETH types; isort / rm_labels of concat_cvTrue.npy);
      a: odd trials (train), b: even trials (test), same order.
      Display: per neuron, (x - median) / (99th pct - median) clipped to [0, 1],
      values below the neuron's 80th percentile set to white, gamma 0.5.
c, d, e  ../sequence_analysis/sparse_tornado.py.
f     ../sequence_analysis/ari_controls.py (xval time-time correlation).
g, h  Beryl pies, computed here (conventions of ../fig_anatomy/regenerate_from_data.py).
i, j  ../fig_si/cortex_rate_analysis.py (i = Fig. S11f), drawn with make_cortex_rate.py.
Helpers shuffle_panel, noise_panels and save are shared with si/make_latency_tiling_controls.py.
Run degrade_tornado.py, rt_split_latency.py and ari_controls.py in
../sequence_analysis and ../fig_si/cortex_rate_analysis.py first.
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEQ_DIR = HERE.parent / "sequence_analysis"
sys.path.insert(0, str(SEQ_DIR))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.transforms  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

from seq_common import use_private_base  # noqa: E402
import sparse_tornado  # noqa: E402
import ari_controls  # noqa: E402
import latency_tiling_groups  # noqa: E402
import seq_rasters  # noqa: E402

sys.path.insert(0, str(HERE.parent / "fig_anatomy"))
import regenerate_from_data as fig_anatomy  # noqa: E402  (pie conventions of Fig. 3f)

FIG_SI = HERE.parent / "fig_si"
sys.path.insert(0, str(FIG_SI))
import make_cortex_rate as cortex_rate  # noqa: E402  (Fig. S11f scatter, region scatter)


MM = 1 / 25.4
ROW_BLOCK = 5
# Groups = cluster sets of the a, b fit (latency_tiling_groups.py)
GROUPS = {"sparse tiling": (latency_tiling_groups.SPARSE, "#1f77b4"), "dense tiling": (latency_tiling_groups.DENSE, "#ff7f0e")}
N_PIE_LABELS = 10
PIE_NMIN = 50  # regions with fewer neurons in the stack are left out of the pies
STIM6 = ["block_change_s", "stimLbLcL", "stimLbRcL", "stimRbRcR", "stimRbLcR", "mistake_s"]


def thresholded(X, q=80, gamma=0.5):
    """Per-row robust scaling to [0, 1]; values below the row's q-th percentile -> 0;
    gamma < 1 darkens the kept (faint but above-threshold) values."""
    med = np.median(X, 1, keepdims=True)
    hi = np.percentile(X, 99, 1, keepdims=True)
    R = np.clip((X - med) / (hi - med + 1e-6), 0, 1)
    return np.where(R >= np.percentile(R, q, 1, keepdims=True), R, 0) ** gamma


def to_image(M):
    n = M.shape[0] // ROW_BLOCK * ROW_BLOCK
    M = M[:n].reshape(-1, ROW_BLOCK, M.shape[1]).mean(1)
    return np.repeat((1 - M)[..., None], 3, axis=2)


cluster_bounds = latency_tiling_groups.cluster_bounds  # row span of each cluster in the a, b order


def runs(clusters):
    """Contiguous runs of cluster numbers, e.g. [0..7, 44, 49..53] -> [(0, 7), (44, 44), (49, 53)]."""
    out, cl = [], sorted(clusters)
    for c in cl:
        if out and c == out[-1][1] + 1:
            out[-1] = (out[-1][0], c)
        else:
            out.append((c, c))
    return out


def bracket(ax, lo, hi, color, text=None):
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    ax.plot([1.02, 1.02], [lo, hi], color=color, lw=2.2, transform=trans,
            clip_on=False, solid_capstyle="butt")
    if text:
        ax.text(1.045, (lo + hi) / 2, text, color=color, rotation=90, ha="left",
                va="center", fontsize=5, transform=trans)


def group_pies(fig, cell, r):
    """Beryl composition of dense tiling and sparse tiling as vector pies, with the
    conventions of Fig. 3f (fig_anatomy.draw_cluster_pies): root/void removed, each
    region's count in the group divided by its count in the whole stack, wedges in
    canonical Beryl order, the largest wedges labelled radially at their middle
    angle, label size linear in wedge fraction on one scale for both pies."""
    br = fig_anatomy.BrainRegions()
    regs = np.asarray(br.id2acronym(np.asarray(r["ids"]), mapping="Beryl")).astype(str)
    keep = ~np.isin(np.char.lower(regs), ["root", "void"])
    total = fig_anatomy.Counter(regs[keep])
    canonical = list(br.id2acronym(np.load(Path(fig_anatomy.iblatlas.__file__).parent / "beryl.npy"),
                                   mapping="Beryl"))
    G = latency_tiling_groups.group_indices(r)
    data = []
    for g, name in (("dense", "dense tiling"), ("sparse", "sparse tiling")):
        idx = G[g][keep[G[g]]]
        counts = fig_anatomy.Counter(regs[idx])
        order = [x for x in canonical if x in counts and total[x] >= PIE_NMIN]
        order += sorted(x for x in counts if x not in set(order) and total[x] >= PIE_NMIN)
        w = np.array([counts[x] / total[x] for x in order])
        data.append((name, order, w / w.sum(), G[g].size))
    top = [np.sort(f)[::-1][:N_PIE_LABELS] for _, _, f, _ in data]
    f_lo, f_hi = min(float(t.min()) for t in top), max(float(t.max()) for t in top)
    sg = cell.subgridspec(1, 2, wspace=0.35)
    labels, titles, axes = [], [], []
    for k, (name, order, frac, n) in enumerate(data):
        ax = fig.add_subplot(sg[0, k], projection="polar")
        edges = np.r_[0.0, 2 * np.pi * np.cumsum(frac)]
        colors = fig_anatomy.region_colors(br, order)
        ax.bar(edges[:-1], np.ones(len(frac)), width=np.diff(edges), bottom=0.0,
               align="edge", color=colors, edgecolor=colors, linewidth=0.15)
        ax.set_ylim(0, fig_anatomy.PIE_RMAX)
        ax.set_axis_off()
        titles.append(ax.text(0.5, 1.2, f"{name} (n = {n:,})", transform=ax.transAxes, ha="center",
                              va="bottom", fontsize=6, color=GROUPS[name][1]))
        for i in np.argsort(-frac)[:N_PIE_LABELS]:
            ax.bar(edges[i], 1.0, width=edges[i + 1] - edges[i], bottom=0.0, align="edge",
                   fill=False, edgecolor="white", linewidth=0.5, zorder=3)
            mid = edges[i] + np.pi * frac[i]
            size = 3.5 + 3.5 * (frac[i] - f_lo) / (f_hi - f_lo)
            t = ax.text(mid, 1.06, order[i], rotation_mode="anchor", va="center",
                        fontsize=size, color=colors[i], clip_on=False)
            fig_anatomy.orient_radially(t, mid)
            t.wedge_mid = mid
            labels.append(t)
        axes.append(ax)
    fig_anatomy.separate_labels(fig, labels, titles)
    fig_anatomy.check_labels_centred(fig, labels)
    return axes


def cortex_rate_panel(ax):
    """i: per Rastermap cluster of a, b, cortical fraction vs median rate
    (../fig_si/results/cortex_rate.npz; drawn as in Fig. S11f)."""
    R = np.load(FIG_SI / "results/cortex_rate.npz", allow_pickle=True)
    cl = R["rm_cluster"].astype(int)
    col = np.where(np.isin(cl, latency_tiling_groups.SPARSE), GROUPS["sparse tiling"][1],
                   np.where(np.isin(cl, latency_tiling_groups.DENSE), GROUPS["dense tiling"][1], "0.6"))
    o = np.argsort(col != "0.6", kind="stable")  # grey first, so blue / orange lie on top
    cortex_rate.scatter_clusters(ax, R["rm_med"][o], R["rm_ctx"][o], R["rm_n"][o], R["base_ctx"],
                                 colors=col[o], size=(0.8, 7))
    k = int(np.argmax(R["rm_ctx"]))  # top left
    x, y = np.log10(R["rm_med"][k]), R["rm_ctx"][k]
    ax.annotate(f"Rastermap cluster {cl[k]}", (x, y), xytext=(x + 0.35, y + 0.12), fontsize=5,
                arrowprops=dict(arrowstyle="-", lw=0.4, color="0.3"), va="bottom")
    ax.set_ylim(0, 0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.98, 0.3, "sparse tiling", color=GROUPS["sparse tiling"][1], transform=ax.transAxes,
            ha="right", fontsize=5)
    ax.text(0.98, 0.2, "dense tiling", color=GROUPS["dense tiling"][1], transform=ax.transAxes,
            ha="right", fontsize=5)
    ax.text(-0.3, 1.06, "i", transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")


def region_rate_panel(ax):
    """j: one dot per Beryl region (>= 20 neurons, cortical and subcortical Cosmos areas),
    Beryl colour: median firing rate vs number of neurons (../fig_si/results/cortex_rate.npz)."""
    R = np.load(FIG_SI / "results/cortex_rate.npz", allow_pickle=True)
    cortex_rate.panel_region_scatter(ax, R, cortex_rate.beryl_colors(R["reg_names"]))
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(-0.3, 1.06, "j", transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")


def shuffle_panel(fig, cell, letter):
    """Neuron-shuffle control (ari_controls): dense tiling, identities permuted per type."""
    A = np.load(SEQ_DIR / "results/ari_controls.npz", allow_pickle=True)
    ax = fig.add_subplot(cell)
    ari_controls.draw_shuffle(ax, A, "shuffle_odd", stim_labels(),
                              "dense tiling, neurons shuffled\nper type, own fit (odd trials)")
    ax.text(-0.2, 1.1, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")


def stim_labels():
    return [seq_rasters.dmn_bwm.peth_dictm[t] for t in STIM6]


def noise_panels(fig, cells, letters):
    """degrade_tornado: dense tiling + matched noise, own fit, odd and even; statistics."""
    D = np.load(SEQ_DIR / "results/degrade_tornado.npz", allow_pickle=True)
    for k, (key, half) in enumerate((("noisy_odd", "odd (train)"), ("noisy_even", "even (test)"))):
        ax = fig.add_subplot(cells[k])
        ax.imshow(to_image(thresholded(D[key]))[..., 0], cmap="gray", vmin=0, vmax=1,
                  aspect="auto", interpolation="antialiased", rasterized=True)
        for b in range(1, 6):
            ax.axvline(72 * b, color="0.5", lw=0.3, ls=":")
        ax.set_xticks([72 * b + 36 for b in range(6)], stim_labels(), rotation=40, fontsize=4.3)
        ax.set_yticks([])
        ax.set_title(f"dense tiling + noise, own fit\n{half}", fontsize=5.5)
        assert abs(ax.get_xlim()[1] - ax.get_xlim()[0] - 6 * 72) < 2, ax.get_xlim()
        L, x1 = 0.05 * seq_rasters.dmn_bwm.c_sec, ax.get_xlim()[1]  # 50 ms; segments are 150 ms
        tr = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.plot([x1 - L, x1], [1.015, 1.015], color="k", lw=1.0, transform=tr,
                clip_on=False, solid_capstyle="butt")
        ax.text(x1 - L - 4, 1.015, "50 ms", ha="right", va="center", fontsize=4.5, transform=tr)
        ax.text(-0.22, 1.06, letters[k], transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")
    ax = fig.add_subplot(cells[2])
    names = ["sparse tiling", "dense tiling + noise", "dense tiling"]
    colors = ["#1f77b4", "0.45", "#ff7f0e"]
    metrics = ["median reliability", "fraction r ≥ 0.3", "xval lag-0 corr", "peak-time ρ"]
    S = D["stats"]
    for j in range(3):
        ax.bar(np.arange(4) + 0.27 * (j - 1), S[j], width=0.26, color=colors[j], label=names[j])
    ax.set_xticks(range(4), metrics, fontsize=5)
    ax.set_title("noise alone turns dense tiling into sparse-tiling statistics", fontsize=6)
    ax.legend(frameon=False, fontsize=4.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(-0.1, 1.06, letters[2], transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")


def save(fig, stem):
    """PDF (600 dpi raster parts), PNG, and a Ghostscript /printer-compressed PDF."""
    fig.savefig(stem.with_suffix(".pdf"), dpi=600, facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white")
    plt.close(fig)
    subprocess.run(["gs", "-q", "-dNOPAUSE", "-dBATCH", "-dSAFER", "-sDEVICE=pdfwrite",
                    "-dCompatibilityLevel=1.5", "-dPDFSETTINGS=/printer",
                    f"-sOutputFile={stem}_printer.pdf", str(stem.with_suffix(".pdf"))], check=True)
    print(f"Saved {stem}.pdf/.png and {stem.name}_printer.pdf")


def raster(fig, rect, img, r, n_rows, title, show_ylabel):
    """Raster with PETH-type labels, dotted segment boundaries, 200 ms bar."""
    ax = fig.add_axes(rect)
    n_bins = sum(r["len"].values())
    ax.imshow(img, aspect="auto", interpolation="antialiased", rasterized=True,
              extent=(0, n_bins, n_rows, 0))
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    start = 0
    for seg, n in r["len"].items():
        ax.text(start + n / 2, 1.01, r["peth_dict"][seg], transform=trans,
                rotation=seq_rasters.SEG_ROTATION, ha="left", va="bottom", rotation_mode="anchor",
                fontsize=seq_rasters.SEG_FONTSIZE, clip_on=False)
        start += n
        if start < n_bins:
            ax.axvline(start, color="0.5", lw=0.35, ls=(0, (2, 2)), zorder=6)
    ax.set_xlim(0, n_bins); ax.set_xticks([])
    if show_ylabel:
        ax.set_ylabel("Rastermap cluster")
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_title(title, fontsize=6, pad=34)
    seq_rasters.scale_bar(ax, 0.2 * seq_rasters.dmn_bwm.c_sec, "200 ms")
    return ax


def main():
    seq_rasters.configure_style()
    d = use_private_base()
    r = d.regional_group("rm", vers="concat", cv=True)
    order, labels = latency_tiling_groups.load_fit(r)  # grid_upsample=0 fit on odd trials
    med, cut = cluster_bounds(labels[order])

    # layout in mm from the top (fits with the caption on one A4 page)
    H = 172.0  # figure height
    R_TOP, R_H = 16.5, 76.0  # rasters a, b: top (below the PETH labels), height
    ROW1 = (105.0, 130.0)  # c-f: top, bottom
    ROW2 = (146.5, 163.5)  # g-j: top, bottom (pie labels extend below)
    fig = plt.figure(figsize=(183 * MM, H * MM))
    for k, (key, title) in enumerate((("concat_z_train", "train: odd trials, sorted by their Rastermap fit"),
                                      ("concat_z", "test: even trials, same order (cross-validated)"))):
        img = to_image(thresholded(np.asarray(r[key], np.float32)[order]))
        ax = raster(fig, [0.06 + k * 0.48, 1 - (R_TOP + R_H) / H, 0.39, R_H / H], img, r, order.size, title,
                    show_ylabel=(k == 0))
        for y in cut[1:-1]:  # cluster boundaries
            ax.axhline(y, color="0.55", lw=0.15, zorder=5)
        ax.set_yticks(med[::5], [str(c) for c in range(0, med.size, 5)])
        ax.tick_params(axis="y", labelsize=4.5, length=1.5, pad=1)
        for text, (clusters, color) in GROUPS.items():
            rr = runs(clusters)
            longest = max(rr, key=lambda x: x[1] - x[0])
            for c0, c1 in rr:
                bracket(ax, cut[c0], cut[c1 + 1], color, text if (c0, c1) == longest else None)
        seq_rasters.label(fig, ax, "ab"[k], dy=0.04 * 180 / H)

    g1 = fig.add_gridspec(1, 4, wspace=0.55, left=0.07, right=0.965,
                          top=1 - ROW1[0] / H, bottom=1 - ROW1[1] / H)
    g2 = fig.add_gridspec(1, 4, wspace=0.55, left=0.07, right=0.965,
                          top=1 - ROW2[0] / H, bottom=1 - ROW2[1] / H)
    # sparse_tornado draws: reliability, matched-rate timing, sparse zoom, dense zoom,
    # held-out latency, RT test, latency distributions, fraction reliable (None = left
    # out; the rest are in si/make_latency_tiling_controls.py)
    cells = [g1[0, 2], None, g1[0, 1], g1[0, 0], None, None, None, None]
    sparse_tornado.main(fig=fig, cells=cells, letters="edc")
    for ax in fig.axes:  # y label on the first zoom (c), not the second (d)
        if ax.get_ylabel().startswith("Rastermap order"):
            ax.set_ylabel("")
        if ax.get_title().startswith("dense tiling, even"):
            ax.set_ylabel("Rastermap order (fit on odd trials)")
        if ax.get_title().startswith(("dense tiling, even", "sparse tiling, even")):
            # each trial-type segment is 72 bins = 150 ms after stimulus onset
            assert abs(ax.get_xlim()[1] - ax.get_xlim()[0] - 6 * 72) < 2, ax.get_xlim()
            L, x1 = 0.05 * seq_rasters.dmn_bwm.c_sec, ax.get_xlim()[1]
            tr = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
            ax.plot([x1 - L, x1], [1.03, 1.03], color="k", lw=1.0, transform=tr,
                    clip_on=False, solid_capstyle="butt")
            ax.text(x1 - L - 4, 1.03, "50 ms", ha="right", va="center", fontsize=4.5,
                    transform=tr)

    A = np.load(SEQ_DIR / "results/ari_controls.npz", allow_pickle=True)
    ari_controls.draw_timetime(fig, g1[0, 3], A, "f")
    pies = group_pies(fig, g2[0, :2], r)
    for ax, letter in zip(pies, "gh"):
        ax.text(-0.55, 1.2, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")
    cortex_rate_panel(fig.add_subplot(g2[0, 2]))
    region_rate_panel(fig.add_subplot(g2[0, 3]))

    save(fig, HERE / "latency_tiling")

if __name__ == "__main__":
    main()
