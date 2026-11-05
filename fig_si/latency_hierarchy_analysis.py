#!/usr/bin/env python3
"""Does stimulus response latency follow the anatomical hierarchy (Harris et al. 2019)?

Per neuron (Fig. 5 CV stack, concat_cvTrue.npy; definitions of sparse_tornado.py /
Fig. 5e): odd/even reliability on the two concordant stimulus types (SEL); peak latency
on odd SEL trials (selection) and, cross-validated, on the four held-out stimulus types,
even trials (EVAL). Cells kept: reliable (r >= R_MIN) with the selection peak inside
the window (6-144 ms, not at an edge).

Per Beryl region with a Harris score (Cre_conf CC+TC+CT, 61 cortical and thalamic
areas; harris_hierarchy_vs_cosine.get_hierarchy_scores) and >= NMIN kept cells: median
held-out latency. Statistics: Spearman rho of regional latency vs hierarchy score, all
areas, cortex only, thalamus only, visual areas; split-half consistency of regional
latency (selection vs held-out estimate); permutation p (region labels shuffled).
Also the latency of Fig. 5 dense-tiling cells only, and window m (150 ms before movement
onset, as Fig. 1e; reliability and peak odd vs even). Circles cortex, squares thalamus.
Writes results/latency_hierarchy(.csv, _m.csv, .txt, .npz, .pdf/.png draft).
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[1]  # code
DMN = Path(os.environ.get("DMN_DATA", Path.home() / "dmn"))  # data
sys.path.insert(0, str(REPO / "sequence_analysis"))
sys.path.insert(0, str(REPO))
import latency_tiling_groups  # noqa: E402
import sparse_tornado as st  # noqa: E402
from harris_hierarchy_vs_cosine import get_hierarchy_scores  # noqa: E402

RES = HERE / "results"
R_MIN = 0.3
NMIN = 10
N_PERM = 10000
VISUAL = ["LGd", "LP", "VISp", "VISl", "VISrl", "VISal", "VISpm", "VISam", "VISa", "VISli",
          "VISpl", "VISpor"]


def perm_p(x, y, rng):
    rho = spearmanr(x, y).statistic
    null = np.array([spearmanr(x, rng.permutation(y)).statistic for _ in range(N_PERM)])
    return rho, (1 + np.sum(np.abs(null) >= abs(rho))) / (1 + N_PERM)


def draw_panel(ax, d, col, ylabel, colors):
    """One dot per Harris area (Beryl colour; circle cortex, square thalamus): median
    held-out latency vs hierarchy score; Spearman rho in the title. Used by the draft
    below and by Fig. 5---figure supplement 1 (fig_latency_tiling/si/make_latency_tiling_controls.py)."""
    for mk, cs in (("o", "Isocortex"), ("s", "TH")):
        m = (d.cosmos == cs).to_numpy()
        ax.scatter(d.hierarchy[m], d[col][m], c=colors[m], marker=mk, s=7, edgecolors="none")
    for x, y, t, cc in zip(d.hierarchy, d[col], d.region, colors):
        ax.text(x, y + 1.5, t, fontsize=3.3, ha="center", color=cc)
    rho = spearmanr(d.hierarchy, d[col]).statistic
    ax.set_title(f"Spearman ρ = {rho:.2f} ({len(d)} areas)", fontsize=5.5)
    ax.set_xlabel("hierarchy score (Harris et al. 2019)"); ax.set_ylabel(ylabel)
    ax.locator_params(nbins=3)
    ax.spines[["top", "right"]].set_visible(False)


def beryl_colors(regions):
    sys.path.insert(0, str(REPO / "fig_anatomy"))
    import regenerate_from_data as fig_anatomy
    return fig_anatomy.region_colors(None, np.asarray(regions), "Beryl")


def draft_plot(df, dm):
    """Draft: regional held-out latency vs Harris hierarchy, stimulus window and window m."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cs, cm = beryl_colors(df.region), beryl_colors(dm.region)
    plt.rcParams.update({"font.size": 6, "axes.linewidth": 0.5, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(120 / 25.4, 50 / 25.4), constrained_layout=True)
    draw_panel(axes[0], df, "lat_eval", "stimulus window: peak latency (ms)", cs)
    draw_panel(axes[1], dm, "lat", "window m: peak latency\nto movement onset (ms)", cm)
    for ext in ("pdf", "png"):
        fig.savefig(RES / f"latency_hierarchy.{ext}", dpi=300, facecolor="white")


def main():
    from iblatlas.regions import BrainRegions
    br = BrainRegions()
    rng = np.random.default_rng(0)
    r = np.load(DMN / "concat_cvTrue.npy", allow_pickle=True).flat[0]
    start = np.cumsum([0] + list(r["len"].values())); names = list(r["len"])
    seg = lambda t: slice(start[names.index(t)], start[names.index(t) + 1])
    Xo, Xe = r["X_odd"], r["X_even"]
    so = np.mean([Xo[:, seg(t)] for t in st.SEL], 0); se = np.mean([Xe[:, seg(t)] for t in st.SEL], 0)
    rel = st.rowcorr(so, se)
    peak_sel = so.argmax(1) / st.C_SEC * 1000
    peak_eval = np.mean([Xe[:, seg(t)] for t in st.EVAL], 0).argmax(1) / st.C_SEC * 1000
    keep = (rel >= R_MIN) & (peak_sel >= 6) & (peak_sel <= 144)
    ids = np.asarray(r["ids"])
    beryl = np.asarray(br.id2acronym(ids, mapping="Beryl"))
    cosmos = np.asarray(br.id2acronym(ids, mapping="Cosmos"))
    dense = np.zeros(ids.size, bool); dense[latency_tiling_groups.group_indices(r)["dense"]] = True
    H = get_hierarchy_scores()

    rows = []
    for reg in sorted(set(beryl[keep]) & set(H)):
        m = keep & (beryl == reg)
        md = m & dense
        n_all = int((beryl == reg).sum())
        rows.append(dict(region=reg, cosmos=cosmos[beryl == reg][0], hierarchy=H[reg], n=int(m.sum()),
                         n_total=n_all, frac_kept=m.sum() / n_all,
                         lat_eval=np.median(peak_eval[m]), lat_sel=np.median(peak_sel[m]),
                         n_dense=int(md.sum()),
                         lat_dense=np.median(peak_eval[md]) if md.sum() >= 5 else np.nan))
    df = pd.DataFrame(rows)
    df = df[df["n"] >= NMIN].sort_values("hierarchy").reset_index(drop=True)
    df.to_csv(RES / "latency_hierarchy.csv", index=False)

    L = [f"cells kept (reliable r>={R_MIN}, interior peak): {keep.sum():,} of {keep.size:,}",
         f"Harris areas with >={NMIN} kept cells: {len(df)} "
         f"({(df.cosmos == 'Isocortex').sum()} cortical, {(df.cosmos == 'TH').sum()} thalamic)"]
    rs = spearmanr(df.lat_sel, df.lat_eval).statistic
    L.append(f"split consistency of regional latency (selection odd vs held-out even): rho={rs:.2f}")
    out = {}
    for name, m in (("all", np.ones(len(df), bool)), ("cortex", (df.cosmos == "Isocortex").to_numpy()),
                    ("thalamus", (df.cosmos == "TH").to_numpy()),
                    ("visual", df.region.isin(VISUAL).to_numpy())):
        if m.sum() < 5:
            continue
        rho, p = perm_p(df.hierarchy[m].to_numpy(), df.lat_eval[m].to_numpy(), rng)
        out[name] = (rho, p, int(m.sum()))
        L.append(f"held-out latency vs hierarchy, {name} (n={m.sum()}): Spearman rho={rho:.2f}, perm p={p:.4f}")
    md = df.lat_dense.notna().to_numpy() & (df.n_dense >= 5).to_numpy()
    rho, p = perm_p(df.hierarchy[md].to_numpy(), df.lat_dense[md].to_numpy(), rng)
    L.append(f"dense-tiling cells only (n={md.sum()} areas with >=5): rho={rho:.2f}, perm p={p:.4f}")
    rho, p = perm_p(df.hierarchy.to_numpy(), df.frac_kept.to_numpy(), rng)
    L.append(f"(control) fraction of reliable cells vs hierarchy: rho={rho:.2f}, perm p={p:.4f}")
    L.append("areas, ordered by hierarchy: " + ", ".join(
        f"{a} {h:.2f}: {l:.0f} ms (n={n})" for a, h, l, n in zip(df.region, df.hierarchy, df.lat_eval, df.n)))
    # window m (150 ms before movement onset, Fig. 1e): reliability odd vs even, peak on
    # odd (selection), held-out peak on even
    sm = seg("motor_init")
    mo, me = Xo[:, sm], Xe[:, sm]
    nb = mo.shape[1]
    po, pe = mo.argmax(1), me.argmax(1)
    km = (st.rowcorr(mo, me) >= R_MIN) & (po >= 6) & (po <= nb - 7)
    lat_m = (pe - nb) / st.C_SEC * 1000  # ms relative to movement onset
    rows_m = []
    for reg in sorted(set(beryl[km]) & set(H)):
        m = km & (beryl == reg)
        if m.sum() >= NMIN:
            rows_m.append(dict(region=reg, cosmos=cosmos[beryl == reg][0], hierarchy=H[reg], n=int(m.sum()),
                               lat=np.median(lat_m[m]), lat_odd=np.median((po[m] - nb) / st.C_SEC * 1000)))
    dm = pd.DataFrame(rows_m).sort_values("hierarchy").reset_index(drop=True)
    dm.to_csv(RES / "latency_hierarchy_m.csv", index=False)
    L.append(f"window m: cells kept {km.sum():,}; areas {len(dm)}; split consistency "
             f"rho={spearmanr(dm.lat_odd, dm.lat).statistic:.2f}")
    for name, m in (("all", np.ones(len(dm), bool)), ("cortex", (dm.cosmos == "Isocortex").to_numpy()),
                    ("thalamus", (dm.cosmos == "TH").to_numpy())):
        rho, p = perm_p(dm.hierarchy[m].to_numpy(), dm.lat[m].to_numpy(), rng)
        L.append(f"window m, held-out latency vs hierarchy, {name} (n={m.sum()}): rho={rho:.2f}, perm p={p:.4f}")
    draft_plot(df, dm)
    (RES / "latency_hierarchy.txt").write_text("\n".join(L) + "\n")
    np.savez(RES / "latency_hierarchy.npz", stats=np.array([(k, *v) for k, v in out.items()], dtype=object),
             allow_pickle=True)
    print("\n".join(L))


if __name__ == "__main__":
    main()
