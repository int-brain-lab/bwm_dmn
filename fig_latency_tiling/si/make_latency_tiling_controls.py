#!/usr/bin/env python3
"""Supplementary figure to Fig. 5 (latency tiling), 183 mm wide (since 2026-10-12: shuffle
control and held-out latency left out).

a, b  dense tiling + noise matched to the sparse tiling' reliability, own fit, odd /
      even trials; c statistics (degrade_tornado.py)
d     peak-latency replication at matched rate (sparse_tornado.py)
e, f  regional held-out peak latency vs Harris et al. 2019 hierarchy: e stimulus window
      (four held-out stimulus types, even trials; cells selected on the two concordant
      types, odd trials), f PETH type m (150 ms before movement onset, as Fig. 1f)
      (../../fig_si/latency_hierarchy_analysis.py -> results/latency_hierarchy*.csv)
g     stimulus- vs movement-locking: fast vs slow reaction-time trials (restored 2026-10-13;
      sparse_tornado.py, ../../sequence_analysis/rt_split_latency.py)
Same groups and data as Fig. 5 (../make_latency_tiling.py, whose helpers are reused).
Writes latency_tiling_controls.pdf/.png/_printer.pdf; with latency_tiling_controls_caption.tex
-> latency_tiling_controls_with_caption.pdf (tectonic, ../elife.cls).
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import make_latency_tiling as lt  # noqa: E402

sys.path.insert(0, str(HERE.parents[1] / "fig_si"))
import latency_hierarchy_analysis as LH  # noqa: E402

plt = lt.plt
MM = lt.MM
W, H = 183.0, 118.0


def load_latency_hierarchy():
    d = LH.pd.read_csv(LH.RES / "latency_hierarchy.csv")
    dm = LH.pd.read_csv(LH.RES / "latency_hierarchy_m.csv")
    return [(d, "lat_eval", "peak latency after stimulus (ms)", LH.beryl_colors(d.region)),
            (dm, "lat", "peak latency to movement onset (ms)", LH.beryl_colors(dm.region))]


def main():
    lt.seq_rasters.configure_style()
    lt.use_private_base()
    fig = plt.figure(figsize=(W * MM, H * MM))
    lh = load_latency_hierarchy()  # imports dmn_bwm via fig_anatomy (rcParams), so before styling
    lt.seq_rasters.configure_style()
    g0 = fig.add_gridspec(1, 4, wspace=0.45, left=0.07, right=0.975, top=0.93, bottom=0.6)
    gs = fig.add_gridspec(1, 8, wspace=1.3, left=0.07, right=0.975, top=0.44, bottom=0.09)
    lt.noise_panels(fig, [g0[0, 0], g0[0, 1], g0[0, 2:4]], "abc")
    # sparse_tornado order: reliability, matched-rate timing, sparse zoom, dense zoom,
    # held-out latency, RT test, latency distributions, fraction reliable
    cells = [None, gs[0, 0:2], None, None, None, gs[0, 6:8], None, None]
    lt.sparse_tornado.main(fig=fig, cells=cells, letters="dg")
    for k, (d, col, ylab, c) in enumerate(lh):
        ax = fig.add_subplot(gs[0, 2 + 2 * k:4 + 2 * k])
        LH.draw_panel(ax, d, col, ylab, c)
        ax.text(-0.2, 1.06, "ef"[k], transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")
    lt.save(fig, HERE / "latency_tiling_controls")


if __name__ == "__main__":
    main()
