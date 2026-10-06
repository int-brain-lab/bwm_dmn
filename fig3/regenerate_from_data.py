#!/usr/bin/env python3
"""Regenerate the anatomy/function correspondence figure from local data."""

from collections import Counter
from pathlib import Path
import subprocess
import sys
from fig3_common import ROOT, use_private_base  # sets the Agg backend first
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator
from scipy.stats import entropy, pearsonr
import iblatlas
from iblatlas.regions import BrainRegions
from iblatlas.plots import plot_swanson_vector

_d = use_private_base()  # all dmn_bwm caches/figures go to fig3/cache
clus_freqs, pal, regional_group = _d.clus_freqs, _d.pal, _d.regional_group


OUT = Path(__file__).resolve().parent
STACK = ROOT / "concat_cvFalse.npy"
CLUS_COUNTS = ROOT / "counts/cf_Beryl_concat_cvFalse_clustkmeans_n25_nrm100_nmin50_norm1_all.npy"
DEC_COUNTS = ROOT / "counts/cf_dec_concat_cvFalse_clustkmeans_n25_nrm100_nmin50_norm1_all.npy"

EXAMPLE_REGIONS = ["CP", "MRN", "ZI", "MOp", "CA1", "CUL4 5"]
G_EXAMPLE_REGIONS = ["PA", "PAA", "MOB", "MEA", "MRN", "SCm", "PRNr", "PGRN"]
HARRIS_HIERARCHY = [
    "VPM", "VPL", "PCN", "LGd", "CL", "IAD", "VISp", "MG", "AM", "IMD", "AUDp", "SSp-n",
    "SSp-ll", "AUDd", "MD", "SSp-ul", "SSp-m", "PT", "SSp-bfd", "SSs", "AIp", "VISl",
    "VISrl", "RSPd", "LD", "MOp", "VISli", "PO", "VISpl", "RSPagl", "RSPv", "VISal", "PVT",
    "CM", "VISpm", "AId", "SSp-tr", "AV", "VAL", "SMT", "LP", "ORBi", "AUDpo", "PL", "ORBm",
    "ILA", "FRP", "VISpor", "ACAv", "VISam", "VISa", "MOs", "TEa", "AIv", "ACAd", "ORBl",
    "PIL", "PF", "RE", "VM", "POL",
]


def configure_style():
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 6, "axes.labelsize": 6, "xtick.labelsize": 5, "ytick.labelsize": 5,
        "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2, "ytick.major.size": 2, "pdf.fonttype": 42, "svg.fonttype": "none",
        # math text ($r$, $p$) in the same sans-serif as all other text
        "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans",
        "mathtext.it": "Liberation Sans:italic", "mathtext.bf": "Liberation Sans:bold",
    })


def region_colors(br, labels, mapping="Beryl"):
    names = np.asarray(labels, dtype=str)
    if mapping == "Beryl":
        # dmn_bwm.get_allen_info replaces the unreadable cerebellar yellow.
        return np.asarray([pal[name][:3] for name in names], dtype=float)
    ids = br.acronym2id(names, mapping=mapping)
    colors = br.get(ids).rgb / 255
    for i, name in enumerate(names):
        if name in pal:
            colors[i] = pal[name][:3]
    return colors


def crop_panel_image(path):
    image = plt.imread(path)
    content = np.any(image[..., :3] < 0.985, axis=2)
    yy, xx = np.where(content)
    return image[yy.min():yy.max() + 1, xx.min():xx.max() + 1]


def source_image(name, box, size):
    """Crop only the white surround of an unchanged source PDF image."""
    image = plt.imread(OUT / name)
    if image.shape[:2] != size:
        raise ValueError(f"Unexpected source image dimensions for {name}: {image.shape[:2]}")
    left, top, right, bottom = box
    return image[top:bottom, left:right]


def inset_axis(ax, left=0.08, right=0.09, bottom=0.065, top=0.035):
    pos = ax.get_position()
    ax.set_position([pos.x0 + left * pos.width,
                     pos.y0 + bottom * pos.height,
                     pos.width * (1 - left - right),
                     pos.height * (1 - bottom - top)])


def xyz_image_triad(ax):
    """Orientation axes for the source anatomical view (elev=45.78, azim=-33.4)."""
    anchor = np.array([-0.01, 0.19])
    directions = {"x": (0.13, -0.045), "y": (-0.11, -0.07), "z": (0.0, 0.15)}
    for name, direction in directions.items():
        tip = anchor + direction
        ax.annotate("", xy=tip, xytext=anchor, xycoords="axes fraction",
                    annotation_clip=False,
                    arrowprops={"arrowstyle": "-|>", "lw": 0.8,
                                "mutation_scale": 6, "color": "black",
                                "shrinkA": 0, "shrinkB": 0})
        ax.text(*(tip + 0.022 * np.asarray(direction) / np.linalg.norm(direction)),
                name, transform=ax.transAxes, fontsize=6, ha="center", va="center")
    ax.text(anchor[0], anchor[1] + directions["z"][1] + 0.07, "anatomical space",
            transform=ax.transAxes, rotation=90, fontsize=5, ha="center", va="bottom")


def cluster_region_fractions(br, nclus=25):
    """Per k-means cluster: Beryl regions, wedge fractions and cluster colour.

    Same wedges as dmn_bwm.plot_cluster_profile(norm_reg_count=True,
    canonical_order=True): root/void removed, each region's count in the cluster
    divided by its total count, wedges in canonical Beryl order.
    """
    rk = regional_group("kmeans", vers="concat", cv=False, nclus=nclus)
    rb = regional_group("Beryl", vers="concat", cv=False)
    regs = np.asarray(rb["acs"]).astype(str)
    keep = ~np.isin(np.char.lower(regs), ["root", "void"])
    regs, clus, cols = regs[keep], np.asarray(rk["acs"])[keep], np.asarray(rk["cols"])[keep]
    canonical = list(br.id2acronym(np.load(Path(iblatlas.__file__).parent / "beryl.npy"),
                                   mapping="Beryl"))
    total = Counter(regs)
    out = []
    for k in range(nclus):
        counts = Counter(regs[clus == k])
        order = [r for r in canonical if r in counts]
        order += sorted(r for r in counts if r not in set(order))
        w = np.array([counts[r] / total[r] for r in order])
        out.append((order, w / w.sum(), cols[np.flatnonzero(clus == k)[0]]))
    return out


PIE_RMAX = 1.2


def draw_cluster_pies(fig, cell, br, fractions):
    """f: one vector pie per cluster; the 5 largest wedges are labelled radially at
    their centre angle. Label font size is a linear function of wedge fraction,
    shared by all pies: the smallest labelled wedge of any pie gets 3.5 pt, the
    largest wedge of any pie 7 pt."""
    sg = cell.subgridspec(5, 5, hspace=0.42, wspace=0.42)
    labelled = [np.sort(frac)[::-1][:5] for _, frac, _ in fractions]
    f_lo = min(float(f.min()) for f in labelled)
    f_hi = max(float(f.max()) for f in labelled)
    labels, axes, numbers = [], [], []
    for k, (regs, frac, ccol) in enumerate(fractions):
        ax = fig.add_subplot(sg[k // 5, k % 5], projection="polar")
        edges = np.r_[0.0, 2 * np.pi * np.cumsum(frac)]
        colors = region_colors(br, regs)
        # Edge in the face colour hides antialiasing seams between the many thin wedges.
        ax.bar(edges[:-1], np.ones(len(frac)), width=np.diff(edges), bottom=0.0,
               align="edge", color=colors, edgecolor=colors, linewidth=0.15)
        ax.set_ylim(0, PIE_RMAX)  # pie radius 1 fills 1/PIE_RMAX of its cell
        ax.set_axis_off()
        numbers.append(ax.text(-0.18, 1.18, str(k + 1), transform=ax.transAxes, fontsize=6,
                               fontweight="bold", color=ccol, ha="left", va="top"))
        for i in np.argsort(-frac)[:5]:
            # Many neighbouring Beryl regions share an Allen colour; a thin white
            # outline keeps each labelled wedge distinguishable from them.
            ax.bar(edges[i], 1.0, width=edges[i + 1] - edges[i], bottom=0.0, align="edge",
                   fill=False, edgecolor="white", linewidth=0.5, zorder=3)
            mid = edges[i] + np.pi * frac[i]
            size = 3.5 + 3.5 * (frac[i] - f_lo) / (f_hi - f_lo)  # linear in wedge size
            t = ax.text(mid, 1.06, regs[i], rotation_mode="anchor", va="center",
                        fontsize=size, color=colors[i], clip_on=False)
            orient_radially(t, mid)
            t.wedge_mid = mid
            labels.append(t)
        axes.append(ax)
    separate_labels(fig, labels, numbers)
    check_labels_centred(fig, labels)
    return axes[0]


def check_labels_centred(fig, labels, tol_deg=1.0):
    """Assert every label's rendered centre lies on its wedge's middle angle.

    Uses the box Matplotlib renders for each label (independent of the layout
    code): for radial, vertically centred text its centre is on the label's ray.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    worst = 0.0
    for t in labels:
        centre = t.axes.transData.transform((0, 0))
        box = t.get_window_extent(renderer)
        dx, dy = (box.x0 + box.x1) / 2 - centre[0], (box.y0 + box.y1) / 2 - centre[1]
        dev = (np.degrees(np.arctan2(dy, dx)) - np.degrees(t.wedge_mid) + 180) % 360 - 180
        worst = max(worst, abs(dev))
    if worst > tol_deg:
        raise RuntimeError(f"pie label off its wedge middle by {worst:.2f} deg")
    print(f"[f] {len(labels)} labels centred on their wedges (max deviation {worst:.3f} deg)")


def orient_radially(t, theta):
    """Rotate a label along the radius at angle theta, reading left to right."""
    ang = np.degrees(theta) % 360
    flip = 90 < ang < 270
    t.set_rotation(ang + 180 if flip else ang)
    t.set_ha("right" if flip else "left")


def text_quad(t, renderer, pad=0.8):
    """Corners (display units) of a rotated text label, padded by ``pad`` points."""
    rot = t.get_rotation()
    t.set_rotation(0)
    box = t.get_window_extent(renderer)
    t.set_rotation(rot)
    pad *= fig_dpi(t) / 72
    w, h = box.width + 2 * pad, box.height + 2 * pad
    x, y = t.get_transform().transform(t.get_position())
    a = np.radians(rot)
    u, v = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
    a0 = -pad if t.get_ha() == "left" else -w + pad
    return np.array([[x, y] + (a0 + da) * u + db * v for da in (0, w) for db in (-h / 2, h / 2)])[[0, 1, 3, 2]]


def fig_dpi(t):
    return t.figure.dpi


def box_overlap(p, q):
    """Overlap area of the upright bounding boxes of two quads (fallback ranking)."""
    w = min(p[:, 0].max(), q[:, 0].max()) - max(p[:, 0].min(), q[:, 0].min())
    h = min(p[:, 1].max(), q[:, 1].max()) - max(p[:, 1].min(), q[:, 1].min())
    return max(w, 0) * max(h, 0)


def quads_overlap(p, q):
    """Separating-axis test for two convex quadrilaterals."""
    for poly in (p, q):
        for k in range(4):
            edge = poly[(k + 1) % 4] - poly[k]
            axis = np.array([-edge[1], edge[0]])
            pp, qq = p @ axis, q @ axis
            if pp.max() < qq.min() or qq.max() < pp.min():
                return False
    return True


def separate_labels(fig, labels, obstacles=()):
    """Avoid label overlaps, larger labels first.

    Labels stay centred on their wedge's middle angle and only move outward
    along that radius (to at most 2 pie radii). Overlap is tested on the
    rotated text rectangles; each label takes the smallest outward move clear
    of the labels already placed and of the cluster numbers (``obstacles``), or
    else the one with the least overlap area.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    radii = 1.06 + 0.04 * np.arange(24)
    shifts = [0.0]
    moves = sorted(((r, d) for r in radii for d in shifts),
                   key=lambda m: (m[0] - 1.06) + 0.6 * abs(m[1]))
    placed = []
    for o in obstacles:
        b = o.get_window_extent(renderer)
        placed.append(np.array([[b.x0, b.y0], [b.x1, b.y0], [b.x1, b.y1], [b.x0, b.y1]]))
    for t in sorted(labels, key=lambda t: -t.get_fontsize()):
        theta0 = t.get_position()[0]
        best = None
        for radius, shift in moves:
            t.set_position((theta0 + shift, radius))
            orient_radially(t, theta0 + shift)
            quad = text_quad(t, renderer)
            hits = [q for q in placed if quads_overlap(quad, q)]
            area = sum(box_overlap(quad, q) for q in hits) if hits else 0.0
            if best is None or area < best[0]:
                best = (area, radius, shift, quad)
            if not hits:
                break
        _, radius, shift, quad = best
        t.set_position((theta0 + shift, radius))
        orient_radially(t, theta0 + shift)
        placed.append(quad)


def specialization(counts):
    out = {}
    for reg, values in counts.items():
        p = np.asarray(values, float)
        if p.sum() <= 0:
            continue
        p /= p.sum()
        out[reg] = 1 - entropy(p) / np.log(len(p))
    return out


def clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.yaxis.set_major_locator(MaxNLocator(3))


def label(ax, letter, x=-0.04, y=1.0):
    method = getattr(ax, "text2D", ax.text)
    method(x, y, letter, transform=ax.transAxes, fontsize=8,
           fontweight="bold", va="bottom", clip_on=False)


def main():
    configure_style()
    # Both panels use the CV cache's displayed concat_z responses. Rastermap
    # ordering was fitted on concat_z_train; Beryl sorting changes rows only.
    subprocess.run([sys.executable, str(OUT / "regenerate_panel_a_kmeans.py")], check=True)
    subprocess.run([sys.executable, str(OUT / "regenerate_panel_b.py")], check=True)
    subprocess.run([sys.executable, str(OUT / "regenerate_panel_c.py")], check=True)
    # Rastermap-free: a-c use all trials (cv=False, 54,719 neurons).
    panel_a_image = plt.imread(OUT / "panel_a_kmeans_beryl_background.png")
    panel_b_image = plt.imread(OUT / "panel_b_anatomical_order_beryl_background.png")
    n_rows = len(np.load(OUT / "panel_a_kmeans_clusters.npy"))

    br = BrainRegions()
    r = np.load(STACK, allow_pickle=True).flat[0]
    x = np.asarray(r["concat_z"], np.float32)
    umap = r["umap_z"]

    fig = plt.figure(figsize=(183 / 25.4, 183 / 25.4))
    outer = fig.add_gridspec(4, 12, height_ratios=[1.35, 0.85, 1.35, 0.65], hspace=0.30, wspace=0.30,
                             left=0.055, right=0.992, bottom=0.055, top=0.992)

    # a: k-means cluster order, Beryl (anatomical) colours.
    ax = fig.add_subplot(outer[0, 0:4]); label(ax, "a")
    ax.imshow(panel_a_image, origin="upper", aspect="auto", rasterized=True,
              extent=(0, x.shape[1], n_rows, 0))
    ax.set_xlabel("Task-aligned response bins")
    ax.set_ylabel("Neurons (k-means cluster order)")
    ax.set_xticks([0, x.shape[1] // 2, x.shape[1]])
    ax.set_yticks([0, n_rows // 2, n_rows])
    clean(ax); inset_axis(ax)

    # b: exactly the panel-a responses, now in canonical Beryl order.
    ax = fig.add_subplot(outer[0, 4:8]); label(ax, "b")
    ax.imshow(panel_b_image, origin="upper", aspect="auto", rasterized=True,
              extent=(0, x.shape[1], n_rows, 0))
    ax.set_xlabel("Task-aligned response bins"); ax.set_ylabel("Neurons (anatomical order)")
    clean(ax); inset_axis(ax)

    # c: all neurons in six Beryl regions, sorted by k-means cluster (cluster colours).
    sg = outer[0, 8:12].subgridspec(3, 2, hspace=0.34, wspace=0.08)
    for q, reg in enumerate(EXAMPLE_REGIONS):
        ax = fig.add_subplot(sg[q // 2, q % 2])
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0 - 0.014, pos.width, pos.height])
        image = plt.imread(OUT / f"panel_c_{reg.replace(' ', '_')}.png")
        ax.imshow(image, origin="upper", aspect="auto", rasterized=True,
                  interpolation="nearest")
        display_reg = "CUL4,5" if reg == "CUL4 5" else reg
        reg_color = region_colors(br, [reg])[0]
        ax.set_title(f"{display_reg}  n={image.shape[0]:,}", fontsize=5.5, color=reg_color, pad=1)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
        if q == 0: label(ax, "c")

    # d/e: use the exact image assets embedded in the alternate manuscript PDF.
    # White margins and embedded asset headings are cropped; point pixels are unchanged.
    source_umap = {
        "d": source_image("source_alternate_d_umap_beryl.jpg", (25, 20, 547, 410), (428, 571)),
        "e": source_image("source_alternate_e_umap_kmeans.jpg", (25, 20, 547, 410), (428, 571)),
    }
    source_xyz = {
        "d": source_image("source_alternate_d_xyz_beryl.jpg", (200, 175, 630, 652), (726, 817)),
        "e": source_image("source_alternate_e_xyz_kmeans.jpg", (212, 175, 645, 655), (726, 843)),
    }
    low, high = np.min(umap, axis=0), np.max(umap, axis=0)
    pad = (high - low) * 0.02
    umap_extent = (low[0] - pad[0], high[0] + pad[0],
                   low[1] - pad[1], high[1] + pad[1])
    ax = fig.add_subplot(outer[1, 0:3]); label(ax, "d")
    ax.imshow(source_umap["d"], origin="upper", extent=umap_extent,
              aspect="auto", interpolation="nearest", rasterized=True)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2"); clean(ax)
    ax = fig.add_subplot(outer[1, 3:6])
    ax.imshow(source_xyz["d"], aspect="equal", interpolation="nearest", rasterized=True)
    ax.set_axis_off(); xyz_image_triad(ax)
    ax = fig.add_subplot(outer[1, 6:9]); label(ax, "e")
    ax.imshow(source_umap["e"], origin="upper", extent=umap_extent,
              aspect="auto", interpolation="nearest", rasterized=True)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2"); clean(ax)
    ax = fig.add_subplot(outer[1, 9:12])
    ax.imshow(source_xyz["e"], aspect="equal", interpolation="nearest", rasterized=True)
    ax.set_axis_off(); xyz_image_triad(ax)

    # The lower row follows the reference: f runs to the bottom, g above h,
    # and two portrait Swanson maps above j.
    lower = outer[2:4, :].subgridspec(1, 3, width_ratios=(6, 3, 3), wspace=0.18)
    mid = lower[0, 1].subgridspec(2, 1, height_ratios=(0.62, 0.38), hspace=0.17)
    right = lower[0, 2].subgridspec(2, 1, height_ratios=(0.62, 0.38), hspace=0.17)

    # f: Beryl composition of each of the 25 k-means clusters, drawn as vector pies.
    first_pie = draw_cluster_pies(fig, lower[0, 0], br, cluster_region_fractions(br))
    label(first_pie, "f", x=-0.55, y=1.25)  # clear of cluster 1's number

    clus_counts = np.load(CLUS_COUNTS, allow_pickle=True).flat[0]
    dec_counts = np.load(DEC_COUNTS, allow_pickle=True).flat[0]
    spec = specialization(clus_counts); spec_dec = specialization(dec_counts)

    # g: the eight examples and plotting logic used by dmn_bwm.clus_freqs.
    # Inset g a little within its slot so it is not crowded by f and i.
    g_slot = mid[0].subgridspec(3, 3, width_ratios=(0.07, 0.86, 0.07),
                                height_ratios=(0.03, 0.91, 0.06), wspace=0, hspace=0)
    sg = g_slot[1, 1].subgridspec(4, 2, hspace=0.12, wspace=0.1)
    g_axes = np.asarray([[fig.add_subplot(sg[row, col]) for col in range(2)]
                         for row in range(4)])
    clus_freqs(foc="Beryl", clustering="kmeans", nmin=50, nclus=25,
               nclus_rm=100, vers="concat", norm_=True, save_=False,
               single_regions=G_EXAMPLE_REGIONS, axs=g_axes, cv=False,
               label_size=2.3)
    for row in range(4):
        for col in range(2):
            ax = g_axes[row, col]
            for bar in ax.patches:
                center = bar.get_x() + bar.get_width() / 2
                bar.set_width(0.66)
                bar.set_x(center - 0.33)
            ax.set_ylim(0, 0.27)
            ax.set_yticks([0, 0.25])
            ax.tick_params(axis="both", labelsize=4, length=1.5, pad=1)
            if row < 3:
                ax.tick_params(axis="x", labelbottom=False, bottom=False)
            else:
                ax.set_xticks([0, 11, 24], ["1", "12", "25"])
            if col == 1:
                ax.tick_params(axis="y", labelleft=False)
            reg = G_EXAMPLE_REGIONS[col * 4 + row]
            ax.text(0.02, 0.98, f"S={spec[reg]:.2f}", transform=ax.transAxes,
                    ha="left", va="top", fontsize=4.2, color="0.2",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8,
                          "pad": 0.2})
    label(g_axes[0, 0], "g")
    g_top = g_axes[0, 0].get_position().y1
    g_bottom = g_axes[-1, 0].get_position().y0
    g_left = g_axes[0, 0].get_position().x0
    fig.text(g_left - 0.026, (g_top + g_bottom) / 2, "Cluster frequencies",
             rotation=90, ha="center", va="center", fontsize=5.5)

    # i: two tightly paired portrait Swanson maps.
    si = right[0].subgridspec(1, 2, wspace=0.01)
    ax = fig.add_subplot(si[0]); label(ax, "i")
    regs = np.array(list(spec)); vals = np.array([spec[z] for z in regs])
    scale = mpl.colors.Normalize(vmin=float(vals.min()), vmax=float(vals.max()))
    plot_swanson_vector(regs, vals, ax=ax, br=br,
                        orientation="portrait", cmap="magma",
                        vmin=scale.vmin, vmax=scale.vmax, show_cbar=False)
    ax.set_title("Specialization", fontsize=5.5, pad=1)
    ax.set_aspect("equal"); ax.set_axis_off()
    # Match the manuscript's compact legend in the white strip left of the map.
    ax.add_patch(Rectangle((-0.40, 0.92), 0.10, 0.04, transform=ax.transAxes,
                           facecolor="silver", edgecolor="none", clip_on=False))
    ax.text(-0.27, 0.94, "no data", transform=ax.transAxes,
            ha="left", va="center", fontsize=4.5, clip_on=False)
    cax = ax.inset_axes((-0.40, 0.61, 0.09, 0.18))
    colorbar = fig.colorbar(mpl.cm.ScalarMappable(norm=scale, cmap="magma"),
                            cax=cax, ticks=[scale.vmin, scale.vmax], format="%.2f")
    colorbar.ax.tick_params(labelsize=4, length=1.5, pad=1)
    colorbar.outline.set_linewidth(0.4)
    ax.text(-0.40, 0.81, "high", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=4.5, clip_on=False)
    ax.text(-0.40, 0.58, "low", transform=ax.transAxes,
            ha="left", va="top", fontsize=4.5, clip_on=False)
    ax = fig.add_subplot(si[1])
    cosmos_names = np.array(["Isocortex", "OLF", "HPF", "CTXsp", "CNU", "TH", "HY", "MB", "HB", "CB"])
    plot_swanson_vector(cosmos_names, np.arange(len(cosmos_names)), ax=ax, br=br, orientation="portrait",
                        cmap=ListedColormap(region_colors(br, cosmos_names, "Cosmos")), show_cbar=False)
    ax.set_aspect("equal"); ax.set_axis_off()

    # h: distributions of the two specialization measures.
    ax = fig.add_subplot(mid[1]); label(ax, "h")
    ax.hist(list(spec.values()), bins=20, histtype="step", lw=1, color="#2455ff", label="Functional clusters")
    ax.hist(list(spec_dec.values()), bins=20, histtype="step", lw=1, color="#e31a1c", label="Decoding")
    ax.set_xlabel("Specialization"); ax.set_ylabel("Regions"); ax.legend(frameon=False, fontsize=4); clean(ax)

    # j: clustering- versus decoding-based specialization (dmn_bwm.scat_dec_clus);
    # the eight regions of panel g are labelled.
    ax = fig.add_subplot(right[1]); label(ax, "j")
    common = sorted(set(spec) & set(spec_dec))
    xv = np.array([spec[z] for z in common]); yv = np.array([spec_dec[z] for z in common])
    cols = region_colors(br, common)
    ax.scatter(xv, yv, s=6, c=cols, linewidths=0)
    offsets = {"PAA": (-2, 1, "right"), "MOB": (2, 3, "left"), "MEA": (2, -5, "left")}
    for reg in G_EXAMPLE_REGIONS:
        i = common.index(reg)
        dx, dy, ha = offsets.get(reg, (2, 1, "left"))
        ax.annotate(reg, (xv[i], yv[i]), xytext=(dx, dy), textcoords="offset points",
                    fontsize=4.5, color=cols[i], ha=ha, va="bottom")
    rr, pp = pearsonr(xv, yv)
    ax.text(.02, .98, f"$r$={rr:.2f}\n$p$={pp:.2g}", transform=ax.transAxes,
            ha="left", va="top", fontsize=5)
    ax.text(.98, .02, f"{len(common)} regions", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_xlabel("Specialization (clusters)"); ax.set_ylabel("Specialization (decoding)")

    panels = OUT / "anatomy_function_correspondence_panels"
    fig.savefig(panels.with_suffix(".pdf"), dpi=1200)
    fig.savefig(panels.with_suffix(".svg"), dpi=1200)
    fig.savefig(panels.with_suffix(".png"), dpi=600)
    plt.close(fig)
    subprocess.run(["gs", "-q", "-dNOPAUSE", "-dBATCH", "-dSAFER", "-sDEVICE=pdfwrite",
                    "-dCompatibilityLevel=1.5", "-dPDFSETTINGS=/printer",
                    f"-sOutputFile={OUT / 'anatomy_function_correspondence_printer.pdf'}",
                    str(panels.with_suffix(".pdf"))], check=True)
    print(f"Saved {panels}.pdf/.svg/.png and anatomy_function_correspondence_printer.pdf")


if __name__ == "__main__":
    main()
