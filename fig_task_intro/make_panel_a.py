#!/usr/bin/env python3
"""Fig. 1 (draft panel d): How homogeneous is task-related neural response diversity?

Left brain: every region task specialized; right brain: every region mixed.

The abstract brain drawing of the earlier Fig. 1a (outline and sulci only,
elements/a_brain_outline.svg) is split into 12 compartments ("regions"): each
grows from a seed until it meets a sulcus line or a neighbouring compartment
(watershed), so open-ended sulci still give closed regions. Each region is
stippled with dots ("neurons"):
  left   one response type per region (distinct colours),
  right  response types sampled from the real 25-cluster proportions of one of
         the 12 most densely recorded Beryl regions (k-means, all trials; the
         cluster colours of Figs. 2-3), one recorded region per compartment.
Callouts define the dots (left brain only): a highlighted dot (black ring) in
each of two regions is linked to a mini feature vector under the brain, the real
mean feature vector (concat_z, all trials) of a k-means cluster in the region's
colour (CALLOUTS: a stimulus-responsive and a movement cluster of Fig. 2). No
cluster numbers (clusters are introduced in Fig. 2).
Writes elements/panel_a.svg (+ .pdf, .png): both brains side by side, left
specialized, right observed, at final size; vector throughout (the original
outline paths, dots as circles, text). Also the raster versions
elements/a_specialized.png, a_observed.png and a_preview.png.
"""

import os
import re
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from iblatlas.regions import BrainRegions
from scipy import ndimage
from skimage.segmentation import watershed

HERE = Path(__file__).resolve().parent
EL = HERE / "elements"
ROOT = Path(os.environ.get("DMN_DATA", Path.home() / "dmn"))
DPI = 1500                 # render of the outline; seeds below are in this pixel grid
SPACING, RADIUS = 24, 8.5  # dot grid spacing and dot radius (px)
# seed (x, y) per compartment, in the DPI render (1524 x 1239 px)
SEEDS = {"frontal_sup": (440, 130), "frontal": (240, 420), "precentral": (660, 300),
         "parietal": (940, 300), "parietal_post": (1280, 440), "temporal_sup": (720, 650),
         "frontal_inf": (220, 690), "temporal": (840, 820), "occipital": (1380, 700),
         "cerebellum": (1160, 1020), "brainstem": (1074, 1190), "insula": (1080, 640)}
BRAINSTEM_GAP = (slice(-9, None), slice(1015, 1131))  # closes the drawing's open brainstem
PT = 0.3528                # mm per point
FONT = "Arial, 'Liberation Sans', Helvetica, sans-serif"
BRAIN_W, GAP, TITLE_H = 28.0, 3.0, 5.2   # final size (mm): brain width, gap, title band
# one title spanning both brains
TITLE = ("How homogeneous is task-related", "neural response diversity?")
# left-brain callouts: (region, k-means cluster as numbered in Fig. 2); each cluster's
# colour equals its region's colour in the left brain (blue: stimulus, green: movement)
CALLOUTS = (("frontal_sup", 21), ("precentral", 5))
TRACE_H, TRACE_GAP, BAND_TOP = 2.6, 0.9, 1.2  # mini feature vectors (mm)


def cluster_colors(n=25):
    """dmn_bwm's k-means colours: tab20 by cluster index, cluster index 1 pink."""
    cmap = mpl.colormaps["tab20"]
    cols = [cmap(i % 20)[:3] for i in range(n)]
    cols[1] = mpl.colors.to_rgb("pink")
    return np.array(cols)


def render_outline():
    png = EL / "a_brain_outline_render.png"
    subprocess.run(["inkscape", str(EL / "a_brain_outline.svg"), "--export-type=png",
                    f"--export-dpi={DPI}", "--export-background=white", "--export-area-drawing",
                    f"--export-filename={png}"], check=True, capture_output=True)
    ink = plt.imread(png)[..., :3].mean(2) < 0.6
    png.unlink()
    return ink


def compartments(ink):
    closed = ink.copy()
    closed[BRAINSTEM_GAP] = True
    lab, _ = ndimage.label(np.pad(~closed, 1, constant_values=True))
    inside = ~(lab == lab[0, 0])[1:-1, 1:-1] & ~closed
    markers = np.zeros(ink.shape, int)
    for k, (x, y) in enumerate(SEEDS.values(), 1):
        assert inside[y, x], list(SEEDS)[k - 1]
        markers[y - 3:y + 4, x - 3:x + 4] = k
    return watershed(np.zeros(ink.shape), markers, mask=inside)


def stipple(seg, ink, colour_of, rng):
    """White image with jittered-grid dots inside each compartment, ink lines on top;
    also the dots as (x, y, rgb) in render pixels."""
    h, w = seg.shape
    img = np.ones((h, w, 3))
    dots = []
    r = int(np.ceil(RADIUS))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    stamp = xx ** 2 + yy ** 2 <= RADIUS ** 2
    for k in range(1, seg.max() + 1):
        room = ndimage.binary_erosion(seg == k, iterations=r + 2)   # dots stay clear of the lines
        gy, gx = np.mgrid[SPACING // 2:h:SPACING, SPACING // 2:w:SPACING]
        gy = (gy + rng.integers(-5, 6, gy.shape)).clip(0, h - 1).ravel()
        gx = (gx + rng.integers(-5, 6, gx.shape)).clip(0, w - 1).ravel()
        ok = room[gy, gx]
        cols = colour_of(k, ok.sum())
        for (y, x), c in zip(zip(gy[ok], gx[ok]), cols):
            sl = (slice(y - r, y + r + 1), slice(x - r, x + r + 1))
            img[sl][stamp] = c
            dots.append((x, y, c, k))
    img[ink] = 0
    return img, dots


def write_svg(dots_left, dots_right, shape, callouts):
    """Both brains side by side at final size: the outline as a reused symbol, dots as circles."""
    src = (EL / "a_brain_outline.svg").read_text()
    q = subprocess.run(["inkscape", str(EL / "a_brain_outline.svg"), "--query-x", "--query-y",
                        "--query-width", "--query-height"], capture_output=True, text=True, check=True)
    # drawing bbox in px (96 dpi) -> the outline's user units (viewBox width / page width)
    vb = [float(v) for v in re.search(r'viewBox="([^"]+)"', src).group(1).split()]
    page_mm = float(re.search(r'width="([\d.]+)mm"', src).group(1))
    unit = (25.4 / 96) / (page_mm / vb[2])
    bx, by, bw, bh = (float(v) * unit for v in q.stdout.split())
    inner = src[src.index(">", src.index("<svg")) + 1: src.rindex("</svg>")]
    h_px, w_px = shape
    sx, sy = bw / w_px, bh / h_px                  # render pixel -> outline user unit
    brain_h = BRAIN_W * bh / bw
    n_tr = max(len(c) for c in callouts) or 1
    band = BAND_TOP + n_tr * TRACE_H + (n_tr - 1) * TRACE_GAP + 0.6
    W, H = 2 * BRAIN_W + GAP, TITLE_H + brain_h + band
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
           f'width="{W:.2f}mm" height="{H:.2f}mm" viewBox="0 0 {W:.3f} {H:.3f}">',
           f'<defs><symbol id="brain" viewBox="{bx:.4f} {by:.4f} {bw:.4f} {bh:.4f}" '
           f'overflow="visible">{inner}</symbol></defs>']
    for j, line in enumerate(TITLE):
        y = TITLE_H - 1.0 - (len(TITLE) - 1 - j) * 2.3
        out.append(f'<text x="{W / 2:.2f}" y="{y:.2f}" font-family="{FONT}" '
                   f'font-size="{6 * PT:.3f}" text-anchor="middle">{line}</text>')
    for k, dots in enumerate((dots_left, dots_right)):
        x0 = k * (BRAIN_W + GAP)
        out.append(f'<svg x="{x0:.3f}" y="{TITLE_H:.3f}" width="{BRAIN_W:.3f}" height="{brain_h:.3f}" '
                   f'viewBox="{bx:.4f} {by:.4f} {bw:.4f} {bh:.4f}">')
        rad = RADIUS * sx
        for x, y, c, _ in dots:
            col = "#%02x%02x%02x" % tuple(int(round(255 * v)) for v in c)
            out.append(f'<circle cx="{bx + (x + 0.5) * sx:.3f}" cy="{by + (y + 0.5) * sy:.3f}" '
                       f'r="{rad:.3f}" fill="{col}"/>')
        out.append(f'<use xlink:href="#brain" href="#brain" x="{bx:.4f}" y="{by:.4f}" '
                   f'width="{bw:.4f}" height="{bh:.4f}"/></svg>')
        # callouts: ringed dot, leader line, mini feature vector in the dot's colour
        scale = BRAIN_W / bw
        for j, ((px, py), col, trace) in enumerate(callouts[k]):
            dx, dy = x0 + (px + 0.5) * sx * scale, TITLE_H + (py + 0.5) * sy * scale
            ty = TITLE_H + brain_h + BAND_TOP + j * (TRACE_H + TRACE_GAP)
            tx0, tx1 = x0 + 0.8, x0 + BRAIN_W - 0.3
            hexc = "#%02x%02x%02x" % tuple(int(round(255 * v)) for v in col)
            t = (trace - trace.min()) / (np.ptp(trace) + 1e-12)
            out.append(f'<polyline points="{dx:.2f},{dy:.2f} {tx0 - 0.3:.2f},{ty + TRACE_H * (1 - t[0]):.2f}" '
                       f'fill="none" stroke="#8a8a8a" stroke-width="0.1"/>')
            out.append(f'<circle cx="{dx:.3f}" cy="{dy:.3f}" r="{RADIUS * sx * scale * 1.35:.3f}" '
                       f'fill="{hexc}" stroke="#000" stroke-width="0.12"/>')
            xs = np.linspace(tx0, tx1, t.size)
            ys = ty + TRACE_H * (1 - t)
            pts = " ".join(f"{a:.2f},{b:.2f}" for a, b in zip(xs, ys))
            out.append(f'<polyline points="{pts}" fill="none" stroke="{hexc}" stroke-width="0.25" '
                       f'stroke-linejoin="round"/>')
    out.append("</svg>")
    path = EL / "panel_a.svg"
    path.write_text("\n".join(out) + "\n")
    for ext, extra in (("pdf", []), ("png", ["--export-dpi=600", "--export-background=white"])):
        subprocess.run(["inkscape", str(path), f"--export-type={ext}",
                        f"--export-filename={path.with_suffix('.' + ext)}", *extra],
                       check=True, capture_output=True)
    print(f"Saved {path} (+ .pdf, .png), {W:.1f} x {H:.1f} mm")


def main():
    rng = np.random.default_rng(0)
    ink = render_outline()
    seg = compartments(ink)
    cols = cluster_colors()

    # real cluster proportions of the 12 most densely recorded Beryl regions
    br = BrainRegions()
    r = np.load(ROOT / "concat_cvFalse.npy", allow_pickle=True).flat[0]
    labels = np.asarray(np.load(ROOT / "kmeans_concat_cvFalse_n25_nclusrm100_zsc1.npy",
                                allow_pickle=True).flat[0]["kmeans_labels"], int)
    beryl = np.array(br.id2acronym(r["ids"], mapping="Beryl"))
    regs, n = np.unique(beryl[~np.isin(beryl, ["root", "void"])], return_counts=True)
    top = regs[np.argsort(-n)][:len(SEEDS)]
    props = [np.bincount(labels[beryl == g], minlength=25) / (beryl == g).sum() for g in top]
    print("regions used for the observed proportions:", ", ".join(top))

    # specialized: one distinct colour per compartment (the ten dark tab20 colours,
    # pink and light orange: no two compartments share a hue)
    special = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1, 3]
    left, dots_l = stipple(seg, ink, lambda k, m: np.repeat(cols[special[k - 1]][None], m, 0), rng)
    right, dots_r = stipple(seg, ink, lambda k, m: cols[rng.choice(25, m, p=props[k - 1])], rng)

    EL.mkdir(exist_ok=True)
    plt.imsave(EL / "a_specialized.png", left)
    plt.imsave(EL / "a_observed.png", right)
    fig, axs = plt.subplots(1, 2, figsize=(7, 3))
    for ax, im in zip(axs, (left, right)):
        ax.imshow(im); ax.set_axis_off()
    fig.suptitle(" ".join(TITLE), fontsize=9)
    fig.savefig(EL / "a_preview.png", dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved {EL}/a_specialized.png, a_observed.png, a_preview.png ({seg.max()} regions)")
    # callouts: mean feature vectors of the clusters, downsampled for a small trace
    X = np.asarray(r["concat_z"], np.float32)
    def mean_vec(c):  # c: cluster number as in Fig. 2 (1-based)
        v = X[labels == c - 1].mean(0)
        return v[: v.size // 4 * 4].reshape(-1, 4).mean(1)
    def pick(dots, region):
        """A dot near the middle of a region."""
        k = list(SEEDS).index(region) + 1
        idx = [i for i, d in enumerate(dots) if d[3] == k]
        c = np.median([[dots[i][0], dots[i][1]] for i in idx], axis=0)
        return min(idx, key=lambda i: (dots[i][0] - c[0]) ** 2 + (dots[i][1] - c[1]) ** 2)
    left_callouts = []
    for region, c in CALLOUTS:
        i = pick(dots_l, region)
        assert np.allclose(dots_l[i][2], cols[c - 1]), (region, c)   # colours must match
        left_callouts.append(((dots_l[i][0], dots_l[i][1]), cols[c - 1], mean_vec(c)))
    callouts = (left_callouts, [])
    write_svg(dots_l, dots_r, seg.shape, callouts)


if __name__ == "__main__":
    main()
