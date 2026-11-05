#!/usr/bin/env python3
"""Fig. 1d brain with the recorded insertions, drawn from the data (vector).

Dorsal view of the Allen mouse brain (iblatlas, 25 um): the outline of the brain
and of the Cosmos divisions at the dorsal surface, and one line per insertion
(515, from concat_cvFalse.npy) through the positions of its recorded neurons
(principal axis of the neuron positions, between the extreme neurons). Replaces
the low-resolution rendering in the earlier figure. Writes
elements/d_insertions_dorsal.svg/.pdf/.png.
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from iblatlas.atlas import AllenAtlas
from iblatlas.regions import BrainRegions

ROOT = Path(os.environ.get("DMN_DATA", Path.home() / "dmn"))
OUT = Path(__file__).resolve().parent / "elements"
MM = 1 / 25.4


def tracks(xyz, pid):
    """Per insertion: the two extreme neuron positions along the principal axis."""
    out = []
    for p in np.unique(pid):
        P = xyz[pid == p]
        P = P[np.isfinite(P).all(1)]
        if len(P) < 2:
            continue
        c = P.mean(0)
        u = np.linalg.svd(P - c, full_matrices=False)[2][0]
        s = (P - c) @ u
        out.append(np.stack([c + s.min() * u, c + s.max() * u]))
    return np.array(out)


def main():
    OUT.mkdir(exist_ok=True)
    r = np.load(ROOT / "concat_cvFalse.npy", allow_pickle=True).flat[0]
    T = tracks(np.asarray(r["xyz"], float), np.asarray(r["pid"]).astype(str))

    ba = AllenAtlas(25)
    br = BrainRegions()
    # Cosmos label of the dorsal-most voxel of each (ML, AP) column
    lab = ba.regions.mappings["Cosmos"][ba.label]               # (AP, ML, DV) indices into br
    inside = lab > 0
    first = np.where(inside.any(2), inside.argmax(2), -1)        # dorsal-most voxel
    ap, ml = np.indices(first.shape)
    top = np.where(first >= 0, lab[ap, ml, np.clip(first, 0, None)], 0)
    xs, ys = ba.bc.xscale, ba.bc.yscale                          # ML (x) and AP (y), metres

    fig, ax = plt.subplots(figsize=(55 * MM, 70 * MM))
    rgb = br.rgb[top] / 255.0
    rgba = np.concatenate([rgb, (top > 0)[..., None] * 0.35], axis=2)
    ax.imshow(rgba, extent=(xs[0], xs[-1], ys[-1], ys[0]), origin="upper",
              interpolation="nearest", rasterized=True, zorder=0)
    ax.contour(xs, ys, top, levels=np.unique(top)[1:] - 0.5, colors="0.55", linewidths=0.25, zorder=1)
    ax.contour(xs, ys, (top > 0).astype(float), levels=[0.5], colors="0.15", linewidths=0.5, zorder=2)
    for a, b in T:
        ax.plot([a[0], b[0]], [a[1], b[1]], color="0.2", lw=0.35, alpha=0.75,
                solid_capstyle="round", zorder=3)
    ax.set_aspect("equal")
    ax.set_xlim(xs.min(), xs.max())
    ax.set_ylim(ys.min(), ys.max())
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    for ext in ("svg", "pdf"):
        fig.savefig(OUT / f"d_insertions_dorsal.{ext}", dpi=600, transparent=True)
    fig.savefig(OUT / "d_insertions_dorsal.png", dpi=600, facecolor="white")
    print(f"{len(T)} insertions; saved {OUT / 'd_insertions_dorsal'}.svg/.pdf/.png")


if __name__ == "__main__":
    main()
