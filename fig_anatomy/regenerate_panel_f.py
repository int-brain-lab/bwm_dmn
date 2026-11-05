#!/usr/bin/env python3
"""Render panel f with the canonical cluster-profile code in dmn_bwm.py."""

import sys
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dmn_bwm import plot_cluster_profile


def overlap_area(a, b):
    width = max(0, min(a.x1, b.x1) - max(a.x0, b.x0))
    height = max(0, min(a.y1, b.y1) - max(a.y0, b.y0))
    return width * height


def separate_region_labels(fig):
    """Move colliding wedge labels outward, preserving their wedge angles."""
    labels = [artist for ax in fig.axes for artist in ax.texts
              if not artist.get_text().isdigit()]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [artist.get_window_extent(renderer=renderer).expanded(1.02, 1.03)
             for artist in labels]
    radii = np.arange(1.04, 1.771, 0.04)
    for _ in range(100):
        pairs = [(i, j) for i, j in combinations(range(len(labels)), 2)
                 if overlap_area(boxes[i], boxes[j]) > 1]
        if not pairs:
            return
        involved = {i for pair in pairs for i in pair}
        best_gain, best_index, best_radius, best_box = 0, None, None, None
        for i in involved:
            artist = labels[i]
            theta, old_radius = artist.get_position()
            origin = artist.axes.transData.transform((theta, old_radius))
            before = sum(overlap_area(boxes[i], boxes[j])
                         for j in range(len(boxes)) if j != i)
            for radius in radii:
                target = artist.axes.transData.transform((theta, radius))
                candidate = boxes[i].translated(*(target - origin))
                after = sum(overlap_area(candidate, boxes[j])
                            for j in range(len(boxes)) if j != i)
                gain = before - after
                if gain > best_gain + 1e-5:
                    best_gain, best_index = gain, i
                    best_radius, best_box = radius, candidate
        if best_index is None:
            break
        artist = labels[best_index]
        artist.set_position((artist.get_position()[0], best_radius))
        boxes[best_index] = best_box
    raise RuntimeError(f"Could not separate all pie labels ({len(pairs)} overlaps remain)")


def main():
    plot_cluster_profile(
        mapping="kmeans",
        vers="concat",
        nclus=25,
        nclus_rm=100,
        cv=False,
        norm_reg_count=True,
        canonical_order=True,
        _full=True,
        pie_only=True,
        savefig=False,
    )
    fig = plt.gcf()
    # Use the available panel area for larger pies, then place the five labels
    # per pie at the nearest non-overlapping radius.
    fig.set_size_inches(9, 8.2)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005,
                        wspace=0.06, hspace=0.52)
    for ax in fig.axes:
        for artist in ax.texts:
            if artist.get_text().isdigit():
                artist.set_fontsize(artist.get_fontsize() * 1.25)
            else:
                artist.set_fontsize(artist.get_fontsize() * 0.92)
    separate_region_labels(fig)
    output = Path(__file__).resolve().parent / "panel_f_cluster_profiles.png"
    fig.savefig(output, dpi=800, facecolor="white", bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
