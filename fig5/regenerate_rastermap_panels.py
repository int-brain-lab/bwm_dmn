#!/usr/bin/env python3
"""Render the two Rastermap images of Fig. 5h with dmn_bwm.plot_rastermap.

left:  responses reconstructed from the real coefficients (syn_control=True)
right: synthetic responses B @ V (syn_control=False)
Each is sorted by Rastermap fitted on the displayed responses. The RGBA image
plot_rastermap draws is captured, averaged over blocks of ROW_BLOCK neurons and
saved as PNG.
"""

import matplotlib

matplotlib.use("Agg")  # plot_rastermap calls plt.ion(); never open windows
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fig5_common import OUT, use_private_base

ROW_BLOCK = 5


def capture(d, syn_control, nclus):
    d.plot_rastermap(vers="concat", feat="concat_z", mapping="kmeans", nclus=nclus,
                     nclus_s=nclus, synthetic=True, syn_control=syn_control,
                     sort_method="rastermap", bg=False, img_only=True, cv=False,
                     zsc=True, vmax=2, bounds=False, rerun=False, clsfig=False)
    fig = plt.gcf()
    rgba = np.asarray(fig.axes[0].images[0].get_array(), dtype=np.float32)
    plt.close(fig)
    n = rgba.shape[0] // ROW_BLOCK * ROW_BLOCK
    return np.clip(rgba[:n].reshape(-1, ROW_BLOCK, *rgba.shape[1:]).mean(axis=1)[..., :3], 0, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nclus", type=int, default=100, help="k-means basis size")
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    d = use_private_base()
    for control, name in ((True, "real_reconstruction"), (False, "synthetic")):
        path = args.out_dir / f"panel_h_rastermap_{name}.png"
        plt.imsave(path, capture(d, control, args.nclus))
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
