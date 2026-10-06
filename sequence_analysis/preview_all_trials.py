#!/usr/bin/env python3
"""Preview (not a figure panel): all-trial feature vectors X (cv=False, 54,719
neurons) sorted by a Rastermap fit on X itself, with the Methods' parameters
(dmn_bwm.regional_group(mapping="rm", cv=False); cached in fig3/cache).
"""

import matplotlib.pyplot as plt
import numpy as np

from seq_common import OUT, PREV, use_private_base
from make_figure3 import configure_style, raster_panel, MM

ROW_BLOCK = 5


def main():
    d = use_private_base()
    d.plot_rastermap(vers="concat", feat="concat_z", mapping="rm", sort_method="rastermap",
                     bg=False, img_only=True, cv=False, zsc=True, vmax=1.5, bounds=False,
                     rerun=False, clsfig=False)
    fig = plt.gcf()
    rgba = np.asarray(fig.axes[0].images[0].get_array(), dtype=np.float32)
    plt.close(fig)
    n = rgba.shape[0] // ROW_BLOCK * ROW_BLOCK
    image = np.clip(rgba[:n].reshape(-1, ROW_BLOCK, *rgba.shape[1:]).mean(1)[..., :3], 0, 1)
    r = d.regional_group("rm", vers="concat", cv=False)
    clusters = np.asarray(r["acs"])[np.asarray(r["isort"])]
    print(f"{clusters.size:,} neurons, {len(np.unique(clusters))} clusters, "
          f"{np.count_nonzero(np.diff(clusters)) + 1} blocks along the order")

    configure_style()
    fig = plt.figure(figsize=(92 * MM, 120 * MM))
    raster_panel(fig, [0.13, 0.05, 0.74, 0.78], image, r, clusters,
                 "all trials (X), sorted by their own Rastermap fit")
    out = PREV / "preview_X_rastermap_all_trials.png"
    fig.savefig(out, dpi=300, facecolor="white")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
