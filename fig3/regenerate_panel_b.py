#!/usr/bin/env python3
"""Fig. 3b: the same neurons and responses as Fig. 3a (all trials, 54,719 neurons),
in canonical Beryl order, row background = Beryl colour.

dmn_bwm.plot_rastermap(mapping="Beryl", sort_method="acs", bg=True, cv=False); the
RGBA image it draws is captured, ROW_BLOCK neurons averaged, saved as PNG.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fig3_common import OUT, use_private_base

ROW_BLOCK = 5


def main() -> None:
    d = use_private_base()
    d.plot_rastermap(vers="concat", feat="concat_z", mapping="Beryl", sort_method="acs",
                     bg=True, bg_bright=0.99, img_only=True, interp="antialiased",
                     cv=False, zsc=True, vmax=2, bounds=False, rerun=False, clsfig=False)
    fig = plt.gcf()
    rgba = np.asarray(fig.axes[0].images[0].get_array(), dtype=np.float32)
    plt.close(fig)
    n = rgba.shape[0] // ROW_BLOCK * ROW_BLOCK
    image = np.clip(rgba[:n].reshape(-1, ROW_BLOCK, *rgba.shape[1:]).mean(1)[..., :3], 0, 1)
    path = OUT / "panel_b_anatomical_order_beryl_background.png"
    plt.imsave(path, image)
    print(f"Saved {path} ({rgba.shape[0]:,} neurons)")


if __name__ == "__main__":
    main()
