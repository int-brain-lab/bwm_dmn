#!/usr/bin/env python3
"""Fig. 3c: all neurons of six example Beryl regions (all trials), each region's
neurons sorted by k-means cluster (25 clusters, as in Fig. 3a), row background =
k-means cluster colour, ink = z-scored response clipped to [0, 2] (as in
dmn_bwm.plot_rastermap). Full resolution: every neuron and all 1,872 bins.
"""

import numpy as np
from matplotlib.colors import to_rgb
from PIL import Image

from fig3_common import OUT, use_private_base

REGIONS = ("CP", "MRN", "ZI", "MOp", "CA1", "CUL4 5")


def main() -> None:
    d = use_private_base()
    r = d.regional_group("kmeans", vers="concat", cv=False, nclus=25)
    X = np.asarray(r["concat_z"], dtype=np.float32)
    acs, beryl = np.asarray(r["acs"]), np.asarray(r["Beryl"])
    cols = np.array([to_rgb(c) for c in r["cols"]], dtype=np.float32)
    for region in REGIONS:
        sel = np.flatnonzero(beryl == region)
        if not sel.size:
            raise ValueError(f"No neurons found in Beryl region {region}")
        sel = sel[np.argsort(acs[sel], kind="stable")]
        ink = np.clip(X[sel], 0, 2) / 2
        background = cols[sel] * 0.99 + 0.01
        rgb = background[:, None, :] * (1 - ink[..., None])
        image = Image.fromarray(np.uint8(np.round(np.clip(rgb, 0, 1) * 255)), "RGB")
        path = OUT / f"panel_c_{region.replace(' ', '_')}.png"
        image.save(path)
        print(f"{region}: {sel.size:,} neurons; {path.name}")


if __name__ == "__main__":
    main()
