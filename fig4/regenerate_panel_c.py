#!/usr/bin/env python3
"""Render full-resolution regional subsets of Fig. 4a's Rastermap raster."""

from pathlib import Path

import matplotlib as mpl
import numpy as np
from PIL import Image
from iblatlas.regions import BrainRegions


from fig4_common import OUT, ROOT
REGIONS = ("CP", "MRN", "ZI", "MOp", "CA1", "CUL4 5")
N_COLOR_BANDS = 100


def main() -> None:
    r = np.load(ROOT / "concat_cvTrue.npy", allow_pickle=True).item()
    responses = np.asarray(r["concat_z"], dtype=np.float32)
    order = np.asarray(r["isort"], dtype=np.intp)
    if len(order) != len(responses) or not np.array_equal(np.sort(order), np.arange(len(order))):
        raise ValueError("The saved Rastermap order is not a permutation of the CV neurons")

    beryl = np.asarray(BrainRegions().id2acronym(r["ids"], mapping="Beryl"))
    # The original cache preserved the Rastermap order but not its cluster IDs.
    # Color 100 successive bands of that saved order, retaining exactly panel a's
    # neuron order. A reversed rainbow puts the high end of the palette at top.
    rank = np.empty(len(order), dtype=np.int32)
    rank[order] = np.arange(len(order))
    bands = np.minimum(rank * N_COLOR_BANDS // len(order), N_COLOR_BANDS - 1)
    palette = mpl.colormaps["turbo"](np.linspace(1, 0, N_COLOR_BANDS))[:, :3]

    for region in REGIONS:
        selected = order[beryl[order] == region]
        if not len(selected):
            raise ValueError(f"No CV neurons found in Beryl region {region}")
        ink = np.clip(responses[selected], 0, 2) / 2
        background = palette[bands[selected]] * 0.99 + 0.01
        rgb = background[:, None, :] * (1 - ink[..., None])
        image = Image.fromarray(np.uint8(np.round(np.clip(rgb, 0, 1) * 255)), "RGB")
        path = OUT / f"panel_c_{region.replace(' ', '_')}.png"
        image.save(path)
        print(f"{region}: {len(selected):,} neurons; {path.name}; {image.width}x{image.height}")


if __name__ == "__main__":
    main()
