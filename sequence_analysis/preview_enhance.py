#!/usr/bin/env python3
"""Compare display mappings for the train/test rasters (canonical odd-trial fit).

All mappings are per neuron (row), display-only, and applied identically to the
odd (train) and even (test) data:
  linear     : z clipped to [0, 1.5]                                   (Methods)
  robust     : (x - row median) / (row 99th pct - row median), [0, 1]
  threshold  : robust, values below the row's 90th percentile set to 0
  peak-norm  : robust, then each PETH-type window normalised to its own max
               (sparse cells: shows where in the window the cell fires)
No row averaging (rows are subsampled for display by the renderer).
"""

import matplotlib.pyplot as plt
import numpy as np

from seq_common import PREV, use_private_base


def robust(X):
    med = np.median(X, 1, keepdims=True)
    hi = np.percentile(X, 99, 1, keepdims=True)
    return np.clip((X - med) / (hi - med + 1e-6), 0, 1)


def threshold(X, q=90):
    R = robust(X)
    return np.where(R >= np.percentile(R, q, 1, keepdims=True), R, 0)


def window_norm(X, lens):
    R = robust(X)
    out, s = np.empty_like(R), 0
    for n in lens:
        w = R[:, s:s + n]
        out[:, s:s + n] = w / (w.max(1, keepdims=True) + 1e-6)
        s += n
    return out ** 2


MAPS = {"linear": lambda X, lens: np.clip(X, 0, 1.5) / 1.5, "robust": lambda X, lens: robust(X),
        "threshold 90th pct": lambda X, lens: threshold(X),
        "window peak-norm (squared)": window_norm}


def main():
    d = use_private_base()
    r = d.regional_group("rm", vers="concat", cv=True)
    order = np.asarray(r["isort"])
    lens = list(r["len"].values())
    data = {"train (odd)": np.asarray(r["concat_z_train"], np.float32)[order],
            "test (even)": np.asarray(r["concat_z"], np.float32)[order]}
    fig, axs = plt.subplots(len(MAPS), 2, figsize=(12, 5.5 * len(MAPS)))
    for i, (mname, fn) in enumerate(MAPS.items()):
        for j, (name, X) in enumerate(data.items()):
            ax = axs[i, j]
            ax.imshow(1 - fn(X, lens), cmap="gray", vmin=0, vmax=1, aspect="auto",
                      interpolation="antialiased", rasterized=True)
            for b in np.cumsum(lens)[:-1]:
                ax.axvline(b, color="tab:red", lw=0.3, alpha=0.5)
            ax.set_title(f"{name}: {mname}", fontsize=9)
            ax.set_xticks([])
    fig.tight_layout()
    out = PREV / "preview_enhance.png"
    fig.savefig(out, dpi=110)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
