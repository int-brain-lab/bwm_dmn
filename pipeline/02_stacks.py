#!/usr/bin/env python3
"""Step 2: stack the per-insertion bundles into the two analysis stacks.

concat_cvFalse.npy  all trials per condition, z-scored ("X"); used by the k-means
                    analyses (Figs. 2-4).
concat_cvTrue.npy   odd/even trial split per condition (Methods): concat_z_train
                    (odd) and concat_z (even), one canonical Rastermap fit on the
                    odd trials (isort, rm_labels); plus row-aligned X_odd, X_even
                    and X (all trials, matched by neuron UUID). Used by everything
                    involving Rastermap (Fig. 5, SI).
Existing stacks are kept unless --force. The CV Rastermap fit takes ~1 h.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import dmn_bwm as d  # noqa: E402

CVF, CVT = d.DMN_BASE / "concat_cvFalse.npy", d.DMN_BASE / "concat_cvTrue.npy"


def add_x_arrays():
    """Add row-aligned X_odd, X_even and X (all trials, matched by UUID) to the CV stack."""
    allstack = np.load(CVF, allow_pickle=True).flat[0]
    cv = np.load(CVT, allow_pickle=True).flat[0]
    row = {u: i for i, u in enumerate(np.asarray(allstack["uuids"]).astype(str))}
    u = np.asarray(cv["uuids"]).astype(str)
    missing = [x for x in u if x not in row]
    if missing:
        raise RuntimeError(f"{len(missing)} CV neurons not in the all-trial stack")
    cv["X_odd"] = np.asarray(cv["concat_z_train"], np.float32)
    cv["X_even"] = np.asarray(cv["concat_z"], np.float32)
    cv["X"] = np.asarray(allstack["concat_z"], np.float32)[[row[x] for x in u]]
    np.save(CVT, cv, allow_pickle=True)
    print(f"CV stack: added X_odd, X_even, X for {len(u)} neurons", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="rebuild existing stacks")
    args = ap.parse_args()
    if args.force or not CVF.exists():
        d.stack_concat(vers="concat", cv=False)
    if args.force or not CVT.exists():
        d.stack_concat(vers="concat", cv=True)
    cv = np.load(CVT, allow_pickle=True).flat[0]
    if args.force or not all(k in cv for k in ("X", "X_odd", "X_even")):
        add_x_arrays()


if __name__ == "__main__":
    main()
