#!/usr/bin/env python3
"""Step 3: shared caches that several figures read from $DMN_DATA.

- 25-cluster k-means on all trials (kmeans_concat_cvFalse_n25_...npy): the
  canonical k-means clustering and sorting of Figs. 2-4.
- Region-by-cluster counts (counts/cf_Beryl_...npy) and decoding-based counts
  (counts/cf_dec_...npy) for Fig. 3f-j. The decoding counts need the BWM
  decoding tables in $DMN_DATA/bwm_decoding/ (see DATA_REQUIREMENTS.md).
Existing caches are reused.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import dmn_bwm as d  # noqa: E402


def main():
    r = d.regional_group("kmeans", vers="concat", cv=False, nclus=25)
    print(f"k-means (25 clusters, all trials): {len(r['acs'])} neurons", flush=True)
    for foc in ("Beryl", "dec"):
        d.clus_freqs(foc=foc, clustering="kmeans", nmin=50, nclus=25, cv=False, get_res=True)
        print(f"counts for {foc} written to {d.pth_dmn / 'counts'}", flush=True)


if __name__ == "__main__":
    main()
