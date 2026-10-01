#!/usr/bin/env python3
"""Compute (or load) the cached synthetic analyses used by Fig. 5 (and Fig. S10).

- synthetic=True, syn_control=False: NCLUS-cluster k-means basis V (100 for
  Fig. 5, 40 for Fig. S10), real
  coefficients C, i.i.d. marginal-matched synthetic coefficients B, synthetic
  responses B @ V with their Rastermap order (panels a-g, h right, i-k).
- synthetic=True, syn_control=True: responses reconstructed from the real
  coefficients, with their Rastermap order (panel h left).
All caches are written to fig5/cache.
"""

import argparse

from fig5_common import use_private_base


def load(syn_control=False, nclus=100):
    """Basis size nclus_s and the k-means of the synthetic responses both = nclus."""
    d = use_private_base()
    return d.regional_group("kmeans", vers="concat", synthetic=True, cv=False,
                            nclus=nclus, nclus_s=nclus, zsc=True, syn_control=syn_control)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nclus", type=int, default=100)
    args = parser.parse_args()
    for control in (False, True):
        r = load(control, args.nclus)
        print(f"syn_control={control}: C {r['C'].shape}")
