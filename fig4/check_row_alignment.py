#!/usr/bin/env python3
"""Regression check for the row alignment of panels j-k.

C = X[r['C_rows']] @ V.T, so C's rows are not in stack order. The manuscript
version grouped C's PC0 by r['Beryl'] (stack order) and r['acs'] (k-means on
the synthetic responses). dmn_bwm.synthetic_row_labels gives each row its own
region and real-data k-means cluster. This prints panels j-k both ways and
verifies the row order directly.
"""

import argparse

import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA

from compute_synthetic import load
from fig4_common import use_private_base


def se_per_group(score, labels, exclude=("root", "void"), nmin=20):
    out = []
    for lab in np.unique(labels):
        if str(lab) in exclude:
            continue
        v = score[labels == lab]
        if v.size >= nmin:
            out.append(np.std(v, ddof=1) / np.sqrt(v.size))
    return np.asarray(out)


def emd(a, b):
    both = np.concatenate([a, b])
    span = both.max() - both.min()
    return wasserstein_distance(a, b) / span if span > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nclus", type=int, default=100, help="k-means basis size")
    nclus = parser.parse_args().nclus
    d = use_private_base()
    r = load(nclus=nclus)
    C = np.asarray(r["C"], float)[:, :nclus]
    X = np.asarray(d.regional_group("Beryl", vers="concat", cv=False)["concat_z"], float)
    rows = np.asarray(r["C_rows"])
    probe = np.array([0, 1, 1000, 30000])
    assert np.allclose(C[probe], X[rows[probe]] @ np.asarray(r["V"]).T), "C_rows does not match C"
    assert not np.allclose(C[probe], X[probe] @ np.asarray(r["V"]).T), "C unexpectedly in stack order"
    print("C == X[C_rows] @ V.T: ok")

    pc0 = PCA(n_components=1).fit(C).transform(C)[:, 0]
    for name, (reg, km) in [
        ("manuscript pairing (r['Beryl'], r['acs'])", (np.asarray(r["Beryl"]), np.asarray(r["acs"]))),
        ("fixed (synthetic_row_labels)", d.synthetic_row_labels(r)),
    ]:
        rng = np.random.default_rng(0)
        x_reg, x_rand, x_km = (se_per_group(pc0, reg), se_per_group(pc0, rng.permutation(reg)),
                               se_per_group(pc0, km))
        print(f"{name}\n  j: Beryl vs random EMD={emd(x_reg, x_rand):.3f}\n"
              f"  k: Beryl vs KMeans EMD={emd(x_reg, x_km):.3f}, "
              f"median SE KMeans={np.median(x_km):.0f}")


if __name__ == "__main__":
    main()
