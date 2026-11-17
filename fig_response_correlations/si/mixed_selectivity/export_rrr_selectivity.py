#!/usr/bin/env python3
"""Export the per-neuron RRR selectivity used by rrr_epairs.py (Fig. 4e, right) to a compact file.

Reads the full rank-5 (lambda = 75) RRR fit of Posani et al. (1.7 GB model file plus per-session
features, see DATA_REQUIREMENTS.md) and writes $DMN_DATA/rrr_selectivity_rank5_l275.npz with,
for the 38,853 neurons with Delta R^2 > 0.015: time-summed absolute and signed coefficients for the
8 task variables, session index and Allen acronym. rrr_epairs.py reads this file when present.
Run (iblenv): python export_rrr_selectivity.py
"""

import numpy as np

import rrr_epairs as RE


def main():
    D = RE.load_rrr_full()
    np.savez_compressed(RE.COMPACT, abs=D["abs"].astype(np.float64), signed=D["signed"].astype(np.float64),
                        sess=D["sess"].astype(np.int64), acronym=D["acronym"].astype("U16"),
                        n_all=D["n_all"], n_sessions=D["n_sessions"], variables=np.array(RE.VARS))
    E = RE.load_rrr()
    for k in ("abs", "signed", "sess"):
        assert np.array_equal(D[k], E[k]), k
    assert np.array_equal(D["acronym"].astype(str), E["acronym"])
    print("wrote", RE.COMPACT, D["abs"].shape, "identical to the full-model load")


if __name__ == "__main__":
    main()
