#!/usr/bin/env python3
import os
"""DMN PETH similarity versus the exact Posani symmetrized log matrix."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from iblatlas.regions import BrainRegions
DATA = Path(os.environ.get("DMN_DATA", Path.home() / "dmn"))  # data stacks
LDATA = DATA / "lorenzo"  # RRR fit, Posani et al.'s code and data (not in the repository)

HERE = Path(__file__).resolve().parent
DMN, LORENZO = HERE.parents[1], HERE.parent
MIN_NEURONS, N_PERM, SEED = 50, 10000, 42


def recompute():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for this standardized computation")
    saved = np.load(DATA / "concat_cvTrue.npy", allow_pickle=True).flat[0]
    features = torch.as_tensor(saved["concat_z"], dtype=torch.float32, device="cuda")
    labels = np.asarray(BrainRegions().id2acronym(saved["ids"], mapping="Beryl"), str)
    area_file = LDATA / "brainwide-RRR-encoding-model/data/area_list.csv"
    all_areas = pd.read_csv(area_file, header=None)[0].astype(str).to_numpy()
    raw_conn = pd.read_csv(
        LDATA / "brainwide-RRR-encoding-model/data/ctx2ctx_conn.csv", header=None
    ).to_numpy()
    log_conn = np.log(raw_conn + 1e-8)
    counts = np.array([(labels == area).sum() for area in all_areas])
    keep = counts >= MIN_NEURONS
    areas = all_areas[keep]
    indices = np.flatnonzero(keep)

    with torch.no_grad():
        means = torch.stack([features[torch.as_tensor(labels == area, device="cuda")].mean(0)
                             for area in areas])
        means = torch.nn.functional.normalize(means, dim=1)
        peth_similarity = (means @ means.T).cpu().numpy()
    anatomy = (log_conn[np.ix_(indices, indices)] + log_conn[np.ix_(indices, indices)].T) / 2
    iu = np.triu_indices(len(areas), 1)
    x, y = anatomy[iu], peth_similarity[iu]
    observed = pearsonr(x, y).statistic
    rng = np.random.default_rng(SEED)
    null = np.empty(N_PERM)
    for iteration in range(N_PERM):
        order = rng.permutation(len(areas))
        null[iteration] = pearsonr(x, peth_similarity[order][:, order][iu]).statistic
    qap_p = (1 + (np.abs(null) >= abs(observed)).sum()) / (N_PERM + 1)
    rows = [{"area_1": areas[i], "area_2": areas[j],
             "anatomical_connectivity": anatomy[i, j],
             "peth_cosine_similarity": peth_similarity[i, j]}
            for i, j in zip(*iu)]
    pd.DataFrame(rows).to_csv(HERE / "panel_d_pairs.csv", index=False)
    np.savez_compressed(HERE / "panel_d_posani_qap_null.npz", null=null)
    summary = {"device": "cuda", "minimum_neurons_per_region": MIN_NEURONS,
               "regions": len(areas), "region_pairs": len(rows),
               "pearson_r": float(observed), "qap_p": float(qap_p),
               "connectivity": "mean of both directions of log(ctx2ctx_conn + 1e-8)",
               "region_names": areas.tolist()}
    (HERE / "panel_d_posani_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    print(json.dumps(recompute(), indent=2))
