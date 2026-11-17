#!/usr/bin/env python3
import os
"""Recompute panel f with at least 50 UUID-matched neurons per region."""
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
MODEL = LDATA / "sillouette_score/reproduction_audit/RRRGD_rank5_l275_full.pt"
SESSIONS = LDATA / "brainwide-RRR-encoding-model/trained_RRR_model_paper178/session_features"
MIN_NEURONS, N_PERM, SEED = 50, 10000, 42


def recompute():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    dmn = np.load(DATA / "concat_cvTrue.npy", allow_pickle=True).flat[0]
    dmn_uuids = dmn["uuids"].astype(str)
    dmn_regions = np.asarray(BrainRegions().id2acronym(dmn["ids"], mapping="Beryl"), str)
    dmn_features = torch.as_tensor(dmn["concat_z"], dtype=torch.float32, device="cuda")

    checkpoint = torch.load(MODEL, map_location="cpu", weights_only=False, mmap=True)["RRRGD_model"]
    state = checkpoint["model"]
    summed_v = state["V"].sum(1).to("cuda")
    rrr_uuids, rrr_alpha = [], []
    with torch.no_grad():
        for eid in checkpoint["eids"]:
            with np.load(SESSIONS / f"{eid}.npz") as session:
                rrr_uuids.append(session["uuids"].astype(str))
            u = state[eid + "_U"].to("cuda")
            rrr_alpha.append(torch.einsum("nvr,r->nv", u, summed_v).cpu())
    rrr_uuids = np.concatenate(rrr_uuids)
    rrr_alpha = torch.cat(rrr_alpha).to("cuda")
    lookup = pd.Series(np.arange(len(rrr_uuids)), index=rrr_uuids)
    matched = np.isin(dmn_uuids, rrr_uuids)
    dmn_indices = np.flatnonzero(matched)
    rrr_indices = lookup.loc[dmn_uuids[matched]].to_numpy(int)
    regions = dmn_regions[matched]

    eligible = [area for area in np.unique(regions)
                if area not in ("root", "void") and (regions == area).sum() >= MIN_NEURONS]
    counts = [(regions == area).sum() for area in eligible]
    with torch.no_grad():
        dmn_means, rrr_means = [], []
        for area in eligible:
            local = np.flatnonzero(regions == area)
            dmn_means.append(dmn_features[torch.as_tensor(dmn_indices[local], device="cuda")].mean(0))
            rrr_means.append(rrr_alpha[torch.as_tensor(rrr_indices[local], device="cuda")].mean(0))
        dmn_means = torch.nn.functional.normalize(torch.stack(dmn_means), dim=1)
        rrr_means = torch.nn.functional.normalize(torch.stack(rrr_means), dim=1)
        dmn_similarity = (dmn_means @ dmn_means.T).cpu().numpy()
        rrr_similarity = (rrr_means @ rrr_means.T).cpu().numpy()

    iu = np.triu_indices(len(eligible), 1)
    x, y = dmn_similarity[iu], rrr_similarity[iu]
    observed = pearsonr(x, y).statistic
    rng = np.random.default_rng(SEED)
    null = np.empty(N_PERM)
    for iteration in range(N_PERM):
        order = rng.permutation(len(eligible))
        null[iteration] = pearsonr(x, rrr_similarity[order][:, order][iu]).statistic
    qap_p = (1 + (np.abs(null) >= abs(observed)).sum()) / (N_PERM + 1)
    rows = [{"area_1": eligible[i], "area_2": eligible[j],
             "dmn_cosine_similarity": dmn_similarity[i, j],
             "rank5_rrr_cosine_similarity": rrr_similarity[i, j]}
            for i, j in zip(*iu)]
    pd.DataFrame(rows).to_csv(HERE / "panel_e_pairs.csv", index=False)
    pd.DataFrame({"area": eligible, "matched_neurons": counts}).to_csv(
        HERE / "panel_f_matched_rank5_area_counts.csv", index=False)
    np.savez_compressed(HERE / "panel_f_matched_rank5_qap_null.npz", null=null)
    summary = {"device": "cuda", "model_rank": 5,
               "matched_neurons_total": int(matched.sum()),
               "minimum_matched_neurons_per_region": MIN_NEURONS,
               "included_neurons": int(sum(counts)), "regions": len(eligible),
               "region_pairs": len(rows), "pearson_r": float(observed),
               "qap_p": float(qap_p), "region_names": eligible}
    (HERE / "panel_f_matched_rank5_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    print(json.dumps(recompute(), indent=2))
