# Fig. S2: comparison with Posani et al. (reduced-rank regression)

**From the saved result tables (default, seconds):** `python compose_si_lorenzo_native.py` draws
all panels from the CSV/JSON/NPZ tables in this folder and the small region tables under
`../clustering-analysis/data/` and `../sillouette_score/`, and writes `../../figs/si_lorenzo_panels.pdf`
(the manuscript's `report/figures/rrr_peth_selectivity_comparison.pdf`; `report/build.sh --figures`
does this).

**Full recomputation (CUDA, hours):** `python reproduce_all.py` recomputes both silhouette analyses,
the regional similarity tables and the QAP tests. It needs, under `$DMN_DATA/lorenzo/`, the code and
data of Posani et al. (`brainwide-RRR-encoding-model/`, with `data/area_list.csv`,
`data/ctx2ctx_conn.csv` and the per-session features in `trained_RRR_model_paper178/session_features/`),
the rank-5 RRR fit (`sillouette_score/reproduction_audit/RRRGD_rank5_l275_full.pt`,
`rank5_l275_cv_r2.npz`; fitted with their code, see the manuscript's Methods) and the Harris et al.
(2019) supplementary table `41586_2019_1716_MOESM5_ESM.xlsx`. `--refit-rrr` also refits the RRR model.

Settings as in the figure: 100 Gaussian null datasets per region, 50 k-means initializations, at
least 50 neurons per region, 10,000 region-label QAP permutations.
