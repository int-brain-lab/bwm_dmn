# k-robustness of the regional distribution (R3 comment 5)

Candidate SI figure (not yet in the manuscript), 2026-10-13.

- `compute_k_robustness.py`: k-means (sklearn, random_state=0, z-scored feature vectors) for k = 10, 15, 25, 40, 60, 100; labels cached in `cache/`. k = 25 matches the paper's labels (ARI 1.0).
- `make_figure.py`: `k_robustness.pdf/.png`, numbers in `k_robustness_values.txt`, caption `k_robustness_caption.tex`.
- `make_regional_cluster_composition.py`: Fig. S4 = frozen per-region grid (a) + types present vs k (b); `regional_cluster_composition.pdf` is copied to the manuscript `figures/`.
