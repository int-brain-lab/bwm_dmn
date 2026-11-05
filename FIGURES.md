# Figure index

Figure folders are named by content, not by manuscript number, because the numbers change
between revisions. This table maps the current manuscript numbers
(`manuscript/IBL_supersession_paper_revision`, 2026-11-04) to folders, scripts and the
PDF that goes into `manuscript/.../figures/`. Update this table when the numbering changes.

Folder renames on 2026-11-04: `fig1` → `fig_task_intro`, `fig2` → `fig_response_types`,
`fig3` → `fig_anatomy`, `fig4` → `fig_response_correlations`, `fig5` → `fig_latency_tiling`
(`fig_si` unchanged). Numbered scripts and modules were renamed with them (e.g.
`make_figure5.py` → `make_latency_tiling.py`, `fig3_common.py` → `anatomy_common.py`,
`sequence_analysis/fig5_groups.py` → `latency_tiling_groups.py`).

## Main figures

| Fig. | Content | Folder | Script | Manuscript PDF |
|---|---|---|---|---|
| 1 | Task, recordings, feature vectors | `fig_task_intro/` | `make_task_intro.py` | `task_data_intro.pdf` (= `task_intro_printer.pdf`) |
| 2 | Response types (k-means, 25 clusters) | `fig_response_types/` | `make_response_types.py` | `functional_response_structure.pdf` |
| 3 | Response types across regions (anatomy) | `fig_anatomy/` | `regenerate_from_data.py` | `anatomy_function_correspondence.pdf` |
| 4 | Stimulus-locked latency tiling | `fig_latency_tiling/` | `make_latency_tiling.py` | `latency_tiling.pdf` |

## Supplementary figures

| Fig. | Content | Folder | Script | Manuscript PDF |
|---|---|---|---|---|
| S1 | Latency-tiling controls | `fig_latency_tiling/si/` | `make_latency_tiling_controls.py` | `latency_tiling_controls.pdf` |
| S2 | RRR (Posani et al.) vs PETH selectivity | `lorenzo/manuscript_comparison_figure/` | `reproduce_all.py` | `rrr_peth_selectivity_comparison.pdf` |
| S3 | Wheel motion before movement onset | `wheel_movement_onset_review/` | `analyze_wheel_movement_onset.py` | `movement_onset_subthreshold_wheel.pdf` |
| S4 | Neuron × neuron correlations; tests of random mixing | `fig_response_correlations/si/`: a, b `neuron_correlation/` (`compute_xcorr.py`, `make_panel.py`); c–e `mixed_selectivity/` (`selectivity_test.py`, `extra_nulls.py`, `rrr_epairs.py` on the S2 RRR fit) | `mixed_selectivity/si_figure/make_response_correlations.py` (assembles a–e) | `response_correlations.pdf` |
| S5 | Cluster composition per region, types vs k | `fig_si/k_robustness/` | `make_regional_cluster_composition.py` | `regional_cluster_composition.pdf` |
| S6 | Specialization metrics, regional diversity | `fig_response_correlations/si/` | `make_specialization_metrics.py` | `specialization_metric_comparison.pdf` |
| S7 | Cortical–subcortical composition | `fig_si/` | `cortical_subcortical_composition.py` | `cortical_subcortical_composition.pdf` |
| S8 | Anatomical hierarchy of regions | (static) | | `structure_tree.pdf` |

## Not in the current manuscript

- `fig_response_correlations/`: the former mixed-selectivity Fig. 4
  (`make_structured_mixed_selectivity.py`) and its 40-cluster version
  (`si/make_mixed_selectivity_k40.py`). Its `compute_synthetic.py` and `compute()` are still
  used by S4 and S6.
- `fig_si/`: the cortex firing-rate figure (`make_cortex_rate.py`; analysis
  `cortex_rate_analysis.py` feeds Fig. 4i, j) and `latency_hierarchy_analysis.py` (feeds S1e, f).
