# Sequence analyses

Code and results for analyses of the "sequence" cells (Fig. 3). Data: the
per-trial PETH bundles in `~/dmn/concat/` (515 insertions) and the odd/even CV
stack (`~/dmn/concat_cvTrue.npy`) for Rastermap cluster labels. Reaction times
from the BWM trials table (`~/Downloads/ONE/bwm_tables/trials.pqt`); bundle trial
indices verified to match its rows.

## 1. Stimulus-locked latency diversity ("tornado") or timing to movement?

```bash
python rt_split_latency.py --workers 10   # per-neuron latencies, ~20 s
python summarize_rt_split.py              # results/summary.txt, results/rt_split_latency.png
```

Method: four correct stimulus-aligned trial types (window 0–150 ms after stimulus)
and the matching movement-aligned types (150–0 ms before first movement). Odd
trials (1st, 3rd, …) give each neuron's peak latency, used for selection and
sorting only; selection = odd/even reliability r ≥ 0.5, rate ≥ 0.5 Hz, peak not at a
window edge (1,128 of 53,027 neurons). Even trials split at the session's median
reaction time (fast median 121 ms, slow 185 ms; gap 65 ms); peak latency measured
on each, in both alignments.

Result (2026-10-05): the responses are **stimulus-locked**. Stimulus-aligned
diagonals are the same on fast and slow trials; peak shifts in the stimulus frame
are 0 ms (peaks < 50 ms) to ~10 ms (50–125 ms), far from the 65 ms reaction-time
gap, while in the movement frame they smear and shift by −20 to −30 ms. Latency
spread broadens with latency (the "tornado"). Caveat: the windows are only 150 ms,
so peaks near the window end (> ~110 ms) are truncated and not informative.
Selected neurons are concentrated in odd-fit Rastermap clusters 20–31, 42, 56–57.

## 2. Rastermap exploration (moved here from `../fig3` on 2026-10-05)

Scripts share `seq_common.py` (private `cache/`); images go to `rastermap_previews/`.

- `regenerate_rasters.py`, `make_figure3.py`: train (odd) / test (even) rasters in
  the order of the canonical odd-trial fit stored in the CV stack (all 21 types).
  Result: sharp train-only diagonals in clusters 61–69, absent on even trials;
  they sit in the rarely sampled block-change columns (noise peaks).
- `preview_all_trials.py`: all-trial X sorted by its own fit.
- `preview_contrast.py`: robust per-neuron normalization + gamma (does not reveal
  test-half sequences in clusters 55–75).
- `fit_rastermap_subset.py` → `results/rastermap_fit9.npz`: fit on the 9 types with
  >= 10 trials per half in >= 98% of insertions (concordant stimulus/movement,
  mistake s/m, motor initiation, L/R movement); 194 neurons silent in these columns
  are appended unclustered. `preview_fit9.py [--fit-only]` shows train/test in that
  order (all 21 columns, fit columns marked; or fit columns only).
  Result: stimulus-window diagonals (clusters ~48–55) replicate on even trials and
  in held-out columns; remaining train-only diagonals are in the mistake columns.

## 3. Do "sequence" neurons differ from stimulus/integrator neurons? (Fig. 3 tests)

`sequence_vs_stim.py` → `results/sequence_vs_stim.png/.pdf/.txt`. Groups on the
fit7z order (odd trials): sequence = clusters 25–30, 46–51; stim/integ = 90–99.
Six stimulus-aligned trial types, four held out of the fit. Reproduces Fig. 3
d–k on even trials, plus cross-validated (odd × even) versions that share no
spikes: the strided 12.5 ms bins overlap, so any within-half time-time
correlation has a narrow constant diagonal band even for noise.

Result (2026-10-05): rate lower and Isocortex/OLF/CTXsp/HPF-biased (reproduced);
but the narrow constant band of the sequence group is the bin-overlap artefact
(cross-validated band absent: lag-0 r 0.12 vs 0.35 for stim/integ), peak times
replicate weakly (rho 0.18 vs 0.54), cross-trial-type xval correlation 0.07 vs
0.33, and the mean response is a weaker transient + ramp rather than flat.

### 3b. Sequence = clusters 60–69 of the canonical fit (user's definition, 2026-10-05)

`python sequence_vs_stim.py --fit canonical --seq 60-69 --stim 17-30 --tag _canonical_60-69`
(canonical = the odd-trial fit stored in the CV stack, as in
`rastermap_previews/contextual_neural_sequences.png`; stim/integ = the replicating
stimulus-aligned shapes, clusters 17–30; grey = rate-matched control from all other
clusters). Sequence group less reproducible than the rate-matched control
(xval lag-0 r 0.08 vs 0.16, peak-time rho 0.24 vs 0.39, cross-type 0.05 vs 0.08);
train diagonal only in change_b,s; mean response ~flat near 0; Isocortex/OLF/HPF/
CTXsp enriched, MB/HB depleted (partly shared with the rate-matched control).

## 4. Are "sequence" cells just sparse tornado cells? (`sparse_tornado.py`)

Canonical fit; sequence = 60–69, tornado = 17–30. Selection on the two concordant
stimulus types (odd vs even), evaluation on the four other stimulus types (even
trials) -> disjoint trials. Results (2026-10-05, `results/sparse_tornado.*`):
- reliability rises with rate for all cells; sequence cells are the least reliable
  (median r 0.12; 29% with r >= 0.3, vs 76% tornado, 42% other);
- at matched rate their timing replicates *less* than other cells' (b): the cluster
  collects the least reliable cells of each rate band (sorted partly by noise);
- their reliable part (1,236 cells) keeps its latency on held-out trial types
  (rho 0.41; tornado 0.75), has the same bimodal latency distribution as the
  tornado (g), and matches fast vs slow RT trials better aligned to the stimulus
  than to movement (pattern r 0.16 vs 0.09; tornado 0.45 vs 0.22): stimulus-locked.

## 5. Noise turns tornado cells into "sequence cells" (`degrade_tornado.py`)

Tornado cells (17–30) + per-neuron smoothed Gaussian noise, matched quantile by
quantile to the sequence cells' reliability distribution and scaled per PETH type by
1/sqrt(trials per half); own Rastermap fit (odd), shown on even. Reproduces
train-only structure in rare columns and sequence-like statistics
(`results/degrade_tornado.*`). Overall conclusion: see `SUMMARY.md`.
