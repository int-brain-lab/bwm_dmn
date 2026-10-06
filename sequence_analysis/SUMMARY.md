# What the "sequence cells" are — mini summary (2026-10-05)

**Bottom line.** The faint "sequence cells" are not a separate class of neurons
with internally generated sequences. They are mostly sparse, low-reliability cells
that Rastermap groups together because their noisy trial averages resemble no
well-defined response cluster; within that group it orders them by where their
(partly noisy) peaks fall. Their reproducible part is the same stimulus-locked
latency tiling seen in the dense "tornado" cells. The defensible result is
**brain-wide, stimulus-locked latency tiling, from dense to sparse cells,
validated on held-out trials.**

## The evidence, in five steps

1. **The train diagonals do not replicate.** Fit Rastermap on odd trials, show the
   even trials in the same order: the sharp "sequence" diagonals vanish
   (`rastermap_previews/contextual_neural_sequences.png`, clusters 60–69), while
   stimulus/movement/feedback structure survives.
2. **They sit in rarely sampled conditions.** The train diagonals are almost only
   in columns with few trials (block change, discordant: median ~12 trials per
   half, vs ~60 for concordant). Neurons' odd-trial peaks there do not predict
   their even-trial peaks (Spearman ~0.1): Rastermap is sorting noise peaks.
   Refitting without those columns moves the artefact to whichever columns remain.
3. **By every cross-validated test, they are noisier than random cells of the
   same firing rate** (`results/sequence_vs_stim_canonical_60-69.png`):
   xval time-time r 0.08 vs 0.16, peak-time ρ 0.24 vs 0.39, cross-trial-type r
   0.05 vs 0.08. The paper's "narrow constant diagonal" (old Fig. 3d) is an
   artefact of overlapping 12.5 ms bins; it disappears when odd bins are
   correlated with even bins.
4. **Their reliable minority looks like the tornado** (`results/sparse_tornado.png`;
   selection and evaluation on disjoint trial types): latency kept on held-out
   trials (ρ 0.41), same bimodal latency distribution, and stimulus-locked —
   fast- vs slow-RT trials match better aligned to the stimulus than to movement
   (0.16 vs 0.09; tornado 0.45 vs 0.22).
5. **Adding noise to tornado cells turns them into "sequence cells"**
   (`results/degrade_tornado.png`). Per-neuron noise matched to the sequence
   cells' reliability distribution, larger in rarely sampled conditions
   (1/sqrt(trials)), then the same Rastermap procedure: train-only structure in
   the rare columns, faint stimulus-locked structure in the others, and
   statistics close to the real sequence cells (6% reliable cells in both;
   peak-time ρ 0.32 vs 0.24; xval r 0.15 vs 0.08). The real cells are even
   noisier (median reliability 0.02 vs 0.08), hence their cleaner noise diagonals.

## What still holds about these cells

Low firing rates and a bias towards isocortex, olfactory areas, hippocampal
formation and cortical subplate (depleted in midbrain/hindbrain), partly beyond a
rate-matched control.

## Caveats

- Groups are Rastermap clusters chosen by eye (canonical fit: sequence 60–69,
  tornado 17–30); the original manuscript clusters came from an unsaved fit.
- 150 ms windows: late latencies (> ~110 ms) are truncated.
- The noise model is Gaussian and smoothed; it reproduces the pattern and the
  order of magnitude of the statistics, not every detail.
