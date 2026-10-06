# Movement onset and sub-threshold wheel analysis

This analysis addresses the reviewer questions about the definition of BWM
`firstMovement_times`, its relationship to choice, and wheel motion during the
stimulus-aligned neural-response window.

## Reproduce

```bash
python analyze_wheel_movement_onset.py
```

The script reads the local `#2025-03-03#` BWM ALF trial revision and native
`_ibl_wheel.position.npy`/`_ibl_wheel.timestamps.npy` objects from the ONE cache (`$ONE_CACHE_DIR`, default `~/Downloads/ONE`); it does
not download them, so the wheel objects must be fetched with ONE first (the
PETH step downloads trials and spikes, not wheel data). Outputs are in
`results/`; `results/movement_onset_subthreshold_wheel.pdf` is the manuscript's
SI figure (`report/figures/movement_onset_subthreshold_wheel.pdf`).

## Operational definition

The installed IBL extractor first detects candidate movements whose wheel
displacement exceeds eight encoder samples within 200 ms. It refines each
candidate onset to the first 1.5-encoder-sample displacement and retains as a
trial's `firstMovement_times` the first candidate with peak amplitude at least
0.1 rad in the search interval from the go cue minus the quiescence period to
feedback. Neither choice nor movement direction enters the detector. Thus,
movement onset is not defined as arbitrary nonzero velocity and need not point
toward the eventual choice.

## Results

The local cache contained 459 BWM trial tables; 396 sessions from 128 subjects
had matching wheel data. The main analysis included 85,226 trials with a valid
go-cue-to-onset interval after excluding the final 20 ms before reported onset.

- Pre-onset net wheel direction (position at onset minus 20 ms, relative to
  position at go cue) matched the recorded final choice on 51.7% of
  direction-defined trials (75,376 trials), close to chance.
- Wheel displacement during the first 50 ms after detected onset matched the
  recorded final choice on 85.7% of trials (84,877 trials). The IBL sign
  convention is positive wheel displacement for `choice == -1`.
- Among 26,493 trials whose movement onset occurred after 170 ms, all had at
  least one encoder change in the 0--150 ms stimulus PETH window. The cumulative
  path reached at least 1.5 encoder steps on 60.8% and at least eight encoder
  steps on 8.8% of trials.

The last result means very small wheel/visual-stimulus displacements commonly
occur during the nominal stimulus-response window. Their direction is not
systematically related to the subsequent choice movement, arguing against a
simple covert onset of the chosen action. Nevertheless, these visual-motion
transients are a plausible sensory confound and should be acknowledged or
controlled in analyses interpreting stimulus-aligned activity.

## Suggested SI caption

**Movement onset and sub-threshold wheel motion.** Analyses used 396 BWM
sessions with locally available wheel data from 128 mice and 11 laboratories;
all 396 sessions contributed to panel **b**. **a**, Example wheel traces
aligned to `firstMovement_times` and normalized to the eventual movement
direction. Candidate wheel movements exceed eight encoder samples within 200
ms; onset is refined using a 1.5-sample displacement, and the first candidate
with peak amplitude at least 0.1 rad is selected without using movement
direction or choice. The gray interval (−20 to 0 ms) was excluded from
pre-onset measurements. **b**, Across-session fractions of trials in which net
wheel displacement between go cue and 20 ms before movement onset matched the
final recorded choice, and in which displacement during the first 50 ms after
onset matched the final choice. The dashed line indicates 50%.
