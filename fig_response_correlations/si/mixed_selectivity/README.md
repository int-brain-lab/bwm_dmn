# Structured vs random mixed selectivity: a test beyond second order

This folder holds a stricter test of the Fig. 4 claim ("structured, non-random
selectivity"). Nothing outside this folder is written. The data are read
through `fig_response_correlations/compute_synthetic.load(nclus=100)` from the existing caches.

## Run (iblenv, GPU)

```bash
source /home/mic/miniforge3/etc/profile.d/conda.sh && conda activate iblenv
cd ~/dmn/fig_response_correlations/si/mixed_selectivity
python selectivity_test.py            # ~5 min on the RTX 3500 Ada; cached in cache/
python selectivity_test.py --rerun    # recompute (--n-null N, default 10)
python make_figure.py                 # selectivity_test.pdf/.png
```

Files:
- `selectivity_test.py` does the analysis. Results go to `cache/results.json`
  (all numbers) and `cache/arrays.npz` (matrices and angle distributions).
- `make_figure.py` draws the figure (`selectivity_test.pdf/.png`).
- `selectivity_test_caption.tex` is the LaTeX caption.

Validation of the GPU code against numpy/sklearn on a 3,000-neuron subset
(`results.json["validation"]`):
- PCA eigenvalues: maximum relative error 3e-15.
- 5-NN angles vs sklearn `NearestNeighbors(metric="cosine")`: maximum error 0.002°.
- k-means inertia (k = 8): 1037719 (torch) vs 1037763 (sklearn).
- Silhouette: 0.13932 (torch) vs 0.13932 (sklearn).

## Methods in brief

- **Data.** X is `concat_z`: 54,719 neurons × 1872 bins, each neuron z-scored
  over the 21 concatenated windows. There are 515 probe insertions. Only
  trial-averaged PETHs are used.
- **Bases.**
  - V is the 100 k-means cluster means, as in Fig. 4. V Vᵀ has condition
    number 7.2e4.
  - W = (V Vᵀ)^(-1/2) V is the Löwdin (symmetric) orthonormalisation of V.
    Each W row has a median cosine of only 0.45 to its V row, because V is
    ill-conditioned.
  - PCA of X: 10 PCs hold 39.9% of the variance, 20 PCs hold 47.2% and 100
    PCs hold 72.4%.
- **Nulls.** Each null draw has the same size as the data; 10 draws per null.
  - *Gaussian*: same mean and covariance as the data.
  - *PC-marginal*: rotate to the data's PCs, permute each PC score
    independently, rotate back. This keeps the covariance and each PC's 1-D
    distribution (e.g. heavy tails) and removes all other dependencies.
  - *Elliptical*: Gaussian directions with the real Mahalanobis radii. This
    controls k-means for radial and heavy-tail structure. Its angles behave
    like the Gaussian null's.
  - *Axis-marginal*: the Fig. 4 null, with each coordinate permuted
    independently. It also destroys the covariance.
- **Positive control ("mixture").** A 25-component Gaussian mixture fitted to
  the same data by k-means (cluster means plus pooled within-cluster
  covariance). Its total covariance is close to the data's. This shows what a
  genuinely clustered population with this covariance looks like to each test.
- **ePAIRS-style test** (Raposo et al. 2014; Hirokawa et al. 2019).
  - For each neuron: the mean angle between its vector and its k = 1, 5 or 20
    nearest neighbours in angle. k = 5 is reported.
  - Neighbours on the same probe insertion are excluded ("xprobe"; results with
    all neighbours are nearly identical). This excludes shared trial noise and
    near-duplicate units.
  - Effect sizes:
    - Δ = median angle (null) − median angle (real), in degrees.
    - AUC = P(angle_real < angle_null), where 0.5 means no clustering.
    - z = Δ divided by the s.d. of the null medians across draws. With
      N = 54,719, every |z| > 10 is "significant", so the AUC and Δ are what
      matter.
- **k-means.**
  - Torch Lloyd's algorithm with k-means++ initialisation, 4 inits, k = 2…50.
  - Gap statistic: mean log W_k of the null minus log W_k of the real data.
  - Silhouette on a fixed random subset of 10,000 neurons.
  - For the selectivity space, the indices are first scaled to unit variance.

## Part 1: orthonormal bases

| | k-means means V (Fig. 4) | Löwdin-orthonormal W |
|---|---|---|
| r(off-diag corr(C), corr of basis rows) | 0.912 | 0.010 |
| r(off-diag corr(C), white noise projected onto basis) | **0.912** | **0.002** |
| mean \|off-diag corr\|, real | 0.517 | 0.174 |
| mean \|off-diag corr\|, white noise | 0.301 | 0 |
| mean \|off-diag corr\|, axis-marginal null | 0.003 | 0.003 |
| mean \|off-diag corr\|, covariance-matched Gaussian | 0.517 (r = 1.00 with real) | 0.174 (r = 1.00) |
| PC1 share of coefficient variance | **71%** | **24%** |

- With V, the correlation pattern comes from the basis (it reproduces the
  r = 0.91 finding).
- With an orthonormal basis, the remaining correlations are the data
  covariance. Neither the marginal null nor any second-order statistic can
  show structure beyond the covariance.

**Higher-order tests (top 10 / 20 PCs and the 100 W coefficients):**

| dataset (5-NN, cross-probe) | median angle real | vs Gaussian: Δ, AUC | vs PC-marginal: Δ, AUC | vs elliptical: AUC |
|---|---|---|---|---|
| top 10 PCs | 18.0° | 2.3°, **0.62** | 1.5°, **0.58** | 0.61 |
| top 10 PCs, top 20% by norm | 14.3° | 2.9°, 0.66 | 3.2°, 0.68 | 0.71 |
| top 20 PCs | 28.5° | 4.2°, 0.65 | 3.2°, 0.62 | 0.65 |
| 100 W coefficients | 44.2° | 5.5°, 0.62 | 4.6°, 0.62 | 0.63 |
| 100 W coefficients vs axis-marginal (Fig. 4) null | | 20.9°, **0.98** | | |
| positive control: 25-cluster mixture (10 PCs) | 18.6° | 1.7°, 0.58 | 1.3°, 0.56 | 0.58 |

- Using k = 1 or k = 20 neighbours gives the same results (top 10 PCs: AUC
  0.61–0.62 vs Gaussian and 0.58 vs PC-marginal).
- **Firing-rate quintiles.** Here each quintile gets its own fitted nulls.
  - Top 10 PCs: AUC vs Gaussian is 0.69 in the lowest quintile and 0.62–0.63
    in the others. Against the PC-marginal null it falls from 0.66 (lowest) to
    0.57 (highest).
  - So different signal-to-noise levels across firing rates do not explain
    the effect. The lowest-rate neurons (sparse, spiky PETHs) inflate it
    somewhat.

**k-means in the top 10 PCs**, at k = 2 / 10 / 50:

| null | gap | real/null inertia (k = 10, 50) |
|---|---|---|
| Gaussian | 0.02 / 0.07 / 0.16 | 0.93, 0.85 |
| PC-marginal | 0.00 / 0.05 / 0.13 | 0.95, 0.88 |
| elliptical | 0.03 / 0.10 / 0.19 | 0.91, 0.83 |

- Silhouette, real vs Gaussian null: 0.235 vs 0.209 at k = 2, and 0.111 vs
  0.092 at k = 10.
- The real silhouette **decreases monotonically with k**. There is no
  preferred cluster number, whereas the 25-cluster mixture control shows a
  bump near k = 20.
- The top 20 PCs give the same picture, with slightly smaller numbers.

**Reading of Part 1.**
- The neural response vectors are not an anisotropic Gaussian cloud.
  Neighbouring neurons are more aligned than any second-order-matched null
  predicts:
  - AUC ≈ 0.62 (top 10 PCs), with Δ of about 2–5° in median neighbour angle.
  - The effect survives cross-probe exclusion, firing-rate stratification and
    a null that keeps the heavy-tailed PC marginals.
  - It is at least as large as for a 25-cluster mixture fitted to the data.
- The structure is **graded/continuous, not discrete clusters**:
  - gap values are small (inertia 5–15% below the null);
  - there is no silhouette peak.
- The effect is modest: the real and null angle distributions overlap
  substantially (AUC 0.62, where 1.0 is full separation).
- The Fig. 4 axis-marginal null is far too permissive. Against it, the real
  data look strongly structured (AUC 0.98) only because that null removes the
  covariance.

## Part 2: task-variable selectivity

Each index is a contrast of window means of z-scored condition-averaged PETHs
(units: s.d. of the neuron's concatenated PETH). The direction of each neuron's
selectivity vector does not depend on the z-scoring. The windows come from
`dmn_bwm.concat_PETHs` (bins 12.5 ms, 2 ms stride; 72 bins = 150 ms):

| index | contrast | epoch |
|---|---|---|
| block_pre | blockL − blockR | −0.4…−0.1 s before stim, all trials |
| block_stim | ½(LbL+RbL) − ½(LbR+RbR) | 0–150 ms after stim, correct, deep in block |
| block_move | same, movement-aligned windows | −150–0 ms before first movement, correct |
| side_stim | ½(LbL+LbR) − ½(RbR+RbL) | stim, correct |
| side_move | same, movement-aligned windows | pre-move, correct |
| choice_post | choiceL − choiceR | 0–150 ms after movement onset, **all** trials |
| congr_stim | ½(LbL+RbR) − ½(LbR+RbL) | stim (block × side interaction = prior-congruence) |
| congr_move | same, movement-aligned windows | pre-move |
| error_stim | mistake_s − mean(4 correct stim windows) | stim |
| error_move | mistake_m − mean(4 correct pre-move windows) | pre-move |
| outcome_fb | fback1 − fback0 | 0–300 ms after feedback |

### What is and is not dissociable

- **Stimulus side vs choice: NOT dissociable.**
  - All side-split windows (`stimXbYcX`, `sXbYchoiceX`) are correct trials, so
    stimulus side = choice side.
  - The error windows (`mistake_s`, `mistake_m`) pool both sides.
  - `choiceL/R` mixes correct and incorrect trials. If a neuron responds
    linearly to stimulus (weight a) and choice (weight b), with error rate e,
    then choiceL − choiceR ∝ b + (1−2e)a, while side ∝ a + b. The expected
    across-neuron correlation is 0.95–0.99 for e = 0.15–0.25 and any
    sd(a)/sd(b) between 0.5 and 2.
  - Every "side" index is therefore a stimulus+choice mixture. Separating them
    needs per-trial data (error trials split by side).
- **Block prior: dissociable from side.** Within each epoch, block is the
  side-balanced contrast on the same correct trials, so it is separable from
  side. It is not separable from slow drifts in firing rate, because blocks
  are separate stretches of the session. The pre-stimulus block index has the
  highest kurtosis (5.6), consistent with a drift contribution.
- **Congruence and outcome.**
  - Congruence (block × side) is separable from both main effects.
  - Error-vs-correct at stimulus/movement is confounded with stimulus contrast
    (errors happen at low contrast) and with not being deep in a block.
  - Feedback outcome is confounded with the reward vs noise-burst sensory
    events.

### Correlations between indices (across neurons)

Large correlations only, with the predicted noise correlation in brackets:

| pair | r | expected from shared windows (noise) |
|---|---|---|
| block_stim–block_move | 0.83 | 0 (different windows) |
| error_stim–error_move | 0.77 | 0 |
| side_stim–side_move | 0.74 | 0 |
| congr_stim–congr_move | 0.72 | 0 |
| block_pre–block_stim / block_move | 0.63 / 0.62 | 0 |
| side_move–choice_post | 0.39 | 0 (yoking predicts ~0.97 within one epoch) |
| side_stim–choice_post | 0.33 | 0 |
| block_stim–side_stim | −0.31 | **−0.20** |
| block_move–side_move | −0.28 | **−0.18** |
| block_stim–choice_post | 0.21 | 0 |
| congr_stim–error_stim | 0.18 | **0.12** |
| error_move–outcome_fb | −0.19 | 0 |

- **Same variable across epochs (r = 0.6–0.8).** This is persistence of the
  same signal. It probably also contains noise shared across epochs through
  the same trials, which the noise model cannot capture.
- **side_move vs choice_post (r = 0.39).** This is far below the ~0.97 that
  yoking predicts within one epoch, so pre- and post-movement side signals
  are only partly the same neurons.
- **Block × side (r ≈ −0.3).** This is mostly sampling noise. The
  prior-incongruent windows (LbR, RbL) have fewer trials, so they are noisier
  than the congruent ones. The noise model predicts −0.20 of the −0.31.
- **Noise model.** Per-bin noise variance is estimated from lag-7-bin
  (14.6 ms) differences. Window means are treated as having len/6 independent
  bins. The estimates are upper bounds.
- **Estimated noise share of each index's variance:**
  - block_pre 0.11
  - block_stim 0.30
  - block_move 0.29
  - side_stim 0.38
  - side_move 0.28
  - choice_post 0.06
  - congr_stim 0.53
  - congr_move 0.49
  - error_stim 0.30
  - error_move 0.27
  - outcome_fb 0.05
- **Dimensionality.** The participation ratio is 5.8 of 11. The first four
  PCs hold 25/23/17/13%.

### Is the distribution of selectivity vectors clustered?

| (5-NN, cross-probe) | median angle real | vs Gaussian: Δ, AUC | vs PC-marginal: Δ, AUC | vs elliptical: AUC |
|---|---|---|---|---|
| all neurons | 19.4° | −0.3°, **0.505** | 0.25°, **0.525** | 0.505 |
| top 20% by norm | 15.7° | 2.6°, 0.68 | 0.8°, **0.57** | 0.74 |
| positive control: 25-cluster mixture | 19.5° | −0.4°, 0.50 | | |

- **k-means on the indices** (unit variance):
  - The gap vs the Gaussian null is *negative* for k ≤ 10 (−0.03), so the
    real data are less clusterable than the Gaussian. It turns slightly
    positive for k ≥ 15 (≤ 0.05).
  - Real silhouette is 0.113 vs 0.137 for the Gaussian null at k = 2.
- **Firing-rate quintiles.** AUC vs Gaussian is 0.48–0.50 in quintiles 2–5
  and 0.59 only in the lowest-rate quintile.

**Reading of Part 2.**
- Over all neurons, the 11-dimensional task-selectivity vectors are
  consistent with a (strongly anisotropic) Gaussian: AUC 0.505, i.e. random
  mixed selectivity in the Fusi sense, given these indices.
- In the most selective 20%, neighbours are more aligned than in a Gaussian
  (AUC 0.68). Most of this comes from heavy-tailed but independent
  selectivity components (excess kurtosis 1.9–5.6): AUC drops to 0.57 against
  the PC-marginal null.
- Caveat on test power: the 25-cluster mixture control is also invisible in
  this space (AUC 0.50). Trial-averaging noise (30–50% of the variance for
  several indices) and only ~6 effective dimensions make ePAIRS insensitive
  here. The result is "no detectable clustering", not proof of isotropy.

## Bottom line

- **Supported:** the full response profiles (PSTH space) deviate from a
  covariance-matched Gaussian in a reproducible way. Neighbouring neurons are
  more similar than chance (AUC ≈ 0.62, Δ ≈ 2–5°), as strongly as for a
  25-cluster mixture fitted to the data, and robust to the controls. This is
  a *graded* non-Gaussian structure (a continuum with dense regions), not
  discrete cell classes.
- **Not supported:**
  - The current Fig. 4 evidence. The corr(C) pattern and the marginal null
    are near-circular, and the marginal null rejects any correlated data.
  - Any claim of non-random structure in *task-variable* selectivity. Here
    the data are indistinguishable from a covariance-matched Gaussian, and the
    excess among strongly selective neurons is mostly explained by
    heavy-tailed independent components.
  - The covariance itself is structured: block indices persist across epochs,
    and side indices persist from stimulus to movement. Except for the block ×
    side term, this is not explained by yoking or noise. But that is
    second-order structure, which a "random" (Gaussian) model with this
    covariance already contains.
