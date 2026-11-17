#!/usr/bin/env python3
"""Plot Nature RRR selectivity and BWM decoding profiles side by side."""
import os
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

OUT = Path(__file__).resolve().parent
DATA = Path(os.environ.get('DMN_DATA', Path.home() / 'dmn'))
SRC = Path('/tmp/nature_fig4_sources/Separability-main/data')
OUT.mkdir(parents=True, exist_ok=True)

# Reconstruct the values used in Nature Fig. 2c from the authors' processed
# time-varying RRR coefficients distributed with their Separability repository.
betas = np.load(SRC / 'RRR_betas.npy')       # neuron x variable x time
areas = np.load(SRC / 'RRR_betas_areas.npy')
area_hierarchy = pd.read_csv(
    '/tmp/nature_fig4_sources/brainwide-RRR-encoding-model-main/data/area_list.csv',
    header=None)[0].tolist()
included = [a for a in area_hierarchy if np.sum(areas == a) >= 30]
ordered = ([a for a in included if 'SS' in a] +
           [a for a in included if 'VIS' in a] +
           [a for a in included if ('AUD' in a) or ('GU' in a)] +
           [a for a in included if 'RSP' in a] +
           [a for a in included if 'AI' in a])
ordered += [a for a in included if a not in ordered]
ordered = ordered[::-1]

alpha_raw = np.asarray([np.abs(betas[areas == a]).sum(2).mean(0)
                        for a in ordered])
alpha_z = (alpha_raw - alpha_raw.mean(0)) / alpha_raw.std(0)
alpha_vars = ['Block', 'Side', 'Contrast', 'Choice', 'Outcome',
              'Wheel', 'Whisker', 'Lick']

# Reconstruct exactly the regional mean scores that feed get_dec_bwm, but retain
# raw means first and z-score each decoder across the same displayed regions.
dec_vars = ['stimside', 'choice', 'feedback', 'wheel-speed', 'wheel-velocity']
dec_labels = ['Stimulus side', 'Choice', 'Feedback', 'Wheel speed',
              'Wheel velocity']
dec_raw = np.full((len(ordered), len(dec_vars)), np.nan)
for j, vari in enumerate(dec_vars):
    d = pd.read_parquet(DATA / 'bwm_decoding' / f'{vari}_stage2.pqt')
    d = d.dropna(subset=['score', 'region'])
    valid = d.groupby('region').score.count()
    valid = valid[valid >= 3].index
    means = d[d.region.isin(valid)].groupby('region').score.mean()
    dec_raw[:, j] = [means.get(a, np.nan) for a in ordered]

dec_z = np.full_like(dec_raw, np.nan)
for j in range(dec_raw.shape[1]):
    good = np.isfinite(dec_raw[:, j])
    dec_z[good, j] = ((dec_raw[good, j] - dec_raw[good, j].mean()) /
                      dec_raw[good, j].std())

# Shared scale makes red intensity quantitatively comparable in z-score units.
vmin = np.nanmin([np.nanmin(alpha_z), np.nanmin(dec_z)])
vmax = np.nanmax([np.nanmax(alpha_z), np.nanmax(dec_z)])
fig, axes = plt.subplots(1, 2, figsize=(7.8, 6.2), sharey=True,
                         gridspec_kw={'width_ratios': [8, 5], 'wspace': .08})
im0 = axes[0].imshow(alpha_z, aspect='auto', cmap='Reds', interpolation='none',
                     vmin=vmin, vmax=vmax)
im1 = axes[1].imshow(dec_z, aspect='auto', cmap='Reds', interpolation='none',
                     vmin=vmin, vmax=vmax)
for ax, labs, title in zip(
        axes, [alpha_vars, dec_labels],
        ['RRR mean absolute selectivity\n(Posani et al., Fig. 2c)',
         'BWM regional decoding score']):
    ax.set_xticks(np.arange(len(labs)), labs, rotation=90)
    ax.set_yticks(np.arange(len(ordered)), ordered)
    ax.set_title(title, fontsize=11)
    ax.tick_params(length=2)
axes[0].set_ylabel('Cortical region (shared order)')
cax = fig.add_axes([.91, .34, .018, .35])
cbar = fig.colorbar(im1, cax=cax)
cbar.set_label('z-scored across displayed regions')
fig.subplots_adjust(left=.14, right=.88, bottom=.20, top=.88)
fig.savefig(OUT / 'rrr_selectivity_vs_bwm_decoding_side_by_side.pdf',
            bbox_inches='tight')
fig.savefig(OUT / 'rrr_selectivity_vs_bwm_decoding_side_by_side.svg',
            bbox_inches='tight')
plt.close(fig)

# Long-form values and closest column-wise comparisons.
rows = []
for i, area in enumerate(ordered):
    row = {'region': area}
    row.update({f'alpha_{v.lower()}_z': alpha_z[i, j]
                for j, v in enumerate(alpha_vars)})
    row.update({f'dec_{v}_raw': dec_raw[i, j] for j, v in enumerate(dec_vars)})
    row.update({f'dec_{v}_z': dec_z[i, j] for j, v in enumerate(dec_vars)})
    rows.append(row)
pd.DataFrame(rows).to_csv(OUT / 'rrr_selectivity_vs_bwm_decoding_values.csv',
                          index=False)

pairs = [('Side', 1, 'Stimulus side', 0),
         ('Choice', 3, 'Choice', 1),
         ('Outcome', 4, 'Feedback', 2),
         ('Wheel', 5, 'Wheel speed', 3),
         ('Wheel', 5, 'Wheel velocity', 4)]
with open(OUT / 'SELECTIVITY_DECODING_COMPARISON.md', 'w') as f:
    f.write('# RRR selectivity versus BWM decoding\n\n')
    f.write(f'{len(ordered)} cortical regions are shown in a shared order. '\
            'Each column is independently z-scored across these regions.\n\n')
    f.write('| RRR variable | BWM decoder | Spearman rho | P | n |\n')
    f.write('|---|---|---:|---:|---:|\n')
    for aname, ai, dname, di in pairs:
        good = np.isfinite(alpha_z[:, ai]) & np.isfinite(dec_z[:, di])
        s = spearmanr(alpha_z[good, ai], dec_z[good, di])
        f.write(f'| {aname} | {dname} | {s.statistic:.3f} | '
                f'{s.pvalue:.4g} | {good.sum()} |\n')

print(f'Saved paired heatmap and values to {OUT}')
