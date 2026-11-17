#!/usr/bin/env python3
"""Anatomical composition of the two large anticorrelated groups of Fig. 4a (Rastermap
clusters 13-35, pre-movement; 78-99, feedback), quoted in the Discussion (2026-11-17).
Run (iblenv): python group_anatomy.py"""
import numpy as np, pandas as pd
from iblatlas.regions import BrainRegions
import os
from pathlib import Path
f = np.load(Path(os.environ.get('DMN_DATA', Path.home() / 'dmn')) / 'concat_cvTrue.npy', allow_pickle=True).flat[0]
br = BrainRegions()
ids = np.asarray(f['ids']); lab = np.asarray(f['rm_labels']); fr = np.asarray(f['fr'])
beryl = br.id2acronym(ids, mapping='Beryl'); cosmos = br.id2acronym(ids, mapping='Cosmos')
grp = np.where((lab >= 13) & (lab <= 35), 'pre-movement (13-35)', np.where(lab >= 78, 'feedback (78-99)', 'other'))
df = pd.DataFrame(dict(grp=grp, beryl=beryl, cosmos=cosmos, fr=fr, pid=np.asarray(f['pid'])))
df = df[~df.cosmos.isin(['root', 'void'])]
print('neurons per group'); print(df.grp.value_counts())
ct = pd.crosstab(df.cosmos, df.grp, normalize='columns').round(3)
ct['all'] = df.cosmos.value_counts(normalize=True).round(3)
print('\nCosmos composition (fraction of each group)'); print(ct)
print('\nmedian firing rate (stack units)'); print(df.groupby('grp').fr.median().round(3))
# enrichment of Beryl regions (>= 100 neurons): fraction of the region's neurons in each group
tot = df.beryl.value_counts(); big = tot[tot >= 100].index
fr_ = pd.crosstab(df.beryl, df.grp, normalize='index').loc[big]
base = df.grp.value_counts(normalize=True)
for g in ('pre-movement (13-35)', 'feedback (78-99)'):
    e = (fr_[g] / base[g]).sort_values()
    print(f'\n{g}: base fraction {base[g]:.3f}')
    print('  most enriched :', ', '.join(f'{r} {fr_.loc[r, g]:.2f}' for r in e.index[::-1][:12]))
    print('  most depleted :', ', '.join(f'{r} {fr_.loc[r, g]:.2f}' for r in e.index[:12]))
# per-session check: is the split driven by a few sessions?
ses = pd.crosstab(df.pid, df.grp, normalize='index')
print('\nprobes (n=%d): fraction of neurons in feedback group, quantiles' % len(ses), ses['feedback (78-99)'].quantile([.1,.25,.5,.75,.9]).round(2).tolist())
print('probes: fraction in pre-movement group, quantiles', ses['pre-movement (13-35)'].quantile([.1,.25,.5,.75,.9]).round(2).tolist())
