#!/usr/bin/env python3
import os
"""Recompute Fig. 2d-style analysis with the fresh rank-5 local RRR fit."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
DATA = Path(os.environ.get("DMN_DATA", Path.home() / "dmn"))  # data stacks
LDATA = DATA / "lorenzo"  # RRR fit, Posani et al.'s code and data (not in the repository)

HERE = Path(__file__).resolve().parent
LORENZO = HERE.parent
MODEL = LDATA / 'sillouette_score/reproduction_audit/RRRGD_rank5_l275_full.pt'
CV = np.load(LDATA / 'sillouette_score/reproduction_audit/rank5_l275_cv_r2.npz')
SESS = LDATA / 'brainwide-RRR-encoding-model/trained_RRR_model_paper178/session_features'
AREAS = pd.read_csv(LDATA / 'brainwide-RRR-encoding-model/data/area_list.csv', header=None)[0].astype(str).to_numpy()
CONN = np.log(pd.read_csv(LDATA / 'brainwide-RRR-encoding-model/data/ctx2ctx_conn.csv', header=None).to_numpy() + 1e-8)
assert torch.cuda.is_available()

ck = torch.load(MODEL, map_location='cpu', weights_only=False, mmap=True)['RRRGD_model']
state = ck['model']
V = state['V'].to('cuda')
profiles, regions = [], []
with torch.no_grad():
    for eid in ck['eids']:
        with np.load(SESS / f'{eid}.npz') as old:
            delta = CV[eid] - old['null_r2']
            acronym = old['acronym'].astype(str)
            mask = (delta > .015) & np.isin(acronym, AREAS)
        U = state[eid + '_U'].to('cuda')
        alpha = (U @ V).abs().sum(2).cpu().numpy()
        profiles.append(alpha[mask])
        regions.append(acronym[mask])

X = np.concatenate(profiles)
region = np.concatenate(regions)
counts = np.array([(region == a).sum() for a in AREAS])
included = AREAS[counts >= 50]
means = np.array([X[region == a].mean(0) for a in included])
standard = (means - means.mean(0)) / means.std(0)
unit = standard / np.linalg.norm(standard, axis=1, keepdims=True)
similarity = unit @ unit.T
lookup = {a: i for i, a in enumerate(AREAS)}
anatomical = np.array([[(CONN[lookup[a], lookup[b]] + CONN[lookup[b], lookup[a]]) / 2
                        for b in included] for a in included])
iu = np.triu_indices(len(included), 1)
xv, yv = anatomical[iu], similarity[iu]
rho, p_naive = spearmanr(xv, yv)

rng = np.random.default_rng(42)
null = np.empty(10000)
for b in range(10000):
    order = rng.permutation(len(included))
    null[b] = spearmanr(xv, similarity[order][:, order][iu]).statistic
p_qap = (1 + (np.abs(null) >= abs(rho)).sum()) / 10001

rows = []
for k, (i, j) in enumerate(zip(*iu)):
    rows.append({'area_1': included[i], 'area_2': included[j],
                 'anatomical_connectivity': xv[k], 'selectivity_similarity': yv[k]})
pd.DataFrame(rows).to_csv(HERE / 'panel_f_rank5_pairs.csv', index=False)
pd.DataFrame({'area': AREAS, 'rank5_selective_neurons': counts,
              'included': counts >= 50}).to_csv(HERE / 'panel_f_rank5_area_counts.csv', index=False)
np.savez_compressed(HERE / 'panel_f_rank5_qap_null.npz', null=null)

fig, ax = plt.subplots(figsize=(5, 4.8))
ax.scatter(xv, yv, s=26, facecolor='white', edgecolor='#ff2d2d', linewidth=1.2)
coef = np.polyfit(xv, yv, 1)
xx = np.linspace(xv.min(), xv.max(), 200)
ax.plot(xx, np.polyval(coef, xx), color='#ff2d2d', lw=2.5)
ax.set(xlabel='Anatomical connectivity score', ylabel='Selectivity similarity', ylim=(-1.05, 1.05),
       title=f'Rank-5 local RRR: all eligible regions\nSpearman ρ={rho:.2f}, QAP P={p_qap:.3g}, n={len(xv)} pairs')
ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
for ext in ('png', 'pdf', 'svg'):
    fig.savefig(HERE / f'panel_f_rank5.{ext}', dpi=300, bbox_inches='tight')
plt.close(fig)

summary = {'device': 'cuda', 'model_rank': 5, 'model_l2': 75,
           'total_local_neurons': int(sum(len(state[e + '_U']) for e in ck['eids'])),
           'selective_cortical_neurons': int(len(X)), 'included_regions': int(len(included)),
           'region_pairs': int(len(xv)), 'spearman_rho': float(rho),
           'ordinary_pairwise_p': float(p_naive), 'qap_p': float(p_qap),
           'regions': included.tolist()}
(HERE / 'panel_f_rank5_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
