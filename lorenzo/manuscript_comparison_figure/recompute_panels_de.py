#!/usr/bin/env python3
import os
"""Audited DMN anatomical and RRR/PETH similarity comparisons."""
from pathlib import Path
import json, pickle
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from iblatlas.regions import BrainRegions
DATA = Path(os.environ.get("DMN_DATA", Path.home() / "dmn"))  # data stacks
LDATA = DATA / "lorenzo"  # RRR fit, Posani et al.'s code and data (not in the repository)

HERE=Path(__file__).resolve().parent; DMN=HERE.parents[1]; LORENZO=HERE.parent
HARRIS_XLS=LDATA/'41586_2019_1716_MOESM5_ESM.xlsx'
MODEL=LDATA/'sillouette_score/reproduction_audit/RRRGD_rank5_l275_full.pt'
SESSION=LDATA/'brainwide-RRR-encoding-model/trained_RRR_model_paper178/session_features'
N_PERM=10000; SEED=42; MIN_N=5

def qap(xmat,ymat,valid=None):
 n=len(xmat); iu=np.triu_indices(n,1)
 if valid is None: valid=np.ones_like(xmat,dtype=bool)
 mask=valid[iu]&np.isfinite(xmat[iu])&np.isfinite(ymat[iu]); x=xmat[iu][mask]; y=ymat[iu][mask]
 obs=pearsonr(x,y).statistic; rng=np.random.default_rng(SEED); null=np.empty(N_PERM)
 for b in range(N_PERM):
  p=rng.permutation(n); yp=ymat[p][:,p]; null[b]=pearsonr(x,yp[iu][mask]).statistic
 pv=(1+(np.abs(null)>=abs(obs)).sum())/(N_PERM+1)
 return x,y,float(obs),float(pv),null

def save_scatter(x,y,xlabel,ylabel,title,stem,r,p):
 fig,ax=plt.subplots(figsize=(5,4.5)); ax.scatter(x,y,s=8,alpha=.35,c='black',edgecolors='none',rasterized=True)
 m,b=np.polyfit(x,y,1); xx=np.linspace(x.min(),x.max(),100); ax.plot(xx,m*xx+b,c='#d62728',lw=1.8)
 ax.set(xlabel=xlabel,ylabel=ylabel,title=f'{title}\nr={r:.2f}, QAP P={p:.3g}, n={len(x)} region pairs')
 ax.spines[['top','right']].set_visible(False); fig.tight_layout()
 for ext in ('png','pdf','svg'): fig.savefig(HERE/f'{stem}.{ext}',dpi=300,bbox_inches='tight')
 plt.close(fig)

# DMN data and Beryl regional means.
d=np.load(DATA/'concat_cvTrue.npy',allow_pickle=True).flat[0]; uu=d['uuids'].astype(str)
acr=np.asarray(BrainRegions().id2acronym(d['ids'],mapping='Beryl'),str); X=np.asarray(d['concat_z'],np.float32)
areas=[]; means=[]; counts=[]
for a in np.unique(acr):
 m=acr==a
 if a not in ('root','void') and m.sum()>=MIN_N: areas.append(a); means.append(X[m].mean(0)); counts.append(int(m.sum()))
means=np.asarray(means); fcos=cosine_similarity(means); n=len(areas)

# Harris unmasked ipsilateral LOG-NPV, symmetric mean when available.
col='LOG NPV (+0.5min 3.06069269616048E-12)'; hd=pd.read_excel(HARRIS_XLS,sheet_name='all CC connection strengths'); hd=hd[hd.hemisphere=='ipsi']
g=hd.groupby(['Consensus Rbp4 anchor Source','Target'])[col].mean(); hmat=np.full((n,n),np.nan)
for i,a in enumerate(areas):
 for j,b in enumerate(areas):
  vals=[g[k] for k in ((a,b),(b,a)) if k in g.index]
  if vals: hmat[i,j]=np.mean(vals)
xd,yd,rd,pd_q,null_d=qap(fcos,hmat,valid=np.isfinite(hmat))
save_scatter(xd,yd,'Cosine similarity of regional mean DMN PETH','Harris anatomical connection strength (LOG NPV)',
             'DMN functional similarity versus anatomy','panel_d_audited',rd,pd_q)
pd.DataFrame({'peth_cosine_similarity':xd,'harris_log_npv':yd}).to_csv(HERE/'panel_d_pairs.csv',index=False)

# Rank-5 full-fit alpha aligned by UUID, then exact common neuron population.
ck=torch.load(MODEL,map_location='cpu',weights_only=False,mmap=True)['RRRGD_model']; st=ck['model']; vs=st['V'].sum(1)
ru=[]; ra=[]
for eid in ck['eids']:
 with np.load(SESSION/f'{eid}.npz') as z: ru.append(z['uuids'].astype(str))
 ra.append(torch.einsum('nvr,r->nv',st[eid+'_U'],vs).numpy().copy())
ru=np.concatenate(ru); ra=np.concatenate(ra); lookup=pd.Series(np.arange(len(ru)),index=ru)
keep=np.isin(uu,ru); di=np.flatnonzero(keep); ri=lookup.loc[uu[keep]].to_numpy(int); ca=acr[keep]
common=[]; dm=[]; rm=[]; cn=[]
for a in np.unique(ca):
 m=ca==a
 if a not in ('root','void') and m.sum()>=MIN_N:
  common.append(a); dm.append(X[di[m]].mean(0)); rm.append(ra[ri[m]].mean(0)); cn.append(int(m.sum()))
dc=cosine_similarity(np.asarray(dm)); rc=cosine_similarity(np.asarray(rm)); xe,ye,re,pe_q,null_e=qap(dc,rc)
save_scatter(xe,ye,'Region-pair cosine similarity\n(mean DMN PETH vector)',
             'Region-pair cosine similarity\n(mean rank-5 RRR time-summed selectivity)',
             f'Matched functional geometries ({len(di):,}/{len(uu):,} neurons)','panel_e_audited',re,pe_q)
pd.DataFrame({'dmn_cosine_similarity':xe,'rank5_rrr_cosine_similarity':ye}).to_csv(HERE/'panel_e_pairs.csv',index=False)

summary={'panel_d':{'regions_with_at_least_5_neurons':n,'observed_pairs':len(xd),'pearson_r':rd,'qap_p':pd_q,
 'computation':'cosine of full-DMN regional mean concat_z PETH; symmetric mean raw unmasked ipsilateral Harris LOG-NPV'},
 'panel_e':{'matched_neurons':len(di),'regions':len(common),'pairs':len(xe),'pearson_r':re,'qap_p':pe_q,
 'computation':'same UUIDs and Beryl labels; cosine of regional mean concat_z versus fresh full-fit rank-5 signed time-summed beta'},
 'inference':f'two-sided {N_PERM}-permutation region-label QAP, seed {SEED}'}
(HERE/'panels_de_audit.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary,indent=2))
