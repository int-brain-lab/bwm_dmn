#!/usr/bin/env python3
from pathlib import Path
import textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent; DMN=HERE.parents[1]; LORENZO=HERE.parent; OUT=DMN/'figs'; OUT.mkdir(exist_ok=True)
panels=[
 LORENZO/'sillouette_score/fig3d_published_reproduction.png',
 LORENZO/'sillouette_score/reproduction_audit/maximal_harris_scored_cortical_fig3d.png',
 LORENZO/'sillouette_score/dmn_hierarchy_silhouette.png',
 HERE/'panel_d_audited.png', HERE/'panel_e_audited.png',
 HERE/'panel_f_rank5.png']

caption=("Comparison between Posani et al. and the DMN analyses. "
"a, Posani et al. Fig. 3d reference, showing decreasing RRR selectivity clustering along the cortical hierarchy. "
"b, Local BWM reproduction using a fresh rank-5 RRR fit and all 24 eligible cortical regions with exact Harris hierarchy scores; the dashed line is the one-sided Bonferroni clustering threshold (z=2.87). "
"c, Full-DMN clustering of 21 time-integrated PETH types does not correlate significantly with ordinal hierarchy position; RRR rank is not applicable to this PETH-only analysis. "
"d, Regional mean DMN PETH similarity is weakly related to Harris anatomical connection strength (r=0.12; two-sided region-label QAP P<10^-4; 732 observed pairs). "
"e, On 12,692 UUID-matched neurons, regional similarity in DMN PETH space covaries with similarity in freshly fitted rank-5 RRR time-summed selectivity space (r=0.57; QAP P<10^-4; 13,530 pairs). "
"f, In the full local rank-5 RRR population, anatomically more strongly connected cortical regions have more similar mean selectivity profiles across all 32 eligible regions (Spearman rho=0.39; region-label QAP P<10^-4; 496 pairs). "
"Panel a retains its reference rendering; panels d--f use audited QAP inference because region pairs are not statistically independent.")

fig=plt.figure(figsize=(18,15),facecolor='white')
gs=fig.add_gridspec(3,3,height_ratios=[1,1,.24],hspace=.12,wspace=.06,left=.025,right=.985,top=.985,bottom=.025)
for i,path in enumerate(panels):
 ax=fig.add_subplot(gs[i//3,i%3]); image=plt.imread(path)
 if i == 5:  # remove the source panel's old "d" before relabelling it as f
  image=image.copy(); image[:int(.12*image.shape[0]),:int(.20*image.shape[1]),:3]=1
 ax.imshow(image); ax.axis('off')
 ax.text(-.015,1.025,chr(97+i),transform=ax.transAxes,va='top',ha='left',fontsize=24,fontweight='bold',clip_on=False,
         bbox=dict(facecolor='white',edgecolor='none',alpha=.88,pad=2))
cax=fig.add_subplot(gs[2,:]); cax.axis('off')
cax.text(0,1,"Figure caption",va='top',ha='left',fontsize=13,fontweight='bold')
cax.text(0,.84,textwrap.fill(caption,210),va='top',ha='left',fontsize=10.5,linespacing=1.35)
fig.savefig(OUT/'si_lorenzo.png',dpi=300,facecolor='white')
fig.savefig(OUT/'si_lorenzo.svg',facecolor='white')
(OUT/'si_lorenzo_caption.txt').write_text(caption+'\n')
print(OUT/'si_lorenzo.png'); print(OUT/'si_lorenzo.svg')
