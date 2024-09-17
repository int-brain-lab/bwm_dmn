from one.api import ONE
from brainbox.singlecell import bin_spikes2D
from brainwidemap import (bwm_query, load_good_units, 
                          load_trials_and_mask, bwm_units)
from iblatlas.atlas import AllenAtlas
from iblatlas.regions import BrainRegions
import iblatlas
from iblatlas.plots import plot_swanson_vector 
from brainbox.io.one import SessionLoader
import ephys_atlas.data
from reproducible_ephys_functions import figure_style, labs

import sys
sys.path.append('Dropbox/scripts/IBL/')
from granger import get_volume, get_centroids, get_res, get_structural, get_ari
from state_space_bwm import get_cmap_bwm, pre_post
from bwm_figs import variverb

from scipy import signal
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import confusion_matrix
from numpy.linalg import norm
from scipy.stats import gaussian_kde, f_oneway, pearsonr, spearmanr, kruskal
from scipy.spatial import distance
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform, cdist
from sklearn.preprocessing import StandardScaler
from random import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
import gc
from pathlib import Path
import random
from copy import deepcopy
import time, sys, math, string, os
from scipy.stats import spearmanr, zscore
import umap
from itertools import combinations, chain
from datetime import datetime
import scipy.ndimage as ndi
import hdbscan

from matplotlib.axis import Axis
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import ListedColormap, LinearSegmentedColormap   
from matplotlib.gridspec import GridSpec   
import mpldatacursor
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import MaxNLocator
from matplotlib.pyplot import cm
from venny4py.venny4py import *

import warnings
warnings.filterwarnings("ignore")
#mpl.use('QtAgg')

# for vari plot
_, b, lab_cols = labs()
plt.ion() 
 
np.set_printoptions(threshold=sys.maxsize)

plt.rcParams.update(plt.rcParamsDefault)
plt.ion()

f_size = 15  # font size

# canonical colors for left and right trial types
blue_left = [0.13850039, 0.41331206, 0.74052025]
red_right = [0.66080672, 0.21526712, 0.23069468]

T_BIN = 0.0125  # bin size [sec] for neural binning
sts = 0.002  # stride size in [sec] for overlapping bins
ntravis = 30  # #trajectories for vis, first 2 real, rest pseudo

one = ONE()

#base_url='https://openalyx.internationalbrainlab.org',
#          password='international', silent=True 
                   
br = BrainRegions()
#units_df = bwm_units(one)  # canonical set of cells


# save results here
pth_dmn = Path(one.cache_dir, 'dmn', 'res')
pth_dmn.mkdir(parents=True, exist_ok=True)

sigl = 0.05  # significance level (for stacking, plotting, fdr)


# order sensitive: must be tts__ = concat_PETHs(pid, get_tts=True).keys()
tts__ = ['inter_trial', 'blockL', 'blockR', 'block50', 'quiescence', 'stimLbLcL', 'stimLbRcL', 'stimLbRcR', 'stimLbLcR', 'stimRbLcL', 'stimRbRcL', 'stimRbRcR', 'stimRbLcR', 'motor_init', 'sLbLchoiceL', 'sLbRchoiceL', 'sLbRchoiceR', 'sLbLchoiceR', 'sRbLchoiceL', 'sRbRchoiceL', 'sRbRchoiceR', 'sRbLchoiceR', 'choiceL', 'choiceR',  'fback1', 'fback0']
#'fback0sRbL', 'fback0sLbR',
     

PETH_types_dict = {
    'concat': [item for item in tts__],
    'resting': ['inter_trial'],
    'quiescence': ['quiescence'],
    'pre-stim-prior': ['blockL', 'blockR'],
    'block50': ['block50'],
    'stim_surp_incon': ['stimLbRcL','stimRbLcR'],
    'stim_surp_con': ['stimLbLcL', 'stimRbRcR'],
    #'resp_surp': ['fback0sRbL', 'fback0sLbR'],
    'motor_init': ['motor_init'],
    'fback1': ['fback1'],
    'fback0': ['fback0']}      


def put_panel_label(ax, k):
    ax.annotate(string.ascii_lowercase[k], (-0.05, 1.15),
                xycoords='axes fraction',
                fontsize=f_size * 1.5, va='top',
                ha='right', weight='bold')


def grad(c, nobs, fr=1):
    '''
    color gradient for plotting trajectories
    c: color map type
    nobs: number of observations
    '''

    cmap = mpl.cm.get_cmap(c)

    return [cmap(fr * (nobs - p) / nobs) for p in range(nobs)]


def eid_probe2pid(eid, probe_name):

    df = bwm_query(one)    
    return df[np.bitwise_and(df['eid'] == eid, 
                             df['probe_name'] == probe_name
                             )]['pid'].values[0]
                        
                             
def cosine_sim(v0, v1):
    # cosine similarity 
    return np.inner(v0,v1)/ (norm(v0) * norm(v1))


def fn2_eid_probe_pid(u):
    '''
    file name u to eid, probe, pid
    '''
    
    return [u.split('_')[0], u.split('_')[1].split('.')[0],
             eid_probe2pid(u.split('_')[0],
                           u.split('_')[1].split('.')[0])]

                         
def get_name(brainregion):
    '''
    get verbose name for brain region acronym
    '''
    regid = br.id[np.argwhere(br.acronym == brainregion)][0, 0]
    return br.name[np.argwhere(br.id == regid)[0, 0]]

    
def get_eid_info(eid):

    '''
    return counter of regions for a given session
    '''
    units_df = bwm_units(one)

    return Counter(units_df[units_df['eid']==eid]['Beryl'])    


def get_sess_per(reg, t='pid'):

    '''
    return bwm insertions that have this region
    '''
    units_df = bwm_units(one)
    print(f'listing {t} per region')
    
    return Counter(units_df[units_df['Beryl'] == reg][t])    
    

def eid2pids(eid):

    
    '''
    return pids for a given eid
    '''
    units_df = bwm_units(one)
    
    return Counter(units_df[units_df['eid'] == eid]['pid'])  


def deep_in_block(trials, pleft, depth=10):

    '''
    get mask for trials object of pleft trials that are 
    "depth" trials into the block
    '''
    
    # pleft trial indices 
    ar = np.arange(len(trials))[trials['probabilityLeft'] == pleft]
    
    # pleft trial indices shifted by depth earlier 
    ar_shift = ar - depth
    
    # trial indices where shifted ones are in block
    ar_ = ar[trials['probabilityLeft'][ar_shift] == pleft]

    # transform into mask for all trials
    bool_array = np.full(len(trials), False, dtype=bool)
    bool_array[ar_] = True
    
    return bool_array


def concat_PETHs(pid, get_tts=False, vers='concat'):

    '''
    for each cell concat all possible PETHs
    
    vers: different PETH set
        vers == 'contrast': extra analyiss for BWM reviewer;
        check for zero contrast effects
        have PETHs aligned to main events; irrespective of type
    
    '''
    
    eid, probe = one.pid2eid(pid)

    # Load in trials data and mask bad trials (False if bad)
    trials, mask = load_trials_and_mask(one, eid,
        saturation_intervals=['saturation_stim_plus04',
                              'saturation_feedback_plus04',
                              'saturation_move_minus02',
                              'saturation_stim_minus04_minus01',
                              'saturation_stim_plus06',
                              'saturation_stim_minus06_plus06'])


    if vers == 'concat':
        # define align, trial type, window length

        # For the 'inter_trial' mask trials with too short iti        
        idcs = [0]+ list(np.where((trials['stimOn_times'].values[1:]
                    - trials['intervals_1'].values[:-1])>1.15)[0]+1)
        mask_iti = [True if i in idcs else False 
            for i in range(len(trials['stimOn_times']))]

        # all sorts of PETHs, some for the surprise conditions
        # need to be 10 trials into a block, see - 10
        
        tts = {

            'inter_trial': ['stimOn_times',
                        np.bitwise_and.reduce([mask, mask_iti]),
                        [1.15, -1]],  
            'blockL': ['stimOn_times', 
                       np.bitwise_and.reduce([mask, 
                       trials['probabilityLeft'] == 0.8]), 
                       [0.4, -0.1]],
            'blockR': ['stimOn_times', 
                       np.bitwise_and.reduce([mask, 
                       trials['probabilityLeft'] == 0.2]),
                       [0.4, -0.1]],
            'block50': ['stimOn_times', 
                       np.bitwise_and.reduce([mask, 
                       trials['probabilityLeft'] == 0.5]),
                       [0.4, -0.1]],                                            
            'quiescence': ['stimOn_times', mask, 
                       [0.4, -0.1]],                       
            'stimLbLcL': ['stimOn_times',
                 np.bitwise_and.reduce([mask, 
                    ~np.isnan(trials[f'contrastLeft']),
                    trials['probabilityLeft'] == 0.8,
                    deep_in_block(trials, 0.8),
                    trials['choice'] == 1]), 
                                        [0, 0.2]], 
            'stimLbRcL': ['stimOn_times',            
                np.bitwise_and.reduce([mask,
                    ~np.isnan(trials[f'contrastLeft']), 
                    trials['probabilityLeft'] == 0.2,                       
                    deep_in_block(trials, 0.2),
                    trials['choice'] == 1]), [0, 0.2]],
            'stimLbRcR': ['stimOn_times',
                 np.bitwise_and.reduce([mask, 
                    ~np.isnan(trials[f'contrastLeft']), 
                    trials['probabilityLeft'] == 0.2,
                    deep_in_block(trials, 0.2),
                    trials['choice'] == -1]), 
                                        [0, 0.2]],           
            'stimLbLcR': ['stimOn_times',
                 np.bitwise_and.reduce([mask,       
                    ~np.isnan(trials[f'contrastLeft']), 
                    trials['probabilityLeft'] == 0.8,
                    deep_in_block(trials, 0.8),
                    trials['choice'] == -1]), 
                                        [0, 0.2]],
            'stimRbLcL': ['stimOn_times',
                 np.bitwise_and.reduce([mask, 
                    ~np.isnan(trials[f'contrastRight']), 
                    trials['probabilityLeft'] == 0.8,
                    deep_in_block(trials, 0.8),
                    trials['choice'] == 1]), 
                                        [0, 0.2]], 
            'stimRbRcL': ['stimOn_times',
                 np.bitwise_and.reduce([mask, 
                    ~np.isnan(trials[f'contrastRight']), 
                    trials['probabilityLeft'] == 0.2,
                    deep_in_block(trials, 0.2),
                    trials['choice'] == 1]), 
                                        [0, 0.2]],
            'stimRbRcR': ['stimOn_times',
                 np.bitwise_and.reduce([mask, 
                    ~np.isnan(trials[f'contrastRight']), 
                    trials['probabilityLeft'] == 0.2,
                    deep_in_block(trials, 0.2),
                    trials['choice'] == -1]), 
                                        [0, 0.2]],        
            'stimRbLcR': ['stimOn_times',
                 np.bitwise_and.reduce([mask, 
                    ~np.isnan(trials[f'contrastRight']), 
                    trials['probabilityLeft'] == 0.8,
                    deep_in_block(trials, 0.8),
                    trials['choice'] == -1]), 
                                        [0, 0.2]],
            'motor_init': ['firstMovement_times', mask, 
                       [0.15, 0]],                                        
            'sLbLchoiceL': ['firstMovement_times',
                 np.bitwise_and.reduce([mask,  
                    ~np.isnan(trials[f'contrastLeft']), 
                    trials['probabilityLeft'] == 0.8,
                    trials['choice'] == 1]), 
                                        [0.15, 0]], 
            'sLbRchoiceL': ['firstMovement_times',
                np.bitwise_and.reduce([mask,
                    ~np.isnan(trials[f'contrastLeft']), 
                    trials['probabilityLeft'] == 0.2,
                    trials['choice'] == 1]), 
                                        [0.15, 0]],
            'sLbRchoiceR': ['firstMovement_times',
                 np.bitwise_and.reduce([mask, 
                    ~np.isnan(trials[f'contrastLeft']), 
                    trials['probabilityLeft'] == 0.2,
                    trials['choice'] == -1]), 
                                        [0.15, 0]],           
            'sLbLchoiceR': ['firstMovement_times',
                 np.bitwise_and.reduce([mask, 
                    ~np.isnan(trials[f'contrastLeft']), 
                    trials['probabilityLeft'] == 0.8,
                    trials['choice'] == -1]), 
                                        [0.15, 0]],
            'sRbLchoiceL': ['firstMovement_times',
                 np.bitwise_and.reduce([mask, 
                    ~np.isnan(trials[f'contrastRight']), 
                    trials['probabilityLeft'] == 0.8,
                    trials['choice'] == 1]), 
                                        [0.15, 0]], 
            'sRbRchoiceL': ['firstMovement_times',
                 np.bitwise_and.reduce([mask, 
                 ~np.isnan(trials[f'contrastRight']), 
                                        trials['probabilityLeft'] == 0.2,
                                        trials['choice'] == 1]), 
                                        [0.15, 0]],
            'sRbRchoiceR': ['firstMovement_times',
                 np.bitwise_and.reduce([mask, 
                 ~np.isnan(trials[f'contrastRight']), 
                                        trials['probabilityLeft'] == 0.2,
                                        trials['choice'] == -1]), 
                                        [0.15, 0]],        
            'sRbLchoiceR': ['firstMovement_times',
                 np.bitwise_and.reduce([mask, 
                 ~np.isnan(trials[f'contrastRight']), 
                                        trials['probabilityLeft'] == 0.8,
                                        trials['choice'] == -1]), 
                                        [0.15, 0]],
            'choiceL': ['firstMovement_times', 
                np.bitwise_and.reduce([mask,
                    trials['choice'] == 1]), 
                        [0, 0.15]],
            'choiceR': ['firstMovement_times', 
                np.bitwise_and.reduce([mask,
                    trials['choice'] == -1]), 
                        [0, 0.15]],            
#            'fback0sRbL': ['feedback_times',    
#                np.bitwise_and.reduce([mask,
#                    trials['feedbackType'] == 0,
#                    ~np.isnan(trials[f'contrastRight']),
#                    trials['probabilityLeft'] == 0.8,
#                    deep_in_block(trials, 0.8)]), 
#                       [0, 0.3]], 
#            'fback0sLbR': ['feedback_times',    
#                np.bitwise_and.reduce([mask,
#                    trials['feedbackType'] == 0,
#                    ~np.isnan(trials[f'contrastLeft']),
#                    trials['probabilityLeft'] == 0.2,
#                    deep_in_block(trials, 0.2)]), 
#                       [0, 0.3]],
            'fback1': ['feedback_times',    
                np.bitwise_and.reduce([mask,
                    trials['feedbackType'] == 1]), 
                       [0, 0.3]],
            'fback0': ['feedback_times', 
                np.bitwise_and.reduce([mask,
                    trials['feedbackType'] == -1]), 
                       [0, 0.3]]}

    elif vers == 'contrast':
        # for latency plot based on coarse PETHs
        # and extra control for auditory signal
        
        tts = {'stim': ['stimOn_times', mask, [0, .15]],         
               'choice': ['firstMovement_times', mask, [0, 0.15]],             
               'fback': ['feedback_times', mask, [0, 0.15]],
               'stim0': ['stimOn_times', np.bitwise_and(mask, 
                            np.bitwise_or(trials[f'contrastRight'] == 0,
                                          trials[f'contrastLeft'] == 0)),
                                          [0, .15]]}       
 
    else:
        print('what set of PETHs??')
        return    
        
        
    if get_tts:
        return tts


    # load in spikes
    spikes, clusters = load_good_units(one, pid)        
    assert len(
            spikes['times']) == len(
            spikes['clusters']), 'spikes != clusters'
            
    D = {}
    D['ids'] = np.array(clusters['atlas_id'])
    D['xyz'] = np.array(clusters[['x','y','z']])
    D['uuids'] = np.array(clusters['uuids'])

    tls = {}  # trial numbers for each type
    ws = []  # list of binned data
    
    for tt in tts:

        event = trials[tts[tt][0]][np.bitwise_and.reduce([mask, tts[tt][1]])]
        tls[tt] = len(event)

        # bin and cut into trials
        # overlapping time bins, bin size = T_BIN, stride = sts
        # tts[key][-1][pre-event time, post-event time]
        bis = []
        st = int(T_BIN // sts)

        for ts in range(st):

            bi, _ = bin_spikes2D(
                spikes['times'],
                clusters['cluster_id'][spikes['clusters']],
                clusters['cluster_id'],
                np.array(event) + ts * sts,
                tts[tt][-1][0], tts[tt][-1][1],
                T_BIN)
            bis.append(bi)

        ntr, nn, nbin = bi.shape
        ar = np.zeros((ntr, nn, st * nbin))

        for ts in range(st):
            ar[:, :, ts::st] = bis[ts]

        # average squared firing rates across trials
        ws.append(np.mean(ar**2, axis=0))        

    D['tls'] = tls
    D['trial_names'] = list(tts.keys())
    D['ws'] = ws  
    return D


def load_atlas_data():
              
    LOCAL_DATA_PATH = Path(one.cache_dir, 'ephys_atlas_data')
    
    D = {}
    (D['df_raw_features'], 
     D['df_clusters'], 
     D['df_channels'], 
     D['df_probes']) = ephys_atlas.data.download_tables(
                        label='latest', 
                        local_path=LOCAL_DATA_PATH, 
                        one=one)                    
                    
    merged_df0 = D['df_raw_features'].merge(D['df_channels'], 
                                     on=['pid','channel'])               
                
    merged_df = merged_df0.merge(
                D['df_clusters'], 
                on=['pid', 'axial_um', 'lateral_um'])                     
                    
    return merged_df       


def get_allen_info(rerun=False):
    '''
    Function to load Allen atlas info, like region colors
    '''
    
    pth_dmna = Path(one.cache_dir, 'dmn', 'alleninfo.npy')
    
    if (not pth_dmna.is_file() or rerun):
        p = (Path(ibllib.__file__).parent /
             'atlas/allen_structure_tree.csv')

        dfa = pd.read_csv(p)

        # replace yellow by brown #767a3a    
        cosmos = []
        cht = []
        
        for i in range(len(dfa)):
            try:
                ind = dfa.iloc[i]['structure_id_path'].split('/')[4]
                cr = br.id2acronym(ind, mapping='Cosmos')[0]
                cosmos.append(cr)
                if cr == 'CB':
                    cht.append('767A3A')
                else:
                    cht.append(dfa.iloc[i]['color_hex_triplet'])    
                        
            except:
                cosmos.append('void')
                cht.append('FFFFFF')
                

        dfa['Cosmos'] = cosmos
        dfa['color_hex_triplet2'] = cht
        
        # get colors per acronym and transfomr into RGB
        dfa['color_hex_triplet2'] = dfa['color_hex_triplet2'].fillna('FFFFFF')
        dfa['color_hex_triplet2'] = dfa['color_hex_triplet2'
                                       ].replace('19399', '19399a')
        dfa['color_hex_triplet2'] = dfa['color_hex_triplet2'].replace(
                                                         '0', 'FFFFFF')
        dfa['color_hex_triplet2'] = '#' + dfa['color_hex_triplet2'].astype(str)
        dfa['color_hex_triplet2'] = dfa['color_hex_triplet2'
                                       ].apply(lambda x:
                                               mpl.colors.to_rgba(x))

        palette = dict(zip(dfa.acronym, dfa.color_hex_triplet2))

        #add layer colors
        bc = ['b', 'g', 'r', 'c', 'm', 'y', 'brown', 'pink']
        for i in range(7):
            palette[str(i)] = bc[i]
        
        palette['thal'] = 'k'    
        r = {}
        r['dfa'] = dfa
        r['palette'] = palette    
        np.save(pth_dmna, r, allow_pickle=True)   

    r = np.load(pth_dmna, allow_pickle=True).flat[0]
    return r['dfa'], r['palette']  


def regional_group(mapping, algo, vers='concat', norm_=False,
                   nclus = 7):

    '''
    mapping: how to color 2d points, say Beryl, layers, kmeans
    find group labels for all cells
    mapping: [Allen, Beryl, Cosmos, layers, clusters, clusters_xyz]
    '''

    r = np.load(Path(pth_dmn, f'{vers}_norm{norm_}.npy'),
                 allow_pickle=True).flat[0]
                 
                              
    # add point names to dict
    r['nums'] = range(len(r[algo][:,0]))
                   

    if mapping == 'kmeans':
        # use kmeans to cluster 2d points
         
        nclus = nclus
        kmeans = KMeans(n_clusters=nclus, random_state=3)
        kmeans.fit(r[algo])
        clusters = kmeans.labels_
        
        cmap = mpl.cm.get_cmap('Spectral')
        cols = cmap(clusters/nclus)
        acs = clusters
        regs = np.unique(clusters)
        
        color_map = dict(zip(list(acs), list(cols)))
        r['els'] = [Line2D([0], [0], color=color_map[reg], 
                    lw=4, label=f'{reg + 1}')
                    for reg in regs]
        
        
        # get average point and color per region
        av = {clus: [np.mean(r[algo][clusters == clus], axis=0), 
                    cmap(clus/nclus)] 
              for clus in range(1,nclus+1)}
              

    elif mapping == 'hdbscan':
        mcs = 10
        clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs)    
        clusterer.fit(r[algo])
        labels = clusterer.labels_
        unique_labels = np.unique(labels)
        mapping = {old_label: new_label 
                      for new_label, old_label in 
                      enumerate(unique_labels)}
        clusters = np.array([mapping[label] for label in labels])

        cmap = mpl.cm.get_cmap('Spectral')
        cols = cmap(clusters/len(unique_labels))
        acs = clusters
        # get average point and color per region
        av = {clus: [np.mean(r[algo][clusters == clus], axis=0), 
                    cols] 
              for clus in range(1,len(unique_labels)+1)} 
        


    elif mapping == 'layers':       
    
        acs = np.array(br.id2acronym(r['ids'], 
                                     mapping='Allen'))
        
        regs0 = Counter(acs)
                                     
        # get regs with number at and of acronym
        regs = [reg for reg in regs0 
                if reg[-1].isdigit()]
        
        for reg in regs:        
            acs[acs == reg] = reg[-1]       
        
        # extra class of thalamic (and hypothalamic) regions 
        names = dict(zip(regs0,[get_name(reg) for reg in regs0]))
        thal = {x:names[x] for x in names if 'thala' in names[x]}
                                          
        for reg in thal: 
            acs[acs == reg] = 'thal'       
        
        mask = np.array([(x.isdigit() or x == 'thal') for x in acs])
        acs[~mask] = '0'
        
        remove_0 = True
        
        if remove_0:
            # also remove layer 6, as there are only 20 neurons 
            zeros = np.arange(len(acs))[
                        np.bitwise_or(acs == '0', acs == '6')]
            for key in r:
                if len(r[key]) == len(acs):
                    r[key] = np.delete(r[key], zeros, axis=0)
                       
            acs = np.delete(acs, zeros)        
        
        _,pa = get_allen_info()
        cols = [pa[reg] for reg in acs]
        regs = Counter(acs)      
        r['els'] = [Line2D([0], [0], color=pa[reg], 
               lw=4, label=f'{reg} {regs[reg]}')
               for reg in regs]
               
        # get average points and color per region
        av = {reg: [np.mean(r[algo][acs == reg], axis=0), pa[reg]] 
              for reg in regs}
               

    elif mapping == 'clusters_xyz':
   
        # use clusters from hierarchical clustering to color
        nclus = 1000
        clusters = fcluster(r['linked_xyz'], t=nclus, 
                            criterion='maxclust')
        cmap = mpl.cm.get_cmap('Spectral')
        cols = cmap(clusters/nclus)
        acs = clusters   
        # get average points per region
        av = {reg: [np.mean(r[algo][clusters == clus], axis=0), 
                    cmap(clus/nclus)] 
              for clus in range(1,nclus+1)}      

    else:
        acs = np.array(br.id2acronym(r['ids'], 
                                     mapping=mapping))
                                     
#        # remove void and root
#        zeros = np.arange(len(acs))[np.bitwise_or(acs == 'root',
#                                                  acs == 'void')]
#        for key in r:
#            if len(r[key]) == len(acs):
#                r[key] = np.delete(r[key], zeros, axis=0)
#                   
#        acs = np.delete(acs, zeros)          
        
                                                              
        _,pa = get_allen_info()
        cols = [pa[reg] for reg in acs]
        
        # get average points and color per region
        regs = Counter(acs)  
        av = {reg: [np.mean(r[algo][acs == reg], axis=0), pa[reg]] 
              for reg in regs}
              

    if 'end' in r['len']:
        del r['len']['end']
              
    r['acs'] = acs
    r['cols'] = cols
    r['av'] = av
              
    return r



def get_umap_dist(rerun=False, algo='umap_z', 
                  mapping='Beryl', vers='concat'):

    pth_ = Path(one.cache_dir, 'dmn', 
                f'{algo}_{mapping}_{vers}_smooth.npy')
    if (not pth_.is_file() or rerun):
        res, regs = smooth_dist(algo=algo, mapping=mapping, vers=vers)    
        d = {'res': res, 'regs' : regs}
        np.save(pth_, d, allow_pickle=True)
    else:
        d = np.load(pth_, allow_pickle=True).flat[0]        
        
    return d     


def get_pw_dist(rerun=False, mapping='Beryl', vers='concat', 
                nclus=7, norm_=False, zscore_=True, nmin=20):

    '''
    get distance for all region pairs by computing
    the Euclidean distance of the feature vectors 
    for all pairs of neurons in the two regions and then 
    average that score 
    '''

    pth_ = Path(one.cache_dir, 'dmn', 
                f'{mapping}_{vers}_zscore_{zscore_}_pw.npy')
                
    if (not pth_.is_file() or rerun):
        r = np.load(Path(pth_dmn, f'{vers}_norm{norm_}.npy'),
                     allow_pickle=True).flat[0] 
        
        vecs = 'concat_z' if zscore_ else 'concat'
                     
        if mapping == 'kmeans':
            # use kmeans to cluster high-dim points
             
            nclus = nclus
            kmeans = KMeans(n_clusters=nclus, random_state=0)
            kmeans.fit(r[vecs])
            acs = kmeans.labels_
            print('kmeans done')
            
        else:
            acs = np.array(br.id2acronym(r['ids'], 
                                         mapping=mapping))

        assert len(r[vecs]) == len(acs), 'mismatch, data != acs'
        
        regs0 = Counter(acs)
        regs = [x for x in regs0 if regs0[x] > nmin]
        res = np.zeros((len(regs),len(regs)))
        print(len(regs), 'regions')
        
        k = 0
        for i in range(len(regs)):
            for j in range(i, len(regs)):
         
                # group of cells a and b
                g_a = r[vecs][acs == regs[i]]
                g_b = r[vecs][acs == regs[j]]

                # compute pairwise distance
                M = cdist(g_a, g_b)
                rows, cols = M.shape
                
                # remove duplicate counts
                mask = np.ones_like(M, dtype=bool)
                min_dim = min(rows, cols)
                mask[:min_dim, :min_dim] = np.triu(
                    np.ones((min_dim, min_dim), dtype=bool), k=1)
                    
                # average across all pairwise scores    
                res[i,j] = np.mean(M[mask])
                res[j,i] = res[i,j]
                
                if np.isnan(res[i,j]):
                    print(regs[i], regs[j],  len(g_a), len(g_b))
                    return
                    
                #res[i,j] = np.mean(M)    
                k += 1
                print(k, 'of', 0.5*(len(regs)**2), 'done')

        d = {'res': res, 'regs' : regs}
        np.save(pth_, d, allow_pickle=True)
    else:
        d = np.load(pth_, allow_pickle=True).flat[0]        
        
    return d



'''
###
### bulk processing
###
'''


def get_all_PETHs(eids_plus=None, vers='concat'):

    '''
    for all BWM insertions, get the PSTHs and acronyms,
    i.e. run get_PETHs
    '''
    
    time00 = time.perf_counter()

    if eids_plus is None:
        df = bwm_query(one)
        eids_plus = df[['eid', 'probe_name', 'pid']].values

    # save results per insertion (eid_probe) in FlatIron folder
    pth = Path(one.cache_dir, 'dmn', vers)
    pth.mkdir(parents=True, exist_ok=True)

    Fs = []
    k = 0
    print(f'Processing {len(eids_plus)} insertions')
    for i in eids_plus:
        eid, probe, pid = i

        time0 = time.perf_counter()
        try:
        
            D = concat_PETHs(pid, vers=vers)
                            
            eid_probe = eid + '_' + probe
            np.save(Path(pth, f'{eid_probe}.npy'), D, 
                    allow_pickle=True)

            gc.collect()
            print(k + 1, 'of', len(eids_plus), 'ok')
        except BaseException:
            Fs.append(pid)
            gc.collect()
            print(k + 1, 'of', len(eids_plus), 'fail', pid)

        time1 = time.perf_counter()
        print(time1 - time0, 'sec')

        k += 1

    time11 = time.perf_counter()
    print((time11 - time00) / 60, f'min for the complete bwm set')
    print(f'{len(Fs)}, load failures:')
    print(Fs)


def stack_concat(vers='concat', get_concat=False, get_tls=False, 
                 ephys=False, norm_=False, n_neighbors=15):

    '''
    stack concatenated PETHs; 
    compute embedding for lower dim
    
    Careful: here 'vers' refers to PETH subsets of concat 
    '''

    pth = Path(one.cache_dir, 'dmn', 'concat')
    ss = os.listdir(pth)  # get insertions
    print(f'combining {len(ss)} insertions for version {vers}') 

    # pool data
    
    r = {}
    for ke in ['ids', 'xyz', 'uuids']:
        r[ke] = []   

    # get PETH type names from first insertion
    D_ = np.load(Path(pth, ss[0]),
                 allow_pickle=True).flat[0]

    # concatenate subsets of PETHs
    
    ttypes = PETH_types_dict[vers]
    tlss = {}  # for stats on trial numbers

    ws = []
    # group results across insertions
    for s in ss:
                   
        eid =  s.split('_')[0]                    
                   
        D_ = np.load(Path(pth, s),
                     allow_pickle=True).flat[0]
        tlss[s] = D_['tls']
        
        if get_tls:
            continue
        
        n_zero_trials = len(np.where(np.array(
                            list(D_['tls'].values())) == 0)[0])
                    
        # remove insertions where a peth is missing (beyond two fbacks)
        if n_zero_trials > 2:
            continue
       
        # pick PETHs to concatenate
        idc = [D_['trial_names'].index(x) for x in ttypes]
        
        if norm_:
            # normalize each PETH type independently
            peths = []
            for x in idc:
                peth = D_['ws'][x]
                mu = np.mean(peth)
                std_ = np.sum(((np.mean(peth,axis=1) - mu)**2))**0.5
                peth_n = (peth - mu)/std_
                peths.append(peth_n)
                if np.isnan(peth_n).flatten().any():
                    print(s,x)
        else:
            peths = [D_['ws'][x] for x in idc]
                     
        # concatenate normalized PETHs             
        ws.append(np.concatenate(peths,axis=1))
        for ke in ['ids', 'xyz', 'uuids']:
            r[ke].append(D_[ke])

    print(len(ws), 'insertions combined')
    
    if get_tls:
        return tlss
    
    
    for ke in ['ids', 'xyz', 'uuids']:  
        r[ke] = np.concatenate(r[ke])
                    
    cs = np.concatenate(ws, axis=0)
    
    # remove cells with nan entries
    goodcells = [~np.isnan(k).any() for k in cs]
    for ke in ['ids', 'xyz', 'uuids']:  
        r[ke] = r[ke][goodcells]       
    cs = cs[goodcells] 

    # remove cells that are zero all the time
    goodcells = [np.any(x) for x in cs]
    for ke in ['ids', 'xyz', 'uuids']:  
        r[ke] = r[ke][goodcells]       
    cs = cs[goodcells]
        
    print(len(cs), 'good cells stacked')

    if get_concat:
        return cs
        
#    r['concat'] = cs

    cs_z = zscore(cs,axis=1)
    r['concat_z'] = cs_z
    
    
    scaler = StandardScaler()
    cs_ss = scaler.fit_transform(cs)
#    r['concat_ss'] = cs_ss
    
    # various dim reduction of PETHs to 2 dims
    print('dimensionality reduction ...')
    ncomp = 2
#    r['umap'] = umap.UMAP(n_components=ncomp).fit_transform(cs)
    r['umap_z'] = umap.UMAP(n_components=ncomp, 
        n_neighbors=n_neighbors).fit_transform(cs_z)
#    r['umap_ss'] = umap.UMAP(n_components=ncomp).fit_transform(cs_ss)
    
    r['len'] = dict(zip(D_['trial_names'],
                    [x.shape[1] for x in D_['ws']]))

    if ephys:
        print('loading and concatenating ephys features ...')

        #  include ephys atlas info
        df = pd.DataFrame({'uuids':r['uuids']})
        merged_df = load_atlas_data()
        dfm = df.merge(merged_df, on=['uuids'])
        
        # remove cells that have no ephys info
        dfr = set(r['uuids']).difference(set(dfm['uuids']))
        rmv = [True if u in dfr else False for u in r['uuids']]

        l0 = deepcopy(len(r['uuids']))
        for key in r:
            if len(r[key]) == l0:
                r[key] = np.delete(r[key], rmv, axis=0)
     
        
        # make ephys feature vector, concat those:
        fts = ['alpha_mean', 'alpha_std', 'depolarisation_slope', 
        'peak_time_secs', 'peak_val', 
        'polarity', 'psd_alpha', 'psd_alpha_csd', 
        'psd_beta', 'psd_beta_csd', 'psd_delta', 
        'psd_delta_csd', 'psd_gamma', 
        'psd_gamma_csd', 'psd_lfp', 'psd_lfp_csd', 
        'psd_theta', 'psd_theta_csd', 
        'recovery_slope', 'recovery_time_secs', 
        'repolarisation_slope', 
        'rms_ap', 'rms_lf', 'rms_lf_csd', 'spike_count_x', 
        'spike_count_y', 'tip_time_secs', 'tip_val', 
        'trough_time_secs', 
        'trough_val']

        r['ephysTF'] = np.array([dfm[dfm['uuids'] == u][fts].values[0] 
                            for u in r['uuids']])
        
        r['fts'] = fts
        
        
        # remove cells with nan/inf/allzero entries
        goodcells = np.bitwise_and.reduce([
                    [~np.isinf(k).any() for k in r['ephysTF']],
                    [np.any(x) for x in r['ephysTF']],
                    [~np.isnan(k).any() for k in r['ephysTF']]])
                    
        l0 = deepcopy(len(r['uuids']))
        for key in r:
            if len(r[key]) == l0:
                r[key] = r[key][goodcells]

        print('hierarchical clustering ...')
        # dim reducing ephys features
        r['umap_e'] = umap.UMAP(
                        n_components=ncomp).fit_transform(r['ephysTF'])      

    np.save(Path(pth_dmn, f'{vers}_norm{norm_}.npy'),
            r, allow_pickle=True)
            
            
def stack_simple(nmin=10):

    '''
    For the latency analysis based on most simple PETHs
    Output PETH per region, latency
    '''
    
    # to fix latency time unit (seg length in sec)          
    pre_post00 = {'stim': [0, 0.15],
                 'choice': [0, 0.15],
                 'fback': [0, 0.15],
                 'stim0': [0, 0.15]}    
    
    pth = Path(one.cache_dir, 'dmn', 'contrast')
    ss = os.listdir(pth)  # get insertions
    print(f'combining {len(ss)} insertions for version contrast') 

    # pool data into df
    
    # get PETH type names from first insertion
    D_ = np.load(Path(pth, ss[0]),
                 allow_pickle=True).flat[0]

    col_keys = ['ids', 'xyz', 'uuids'] + D_['trial_names']
    r = {ke: [] for ke in col_keys}

    # group results across insertions
    for s in ss:           
                   
        D_ = np.load(Path(pth, s),
                     allow_pickle=True).flat[0]
                     
        for ke in ['ids', 'xyz', 'uuids']:
            r[ke].append(D_[ke])
            
        i = 0    
        for ke in D_['trial_names']:
            r[ke].append(D_['ws'][i])
            i += 1

    for ke in r:  
        r[ke] = np.concatenate(r[ke])
                    

    # remove cells with nan entries
    goodcells = np.bitwise_and.reduce(
        [[~np.isnan(k).any() for k in r[ke]] for ke in D_['trial_names']])
        
    for ke in r:  
        r[ke] = r[ke][goodcells]       
                      

    # get average PETH and latency per region
    r['acs'] = np.array(br.id2acronym(r['ids'], 
                                     mapping='Beryl'))    
    
    
    lengths = [len(value) for key, value in r.items() 
        if isinstance(value, (list, np.ndarray))]

    # Check if all elements have the same length
    assert len(set(lengths)) == 1, ("Not all data "
        "elements have the same length.")                    
    
    # get average PETH and latency per region
    rr = {}
    
    regs = np.unique(r['acs'])
    for reg in regs:
        d = {}
        if sum(r['acs'] == reg) < nmin:
            continue
            
        for ke in D_['trial_names']:
            d[ke] = np.mean(r[ke][r['acs'] == reg], axis=0)    
            seg = zscore(d[ke])
            seg = seg - np.min(seg)
            loc = np.where(seg > 0.7 * (np.max(seg)))[0][0]
    
            # convert time unit
            pre,post = pre_post00[ke]
            rrr = np.linspace(0, pre+post,len(seg))

            d[ke+'_lat'] = rrr[loc]
            
        # extra diff
        ke = 'stimdiff'
        d[ke] = np.mean(r['stim'][r['acs'] == reg]
                       -r['stim0'][r['acs'] == reg], axis=0)    
        seg = zscore(d[ke])
        seg = seg - np.min(seg)
        loc = np.where(seg > 0.7 * (np.max(seg)))[0][0]

        # convert time unit
        pre,post = pre_post00['stim']
        rrr = np.linspace(0, pre+post,len(seg))

        d[ke+'_lat'] = rrr[loc]            

        rr[reg] = d                       
                       
    np.save(Path(one.cache_dir, 'dmn', 'stack_simple.npy'),
            rr, allow_pickle=True)
            


'''
#####################################################
### plotting
#####################################################
'''
        

def plot_dim_reduction(algo='umap_z', mapping='Beryl',norm_=False , 
                       means=False, exa=False, shuf=False,
                       exa_squ=False, vers='concat', ax=None, ds=0.5,
                       axx=None, exa_kmeans=False, leg=False, restr=None,
                       nclus = 10, n_neighbors=15):
                       
    '''
    2 dims being pca on concat PETH; 
    colored by region
    algo in ['umap','tSNE','PCA','ICA']
    means: plot average dots per region
    exa: plot some example feature vectors
    exa_squ: highlight example squares in embedding space,
             make and extra plot for each with mean feature vector 
             and those of cells in square in color of mapping
    space: 'concat'  # can also be tSNE, PCA, umap, for distance space
    ds: marker size in main scatter
    restr: list of Beryl regions to restrict plot to
    '''
    
    feat = 'concat_z'
    
    r = regional_group(mapping, algo, vers=vers, norm_=norm_, 
                       nclus=nclus)
    alone = False
    if not ax:
        alone = True
        fig, ax = plt.subplots(label=f'{vers}_{mapping}')
        #ax.set_title(vers)
    
    if shuf:
        shuffle(r['cols'])
    
    if restr:
        # restrict to certain Beryl regions
        #r2 = regional_group('Beryl', algo, vers=vers)
        ff = np.bitwise_or.reduce([r['acs'] == reg for reg in restr]) 
    
    
        im = ax.scatter(r[algo][:,0][ff], r[algo][:,1][ff], 
                        marker='o', c=r['cols'][ff], s=ds, rasterized=True)
                        
    else: 
        im = ax.scatter(r[algo][:,0], r[algo][:,1], 
                        marker='o', c=r['cols'], s=ds,
                        rasterized=True)                            
                        
    
    if means:
        # show means
        emb1 = [r['av'][reg][0][0] for reg in r['av']] 
        emb2 = [r['av'][reg][0][1] for reg in r['av']]
        cs = [r['av'][reg][1] for reg in r['av']]
        ax.scatter(emb1, emb2, marker='o', facecolors='none', 
                   edgecolors=cs, s=600, linewidths=4, rasterized=True)
    
#    ax.set_xlabel(f'{algo} dim1')
#    ax.set_ylabel(f'{algo} dim2')
    zs = True if algo == 'umap_z' else False
    if alone:
        ax.set_title(f'norm: {norm_}, z-score: {zs}')
    ax.axis('off')
    ss = 'shuf' if shuf else ''
       
    
    if mapping in ['layers', 'kmeans']:
        if leg:
            ax.legend(handles=r['els'], ncols=1,
                      frameon=False).set_draggable(True)

    elif 'clusters' in mapping:
        nclus = len(Counter(r['acs']))
        cax = fig.add_axes([0.27, 0.2, 0.5, 0.01])
        norm = mpl.colors.Normalize(vmin=0, 
                                    vmax=nclus)
        cmap = mpl.cm.get_cmap('Spectral')                            
        fig.colorbar(mpl.cm.ScalarMappable(
                                norm=norm, 
                                cmap=cmap), 
                                cax=cax, orientation='horizontal')

    if alone:
        fig.tight_layout()
#    fig.savefig(Path(one.cache_dir,'dmn', 'figs',
#        f'{algo}_{vers}_{mapping}.png'), dpi=150, bbox_inches='tight')


    if exa:
        # plot a cells' feature vector
        # in extra panel when hovering over point
        fig_extra, ax_extra = plt.subplots()
        
        line, = ax_extra.plot(r[feat][0], 
                              label='Extra Line Plot')

        # Define a function to update the extra line plot 
        # based on the selected point
        
        def update_line(event):
            if event.mouseevent.inaxes == ax:
                x_clicked = event.mouseevent.xdata
                y_clicked = event.mouseevent.ydata
                
                selected_point = None
                for key, value in zip(r['nums'], r[algo]):
                    if (abs(value[0] - x_clicked) < 0.01 and 
                       abs(value[1] - y_clicked) < 0.01):
                        selected_point = key
                        break
                
                if selected_point:

                    line.set_data(T_BIN *np.arange(len(r[feat][key])),
                                  r[feat][key])
                    ax_extra.relim()
                    ax_extra.set_ylabel(feat)
                    ax_extra.set_xlabel('time [sec]')
                    ax_extra.autoscale_view()              
                    ax_extra.set_title(
                        f'Line Plot for x,y ='
                        f' {np.round(x_clicked,2), np.round(y_clicked,2)}')
                    fig_extra.canvas.draw()   
    
        # Connect the pick event to the scatter plot
        fig.canvas.mpl_connect('pick_event', update_line)
        im.set_picker(5)  # Set the picker radius for hover detection

    if exa_kmeans:
        # show for each kmeans cluter the mean PETH
        if mapping != 'kmeans':
            print('mapping must be kmeans')
            return
            
        if axx is None:
            fg, axx = plt.subplots(nrows=len(np.unique(r['acs'])),
                                   sharex=True, sharey=False,
                                   figsize=(6,6))
                
        maxys = [np.max(np.mean(r[feat][
                 np.where(r['acs'] == clu)], axis=0)) 
                 for clu in np.unique(r['acs'])]
        
        kk = 0             
        for clu in np.unique(r['acs']):
                    
            #cluster mean
            xx = np.arange(len(r[feat][0])) /480
            yy = np.mean(r[feat][np.where(r['acs'] == clu)], axis=0)

            axx[kk].plot(xx, yy,
                     color=r['cols'][np.where(r['acs'] == clu)][0],
                     linewidth=2)
                     

            
            if kk != (len(np.unique(r['acs'])) - 1):
                axx[kk].axis('off')
            else:

                axx[kk].spines['top'].set_visible(False)
                axx[kk].spines['right'].set_visible(False)
                axx[kk].spines['left'].set_visible(False)      
                axx[kk].tick_params(left=False, labelleft=False)
                
            d2 = {}
            for sec in PETH_types_dict[vers]:
                d2[sec] = r['len'][sec]
                                
            # plot vertical boundaries for windows
            h = 0
            for i in d2:
            
                xv = d2[i] + h
                axx[kk].axvline(xv/480, linestyle='--', linewidth=1,
                            color='grey')
                
                if  kk == 0:            
                    axx[kk].text(xv/480 - d2[i]/(2*480), max(yy),
                             '   '+i, rotation=90, color='k', 
                             fontsize=10, ha='center')
            
                h += d2[i] 
            kk += 1                

#        #axx.set_title(f'{s} \n {len(pts)} points in square')
        axx[kk - 1].set_xlabel('time [sec]')
#        axx.set_ylabel(feat)
        if alone:
            fg.tight_layout()
#        fg.savefig(Path(one.cache_dir,'dmn', 'figs',
#            f'{vers}_kmeans_clusters.png'), dpi=150, bbox_inches='tight')


    if exa_squ:
    
        # get squares
        ns = 10  # number of random square regions of interest
        ss = 0.01  # square side length as a fraction of total area
        x_min = np.floor(np.min(r[algo][:,0]))
        x_max = np.ceil(np.max(r[algo][:,0]))
        y_min = np.floor(np.min(r[algo][:,1]))
        y_max = np.ceil(np.max(r[algo][:,1]))
        
        
        side_length = ss * (x_max - x_min)
        
        sqs = []
        for _ in range(ns):
            # Generate random x and y coordinates within the data range
            x = random.uniform(x_min, x_max - side_length)
            y = random.uniform(y_min, y_max - side_length)
            
            # Create a square represented as (x, y, side_length)
            square = (x, y, side_length)
            
            # Add the square to the list of selected squares
            sqs.append(square)
            

        
        r['nums'] = range(len(r[algo][:,0]))
        
        
        k = 0
        for s in sqs:
    
            
            # get points within square
            
            pts = []
            sq_x, sq_y, side_length = s
            
            for ke, value in zip(r['nums'], r[algo]):
                if ((sq_x <= value[0] <= sq_x + side_length) 
                    and (sq_y <= value[1] <= sq_y + side_length)):
                    pts.append(ke)            
          
            if len(pts) == 0:
                continue
          
            # plot squares in main figure
            rect = plt.Rectangle((s[0], s[1]), s[2], s[2], 
                    fill=False, color='r', linewidth=2)
            ax.add_patch(rect)
          
          
            # plot mean and individual feature line plots
            fg, axx = plt.subplots()          
          
            # each point individually
            maxys = []
            for pt in pts:
                axx.plot(T_BIN * np.arange(len(r[feat][pt])),
                         r[feat][pt],color=r['cols'][pt], linewidth=0.5)
                maxys.append(np.max(r[feat][pt]))         
                         
                
            #square mean
            axx.plot(T_BIN * np.arange(len(r[feat][pt])),
                     np.mean(r[feat][pts],axis=0),
                color='k', linewidth=2)    

            axx.set_title(f'{s} \n {len(pts)} points in square')
            axx.set_xlabel('time [sec]')
            axx.set_ylabel(feat)
            
            # plot vertical boundaries for windows
            h = 0
            for i in r['len']:
            
                xv = r['len'][i] + h
                axx.axvline(T_BIN * xv, linestyle='--',
                            color='grey')
                            
                axx.text(T_BIN * xv, 0.8 * np.max(maxys), 
                         i, rotation=90, 
                         fontsize=12, color='k')
            
                h += r['len'][i]


def smooth_dist(algo='umap_z', mapping='Beryl', show_imgs=False,
                norm_=True, dendro=True, nmin=30, vers='concat'):

    '''
    smooth 2d pointclouds, show per class
    norm_: normalize smoothed image by max brightness
    '''

    r = regional_group(mapping, algo, vers=vers)
    feat = 'concat_z' if algo[-1] == 'z' else 'concat'
    fontsize = 12
    
    # Define grid size and density kernel size
    x_min = np.floor(np.min(r[algo][:,0]))
    x_max = np.ceil(np.max(r[algo][:,0]))
    y_min = np.floor(np.min(r[algo][:,1]))
    y_max = np.ceil(np.max(r[algo][:,1]))
    
    imgs = {}
    xys = {}
    
    regs00 = Counter(r['acs'])
    regcol = {reg: np.array(r['cols'])[r['acs'] == reg][0] 
              for reg in regs00}    

    if mapping == 'Beryl':
        # oder regions 
        p = (Path(iblatlas.__file__).parent / 'beryl.npy')
        regsord = dict(zip(br.id2acronym(np.load(p), 
                           mapping='Beryl'),
                           br.id2acronym(np.load(p), 
                           mapping='Cosmos')))
        regs = []
        
        for reg in regsord:
            if ((reg in regs00) and (regs00[reg] > nmin)):
                regs.append(reg)
    
    else:
        regs = [reg for reg in regs00 if 
                regs00[reg] > nmin]

    for reg in regs:
    
        # scale values to lie within unit interval
        x = (r[algo][np.array(r['acs'])==reg,0] - x_min)/ (x_max - x_min)    
        y = (r[algo][np.array(r['acs'])==reg,1] - y_min)/ (y_max - y_min)

        data = np.array([x,y]).T         
        inds = (data * 255).astype('uint')  # convert to indices

        img = np.zeros((256,256))  # blank image
        for i in np.arange(data.shape[0]):  # draw pixels
            img[inds[i,0], inds[i,1]] += 1
        
        imsm = ndi.gaussian_filter(img.T, (10,10))
        imgs[reg] = imsm/np.max(imsm) if norm_ else imsm
        xys[reg] = [x,y]
  

    if show_imgs:

        # tweak for other mapping than "layers"
        fig, axs = plt.subplots(nrows=3, ncols=len(regs),
                                figsize=(18.6, 5.8))        
        axs = axs.flatten()    
        #[ax.set_axis_off() for ax in axs]

        vmin = np.min([np.min(imgs[reg].flatten()) for reg in imgs])
        vmax = np.max([np.max(imgs[reg].flatten()) for reg in imgs])
        
        k = 0 

        # row of images showing point clouds     
        for reg in imgs:
            axs[k].scatter(xys[reg][0], xys[reg][1], color=regcol[reg], s=0.1)
            axs[k].set_title(f'{reg}, ({regs00[reg]})')
            #axs[k].set_axis_off()
            axs[k].set_aspect('equal')
            axs[k].spines['right'].set_visible(False)
            axs[k].spines['top'].set_visible(False)
            axs[k].set_xlabel('umap dim 1')
            axs[k].set_ylabel('umap dim 2')             
            k+=1
            
        # row of panels showing smoothed point clouds
        for reg in imgs:
            axs[k].imshow(imgs[reg], origin='lower', vmin=vmin, vmax=vmax,
                          interpolation=None)
            axs[k].set_title(f'{reg}, ({regs00[reg]})')
            axs[k].set_axis_off()
            k+=1                            
            
        # row of images showing mean feature vector
        for reg in imgs:
            pts = np.arange(len(r['acs']))[r['acs'] == reg]
            
            xss = T_BIN * np.arange(len(np.mean(r[feat][pts],axis=0)))
            yss = np.mean(r[feat][pts],axis=0)
            yss_err = np.std(r[feat][pts],axis=0)/np.sqrt(len(pts))
                         
            axs[k].fill_between(xss, yss - yss_err, yss + yss_err, 
                                alpha=0.2, color = regcol[reg])    
                
            maxys = [yss + yss_err]  
              
            #region mean
            axs[k].plot(xss,yss, color='k', linewidth=2)    

            axs[k].set_title(reg)
            axs[k].set_xlabel('time [sec]')
            axs[k].set_ylabel(feat)
            axs[k].set_axis_off()      
        
            # plot vertical boundaries for windows
            h = 0
            for i in r['len']:
            
                xv = r['len'][i] + h
                axs[k].axvline(T_BIN * xv, linestyle='--',
                            color='grey', linewidth=0.1)
                            
                axs[k].text(T_BIN * xv, 0.8 * np.max(maxys), 
                         i, rotation=90, 
                         fontsize=5, color='k')
            
                h += r['len'][i]
            
            k+=1
            
        
        fig.suptitle(f'algo: {algo}, mapping: {mapping}, norm:{norm_}')
        fig.tight_layout()    

    # show cosine similarity of density vectors
    

    
    res = np.zeros((len(regs),len(regs)))
    i = 0
    for reg_i in imgs:
        j = 0
        for reg_j in imgs:
            v0 = imgs[reg_i].flatten()
            v1 = imgs[reg_j].flatten()
            
            res[i,j] = cosine_sim(v0, v1)
            j+=1
        i+=1            

    if dendro:
        fig0, axs = plt.subplots(ncols=2, figsize=(10,8), 
            gridspec_kw={'width_ratios': [1, 11]})
        res = np.round(res, decimals=8)
        
        cres = squareform(1 - res)
        linkage_matrix = hierarchy.linkage(cres)
        

        # Order the matrix using the hierarchical clustering
        ordered_indices = hierarchy.leaves_list(linkage_matrix)
        res = res[:, ordered_indices][ordered_indices, :]
        
        row_dendrogram = hierarchy.dendrogram(linkage_matrix,labels =regs,
                     orientation="left", color_threshold=np.inf, ax=axs[0])
        regs = np.array(regs)[ordered_indices]
        
        [t.set_color(i) for (i,t) in    
            zip([regcol[reg] for reg in regs],
                 axs[0].yaxis.get_ticklabels())]
                                     
                     
        ax0 = axs[1]
        
        axs[0].axis('off')
#        axs[0].tick_params(axis='both', labelsize=fontsize)
#        axs[0].spines['top'].set_visible(False)
#        axs[0].spines['bottom'].set_visible(False)    
#        axs[0].spines['right'].set_visible(False)
#        axs[0].spines['left'].set_visible(False)
#        axs[0].set_xticks([])
        
        
    else:
        fig0, ax0 = plt.subplots(figsize=(4,4))
    
                   
    ims = ax0.imshow(res, origin='lower', interpolation=None)
    ax0.set_xticks(np.arange(len(regs)), regs,
                   rotation=90, fontsize=fontsize)
    ax0.set_yticks(np.arange(len(regs)), regs, fontsize=fontsize)               
                   
    [t.set_color(i) for (i,t) in
        zip([regcol[reg] for reg in regs],
        ax0.xaxis.get_ticklabels())] 
         
    [t.set_color(i) for (i,t) in    
        zip([regcol[reg] for reg in regs],
        ax0.yaxis.get_ticklabels())]
    
    #ax0.set_title(f'cosine similarity of smooth images, norm:{norm_}')
    #ax0.set_ylabel(mapping)
    cb = plt.colorbar(ims,fraction=0.046, pad=0.04)
    cb.set_label('regional similarity')
    fig0.tight_layout()
    #fig0.suptitle(f'{algo}, {mapping}')
    
    return res, regs


def plot_ave_PETHs(feat = 'concat', vers='concat', rerun=False):

    '''
    average PETHs across cells
    plot as lines within average trial times
    '''   
    evs = {'stimOn_times':'gray', 'firstMovement_times':'cyan',
           'feedback_times':'orange'}
    
    # needs update; do via if in dict else grey       
    win_cols = {'inter_trial': 'grey',
                 'stimL': [0.13850039, 0.41331206, 0.74052025],
                 'stimR': [0.66080672, 0.21526712, 0.23069468],
                 'blockL': [0.13850039, 0.41331206, 0.74052025],
                 'blockR': [0.66080672, 0.21526712, 0.23069468],
                 'choiceL': [0.13850039, 0.41331206, 0.74052025],
                 'choiceR': [0.66080672, 0.21526712, 0.23069468],
                 'fback1': 'g',
                 'fback0': 'k',
                 'block50': 'grey',
                 'stimLbLcL': 'grey',
                 'stimLbRcL': 'grey',
                 'stimLbRcR': 'grey',
                 'stimLbLcR': 'grey',
                 'stimRbLcL': 'grey',
                 'stimRbRcL': 'grey',
                 'stimRbRcR': 'grey',
                 'stimRbLcR': 'grey',
                 'sLbLchoiceL': 'grey',
                 'sLbRchoiceL': 'grey',
                 'sLbRchoiceR': 'grey',
                 'sLbLchoiceR': 'grey',
                 'sRbLchoiceL': 'grey',
                 'sRbRchoiceL': 'grey',
                 'sRbRchoiceR': 'grey',
                 'sRbLchoiceR': 'grey',
                 'quiescence': 'grey',
                 'motor_init': 'grey'}


    # trial split types, with string to define alignment
    def align(win):
        if ('stim' in win) or ('block' in win):
            return 'stimOn_times'
        elif 'choice' in win:
            return 'firstMovement_times'
        elif 'fback' in win:
            return 'feedback_times'

    def pre_post(win):
        '''
        [pre_time, post_time] relative to alignment event
        split could be contr or restr variant, then
        use base window
        '''

        pid = '1a60a6e1-da99-4d4e-a734-39b1d4544fad'
        tts = concat_PETHs(pid, get_tts=True, vers=vers)
        
        return tts[win][2]


    # get average temporal distances between events    
    pth_dmnm = Path(pth_dmn.parent, 'mean_event_diffs.npy')
    
    if not pth_dmnm.is_file() or rerun:      

        eids = list(np.unique(bwm_query(one)['eid']))
                     
        
        diffs = []
        for eid in eids:
            trials, mask = load_trials_and_mask(one, eid)    
            trials = trials[mask][:-100]
            diffs.append(np.mean(np.diff(
                        trials[list(evs.keys())]),axis=0))
        
        d = {}
        d['mean'] = np.nanmean(diffs,axis=0) 
        d['std'] = np.nanstd(diffs,axis=0)
        d['diffs'] = diffs
        d['av_tr_times'] = [np.cumsum([0]+ list(x)) for x in d['diffs']]

        d['av_times'] = dict(zip(list(evs.keys()), 
                             zip(np.cumsum([0]+ list(d['mean'])),
                                 np.cumsum([0]+ list(d['std'])))))
        
        np.save(pth_dmnm, d, allow_pickle=True)   

    d = np.load(pth_dmnm, allow_pickle=True).flat[0]
    
    fig, ax = plt.subplots(figsize=(8.57, 2.75))
    r = np.load(Path(pth_dmn, 'concat_normTrue.npy'),allow_pickle=True).flat[0]
    r['mean'] = np.mean(r['concat'],axis=0)

    # get alignment event per PETH type
    pid = '1a60a6e1-da99-4d4e-a734-39b1d4544fad'
    ttt =  concat_PETHs(pid = pid,get_tts=True)
    
    # plot trial averages
    yys = []  # to find maxes for annotation
    st = 0
    for tt in ttt:
  
        xx = np.linspace(-ttt[tt][-1][0],
                         ttt[tt][-1][1],
                         r['len'][tt]) + d['av_times'][ttt[tt][0]][0]

        yy = r['mean'][st: st + r['len'][tt]]
        yys.append(max(yy))

        st += r['len'][tt]


        ax.plot(xx, yy, label=tt, color=win_cols[tt])
        ax.annotate(tt, (xx[-1], yy[-1]), color=win_cols[tt])
        
    
    for ev in d['av_times']:
        if ev == 'intervals_1':
            continue
        ax.axvline(x=d['av_times'][ev][0], label=ev,
                   color=evs[ev], linestyle='-')
        ax.annotate(ev, (d['av_times'][ev][0], 0.8*max(yys)), 
                    color=evs[ev], rotation=90, 
                    textcoords='offset points', xytext=(-15, 0))
    
                   
    d['av_tr_times'] = [np.cumsum([0]+ list(x)) 
                        for x in d['diffs']]

#    for s in d['av_tr_times']:
#        k = 0
#        for t in s:
#            ax.axvline(x=t, color=evs[list(evs)[k]], 
#                       linestyle='-', linewidth=0.01)
#            k +=1 
    # Use KDE instead of vertical lines
#    k=0
#    for t in d['av_tr_times']:
#        
#        kde = gaussian_kde(t)
#        ax.plot(t, kde(t), color=evs[list(evs)[k]], alpha=0.5)
#        k += 1            
            
                       
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)    
    ax.set_xlabel('time [sec]')
    ax.set_ylabel('trial averaged fr [Hz]')
    ax.set_title('PETHs averaged across all BWM cells')
    ax.set_xlim(-1.15,3)
    fig.tight_layout()
#    fig.savefig(Path(one.cache_dir,'dmn', 'figs',
#                'intro', 'avg_PETHs.svg'))



def plot_xyz(mapping='Beryl', vers='concat', add_cents=False,
             restr=False, smooth=False, nclus=7, ax = None):

    '''
    3d plot of feature per cell
    add_cents: superimpose stars for region volumes and centroids
    '''

    r = regional_group(mapping, 'umap_z', vers=vers, nclus=nclus)
    xyz = r['xyz']*1000  #convert to mm
    
    alone = False
    if not ax:
        alone = True
        fig = plt.figure(figsize=(8.43,7.26), label=mapping)
        ax = fig.add_subplot(111,projection='3d')

    if isinstance(restr, list):
        idcs = np.bitwise_or.reduce([r['acs'] == reg for reg in restr])   
        xyz = xyz[idcs]
        r['cols'] = np.array(r['cols'])[idcs]
        r['acs'] = np.array(r['acs'])[idcs]
       
    ax.scatter(xyz[:,0], xyz[:,1],xyz[:,2], 
               marker='o', s = 1 if alone else 0.5, c=r['cols'])
               
    if smooth:
        # overlay smooth kernels
        
        reso = 50j
        xi, yi, zi = np.mgrid[
            min((xyz.T)[0]):max((xyz.T)[0]):reso,
            min((xyz.T)[1]):max((xyz.T)[1]):reso,
            min((xyz.T)[2]):max((xyz.T)[2]):reso]
        
        coords = np.vstack([item.ravel() for item in [xi, yi, zi]])
        
        cs = np.unique(r['acs'])
                               
        for c in cs:
              kde_ = gaussian_kde(xyz[r['acs'] == c].T)
              density = kde_(coords).reshape(xi.shape)
              
              # create colormap
              custom_rgba = list(np.array(r['cols'])[r['acs'] == c][0])
              density_flat = density.ravel()
              norm = plt.Normalize(density_flat.min(), density_flat.max())
              colors = list(reversed([tuple(custom_rgba[:3] + [alpha]) 
                            for alpha in np.linspace(0, 1, 256)]))
              custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', 
                     colors, N=256)
                     
              ax.scatter(coords[0], coords[1], coords[2],
                         c=custom_cmap(norm(density_flat)), 
                         s=1, alpha=0.5)
              

    if add_cents:
        # add centroids with size relative to volume
        if mapping !='Beryl':
            print('add cents only for Beryl')
            
        else:    
            regs = list(Counter(r['acs']))
            centsd = get_centroids()
            cents = np.array([centsd[x] for x in regs])          
            volsd = get_volume()
            vols = [volsd[x] for x in regs]
            
            scale = 5000
            vols = scale * np.array(vols)/np.max(vols)
            
            _,pa = get_allen_info()
            cols = [pa[reg] for reg in regs]
            ax.scatter(cents[:,0], cents[:,1], cents[:,2], 
                       marker='*', s = vols, color=cols)
                       
    scalef = 1.2                  
    ax.view_init(elev=45.78, azim=-33.4)
    ax.set_xlim(min(xyz[:,0])/scalef, max(xyz[:,0])/scalef)
    ax.set_ylim(min(xyz[:,1])/scalef, max(xyz[:,1])/scalef)
    ax.set_zlim(min(xyz[:,2])/scalef, max(xyz[:,2])/scalef)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    fontsize = 14
    ax.set_xlabel('x [mm]', fontsize = fontsize)
    ax.set_ylabel('y [mm]', fontsize = fontsize)
    ax.set_zlabel('z [mm]', fontsize = fontsize)
    ax.tick_params(axis='both', labelsize=12)
    #ax.set_title(f'Mapping: {mapping}')
    ax.grid(False)
    nbins = 3
    ax.xaxis.set_major_locator(MaxNLocator(nbins=nbins))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=nbins))
    
    if alone:
        ax.set_title(f'{mapping}_{vers}_{nclus}')
    
#    if alone:
#        fig.tight_layout()
#        fig.savefig(Path(one.cache_dir,'dmn', 'imgs',
#            f'{mapping}_{vers}_{nclus}_3d.png'),dpi=150)


def clus_grid():

    fig = plt.figure(figsize=(14,10))
    
    axs = []
    for cl in range(7):
        axs.append(fig.add_subplot(2, 4, cl + 1, projection='3d'))
        plot_xyz(mapping='kmeans',restr=[cl], ax=axs[-1])    



def plot_sim():

    '''
    Group and plot cosine similarity results
    
    nmin: minimum number of neurons per region to be included
    sessmin: min number of sessions with region combi
    '''
    

    res, regs = smooth_dist(algo='umap_z', mapping='Beryl')
    
    regsl = list(regs)
    dm = {}
    for i in range(len(regs)):
        for j in range(len(regs)):
            if i == j:
                continue
            dm[f'{regsl[i]}_{regsl[j]}'] = res[i,j]    


    dm_sorted = dict(sorted(dm.items(), key=lambda item: item[1]))

    exs = np.concatenate([list(dm_sorted.keys())[:5],
                          list(dm_sorted.keys())[-150:]])
    
    d_exs = {x:dm[x] for x in exs}

    fig, ax = plt.subplots(figsize=(5,4))
    ax.plot(np.arange(len(d_exs)), d_exs.values(), 
            marker='o', linestyle='')
    ax.set_xticks(np.arange(len(d_exs)))       
    ax.set_xticklabels(list(d_exs.keys()), rotation=90)
    ax.set_title(f'min and max region pairs')
    ax.set_ylabel('cosine similarity')
    fig.tight_layout()


def plot_connectivity_matrix(metric='umap_z', mapping='Beryl',
                             vers = 'concat', ax0=None, ari=False, rerun=True):

    '''
    all-to-all matrix for some measures
    '''


    if metric == 'cartesian':
        d = get_centroids(dist_=False)
    elif metric == 'pw':
        d = get_pw_dist(mapping=mapping, vers=vers)        
    else:     
        d = get_umap_dist(algo=metric, vers=vers, rerun=rerun)
                
    res = d['res']
    regs = d['regs']
    
    _,pal = get_allen_info()
    
    alone = False
    if not ax0:
        alone=True
        fig, (ax_dendro, ax0) = plt.subplots(1, 2, 
            figsize=(5, 4), 
            gridspec_kw={'width_ratios': [1, 5]})
    
    if ari:
        rs = get_ari()
    
        ints = []
        for reg in rs:
            if reg in regs:
                ints.append(reg)
        
        rems = [reg for reg in regs if reg not in ints] 
        print(list(ints)[0], rems[0])
        node_order = list(ints) + rems
        
        ordered_indices = [list(regs).index(reg) for reg in node_order]
        regs = np.array(regs)[ordered_indices]
        res = res[:, ordered_indices][ordered_indices, :]        
        
    else:    
        # Order the matrix using hierarchical clustering
        
        cres = squareform(np.max(res) - res)
        linkage_matrix = hierarchy.linkage(cres)
        ordered_indices = hierarchy.leaves_list(linkage_matrix)
            
            
        regs = np.array(regs)[ordered_indices]    
        res = res[:, ordered_indices][ordered_indices, :]
           
    ims = ax0.imshow(res, origin='lower', interpolation=None)
    ax0.set_xticks(np.arange(len(regs)), regs,
                   rotation=90, fontsize=5)
    ax0.set_yticks(np.arange(len(regs)), regs, fontsize=5)               
                   
    [t.set_color(i) for (i,t) in
        zip([pal[reg] for reg in regs],
        ax0.xaxis.get_ticklabels())] 
         
    [t.set_color(i) for (i,t) in    
        zip([pal[reg] for reg in regs],
        ax0.yaxis.get_ticklabels())]
    
    if metric[-1] == 'e':
        vers = '30 ephysAtlas'
        
    ax0.set_title(f'{metric}, {vers}')
    #ax0.set_ylabel(mapping)
    cbar = plt.colorbar(ims,fraction=0.046, pad=0.04, 
                        extend='neither')#, ticks=[0, 0.5, 1]
    #cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=3))

    if not ari:
        # plot dendrogram
        with plt.rc_context({'lines.linewidth': 0.5}):
            hierarchy.dendrogram(linkage_matrix, ax=ax_dendro, 
                orientation='left', labels=regs)

            
        ax_dendro.set_axis_off()
    
#    ax_dendro.set_yticklabels(regs)
#    [ax_dendro.spines[s].set_visible(False) for s in
#        ['left', 'right', 'top', 'bottom']]
#    ax_dendro.get_xaxis().set_visible(False)
#    [t.set_color(i) for (i,t) in    
#        zip([pal[reg] for reg in regs],
#        ax_dendro.yaxis.get_ticklabels())]    
    
    plt.subplots_adjust(wspace=0.05)
    
    if alone:
        fig.tight_layout()

    #fig0.suptitle(f'{algo}, {mapping}')
    else:
        return ordered_indices
    

def plot_multi_matrices(ticktype='rectangles', add_clus=True, 
                        get_matrices=False, rerun=False):

    '''
    for various subsets of the PETHs 
    plot adjacency all in one row
    repeat rows, each ordered by the type of the row index
    
    add_clus: add a row of cluster 2d scatter plots  
    '''


    pth_matrices = Path(one.cache_dir, 'dmn', 'd.npy')
    
    if (not pth_matrices.is_file() or rerun):        
        #verss = list(PETH_types_dict.keys()) + ['cartesian']
        verss = ['concat','stim_surp_incon', 'resting']
        D = {}
        for vers in verss:
            if vers == 'cartesian':
                D[vers] = trans_(get_centroids(dist_=True))
            elif vers == 'ephysAtlas':
                D[vers] = trans_(get_umap_dist(algo='umap_e', vers='concat'))
            elif vers == 'pw':
                D[vers] = trans_(get_pw_dist(vers='concat'))   
            else:     
                D[vers] = trans_(get_umap_dist(algo='umap_z', vers=vers))

        # get intersection of regions across versions
        set_list = [set(list(D[vers].keys())) for vers in D]
        pairs = list(set.intersection(*set_list))
        regs = list(Counter(np.array([p.split(' --> ') 
                    for p in pairs]).flatten()))
        
          
        D2 = {}
        for vers in D:
            res = np.ones((len(regs), len(regs)))   
            for i in range(len(regs)):
                for j in range(len(regs)):
                    if i == j:
                        continue
                    res[i,j] = D[vers][f'{regs[i]} --> {regs[j]}']
            
            D2[vers] = res
            
        D2['regs'] = regs         
        np.save(pth_matrices, D2, allow_pickle=True)   

    D2 = np.load(pth_matrices, allow_pickle=True).flat[0]

    if get_matrices:
        return D2    

    verss = list(D2.keys())
    verss.remove('regs')
    regs = D2['regs']
    
    _,pal = get_allen_info()
    nrows = len(verss)+1 if add_clus else len(verss)
    fig, axs = plt.subplots(nrows=nrows, ncols=len(verss),
                            figsize=(8.67,9.77))
    axs = axs.flatten()
    
    k = 0 
    for row in range(len(verss)):   
        # use dendro order of first version for all 
        res0 = np.round(D2[verss[row]], decimals=10)
        
        # turn similarity into distance and put into vector form 
        cres = squareform(np.max(res0) - res0)
        linkage_matrix = hierarchy.linkage(cres)
        ordered_indices = hierarchy.leaves_list(linkage_matrix) 
        regso = np.array(regs)[ordered_indices]
        
            
        for col in range(len(verss)):
           
            res = D2[verss[col]][:, ordered_indices][ordered_indices, :]
                   
            ims = axs[k].imshow(res, origin='lower', interpolation=None)
            
            
            if ticktype == 'acronyms':
                axs[k].set_xticks(np.arange(len(regso)), regso,
                               rotation=90)
                axs[k].set_yticks(np.arange(len(regso)), regso)               
                               
                [t.set_color(i) for (i,t) in
                    zip([pal[reg] for reg in regso],
                    axs[k].xaxis.get_ticklabels())] 
                     
                [t.set_color(i) for (i,t) in    
                    zip([pal[reg] for reg in regso],
                    axs[k].yaxis.get_ticklabels())]
            
            else:
                # plot region rectangles
                rect_height = 15 
                data_height = len(regso)
                
                x_tick_colors = [to_rgba(pal[reg]) for reg in regso]
                axs[k].axis('off')

                for i, color in enumerate(x_tick_colors):
                    rect = Rectangle((i - 0.5, -rect_height - 0.5), 1, 
                               rect_height, color=color, clip_on=False,
                               transform=axs[k].transData)
                    axs[k].add_patch(rect)

                # Create colored rectangles for y-axis ticks
                y_tick_colors = [to_rgba(pal[reg]) for reg in regso]


                for i, color in enumerate(y_tick_colors):
                    rect = Rectangle((-rect_height - 0.5, i - 0.5),
                                      rect_height, 
                                      1, color=color, clip_on=False,
                                      transform=axs[k].transData)
                    axs[k].add_patch(rect)         
                        
            if k < len(verss):    
                axs[k].set_title(verss[col])
            k += 1


    if add_clus:
        # add cluster 2d plots as bottom row
        plot_dist_clusters(anno=False, axs=axs[-len(verss):])
        [ax.axis('off') for ax in axs]
        
#    #fig.suptitle('all matrices ordered by data of row number')
#    fig.tight_layout()
#    fig.savefig(Path(one.cache_dir,'dmn', 'figs','matrices.svg'))    
#    fig.savefig(Path(one.cache_dir,'dmn', 'figs','matrices.pdf'),
#                dpi=150)


def plot_dendrograms():

    """
    Displays a dendrogram for each 'vers' in 'verss' with colored x-tick labels.
    
    Parameters:
    - verss: Versions to plot dendrograms for.
    - D: Precomputed distances or similarities for each 'vers'.
    - regs: List of brain regions.
    - pal: Palette dictionary mapping regions to colors.
    """
    
    
    verss = list(PETH_types_dict.keys()) + ['cartesian','ephysAtlas']
    D = {}
    for vers in verss:
        if vers == 'cartesian':
            D[vers] = trans_(get_centroids(dist_=True))
        elif vers == 'ephysAtlas':
            D[vers] = trans_(get_umap_dist(algo='umap_e', vers='concat'))
        else:     
            D[vers] = trans_(get_umap_dist(algo='umap_z', vers=vers))

    # get intersection of regions across versions
    set_list = [set(list(D[vers].keys())) for vers in D]
    pairs = list(set.intersection(*set_list))
    regs = list(Counter(np.array([p.split(' --> ') 
                for p in pairs]).flatten()))
    
    D2 = {}
    for vers in D:
        res = np.ones((len(regs), len(regs)))   
        for i in range(len(regs)):
            for j in range(len(regs)):
                if i == j:
                    continue
                res[i,j] = D[vers][f'{regs[i]} --> {regs[j]}']
        
        D2[vers] = res
        
    _,pal = get_allen_info()         
    nrows = len(verss)
    fig, axs = plt.subplots(nrows=nrows, figsize=(10, 7.93))

    cmap = cm.Greys(np.linspace(0.1, 1, 10))
    hierarchy.set_link_color_palette([mpl.colors.rgb2hex(rgb[:3]) for rgb in cmap])


    if nrows == 1:
        axs = [axs]  # Ensure axs is iterable for a single row

    for ax, vers in zip(axs, verss):
        # Compute hierarchical clustering and order regions
        
        # turn similarity matrix into distance matrix
        res = np.amax(D2[vers]) - D2[vers]
        cres = squareform(res)
        
        linkage_matrix = hierarchy.linkage(cres)
            
        dendro = hierarchy.dendrogram(linkage_matrix, ax=ax,
            above_threshold_color='k', orientation='top')
            
            
        # Set x-tick labels with brain regions, ordered by dendrogram
        ordered_regions = [regs[i] for i in dendro['leaves']]
        ax.set_xticklabels(ordered_regions, rotation=90, fontsize=5.6)

        # Color x-tick labels based on brain regions
        for xtick, reg in zip(ax.get_xticklabels(), ordered_regions):
            xtick.set_color(pal[reg])

        # ax.set_title(f"{vers}")
        ax.text(0.5, 0.9, f"{vers}", transform=ax.transAxes, 
        ha='center', va='bottom', fontsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_yticks([])
        
    fig.subplots_adjust(
top=0.965,
bottom=0.069,
left=0.016,
right=0.984,
hspace=1.0,
wspace=0.335)

    fig.savefig(Path(one.cache_dir,'dmn', 'figs','dendros.svg'))    
    fig.savefig(Path(one.cache_dir,'dmn', 'figs','dendros.pdf'),
                dpi=150)


def trans_(d):
    '''
    turn adjacency matrix into A --> B format dictionary
    '''
    d0 = {}
    res = d['res']
    regs = d['regs']
    for i in range(len(regs)):
        for j in range(len(regs)):
            if i == j:
                continue
            d0[f'{regs[i]} --> {regs[j]}'] = res[i,j]
                
    return d0
   

def scatter_Beryl_similarity(ranks=False, hexbin_=False, anno=False):

    '''
    for pairs of two similarity metrics, 
    scatter and correlate region pairs;
    looking for max intersection of data
    '''

    D = {'cartesian': trans_(get_centroids(dist_=True)),
#         'concat': trans_(get_umap_dist(algo='umap_z', 
#                                                vers='concat')),
#         'surprise': trans_(get_umap_dist(algo='umap_z', 
#                                                vers='surprise')),
#         'reward': trans_(get_umap_dist(algo='umap_z',
#                                                vers='reward')),
#         'quiescence': trans_(get_umap_dist(algo='umap_z',
#                                                vers='quiescence')),
#         'resting': trans_(get_umap_dist(algo='umap_z',
#                                                vers='resting')),
#         '30ephys': trans_(get_umap_dist(algo='umap_e')),
#         #'coherence': get_res(metric='coherence', 
                              #sig_only=True, combine_=True),
         'granger': get_res(metric='granger', 
                            sig_only=True, combine_=True),
         #'structural3_sp': get_structural(fign=3, shortestp=True),
         'axonal': get_structural(fign=3)
         }
     
#    tt = len(list(combinations(list(D.keys()),2)))
#    nrows = 1 if tt == 1 else 4
#    ncols = 1 if tt == 1 else tt//nrows 
        
    nrows = 3
    ncols = 1    
        
     
    fig, ax = plt.subplots(ncols=ncols, nrows=nrows,
                           figsize=[10.34, 9.74])     
    ax = np.array(ax).flatten()
    
    metrics = list(D.keys())   
    nf = list(combinations(range(len(D)),2))
     
    for k in range(len(nf)):
       
        dg,dc = D[metrics[nf[k][0]]], D[metrics[nf[k][1]]]
                        
        pairs = list(set(dg.keys()).intersection(set(dc.keys())))
        
        pts = []
        gs = []
        cs = []
        
        for p in pairs:
            gs.append(np.mean(dg[p]))
            cs.append(np.mean(dc[p]))
            pts.append(p)        

        gs = gs
        cs = cs
        pts = pts
        
        corp,pp = pearsonr(gs, cs)
        cors,ps = spearmanr(gs, cs)
        

        if ranks:
            gs = np.argsort(np.argsort(gs))
            cs = np.argsort(np.argsort(cs))

            
        if hexbin_:
            ax[k].hexbin(gs, cs, cmap='Greys', gridsize=150)
        else:                
            ax[k].scatter(gs, cs, color='b' if ranks else 'k', 
                          s=0.5, alpha=0.1, rasterized=True)

        if anno:
     
            for i in range(len(pts)):
                ax[k].annotate('  ' + pts[i], 
                    (gs[i], cs[i]),
                    fontsize=5,color='b' if ranks else 'k')
                       
        a = 'ranks' if ranks else ''
        ax[k].set_xlabel(f'{metrics[nf[k][0]]} ' + a)       
        ax[k].set_ylabel(f'{metrics[nf[k][1]]} ' + a)
        ax[k].spines['right'].set_visible(False)
        ax[k].spines['top'].set_visible(False)
        ss = (f"{np.round(corp,2) if pp<0.05 else '_'}, "
              f"{np.round(cors,2) if ps<0.05 else '_'}")
        ax[k].set_title(ss + f'\n {len(pts)}')
    
        print(metrics[nf[k][0]], metrics[nf[k][1]], len(pts))
        print(f'pe: (r,p)=({np.round(corp,2)},{np.round(pp,2)})')
        print(f'sp: (r,p)=({np.round(cors,2)},{np.round(ps,2)})')
        
    # Check if axes is taken, if not, switch axes off
    [a.axis('off') for a in ax if not a.title.get_text()]
            
            
#    fig.tight_layout()
#    fig.savefig(Path(one.cache_dir,'dmn', 'figs',
#                    'scatters.svg'))
#    fig.savefig(Path(one.cache_dir,'dmn', 'figs',
#                    'scatters.pdf'), dpi=250,
#                     bbox_inches='tight', format='pdf')

    
def plot_venn():

    '''
    illustrate overlap of region pair data
    '''
    
    D = {#'euc_centroid': trans_(get_centroids(dist_=True)),
         'umap_z_concat': trans_(get_umap_dist(algo='umap_z', vers='concat')),
         'umap_z_reward': trans_(get_umap_dist(algo='umap_z', vers='reward')),
         'umap_z_surprise': trans_(get_umap_dist(algo='umap_z',
                                     vers='surprise')),
         'umap_z_resting': trans_(get_umap_dist(algo='umap_z',
                                     vers='resting'))}#,
##         'coherence': get_res(metric='coherence', 
##                              sig_only=True, combine_=True),
#         'granger': get_res(metric='granger', 
#                            sig_only=True, combine_=True),          
#         #'structural3': get_structural(fign=3, rerun=True),
#         'structural4': get_structural(fign=4, rerun=True)}        
    
    
    sets = dict(zip(list(D.keys()), 
                    [set(list(D[s].keys())) for s in D]))
    
    venny4py(sets=sets)



def swansons_all(metric='latency', minreg=10, annotate=True,
             vers='contrast', restrict=False, thres=10):

    '''
    Per window of the BWM, average PETHS, get latency (latency_all) or fr
    put on swanson, one per aligment event
    
    if metric == 'ephysTF', plot region average score
    of each of the 30 features (or just two if restric=True)


    Can also be used as extra figure for BWM rebuttal on latency alternative
    metric ='latency' with vers = 'contrast'
    
    '''
    if metric == 'latency':

        r = np.load(Path(one.cache_dir, 'dmn', 'stack_simple.npy'),
                    allow_pickle=True).flat[0]

        xs = {'stim': np.linspace(0,0.15,len(r['MRN']['stim'])),
              'choice': np.linspace(0,0.15,len(r['MRN']['choice'])),
              'fback': np.linspace(0,0.15,len(r['MRN']['fback'])),
              'stim0': np.linspace(0,0.15,len(r['MRN']['stim0']))}
        
        del r['root']
        del r['void']
        regs2 = list(r.keys())
        avs = r
        lattypes = [x for x in r['MRN'].keys() if 'lat' in x][:3]
        reg = 'MRN'
        
        
    else:
        r = regional_group('Beryl', 'umap_z', vers=vers)    

        # get average z-scored PETHs per Beryl region 
        regs = Counter(r['acs'])
        regs2 = [reg for reg in regs if regs[reg]>minreg]

    # average all PETHs per region, then z-score and get latency
    # plot latency in swanson; put average peths on top

        avs = {}
        for reg in regs2:
        
            if metric == 'ephysTF':
                orgl = np.mean(r['ephysTF'][r['acs'] == reg],axis=0)
                avs[reg] = dict(zip(r['fts'],orgl))
       
            else:
                orgl = np.mean(r['concat'][r['acs'] == reg],axis=0)
                lats = []              
                for length in r['len'].values():
                    seg = orgl[:length]
                    frs.append(np.max(seg))
                    seg = zscore(seg)
                    seg = seg - np.min(seg)
                    loc = np.where(seg > 0.7 * (np.max(seg)))[0][0]
                    lats.append(loc)
                    orgl = orgl[length:]     
                
                avs[reg] = dict(zip(list(r['len'].keys()),
                                    lats if metric == 'latency_all' else frs))
    
    if metric == 'latency':
        nrows = 1
        ncols = 3
    else:
        nrows = 3
        ncols = len(avs[reg].keys())//nrows

    if restrict:                           
        # restrict to example features
        features_to_keep = ['psd_alpha', 'trough_time_secs']
                                
        restricted_data = {region: {feature: avs[region][feature]
            for feature in features_to_keep} for region in avs}
            
        avs = restricted_data    
        ncols = len(features_to_keep)
        nrows = 1
        
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(9.72, 8.4),
                            sharex=True if metric!= 'ephysTF' else False, 
                            sharey=True if metric!= 'ephysTF' else False)
    axs = axs.flatten('F')
    

    
    if metric!= 'ephysTF':
        # assure all panels have same scale
        lats_all = np.array([np.array([avs[x][s] for x in avs]) 
                    for s in avs[regs2[0]].keys() if 'lat' in s]).flatten()
        lats_all = lats_all * 1000 # ms    
        vmin, vmax = (np.nanmin(lats_all), np.nanmax(lats_all))
   
    
    # loop through PETH types
    k = 0
    
    for s in lattypes:
        
        if metric == 'latency':
            #cmap_ = get_cmap_bwm(s).reversed()
            cmap_ = 'viridis_r'
        else:
            cmap_ = 'viridis'        
        

        lats = np.array([avs[x][s] for x in avs]) * 1000 # ms
        
        print(s)
        asort = np.argsort(lats)
        print(lats[asort][:10])
        print(np.array(list(avs.keys()))[asort][:10])
        
               
        vmin = vmin if metric != 'ephysTF' else np.min(lats)
        vmax = vmax if metric != 'ephysTF' else np.max(lats)
        norm = mpl.colors.Normalize(vmin=vmin, 
                                    vmax=vmax)        
        
        plot_swanson_vector(np.array(list(avs.keys())),
                            np.array(lats), 
                            cmap=cmap_, 
                            ax=axs[k], br=br, 
                            orientation='portrait',
                            vmin=vmin, 
                            vmax=vmax,
                            annotate= annotate,
                            annotate_list=None,
                            annotate_n=500,
                            thres=thres)

        axs[k].axes.invert_xaxis()                     
        cbar = fig.colorbar(mpl.cm.ScalarMappable(
                                norm=norm, 
                                cmap=cmap_), 
                                ax=axs[k],
                                location='bottom', pad=0.04)
        cbar.formatter = ScalarFormatter(useMathText=True)
        cbar.set_ticks([0, 75, 150])
        cbar.set_ticklabels(['0', '75', '150'])
        cbar.set_label('latency [ms]' 
                       if metric == 'latency' 
                       else 'fr [Hz]')
                       
        axs[k].axis('off')
        axs[k].set_title(variverb[s.split('_')[0]])
        
        # change annotation fontsize
        text_objects = axs[k].texts
        for text_obj in text_objects:
            text_obj.set_fontsize(8)
        
        #put_panel_label(axs[k], k)
        k+=1

    fig.subplots_adjust(top=.955,
                        bottom=0.0,
                        left=0.031,
                        right=0.977,
                        hspace=0.2,
                        wspace=0.106)
                        
    print(metric, vers)
    
#    fig.savefig(Path(one.cache_dir,'dmn', 'figs', 'intro', 
#                 'ephysTF_example_swansons.svg'))

    if metric == 'latency':
#        fig.savefig(Path(one.cache_dir, 'bwm_res', 'bwm_figs_imgs',
#                         'si', 'n6_supp_figure_peth_latency_swanson.svg'))
        fig.savefig(Path(one.cache_dir, 'bwm_res', 'bwm_figs_imgs',
                         'si', 'n6_supp_figure_peth_latency_swanson.pdf'),
                        dpi=150,bbox_inches='tight')                         

    #verss = ['concat', 'surprise', 'reward', 'resting']



def swansons_means(minreg=10, annotate=True, nanno=5):

    '''
    Plot on swansons mean PETH for 5 types
    and differences from concat
    '''
    
    r = regional_group('Beryl', 'umap_z', vers='concat')
     
    # get average z-scored PETHs per Beryl region 
    regs = Counter(r['acs'])
    regs2 = [reg for reg in regs if regs[reg]>minreg]

    # average all PETHs per region, then z-score and get latency
    # plot latency in swanson; put average peths on top
    avs = {}

    for reg in regs2:
    
        # average across neurons per region
        orgl = np.mean(r['concat'][r['acs'] == reg],axis=0)   

        rd = {}

        # cumulative [start, end] indices of each segment
        start_end = {}
        start_idx = 0

        # Calculate cumulative [start, end] indices for each segment
        for key, length in r['len'].items():
            end_idx = start_idx + length 
            start_end[key] = [start_idx, end_idx]
            # Update the start index for the next segment
            start_idx += length

        for subset, segments in PETH_types_dict.items():
            # Calculate start and end indices for the current subset
            ranges = []
            for seg in segments:
                ranges.append(np.arange(start_end[seg][0], 
                                        start_end[seg][1]))
                                        
            # take mean of subset PETHs (average across time bins)
            rd[subset] = np.mean(orgl[np.concatenate(ranges)])
            
        avs[reg] = rd

    # add for each condition the difference to "concat"
    conds = list(avs[reg].keys())
    avs_d = {}
    for reg in avs:
        dd = {}
        for c in conds:
            dd[f'concat-{c}'] = abs(avs[reg]['concat'] - avs[reg][c])
        avs_d[reg] = dd
     
    nrows = 2
    ncols = len(avs[reg].keys())
        
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, 
                            figsize=[8.33, 8.61])

    cmap_ = 'viridis_r'
    

    lats_all = np.array([np.array([avs[x][s] for x in avs]) 
                for s in avs[reg].keys()]).flatten()
    lats_all_d = np.array([np.array([avs_d[x][s] for x in avs_d]) 
                for s in avs_d[reg].keys()]).flatten()
                
    vmin, vmax = (np.nanmin(lats_all), np.nanmax(lats_all))
    vmin_d, vmax_d = (np.nanmin(lats_all_d), np.nanmax(lats_all_d))
    
    # loop through PETH subset types
    k = 0  
    for s in avs[reg].keys():     
        
        print('means')
        
        aord = np.argsort(np.array([avs[x][s] for x in avs]))
        print(s, list(reversed(
                    np.array(list(avs.keys()))[aord][-nanno:])))
        
        plot_swanson_vector(np.array(list(avs.keys())),
                            np.array([avs[x][s] for x in avs]), 
                            cmap=cmap_, 
                            ax=axs[0,k], br=br, 
                            orientation='portrait',
                            vmin=vmin, 
                            vmax=vmax,
                            thres=20000,
                            annotate= annotate,
                            annotate_n=nanno,
                            annotate_order='top')
                            
        norm = mpl.colors.Normalize(vmin=vmin, 
                                    vmax=vmax)        

        num_ticks = 3  # Adjust as needed

        # Use MaxNLocator to select a suitable number of ticks
        locator = MaxNLocator(nbins=num_ticks)
                   
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(
                   mpl.cm.ScalarMappable(norm=norm,cmap=cmap_),
                   ax=axs[0,k],shrink=0.8,aspect=12,pad=.025,
                   orientation="horizontal", ticks=locator)
                   
        cbar.ax.tick_params(axis='both', which='major', size=6)
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(size=2)
        cbar.ax.xaxis.set_tick_params(pad=5)
        #cbar.set_label('firing rate (Hz)') 


        print('diffs')
        aord_d = np.argsort(np.array([avs_d[x][f'concat-{s}'] 
                                for x in avs_d]))
        print(s, list(reversed(
                    np.array(list(avs_d.keys()))[aord_d][-nanno:])))
        
        # same for differences                           
        plot_swanson_vector(np.array(list(avs_d.keys())),
                            np.array([avs_d[x][f'concat-{s}'] 
                                for x in avs_d]), 
                            cmap=cmap_, 
                            ax=axs[1,k], br=br, 
                            orientation='portrait',
                            vmin=vmin_d, 
                            vmax=vmax_d,
                            thres=20000,
                            annotate= annotate,
                            annotate_n=nanno,
                            annotate_order='top')                            
                            
                            

        norm = mpl.colors.Normalize(vmin=vmin_d, 
                                    vmax=vmax_d)        

        num_ticks = 3  # Adjust as needed

        # Use MaxNLocator to select a suitable number of ticks
        locator = MaxNLocator(nbins=num_ticks)
                   
        norm = mpl.colors.Normalize(vmin=vmin_d, vmax=vmax_d)
        cbar = fig.colorbar(
                   mpl.cm.ScalarMappable(norm=norm,cmap=cmap_),
                   ax=axs[1,k],shrink=0.8,aspect=12,pad=.025,
                   orientation="horizontal", ticks=locator)
                   
        cbar.ax.tick_params(axis='both', which='major', size=6)
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(size=2)
        cbar.ax.xaxis.set_tick_params(pad=5)
        cbar.set_label('firing rate (Hz)')

        
        axs[0,k].set_title(s)
 
           
 
        [axs[row,k].axis('off') for row in range(nrows)]    


        #put_panel_label(axs[k], k)
        k+=1
    
    fig.subplots_adjust(top=0.963,
                        bottom=0.001,
                        left=0.018,
                        right=0.982,
                        hspace=0.0,
                        wspace=0.035)
    
    
    fig.savefig(Path(one.cache_dir,'dmn', 'figs','swansons.svg'))
    fig.savefig(Path(one.cache_dir,'dmn', 'figs','swansons.pdf'),
                dpi=150, bbox_inches='tight')


def plot_multi_umap_cell(ds=0.1):

    '''
    5 columns for the 5 network types;
    for each a cell-level umap embedding
    colored by Beryl, layers, kmeans
    Below k-means average cluster PETHs
    
    ds: scatter dot size
    '''
      
    nrows = 3 + 2 + 7  # 3 scatters, 7 lineplots, 2 filler
    ncols = len(PETH_types_dict)
        
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, 
                            figsize=[8.33, 8.61],
                            gridspec_kw={'height_ratios': 
                            [6,6,6,1,1,1,1,1,1,1,1,1]})
    
    mappings = ['Beryl', 'layers', 'kmeans']

    c = 0
    for s in PETH_types_dict:    
        r = 0
        leg = True if c == 4 else False
        axs[0,c].set_title(s)
        for mapping in mappings:
            exa_kmeans = True if mapping == 'kmeans' else False
            plot_dim_reduction(algo='umap_z', 
                mapping=mapping, vers=s, ax=axs[r,c], ds=ds,
                leg=leg, exa_kmeans = exa_kmeans,
                axx = axs[-7:,c] if mapping == 'kmeans' else None)  
            r+=1    
        c +=1
    
    # multi panel arrangement    
    [[axs[r,c].axis('off') for c in range(len(PETH_types_dict))] 
        for r in range(nrows)[:-1]]
    
    [[axs[-1,c].tick_params(axis='y', left=False, labelleft=False),
      axs[-1,c].spines['left'].set_visible(False)]
        for c in range(len(PETH_types_dict))]
        
    fig.subplots_adjust(top=0.96,
bottom=0.055,
left=0.01,
right=0.98,
hspace=0.05,
wspace=0.05)

#    fig.savefig(Path(one.cache_dir,'dmn', 'figs','umap_cell.svg'))
#    fig.savefig(Path(one.cache_dir,'dmn', 'figs','umap_cell.pdf'),
#                dpi=150, bbox_inches='tight')    



def plot_peth():

    '''
    for 4 simple trial types, plot PETH for all regions;
    indicate latency also
    '''
    _,pal = get_allen_info()   
    d = np.load(Path(one.cache_dir, 'dmn', 'stack_simple.npy'),
                allow_pickle=True).flat[0]
    
    # order regions by beryl, called regs1
    regs0 = list(d.keys()) 
    p = (Path(iblatlas.__file__).parent / 'beryl.npy')
    regs = br.id2acronym(np.load(p), mapping='Beryl')
    regs1 = []
    for reg in regs:
        if reg in regs0:
            regs1.append(reg)

    xs = {'stim': np.linspace(0,0.15,len(d['MRN']['stim'])),
          'choice': np.linspace(0,0.15,len(d['MRN']['choice'])),
          'fback': np.linspace(0,0.15,len(d['MRN']['fback'])),
          'stim0': np.linspace(0,0.15,len(d['MRN']['stim0'])),
          'stimdiff': np.linspace(0,0.15,len(d['MRN']['stim0']))}

    # scale lines
    yoffset = np.max([np.max([np.max(d[reg][trial_type]) 
                              for reg in regs1])
                              for trial_type in xs])/20

    
    # split regions into q columns
    q = 3
    regions_sorted = regs1
    num_regions = len(regs1)
    regs_per_col = num_regions // q + (num_regions % q > 0)  
    yticks = np.arange(regs_per_col)
    
    # Split regions into three groups
    reg_groups = [regions_sorted[regs_per_col*k:regs_per_col*(k+1)] 
                  for k in range(q)]

    # Initialize the figure and axes
    fig, axs = plt.subplots(1, len(xs)*q, figsize=(18, 10))

    ii = 0
    for trial_type in xs:

        for k in range(q):  # per column
            for i, reg in enumerate(reg_groups[k]):
                x = xs[trial_type]  # converted to sec
                y = ((d[reg][trial_type] - np.min(d[reg][trial_type])) / yoffset 
                      + yticks[i])
                                  
                axs[ii].plot(x, y, color=pal[reg])
                
            axs[ii].set_title(trial_type)   
            axs[ii].set_yticks(yticks[:len(reg_groups[k])])
            axs[ii].set_yticklabels(reg_groups[k])
            axs[ii].set_xlabel('time [sec]')
            colors = [pal[reg] for reg in reg_groups[k]]
            for ticklabel, color in zip(axs[ii].get_yticklabels(), colors):
                ticklabel.set_color(color) 

            ii+=1             
   
    fig.tight_layout()
    
        

def plot_dist_clusters(anno=True, axs=None):

    '''
    for the 5 subsets of PETHS, 
    plot umap per region by using the
    similarity scores obtained by smoothing over neurons
    
    also for cartesian and ephysAtlas 
    '''

    alone = False
    if 'axs' not in locals():
        alone = True
        fig, axs = plt.subplots(nrows=1, ncols=len(verss), 
                                layout="constrained", figsize=(17,4))
    _, pa =get_allen_info()
    
    D = plot_multi_matrices(get_matrices=True)
    
    k = 0
    for vers in D:
    
        if vers == 'regs':
            continue

        
        d = D[vers]        
                  
        dist = np.max(d) - d  # invert similarity to distances
        reducer = umap.UMAP(metric='precomputed')
        emb = reducer.fit_transform(dist)

        cols = [pa[reg] for reg in D['regs']]
        
        axs[k].scatter(emb[:,0], emb[:,1], 
                 color=cols, s=5, rasterized=True)

        if anno:
            for i in range(len(emb)):
                axs[k].annotate('  ' + D['regs'][i], 
                (emb[:,0][i], emb[:,1][i]),
                fontsize=5,color=cols[i])     

            axs[k].set_title(vers)
        axs[k].set_xlabel('umap dim 1')
        axs[k].set_ylabel('umap dim 2')

        k+=1


def plot_single_feature(algo='umap_z', vers='concat', mapping='Beryl',
                        reg = 'MOp'):

    '''
    For a single cell, plot feature vector with PETH labels
    '''
    
    feat = 'concat_z' if algo[-1] == 'z'  else 'concat'
    
    r = regional_group(mapping, algo, vers=vers)    
    
    fig, ax = plt.subplots(figsize=(6.97, 3.01))
    
    xx = np.arange(len(r[feat][0])) /480  # convert to sec
    
    # pick random cell from region
    samp = random.choices(np.where(r['acs'] == reg)[0],k=1)[0]
    print(reg, samp)
    yy = r[feat][samp]

    ax.plot(xx, yy,
             color=r['cols'][samp],
             linewidth=2)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    
    d2 = {}
    for sec in PETH_types_dict[vers]:
        d2[sec] = r['len'][sec]
                        
    # plot vertical boundaries for windows
    h = 0
    for i in d2:
    
        xv = d2[i] + h
        ax.axvline(xv/480, linestyle='--', linewidth=1,
                    color='grey')
        
        # place text in middle of interval
        ax.text(xv/480 - d2[i]/(2*480), max(yy),
                 '   '+i, rotation=90, color='k', 
                 fontsize=10, ha='center')
    
        h += d2[i] 


    ax.set_ylabel('z-scored firing rate')    
    ax.set_xlabel('time [sec]')    
    fig.tight_layout()    
   
    fig.savefig(Path(one.cache_dir,'dmn', 'figs','ex_cells',
        f'{reg}_{samp}.png'),
        dpi=150, bbox_inches='tight')   
    
    
def var_expl(minreg=20):

    '''
    plot variance explained 
    '''
    
    r = regional_group('Beryl', 'umap_z', vers='concat')
    regs = Counter(r['acs'])

    # restrict to regions with minreg cells
    regs2 = [reg for reg in regs if regs[reg]>minreg]

    d = {}
    d2 = {}
    
    for reg in regs2:
        scaler = StandardScaler()
        data_standardized = scaler.fit_transform(
            r['concat'][r['acs']==reg])        
        pca = PCA()
        pca.fit(data_standardized)        
        explained_variance_ratio = pca.explained_variance_ratio_
        d2[reg] = pca.explained_variance_ratio_
        cumulative_explained_variance = np.cumsum(explained_variance_ratio)
        d[reg] = np.argmax(cumulative_explained_variance >= 0.90) + 1

    return d2
    
    sorted_brain_regions = dict(sorted(d.items(), 
        key=lambda item: item[1]))
        
    _,pal = get_allen_info()   
        
    # Extracting keys, values, and colors
    regions = list(sorted_brain_regions.keys())
    values = list(sorted_brain_regions.values())
    colors = [pal[region] for region in regions]

    # Creating the bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.bar(regions, values, color=colors)
    ax.set_xlabel('Brain Regions')
    ax.set_ylabel('PCA dims to explain at least 90% of variance')
    # Rotate x-tick labels and set their colors
    ax.set_xticks(range(len(regions)))
    ax.set_xticklabels(regions, rotation=90)
    [t.set_color(pal[reg]) for reg, t in zip(regions, ax.get_xticklabels())]
  

    
def clus_freqs(foc='kmeans', nmin=50, nclus=7, vers='concat'):

    '''
    For each k-means cluster, show an Allen region bar plot of frequencies,
    or vice versa
    foc: focus, either kmeans or Allen 
    '''
    
    r_a = regional_group('Beryl', 'umap_z', vers=vers, nclus=nclus)    
    r_k = regional_group('kmeans', 'umap_z', vers=vers, nclus=nclus)

    if foc == 'kmeans':
    
        # show frequency of regions for all clusters
        cluss = sorted(Counter(r_k['acs']))
        fig, axs = plt.subplots(nrows = len(cluss), ncols = 1,
                               figsize=(18.79,  15),
                               sharex=True, sharey=False)
        
        fig.canvas.manager.set_window_title(
            f'Frequency of Beryl region label per'
            f' kmeans cluster ({nclus}); vers ={vers}')                      
                               
        cols_dict = dict(list(Counter(zip(r_a['acs'], r_a['cols']))))
        
        # order regions by canonical list 
        p = (Path(iblatlas.__file__).parent / 'beryl.npy')
        regs_can = br.id2acronym(np.load(p), mapping='Beryl')
        regs_ = Counter(r_a['acs'])
        reg_ord = []
        for reg in regs_can:
            if reg in regs_:
                reg_ord.append(reg)        
        
        k = 0                       
        for clus in cluss:                       
            counts = Counter(r_a['acs'][r_k['acs'] == clus])
            reg_order = {reg: 0 for reg in reg_ord}
            for reg in reg_order:
                if reg in counts:
                    reg_order[reg] = counts[reg] 
                    
            # Preparing data for plotting
            labels = list(reg_order.keys())
            values = list(reg_order.values())        
            colors = [cols_dict[label] for label in labels]                
                               
            # Creating the bar chart
            bars = axs[k].bar(labels, values, color=colors)
            axs[k].set_ylabel(f'clus {clus}')
            axs[k].set_xticklabels(labels, rotation=90, 
                                   fontsize=6)
            
            for ticklabel, bar in zip(axs[k].get_xticklabels(), bars):
                ticklabel.set_color(bar.get_facecolor())        

            axs[k].set_xlim(-0.5, len(labels)-0.5)

            k += 1
        
        fig.tight_layout()        
        fig.subplots_adjust(top=0.951,
                            bottom=0.059,
                            left=0.037,
                            right=0.992,
                            hspace=0.225,
                            wspace=0.2)       

    else:

        # show frequency of clusters for all regions

        # order regions by canonical list 
        p = (Path(iblatlas.__file__).parent / 'beryl.npy')
        regs_can = br.id2acronym(np.load(p), mapping='Beryl')
        regs_ = Counter(r_a['acs'])
        reg_ord = []
        for reg in regs_can:
            if reg in regs_ and regs_[reg] >= nmin:
                reg_ord.append(reg)        

        print(len(reg_ord), f'regions with at least {nmin} cells')
        ncols = int((len(reg_ord) ** 0.5) + 0.999)
        nrows = (len(reg_ord) + ncols - 1) // ncols
        
        fig, axs = plt.subplots(nrows = nrows, 
                                ncols = ncols,
                                figsize=(18.79,  15),
                                sharex=True)
        
        axs = axs.flatten()
                               
        cols_dict = dict(list(Counter(zip(r_k['acs'],
                    [tuple(color) for color in r_k['cols']]))))
                    
        cols_dictr = dict(list(Counter(zip(r_a['acs'],
                                          r_a['cols']))))
        
        cluss = sorted(list(Counter(r_k['acs'])))
        
        k = 0                       
        for reg in reg_ord:                       
            counts = Counter(r_k['acs'][r_a['acs'] == reg])
            clus_order = {clus: 0 for clus in cluss}
            for clus in clus_order:
                if clus in counts:
                    clus_order[clus] = counts[clus] 
                    
            # Preparing data for plotting
            labels = list(clus_order.keys())
            values = list(clus_order.values())        
            colors = [cols_dict[label] for label in labels]                
                               
            # Creating the bar chart
            bars = axs[k].bar(labels, values, color=colors)
            axs[k].set_ylabel(reg, color=cols_dictr[reg])
            axs[k].set_xticks(labels)
            axs[k].set_xticklabels(labels, fontsize=8)
            
            for ticklabel, bar in zip(axs[k].get_xticklabels(), bars):
                ticklabel.set_color(bar.get_facecolor())        

            axs[k].set_xlim(-0.5, len(labels)-0.5)

            k += 1
            
        fig.canvas.manager.set_window_title(
            f'Frequency of kmeans cluster ({nclus}) per'
            f' Beryl region label per; vers = {vers}')
                     
        fig.tight_layout()        


    fig.savefig(Path(pth_dmn.parent, 'imgs',
                     f'{foc}_{nclus}_{vers}.png')) 
    


def count_trials():

    '''
    For each peth type, count insertions that have zero trials
    '''
    
    t = stack_concat(get_tls=True)
    
    npeths = len(t[list(t.keys())[0]].keys())
    d = dict(zip(np.arange(npeths), [0]*npeths))
    
    for pid in t: 
        y = list(t[pid].values())
        ids = np.where(np.array(y)==0)[0]
        for x in ids:
            d[x] +=1
    
    fig, ax = plt.subplots()
    ax.bar(list(d.keys()),
           (len(t) - np.array(list(d.values())))/len(t))
    ax.set_xticks(list(d.keys()))
    ax.set_xticklabels(t[list(t.keys())[0]].keys(),rotation=90)
    ax.set_ylabel('percentage of insertions \n with trials for peth type')
    ax.set_xlabel('peth types')    
    
    
def compare_two_goups(vers='concat', filt = 'VISp'):

    '''
    compare average feature vecotr for two groups of cells
    '''

    r = regional_group('Beryl', 'umap_z', vers=vers)        
    df = pd.DataFrame({'acs': r['acs'], 'x': r['xyz'][:,0]})    
    regs = np.unique(r['acs'])
    
    
    vis = [x for x in regs if filt in x]
    print(vis)
    
    colsd = {'left': blue_left, 'right': red_right}
    
    fig, axs = plt.subplots(nrows=2, sharex=True, sharey=True,
                            figsize=[18 , 7.63])
    
    kk = 0
    for hem in ['left', 'right']:
        idc = np.bitwise_and(df['acs'].isin(vis), 
            (df['x'] < 0) if (hem == 'left') else (df['x'] >= 0))

        n_cells = sum(idc)
        yy = np.mean(r['concat_z'][idc],axis=0)
        xx = np.arange(len(yy)) /480
        axs[kk].plot(xx, yy,
                 color=colsd[hem],
                 linewidth=2, 
                 label=f'{n_cells} cells'
                       f' in {filt} areas in {hem} hemisphere')                     

        axs[kk].spines['top'].set_visible(False)
        axs[kk].spines['right'].set_visible(False)
        axs[kk].set_xlabel('time [sec]')
        axs[kk].set_ylabel('firing rate')
            
        axs[kk].legend(loc='lower right')    
        d2 = {}
        for sec in PETH_types_dict[vers]:
            d2[sec] = r['len'][sec]
                            
        # plot vertical boundaries for windows
        h = 0
        for i in d2:
        
            xv = d2[i] + h
            axs[kk].axvline(xv/480, linestyle='--', linewidth=1,
                        color='grey')
            
            if kk == 0: 
                axs[kk].text(xv/480 - d2[i]/(2*480), max(yy),
                         '   '+i, rotation=90, color='k', 
                         fontsize=10, ha='center')        
            h += d2[i]
        kk += 1

    fig.tight_layout()



















