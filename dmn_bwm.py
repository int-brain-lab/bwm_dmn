from one.api import ONE
from brainbox.singlecell import bin_spikes2D
from brainwidemap import (bwm_query, load_good_units, 
                          load_trials_and_mask, bwm_units)
from iblatlas.atlas import AllenAtlas
from iblatlas.regions import BrainRegions
import ibllib
from iblatlas.plots import plot_swanson_vector 
from brainbox.io.one import SessionLoader
import ephys_atlas.data
from reproducible_ephys_functions import figure_style, labs

import sys
sys.path.append('Dropbox/scripts/IBL/')
from granger import get_volume, get_centroids, get_res, get_structural



from scipy import signal
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import confusion_matrix
from numpy.linalg import norm
from scipy.stats import gaussian_kde, f_oneway, pearsonr, spearmanr, kruskal
from scipy.spatial import distance
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
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

one = ONE(base_url='https://openalyx.internationalbrainlab.org',
          password='international', silent=True)
          
br = BrainRegions()
#units_df = bwm_units(one)  # canonical set of cells


# save results here
pth_dmn = Path(one.cache_dir, 'dmn', 'res')
pth_dmn.mkdir(parents=True, exist_ok=True)

sigl = 0.05  # significance level (for stacking, plotting, fdr)


bad_eids = ['4e560423-5caf-4cda-8511-d1ab4cd2bf7d',
            '3a3ea015-b5f4-4e8b-b189-9364d1fc7435',
            'd85c454e-8737-4cba-b6ad-b2339429d99b',
            'de905562-31c6-4c31-9ece-3ee87b97eab4',
            '2d9bfc10-59fb-424a-b699-7c42f86c7871',
            '7cc74598-9c1b-436b-84fa-0bf89f31adf6',
            '642c97ea-fe89-4ec9-8629-5e492ea4019d',
            'a2ec6341-c55f-48a0-a23b-0ef2f5b1d71e', # clear artefact
            '195443eb-08e9-4a18-a7e1-d105b2ce1429',
            '549caacc-3bd7-40f1-913d-e94141816547',
            '90c61c38-b9fd-4cc3-9795-29160d2f8e55',
            'ebe090af-5922-4fcd-8fc6-17b8ba7bad6d',
            'a9138924-4395-4981-83d1-530f6ff7c8fc',
            '8c025071-c4f3-426c-9aed-f149e8f75b7b',
            '29a6def1-fc5c-4eea-ac48-47e9b053dcb5',
            '0cc486c3-8c7b-494d-aa04-b70e2690bcba']


tts__ = ['inter_trial',
     'blockL',
     'blockR',
     'stimLbLcL',
     'stimLbRcL',
     'stimLbRcR',
     'stimLbLcR',
     'stimRbLcL',
     'stimRbRcL',
     'stimRbRcR',
     'stimRbLcR',
     'sLbLchoiceL',
     'sLbRchoiceL',
     'sLbRchoiceR',
     'sLbLchoiceR',
     'sRbLchoiceL',
     'sRbRchoiceL',
     'sRbRchoiceR',
     'sRbLchoiceR',
     'choiceL',
     'choiceR',
     'fback1',
     'fback0']
     

PETH_types_dict = {
    'concat': [item for item in tts__],  # Construct the list without 'end'
    'surprise': ['stimLbRcL', 'stimLbRcR', 'stimRbLcL', 'stimRbLcR', 'sLbRchoiceL', 'sLbRchoiceR', 'sRbLchoiceL', 'sRbLchoiceR'],
    'reward': ['fback1', 'fback0'],
    'resting': ['inter_trial'],
    'quiescence': ['blockL', 'blockR']
}    
    

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


def concat_PETHs(pid, get_tts=False, vers='concat'):

    '''
    for each cell concat all possible PETHs
    '''
    
    eid, probe = one.pid2eid(pid)

    # Load in trials data and mask bad trials (False if bad)
    trials, mask = load_trials_and_mask(one, eid)     

    if vers == 'concat0':        
        # define align, trial type, window length   
        tts = {
            'stimL': ['stimOn_times', 
            np.bitwise_and.reduce([mask,~np.isnan(trials[f'contrastLeft'])]),
            [0, 0.15]],
            'stimR': ['stimOn_times', 
            np.bitwise_and.reduce([mask,~np.isnan(trials[f'contrastRight'])]), 
            [0, 0.15]],
            'blockL': ['stimOn_times', 
            np.bitwise_and.reduce([mask,trials['probabilityLeft'] == 0.8]),
             [0.4, -0.1]],
            'blockR': ['stimOn_times', 
            np.bitwise_and.reduce([mask,trials['probabilityLeft'] == 0.2]),
             [0.4, -0.1]],
            'choiceL': ['firstMovement_times', 
            np.bitwise_and.reduce([mask,trials['choice'] == 1]), 
            [0.15, 0]],
            
            'choiceR': ['firstMovement_times', 
            np.bitwise_and.reduce([mask,trials['choice'] == -1]), 
            [0.15, 0]],
            'fback1': ['feedback_times', 
            np.bitwise_and.reduce([mask,trials['feedbackType'] == 1]), 
            [0, 0.3]],
            'fback0': ['feedback_times', 
            np.bitwise_and.reduce([mask,trials['feedbackType'] == -1]), 
            [0, 0.3]]}

    else:
        # define align, trial type, window length

        # For the 'inter_trial' mask trials with too short iti        
        idcs = [0]+ list(np.where((trials['stimOn_times'].values[1:]
                    - trials['intervals_1'].values[:-1])>1.15)[0]+1)
        mask_iti = [True if i in idcs else False 
            for i in range(len(trials['stimOn_times']))]

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
            
            'stimLbLcL': ['stimOn_times',
                 np.bitwise_and.reduce([mask, 
                    ~np.isnan(trials[f'contrastLeft']),
                    trials['probabilityLeft'] == 0.8,
                    trials['choice'] == 1]), 
                                        [0, 0.15]], 
            'stimLbRcL': ['stimOn_times',            
                np.bitwise_and.reduce([mask,
                    ~np.isnan(trials[f'contrastLeft']), 
                    trials['probabilityLeft'] == 0.2,
                    trials['choice'] == 1]), [0, 0.15]],
            'stimLbRcR': ['stimOn_times',
                 np.bitwise_and.reduce([mask, 
                    ~np.isnan(trials[f'contrastLeft']), 
                    trials['probabilityLeft'] == 0.2,
                    trials['choice'] == -1]), 
                                        [0, 0.15]],           
            'stimLbLcR': ['stimOn_times',
                 np.bitwise_and.reduce([mask,       
                    ~np.isnan(trials[f'contrastLeft']), 
                    trials['probabilityLeft'] == 0.8,
                    trials['choice'] == -1]), 
                                        [0, 0.15]],
            'stimRbLcL': ['stimOn_times',
                 np.bitwise_and.reduce([mask, 
                    ~np.isnan(trials[f'contrastRight']), 
                    trials['probabilityLeft'] == 0.8,
                    trials['choice'] == 1]), 
                                        [0, 0.15]], 
            'stimRbRcL': ['stimOn_times',
                 np.bitwise_and.reduce([mask, 
                    ~np.isnan(trials[f'contrastRight']), 
                    trials['probabilityLeft'] == 0.2,
                    trials['choice'] == 1]), 
                                        [0, 0.15]],
            'stimRbRcR': ['stimOn_times',
                 np.bitwise_and.reduce([mask, 
                    ~np.isnan(trials[f'contrastRight']), 
                    trials['probabilityLeft'] == 0.2,
                    trials['choice'] == -1]), 
                                        [0, 0.15]],        
            'stimRbLcR': ['stimOn_times',
                 np.bitwise_and.reduce([mask, 
                    ~np.isnan(trials[f'contrastRight']), 
                    trials['probabilityLeft'] == 0.8,
                    trials['choice'] == -1]), 
                                        [0, 0.15]],
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

            'fback1': ['feedback_times',    
                np.bitwise_and.reduce([mask,
                    trials['feedbackType'] == 1]), 
                       [0, 0.3]],
            'fback0': ['feedback_times', 
                np.bitwise_and.reduce([mask,
                    trials['feedbackType'] == -1]), 
                       [0, 0.3]]}

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

    one = ONE(base_url="https://alyx.internationalbrainlab.org", 
              mode='local')
              
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


def regional_group(mapping, algo, vers='concat'):

    '''
    find group labels for all cells
    mapping: [Allen, Beryl, Cosmos, layers, clusters, clusters_xyz]
    '''

    r = np.load(Path(pth_dmn, f'{vers}.npy'),
                 allow_pickle=True).flat[0]
                 
                              
    # add point names to dict
    r['nums'] = range(len(r[algo][:,0]))
                   

    if 'clusters' in mapping:
        # use clusters from hierarchical clustering to color
        
         
        if mapping[-1] =='z':
            linked_ = 'linked_z'
        elif mapping[-1] =='e':       
            linked_ = 'linked_e'
        else:
            linked_ = 'linked'   
            
        nclus = 10
        clusters = fcluster(r[linked_], t=nclus, criterion='maxclust')
        cmap = mpl.cm.get_cmap('Spectral')
        cols = cmap(clusters/nclus)
        acs = clusters
        # get average point and color per region
        av = {clus: [np.mean(r[algo][clusters == clus], axis=0), 
                    cmap(clus/nclus)] 
              for clus in range(1,nclus+1)} 

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
                                     
        # remove void and root
        zeros = np.arange(len(acs))[np.bitwise_or(acs == 'root',
                                                  acs == 'void')]
        for key in r:
            if len(r[key]) == len(acs):
                r[key] = np.delete(r[key], zeros, axis=0)
                   
        acs = np.delete(acs, zeros)          
        
                                                              
        _,pa = get_allen_info()
        cols = [pa[reg] for reg in acs]
        
        # get average points and color per region
        regs = Counter(acs)  
        av = {reg: [np.mean(r[algo][acs == reg], axis=0), pa[reg]] 
              for reg in regs}
              
              
    r['acs'] = acs
    r['cols'] = cols
    r['av'] = av
              
    return r


def NN(x, y, decoder='LDA', CC=1.0, confusion=False,
       return_weights=False, shuf=False, verb=True):
    '''
    decode region label y from activity x
    '''
    
    nclasses = len(Counter(y))
    startTime = datetime.now()
    
    if shuf:
        np.random.shuffle(y)

    if len(x.shape) == 1:
        x = x.reshape(-1, 1)

    acs = []

    # predicted labels for train/test
    yp_train = []  
    yp_test = []
    
    # true labels for train/test
    yt_train = []  
    yt_test = []  
    
    ws = []
    
    folds = 5  # 20
    
    kf = KFold(n_splits=folds, shuffle=True)

    if verb:
        print('input dimension:', np.shape(x))    
        print(f'# classes = {nclasses}')
        print(f'uniform chance at 1 / {nclasses} = {np.round(1/nclasses,5)}')
        print('x.shape:', x.shape, 'y.shape:', y.shape)
        print(f'{folds}-fold cross validation')
        if shuf: 
            print('labels are SHUFFLED')

    k = 1
    for train_index, test_index in kf.split(x):

        sc = StandardScaler()
        train_X = sc.fit_transform(x[train_index])
        test_X = sc.fit_transform(x[test_index])

        train_y = y[train_index]
        test_y = y[test_index]

        if k == 1:
            if verb:    
                print('train/test samples:', len(train_y), len(test_y))
            

        if decoder == 'LR':
            # CC = 1  #0.00001
            clf = LogisticRegression(C=CC, random_state=0, n_jobs=-1)
            clf.fit(train_X, train_y)

            y_pred_test = clf.predict(test_X)
            y_pred_train = clf.predict(train_X)

        elif decoder == 'LDA':

            clf = LinearDiscriminantAnalysis()
            clf.fit(train_X, train_y)

            y_pred_test = clf.predict(test_X)
            y_pred_train = clf.predict(train_X)

        else:
            return 'what model??'

        yp_train.append(y_pred_train)
        yt_train.append(train_y)

        yp_test.append(y_pred_test)
        yt_test.append(test_y)

        res_test = np.mean(test_y == y_pred_test)
        res_train = np.mean(train_y == y_pred_train)


        ac_test = round(np.mean(res_test), 4)
        ac_train = round(np.mean(res_train), 4)
        acs.append([ac_train, ac_test])

        k += 1

    r_train = round(np.mean(np.array(acs)[:, 0]), 3)
    r_test = round(np.mean(np.array(acs)[:, 1]), 3)
    
    if verb:
        print('')
        print('Mean train accuracy:', r_train)
        print('Mean test accuracy:', r_test)
        print('')
        print('time to compute:', datetime.now() - startTime)
        print('')
        
    if return_weights:
        if decoder == 'LR':
            clf = LogisticRegression(C=CC, random_state=0, n_jobs=-1)
        else:
            clf = LinearDiscriminantAnalysis()

        clf.fit(x, y)

        return clf.coef_

    if confusion:
    
        yt_train = list(chain.from_iterable(yt_train))
        yp_train = list(chain.from_iterable(yp_train))
        
        cm_train = confusion_matrix(yt_train, yp_train, normalize='pred')
        
        yt_test = list(chain.from_iterable(yt_test))
        yp_test = list(chain.from_iterable(yp_test))        
        
        cm_test = confusion_matrix(yt_test, yp_test, normalize='pred')
                   
        return cm_train, cm_test, r_train, r_test                              

    return np.array(acs)


def decode(src='concat', mapping='Beryl', minreg=20, decoder='LDA', 
           algo='umap_z', n_runs = 1, confusion=False):
    
    '''
    src in ['concat', 'concat_z', 'ephysTF']
    '''
    
           
    print(src, mapping, f', minreg: {minreg},', decoder)
                               
    r = regional_group(mapping, 'umap')
     
    # get average points and color per region
    regs = Counter(r['acs'])
    
    x = r[src]
    y = r['acs']

    # restrict to regions with minreg cells
    regs2 = [reg for reg in regs if regs[reg]>minreg]
    mask = [True if ac in regs2 else False for ac in r['acs']]

    x = x[mask]    
    y = y[mask]
    
    # remove void and root
    mask = [False if ac in ['void', 'root'] else True for ac in y]
    x = x[mask]    
    y = y[mask]  
    regs = Counter(y)  

    if confusion:
        cm_train, cm_test, r_train, r_test = NN(x, y, 
            decoder=decoder, confusion=True)
        return cm_train, cm_test, regs, r_train, r_test    

    
    res = []
    res_shuf = []  
    for i in range(n_runs):
        y_ = deepcopy(y)
        
        res.append(NN(x, y_, decoder=decoder, shuf=False, 
                      verb=False if n_runs > 1 else True))
        res_shuf.append(NN(x, y_, decoder=decoder, shuf=True, 
                      verb=False if n_runs > 1 else True))        
            
    return res, res_shuf
        


'''
###
### bulk processing
###
'''


def get_all_PETHs(eids_plus=None):

    '''
    for all BWM insertions, get the PSTHs and acronyms,
    i.e. run get_PETHs
    '''
    
    time00 = time.perf_counter()

    if eids_plus is None:
        df = bwm_query(one)
        eids_plus = df[['eid', 'probe_name', 'pid']].values

    # save results per insertion (eid_probe) in FlatIron folder
    pth = Path(one.cache_dir, 'dmn', 'concat')
    pth.mkdir(parents=True, exist_ok=True)

    Fs = []
    k = 0
    print(f'Processing {len(eids_plus)} insertions')
    for i in eids_plus:
        eid, probe, pid = i

        # remove lick artefact eid and late fire only
        if eid in bad_eids:
            print('exclude', eid)
            continue

        time0 = time.perf_counter()
        try:
        
            D = concat_PETHs(pid)
                            
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


def stack_concat(vers='concat', get_concat=False):

    '''
    stack concatenated PETHs; 
    compute embedding for lower dim 
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

    ws = []
    # group results across insertions
    for s in ss:
                   
        eid =  s.split('_')[0]          
        if eid in bad_eids:
            continue           
                   
        D_ = np.load(Path(pth, s),
                     allow_pickle=True).flat[0]
        
        # pick PETHs to concatenate
        idc = [D_['trial_names'].index(x) for x in ttypes]             
        ws.append(np.concatenate([D_['ws'][x] for x in idc],axis=1))
        for ke in ['ids', 'xyz', 'uuids']:
            r[ke].append(D_[ke])
    
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


    if get_concat:
        return cs
        
    r['concat'] = cs
    cs_z = zscore(cs,axis=1)
    r['concat_z'] = cs_z
    
    # various dim reduction of PETHs to 2 dims
    print('dimensionality reduction ...')
    ncomp = 2
    r['umap'] = umap.UMAP(n_components=ncomp).fit_transform(cs)
    r['umap_z'] = umap.UMAP(n_components=ncomp).fit_transform(cs_z)

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
    
    r['len'] = dict(zip(D_['trial_names'],
                    [x.shape[1] for x in D_['ws']]))

    np.save(Path(pth_dmn, f'{vers}.npy'),
            r, allow_pickle=True)            


def get_umap_dist(rerun=False, algo='umap_z', 
                  mapping='Beryl', vers='concat'):

    pth_ = Path(one.cache_dir, 'granger', 
                f'{algo}_{mapping}_{vers}_smooth.npy')
    if (not pth_.is_file() or rerun):
        res, regs = smooth_dist(algo=algo, mapping=mapping, vers=vers)    
        d = {'res': res, 'regs' : regs}
        np.save(pth_, d, allow_pickle=True)
    else:
        d = np.load(pth_, allow_pickle=True).flat[0]        
        
    return d     



'''
#####################################################
### plotting
#####################################################
'''
        

def plot_dim_reduction(algo='umap_z', mapping='layers', 
                       means=False, exa=False, shuf=False,
                       exa_squ=False, vers='concat', ax=None, ds=0.5):
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
    '''
    
    feat = 'concat_z' if algo[-1] == 'z'  else 'concat'
    
    r = regional_group(mapping, algo, vers=vers)

    if not ax:
        fig, ax = plt.subplots()
        ax.set_title(vers) 
    if shuf:
        shuffle(r['cols'])
    
    im = ax.scatter(r[algo][:,0], r[algo][:,1], 
                    marker='o', c=r['cols'], s=ds, rasterized=True)
    
    if means:
        # show means
        emb1 = [r['av'][reg][0][0] for reg in r['av']] 
        emb2 = [r['av'][reg][0][1] for reg in r['av']]
        cs = [r['av'][reg][1] for reg in r['av']]
        ax.scatter(emb1, emb2, marker='o', facecolors='none', 
                   edgecolors=cs, s=600, linewidths=4, rasterized=True)
    
    ax.set_xlabel(f'{algo} dim1')
    ax.set_ylabel(f'{algo} dim2')
    ss = 'shuf' if shuf else ''
       
    
    if mapping == 'layers':
        ax.legend(handles=r['els'], ncols=1).set_draggable(True)

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

    if exa:
        # plot a cells' feature vector in extra panel when hovering over point
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
        p = (Path(ibllib.__file__).parent / 'atlas/beryl.npy')
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
        fig0, axs = plt.subplots(ncols=2, figsize=(10,8))
        
        
        linkage_matrix = hierarchy.linkage(1-res)    
        # Order the matrix using the hierarchical clustering
        ordered_indices = hierarchy.leaves_list(linkage_matrix)
        res = res[:, ordered_indices][ordered_indices, :]
        
        row_dendrogram = hierarchy.dendrogram(linkage_matrix,labels =regs,
                     orientation="left", color_threshold=np.inf, ax=axs[0])
        regs = np.array(regs)[ordered_indices]
        
        [t.set_color(i) for (i,t) in    
            zip([regcol[reg] for reg in regs],
                 axs[0].yaxis.get_ticklabels())]
#                              
                     
        ax0 = axs[1]

            
        
    else:
        fig0, ax0 = plt.subplots(figsize=(4,4))
    
                   
    ims = ax0.imshow(res, origin='lower', interpolation=None)
    ax0.set_xticks(np.arange(len(regs)), regs,
                   rotation=90)
    ax0.set_yticks(np.arange(len(regs)), regs)               
                   
    [t.set_color(i) for (i,t) in
        zip([regcol[reg] for reg in regs],
        ax0.xaxis.get_ticklabels())] 
         
    [t.set_color(i) for (i,t) in    
        zip([regcol[reg] for reg in regs],
        ax0.yaxis.get_ticklabels())]
    
    ax0.set_title(f'cosine similarity of smooth images, norm:{norm_}')
    ax0.set_ylabel(mapping)
    plt.colorbar(ims,fraction=0.046, pad=0.04)
    fig0.tight_layout()
    fig0.suptitle(f'{algo}, {mapping}')
    
    return res, regs


def plot_dec(r, rs):
    
    '''
    scatter decoding accuracies for actual data
    and below for data with shuffled labels
    '''
    
    r = np.array(r)
    rs = np.array(rs)

    fig, ax = plt.subplots()
    ax.scatter(np.concatenate(r[:,:,0]), 
               np.concatenate(r[:,:,1]), color='r', label='true')
               
    # borders
    ax.axvline(min(np.concatenate(r[:,:,0])), color='r')
    ax.axhline(min(np.concatenate(r[:,:,1])), color='r')           
    ax.scatter(np.concatenate(rs[:,:,0]), 
               np.concatenate(rs[:,:,1]), color='b', label='shuf')
    ax.set_xlabel('train accuracy')
    ax.set_ylabel('test accuracy')
   

def plot_dec_confusion(src ='concat', mapping='Beryl', 
                       minreg=20, decoder='LDA', n_runs = 1):
           
    '''
    For train and test, plot a confusion matrix
    src in ['concat', 'concat_z', 'ephysTF']
    '''       

    cm_train, cm_test, regs, r_train, r_test  = decode(src=src,
        mapping=mapping, minreg=minreg, 
        decoder=decoder,
        confusion=True)
    
    cms = {'train': cm_train, 'test': cm_test}
    
                                     
    vmin, vmax = np.min([cm_train, cm_test]), np.max([cm_train, cm_test])
        
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15,6))
    k = 0
    for ty in cms:
        ims = axs[k].imshow(cms[ty], interpolation=None, 
               cmap=plt.get_cmap('Blues'))
        axs[k].set_title(f'Confusion Matrix {ty}')
        
        tick_marks = range(len(regs))
        axs[k].set_xticks(tick_marks, list(regs), rotation=45)
        axs[k].set_yticks(tick_marks, list(regs))

        if cm_train.shape[0]<15:
            for i in range(len(regs)):
                for j in range(len(regs)):
                    axs[k].text(j, i, np.round(cms[ty][i, j],2), ha='center', 
                             va='center', color='k')

        axs[k].set_ylabel('True label')
        axs[k].set_xlabel('Predicted label')
        k+=1    
    
        cb = plt.colorbar(ims,fraction=0.046, pad=0.04)
    
    fig.suptitle(f'source:{src}; '
                 f'ac. train:{r_train}, ac. test:{r_test}')   
    fig.tight_layout()

    return cms                                         


def plot_ave_PETHs(feat = 'concat', vers='concat', rerun=False):

    '''
    average PETHs across cells
    plot as lines within average trial times
    '''   
    evs = {'stimOn_times':'gray', 'firstMovement_times':'cyan',
           'feedback_times':'orange', 'intervals_1':'purple'}
           
    win_cols = {
         'inter_trial': 'grey', 
         'stimL': blue_left,
         'stimR': red_right,
         'blockL': blue_left,
         'blockR': red_right,
         'choiceL': blue_left,
         'choiceR': red_right,
         'fback1': 'g',
         'fback0': 'k',
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
         'sRbLchoiceR': 'grey'}


    # trial split types, with string to define alignment
    def align(win):
        if ('stim' in win) or ('block' in win) or ('inter' in win):
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

        eids = list(set(np.unique(bwm_query(one)['eid']))
                     - set(bad_eids))
        
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
    r = np.load(Path(pth_dmn, f'{vers}.npy'),allow_pickle=True).flat[0]
    r['mean'] = np.mean(r[feat],axis=0)
    
    
    # plot trial averages
    yys = []  # to find maxes for annotation
    st = 0
    for tt in r['len']:
  
        xx = np.linspace(-pre_post(tt)[0],
                         pre_post(tt)[1],
                         r['len'][tt]) + d['av_times'][align(tt)][0]

        yy = r['mean'][st: st + r['len'][tt]]
        yys.append(max(yy))

        st += r['len'][tt]


        ax.plot(xx, yy, label=tt, color=win_cols[tt])
        ax.annotate(tt, (xx[-1], yy[-1]), color=win_cols[tt])
        
    
    for ev in d['av_times']:
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
    fig.savefig(Path(one.cache_dir,'dmn', 'figs',
                'intro', 'avg_PETHs.svg'))



def plot_xyz(mapping='Beryl', vers='concat', add_cents=True):

    '''
    3d plot of feature per cell
    '''
    
    r = regional_group(mapping, 'umap_z', vers=vers)
    xyz = r['xyz']
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111,projection='3d')
        

    ax.scatter(xyz[:,0], xyz[:,1],xyz[:,2], marker='o', s = 1,
               c=r['cols'])

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
                       
                       

    ax.set_xlim(min(xyz[:,0]), max(xyz[:,0]))
    ax.set_ylim(min(xyz[:,1]), max(xyz[:,1]))
    ax.set_zlim(min(xyz[:,2]), max(xyz[:,2]))
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(f'Mapping: {mapping}')
    ax.grid(False)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=3))
    fig.tight_layout()



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


def plot_connectivity_matrix(metric='umap_z',
                             vers = 'concat', ax0=None):

    '''
    all-to-all matrix for some measures
    '''


    if metric == 'euc_centroid':
        d = get_centroids(dist_=True)
    else:     
        d = get_umap_dist(algo=metric, vers=vers)
                
    res = d['res']
    regs = d['regs']
    
    _,pal = get_allen_info()
    
    alone = False
    if not ax0:
        alone=True
        fig, (ax_dendro, ax0) = plt.subplots(1, 2, 
            figsize=(5, 4), 
            gridspec_kw={'width_ratios': [1, 5]})
    

    # Order the matrix using hierarchical clustering
    linkage_matrix = hierarchy.linkage(1-res)
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
                        extend='neither', ticks=[0, 0.5, 1])
    #cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=3))


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
    return ordered_indices
    

def plot_multi_matrices(ticktype='rectangles', add_clus=True):

    '''
    for various subsets of the PETHs 
    plot adjacency all in one row
    repeat rows, each ordered by the type of the row index
    
    add_clus: add a row of cluster 2d scatter plots  
    '''

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
    nrows = len(verss)+1 if add_clus else len(verss)
    fig, axs = plt.subplots(nrows=nrows, ncols=len(verss),
                            figsize=(8.67,9.77))
    axs = axs.flatten()
    
    k = 0 
    for row in range(len(verss)):   
        # use dendro order of first version for all
        res0 = np.amax(D2[verss[row]]) - D2[verss[row]] 
        cres = squareform(res0)  # get input form for next line
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
        
    #fig.suptitle('all matrices ordered by data of row number')
    fig.tight_layout()
    fig.savefig(Path(one.cache_dir,'dmn', 'figs','matrices.svg'))    
    fig.savefig(Path(one.cache_dir,'dmn', 'figs','matrices.pdf'),
                dpi=150)


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
         'concat': trans_(get_umap_dist(algo='umap_z', 
                                                vers='concat')),
         'surprise': trans_(get_umap_dist(algo='umap_z', 
                                                vers='surprise')),
         'reward': trans_(get_umap_dist(algo='umap_z',
                                                vers='reward')),
         'quiescence': trans_(get_umap_dist(algo='umap_z',
                                                vers='quiescence')),
         'resting': trans_(get_umap_dist(algo='umap_z',
                                                vers='resting')),
         '30ephys': trans_(get_umap_dist(algo='umap_e')),
         #'coherence': get_res(metric='coherence', 
                              #sig_only=True, combine_=True),
         'granger': get_res(metric='granger', 
                            sig_only=True, combine_=True),
         #'structural3_sp': get_structural(fign=3, shortestp=True),
         'axonal': get_structural(fign=3)
         }
     
    tt = len(list(combinations(list(D.keys()),2)))
    nrows = 1 if tt == 1 else 4
    ncols = 1 if tt == 1 else tt//nrows 
        
#    nrows = 3
#    ncols = 1    
        
     
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
            
            
    fig.tight_layout()
    fig.savefig(Path(one.cache_dir,'dmn', 'figs',
                    'scatters.svg'))
    fig.savefig(Path(one.cache_dir,'dmn', 'figs',
                    'scatters.pdf'), dpi=250,
                     bbox_inches='tight', format='pdf')

    
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



def swansons_all(metric='latency', minreg=10, annotate=False,
             vers='concat', mapping='Beryl', restrict=False):

    '''
    Per window of the BWM, average PETHS, get latecy or fr
    put on swanson, one per aligment event
    
    if metric == 'ephysTF', plot region average score
    of each of the 30 features (or just two if restric=True)
    '''
    r = regional_group(mapping, 'umap_z', vers='concat')    
    
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
            frs = []           
            lats = []

            for length in r['len'].values():
                seg = orgl[:length]
                frs.append(np.max(seg))
                seg = zscore(seg)
                seg = seg - np.min(seg)
                loc = np.where(seg > 0.7 * (np.max(seg)))[0][0]
                lats.append(loc * T_BIN)
                orgl = orgl[length:]     
            
            avs[reg] = dict(zip(list(r['len'].keys()),
                                lats if metric == 'latency' else frs))
     
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
        
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols,
                            sharex=True if metric!= 'ephysTF' else False, 
                            sharey=True if metric!= 'ephysTF' else False)
    axs = axs.flatten('F')
    
    cmap_ = 'viridis_r' if metric == 'latency' else 'viridis'
    
    if metric!= 'ephysTF':
        # assure all panels have same scale

        

        lats_all = np.array([np.array([avs[x][s] for x in avs]) 
                    for s in avs[reg].keys()]).flatten()
            
        vmin, vmax = (np.nanmin(lats_all), np.nanmax(lats_all))
   
    
    # loop through PETH types
    k = 0
    for s in avs[reg].keys():
        
        
               
        lats = np.array([avs[x][s] for x in avs])
        
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
                            annotate_n=500,
                            annotate_order=('bottom' 
                            if metric == 'latency' else 'top'))

                             
        cbar = fig.colorbar(mpl.cm.ScalarMappable(
                                norm=norm, 
                                cmap=cmap_), 
                                ax=axs[k],
                                location='bottom')
        cbar.formatter = ScalarFormatter(useMathText=True)
        
        cbar.locator = plt.MaxNLocator(nbins=2)
        cbar.update_ticks()

        axs[k].axis('off')
        axs[k].set_title(s)
        #put_panel_label(axs[k], k)
        k+=1

    fig.suptitle(metric)
    fig.tight_layout()    
    fig.savefig(Path(one.cache_dir,'dmn', 'figs', 'intro', 
                 'ephysTF_example_swansons.svg'))


    #verss = ['concat', 'surprise', 'reward', 'resting']



def illustrate_data(minreg=10, annotate=False, ds=0.1):

    '''
    Plot on swansons mean PETH for 5 types
    put 5 scatter below for PETH types
    ds: scatter dot size
    '''
    
    r = regional_group('Beryl', 'umap_z', vers='concat')    
    del r['len']['end']
     
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
     
    nrows = 2
    ncols = len(avs[reg].keys())
        
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, 
                            figsize=[8.51, 6.07], 
                            gridspec_kw={'height_ratios': [3, 1]})

    cmap_ = 'viridis'
    

    lats_all = np.array([np.array([avs[x][s] for x in avs]) 
                for s in avs[reg].keys()]).flatten()
        
    vmin, vmax = (np.nanmin(lats_all), np.nanmax(lats_all))
   
    
    # loop through PETH subset types
    k = 0  
    for s in avs[reg].keys():

        lats = np.array([avs[x][s] for x in avs])       
        
        plot_swanson_vector(np.array(list(avs.keys())),
                            np.array(lats), 
                            cmap=cmap_, 
                            ax=axs[0,k], br=br, 
                            orientation='portrait',
                            vmin=vmin, 
                            vmax=vmax,
                            annotate= annotate,
                            annotate_n=5,
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
        cbar.set_label('firing rate (Hz)')

        
        axs[0,k].set_title(s)
        axs[0,k].axis('off')
        
        
        plot_dim_reduction(algo='umap_z', 
            mapping='Beryl', vers=s, ax=axs[1,k], ds=ds)
        axs[1,k].axis('off')
        #put_panel_label(axs[k], k)
        k+=1

    
    fig.tight_layout()    
    fig.savefig(Path(one.cache_dir,'dmn', 'figs','swansons_umap.svg'))
    fig.savefig(Path(one.cache_dir,'dmn', 'figs','swansons_umap.pdf'),
                dpi=150, bbox_inches='tight')    
    



def plot_peth(regs_=[], win='inter_trial', minreg=10):

    '''
    plot the PETHs:
    win: PETH type
    regs: regions to show. If [] all regions are shown
    '''
    
    r = regional_group('Beryl', 'umap_z', vers='concat')        
    
    regs = Counter(r['acs'])    
    regs2 = [reg for reg in regs if regs[reg]>minreg]
    
    
    avs = {}

    for reg in regs2:
  
    
        orgl = np.mean(r['concat'][r['acs'] == reg],axis=0)
        frs = []           
        lats = []

        for length in r['len'].values():
            seg = orgl[:length]
            frs.append(seg)
            seg = zscore(seg)
            seg = seg - np.min(seg)
            loc = np.where(seg > 0.7 * (np.max(seg)))[0][0]
            lats.append(loc * T_BIN)
            orgl = orgl[length:]     
        
        avs[reg] = dict(zip(list(r['len'].keys()),zip(lats, frs)))    
    
    fig, ax = plt.subplots(figsize=(3,3))
    _, pal = get_allen_info()
    
    for reg in regs_:
        x = np.arange(len(avs[reg][win][1]))*T_BIN
        y = avs[reg][win][1]
        ax.plot(x, y, c=pal[reg])
        ax.text(x[-1], y[-1], reg, 
                c=pal[reg])
#        ax.axhline(y=np.mean(y), c=pal[reg], 
#                   linestyle='--', linewidth=0.5)
    
    ax.set_title(win)
    ax.set_xlabel('time [ sec]')
    ax.set_ylabel('firing rate')

    fig.tight_layout()    
    

def plot_dist_clusters(anno=True, axs=None):

    '''
    for the 5 subsets of PETHS, 
    plot umap per region by using the
    similarity scores obtained by smoothing over neurons
    
    also for cartesian and ephysAtlas
    
    '''
    
    verss = list(PETH_types_dict.keys()) + ['cartesian','ephysAtlas']
    
    alone = False
    if 'axs' not in locals():
        alone = True
        fig, axs = plt.subplots(nrows=1, ncols=len(verss), 
                                layout="constrained", figsize=(17,4))
    _, pa =get_allen_info()
    
    k = 0
    for vers in verss:

        if vers == 'cartesian':
            d = get_centroids(dist_=True)
        elif vers == 'ephysAtlas':
            d = get_umap_dist(algo='umap_e', vers='concat')
        else:     
            d = get_umap_dist(algo='umap_z', vers=vers)
           
        dist = np.max(d['res']) - d['res']  # invert similarity to distances
        reducer = umap.UMAP(metric='precomputed')
        emb = reducer.fit_transform(dist)

        cols = [pa[reg] for reg in d['regs']]
        
        axs[k].scatter(emb[:,0], emb[:,1], 
                 color=cols, s=5, rasterized=True)

        if anno:
            for i in range(len(emb)):
                axs[k].annotate('  ' + d['regs'][i], 
                (emb[:,0][i], emb[:,1][i]),
                fontsize=5,color=cols[i])     

            axs[k].set_title(vers)
        axs[k].set_xlabel('umap dim 1')
        axs[k].set_ylabel('umap dim 2')

        k+=1




