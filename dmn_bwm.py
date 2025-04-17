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
#from reproducible_ephys_functions import figure_style, labs

import sys
sys.path.append('Dropbox/scripts/IBL/')
from granger import get_volume, get_centroids, get_res, get_structural, get_ari
from state_space_bwm import get_cmap_bwm, pre_post

from scipy import signal
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import KMeans, SpectralClustering, SpectralCoclustering
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from statsmodels.stats.multitest import multipletests
from numpy.linalg import norm
from scipy.stats import (gaussian_kde, f_oneway, 
    pearsonr, spearmanr, kruskal, rankdata, linregress, entropy)
from scipy.spatial import distance
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform, cdist

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from random import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from joblib import Parallel, delayed
from sklearn.utils import parallel_backend

import gc
from pathlib import Path
import random
from copy import deepcopy
import time, sys, math, string, os
from scipy.stats import spearmanr, zscore
import umap.umap_ as umap
from rastermap import Rastermap
from scipy.stats import wasserstein_distance
from itertools import combinations, chain
from datetime import datetime
import scipy.ndimage as ndi
import hdbscan
import subprocess
from PIL import Image

from venny4py.venny4py import *
from matplotlib.axis import Axis
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import ListedColormap, LinearSegmentedColormap   
from matplotlib.gridspec import GridSpec   
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import MaxNLocator
from matplotlib.pyplot import cm
import matplotlib.colors as mcolors
from ibl_style.style import figure_style
from ibl_style.utils import get_coords, add_label, MM_TO_INCH
import figrid as fg
import matplotlib.ticker as ticker
from matplotlib import cm

import warnings
warnings.filterwarnings("ignore")
#mpl.use('QtAgg')

plt.ion() 


np.set_printoptions(threshold=sys.maxsize)

figure_style()
f_size = mpl.rcParams['font.size']
f_size_s = mpl.rcParams['xtick.labelsize']

title_size = 7
label_size = 7
text_size = 6
f_size_l = title_size
f_size = label_size
f_size_s = text_size
f_size_xs = 5

mpl.rcParams['xtick.minor.visible'] = False
mpl.rcParams['ytick.minor.visible'] = False

mpl.rcParams['pdf.fonttype']=42

handle_length = 1
handle_pad = 0.5


def set_max_ticks(ax, num_ticks=4):
    x_ticks = len(ax.get_xticks())
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=np.min([x_ticks, num_ticks])))
    y_ticks = len(ax.get_yticks())
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=np.min([y_ticks, num_ticks])))

# -------------------------------------------------------------------------------------------------
# Plotting utils
# -------------------------------------------------------------------------------------------------

def adjust_subplots(fig, adjust=5, extra=2):
    width, height = fig.get_size_inches() / MM_TO_INCH
    if not isinstance(adjust, int):
        assert len(adjust) == 4
    else:
        adjust = [adjust] *  4
    fig.subplots_adjust(top=1 - adjust[0] / height, bottom=(adjust[1] + extra) / height,
                        left=adjust[2] / width, right=1 - adjust[3] / width)

plt.rcParams.update(plt.rcParamsDefault)
plt.ion()

f_size = 15  # font size

# canonical colors for left and right trial types
blue_left = [0.13850039, 0.41331206, 0.74052025]
red_right = [0.66080672, 0.21526712, 0.23069468]

T_BIN = 0.0125  # bin size [sec] for neural binning
sts = 0.002  # stride size in [sec] for overlapping bins
ntravis = 30  # #trajectories for vis, first 2 real, rest pseudo

# conversion divident to get bins in seconds 
# (taking striding into account)
c_sec =  int(T_BIN // sts) / T_BIN

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
tts__ = ['inter_trial', 'blockL', 'blockR', 'block50', 
         'quiescence', 'stimLbLcL', 'stimLbRcL', 'stimLbRcR', 
         'stimLbLcR', 'stimRbLcL', 'stimRbRcL', 'stimRbRcR', 
         'stimRbLcR', 'motor_init', 'sLbLchoiceL', 'sLbRchoiceL', 
         'sLbRchoiceR', 'sLbLchoiceR', 'sRbLchoiceL', 'sRbRchoiceL', 
         'sRbRchoiceR', 'sRbLchoiceR', 'choiceL', 'choiceR',  
         'fback1', 'fback0']
     
peth_ila = [
    r"$\mathrm{rest}$",
    r"$\mathrm{L_b}$",
    r"$\mathrm{R_b}$",
    r"$\mathrm{50_b}$",
    r"$\mathrm{quies}$",
    r"$\mathrm{L_sL_cL_b, s}$",
    r"$\mathrm{L_sL_cR_b, s}$",
    r"$\mathrm{L_sR_cR_b, s}$",
    r"$\mathrm{L_sR_cL_b, s}$",
    r"$\mathrm{R_sL_cL_b, s}$",
    r"$\mathrm{R_sL_cR_b, s}$",
    r"$\mathrm{R_sR_cR_b, s}$",
    r"$\mathrm{R_sR_cL_b, s}$",
    r"$\mathrm{m}$",
    r"$\mathrm{L_sL_cL_b, m}$",
    r"$\mathrm{L_sL_cR_b, m}$",
    r"$\mathrm{L_sR_cR_b, m}$",
    r"$\mathrm{L_sR_cL_b, m}$",
    r"$\mathrm{R_sL_cL_b, m}$",
    r"$\mathrm{R_sL_cR_b, m}$",
    r"$\mathrm{R_sR_cR_b, m}$",
    r"$\mathrm{R_sR_cL_b, m}$",
    r"$\mathrm{L_{move}}$",
    r"$\mathrm{R_{move}}$",
    r"$\mathrm{feedbk1}$",
    r"$\mathrm{feedbk0}$"
]


peth_dict = dict(zip(tts__, peth_ila))


PETH_types_dict = {
    'concat': [item for item in tts__],
    'Resting': ['inter_trial'],
    'Quiescence': ['quiescence'],
    'Pre-stim prior': ['blockL', 'blockR'],
    # 'Block 50': ['block50'],
    'Stim surprise': ['stimLbRcL', 'stimRbLcR'],
    'Stim congruent': ['stimLbLcL', 'stimRbRcR'],
    # 'stim_all': ['stimLbRcL', 'stimRbLcR', 'stimLbLcL', 'stimRbRcR'],
    'Mistake': ['stimLbRcR', 'stimLbLcR', 'stimRbLcL', 'stimRbRcL',
                'sLbRchoiceR', 'sLbLchoiceR', 'sRbLchoiceL', 'sRbRchoiceL'],
    'Motor initiation': ['motor_init'],
    'Movement': ['choiceL', 'choiceR'],
    'Feedback correct': ['fback1'],
    'Feedback incorrect': ['fback0']
}
    

# https://www.nature.com/articles/s41586-019-1716-z/figures/6
harris_hierarchy = [
    "VPM", "VPL", "PCN", "LGd", "CL", "IAD",
    "VISp", "MG", "AM", "IMD", "AUDp", "SSp-n",
    "SSp-ll", "AUDd", "MD", "SSp-ul", "SSp-m", "PT",
    "SSp-bfd", "SSs", "AIp", "VISl", "VISrl", "RSPd",
    "LD", "MOp", "VISli", "PO", "VISpl", "RSPagl",
    "RSPv", "VISal", "PVT", "CM", "VISpm", "AId",
    "SSp-tr", "AV", "VAL", "SMT", "LP", "ORBi",
    "AUDpo", "PL", "ORBm", "ILA", "FRP", "VISpor",
    "ACAv", "VISam", "VISa", "MOs", "TEa", "AIv",
    "ACAd", "ORBl", "PIL", "PF", "RE", "VM",
    "POL"
]


def put_panel_label(ax, k):
    ax.annotate(string.ascii_lowercase[k], (-0.05, 1.15),
                xycoords='axes fraction',
                fontsize=f_size * 1.5, va='top',
                ha='right', weight='bold')


def beryl_to_cosmos(beryl_acronym, br):
    beryl_id = br.id[br.acronym==beryl_acronym]
    return br.get(ids=br.remap(beryl_id, source_map='Beryl', 
        target_map='Cosmos'))['acronym'][0]


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
    D['channels'] = np.array(clusters['channels'])
    D['axial_um'] = np.array(clusters['axial_um'])
    D['lateral_um'] = np.array(clusters['lateral_um'])
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

        # average firing rates across trials
        ws.append(np.mean(a, axis=0))        

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
                        label='2024_W50', 
                        local_path=LOCAL_DATA_PATH, 
                        one=one)                    
                    
    merged_df0 = D['df_raw_features'].merge(D['df_channels'], 
                                     on=['pid','channel'])               
    merged_df0.reset_index(inplace=True)            
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
        p = (Path(iblatlas.__file__).parent /
             'allen_structure_tree.csv')

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
        palette['CB'] = (0.4627450980392157, 0.47843137254901963, 
                         0.22745098039215686, 1.0)

        palette['void'] = (0, 0, 0, 1)
        palette['root'] = (0, 0, 0, 1)

        #add layer colors
        bc = ['b', 'g', 'r', 'c', 'm', 'y', 'brown', 'pink']
        for i in range(7):
            palette[str(i)] = mcolors.to_rgba(bc[i])
        
        palette['thal'] = mcolors.to_rgba('k') 
        palette['~layer'] = (0, 0, 0, 0)


        r = {}
        r['dfa'] = dfa
        r['palette'] = palette    
        np.save(pth_dmna, r, allow_pickle=True)   

    r = np.load(pth_dmna, allow_pickle=True).flat[0]
    return r['dfa'], r['palette']  

_,pal = get_allen_info()


def regional_group(mapping, vers='concat', ephys=False,
                   nclus = 13, rerun=False):

    '''
    mapping: how to color 2d points, say Beryl, layers, kmeans
    find group labels for all cells
    mapping: [Allen, Beryl, Cosmos, layers, clusters, clusters_xyz, 'in tts__']
    '''

    if all([mapping == 'kmeans', rerun == False, nclus == 13]):
        pth__ = Path(one.cache_dir, 'dmn', f'kmeans_{vers}_ephys{ephys}.npy')
    
        if ((pth__.is_file()) and (not rerun)):
            return np.load(pth__,allow_pickle=True).flat[0] 

    r = np.load(Path(pth_dmn, f'{vers}_ephys{ephys}.npy'),
                 allow_pickle=True).flat[0]

    # add point names to dict
    r['nums'] = range(len(r['xyz'][:,0]))

    if mapping == 'kmeans':
        print('computing k-means')

        feat = 'concat_z'

        kmeans = KMeans(n_clusters=nclus, random_state=0)
        kmeans.fit(r[feat])  # kmeans in feature space

        clusters = kmeans.labels_
        
        cmap = mpl.cm.get_cmap('Spectral')
        cols = cmap(clusters/nclus)
        acs = [int(x) for x in clusters]
        regs = np.unique(clusters)
        
        color_map = dict(zip(list(acs), list(cols)))
        r['els'] = [Line2D([0], [0], color=color_map[reg], 
                    lw=4, label=f'{reg + 1}')
                    for reg in regs]

        r['Beryl'] = np.array(br.id2acronym(r['ids'], 
                                     mapping='Beryl'))
        av = None
              

    elif mapping == 'cocluster':
        feat = 'concat_z'

        clusterer = SpectralCoclustering(n_clusters=nclus, random_state=0) 
        clusterer.fit(r[feat])
        labels = clusterer.row_labels_
        unique_labels = np.unique(labels)
        mapping = {old_label: new_label 
                      for new_label, old_label in 
                      enumerate(unique_labels)}
        clusters = labels

        cmap = mpl.cm.get_cmap('Spectral')
        cols = cmap(clusters/len(unique_labels))
        acs = clusters

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
        acs[~mask] = '~layer'
        
        remove_0 = False
        
        if remove_0:
            # also remove layer 6, as there are only 20 neurons 
            zeros = np.arange(len(acs))[
                        np.bitwise_or(acs == '~layer', acs == '6')]
            for key in r:
                if len(r[key]) == len(acs):
                    r[key] = np.delete(r[key], zeros, axis=0)
                       
            acs = np.delete(acs, zeros)        
        

        cols = np.array([pal[reg] for reg in acs])
        regs = Counter(acs)      
        r['els'] = [Line2D([0], [0], color=pal[reg], 
               lw=4, label=f'{reg} {regs[reg]}')
               for reg in regs]

    elif mapping == 'clusters_xyz':
   
        # use clusters from hierarchical clustering to color
        nclus = 1000
        clusters = fcluster(r['linked_xyz'], t=nclus, 
                            criterion='maxclust')
        cmap = mpl.cm.get_cmap('Spectral')
        cols = cmap(clusters/nclus)
        acs = clusters   

    # elif mapping in PETH_types_dict:
    #     # For a group of PETH types defined in PETH_types_dict,
    #     # concatenate the defined PETH segments and rank cells by mean score
    #     feat = 'concat_z'  # was _bd
    #     assert vers == 'concat', 'vers needs to be concat for this'
    #     assert feat == 'concat_z'

    #     # Retrieve the list of PETH types to concatenate from PETH_types_dict
    #     peth_types_to_concat = PETH_types_dict[mapping]

    #     # Initialize a list to store the concatenated segments
    #     concatenated_data = []

    #     # Iterate through each PETH type in the list and concatenate the data
    #     for peth_type in peth_types_to_concat:
    #         # Extract relevant information from the data
    #         # Get the start and end indices of the PETH segment
    #         segment_names = list(r['len'].keys())
    #         segment_lengths = list(r['len'].values())
    #         start_idx = sum(segment_lengths[:segment_names.index(peth_type)])
    #         end_idx = start_idx + r['len'][peth_type]

    #         # Extract the segment of interest and append to concatenated_data
    #         segment_data = r[feat][:, start_idx:end_idx]
    #         concatenated_data.append(segment_data)

    #     # Concatenate all the segments along the column axis
    #     concatenated_data = np.concatenate(concatenated_data, axis=1)

    #     # Compute the mean (max) for each neuron across the concatenated segment
    #     means = np.mean(abs(concatenated_data), axis=1)

    #     # Rank neurons based on their mean values
    #     df = pd.DataFrame({'means': means})
    #     df['rankings'] = df['means'].rank(method='min', ascending=False).astype(int)
    #     r['rankings'] = df['rankings'].values

    #     # Map neuron IDs to brain region acronyms (if needed)
    #     acs = np.array(br.id2acronym(r['ids'], mapping='Beryl'))

    #     # Apply a color map based on the rankings
    #     cmap = mpl.cm.get_cmap('Spectral')
    #     cols = cmap(r['rankings'] / max(r['rankings']))

    #     # Count the occurrence of each brain region acronym
    #     regs = Counter(acs)

    elif mapping in tts__:
        # for a specific PETH type, 
        # rank cells by mean score
        feat = 'concat_z'  # was _bd
        assert vers == 'concat', 'vers needs to be concat for this'
        assert feat == 'concat_z'

        # Extract relevant information from the data
        # Get the start and end indices of the PETH segment
        segment_names = list(r['len'].keys())
        segment_lengths = list(r['len'].values())
        start_idx = sum(segment_lengths[:segment_names.index(mapping)])
        end_idx = start_idx + r['len'][mapping]

        # Extract the segment of interest
        segment_data = r[feat][:, start_idx:end_idx]

        # Compute the mean for each neuron across the segment
        means = np.mean(abs(segment_data), axis=1)

        # Get the ranking of neurons based on their mean values 
        df = pd.DataFrame({'means': means})
        df['rankings'] = df['means'].rank(method='min', 
                                ascending=False).astype(int)
        r['rankings'] = df['rankings'].values 

        acs = np.array(br.id2acronym(r['ids'], 
                                     mapping='Beryl'))
                                     
        cmap = mpl.cm.get_cmap('Spectral')
        cols = cmap(r['rankings']/max(r['rankings'])) 
        regs = Counter(acs)  

    elif mapping == 'functional':

        funct = {
            "FRP": "Prefrontal", "ACAd": "Prefrontal", "ACAv": "Prefrontal", "PL": "Prefrontal", "ILA": "Prefrontal",
            "ORBl": "Prefrontal", "ORBm": "Prefrontal", "ORBvl": "Prefrontal",
            "AId": "Lateral", "AIv": "Lateral", "AIp": "Lateral", "GU": "Lateral", "VISc": "Lateral",
            "TEa": "Lateral", "PERI": "Lateral", "ECT": "Lateral",
            "SSs": "Somatomotor", "SSp-bfd": "Somatomotor", "SSp-tr": "Somatomotor", "SSp-ll": "Somatomotor",
            "SSp-ul": "Somatomotor", "SSp-un": "Somatomotor", "SSp-n": "Somatomotor", "SSp-m": "Somatomotor",
            "MOp": "Somatomotor", "MOs": "Somatomotor",
            "VISal": "Visual", "VISl": "Visual", "VISp": "Visual", "VISpl": "Visual",
            "VISli": "Visual", "VISpor": "Visual", "VISrl": "Visual",
            "VISa": "Medial", "VISam": "Medial", "VISpm": "Medial",
            "RSPagl": "Medial", "RSPd": "Medial", "RSPv": "Medial",
            "AUDd": "Auditory", "AUDp": "Auditory", "AUDpo": "Auditory", "AUDv": "Auditory"
        }

        cols0 = {
            "Prefrontal": (0.78, 0.16, 0.16, 1.0),     # Deep Red
            "Lateral": (0.83, 0.79, 0.36, 1.0),        # Yellow-green
            "Somatomotor": (0.89, 0.63, 0.33, 1.0),    # Light Orange
            "Visual": (0.49, 0.73, 0.50, 1.0),         # Light Green
            "Medial": (0.53, 0.63, 0.83, 1.0),         # Light Blue
            "Auditory": (0.65, 0.51, 0.84, 1.0),       # Purple
            "Other": (0.,0.,0.,1.)                     # Black
        }

        acs0 = np.array(br.id2acronym(r['ids'], 
                                     mapping='Beryl'))

        acs = [funct[reg] if (reg in funct) else 'Other' for reg in acs0]
        cols = np.array([cols0[reg] for reg in acs])

        # get average points and color per region
        regs = Counter(acs) 

    else:
        acs = np.array(br.id2acronym(r['ids'], 
                                     mapping=mapping))
                                                                                 
        # get color per region
        cols = np.array([pal[reg] for reg in acs])
        regs = Counter(acs)

              
    if 'end' in r['len']:
        del r['len']['end']
              
    r['acs'] = acs
    r['cols'] = cols

    if mapping == 'kmeans' and nclus == 13:
        pth__ = Path(one.cache_dir, 'dmn', f'kmeans_{vers}_ephys{ephys}.npy')
        np.save(pth__, r, allow_pickle=True)

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


# def get_pw_dist(rerun=False, mapping='Beryl', vers='concat', 
#                 nclus=7, norm_=False, zscore_=True, nmin=20):

#     '''
#     get distance for all region pairs by computing
#     the Euclidean distance of the feature vectors 
#     for all pairs of neurons in the two regions and then 
#     average that score 
#     '''

#     pth_ = Path(one.cache_dir, 'dmn', 
#                 f'{mapping}_{vers}_zscore_{zscore_}_pw.npy')
                
#     if (not pth_.is_file() or rerun):
#         r = np.load(Path(pth_dmn, f'{vers}_norm{norm_}.npy'),
#                      allow_pickle=True).flat[0] 
        
#         vecs = 'concat_z' if zscore_ else 'concat'
                     
#         if mapping == 'kmeans':
#             # use kmeans to cluster high-dim points
             
#             nclus = nclus
#             kmeans = KMeans(n_clusters=nclus, random_state=0)
#             kmeans.fit(r[vecs])
#             acs = kmeans.labels_
#             print('kmeans done')
            
#         else:
#             acs = np.array(br.id2acronym(r['ids'], 
#                                          mapping=mapping))

#         assert len(r[vecs]) == len(acs), 'mismatch, data != acs'
        
#         regs0 = Counter(acs)
#         regs = [x for x in regs0 if regs0[x] > nmin]
#         res = np.zeros((len(regs),len(regs)))
#         print(len(regs), 'regions')
        
#         k = 0
#         for i in range(len(regs)):
#             for j in range(i, len(regs)):
         
#                 # group of cells a and b
#                 g_a = r[vecs][acs == regs[i]]
#                 g_b = r[vecs][acs == regs[j]]

#                 # compute pairwise distance
#                 M = cdist(g_a, g_b)
#                 rows, cols = M.shape
                
#                 # remove duplicate counts
#                 mask = np.ones_like(M, dtype=bool)
#                 min_dim = min(rows, cols)
#                 mask[:min_dim, :min_dim] = np.triu(
#                     np.ones((min_dim, min_dim), dtype=bool), k=1)
                    
#                 # average across all pairwise scores    
#                 res[i,j] = np.mean(M[mask])
#                 res[j,i] = res[i,j]
                
#                 if np.isnan(res[i,j]):
#                     print(regs[i], regs[j],  len(g_a), len(g_b))
#                     return
                    
#                 #res[i,j] = np.mean(M)    
#                 k += 1
#                 print(k, 'of', 0.5*(len(regs)**2), 'done')

#         d = {'res': res, 'regs' : regs}
#         np.save(pth_, d, allow_pickle=True)
#     else:
#         d = np.load(pth_, allow_pickle=True).flat[0]        
        
#     return d


# def NN(x, y, decoder='LDA', CC=1.0, confusion=False,
#        return_weights=False, shuf=False, verb=True):
#     '''
#     Decode region label y from activity x with parallelized cross-validation.
#     '''
    
#     nclasses = len(Counter(y))
#     startTime = datetime.now()
    
#     if shuf:
#         np.random.shuffle(y)

#     if len(x.shape) == 1:
#         x = x.reshape(-1, 1)

#     acs = []

#     # predicted labels for train/test
#     yp_train = []  
#     yp_test = []
    
#     # true labels for train/test
#     yt_train = []  
#     yt_test = []  
    
#     folds = 5
#     kf = StratifiedKFold(n_splits=folds, shuffle=True)

#     if verb:
#         print('input dimension:', np.shape(x))    
#         print(f'# classes = {nclasses}')
#         print('x.shape:', x.shape, 'y.shape:', y.shape)
#         print(f'{folds}-fold cross validation')
#         if shuf: 
#             print('labels are SHUFFLED')

#     def process_fold(train_index, test_index):
#         """
#         Process a single fold of cross-validation.
#         """
#         sc = StandardScaler()
#         train_X = sc.fit_transform(x[train_index])
#         test_X = sc.fit_transform(x[test_index])

#         train_X = x[train_index]
#         test_X = x[test_index]

#         train_y = y[train_index]
#         test_y = y[test_index]

#         if decoder == 'LR':
#             clf = LogisticRegression(C=CC, random_state=0, n_jobs=-1,
#                 max_iter=1000)
#         elif decoder == 'LDA':
#             clf = LinearDiscriminantAnalysis()
#         else:
#             raise ValueError('Unsupported model type. Use "LR" or "LDA".')

#         clf.fit(train_X, train_y)
#         y_pred_test = clf.predict(test_X)
#         y_pred_train = clf.predict(train_X)

#         res_test = np.mean(test_y == y_pred_test)
#         res_train = np.mean(train_y == y_pred_train)

#         return (res_train, res_test, y_pred_train, y_pred_test, train_y, test_y)

#     with parallel_backend('loky'):
#         results = Parallel(n_jobs=-1)(
#             delayed(process_fold)(train_index, test_index)
#             for train_index, test_index in kf.split(x, y)
#         )

#     # Collect results from parallel processing
#     for res_train, res_test, y_pred_train, y_pred_test, train_y, test_y in results:
#         acs.append([res_train, res_test])
#         yp_train.append(y_pred_train)
#         yp_test.append(y_pred_test)
#         yt_train.append(train_y)
#         yt_test.append(test_y)

#     r_train = round(np.mean(np.array(acs)[:, 0]), 3)
#     r_test = round(np.mean(np.array(acs)[:, 1]), 3)
    
#     if verb:
#         print('')
#         print('Mean train accuracy:', r_train)
#         print('Mean test accuracy:', r_test)
#         print('')
#         print('time to compute:', datetime.now() - startTime)
#         print('')
        
#     if return_weights:
#         if decoder == 'LR':
#             clf = LogisticRegression(C=CC, random_state=0, n_jobs=-1,
#                 max_iter=1000)
#         else:
#             clf = LinearDiscriminantAnalysis()

#         clf.fit(x, y)
#         return clf.coef_

#     if confusion:
#         yt_train = np.concatenate(yt_train)
#         yp_train = np.concatenate(yp_train)
#         yt_test = np.concatenate(yt_test)
#         yp_test = np.concatenate(yp_test)
        
#         cm_train = confusion_matrix(yt_train, yp_train, normalize='pred')
#         cm_test = confusion_matrix(yt_test, yp_test, normalize='pred')
                   
#         return cm_train, cm_test, r_train, r_test                              

#     return np.array(acs)


def decode(src='concat_z', mapping='Beryl', minreg=20, decoder='LR', 
           algo='umap_z', n_runs = 1, confusion=False, apply_pca=True):
    
    '''
    src in ['concat_z', 'ephysTF']
    ''' 
           
    print(src, mapping, f', minreg: {minreg},', decoder)
                               
    r = regional_group(mapping)
     
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
    mask = [False if ac in ['void', 'root', 'Other'] else True for ac in y]
    x = x[mask]    
    y = y[mask]  
    regs = Counter(y)

    print('x shape', x.shape, 'n classes', len(regs))

    if apply_pca and x.shape[1] >= 100:
        n_components=100
        pca = PCA(n_components=n_components)
        print(f'applying pca, reducing {x.shape[1]} to {n_components}')
        x = pca.fit_transform(x)   

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


def decode_bulk():
    re = {}
    sources = ['concat_z', 'ephysTF']
    targets = ['kmeans', 'Beryl', 'Cosmos', 
                        'layers', 'functional']

    k = 0 
    for src in sources:
        for mapping in targets:

            re[f'{src} {mapping}'] = decode(src=src,n_runs=10,
                 mapping=mapping)
            k +=1
            print(f'{k} out of {len(sources)*len(targets)} done')

    np.save(Path(pth_dmn,'decode.npy'), re, allow_pickle=True)


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
                 ephys=False, n_neighbors=15, concat_only=False):

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
    kes = ['ids', 'xyz', 'uuids','pid','axial_um', 'lateral_um', 'channels']
    for ke in kes:
        r[ke] = []   

    # get PETH type names from first insertion
    D_ = np.load(Path(pth, ss[0]),
                 allow_pickle=True).flat[0]

    # concatenate subsets of PETHs
    
    ttypes = PETH_types_dict[vers]
    tlss = {}  # for stats on trial numbers


    df = bwm_query(one)   
    def pid__(eid,probe_name):
        return df[np.bitwise_and(df['eid'] == eid, 
                          df['probe_name'] == probe_name
                          )]['pid'].values[0]

    ws = []
    # group results across insertions
    for s in ss:
                   
        eid =  s.split('_')[0]
        probe_name = s.split('_')[1].split('.')[0]

        pid = pid__(eid,probe_name)                  
                   
        D_ = np.load(Path(pth, s),
                     allow_pickle=True).flat[0]

        D_['pid'] = [pid]*len(D_['ids'])

        tlss[s] = D_['tls']
        
        if get_tls:
            continue
        
        n_zero_trials = len(np.where(np.array(
                            list(D_['tls'].values())) == 0)[0])
                    
        # remove insertions where a peth is missing (beyond two fbacks)
        if n_zero_trials > 0:
            continue
       
        # pick PETHs to concatenate
        idc = [D_['trial_names'].index(x) for x in ttypes]
        
        peths = [D_['ws'][x] for x in idc]
                     
        # concatenate normalized PETHs             
        ws.append(np.concatenate(peths,axis=1))
        for ke in kes:
            r[ke].append(D_[ke])

    print(len(ws), 'insertions combined')
    
    if get_tls:
        return tlss
    
    
    for ke in kes:  
        r[ke] = np.concatenate(r[ke])
                    
    cs = np.concatenate(ws, axis=0)
    
    # remove cells with nan entries
    goodcells = [~np.isnan(k).any() for k in cs]
    for ke in kes:  
        r[ke] = r[ke][goodcells]       
    cs = cs[goodcells] 

    # remove cells that are zero all the time
    goodcells = [np.any(x) for x in cs]
    for ke in kes:  
        r[ke] = r[ke][goodcells]       
    cs = cs[goodcells]
        
    print(len(cs), 'good cells stacked before ephys')

    ptypes = [x for x in D_['trial_names']
              if x in ttypes]

    idc = [D_['trial_names'].index(x) for x in ttypes]          
    r['len'] = dict(zip(ptypes,
                    [D_['ws'][x].shape[1] for x in idc]))

    if concat_only:  
        r['concat'] = cs              
        np.save(Path(pth_dmn, f'concat.npy'),
                r, allow_pickle=True)
        return  
        

    r['concat_z'] = zscore(cs,axis=1)

    if get_concat:
        return cs

    # n components for PCA, umap ... 
    ncomp = 2

    if ephys:
        print('loading and concatenating ephys features ...')

        #  include ephys atlas info per channel, 
        #  so some clusters have the same!

        df = pd.DataFrame({'uuids':r['uuids'], 
                           'pid': r['pid'],
                           'channels': r['channels'],
                           'axial_um': r['axial_um'],
                           'lateral_um':r['lateral_um'],
                           'concat_z':[row for row in r['concat_z']]})

        atlas_df = load_atlas_data()
        atlas_df = atlas_df.drop_duplicates(
            subset=['pid', 'axial_um', 'lateral_um'])

        dfm = df.merge(atlas_df, 
                     on=['pid', 'axial_um', 'lateral_um'], 
                     how='left')

        assert len(df) == len(dfm)
        
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

        dfm['ephysTF'] = dfm[fts].apply(
            lambda row: np.concatenate([np.atleast_1d(val) 
            for val in row.values if pd.notna(val)])
            if any(pd.notna(row.values)) else np.array([]),  
            axis=1)
        

        # Add 'ephysTF' to the dictionary
        r['ephysTF'] = dfm['ephysTF'].values
        r['fts'] = fts
               
        # remove cells with nan/inf/allzero entries
        goodcells = np.bitwise_and.reduce([
            [~np.isinf(k).any() for k in r['ephysTF']],   # No inf values
            [np.any(x) for x in r['ephysTF']], # At least one non-zero value
            [~np.isnan(k).any() for k in r['ephysTF']],  # No NaN values
            [k.size > 0 for k in r['ephysTF']]           # Non-empty arrays
        ])
                    
        l0 = deepcopy(len(r['uuids']))
        for key in r:
            if len(r[key]) == l0:
                r[key] = r[key][goodcells]

        print('z-scoring ephys features')
        r['ephysTF'] = zscore(np.stack(r['ephysTF'], axis=0),axis=1)
        print('umap of ephys ...')
        # dim reducing ephys features
        r['umap_e'] = umap.UMAP(
                        n_components=ncomp).fit_transform(r['ephysTF'])

        # various dim reduction of PETHs to 2 dims
        print('embedding rastermap on ephys')

        # Fit the Rastermap model
        model = Rastermap(n_PCs=200, n_clusters=100,
                        locality=0.75, time_lag_window=5, bin_size=1).fit(r['ephysTF'])
        r['isort_e'] = model.isort       


    # various dim reduction of PETHs to 2 dims
    print(f'embedding rastermap on {vers}...')

    try:
        # Fit the Rastermap model
        model = Rastermap(n_PCs=200, n_clusters=100,
                        locality=0.75, time_lag_window=5, 
                        bin_size=1).fit(r['concat_z'])

        r['isort'] = model.isort
    except:
        print('Rastermap failed')
        


    print(f'embedding umap on {vers}...')
    

    r['umap_z'] = umap.UMAP(n_components=ncomp, random_state=0, 
        n_neighbors=n_neighbors).fit_transform(r['concat_z'])

    print(f'embedding pca on {vers}...')
    r['pca_z'] = PCA(n_components=ncomp).fit_transform(r['concat_z'])
    

    np.save(Path(pth_dmn, f'{vers}_ephys{ephys}.npy'),
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
        

def plot_dim_reduction(algo='umap_z', mapping='Beryl',ephys=True,
                       feat = 'concat_z', means=False, exa=False, shuf=False,
                       exa_squ=False, vers='concat', ax=None, ds=0.5,
                       axx=None, exa_kmeans=False, leg=False, restr=None,
                       nclus = 13, n_neighbors=15):
                       
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

    r = regional_group(mapping, vers=vers, ephys=ephys, nclus=nclus)

    print(len(r['concat_z']), 'cells in', mapping, vers)
    

    alone = False
    if not ax:
        alone = True
        fig, ax = plt.subplots(label=f'{vers}_{mapping}')
        #ax.set_title(vers)
    
    if shuf:
        shuffle(r['cols'])
    
    if restr:
        # restrict to certain Beryl regions
        ff = np.bitwise_or.reduce([r['acs'] == reg for reg in restr]) 
    
    
        im = ax.scatter(r[algo][:,0][ff], r[algo][:,1][ff], 
                        marker='o', c=r['cols'][ff], s=ds, rasterized=True)
                        
    else: 
        im = ax.scatter(r[algo][:,0], r[algo][:,1], 
                        marker='o', c=r['cols'], s=ds,
                        rasterized=True)                            
                        
    
    if means:
        # show means
        regs = list(Counter(r['acs']))
        r['av'] = {reg: [np.mean(r[algo][acs == reg], axis=0), pal[reg]] 
              for reg in regs}

        emb1 = [r['av'][reg][0][0] for reg in r['av']] 
        emb2 = [r['av'][reg][0][1] for reg in r['av']]
        cs = [r['av'][reg][1] for reg in r['av']]
        ax.scatter(emb1, emb2, marker='o', facecolors='none', 
                   edgecolors=cs, s=600, linewidths=4, rasterized=True)
    
    zs = True if algo == 'umap_z' else False
    if alone:
        ax.set_title(f'z-score: {zs}')
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
        fff = plt.gcf()
        fff.savefig(Path(one.cache_dir, 'dmn',  'imgs',
            f'{nclus}_kmeans_umap.png'), dpi=150)

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
        plot_cluster_mean_PETHs(r, mapping, feat, 
            vers='concat', axx=axx, alone=True)
        ff = plt.gcf()
        ff.savefig(Path(one.cache_dir, 'dmn', 'imgs',
            f'{nclus}_kmeans_lines.svg'), dpi=150)


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


def plot_cluster_mean_PETHs(r, mapping, feat, vers='concat', 
                         axx=None, alone=True):
    """
    Plots the mean PETH for each k-means cluster.

    Parameters:
    r : dict
        A dictionary with the data needed for plotting, including cluster assignments and feature vectors.
    feat : str
        The key in `r` corresponding to the feature vectors to be plotted.
    vers : str, optional
        The version of the data (default is 'concat').
    axx : list of Axes, optional
        A list of Axes for plotting, one for each cluster.
    alone : bool, optional
        If True, creates a new figure for the plot (default is True).
    c_sec : int, optional
        Conversion factor for time bins to seconds.
    """

    if len(np.unique(r['acs'])) > 50:
        print('too many (>50) line plots!')
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

        # Cluster mean
        xx = np.arange(len(r[feat][0])) / c_sec
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
            d2[sec] =  r['len'][sec]

        # Plot vertical boundaries for windows
        h = 0
        for i in d2:
            xv = d2[i] + h
            axx[kk].axvline(xv / c_sec, linestyle='--', linewidth=1, color='grey')

            if kk == 0:
                axx[kk].text(xv / c_sec - d2[i] / (2 * c_sec), max(yy),
                             '   ' + peth_dict[i], rotation=90, color='k',
                             fontsize=10, ha='center')

            h += d2[i]
        kk += 1

    axx[kk - 1].set_xlabel('time [sec]')

    if alone:
        fg.tight_layout()


def smooth_dist(dim=2, algo='umap_z', mapping='Beryl', 
                show_imgs=False, restr=False,
                norm_=True, dendro=True, nmin=50, vers='concat'):
    """
    Generalized smoothing and analysis of N-dimensional point clouds.
    
    Parameters:
        dim (int): Number of dimensions (2-5). 
        algo, mapping, show_imgs, restr, norm_, dendro, nmin, vers: As in original functions.
        
    Returns:
        res (np.array): Cosine similarity matrix.
        regs (list): List of region labels.
    """

    assert 2 <= dim <= 5, "dim must be between 2 and 5."
    feat = 'concat_z' if algo[-1] == 'z' else 'concat'
    r = regional_group(mapping, vers=vers)
    if algo == 'xyz':
        dim =3
        feat = 'xyz'
        r[algo] = r[algo]*100000

    meshsize = 256 if dim == 2 else 64

    fontsize = 12

    # Define grid boundaries
    mins, maxs = [], []
    for i in range(dim):
        mins.append(np.floor(np.min(r[algo][:, i])))
        maxs.append(np.ceil(np.max(r[algo][:, i])))

    # deal with edge case, add 1% to each max value
    for i in range(dim):
        maxs[i] = maxs[i] * 0.01 + maxs[i]

    imgs = {}
    coords = {}

    regs00 = Counter(r['acs'])
    regcol = {reg: np.array(r['cols'])[r['acs'] == reg][0] for reg in regs00}

    if mapping == 'Beryl':
        p = (Path(iblatlas.__file__).parent / 'beryl.npy')
        regsord = dict(zip(
            br.id2acronym(np.load(p), mapping='Beryl'),
            br.id2acronym(np.load(p), mapping='Cosmos')))
        regs = [reg for reg in regsord if reg in regs00 and regs00[reg] > nmin]
    else:
        regs = [reg for reg in regs00 if regs00[reg] > nmin]

    if restr:
        regs = regs[:10]

    for reg in regs:
        # Scale values to unit interval
        scaled_data = [
            (r[algo][np.array(r['acs']) == reg, i] - mins[i]) / (maxs[i] - mins[i])
            for i in range(dim)
        ]

        data = np.array(scaled_data).T
        inds = (data * meshsize).astype('uint')  # Convert to voxel indices

        img = np.zeros([meshsize] * dim)  # Blank n-dimensional volume
        for pt in inds:
            img[tuple(pt)] += 1

        imsm = ndi.gaussian_filter(img.T, [5] * dim)
        imgs[reg] = imsm / np.max(imsm) if norm_ else imsm
        coords[reg] = scaled_data

    if show_imgs and dim <= 3:
        fig, axs = plt.subplots(nrows=3, ncols=len(regs), figsize=(18.6, 8))

        if dim == 2:
            for i in range(1, len(regs)):
                axs[0, i].sharex(axs[0, 0])
                axs[0, i].sharey(axs[0, 0])

        axs = axs.flatten()
        k = 0

        # First row: Scatter plots
        for reg in regs:
            ax = axs[k]

            if dim == 2:
                ax.scatter(coords[reg][0], coords[reg][1], color=regcol[reg], 
                           s=0.1)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.set_xlim([0, 1])
                ax.set_ylim([0, 1])

            elif dim == 3:
                ax.axis('off')
                ax = fig.add_subplot(3, len(regs), k + 1, projection='3d')
                ax.scatter(coords[reg][0], coords[reg][1], coords[reg][2], 
                           s=0.1, c=regcol[reg])
                ax.set_xlim([0, 1])
                ax.set_ylim([0, 1])
                ax.set_zlim([0, 1])                 


            ax.set_aspect('equal')    
            ax.set_title(f"{reg}, ({regs00[reg]})")
            
            k += 1

        fig.text(0.02, 0.8, f'{algo} \n embedded activity', fontsize=14, 
            rotation='vertical', va='center', ha='center')

        # Second row: Smoothed density (Max projection if dim > 3)
        for reg in regs:
            img = imgs[reg]
            ax = axs[k]
            if dim == 2:
                ax.imshow(img, origin='lower', cmap='viridis')
            elif dim == 3:
                ax.imshow(np.max(img, axis=0), origin='lower', cmap='viridis')
            axs[k].set_aspect('equal')
            ax.axis('off')
            k += 1

        fig.text(0.02, 0.5, 'Smoothed 2d \n projected Density', fontsize=14, 
            rotation='vertical', va='center', ha='center')

        k3 = k
        # Third row: Feature vectors
        for reg in regs:
            pts = np.arange(len(r['acs']))[r['acs'] == reg]
            xss = T_BIN * np.arange(len(np.mean(r[feat][pts], axis=0)))
            yss = np.mean(r[feat][pts], axis=0)
            yss_err = np.std(r[feat][pts], axis=0) / np.sqrt(len(pts))
            maxys = [yss + yss_err]  
            ax = axs[k]
            ax.fill_between(xss, yss - yss_err, yss + yss_err, 
                alpha=0.2, color=regcol[reg])
            ax.plot(xss, yss, color='k', linewidth=0.5)
            axs[k].set_xlabel('time [sec]' if algo != 'xyz' else 'xyz')

            ax.axis('off')
            if algo != 'xyz':     
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

            k += 1

        # shar x and y for last row of panels
        for axx in axs[k3:]:
            axx.sharex(axs[k3])
            axx.sharey(axs[k3])

        fig.text(0.02, 0.25, 'Avg. Feature \n Vectors', 
            fontsize=14, rotation='vertical', va='center', ha='center')

        fig.tight_layout()

        fig.subplots_adjust(top=0.981, bottom=0.019, left=0.04, 
            right=0.992, hspace=0.023, wspace=0.092)

    # Compute similarity between regions
    res = np.zeros((len(regs), len(regs)))
    for i, reg_i in enumerate(imgs):
        for j, reg_j in enumerate(imgs):
            v0 = imgs[reg_i].flatten()
            v1 = imgs[reg_j].flatten()
            res[i, j] = cosine_sim(v0, v1)

    # Plot similarity matrix and dendrogram
    if dendro:
        fig0, axs = plt.subplots(1, 2, figsize=(4, 4), 
            gridspec_kw={'width_ratios': [1, 11]})
        dist = np.max(res) - res
        np.fill_diagonal(dist, 0)
        cres = squareform(dist)
        linkage_matrix = hierarchy.linkage(cres)

        ordered_indices = hierarchy.leaves_list(linkage_matrix)
        res = res[:, ordered_indices][ordered_indices, :]
        row_dendrogram = hierarchy.dendrogram(linkage_matrix, labels=regs, orientation="left",
                                               color_threshold=np.inf, ax=axs[0])
        axs[0].axis('off')

        ax0 = axs[1]
    else:
        fig0, ax0 = plt.subplots(figsize=(4, 4))

    ax0.set_title(f'{algo}, {mapping}, {dim} dims')               
    ims = ax0.imshow(res, origin='lower', interpolation=None,
                     vmin=0, vmax=1)
    ax0.set_xticks(np.arange(len(regs)), regs,
                   rotation=90, fontsize=fontsize)
    ax0.set_yticks(np.arange(len(regs)), regs, fontsize=fontsize)               
                   
    [t.set_color(i) for (i,t) in
        zip([regcol[reg] for reg in regs],
        ax0.xaxis.get_ticklabels())] 
         
    [t.set_color(i) for (i,t) in    
        zip([regcol[reg] for reg in regs],
        ax0.yaxis.get_ticklabels())]
    
    cb = plt.colorbar(ims,fraction=0.046, pad=0.04)
    cb.set_label('regional similarity')
    fig0.tight_layout()

    return res, regs


def plot_ave_PETHs(feat = 'concat', vers='concat',
                   rerun=False, anno=True):

    '''
    average PETHs across cells
    plot as lines within average trial times
    '''   
    evs = {'stimOn_times':'gray', 'firstMovement_times':'cyan',
           'feedback_times':'orange'}

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
            try:
                trials, mask = load_trials_and_mask(one, eid, 
                    revision='2024-07-10')
                trials = trials[mask][:-100]

                diffs.append(np.mean(np.diff(
                            trials[list(evs.keys())]),axis=0))
            except:
                print(f'error with {eid}')
                continue
                
        
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
    
    fig, ax = plt.subplots(figsize=(7, 2.75))
    r = np.load(Path(pth_dmn, 'concat.npy'),allow_pickle=True).flat[0]
    r['mean'] = np.mean(r[feat],axis=0)

    # get alignment event per PETH type
    pid = '1a60a6e1-da99-4d4e-a734-39b1d4544fad'
    ttt =  concat_PETHs(pid, get_tts=True, vers=vers)

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


        ax.plot(xx, yy, label=tt, color='k')
        if anno:
            ax.annotate(tt, (xx[-1], yy[-1]), color='k')
        
    
    for ev in d['av_times']:
        if ev == 'intervals_1':
            continue
        ax.axvline(x=d['av_times'][ev][0], label=ev,
                   color=evs[ev], linestyle='-')

        if anno:           
            ax.annotate(ev, (d['av_times'][ev][0], 0.8*max(yys)), 
                        color=evs[ev], rotation=90, 
                        textcoords='offset points', xytext=(-15, 0))
    
                   
    d['av_tr_times'] = [np.cumsum([0]+ list(x)) 
                        for x in d['diffs']]
                 
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)    
    ax.set_xlabel('time [sec]')
    ax.set_ylabel('trial averaged fr [Hz]')
    fig.canvas.manager.set_window_title('PETHs averaged across all BWM cells')
    #ax.set_xlim(-1.15,3)
    fig.tight_layout()
#    fig.savefig(Path(one.cache_dir,'dmn', 'figs',
#                'intro', 'avg_PETHs.svg'))



def plot_xyz(mapping='Beryl', vers='concat', add_cents=False,
             restr=False, ax = None, axoff=True,
             exa=True):

    '''
    3d plot of feature per cell
    add_cents: superimpose stars for region volumes and centroids
    exa: show example average feature vectors
    '''

    r = regional_group(mapping, vers=vers)

    if ((mapping in tts__) or (mapping in PETH_types_dict)):
        cmap = mpl.cm.get_cmap('Spectral')
        norm = mpl.colors.Normalize(vmin=min(r['rankings']), 
                                    vmax=max(r['rankings']))
        cols = cmap(norm(r['rankings']))
        r['cols'] = cols

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
       
    ax.scatter(xyz[:,0], xyz[:,1],xyz[:,2], depthshade=False,
               marker='o', s = 1 if alone else 0.5, c=r['cols'])
               
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
            
            cols = [pal[reg] for reg in regs]
            ax.scatter(cents[:,0], cents[:,1], cents[:,2], 
                       marker='*', s = vols, color=cols, depthshade=False)
                       
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

    if axoff:
        ax.axis('off')


    if ((mapping in tts__) or (mapping in PETH_types_dict)):
        # Create a colorbar based on the colormap and normalization
        mappable = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        mappable.set_array(r['rankings'])  # Set the data array for the colorbar
        cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10)
        cbar.set_label(f'mean {mapping} rankings')

    if alone:
        ax.set_title(f'{mapping}')

#    if alone:
#        fig.tight_layout()
#        fig.savefig(Path(one.cache_dir,'dmn', 'imgs',
#            f'{mapping}_{vers}_{nclus}_3d.png'),dpi=150)

    if exa:
        # add example time series; 20 from max to min equally spaced 
        if (mapping not in tts__) and (mapping not in PETH_types_dict):
            print('not implemented for other mappings')
            return

        
        feat = 'concat_z'
        nrows = 10  # show 5 top cells in the ranking and 5 last
        rankings_s = sorted(r['rankings'])
        indices = [list(r['rankings']).index(x) for x in
                    np.concatenate([rankings_s[:nrows//2], 
                                    rankings_s[-nrows//2:]])]

        fg, axx = plt.subplots(nrows=nrows,
                                   sharex=True, sharey=False,
                                   figsize=(7,7))                   

        xx = np.arange(len(r[feat][0]))/c_sec 

        kk = 0             
        for ind in indices:
                                
            yy = r[feat][ind]

            axx[kk].plot(xx, yy,
                     color=r['cols'][ind],
                     linewidth=2)

            sss = (r['acs'][ind] + '\n' + str(r['pid'][ind][:3]))


            axx[kk].set_ylabel(sss)

            if kk != (len(indices) -1):
                axx[kk].spines['top'].set_visible(False)
                axx[kk].spines['right'].set_visible(False)
                axx[kk].spines['bottom'].set_visible(False)                
            else:

                axx[kk].spines['top'].set_visible(False)
                axx[kk].spines['right'].set_visible(False)
                #axx[kk].spines['left'].set_visible(False)      
                #axx[kk].tick_params(left=False, labelleft=False)
                       
            # plot vertical boundaries for windows
            h = 0
            for i in r['len']:
            
                xv = r['len'][i] + h
                axx[kk].axvline(xv/c_sec, linestyle='--', linewidth=1,
                            color='grey')
                
                ccc = 'r' if i == mapping else 'k'
                if mapping in PETH_types_dict:
                    if i in PETH_types_dict[mapping]:
                        ccc = 'r'

                if  kk == 0:            
                    axx[kk].text(xv/c_sec - r['len'][i]/(2*c_sec), max(yy),
                             '   '+i, rotation=90, 
                             color=ccc, 
                             fontsize=10, ha='center')
            
                h += r['len'][i] 
            kk += 1                

#        #axx.set_title(f'{s} \n {len(pts)} points in square')
        axx[kk - 1].set_xlabel('time [sec]')

        fg.suptitle(f'mapping: {mapping}, feat: {feat}')
        fg.tight_layout()


def clus_grid():

    fig = plt.figure(figsize=(14,10))
    
    axs = []
    for cl in range(13):
        axs.append(fig.add_subplot(3, 5, cl + 1, projection='3d'))
        plot_xyz(mapping='kmeans',restr=[cl], ax=axs[-1],
            axoff=False if cl == 0 else True)    



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

    D = {k: trans_(get_umap_dist(algo='umap_z',vers=k))
         for k in PETH_types_dict}
    
    D['cartesian']= trans_(get_centroids(dist_=True))

#         '30ephys': trans_(get_umap_dist(algo='umap_e')),
#         #'coherence': get_res(metric='coherence', 
                              #sig_only=True, combine_=True),
         #'granger': get_res(metric='granger', 
                           # sig_only=True, combine_=True),
         #'structural3_sp': get_structural(fign=3, shortestp=True),
         #'axonal': get_structural(fign=3)
         
     
    tt = len(list(combinations(list(D.keys()),2)))
    ncols = math.ceil(math.sqrt(tt))
    nrows = math.ceil(tt / ncols)

       
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
                          s=0.1, alpha=0.1, rasterized=True)

        if anno:
     
            for i in range(len(pts)):
                ax[k].annotate('  ' + pts[i], 
                    (gs[i], cs[i]),
                    fontsize=5,color='b' if ranks else 'k')
                       
        cc = ('r' if 'cartesian' in (metrics[nf[k][0]], metrics[nf[k][1]]) 
              else 'k')              
        a = 'ranks' if ranks else ''
        ax[k].set_xlabel(f'{metrics[nf[k][0]]} ' + a, color=cc)       
        ax[k].set_ylabel(f'{metrics[nf[k][1]]} ' + a, color=cc)
        ax[k].spines['right'].set_visible(False)
        ax[k].spines['top'].set_visible(False)
        ss = (f"{np.round(corp,2) if pp<0.05 else '_'}, "
              f"{np.round(cors,2) if ps<0.05 else '_'}")
        ax[k].set_title(ss)# + f'\n {len(pts)}')
    
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
        r = regional_group('Beryl', vers=vers)    

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

        axs[k].axes.inpth_dmnvert_xaxis()                     
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
        axs[k].set_title(s.split('_')[0])
        
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
    Plot on Swanson's mean PETH for 11 types
    and differences from concat
    '''
    
    r = regional_group('Beryl', vers='concat')
     
    # get average z-scored PETHs per Beryl region 
    regs = Counter(r['acs'])
    regs2 = [reg for reg in regs if regs[reg] > minreg]

    # average all PETHs per region, then z-score and get latency
    # plot latency in swanson; put average peths on top
    avs = {}

    for reg in regs2:
    
        # average across neurons per region
        orgl = np.mean(r['concat_z'][r['acs'] == reg], axis=0)   

        rd = {}

        # cumulative [start, end] indices of each segment
        start_end = {}
        start_idx = 0

        # Calculate cumulative [start, end] indices for each segment
        for key, length in r['len'].items():
            end_idx = start_idx + length 
            start_end[key] = [start_idx, end_idx]
            start_idx += length

        for subset, segments in PETH_types_dict.items():
            ranges = []
            for seg in segments:
                ranges.append(np.arange(start_end[seg][0], start_end[seg][1]))
                                        
            # take mean of subset PETHs (average across time bins)
            rd[subset] = np.mean(orgl[np.concatenate(ranges)])
            
        avs[reg] = rd

    # Compute differences from concat
    conds = list(avs[reg].keys())
    avs_d = {}
    for reg in avs:
        avs_d[reg] = {f'concat-{c}': abs(avs[reg]['concat'] - avs[reg][c]) for c in conds}

    # New structure: 4 rows
    first_6 = list(PETH_types_dict.keys())[:6]
    last_5 = list(PETH_types_dict.keys())[6:]

    nrows = 4
    ncols = max(len(first_6), len(last_5))
        
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=[7.09, 8.61])

    cmap_ = 'viridis_r'
    
    # Determine global min/max for color scaling
    lats_all = np.array([avs[x][s] for s in conds for x in avs]).flatten()
    lats_all_d = np.array([avs_d[x][f'concat-{s}'] for s in conds for x in avs_d]).flatten()

    vmin, vmax = np.nanmin(lats_all), np.nanmax(lats_all)
    vmin_d, vmax_d = np.nanmin(lats_all_d), np.nanmax(lats_all_d)

    def plot_row(subset_keys, row_idx, is_diff=False):
        for k, s in enumerate(subset_keys):
            if is_diff:
                aord = np.argsort([avs_d[x][f'concat-{s}'] for x in avs_d])
                values = np.array([avs_d[x][f'concat-{s}'] for x in avs_d])
                vmin_, vmax_ = vmin_d, vmax_d
            else:
                aord = np.argsort([avs[x][s] for x in avs])
                values = np.array([avs[x][s] for x in avs])
                vmin_, vmax_ = vmin, vmax
            
            print(s, list(reversed(np.array(list(avs.keys()))[aord][-nanno:])))

            plot_swanson_vector(
                np.array(list(avs.keys()) if not is_diff else list(avs_d.keys())),
                values, cmap=cmap_, ax=axs[row_idx, k], br=br,
                orientation='portrait', vmin=vmin_, vmax=vmax_,
                annotate=annotate, annotate_n=nanno,
                annotate_order='top')

            norm = mpl.colors.Normalize(vmin=vmin_, vmax=vmax_)
            locator = MaxNLocator(nbins=3)

            cbar = fig.colorbar(
                mpl.cm.ScalarMappable(norm=norm, cmap=cmap_),
                ax=axs[row_idx, k], shrink=0.8, aspect=12, pad=.025,
                orientation="horizontal", ticks=locator)

            cbar.ax.tick_params(axis='both', which='major', size=6)
            cbar.outline.set_visible(False)
            cbar.ax.tick_params(size=2)
            cbar.ax.xaxis.set_tick_params(pad=5)

            if is_diff:
                cbar.set_label('firing rate (Hz)')

            axs[row_idx, k].set_title(s)
            axs[row_idx, k].axis('off')

    # First row: First 6 items
    plot_row(first_6, row_idx=0, is_diff=False)

    # Second row: Differences of first 6 items
    plot_row(first_6, row_idx=1, is_diff=True)

    # Third row: Remaining 5 items
    plot_row(last_5, row_idx=2, is_diff=False)

    # Fourth row: Differences of last 5 items
    plot_row(last_5, row_idx=3, is_diff=True)

    fig.subplots_adjust(
        top=0.963, bottom=0.001, left=0.018, right=0.982,
        hspace=0.0, wspace=0.035)

    fig.savefig(Path(one.cache_dir, 'dmn', 'imgs', 'swansons.svg'))
    fig.savefig(Path(one.cache_dir, 'dmn', 'imgs', 'swansons.pdf'), dpi=150, bbox_inches='tight')


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
    
    D = plot_multi_matrices(get_matrices=True)
    
    k = 0
    for vers in D:
    
        if vers == 'regs':
            continue

        
        d = D[vers]        
                  
        dist = np.max(d) - d  # invert similarity to distances
        reducer = umap.UMAP(metric='precomputed')
        emb = reducer.fit_transform(dist)

        cols = [pal[reg] for reg in D['regs']]
        
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
    
    r = regional_group(mapping, vers=vers)    
    
    fig, ax = plt.subplots(figsize=(6, 3.01))
    
    xx = np.arange(len(r[feat][0])) /c_sec  # convert to sec
    
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
        ax.axvline(xv/c_sec, linestyle='--', linewidth=1,
                    color='grey')
        
        # place text in middle of interval
        ax.text(xv/c_sec - d2[i]/(2*c_sec), max(yy),
                 '   '+peth_dict[i], rotation=90, color='k', 
                 fontsize=10, ha='center')
    
        h += d2[i] 


    ax.set_ylabel('z-scored firing rate')    
    ax.set_xlabel('time [sec]')    
    fig.tight_layout()    
     
    
    
def var_expl(minreg=20):

    '''
    plot variance explained 
    '''
    
    r = regional_group('Beryl', vers='concat')
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
  

    
def clus_freqs(foc='kmeans', nmin=50, nclus=13, vers='concat', get_res=False,
               rerun=False, norm_=True, save_=False, single_regions=[],
               axs = None):

    '''
    For each k-means cluster, show an Allen region bar plot of frequencies,
    or vice versa
    foc: focus, either kmeans or Allen
    get_res: return results
    norm_: normalize distribution so they all sum up to 1
    save_: save results to file
    single_regions: list of regions to plot separately

    single_regions=['PA','SIM','MOB','SCm', 'MS','MRN', 
                    'VISpor','PRNr']
    '''
    alone = True if axs is None else False

    pthres = Path(pth_dmn.parent, f'nclus{nclus}_{foc}_nrm_{norm_}.npy')

    if get_res and not rerun:
        if pthres.is_file():
            return np.load(pthres, allow_pickle=True).flat[0]

    
    r_a = regional_group('Beryl', vers=vers, nclus=nclus)    
    r_k = regional_group('kmeans', vers=vers, nclus=nclus)

    if foc == 'kmeans':
    
        # show frequency of regions for all clusters
        cluss = sorted(Counter(r_k['acs']))
        fig, axs = plt.subplots(nrows = len(cluss), ncols = 1,
                               figsize=(18.79,  15),
                               sharex=True, sharey=True if norm_ else False)
        
        fig.canvas.manager.set_window_title(
            f'Frequency of Beryl region label per'
            f' kmeans cluster ({nclus}); vers ={vers}')                      
                               
        cols_dict = dict(list(Counter(zip(list(r_a['acs']),
                    [tuple(color) for color in r_a['cols']]))))  
        
        # order regions by canonical list 
        p = (Path(iblatlas.__file__).parent / 'beryl.npy')
        regs_can = br.id2acronym(np.load(p), mapping='Beryl')
        regs_ = Counter(r_a['acs'])
        reg_ord = []
        for reg in regs_can:
            if reg in regs_:
                reg_ord.append(reg)

        # keep results for output
        d = {} 

        k = 0                       
        for clus in cluss:                       
            counts = Counter(r_a['acs'][r_k['acs'] == clus])
            reg_order = {reg: 0 for reg in reg_ord}
            for reg in reg_order:
                if reg in counts:
                    reg_order[reg] = counts[reg] 

            values = list(reg_order.values())

            if norm_:
               values = np.array(values)/float(sum(values))           

            # Preparing data for plotting
            labels = list(reg_order.keys())
            values = list(values)        
            colors = [cols_dict[label] for label in labels]                

            d[clus] = values 

            # Creating the bar chart
            bars = axs[k].bar(labels, values, color=colors)
            axs[k].set_ylabel(f'clus {clus}')
            axs[k].set_xticklabels(labels, rotation=90, 
                                   fontsize=6)
            
            for ticklabel, bar in zip(axs[k].get_xticklabels(), bars):
                ticklabel.set_color(bar.get_facecolor())        

            axs[k].set_xlim(-0.5, len(labels)-0.5)
            axs[k].spines['top'].set_visible(False)
            axs[k].spines['right'].set_visible(False)

            k += 1

        for ax in axs:
            if not has_data(ax):
                ax.set_visible(False) 

        fig.tight_layout()        
        fig.subplots_adjust(top=0.951,
                            bottom=0.059,
                            left=0.037,
                            right=0.992,
                            hspace=0.225,
                            wspace=0.2)

    elif foc == 'Beryl':

        # show frequency of clusters for all regions

        # Load canonical region order
        p = (Path(iblatlas.__file__).parent / 'beryl.npy')
        regs_can = br.id2acronym(np.load(p), mapping='Beryl')

        # Determine which regions to plot
        if single_regions:
            reg_ord = [reg for reg in single_regions if reg in r_a['acs']]
            print(f'Plotting {len(reg_ord)} selected regions (single_regions)')
        else:
            regs_ = Counter(r_a['acs'])
            reg_ord = [reg for reg in regs_can if reg in regs_ and regs_[reg] >= nmin]
            print(f'{len(reg_ord)} regions with at least {nmin} cells')

        ncols = int((len(reg_ord) ** 0.5) + 0.999)
        nrows = (len(reg_ord) + ncols - 1) // ncols

        if axs is None:
            fig, axs = plt.subplots(nrows=nrows, 
                                    ncols=ncols,
                                    figsize=(18.79,  15),
                                    sharex=True,
                                    sharey=True if norm_ else False)
            
            axs = axs.flatten()
                               
        cols_dict = {}
        for lab, col in zip(r_k['acs'], r_k['cols']):
            lab_int = int(lab)  # Convert np.int32 → int
            if lab_int not in cols_dict:
                cols_dict[lab_int] = col
                    
        cols_dictr = dict(list(Counter(zip(list(r_a['acs']),
                    [tuple(color) for color in r_a['cols']]))))  

        cluss = sorted(list(Counter(r_k['acs'])))
        
        # keep results for output
        d = {}
        for k, reg in enumerate(reg_ord):
            counts = Counter(np.array(r_k['acs'])[r_a['acs'] == reg])
            clus_order = {clus: 0 for clus in cluss}
            for clus in cluss:
                clus_order[clus] = counts.get(clus, 0)

            values = list(clus_order.values())
            if norm_:
                values = np.array(values)/float(sum(values))    

            labels = list(clus_order.keys())

            if single_regions:
                colors = ['k'] * len(labels)
            else:
                colors = [cols_dict[label] for label in labels]

            d[reg] = values          

            bars = axs[k].bar(labels, values, color=colors)
            axs[k].set_ylabel(reg, color = pal[reg])
            axs[k].set_xticks(labels)
            axs[k].set_xticklabels(labels, fontsize=8, rotation=90)

            if not single_regions:
                for ticklabel, bar in zip(axs[k].get_xticklabels(), bars):
                    ticklabel.set_color(bar.get_facecolor())        

            axs[k].set_xlim(-0.5, len(labels)-0.5)
            axs[k].spines['top'].set_visible(False)
            axs[k].spines['right'].set_visible(False)

        for ax in axs:
            if not has_data(ax):
                ax.set_visible(False)

        if alone:
            fig.canvas.manager.set_window_title(
                f'Frequency of kmeans cluster ({nclus}) per'
                f' Beryl region label; vers = {vers}')
            fig.tight_layout()

    elif foc == 'dec':
        # Get data from get_dec_bwm()
        d = get_dec_bwm()

        # Load canonical region order
        p = (Path(iblatlas.__file__).parent / 'beryl.npy')
        regs_can = br.id2acronym(np.load(p), mapping='Beryl')

        # Determine which regions to plot
        if single_regions:
            reg_ord = [reg for reg in single_regions if reg in d]
            print(f'Plotting {len(reg_ord)} selected regions (single_regions)')
        else:
            reg_ord = [reg for reg in regs_can if reg in d]
            print(f'{len(reg_ord)} regions')

        ncols = int((len(reg_ord) ** 0.5) + 0.999)
        nrows = (len(reg_ord) + ncols - 1) // ncols

        if axs is None:
            alone = False
            fig, axs = plt.subplots(nrows=nrows, 
                                    ncols=ncols,
                                    figsize=(18.79,  15),
                                    sharex=True,
                                    sharey=True if norm_ else False)
            
            axs = axs.flatten()

        rrcols = ['r', 'g', 'b', 'y', 'c', 'm', 'k', 'o', 'p']
        cols_dictr = dict(Counter(zip(list(r_a['acs']),
                                      [tuple(color) for color in r_a['cols']])))

        for k, reg in enumerate(reg_ord):
            values = d[reg]
            labels = list(range(len(values)))

            if norm_:
                values = np.array(values) / float(sum(values))

            if single_regions:
                colors = ['k'] * len(labels)
            else:
                colors = rrcols[:len(labels)]

            bars = axs[k].bar(labels, values, color=colors)
            axs[k].set_ylabel(reg, color = pal[reg])
            axs[k].set_xticks(labels)
            axs[k].set_xticklabels(labels, fontsize=8, rotation=90)

            if not single_regions:
                for ticklabel, bar in zip(axs[k].get_xticklabels(), bars):
                    ticklabel.set_color(bar.get_facecolor())

            axs[k].set_xlim(-0.5, len(labels) - 0.5)
            axs[k].spines['top'].set_visible(False)
            axs[k].spines['right'].set_visible(False)

        for ax in axs:
            if not has_data(ax):
                ax.set_visible(False)

        if alone:
            fig.canvas.manager.set_window_title(
                f'Values from get_dec_bwm per Beryl region; vers = {vers}')
            fig.tight_layout()

    if save_:    
        fig.savefig(Path(pth_dmn.parent, 'imgs',
                        f'{foc}_{nclus}_{vers}_nrm_{norm_}.svg'), dpi=150)
        fig.savefig(Path(pth_dmn.parent, 'imgs',
                        f'{foc}_{nclus}_{vers}_nrm_{norm_}.pdf'), dpi=150)
        np.save(pthres, d, allow_pickle=True)
    if get_res:
        return d

def has_data(ax):
    # Check if there are any plot lines
    if len(ax.lines) > 0:
        return True
    # Check if there are any bar containers
    if len(ax.containers) > 0:
        return True
    # Check if there are any collections (like scatter, bar, etc.)
    if len(ax.collections) > 0:
        return True
    # Check for images, patches, and other data types if necessary
    if len(ax.images) > 0 or len(ax.patches) > 0:
        return True
    return False


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

    r = regional_group('Beryl', vers=vers)        
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
        yy = np.mean(r['concat_bd'][idc],axis=0)
        xx = np.arange(len(yy)) /c_sec
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
            axs[kk].axvline(xv/c_sec, linestyle='--', linewidth=1,
                        color='grey')
            
            if kk == 0: 
                axs[kk].text(xv/c_sec - d2[i]/(2*c_sec), max(yy),
                         '   '+i, rotation=90, color='k', 
                         fontsize=10, ha='center')        
            h += d2[i]
        kk += 1

    fig.tight_layout()


def plot_rastermap(vers='concat', feat='concat_z', regex='Isocortex', 
                   exa = False, mapping='kmeans', bg=True, img_only=False,
                   interp='antialiased', single_reg=False):
    """
    Function to plot a rastermap with vertical segment boundaries 
    and labels positioned above the segments.

    Extra panel with colors of mapping.

    feat = 'single_reg' will show only cells in example region regex

    """
    r = regional_group(mapping, vers=vers)

    if exa:
        plot_cluster_mean_PETHs(r,mapping, feat)

    spks = r[feat]
    isort = r['isort' if feat != 'ephysTF' else 'isort_e'] 
    data = spks[isort]
    row_colors = np.array(r['cols'])[isort]  # Reorder colors by sorted index
    alpha = 0.3
    lw =.01

    if single_reg:
        
        acs = np.array(r['Beryl'])[isort]

        # adjust background line width according to example region size
        n = len(acs)
        n_ex = sum(acs == regex)
        print('number of cells in example region:', n_ex)

        lw = n / (n_ex * 100)

        data = data[acs == regex]
        row_colors = row_colors[acs == regex]
        print(f'filtering rastermap for {regex} cells only')

    n_rows, n_cols = data.shape
    # Create a figure for the rastermap
    fig, ax = plt.subplots(figsize=(6, 8))

    if feat != 'ephysTF':
        # plot in greys, then overlay color (good for big picture); 
        im = ax.imshow(data,  cmap="gray_r", vmin=0, vmax=5,
                    aspect="auto", interpolation=interp)#

        # Plot vertical boundaries and add text labels
        ylim = ax.get_ylim()  
        h = 0
        for segment in r['len']:
            xv = h + r['len'][segment]  # Cumulative position of the vertical line
            ax.axvline(xv, linestyle='--', linewidth=1, color='grey')  # Draw vertical line
            
            # Add text label above the segment boundary
            midpoint = h + r['len'][segment] / 2  # Midpoint of the segment

            if not img_only:
                # Label positioned above the plot
                ax.text(midpoint, ylim[1] + 0.05 * (ylim[1] - ylim[0]), 
                        peth_dict[segment], rotation=90, color='k', 
                        fontsize=10, ha='center')  
            
            h += r['len'][segment]  # Update cumulative sum for the next segment

        x_ticks = np.arange(0, n_cols, c_sec)
        ax.set_xticks(x_ticks)
        x_labels = [f"{int(tick / c_sec)}" for tick in x_ticks]
        ax.set_xticklabels(x_labels)

    else:
        im = ax.imshow(data, cmap="gray_r", aspect="auto")

        ax.set_xticks(range(data.shape[1]))
        ax.set_xticklabels(r['fts'], rotation=90)       

    if bg:   
        for i, color in enumerate(row_colors):
            ax.hlines(i, xmin=0, xmax=n_cols, colors=color, 
            lw=lw, alpha=alpha)

    ax.set_xlabel('time [sec]')    
    ax.set_ylabel(f'cells in {regex}' if feat == 'single_reg' else 'cells')

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if img_only:
        ax.axis('off')

    plt.tight_layout()  # Adjust the layout to prevent clipping
    plt.show()


def non_flatness_score(d, get_cells=False, norm_=True):
    '''
    for each item compute the non-flatness score as the 
    wasserstein metric to a flat distibution
    i.e. zero when flat, high when far from it
    d: dict with region acronyms as keys and distributions as values
    norm_: distribution sums to 1
    '''

    scores = {}
    for reg in d:

        n_cells = np.sum(d[reg])
        if norm_:
            d[reg] = np.array(d[reg])/n_cells

        flat_dist = [np.sum(d[reg])/len(d[reg])] * len(d[reg])

        emd = wasserstein_distance(d[reg], flat_dist)
        
        if get_cells:
            scores[reg] = [n_cells, emd]
        else:    
            scores[reg] = emd

    return dict(sorted(scores.items(), key=lambda item: item[1]))


def flatness_entropy_score(d, get_cells=False):
    '''
    Measures how close the distribution is to uniform using normalized entropy.
    1 = perfectly flat (uniform), 0 = most peaked (Dirac delta-like).
    '''
    scores = {}
    for reg in d:
        p = np.array(d[reg])
        p = p / p.sum()  # normalize to probability
        n_bins = len(p)
        H = entropy(p)  # in nats
        max_H = np.log(n_bins)
        score = H / max_H if max_H > 0 else 0  # normalize to [0, 1]
        score = 1 - score
        if get_cells:
            scores[reg] = [np.sum(d[reg]), score]
        else:
            scores[reg] = score

    return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))



def plot_xyz_cents(foc='Beryl', ax=None, norm_=True):

    '''
    3d plot of feature per Beryl region centroid
    stars for region volumes and centroids
    colored by 'Beryl' and 'foc',
    score in 'clus_flat', 'dec_flat'
    i.e. flatness of distribution of clusters or decoding scores
    '''
    
    alone = False
    if not ax:
        alone = True
        fig = plt.figure(figsize=(8.43,7.26), label=f'{foc}')
        ax = fig.add_subplot(111,projection='3d')

    if foc == 'Beryl':
        d = clus_freqs(foc=foc, get_res=True, norm_=norm_) 

    elif foc == 'dec':
        d = get_dec_bwm()

    else:
        print('??? what score')
        return

    fs = non_flatness_score(d, norm_=norm_)

    regs = list(Counter(fs.keys()))
    centsd = get_centroids()
    cents = np.array([centsd[x] for x in regs])
    xyz = cents          
    volsd = get_volume()
    vols = [volsd[x] for x in regs]
    
    scale = 5000
    vols = scale * np.array(vols)/np.max(vols)
    

    colsB = [pal[reg] for reg in regs]


    cmap = plt.cm.Blues
    scores = [fs[reg] for reg in fs] 
    norm = plt.Normalize(vmin=min(scores), vmax=max(scores))
    cols = cmap(norm(scores))

    scatter = ax.scatter(cents[:,0], cents[:,1], cents[:,2], linewidths=3,
                marker='o', s = vols, color=cols, edgecolor=colsB,
                depthshade=False)
                       
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
    ax.grid(False)
    nbins = 3
    ax.xaxis.set_major_locator(MaxNLocator(nbins=nbins))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=nbins))

    ax.set_title(f'{foc}, norm:{norm_}')

    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, 
                        fraction=0.02, pad=0.1)
    cbar.set_label('the lower the flatter the distribution',
                   fontsize=fontsize)
    cbar.ax.tick_params(labelsize=12)


def get_dec_bwm(nscores=3):
    """
    Calculate the average score per variable for each region, 
    averaged across recordings.

    Parameters:
    - nscores: Minimum number of scores for a region to be included.

    Returns:
    - A dictionary with region acronyms as keys and lists of 
      average scores per variable as values.
    """        
            
    dec_pth = Path(one.cache_dir, 'bwm_res', 'bwm_figs_data', 'decoding') 
    varis = ['stimside', 'choice', 'feedback', 'wheel-speed', 'wheel-velocity']

    res = {}
    

    for vari in varis:
        # Load pooled data based on variable
        data_file = dec_pth / f'{vari}_stage2.pqt'
        d = pd.read_parquet(data_file)      
        d = d.dropna(subset=['score', 'region'])
        
        # Filter out regions that do not meet the minimum score count
        score_count = d.groupby(['region'])[
            'score'].count().reset_index(name='score_count')

        valid_regions = score_count[
            score_count['score_count'] >= nscores]['region']

        d = d[d['region'].isin(valid_regions)]

        # Calculate mean score for each region
        mean_scores = d.groupby('region')['score'].mean()

        # Normalize scores for the current variable
        min_score, max_score = mean_scores.min(), mean_scores.max()
        normalized_scores = (mean_scores - min_score) / (max_score - min_score)

        # Store normalized scores in the result dictionary
        res[vari] = normalized_scores

    # Combine dictionaries so each region has a list of scores across variables
    combined_res = {}
    for vari in varis:
        for region, score in res[vari].items():
            if region not in combined_res:
                combined_res[region] = [None] * len(varis)  # Initialize list for each region
            combined_res[region][varis.index(vari)] = score  # Assign normalized score to the correct index

    return combined_res


def scat_dec_clus(norm_=True, ari=False, harris=False, 
                  log_scale=True, axs=None):
    # Replace the example dictionaries with the processed scores
    be = flatness_entropy_score(clus_freqs(foc='Beryl', get_res=True)) 

    if ari:
        d0 = np.load('/home/mic/wasserstein_fromflatdist_13_concat_nd2.npy', allow_pickle=True).flat[0]
        de = {}
        k = 0
        for reg in d0['regs']:
            de[reg] = d0['res'][k]
            k += 1
    else:                             
        de = flatness_entropy_score(clus_freqs(foc='dec', get_res=True))  

    if harris:
        # Harris hierarchy scores
        harris_hierarchy_scores = {region: idx for idx, region 
            in enumerate(harris_hierarchy)}

        # Merge scores for common regions
        common_regions = set(be.keys()).intersection(de.keys()).intersection(
            harris_hierarchy_scores.keys())
        merged_data = {region: (be[region], de[region], 
            harris_hierarchy_scores[region]) for region in common_regions}

    else:
        # Merge scores for common regions
        common_regions = set(be.keys()).intersection(de.keys())
        merged_data = {region: (be[region], de[region]) 
            for region in common_regions} 

    print(f"Number of common regions: {len(merged_data)}")
    # Separate values for plotting
    x_be = [merged_data[region][0] for region in merged_data]
    y_de = [merged_data[region][1] for region in merged_data]

    if harris:
        z_harris = [merged_data[region][2] for region in merged_data]
    colors = [pal[region] for region in merged_data]
    labels = list(merged_data.keys())

    # Set up subplots
    if axs is None:
        fig, axs = plt.subplots(1, 1, 
            figsize=(3, 3)) 

    # Define scatter plot function for reuse
    def scatter_panel(ax, x, y, xlabel, ylabel):
        
        # Pearson correlation
        corr, p = pearsonr(x, y)
        print(f"Pearson r,p = {corr:.2f},{p:.4f}")
        if p < 0.05:

            # ax.text(0.95, 0.95, '*', transform=ax.transAxes, fontsize=20, verticalalignment='top')

            # Fit regression line
            slope, intercept, r_value, _, _ = linregress(x, y)
            xx = np.linspace(min(x), max(x), 100)
            yy = slope * xx + intercept
            ax.plot(xx, yy, color='black', linestyle='--', 
                label=f'Fit: y={slope:.2f}x+{intercept:.2f}')

        # Scatter points with labels
        for i, region in enumerate(labels):
            ax.scatter(x[i], y[i], color=colors[i], label=region)
            #ax.text(x[i], y[i], region, color=colors[i], fontsize=8, ha='left')

        if log_scale:
            ax.set_xscale('log')
            ax.set_yscale('log')
        # Set labels
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)


    if harris:
        # Panel 2: `be` vs `harris`
        scatter_panel(
            axs,
            x_be, z_harris,
            xlabel='specialization (clu)',
            ylabel='Harris hierarchy score'
        )

    else:    
        # Panel 1: `be` vs `de`
        scatter_panel(
            axs,
            x_be, y_de,
            xlabel='specialization (clu)',
            ylabel='specialization (dec)' if not ari else "ari's non-flatness-scores"
        )
        # # Panel 3: `de` vs `harris`
        # scatter_panel(
        #     axes[2],
        #     y_de, z_harris,
        #     xlabel='BWM dec (non_flatness_score(d_d))' if not ari else "ari's non-flatness-scores",
        #     ylabel='Harris hierarchy score'
        # )

    # Adjust layout and show
    plt.tight_layout()
    plt.show()


def scatter_harris_correlation(acronyms=False, axs=None):

    # File paths based on the image names

    nets = ['concat']
    # , 'motor_init', 'pre-stim-prior', 
    #     'stim_all', 'stim_surp_con', 'stim_surp_incon']

    file_paths = [Path(one.cache_dir, 'dmn', 
        f'wasserstein_fromflatdist_13_{net}_nd2.npy') for net in nets]


    # Load Harris hierarchy scores
    harris_hierarchy_scores = {region: idx for idx, region in enumerate(harris_hierarchy)}

    # Prepare data for each file
    datasets = []
    for path in file_paths:
        data = np.load(path, allow_pickle=True).flat[0]
        de = {}
        for k, reg in enumerate(data['regs']):
            de[reg] = data['res'][k]
        datasets.append(de)

    # Identify common regions across all datasets and Harris scores
    common_regions = set.intersection(*(set(dataset.keys()) for dataset in datasets), set(harris_hierarchy_scores.keys()))

    # Extract merged data for scatter plots
    merged_data = []
    for dataset in datasets:
        merged_data.append({region: (dataset[region], harris_hierarchy_scores[region]) for region in common_regions})

    alone = False
    if axs is None:
        alone = True
        # Set up subplots (1 row, 6 columns)
        fig, axs = plt.subplots(3, 2, figsize=(7, 10),sharey=True)
        axes = axs.flatten()

    # Define scatter plot function for reuse
    def scatter_panel(ax, x, y, labels, colors, xlabel, ylabel, title):

        # Pearson correlation
        corr, p = pearsonr(x, y)

        # Fit regression line
        slope, intercept, r_value, _, _ = linregress(x, y)
        xx = np.linspace(min(x), max(x), 100)
        yy = slope * xx + intercept
        ax.plot(xx, yy, color='black', linestyle='--',linewidth=0.5 if p > 0.05 else 1, 
            label=f'Fit: y={slope:.2f}x+{intercept:.2f}')

        # if p < 0.05:
        #     # put an astreics at the end of lie plot    
        #     ax.text(0.95, 0.95, '*', transform=ax.transAxes, fontsize=20, verticalalignment='top')

        # Scatter points with region-specific colors and labels
        for i, region in enumerate(labels):
            ax.scatter(x[i], y[i], color=colors[i])
            if acronyms:
                ax.text(x[i], y[i], region, color=colors[i], ha='left')

        # Set labels and title
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        # put tile left
        #ax.set_title(f"{title}", fontsize=10, loc="left")  
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Generate scatter plots for each dataset
    for i, data in enumerate(merged_data):
        x = [data[region][0] for region in data]
        y = [data[region][1] for region in data]
        labels = list(data.keys())
        colors = [pal[region] for region in labels]  # Assume `pal` maps regions to colors
        print( 'i', i)
        scatter_panel(
            axs[i], x, y, labels, colors,
            xlabel='specialization' if i == 0 else '',
            ylabel='hierarchy score',
            title=f'{nets[i].replace("_", " ").title()}'
        )
        if i != 0:
           axs[i].set_ylabel('')

    if alone:
        # Adjust layout and show
        plt.tight_layout()
        plt.show()



def plot_brain_region_counts(start=None, end=None, nmin=50):
    """
    Plot a bar chart of brain region counts, color-coded by a given palette.
    
    Parameters:
    - data (dict or Counter): Dictionary or Counter object with region names as keys and counts as values.
    - norm_ binary, control for total cell counts across regions    
    Output:
    - Displays a bar chart with regions sorted by count.
    """

    r = regional_group('Beryl')

    start = start if start is not None else 0
    end = end if end is not None else len(r['acs'])    
    d0 = Counter(np.array(r['acs'])[r['isort']][start:end])
    d00 = Counter(np.array(r['acs'])[r['isort']])

    data = {}
    for reg in d0:
        if d00[reg] < nmin:
            continue
        data[reg] = d0[reg]/d00[reg]

    # Sort data by values
    sorted_data = dict(sorted(data.items(), key=lambda item: item[1], reverse=True))

    # Extract keys, values, and colors
    regions = list(sorted_data.keys())
    values = list(sorted_data.values())
    colors = [pal[region] for region in regions]

    # Create the bar plot
    plt.figure(figsize=(17.27,  6.  ))
    plt.bar(regions, values, color=colors)

    # Customize plot
    plt.xticks(rotation=90, fontsize=8)
    plt.ylabel('Count', fontsize=12)
    plt.title(f'cells {start} to {end} in rastermap ordering', fontsize=14)
    # Set x-tick labels to the same color as bars
    ax = plt.gca()
    for tick, color in zip(ax.get_xticklabels(), colors):
        tick.set_color(color)
    # Show the plot
    plt.tight_layout()
    plt.show()


def embed_histograms_scatter(foc='dec', ax=None):

    '''
    per region, get histogram (bwm dec or PETH based k-means counts)
    embed in 2d via umap
    '''

    alone = False
    if not ax:
        alone = True
        fig, ax = plt.subplots(figsize=(8.43,7.26), label=f'{foc}')

    if foc == 'Beryl':
        d = clus_freqs(foc=foc, get_res=True, norm_=False) 

    elif foc == 'dec':
        d = get_dec_bwm()

    regs, data = [reg for reg in d], np.array([d[reg] for reg in d])
    cols = [pal[reg] for reg in regs]

    emb = umap.UMAP(n_components=2).fit_transform(data)

    ax.scatter(emb[:,0], emb[:,1], color='w')
    # Plot colored text instead of scatter markers
    for i, reg in enumerate(regs):
        ax.text(emb[i, 0], emb[i, 1], reg, color=cols[i], 
            fontsize=9, ha='center', va='center')
  
    ax.set_title(f'embed histograms of {foc}')
    ax.set_xlabel('umap dim 1')
    ax.set_ylabel('umap_dim 2')


def plot_decoding_results():
    re = np.load(Path(pth_dmn,'decode.npy'), allow_pickle=True).flat[0]
    # Prepare the data for plotting
    sources, mappings = [], []
    [[sources.append(k.split(' ')[0]), mappings.append(k.split(' ')[1])] for k in re]

    mappings = np.unique(mappings) 
    sources = np.unique(sources) 

    data = []

    for source in sources:
        for mapp in mappings:
            key = f"{source} {mapp}"
            if key not in re:
                continue
            normal_results, shuffled_results = re[key]  
            for split in normal_results:
                test_results = split[:, 1]
                data.extend([{'Mapping': mapp, 'Source': source, 'Type': 'Test Normal', 'Accuracy': acc} for acc in test_results])
            for split in shuffled_results:
                test_results = split[:, 1]
                data.extend([{'Mapping': mapp, 'Source': source, 'Type': 'Test Shuffled', 'Accuracy': acc} for acc in test_results])

    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(3, 3))
    mapping_colors = sns.color_palette("tab10", len(mappings))
    mapping_palette = {str(mapp): color for mapp, 
        color in zip(mappings, mapping_colors)}
    expanded_palette = {
        mapp: {'Test Normal': color, 'Test Shuffled': color}
        for mapp, color in mapping_palette.items()
    }
    for mapp in mappings:    
        sns.stripplot(
            data=df[df['Mapping'] == mapp],
            x='Source', y='Accuracy', hue='Type',
            dodge=True, jitter=True, ax=ax,
            palette=expanded_palette[mapp] ,  # Use the defined color palette
            size=3
        )
    
    ax.axvline(x=0.5, color='k', linestyle='-', linewidth=1)
    ax.axvline(x=0, color='k', linestyle='--', linewidth=1)
    ax.axvline(x=1, color='k', linestyle='--', linewidth=1)



    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    custom_handles = [plt.Line2D([0], [0], color=palette, marker='o', 
                        linestyle='', markersize=5)
                      for palette in mapping_colors]
    ax.legend(custom_handles, mappings, title='Targets', loc='best', 
         frameon=False).set_draggable(True)

    ax.set_xticks([-0.25,  0.25, 0.75, 1.25])
    ax.set_xticklabels(['test','shuffle', 'test','shuffle'], rotation=90)
    ax.set_xlabel('concat_z | ephysTF')
    ax.set_ylabel('Decoding Accuracy')
     

def ghostscript_compress_pdf(level='/printer'):

    '''
    Compress figs (inkscape pdfs)
    
    levels in [/screen, /ebook,  /printer]
    '''

    input_path = input("Please enter pdf input_path: ")
    print(f"Received input path: {input_path}")

    output_path = input("Please enter pdf output_path: ")
    print(f"Received output path: {output_path}")
    
    input_path = Path(input_path.strip("'\""))
    output_path = Path(output_path.strip("'\""))
    
    print('input_path', input_path)
    print('output_path', output_path)                 

    # Ghostscript command to compress PDF
    command = [
        'gs', '-sDEVICE=pdfwrite', '-dCompatibilityLevel=1.4',
        '-dAutoRotatePages=/None',
        f'-dPDFSETTINGS={level}', '-dNOPAUSE', '-dQUIET', '-dBATCH',
        f'-sOutputFile={output_path}', input_path
    ]

    # Execute the command
    try:
        subprocess.run(command, check=True)
        print(f"PDF successfully compressed and saved to {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")


def float_array_to_rgba(img_float):
    cmap = cm._colormaps['gray_r']
    rgba = cmap(img_float)  # Returns float32 RGBA
    return (rgba * 255).astype(np.uint8)  # Convert to uint8


def save_rastermap_pdf(feat='concat_z', mapping='kmeans', bg=False, cv=False):

    if cv:
        # load Ari's files
        r_test = np.load(Path(pth_dmn, 'cross_val_test.npy'),
                         allow_pickle=True).flat[0]
        r_train = np.load(Path(pth_dmn, 'cross_val_train.npy'), 
                          allow_pickle=True).flat[0]
        
        spks = r_test[feat]
        isort = r_train['isort'] 
        data = spks[isort]

        bg = False      


    else:
        r = regional_group(mapping)
        spks = r[feat]
        isort = r['isort'] 
        data = spks[isort]
    
    # Normalize and convert to RGBA image
    norm_data = data - data.min()
    norm_data = norm_data / norm_data.max()
    image_rgba = float_array_to_rgba(norm_data)
    image_rgba[..., 3] = 255

    if bg:
        # RGBA rows
        row_colors = np.array(r['cols'])[isort]  

        # Blend row colors onto grayscale image
        alpha_overlay = 0.2  # adjust intensity of color overlay

        for i in range(image_rgba.shape[0]):
            r_c, g_c, b_c, _ = row_colors[i] * 255  # get color per row
            overlay = np.array([r_c, g_c, b_c], dtype=np.uint8)

            # Blend overlay with original grayscale values (RGB only)
            image_rgba[i, :, :3] = (
                (1 - alpha_overlay) * image_rgba[i, :, :3] +
                alpha_overlay * overlay[None, :]
            ).astype(np.uint8)

    # Convert to RGB (drop alpha) for PDF compatibility
    img = Image.fromarray(image_rgba[..., :3], mode='RGB')
    s = f'_{mapping}' if bg else ''
    if cv:
        s = 'cross_val'
    img.save(f"rastermap{s}.pdf", "PDF")



def plot_umap_SI(algo='umap_z', mapping='Beryl',smooth=True, norm_=True):

    '''
    for 10 example regions (columns)
    and 11 networks (rows)
    plot a grid of 2d umap embeddings
    '''

    regs = ['PA', 'MOB', 'MS', 'VISpor',
            'SIM', 'SCm', 'MRN', 'PRNr', 'CP', 'GRN']

    verss = PETH_types_dict

    fig, axs = plt.subplots(nrows = len(verss), ncols = len(regs),
                figsize=(183 * MM_TO_INCH, 240 * MM_TO_INCH),
                sharex=True, sharey=True)

    dim = 2
    row = 0
    for vers in verss:
        r = regional_group(mapping, vers=vers)

        # Define grid boundaries
        mins, maxs = [], []
        for i in range(dim):
            mins.append(np.floor(np.min(r[algo][:, i])))
            maxs.append(np.ceil(np.max(r[algo][:, i])))

    # deal with edge case, add 1% to each max value
        for i in range(dim):
            maxs[i] = maxs[i] * 0.01 + maxs[i]

        coords = {}
        regcol = {reg: np.array(r['cols'])[r['acs'] == reg][0] for reg in regs}

        imgs = {}
        meshsize = 256

        for reg in regs:
            # Scale values to unit interval
            scaled_data = [
                (r[algo][np.array(r['acs']) == reg, i] - mins[i]) / (maxs[i] - mins[i])
                for i in range(dim)
            ]
            coords[reg] = scaled_data

            if smooth:
                data = np.array(scaled_data).T
                inds = (data * (meshsize - 1)).astype('uint')  # Convert to voxel indices

                img = np.zeros([meshsize] * dim)  # Blank n-dimensional volume
                for pt in inds:
                    img[tuple(pt)] += 1

                imsm = ndi.gaussian_filter(img.T, [5] * dim)
                imgs[reg] = imsm / np.max(imsm) if norm_ else imsm

        col = 0

        # For first panel of row, print version as y label

        axs[row, 0].set_ylabel(vers, fontsize=6)

        for reg in regs:
            ax = axs[row,col]

            if smooth:
                ax.imshow(imgs[reg], origin='lower', cmap='viridis')
                ax.set_aspect('equal')
                ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
                for spine in ['top', 'right', 'bottom']:
                    ax.spines[spine].set_visible(False)
            else:    
                ax.scatter(coords[reg][0], coords[reg][1], color=regcol[reg], 
                        s=0.05)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.set_xlim([0, 1])
                ax.set_ylim([0, 1])

            ax.set_aspect('equal')

            # only print titles of regions for first row
            if row == 0:
                ax.set_title(f'{reg} \n {sum(r["acs"] == reg)}', color=pal[reg])

            col += 1

        row += 1

    fig.tight_layout()
    fig.savefig(Path(pth_dmn.parent, 'imgs', 'overleaf_pdf',
         f'umap_si_norm{norm_}_smooth{smooth}.png'), dpi=100, bbox_inches='tight')


##############
# multi panel plotting
##############


def adjust_subplots(fig, adjust=5, extra=2):
    width, height = fig.get_size_inches() / MM_TO_INCH
    if not isinstance(adjust, int):
        assert len(adjust) == 4
    else:
        adjust = [adjust] *  4
    fig.subplots_adjust(top=1 - adjust[0] / height, bottom=(adjust[1] + extra) / height,
                        left=adjust[2] / width, right=1 - adjust[3] / width)


def region_level_mixed_selectivity():
    '''
    One row, 4 columns of panels.
    Panel a: 
        selection of 4 regions with max entropy 
        and 4 with min, of the cluster counts
    Panel b: 
        selection of 4 regions with max entropy 
        and 4 with min, of the decoding scores
    Panel c:
        scatter plot of specialization scores
        of cluster counts versus decoding scores
    Panel d:
        scatter plot of specialization scores
        of cluster counts versus Harris hierarchy
    '''     

    fig = plt.figure(figsize=(183 * MM_TO_INCH, 80 * MM_TO_INCH))
    width, height = fig.get_size_inches() / MM_TO_INCH

    xspans1 = get_coords(width, ratios=[0.5, 0.5, 0.5, 0.5, 1, 1],  
                         space=3, pad=3, span=(0, 1))

    yspans0 = get_coords(height, ratios=[1, 1, 1, 1], 
                        space=3, pad=3, span=(0, 1))

    yspans1 = get_coords(height, ratios=[1], 
                            space=3, pad=3, span=(0, 1))

    a_0 = fg.place_axes_on_grid(fig, xspan=xspans1[0],
                        yspan=yspans0[0])

    pan_a0 = {f'a_{i}': fg.place_axes_on_grid(fig, xspan=xspans1[0],
                        yspan=yspans0[i], sharex=a_0, sharey=a_0) 
                        for i in range(1,4)}

    pan_a1 = {f'a_{i + 4}': fg.place_axes_on_grid(fig, xspan=xspans1[1],
                        yspan=yspans0[i], sharex=a_0, sharey=a_0) 
                        for i in range(4)}   



    b_0 = fg.place_axes_on_grid(fig, xspan=xspans1[2],
                        yspan=yspans0[0])

    pan_b0 = {f'b_{i}': fg.place_axes_on_grid(fig, xspan=xspans1[2],
                        yspan=yspans0[i], sharex=b_0, sharey=b_0) 
                        for i in range(1,4)}

    pan_b1 = {f'b_{i + 4}': fg.place_axes_on_grid(fig, xspan=xspans1[3],
                        yspan=yspans0[i], sharex=b_0, sharey=b_0) 
                        for i in range(4)}


    axs = {'a_0': a_0,
            'b_0': b_0,
            'c': fg.place_axes_on_grid(fig, xspan=xspans1[4], yspan=yspans1[0]),
           'd': fg.place_axes_on_grid(fig, xspan=xspans1[5], yspan=yspans1[0])}

    for h in [pan_a0, pan_a1, pan_b0, pan_b1]:       
        axs.update(h)

    for i in range(1, 8):
        axs[f'a_{i}'].sharex(axs['a_0'])
        axs[f'a_{i}'].sharey(axs['a_0'])
        axs[f'b_{i}'].sharex(axs['b_0'])
        axs[f'b_{i}'].sharey(axs['b_0'])
        axs[f'a_{i}'].tick_params(labelleft=False)
        axs[f'b_{i}'].tick_params(labelleft=False)

    single_regions=['PA', 'MOB', 'MS', 'VISpor',
                    'SIM', 'SCm', 'MRN', 'PRNr']

    clus_freqs(foc='Beryl', single_regions=single_regions, 
               axs = [axs[f'a_{i}'] for i in range(4)] +
                     [axs[f'a_{i + 4}'] for i in range(4)])

    clus_freqs(foc='dec', single_regions=single_regions, 
               axs = [axs[f'b_{i}'] for i in range(4)] +
                     [axs[f'b_{i + 4}'] for i in range(4)])        

    scat_dec_clus(axs=axs['c'])
    scat_dec_clus(axs=axs['d'], harris=True, log_scale=False)

    labels = []
    padx = 2
    pady = 2
    labels.append(add_label('a', fig, xspans1[0], yspans0[0], 
        padx, pady, fontsize=8))
    labels.append(add_label('b', fig, xspans1[2], yspans0[0], 
        padx, pady, fontsize=8))
    labels.append(add_label('c', fig, xspans1[-2], yspans0[0], 
        padx, pady, fontsize=8))
    labels.append(add_label('d', fig, xspans1[-1], yspans0[0], 
        padx, pady, fontsize=8))

    fg.add_labels(fig, labels)

    adjust = 1
    # Depending on the location of axis labels leave a bit more space
    extra =  1
    fig.subplots_adjust(top=1-adjust/height, bottom=(adjust + extra)/height, 
                        left=adjust/width, right=1-adjust/width)

    
