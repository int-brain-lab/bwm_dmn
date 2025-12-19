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
from sklearn.manifold import SpectralEmbedding
import sys
sys.path.append('Dropbox/scripts/IBL/')
from granger import get_volume, get_centroids, get_res, get_structural, get_ari
from state_space_bwm import get_cmap_bwm, pre_post

from scipy.signal import hilbert
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
    pearsonr, spearmanr, kruskal, rankdata, 
    linregress, entropy, energy_distance, ttest_ind)
from scipy.spatial import distance
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform, cdist, pdist
from skbio.stats.distance import DistanceMatrix, permanova

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from random import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from joblib import Parallel, delayed
from sklearn.utils import parallel_backend
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time, os, re 
from PIL import Image, ImageDraw, ImageFont

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
from uuid import UUID
from venny4py.venny4py import *
from matplotlib.axis import Axis
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import ListedColormap, LinearSegmentedColormap   
from matplotlib.gridspec import GridSpec   
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba, Normalize
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
from matplotlib.colors import to_rgba
from matplotlib.cm import ScalarMappable
from typing import Optional, List, Tuple, Dict, Sequence, Union
from collections import OrderedDict

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
tts__ = [
        'inter_trial', 
        'blockL', 
        'blockR', 
        'quiescence', 
        'block_change_s', 
        'stimLbLcL', 
        'stimLbRcL', 
        'stimRbRcR', 
        'stimRbLcR', 
        'mistake_s', 
        'motor_init', 
        'block_change_m', 
        'sLbLchoiceL', 
        'sLbRchoiceL', 
        'sRbRchoiceR', 
        'sRbLchoiceR', 
        'mistake_m', 
        'choiceL', 
        'choiceR', 
        'fback1', 
        'fback0']
     
peth_ila = [
    r"$\mathrm{rest}$",
    r"$\mathrm{L_b}$",
    r"$\mathrm{R_b}$",
    r"$\mathrm{quies}$",
    r"$\mathrm{change_b, s}$",
    r"$\mathrm{L_sL_cL_b, s}$",
    r"$\mathrm{L_sL_cR_b, s}$",
    r"$\mathrm{R_sR_cR_b, s}$",
    r"$\mathrm{R_sR_cL_b, s}$",
    r"$\mathrm{mistake, s}$",
    r"$\mathrm{m}$",
    r"$\mathrm{change_b, m}$",
    r"$\mathrm{L_sL_cL_b, m}$",
    r"$\mathrm{L_sL_cR_b, m}$",
    r"$\mathrm{R_sR_cR_b, m}$",
    r"$\mathrm{R_sR_cL_b, m}$",
    r"$\mathrm{mistake, m}$",
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
    'Stim surprise': ['stimLbRcL', 'stimRbLcR'],
    'Stim congruent': ['stimLbLcL', 'stimRbRcR'],
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


# ----------------- helper: concatenate members on trials axis -----------------
def _concat_trials_over_members(D, members, extractor):
    """
    Pull (N,T,M_i) per member; concatenate along trials axis -> (N,T,sum M_i).
    Skips missing/empty members. Requires consistent N,T across present members.
    """
    Xs = []
    for name in members:
        try:
            X = extractor(D, name)  # (N,T,M)
            if X.size and X.shape[2] > 0:
                Xs.append(X)
        except (KeyError, ValueError):
            continue
    if not Xs:
        N = len(D.get('ids', []))
        return np.empty((N, 0, 0), dtype=np.float32)
    N0, T0 = Xs[0].shape[:2]
    for X in Xs[1:]:
        if X.shape[0] != N0 or X.shape[1] != T0:
            raise ValueError(f"Inconsistent shapes among members: expected (N={N0},T={T0}), got {X.shape}")
    return np.concatenate(Xs, axis=2)  # stack trials


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

def _count_real_trials_3d(X: np.ndarray) -> tuple[int, np.ndarray]:
    """
    Count 'real' trials in a (N, T, M) array, defining placeholder trials
    as those with all-NaN across (N,T) for a given trial index.
    Returns (count, mask[M]).
    """
    if X is None or X.size == 0 or X.ndim != 3:
        return 0, np.zeros(0, dtype=bool)
    # trial is real if it contains at least one finite value
    real_mask = ~np.all(~np.isfinite(X), axis=(0, 1))
    return int(np.sum(real_mask)), real_mask


def _concat_trials_over_members_real(D, members, extractor):
    """
    Like _concat_trials_over_members but drops placeholder (all-NaN) trials
    before concatenating members along the trial axis.
    """
    mats = []
    for m in members:
        Xm = extractor(D, m)     # (N,T,Mm) or empty
        if Xm is None or Xm.size == 0:
            continue
        _, mk = _count_real_trials_3d(Xm)
        if mk.size:
            Xm = Xm[:, :, mk]
            if Xm.shape[2] > 0:
                mats.append(Xm.astype(np.float32))
    if not mats:
        return np.empty((len(D['ids']), 0, 0), dtype=np.float32)
    return np.concatenate(mats, axis=2)

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


def deep_in_block(trials, pleft, depth=3):

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


def first_three_after_block_switch(trials):
    pl = trials['probabilityLeft'].values
    n = len(pl)

    # indices where block changes (start of new block)
    bs = np.where(np.diff(pl) != 0)[0] + 1

    mask = np.zeros(n, dtype=bool)

    # mark block_start, block_start+1, block_start+2
    for b in bs:
        mask[b : min(b+3, n)] = True

    return mask


def concat_PETHs(pid, get_tts: bool = False, vers: str = 'concat', 
                 require_all: bool = False):
    """
    Build PETH bundles containing per-trial binned activity (no averaging, no CV).

    Returns a single dict D with:
      - ids, xyz, channels, axial_um, lateral_um, uuids
      - trial_names : list[str]
      - tls         : dict[str, int]   (trial counts)
      - ws          : list[np.ndarray] (per PETH, array of shape (n_trials, n_neurons, n_timebins), dtype float32)

    Notes
    -----
    - This function bins spikes into time bins but DOES NOT average across trials.
    - All arrays are stored as float32 for memory efficiency.
    - Cross-validation has been removed entirely; do it later from the per-trial arrays.
    """
    
    eid, probe = one.pid2eid(pid)

    print('eid', eid)
    # Load in trials data and mask bad trials (False if bad)
    trials, mask = load_trials_and_mask(one, str(eid),
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
            'quiescence': ['stimOn_times', mask, 
                       [0.4, -0.1]],
            'block_change_s': ['stimOn_times', np.bitwise_and(
                mask, first_three_after_block_switch(trials)),
                       [0, 0.15]],
            'stimLbLcL': ['stimOn_times',
                 np.bitwise_and.reduce([mask, 
                    ~np.isnan(trials[f'contrastLeft']),
                    trials['probabilityLeft'] == 0.8,
                    deep_in_block(trials, 0.8),
                    trials['choice'] == 1]), [0, 0.15]], 
            'stimLbRcL': ['stimOn_times',            
                np.bitwise_and.reduce([mask,
                    ~np.isnan(trials[f'contrastLeft']), 
                    trials['probabilityLeft'] == 0.2,                       
                    deep_in_block(trials, 0.2),
                    trials['choice'] == 1]), [0, 0.15]],
            'stimRbRcR': ['stimOn_times',
                 np.bitwise_and.reduce([mask, 
                    ~np.isnan(trials[f'contrastRight']), 
                    trials['probabilityLeft'] == 0.2,
                    deep_in_block(trials, 0.2),
                    trials['choice'] == -1]), 
                                        [0, 0.15]],        
            'stimRbLcR': ['stimOn_times',
                 np.bitwise_and.reduce([mask, 
                    ~np.isnan(trials[f'contrastRight']), 
                    trials['probabilityLeft'] == 0.8,
                    deep_in_block(trials, 0.8),
                    trials['choice'] == -1]), 
                                        [0, 0.15]],
            'mistake_s': ['stimOn_times',
                np.bitwise_and.reduce([mask,
                    trials['feedbackType'] == -1]), 
                                        [0, 0.15]],
            'motor_init': ['firstMovement_times', mask, 
                       [0.15, 0]],  
            'block_change_m': ['firstMovement_times', np.bitwise_and(
                mask, first_three_after_block_switch(trials)),
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
            'mistake_m': ['firstMovement_times',
                np.bitwise_and.reduce([mask,
                    trials['feedbackType'] == -1]), 
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


    # Metadata
    meta = {
        'ids':        np.array(clusters['atlas_id']),
        'xyz':        np.array(clusters[['x', 'y', 'z']]),
        'channels':   np.array(clusters['channels']),
        'axial_um':   np.array(clusters['axial_um']),
        'lateral_um': np.array(clusters['lateral_um']),
        'uuids':      np.array(clusters['uuids']),
        'trial_names': list(tts.keys()),
    }

    # Helper: bin per-trial, concatenate within-bin shifts, no averaging
    def _bin_peth(event_times: np.ndarray, pre: float, post: float):
        if len(event_times) == 0:
            return None
        st = int(T_BIN // sts)  # number of within-bin shifts; assumes globals T_BIN and sts
        bis = []
        for ts in range(st):
            bi, _ = bin_spikes2D(
                spikes['times'],
                clusters['cluster_id'][spikes['clusters']],
                clusters['cluster_id'],
                np.asarray(event_times) + ts * sts,
                pre, post, T_BIN
            )  # bi: (n_trials, n_neurons, n_bins)
            bis.append(bi.astype(np.float32))
        if bis[0].shape[0] == 0:
            return None
        ntr, nn, nbin = bis[0].shape
        ar = np.zeros((ntr, nn, st * nbin), dtype=np.float32)
        for ts in range(st):
            ar[:, :, ts::st] = bis[ts]
        return ar  # (n_trials, n_neurons, n_timebins_concat), float32


    # Utility: first finite event time in a column, else 0.0
    def _first_finite(ev_col: np.ndarray) -> float:
        ev = np.asarray(ev_col, dtype=float)
        finite = np.isfinite(ev)
        return float(ev[finite][0]) if np.any(finite) else 0.0

    # Build outputs (no CV)
    tls: dict = {}
    ws: list = []

    for key in meta['trial_names']:
        align_col, trial_mask, (pre, post) = tts[key]
        # combine global + per-PETH mask
        events_all = trials[align_col][np.bitwise_and.reduce([mask, trial_mask])].to_numpy()
        tls[key] = int(len(events_all))

        if tls[key] == 0:
            if require_all:
                raise ValueError(f"Missing PETH '{key}' for pid={pid} (0 trials).")
            # Create placeholder with correct (1, n_neurons, n_timebins)
            # by binning a one-event dummy to infer shape, then fill with NaNs.
            dummy_t = _first_finite(trials[align_col].to_numpy())
            ref = _bin_peth(np.array([dummy_t], dtype=float), pre, post)
            if ref is None:
                # Degenerate fallback: infer neurons and try again with t=0.0
                ref = _bin_peth(np.array([0.0], dtype=float), pre, post)
            if ref is None:
                # As last resort, assume n_neurons from clusters, n_timebins=1
                nn = int(len(clusters))
                ws.append(np.full((1, nn, 1), np.nan, dtype=np.float32))
            else:
                _, nn, nt = ref.shape
                ws.append(np.full((1, nn, nt), np.nan, dtype=np.float32))
            continue

        # Normal path
        w = _bin_peth(events_all, pre, post)
        ws.append(w)

    D = dict(meta)
    D['tls'] = tls
    D['ws']  = ws

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



def rgb_to_hex(rgb):
    return '#{:02X}{:02X}{:02X}'.format(
        int(rgb[0] * 255),
        int(rgb[1] * 255),
        int(rgb[2] * 255)
    )


def get_hierarchy(reg):
    '''
    Get structural hierarchy tree for a given Beryl region.
    Beryl abbreviation is bolded using HTML-style <b> tags.
    '''
    a, _ = get_allen_info()
    a['Beryl'] = br.id2acronym(a['id'].values, mapping='Beryl')
    a['Cosmos'] = br.id2acronym(a['id'].values, mapping='Cosmos')

    cdict = Counter(a['Cosmos'])
    del cdict['void']
    del cdict['root']
    cosmos_ids = br.acronym2id(list(cdict.keys()))


    # Get path and truncate to last 5 ancestors
    idp = a['structure_id_path'][a['Beryl'] == reg].values[0]
    idp = idp.split('/')[-6:-1]

    # Find index of first parent in Cosmos
    cos_i = next(i for i, x in enumerate(idp) if int(x) in cosmos_ids)
    idp = idp[cos_i:]
    idp_int = list(map(int, idp))

    col = rgb_to_hex(pal[reg])


    return ' / '.join([
        f'<font color="{col}">{get_name(br.id2acronym(x))} (<b>{br.id2acronym(x)[0]}</b>)</font>'
        for x in idp_int
    ])


def print_full_structure_tree(filename='structure_tree.pdf'):
    '''
    Print all Beryl region hierarchies line by line in two-column PDF.
    Each line is colored according to its Beryl region color (pal[reg]).
    Font is very small (3 pt) for compact layout.
    '''
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Frame, PageTemplate
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.units import mm
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.lib import colors

    # Load region data and palette
    r = regional_group('Beryl', vers='concat', ephys=False, rerun=False)
    p = Path(iblatlas.__file__).parent / 'beryl.npy'
    regs_can = br.id2acronym(np.load(p), mapping='Beryl')
    regs_ = Counter(r['acs'])
    reg_ord = [reg for reg in regs_can if reg in regs_]

    a, pal_raw = get_allen_info()
    pal = {k: rgb_to_hex(v) for k, v in pal_raw.items()}  # acronym → hex

    a['Cosmos'] = br.id2acronym(a['id'].values, mapping='Cosmos')
    cosmos_acronyms = set(a['Cosmos']) - {'root', 'void'}
    cosmos_ids = set(br.acronym2id(list(cosmos_acronyms)))

    id2name = dict(zip(br.id, br.name))
    id2acr = dict(zip(br.id, br.acronym))


    # Page setup: 2 columns
    width = 180 * mm
    height = 170 * mm
    margin = 1 * mm
    gap = 1 * mm
    usable_width = width - 2 * margin - gap
    column_height = height - 2 * margin
    left_width = usable_width * 3.2 / 5
    right_width = usable_width * 1.8 / 5

    frame_left = Frame(margin, margin, left_width, column_height,
                       leftPadding=0, rightPadding=2, topPadding=0, bottomPadding=0)
    frame_right = Frame(margin + left_width + gap, margin, right_width, column_height,
                        leftPadding=2, rightPadding=0, topPadding=0, bottomPadding=0)


    template = PageTemplate(id='TwoCol', frames=[frame_left, frame_right])
    doc = SimpleDocTemplate(filename, pagesize=(width, height))
    doc.addPageTemplates([template])

    # Very tight style
    style = ParagraphStyle(
        name='Tight',
        fontSize=3,
        leading=3.5,
        spaceBefore=0,
        spaceAfter=0,
        alignment=TA_LEFT
    )

    story = []
    for reg in reg_ord:
        try:
            hierarchy_text = get_hierarchy(reg)
            story.append(Paragraph(hierarchy_text, style))
        except Exception as e:
            print(f"Skipping {reg} due to error: {e}")

    doc.build(story)



def regional_group(
    mapping,
    vers: str = "concat",
    ephys: bool = False,
    grid_upsample: int = 0,
    nclus: int = 100,
    cv: bool = True,
    locality: float = 0.75,
    time_lag_window: int = 5,
    symmetric: bool = False,
    rerun: bool = False,
):
    """
    Group / color neurons for visualization and downstream analyses.
    """
    pth_res = Path(one.cache_dir, "dmn", "res")

    def _cache_path(kind: str) -> Path:
        """
        kind:
          - 'stack': precomputed stacked features from stack_concat (DO NOT include rm hyperparams)
          - 'rm'   : rastermap cache (include rm hyperparams + nclus)
        """
        if kind == "stack":
            base = f"{vers}"
            base += f"_cv{cv}"
            base += f"_ephys{ephys}"
            return pth_res / (base + ".npy")

        if kind == "rm":
            base = f"rm_{vers}"
            base += f"_cv{cv}"
            base += f"_ephys{ephys}"
            base += f"_n{int(nclus)}"
            base += f"_grid{int(grid_upsample)}"
            base += f"_loc{float(locality):.3f}"
            base += f"_tlag{int(time_lag_window)}"
            base += f"_sym{int(bool(symmetric))}"
            return pth_res / (base + ".npy")

        raise ValueError(f"Unknown cache kind: {kind}")

    # ---------- load stack ----------
    stack_path = _cache_path("stack")
    if not stack_path.is_file():
        raise FileNotFoundError(
            f"Stack file not found: {stack_path}\n"
            "Expected stack caches to depend only on vers/cv/ephys (not rm hyperparams)."
        )

    r = np.load(stack_path, allow_pickle=True).flat[0]
    print(
        f"mapping {mapping}, vers {vers}, ephys {ephys}, nclus {nclus}, rerun {rerun}, cv {cv}, "
        f"{len(r['ids'])} neurons loaded."
    )

    r["len"] = OrderedDict((k, int(r["len"][k])) for k in r["ttypes"])

    if "xyz" not in r:
        raise KeyError("Saved stack lacks 'xyz'.")
    r["nums"] = np.arange(r["xyz"].shape[0], dtype=int)

    feat_key = "concat_z"
    r["_order_signature"] = (
        "|".join(f"{k}:{r['len'][k]}" for k in r["ttypes"])
        + f"|shape:{r[feat_key].shape}"
    )

    # ---------- mapping ----------
    if mapping == "rm":
        feat = feat_key
        if feat not in r:
            raise KeyError(f"Feature '{feat}' not found in stack.")

        rm_cache_path = _cache_path("rm")
        labels = None
        isort = None

        # try load cache; if missing/invalid -> compute from scratch
        if (not rerun) and rm_cache_path.is_file():
            try:
                cached = np.load(rm_cache_path, allow_pickle=True).flat[0]
                if (
                    isinstance(cached, dict)
                    and cached.get("order_sig") == r["_order_signature"]
                    and "rm_labels" in cached
                    and "isort" in cached
                ):
                    labels = np.asarray(cached["rm_labels"], dtype=int).reshape(-1)
                    isort = np.asarray(cached["isort"], dtype=int).reshape(-1)
                    if labels.shape[0] != r[feat].shape[0] or isort.shape[0] != r[feat].shape[0]:
                        labels = None
                        isort = None
                    else:
                        print(f"[rm] using cached labels/isort ({rm_cache_path.name})")
            except Exception as e:
                print(f"[rm] cache read error; recomputing Rastermap: {e}")
                labels = None
                isort = None

        if labels is None or isort is None:
            feat_used = "concat_z_train" if cv else feat
            if feat_used not in r:
                raise KeyError(f"Feature '{feat_used}' not found in stack.")

            print(f"[rm] computing Rastermap (n_clusters={nclus}) on {feat_used}")
            model = Rastermap(
                n_PCs=200,
                n_clusters=nclus,
                grid_upsample=grid_upsample,
                locality=locality,
                time_lag_window=time_lag_window,
                bin_size=1,
                symmetric=symmetric,
            ).fit(r[feat_used])

            labels = np.asarray(model.embedding_clust, dtype=int)
            if labels.ndim > 1:
                labels = labels[:, 0]
            isort = np.asarray(model.isort, dtype=int).reshape(-1)

            if labels.shape[0] != r[feat].shape[0] or isort.shape[0] != r[feat].shape[0]:
                raise ValueError("Rastermap outputs do not match data length.")

            np.save(
                rm_cache_path,
                {"rm_labels": labels, "isort": isort, "order_sig": r["_order_signature"]},
                allow_pickle=True,
            )
            print(f"[rm] wrote cache ({rm_cache_path.name})")

        # colors/legend as before (abbrev; keep your existing block)
        clusters = labels.copy()
        unique = np.unique(clusters)
        norm_vals = clusters / float(unique.max()) if unique.size > 1 else np.zeros_like(clusters, dtype=float)
        cmap = mpl.cm.get_cmap("Spectral")
        cols = cmap(norm_vals)
        regs = unique
        color_map = {reg: cols[clusters == reg][0] for reg in regs}
        r["els"] = [Line2D([0], [0], color=color_map[reg], lw=4, label=f"{reg}") for reg in regs]
        r["Beryl"] = np.array(br.id2acronym(r["ids"], mapping="Beryl"))
        r["acs"] = labels
        r["cols"] = cols
        r["isort"] = isort

        return r

    # other mappings unchanged...
    # ...
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


def NN(x, y, decoder='LDA', CC=1.0, confusion=False,
       return_weights=False, shuf=False, verb=True):
    '''
    Decode region label y from activity x with parallelized cross-validation.
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
    
    folds = 5
    kf = StratifiedKFold(n_splits=folds, shuffle=True)

    if verb:
        print('input dimension:', np.shape(x))    
        print(f'# classes = {nclasses}')
        print('x.shape:', x.shape, 'y.shape:', y.shape)
        print(f'{folds}-fold cross validation')
        if shuf: 
            print('labels are SHUFFLED')

    def process_fold(train_index, test_index):
        """
        Process a single fold of cross-validation.
        """
        sc = StandardScaler()
        train_X = sc.fit_transform(x[train_index])
        test_X = sc.fit_transform(x[test_index])

        train_X = x[train_index]
        test_X = x[test_index]

        train_y = y[train_index]
        test_y = y[test_index]

        if decoder == 'LR':
            clf = LogisticRegression(C=CC, random_state=0, n_jobs=-1,
                max_iter=1000)
        elif decoder == 'LDA':
            clf = LinearDiscriminantAnalysis()
        else:
            raise ValueError('Unsupported model type. Use "LR" or "LDA".')

        clf.fit(train_X, train_y)
        y_pred_test = clf.predict(test_X)
        y_pred_train = clf.predict(train_X)

        res_test = np.mean(test_y == y_pred_test)
        res_train = np.mean(train_y == y_pred_train)

        return (res_train, res_test, y_pred_train, y_pred_test, train_y, test_y)

    with parallel_backend('loky'):
        results = Parallel(n_jobs=-1)(
            delayed(process_fold)(train_index, test_index)
            for train_index, test_index in kf.split(x, y)
        )

    # Collect results from parallel processing
    for res_train, res_test, y_pred_train, y_pred_test, train_y, test_y in results:
        acs.append([res_train, res_test])
        yp_train.append(y_pred_train)
        yp_test.append(y_pred_test)
        yt_train.append(train_y)
        yt_test.append(test_y)

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
            clf = LogisticRegression(C=CC, random_state=0, n_jobs=-1,
                max_iter=1000)
        else:
            clf = LinearDiscriminantAnalysis()

        clf.fit(x, y)
        return clf.coef_

    if confusion:
        yt_train = np.concatenate(yt_train)
        yp_train = np.concatenate(yp_train)
        yt_test = np.concatenate(yt_test)
        yp_test = np.concatenate(yp_test)
        
        cm_train = confusion_matrix(yt_train, yp_train, normalize='pred')
        cm_test = confusion_matrix(yt_test, yp_test, normalize='pred')
                   
        return cm_train, cm_test, r_train, r_test                              

    return np.array(acs)


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



def lz76_complexity(s: str) -> int:
    """
    Fast LZ76 parser for a binary string.
    Returns the number of parsed phrases (c).
    """
    n = len(s)
    i = 0
    c = 0
    dictionary = set()

    while i < n:
        k = 1
        # extend substring as long as it exists
        while i + k <= n and s[i:i+k] in dictionary:
            k += 1
        dictionary.add(s[i:i+k])
        c += 1
        i += k

    return c


def lzs_pci(x: np.ndarray, rng: np.random.Generator) -> float:
    """
    PCI-style Lempel-Ziv complexity as used in Casali et al. (2013).
    - detrend, z-score
    - Hilbert envelope
    - threshold at its mean
    - binarize to string
    - compute LZ(s) / LZ(shuffled(s))

    Parameters
    ----------
    x : 1D array
    rng : np.random.Generator for reproducible shuffle

    Returns
    -------
    float
        LZ complexity normalized by shuffled surrogate.
    """
    x = np.asarray(x, float)

    # Hilbert envelope
    env = np.abs(hilbert(x))
    th = env.mean()

    # binary string
    s = (env > th).astype(np.uint8)
    s_str = ''.join('1' if b else '0' for b in s)

    # shuffle surrogate (same 0/1 counts)
    M = np.array(list(s_str))
    rng.shuffle(M)
    w_str = ''.join(M.tolist())

    # LZ complexity
    c_s = lz76_complexity(s_str)
    c_w = lz76_complexity(w_str)

    if c_w == 0:
        return 0.0
    return c_s / c_w


def add_lz_to_stack(vers: str = 'concat',
                    ephys: bool = False,
                    cv: bool = False,
                    cv2: bool = False,
                    overwrite: bool = True,
                    seed: int = 0) -> Path:
    """
    Load the appropriate 'vers_*' .npy stack, compute LZ complexity per neuron
    using PCI-style LZs (Hilbert envelope threshold + LZ76 ratio),
    store as r['lz'], and save back to disk.
    """
    rng = np.random.default_rng(seed)

    pth_res = Path(one.cache_dir, 'dmn', 'res')

    def _stack_fname() -> Path:
        if cv and cv2:
            raise ValueError("cv and cv2 cannot both be True.")
        if cv:
            return pth_res / f"{vers}_cvTrue_ephysFalse.npy"
        if cv2:
            return pth_res / f"{vers}_cv2True_ephysFalse.npy"
        return pth_res / f"{vers}_cvFalse_ephysFalse.npy"

    stack_path = _stack_fname()
    if not stack_path.is_file():
        raise FileNotFoundError(stack_path)

    # Load
    t0 = time.perf_counter()
    r = np.load(stack_path, allow_pickle=True).flat[0]

    if 'lz' in r and not overwrite:
        print("[info] lz already exists; skipping computation.")
        return stack_path

    if 'concat_z' not in r:
        raise KeyError(f"'concat_z' not found in {stack_path}")

    data = np.asarray(r['concat_z'])
    if data.ndim != 2:
        raise ValueError("r['concat_z'] must be 2D (neurons × time).")

    N = data.shape[0]
    print(f"[info] Computing LZs for {N} neurons…")

    # Compute LZs per neuron
    t1 = time.perf_counter()
    lz_vals = np.zeros(N, float)
    for i in range(N):
        lz_vals[i] = lzs_pci(data[i], rng)

    t2 = time.perf_counter()

    # Store and save
    r['lz'] = lz_vals
    np.save(stack_path, r, allow_pickle=True)

    t3 = time.perf_counter()

    print(f"[timing] load:       {t1 - t0:6.3f} s")
    print(f"[timing] LZ compute: {t2 - t1:6.3f} s (≈ {(t2 - t1)/N*1000:.2f} ms/neuron)")
    print(f"[timing] save:       {t3 - t2:6.3f} s")
    print(f"[timing] total:      {t3 - t0:6.3f} s")

    return stack_path


'''
##############################################################
### bulk processing
##############################################################
'''


bad_eids = ['642c97ea-fe89-4ec9-8629-5e492ea4019d', '1b715600-0cbc-442c-bd00-5b0ac2865de1', '3a3ea015-b5f4-4e8b-b189-9364d1fc7435', '09394481-8dd2-4d5c-9327-f2753ede92d7', '25d1920e-a2af-4b6c-9f2e-fc6c65576544', '5ae68c54-2897-4d3a-8120-426150704385', 'f819d499-8bf7-4da0-a431-15377a8319d5', '0c828385-6dd6-4842-a702-c5075f5f5e81', '4e560423-5caf-4cda-8511-d1ab4cd2bf7d', '0a018f12-ee06-4b11-97aa-bbbff5448e9f', '3dd347df-f14e-40d5-9ff2-9c49f84d2157', 'e56541a5-a6d5-4750-b1fe-f6b5257bfe7c', 'e6594a5b-552c-421a-b376-1a1baa9dc4fd', '81a1dca0-cc90-47c5-afe3-c277319c47c8', 'ebe090af-5922-4fcd-8fc6-17b8ba7bad6d', '004d8fd5-41e7-4f1b-a45b-0d4ad76fe446', '9dd72e52-5393-4c08-9eca-f7dace2e59f6', '9fc31d79-b56f-46d0-92a0-e9563caf4a7a', '2d9bfc10-59fb-424a-b699-7c42f86c7871', '88224abb-5746-431f-9c17-17d7ef806e6a', '195443eb-08e9-4a18-a7e1-d105b2ce1429', 'ebe2efe3-e8a1-451a-8947-76ef42427cc9', 'a9138924-4395-4981-83d1-530f6ff7c8fc', '6668c4a0-70a4-4012-a7da-709660971d7a', '0802ced5-33a3-405e-8336-b65ebc5cb07c', 'bb8d9451-fdbd-4f46-b52e-9290e8f84d2e', 'f88d4dd4-ccd7-400e-9035-fa00be3bcfa8', '4a45c8ba-db6f-4f11-9403-56e06a33dfa4', '9a629642-3a9c-42ed-b70a-532db0e86199', '08102cfc-a040-4bcf-b63c-faa0f4914a6f', '14127fdb-2e66-4823-b124-f49c128ba94d', '5455a21c-1be7-4cae-ae8e-8853a8d5f55e', '781b35fd-e1f0-4d14-b2bb-95b7263082bb', '3f6e25ae-c007-4dc3-aa77-450fd5705046', 'dc962048-89bb-4e6a-96a9-b062a2be1426', 'a4a74102-2af5-45dc-9e41-ef7f5aed88be', '2f63c555-eb74-4d8d-ada5-5c3ecf3b46be', '5bcafa14-71cb-42fa-8265-ce5cda1b89e0', 'f7335a49-4a98-46d2-a8ce-d041d2eac1d6', 'd57df551-6dcb-4242-9c72-b806cff5613a', '8c025071-c4f3-426c-9aed-f149e8f75b7b', '196a2adf-ff83-49b2-823a-33f990049c2e', 'f27e6cd6-cdd3-4524-b8e3-8146046e2a7d', '51e53aff-1d5d-4182-a684-aba783d50ae5', 'ff96bfe1-d925-4553-94b5-bf8297adf259', 'c23b4118-db40-4333-af1d-933154b533c6', '28741f91-c837-4147-939e-918d38d849f2', 'e9fc0a2d-c69d-44d1-9fa3-314782387cae', '9545aa05-3945-4054-a5c3-a259f7209d61', 'e49d8ee7-24b9-416a-9d04-9be33b655f40', '30af8629-7b96-45b7-8778-374720ddbc5e', '8928f98a-b411-497e-aa4b-aa752434686d', '62902992-8432-46fb-af12-6392012e58c7', '54238fd6-d2d0-4408-b1a9-d19d24fd29ce', '23c75e0b-05d8-452e-8efb-a3687ab94079', '239dd3c9-35f3-4462-95ee-91b822a22e6b', 'd32876dd-8303-4720-8e7e-20678dc2fd71', '8a3a0197-b40a-449f-be55-c00b23253bbf', 'ca4ecb4c-4b60-4723-9b9e-2c54a6290a53', 'a2ec6341-c55f-48a0-a23b-0ef2f5b1d71e', '3537d970-f515-4786-853f-23de525e110f', 'c728f6fd-58e2-448d-aefb-a72c637b604c', 'a34b4013-414b-42ed-9318-e93fbbc71e7b', '7af49c00-63dd-4fed-b2e0-1b3bd945b20b', 'f8041c1e-5ef4-4ae6-afec-ed82d7a74dc1', 'caa5dddc-9290-4e27-9f5e-575ba3598614', '687017d4-c9fc-458f-a7d5-0979fe1a7470', '4364a246-f8d7-4ce7-ba23-a098104b96e4', '158d5d35-a2ab-4a76-87b0-51048c5d5283', '7f5df7eb-cf36-4589-a20a-14b535441142', '7cc74598-9c1b-436b-84fa-0bf89f31adf6', 'bb6a5aae-2431-401d-8f6a-9fdd6de655a9', 'd85c454e-8737-4cba-b6ad-b2339429d99b', '239cdbb1-68e2-4eb0-91d8-ae5ae4001c7a', '3f859b5c-e73a-4044-b49e-34bb81e96715', 'b69b86be-af7d-4ecf-8cbf-0cd356afa1bd', '549caacc-3bd7-40f1-913d-e94141816547', '4ddb8a95-788b-48d0-8a0a-66c7c796da96', '810b1e07-009e-4ebe-930a-915e4cd8ece4', '2d5f6d81-38c4-4bdc-ac3c-302ea4d5f46e', 'a7eba2cf-427f-4df9-879b-e53e962eae18', '75b6b132-d998-4fba-8482-961418ac957d', '1d4a7bd6-296a-48b9-b20e-bd0ac80750a5', '15948667-747b-4702-9d53-354ac70e9119', '7235b10c-6621-44ea-abe9-01559633472d', 'f25642c6-27a5-4a97-9ea0-06652db79fbd', '6434f2f5-6bce-42b8-8563-d93d493613a2', '6bf810fd-fbeb-4eea-9ea7-b6791d002b22', '0deb75fb-9088-42d9-b744-012fb8fc4afb', '3f71aa98-08c6-4e79-b4c8-00eae4f03eff', '90c61c38-b9fd-4cc3-9795-29160d2f8e55', '8c552ddc-813e-4035-81cc-3971b57efe65', '5339812f-8b91-40ba-9d8f-a559563cc46b', '8c2f7f4d-7346-42a4-a715-4d37a5208535', 'f9860a11-24d3-452e-ab95-39e199f20a93', 'b22f694e-4a34-4142-ab9d-2556c3487086', 'f359281f-6941-4bfd-90d4-940be22ed3c3', 'de905562-31c6-4c31-9ece-3ee87b97eab4', '8a1cf4ef-06e3-4c72-9bc7-e1baa189841b', '58c4bf97-ec3b-45b4-9db4-d5d9515d5b00', '768a371d-7e88-47f8-bf21-4a6a6570dd6e', '61caa69d-088b-465a-b9d0-d75341dabac6', '0cbeae00-e229-4b7d-bdcc-1b0569d7e0c3', 'f99ac31f-171b-4208-a55d-5644c0ad51c3', 'd832d9f7-c96a-4f63-8921-516ba4a7b61f', '251ece37-7798-477c-8a06-2845d4aa270c', '5157810e-0fff-4bcf-b19d-32d4e39c7dfc',
'1928bf72-2002-46a6-8930-728420402e01',
  '283ecb4c-e529-409c-9f0a-8ea5191dcf50',
  '2ab7d2c2-bcb7-4ae6-9626-f3786c22d970',
  '35ed605c-1a1a-47b1-86ff-2b56144f55af',
  '5d01d14e-aced-4465-8f8e-9a1c674f62ec',
  '7be8fec4-406b-4e74-8548-d2885dcc3d5e',
  '8ca740c5-e7fe-430a-aa10-e74e9c3cbbe8',
  '91a3353a-2da1-420d-8c7c-fad2fedfdd18',
  'b52182e7-39f6-4914-9717-136db589706e',
  'c557324b-b95d-414c-888f-6ee1329a2329',
  'd23a44ef-1402-4ed7-97f5-47e9a7a504d9',
  'd2918f52-8280-43c0-924b-029b2317e62c',
  'd71e565d-4ddb-42df-849e-f99cfdeced52',
  'd7e60cc3-6020-429e-a654-636c6cc677ea',
  'e349a2e7-50a3-47ca-bc45-20d1899854ec',
  'e535fb62-e245-4a48-b119-88ce62a6fe67',
  'ecb5520d-1358-434c-95ec-93687ecd1396',
  'f4ffb731-8349-4fe4-806e-0232a84e52dd',
  'fb70ebf7-8175-42b0-9b7a-7c6e8612226e']



def get_all_PETHs_parallel(
    eids_plus=None,
    vers: str = 'concat',
    require_all: bool = False,
    n_workers: int = 5,
    bad_eids: list = bad_eids,
    overwrite: bool = False,
):
    """
    Compute and save PETH bundles for BWM insertions in parallel (threads).

    Parameters
    ----------
    eids_plus : array-like of (eid, probe_name, pid), optional
        If None, uses bwm_query(one) to get all insertions.
    vers : str
        PETH variant ('concat', 'contrast', ...); becomes subfolder under cache.
    require_all : bool
        Skip insertions with missing/zero-trial conditions inside concat_PETHs.
    n_workers : int
        Number of concurrent worker threads (default 5).
    bad_eids : list/sequence of str, optional
        EIDs to exclude up front (pre-known problematic sessions).
    overwrite : bool
        If False, skip saving if the target .npy already exists.

    Returns
    -------
    dict
        {
          'n_total': int,          # total candidate insertions (after filtering bad_eids)
          'n_done': int,           # saved successfully
          'n_skipped_existing': int,
          'n_failed': int,
          'failures': List[(eid, probe, pid, exc_str)],
          'suggest_bad_eids': List[str],  # EIDs that failed, de-duplicated
          'out_dir': str
        }
    """
    t0_all = time.perf_counter()

    # Discover insertions if not provided
    if eids_plus is None:
        df = bwm_query(one)
        eids_plus = df[['eid', 'probe_name', 'pid']].values

    # Exclude known-bad EIDs up front
    bad_set = set(bad_eids or [])
    eids_plus = [(eid, probe, pid) for (eid, probe, pid) in eids_plus if eid not in bad_set]

    out_dir = Path(one.cache_dir, 'dmn', vers)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Shared counters (thread-safe)
    lock = threading.Lock()
    n_done = 0
    n_failed = 0
    n_skipped_existing = 0
    failures = []
    suggest_bad_eids = set()

    def _target_path(eid: str, probe: str) -> Path:
        return out_dir / f"{eid}_{probe}.npy"

    def _worker(eid: str, probe: str, pid: str):
        nonlocal n_done, n_failed, n_skipped_existing
        t0 = time.perf_counter()
        target = _target_path(eid, probe)

        # Skip existing file unless overwrite=True
        if (not overwrite) and target.exists():
            with lock:
                n_skipped_existing += 1
            print(f"[skip existing] {eid}_{probe}  ({time.perf_counter()-t0:.2f}s)")
            return

        try:
            D = concat_PETHs(pid, vers=vers, require_all=require_all)
            np.save(target, D, allow_pickle=True)
            gc.collect()
            with lock:
                n_done += 1
            print(f"[ok] {eid}_{probe}  ({time.perf_counter()-t0:.2f}s)")
        except BaseException as ex:
            with lock:
                n_failed += 1
                failures.append((eid, probe, pid, f"{type(ex).__name__}: {ex}"))
                suggest_bad_eids.add(eid)
            gc.collect()
            print(f"[fail] {eid}_{probe} | {type(ex).__name__}: {ex}  ({time.perf_counter()-t0:.2f}s)")

    tasks = list(eids_plus)
    n_total = len(tasks)
    print(f"Processing {n_total} insertions with {n_workers} threads "
          f"(require_all={require_all}, overwrite={overwrite})")

    # Thread pool
    with ThreadPoolExecutor(max_workers=max(1, int(n_workers))) as ex:
        futures = [ex.submit(_worker, eid, probe, pid) for (eid, probe, pid) in tasks]
        # Optional: iterate for early exception surfacing
        for _ in as_completed(futures):
            pass

    dt = time.perf_counter() - t0_all
    print(f"\nDone in {dt/60:.2f} min  |  ok={n_done}  fail={n_failed}  skipped_existing={n_skipped_existing}")
    if failures:
        print(f"{len(failures)} failures:")
        for eid, probe, pid, msg in failures[:25]:
            print("  ", eid, probe, pid, "|", msg)
        if len(failures) > 25:
            print(f"  ... and {len(failures) - 25} more")

    # Suggest updating bad_eids for next runs
    if suggest_bad_eids:
        print("\nConsider adding these EIDs to bad_eids for future runs:")
        for eid in sorted(suggest_bad_eids):
            print("  ", eid)

    return {
        'n_total': n_total,
        'n_done': n_done,
        'n_skipped_existing': n_skipped_existing,
        'n_failed': n_failed,
        'failures': failures,
        'suggest_bad_eids': sorted(suggest_bad_eids),
        'out_dir': str(out_dir),
    }


def _attach_ephys_features(r):
    """
    Attach atlas-based ephys features to dict r, applying consistent
    cleaning; returns (dfm, r). Only used in non-CV mode.
    """
    print('attaching ephys features...')
    n_cells = len(r['uuids'])
    df = pd.DataFrame({
        'uuids':    r['uuids'],
        'pid':      r['pid'],
        'channels': r['channels'],
        'axial_um': r['axial_um'],
        'lateral_um': r['lateral_um'],
        # keep for debugging / alignment checks
        'concat_z_len': [row.shape[0] if hasattr(row, 'shape') else np.nan for row in r['concat_z']]
                          if isinstance(r['concat_z'], np.ndarray) else [np.nan]*n_cells
    })

    atlas_df = load_atlas_data().copy()
    # enforce many-to-one merge (each (pid,axial_um,lateral_um) maps to <=1 row)
    atlas_df = atlas_df.drop_duplicates(subset=['pid', 'axial_um', 'lateral_um'])
    dfm = df.merge(atlas_df, on=['pid', 'axial_um', 'lateral_um'],
                   how='left', validate='many_to_one')
    assert len(dfm) == len(df), "Merge changed row count; check atlas_df keys/duplicates."

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
           'trough_time_secs', 'trough_val']

    # build fixed-length vectors per row (concatenate available scalars)
    def _row_to_vec(row):
        vals = []
        any_ok = False
        for k in fts:
            v = row.get(k, np.nan)
            if pd.notna(v):
                any_ok = True
                vals.append(np.atleast_1d(v))
        return np.concatenate(vals) if any_ok else np.array([])

    print('building ephys feature vectors...')
    dfm['ephysTF'] = dfm.apply(_row_to_vec, axis=1)

    r['ephysTF'] = dfm['ephysTF'].to_numpy()
    r['fts'] = fts

    # remove cells with nan/inf/allzero/empty ephysTF
    good_nan  = np.array([not np.isnan(vec).any() for vec in r['ephysTF']], dtype=bool)
    good_inf  = np.array([not np.isinf(vec).any() for vec in r['ephysTF']], dtype=bool)
    good_len  = np.array([vec.size > 0 for vec in r['ephysTF']], dtype=bool)
    good_any  = np.array([np.any(vec) if vec.size else False for vec in r['ephysTF']], dtype=bool)
    goodcells = good_nan & good_inf & good_len & good_any

    print(f"[ephys] {goodcells.sum()} / {len(goodcells)} neurons keepable by ephys filters")

    l0 = len(r['uuids'])
    for key in list(r.keys()):
        arr = r[key]
        if isinstance(arr, np.ndarray) and len(arr) == l0:
            r[key] = arr[goodcells]
    return dfm, r


def stack_concat(
    vers: str = "concat",
    get_tls: bool = False,
    ephys: bool = False,
    concat_only: bool = False,
    cv: bool = False,
    min_trials: int = 10,
):
    """
    Stack concatenated PETHs from per-trial data on disk and optionally compute embeddings.

    - Non-CV:       average trials per segment -> concat time -> one matrix (neurons x time)
    - CV (half0/1): split trials per segment into two halves -> train & test matrices
    """
    start_time = time.time()

    # ---- paths ----
    pth = Path(one.cache_dir, "dmn", vers)
    pth.mkdir(parents=True, exist_ok=True)
    pth_res = Path(one.cache_dir, "dmn", "res")
    pth_res.mkdir(parents=True, exist_ok=True)

    # ---- discover per-insertion files "<eid>_<probe>.npy" ----
    ss_all = [fn for fn in os.listdir(pth) if fn.endswith(".npy")]
    ss = [fn for fn in ss_all if "_" in fn and not fn.startswith(f"{vers}_")]
    if not ss:
        raise RuntimeError(f"No per-insertion .npy files found in {pth}")
    print(f"combining {len(ss)} insertions for version {vers}")

    # ---- authoritative order from PETH_types_dict and sample file ----
    ttypes_atomic = list(PETH_types_dict[vers])
    D_sample = np.load(Path(pth, ss[0]), allow_pickle=True).flat[0]
    ttypes_0 = D_sample["trial_names"]
    assert ttypes_atomic == ttypes_0, "ttypes in sample file do not match PETH_types_dict."
    ttypes_eff = ttypes_atomic  # combine_mistake already handled upstream

    # ---- pid helper ----
    df = bwm_query(one)

    def pid__(eid, probe_name):
        return df[np.bitwise_and(df["eid"] == eid, df["probe_name"] == probe_name)]["pid"].values[0]

    # ---- I/O + containers ----
    def _load_D(p):
        return np.load(p, allow_pickle=True).flat[0]

    def _init_r_dict():
        return {k: [] for k in ["ids", "xyz", "uuids", "pid", "axial_um", "lateral_um", "channels"]}

    # ---- helpers ----
    def _avg_trials(ar: np.ndarray) -> np.ndarray:
        # (N,T,M) -> (N,T)
        if ar.ndim != 3:
            raise ValueError(f"_avg_trials expects (N,T,M), got {ar.shape}")
        M = ar.shape[2]
        if M == 0:
            return np.zeros(ar.shape[:2], dtype=np.float32)
        return np.mean(ar, axis=2).astype(np.float32)  # median results in too many zero

    def _extract_trials_3d(D, tname: str) -> np.ndarray:
        """
        Input on disk: D['ws'][idx] has shape (n_trials, n_neurons, n_timebins) = (M,N,T).
        Return: (N,T,M).
        """
        if "ws" not in D:
            raise KeyError("Expected per-trial data under 'ws' (shape: M,N,T).")

        try:
            idx = D["trial_names"].index(tname)
        except ValueError:
            raise KeyError(f"Trial name '{tname}' not found in D['trial_names']")

        X = D["ws"][idx]
        if X is None:
            return np.empty((len(D["ids"]), 0, 0), dtype=np.float32)
        if X.ndim != 3:
            raise ValueError(f"ws['{tname}'] must be 3D (M,N,T); got {X.shape}")

        M, N, T = X.shape
        if N != len(D["ids"]):
            raise ValueError(f"ws['{tname}'] neurons ({N}) != len(ids) ({len(D['ids'])})")

        # (M,N,T) -> (N,T,M)
        return np.transpose(X, (1, 2, 0)).astype(np.float32)

    def _compute_lens_eff_from_sample(D_sample, ttypes_eff):
        lens = []
        for t in ttypes_eff:
            Xt = _extract_trials_3d(D_sample, t)  # (N,T,M)
            lens.append(int(Xt.shape[1]))
        return lens

    def _half_means_concat(D, ttypes_eff):
        segs0, segs1 = [], []
        for t in ttypes_eff:
            X = _extract_trials_3d(D, t)  # (N,T,M)
            if X.size:
                _, mk = _count_real_trials_3d(X)
                if mk.size:
                    X = X[:, :, mk]

            M = X.shape[2]
            if M <= 1:
                idx0 = np.arange(M, dtype=int)
                idx1 = np.array([], dtype=int)
            else:
                k = (M + 1) // 2
                idx0 = np.arange(0, k, dtype=int)
                idx1 = np.arange(k, M, dtype=int)

            A0 = _avg_trials(X[:, :, idx0])
            A1 = _avg_trials(X[:, :, idx1])
            segs0.append(A0)
            segs1.append(A1)

        P0 = np.concatenate(segs0, axis=1) if segs0 else np.empty((0, 0))
        P1 = np.concatenate(segs1, axis=1) if segs1 else np.empty((0, 0))
        return P0, P1

    # =========================================================
    #                      NON-CV PATH
    # =========================================================
    if not cv:
        r = _init_r_dict()
        ws = []
        tlss = {}

        for s in ss:
            eid = s.split("_")[0]
            probe_name = s.split("_")[1].split(".")[0]
            pid = pid__(eid, probe_name)

            D_ = _load_D(Path(pth, s))
            D_["pid"] = [pid] * len(D_["ids"])
            tlss[s] = D_.get("tls", {})

            if get_tls:
                continue

            def _count_grouped_trials(tname: str) -> int:
                X = _extract_trials_3d(D_, tname)
                c, _ = _count_real_trials_3d(X)
                return int(c)

            counts = {t: _count_grouped_trials(t) for t in ttypes_eff}
            failing = {t: c for t, c in counts.items() if c < min_trials}
            if failing:
                detail = ", ".join(f"{t}={c}" for t, c in sorted(failing.items()))
                print(f"[skip] {eid}_{probe_name}: <{min_trials} real trials for types: {detail}")
                continue

            segs = []
            for t in ttypes_eff:
                X = _extract_trials_3d(D_, t)  # (N,T,M)
                if X.size:
                    _, mk = _count_real_trials_3d(X)
                    if mk.size:
                        X = X[:, :, mk]
                A = _avg_trials(X)
                segs.append(A)

            P = np.concatenate(segs, axis=1)
            ws.append(P)

            for ke in r.keys():
                r[ke].append(D_[ke])

        print(len(ws), "insertions combined")

        if get_tls:
            return tlss

        for ke in r.keys():
            r[ke] = np.concatenate(r[ke]) if len(r[ke]) else np.array([])
        cs = np.concatenate(ws, axis=0) if len(ws) else np.empty((0, 0))
        print(f"[non-CV] MERGED raw size: {cs.shape[0]} neurons, {cs.shape[1] if cs.size else 0} timebins")

        good = (~np.isnan(cs).any(axis=1)) & np.any(cs, axis=1)
        for ke in r.keys():
            r[ke] = r[ke][good]
        cs = cs[good]
        print(f"[non-CV] After cleaning: {cs.shape[0]} neurons kept")

        lens_eff = _compute_lens_eff_from_sample(D_sample, ttypes_eff)
        r["ttypes"] = list(ttypes_eff)
        r["len"] = dict(zip(ttypes_eff, lens_eff))

        if concat_only:
            r["concat"] = cs
            out = pth_res / f"{vers}_concat_only.npy"
            np.save(out, r, allow_pickle=True)
            print(f"saved concatenated-only data to {out}")
            print(f"Function 'stack_concat' executed in: {time.time() - start_time:.4f} s")
            return

        r["fr"] = np.array([np.mean(x) for x in cs], dtype=np.float32)

        # IMPORTANT: define concat_z before using it
        r["concat_z"] = zscore(cs, axis=1) if cs.size else cs

        data = np.asarray(r["concat_z"])
        rng = np.random.default_rng(0)

        if data.ndim != 2:
            raise ValueError("r['concat_z'] must be 2D (neurons x time).")

        N = data.shape[0]
        print(f"[info] Computing LZs for {N} neurons…")

        lz_vals = np.zeros(N, float)
        for i in range(N):
            lz_vals[i] = lzs_pci(data[i], rng)
        r["lz"] = lz_vals

        if ephys and len(r["uuids"]):
            print("loading and concatenating ephys features ...")
            _, r = _attach_ephys_features(r)
            print(f"{r['concat_z'].shape[0]} neurons retained after ephys cleaning")
            print("z-scoring ephys features")
            r["ephysTF"] = zscore(np.stack(r["ephysTF"], axis=0), axis=1)
            print("embedding Rastermap on ephys")
            model_e = Rastermap(
                n_PCs=200, n_clusters=100, locality=0.75, time_lag_window=5, bin_size=1
            ).fit(r["ephysTF"])
            r["isort_e"] = model_e.isort
            print("UMAP on ephys...")
            r["umap_e"] = umap.UMAP(n_components=2, random_state=0).fit_transform(r["ephysTF"])

        print(f"embedding Rastermap on {vers}...")
        try:
            model = Rastermap(
                n_PCs=200, n_clusters=100, locality=0.75, time_lag_window=5, bin_size=1
            ).fit(r["concat_z"])
            r["isort"] = model.isort
        except Exception as e:
            print("Rastermap failed:", e)

        print(f"embedding UMAP on {vers}...")
        r["umap_z"] = umap.UMAP(n_components=2, random_state=0).fit_transform(r["concat_z"])

        print(f"embedding PCA on {vers}...")
        pca = PCA(n_components=2)
        r["pca_z"] = pca.fit_transform(r["concat_z"])

        out = pth_res / f"{vers}_cvFalse_ephys{ephys}.npy"
        np.save(out, r, allow_pickle=True)
        print(f"saved combined data to {out}")
        print(f"Function 'stack_concat' executed in: {time.time() - start_time:.4f} s")
        return

    # =========================================================
    #                        CV PATH
    # =========================================================
    r = _init_r_dict()
    ws_train, ws_test = [], []
    tot0_raw = tot1_raw = tot_after = 0

    lens_eff = _compute_lens_eff_from_sample(D_sample, ttypes_eff)

    for fn in ss:
        eid = fn.split("_")[0]
        probe_name = fn.split("_")[1].split(".")[0]
        pid = pid__(eid, probe_name)
        D = _load_D(Path(pth, fn))

        def _count_grouped_trials_cv(tname: str) -> int:
            X = _extract_trials_3d(D, tname)
            c, _ = _count_real_trials_3d(X)
            return int(c)

        counts_cv = {t: _count_grouped_trials_cv(t) for t in ttypes_eff}
        failing_cv = {t: c for t, c in counts_cv.items() if c < min_trials}
        if failing_cv:
            detail = ", ".join(f"{t}={c}" for t, c in sorted(failing_cv.items()))
            print(f"[CV skip] {eid}_{probe_name}: <{min_trials} real trials for types: {detail}")
            continue

        try:
            P0, P1 = _half_means_concat(D, ttypes_eff)
        except Exception as ex:
            print(f"[CV] Skipping {eid}_{probe_name}: {type(ex).__name__}: {ex}")
            continue

        valid0 = (~np.isnan(P0).any(axis=1)) & np.any(P0, axis=1)
        valid1 = (~np.isnan(P1).any(axis=1)) & np.any(P1, axis=1)
        common_good = valid0 & valid1
        if not np.any(common_good):
            print(f"[CV] Skipping {eid}_{probe_name}: no valid neurons after joint mask")
            continue

        P0c = P0[common_good, :]
        P1c = P1[common_good, :]
        ws_train.append(P0c)
        ws_test.append(P1c)

        n_raw = len(D["ids"])
        for ke in r.keys():
            base = np.array([pid] * n_raw) if ke == "pid" else np.asarray(D[ke])
            r[ke].append(base[common_good])

        tot0_raw += n_raw
        tot1_raw += n_raw
        tot_after += P0c.shape[0]

    print(len(ws_train), "CV train insertions combined; ", len(ws_test), "CV test insertions combined")
    print(f"[CV] TOTALS (before cleaning): half0={tot0_raw}, half1={tot1_raw} neurons")
    print(f"[CV] TOTALS (after joint mask): kept={tot_after} neurons")

    for ke in r.keys():
        r[ke] = np.concatenate(r[ke]) if len(r[ke]) else np.array([])
    X_train = np.concatenate(ws_train, axis=0) if len(ws_train) else np.empty((0, 0))
    X_test = np.concatenate(ws_test, axis=0) if len(ws_test) else np.empty((0, 0))
    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        raise RuntimeError("Empty TRAIN or TEST matrix after joint cleaning; cannot run Rastermap CV.")

    print(f"[CV] MERGED sizes: TRAIN={X_train.shape[0]} neurons, TEST={X_test.shape[0]} neurons")

    Z_train = zscore(X_train, axis=1) if X_train.size else X_train
    r["concat_z_train"] = Z_train
    r["concat_z"] = zscore(X_test, axis=1) if X_test.size else X_test
    r["fr"] = np.array([np.mean(x) for x in X_test], dtype=np.float32) if X_test.size else np.array([], dtype=np.float32)

    data = np.asarray(r["concat_z"])
    rng = np.random.default_rng(0)
    if data.ndim != 2:
        raise ValueError("r['concat_z'] must be 2D (neurons x time).")

    N = data.shape[0]
    print(f"[info] Computing LZs for {N} neurons…")

    lz_vals = np.zeros(N, float)
    for i in range(N):
        lz_vals[i] = lzs_pci(data[i], rng)
    r["lz"] = lz_vals

    r["ttypes"] = list(ttypes_eff)
    r["len"] = dict(zip(ttypes_eff, lens_eff))

    print("[CV] fitting Rastermap on TRAIN (half0) and storing sorting for TEST (half1)...")
    model = Rastermap(n_PCs=200, n_clusters=100, locality=0.75, time_lag_window=5, bin_size=1).fit(Z_train)
    r["isort"] = model.isort

    print("embedding UMAP on TEST concat_z...")
    r["umap_z"] = umap.UMAP(n_components=2, random_state=0, n_neighbors=8, min_dist=0.2).fit_transform(
        r["concat_z"]
    )

    out = pth_res / f"{vers}_cvTrue_ephysFalse.npy"
    np.save(out, r, allow_pickle=True)
    print(f"saved combined data to {out}")
    print(f"Function 'stack_concat' executed in: {time.time() - start_time:.4f} s")




'''
#####################################################
### plotting
#####################################################
'''
        

def plot_dim_reduction(algo='umap_z', mapping='rm', ephys=False,
                       feat='concat_z', means=False, exa=False, shuf=False,
                       exa_squ=False, vers='concat', ax=None, ds=0.5,
                       axx=None, exa_kmeans=False, leg=False, restr=None,
                       nclus=7, rerun=False, cv=True, save_only=False):
    """
    2D embedding (e.g., UMAP) colored by mapping.
    Segment boundaries/labels are taken from r['len'] (ordered) and r['peth_dict'].
    """

    if save_only:
        plt.ioff()


    # --- load result dict with the exact data-derived order/labels ---
    r = regional_group(mapping, vers=vers, ephys=ephys,
                       nclus=nclus, rerun=rerun, cv=cv)

    if feat not in r:
        raise KeyError(f"Feature '{feat}' not found.")
    if algo not in r:
        raise KeyError(f"Embedding '{algo}' not found.")

    print(len(r['concat_z']), 'cells in', mapping, vers)

    # --- axes ---
    alone = False
    if ax is None:
        alone = True
        fig, ax = plt.subplots(label=f'{vers}_{mapping}')


    # --- optional color shuffle ---
    if shuf:
        shuffle(r['cols'])

    # --- scatter ---
    if restr:
        ff = np.bitwise_or.reduce([r['acs'] == reg for reg in restr])
        im = ax.scatter(r[algo][:, 0][ff], r[algo][:, 1][ff],
                        marker='o', c=r['cols'][ff], s=ds, rasterized=True)
    else:
        im = ax.scatter(r[algo][:, 0], r[algo][:, 1],
                        marker='o', c=r['cols'], s=ds, rasterized=True)

    # --- means overlay ---
    if means:
        regs = list(Counter(r['acs']))
        r['av'] = {reg: [np.mean(r[algo][r['acs'] == reg], axis=0), pal[reg]] for reg in regs}
        emb1 = [r['av'][reg][0][0] for reg in r['av']]
        emb2 = [r['av'][reg][0][1] for reg in r['av']]
        cs = [r['av'][reg][1] for reg in r['av']]
        ax.scatter(emb1, emb2, marker='o', facecolors='none',
                   edgecolors=cs, s=600, linewidths=4, rasterized=True)

    # --- cosmetics ---
    ax.set_title(f'z-score: {algo == "umap_z"}') if alone else None
    ax.axis('off')

    # --- legend/colorbar ---
    if mapping in ['layers', 'kmeans']:
        if leg and ('els' in r):
            ax.legend(handles=r['els'], ncols=1, frameon=False).set_draggable(True)
    elif 'clusters' in mapping:
        nclus_dyn = len(Counter(r['acs']))
        cax = fig.add_axes([0.27, 0.2, 0.5, 0.01])
        norm = mpl.colors.Normalize(vmin=0, vmax=nclus_dyn)
        cmap = mpl.cm.get_cmap('Spectral')
        fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                     cax=cax, orientation='horizontal')

    # --- save main figure ---
    if alone:
        fig.tight_layout()


                # --- SET ACTUAL WINDOW TITLE WITH nclus ---
        try:
            fig.canvas.manager.set_window_title(
                f"umap | {algo} | {mapping} | nclus={nclus} | cv={int(cv)} | {vers}"
            )
        except Exception:
            pass

        out = Path(one.cache_dir, 'dmn', 'imgs', f'{nclus}_{mapping}_umap_cv{cv}.png')
        fig.savefig(out, dpi=150)
        if save_only:
            plt.close(fig)
            

    # --- interactive example ---
    if exa:
        fig_extra, ax_extra = plt.subplots()
        line, = ax_extra.plot(r[feat][0], label='Extra Line Plot')

        def update_line(event):
            if event.mouseevent.inaxes == ax:
                x_clicked = event.mouseevent.xdata
                y_clicked = event.mouseevent.ydata
                selected_point = None
                for key, emb in zip(r['nums'], r[algo]):
                    if (abs(emb[0] - x_clicked) < 0.01 and
                        abs(emb[1] - y_clicked) < 0.01):
                        selected_point = key
                        break
                if selected_point is not None:
                    line.set_data(T_BIN * np.arange(len(r[feat][selected_point])),
                                  r[feat][selected_point])
                    ax_extra.relim(); ax_extra.autoscale_view()
                    ax_extra.set_ylabel(feat); ax_extra.set_xlabel('time [sec]')
                    ax_extra.set_title(
                        f'Line Plot at ({np.round(x_clicked,2)}, {np.round(y_clicked,2)})')
                    fig_extra.canvas.draw()



        fig.canvas.mpl_connect('pick_event', update_line)
        im.set_picker(5)

    # --- cluster mean PETHs (uses r['len'] + r['peth_dict']) ---
    if exa_kmeans:
        plot_cluster_mean_PETHs(r, mapping, feat, vers=vers, axx=axx, alone=True)
        ff = plt.gcf()

                        # --- SET ACTUAL WINDOW TITLE WITH nclus ---
        try:
            ff.canvas.manager.set_window_title(
                f"Avg | {algo} | {mapping} | nclus={nclus} | cv={int(cv)} | {vers}"
            )
        except Exception:
            pass

        out2 = Path(one.cache_dir, 'dmn', 'imgs',
                    f'{nclus}_{mapping}_lines_cv{cv}.png')
        ff.savefig(out2, dpi=150)
        if save_only:
            plt.close(ff)

    # --- square ROIs and mean/individual PETHs with correct segment order ---
    if exa_squ:
        ns = 10; ssq = 0.01
        x_min = np.floor(np.min(r[algo][:, 0])); x_max = np.ceil(np.max(r[algo][:, 0]))
        y_min = np.floor(np.min(r[algo][:, 1])); y_max = np.ceil(np.max(r[algo][:, 1]))
        side_length = ssq * (x_max - x_min)

        sqs = [(random.uniform(x_min, x_max - side_length),
                random.uniform(y_min, y_max - side_length),
                side_length) for _ in range(ns)]

        for sq_x, sq_y, L in sqs:
            pts = [ke for ke, emb in zip(r['nums'], r[algo])
                   if (sq_x <= emb[0] <= sq_x + L) and (sq_y <= emb[1] <= sq_y + L)]
            if not pts:
                continue

            rect = plt.Rectangle((sq_x, sq_y), L, L, fill=False, color='r', linewidth=2)
            ax.add_patch(rect)

            fg, axp = plt.subplots()
            maxys = []
            for pt in pts:
                axp.plot(T_BIN * np.arange(len(r[feat][pt])), r[feat][pt],
                         color=r['cols'][pt], linewidth=0.5)
                maxys.append(np.max(r[feat][pt]))
            axp.plot(T_BIN * np.arange(len(r[feat][pts][0])),
                     np.mean(r[feat][pts], axis=0), color='k', linewidth=2)

            axp.set_title(f'ROI {len(pts)} points'); axp.set_xlabel('time [sec]'); axp.set_ylabel(feat)

            if 'len' not in r or not isinstance(r['len'], dict) or len(r['len']) == 0:
                raise KeyError("r['len'] missing or empty; cannot draw boundaries.")
            ordered_segments = list(r['len'].keys())
            labels = r.get('peth_dict', {k: k for k in ordered_segments})

            h = 0
            ymax = 0.8 * (np.max(maxys) if len(maxys) else 1.0)
            for seg in ordered_segments:
                seg_len = r['len'][seg]
                xv = T_BIN * (h + seg_len)
                axp.axvline(xv, linestyle='--', color='grey', linewidth=0.8)
                axp.text(T_BIN * (h + seg_len / 2.0), ymax,
                         labels.get(seg, seg), rotation=90,
                         fontsize=10, color='k', ha='center')
                h += seg_len
    plt.show()
    #plt.close('all')



def plot_cluster_mean_PETHs(r, mapping, feat, vers='concat',
                            axx=None, alone=True): # MODIFIED
    """
    Plot mean PETH per cluster using the segment order/labels from the data file.
    """

    if feat not in r:
        raise KeyError(f"Feature '{feat}' not in result dict.")
    if 'acs' not in r or 'cols' not in r:
        raise KeyError("Result dict must contain 'acs' and 'cols'.")
    if 'len' not in r or not isinstance(r['len'], dict) or len(r['len']) == 0:
        raise KeyError("r['len'] (segment lengths) missing or empty.")
    if 'peth_dict' not in r:
        r['peth_dict'] = {k: k for k in r['len'].keys()}

    clu_vals = np.array(sorted(np.unique(r['acs'])))
    n_clu = len(clu_vals)
    if n_clu > 50:
        print('too many (>50) line plots!')
        return

    if axx is None:
        fg, axx = plt.subplots(nrows=n_clu, sharex=True, sharey=False, figsize=(6, 10))
    if not isinstance(axx, (list, np.ndarray)):
        axx = [axx]
    if len(axx) != n_clu:
        raise ValueError(f"Expected {n_clu} axes, got {len(axx)}.")

    n_bins = r[feat].shape[1]
    xx = np.arange(n_bins) / c_sec

    ordered_segments = list(r['len'].keys())
    seg_lengths = [r['len'][s] for s in ordered_segments]
    if sum(seg_lengths) != n_bins:
        print(f"[warn] sum(r['len'])={sum(seg_lengths)} != n_bins={n_bins}")

    for k, clu in enumerate(clu_vals):
        # Determine the base indices for the current cluster
        idx = np.where(r['acs'] == clu)[0] 

        if idx.size == 0:
            axx[k].axis('off')
            continue

        yy = np.mean(r[feat][idx, :], axis=0)
        col = r['cols'][idx[0]]
        axx[k].plot(xx, yy, color=col, linewidth=2)

        if k != (n_clu - 1):
            axx[k].axis('off')
        else:
            axx[k].spines['top'].set_visible(False)
            axx[k].spines['right'].set_visible(False)
            axx[k].spines['left'].set_visible(False)
            axx[k].tick_params(left=False, labelleft=False)

        # segment boundaries + labels from file order
        h = 0
        ymax = float(np.max(yy)) if yy.size else 0.0
        for s in ordered_segments:
            seg_len = r['len'][s]
            xv_bins = h + seg_len

            if xv_bins > n_bins:
                break   

            axx[k].axvline(xv_bins / c_sec, linestyle='--', linewidth=1, color='grey')
            if k == 0:
                seg_mid = h + seg_len / 2.0
                axx[k].text(seg_mid / c_sec, ymax,
                            '   ' + r['peth_dict'].get(s, s),
                            rotation=90, color='k', fontsize=10, ha='center')
            h += seg_len

        axx[k].set_xlim(0, n_bins / c_sec)

    axx[-1].set_xlabel('time [sec]')

    if alone:
        plt.tight_layout()



def smooth_dist(dim=2, algo='umap_z', mapping='Beryl', 
                show_imgs=False, restr=False, global_norm=True,
                norm_=True, dendro=False, nmin=50, vers='concat',
                combine_mistake=False):
    """
    Generalized smoothing/analysis of embeddings.
    Uses r['len'] and r['peth_dict'] for segment boundaries/labels in feature plots.
    """

    assert 2 <= dim <= 5, "dim must be between 2 and 5."
    feat = 'concat_z' if algo[-1] == 'z' else 'concat'

    # pull the same data-derived order/labels
    r = regional_group(mapping, vers=vers, combine_mistake=combine_mistake)

    if algo == 'xyz':
        r[algo] = r[algo] * 100000

    meshsize = 256 if dim == 2 else 64
    fontsize = 12

    # grid limits
    mins = [np.floor(np.min(r[algo][:, i])) for i in range(dim)]
    maxs = [np.ceil(np.max(r[algo][:, i])) for i in range(dim)]
    for i in range(dim):
        maxs[i] = maxs[i] * 0.01 + maxs[i]
        mins[i] = mins[i] * 0.01 + mins[i]

    if algo == 'xyz':
        dim = 3
        feat = 'xyz'
        global_scaled = [ (r[algo][:, i] - mins[i]) / (maxs[i] - mins[i]) for i in range(dim) ]
        global_inds = np.clip((np.array(global_scaled).T * meshsize), 0, meshsize - 1).astype('uint')
        global_img = np.zeros([meshsize] * dim)
        for pt in global_inds:
            global_img[tuple(pt)] += 1
        global_smoothed = ndi.gaussian_filter(global_img.T, [5] * dim)
        if norm_:
            global_smoothed /= np.max(global_smoothed)

    imgs, coords = {}, {}
    regs00 = Counter(r['acs'])
    regcol = {reg: np.array(r['cols'])[np.array(r['acs']) == reg][0] for reg in regs00}

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
    for drop in ('root', 'void'):
        if drop in regs:
            regs.remove(drop)

    for reg in regs:
        scaled = [ (r[algo][np.array(r['acs']) == reg, i] - mins[i]) / (maxs[i] - mins[i]) for i in range(dim) ]
        coords[reg] = scaled
        data = np.array(scaled).T
        inds = np.clip(data * meshsize, 0, meshsize - 1).astype('uint')
        img = np.zeros([meshsize] * dim)
        for pt in inds:
            img[tuple(pt)] += 1
        imsm = ndi.gaussian_filter(img.T, [5] * dim)
        if (algo == 'xyz') and global_norm:
            with np.errstate(divide='ignore', invalid='ignore'):
                imsm = np.divide(imsm, global_smoothed)
                imsm[~np.isfinite(imsm)] = 0
        imgs[reg] = imsm / np.max(imsm) if norm_ else imsm

    if show_imgs and dim <= 3:
        fig, axs = plt.subplots(nrows=3, ncols=len(regs), figsize=(18.6, 8))
        if dim == 2:
            for i in range(1, len(regs)):
                axs[0, i].sharex(axs[0, 0]); axs[0, i].sharey(axs[0, 0])
        axs = axs.flatten(); k = 0

        # row 1: scatter
        for reg in regs:
            ax = axs[k]
            if dim == 2:
                ax.scatter(coords[reg][0], coords[reg][1], color=regcol[reg], s=0.1, rasterized=True)
                ax.spines['right'].set_visible(False); ax.spines['top'].set_visible(False)
                ax.set_xlim([0, 1]); ax.set_ylim([0, 1]); ax.set_aspect('equal')
            elif dim == 3:
                ax.axis('off')
                ax = fig.add_subplot(3, len(regs), k + 1, projection='3d')
                ax.scatter(coords[reg][0], coords[reg][1], coords[reg][2], s=0.1, c=regcol[reg], rasterized=True)
                ax.set_xlim([0, 1]); ax.set_ylim([0, 1]); ax.set_zlim([0, 1])
            ax.set_title(f"{reg}, ({regs00[reg]})"); k += 1
        fig.text(0.02, 0.8, f'{algo} \n embedded activity', fontsize=14, rotation='vertical', va='center', ha='center')

        # row 2: smoothed density
        for reg in regs:
            img = imgs[reg]; ax = axs[k]
            if dim == 2:
                ax.imshow(img, origin='lower', cmap='viridis', rasterized=True)
            elif dim == 3:
                ax.imshow(np.max(img, axis=0), origin='lower', cmap='viridis', rasterized=True)
            ax.set_aspect('equal'); ax.axis('off'); k += 1
        fig.text(0.02, 0.5, 'Smoothed 2d \n projected Density', fontsize=14, rotation='vertical', va='center', ha='center')

        # row 3: feature vectors (respect file order/labels)
        k3 = k
        ordered_segments = list(r['len'].keys())
        labels = r.get('peth_dict', {k: k for k in ordered_segments})

        for reg in regs:
            pts = np.arange(len(r['acs']))[np.array(r['acs']) == reg]
            xss = T_BIN * np.arange(r[feat].shape[1])
            yss = np.mean(r[feat][pts], axis=0)
            yss_err = np.std(r[feat][pts], axis=0) / np.sqrt(len(pts))
            maxys = [yss + yss_err]
            ax = axs[k]
            ax.fill_between(xss, yss - yss_err, yss + yss_err, alpha=0.2, color=regcol[reg])
            ax.plot(xss, yss, color='k', linewidth=0.5)
            ax.axis('off')

            h = 0
            for seg in ordered_segments:
                seg_len = r['len'][seg]
                xv = T_BIN * (h + seg_len)
                ax.axvline(xv, linestyle='--', color='grey', linewidth=0.1)
                ax.text(T_BIN * (h + seg_len / 2.0), 0.8 * np.max(maxys),
                        labels.get(seg, seg), rotation=90, fontsize=5, color='k', ha='center')
                h += seg_len
            k += 1

        for axx in axs[k3:]:
            axx.sharex(axs[k3]); axx.sharey(axs[k3])

        fig.text(0.02, 0.25, 'Avg. Feature \n Vectors', fontsize=14, rotation='vertical', va='center', ha='center')
        fig.tight_layout()
        fig.subplots_adjust(top=0.981, bottom=0.019, left=0.04, right=0.992, hspace=0.023, wspace=0.092)

    # similarity matrix
    regs_aug = list(regs); imgs_aug = dict(imgs)
    if algo == 'xyz':
        v_all = global_smoothed.flatten()
        imgs_aug['ALL'] = global_smoothed
        regs_aug.append('ALL')

    res = np.zeros((len(regs_aug), len(regs_aug)))
    for i, reg_i in enumerate(regs_aug):
        for j, reg_j in enumerate(regs_aug):
            v0 = imgs_aug[reg_i].flatten(); v1 = imgs_aug[reg_j].flatten()
            res[i, j] = cosine_sim(v0, v1)

    fig0, ax0 = plt.subplots(figsize=(4, 4))
    if dendro:
        dist = np.max(res) - res
        np.fill_diagonal(dist, 0)
        cres = squareform(dist)
        linkage_matrix = hierarchy.linkage(cres)
        ordered_indices = hierarchy.leaves_list(linkage_matrix)
        regs_aug = [regs_aug[i] for i in ordered_indices]
        res = res[:, ordered_indices][ordered_indices, :]

    ax0.set_title(f'{algo}, {mapping}, {dim} dims')
    ims = ax0.imshow(res, origin='lower', interpolation=None, vmin=0, vmax=1)
    ax0.set_xticks(np.arange(len(regs_aug))); ax0.set_yticks(np.arange(len(regs_aug)))
    ax0.set_xticklabels(regs_aug, rotation=90, fontsize=fontsize)
    ax0.set_yticklabels(regs_aug, fontsize=fontsize)
    for i, reg in enumerate(regs_aug):
        col = regcol[reg] if reg in regcol else 'black'
        ax0.xaxis.get_ticklabels()[i].set_color(col)
        ax0.yaxis.get_ticklabels()[i].set_color(col)
    cb = plt.colorbar(ims, fraction=0.046, pad=0.04)
    cb.set_label('regional similarity')
    fig0.tight_layout()

    # save
    fig_basepath = Path(one.cache_dir, 'dmn', 'figs'); fig_basepath.mkdir(parents=True, exist_ok=True)
    base_name = f"{algo}_{mapping}_{dim}D_globalnorm{global_norm}"
    if 'fig' in locals():
        fig.savefig(fig_basepath / f"{base_name}_all_panels.svg", format='svg', dpi=300, bbox_inches='tight')
    if 'fig0' in locals():
        fig0.savefig(fig_basepath / f"{base_name}_similarity_matrix.svg", format='svg', dpi=300, bbox_inches='tight')

    return res, regs_aug


def _build_event_stats(rerun=False):
    pth_dmnm = Path(Path(one.cache_dir, 'dmn'), 'mean_event_diffs.npy')
    if (not pth_dmnm.is_file()) or rerun:

        pids = np.unique(np.asarray(r['pid']))

        eids_from_r = []

        diffs = []
        for pid in pids:
            try:
                eid = one.pid2eid(pid)
                if eid in eids_from_r:
                    continue
                eids_from_r.append(eid)

                trials, mask = load_trials_and_mask(one, eid, revision='2024-07-10')
                trials = trials[mask][:-100]
                diffs.append(np.mean(np.diff(trials[list(evs.keys())]), axis=0))
            except Exception:
                continue

        d = {}
        d['mean'] = np.nanmean(diffs, axis=0)
        d['std'] = np.nanstd(diffs, axis=0)
        d['diffs'] = diffs
        d['av_tr_times'] = [np.cumsum([0] + list(x)) for x in d['diffs']]
        d['av_times'] = dict(zip(list(evs.keys()),
                                    zip(np.cumsum([0] + list(d['mean'])),
                                        np.cumsum([0] + list(d['std'])))))
        np.save(pth_dmnm, d, allow_pickle=True)
    return np.load(pth_dmnm, allow_pickle=True).flat[0]


def plot_ave_PETHs(feat='concat_z', vers='concat',
                   rerun=False, anno=True, separate_cols=False, cv=True):
    """
    Average PETHs across cells and plot.
    Critically: segment order and labels come from the data file (r['len'], r['peth_dict']).
    """

    # mean timing between task events (unchanged)
    evs = {'stimOn_times': 'gray',
           'firstMovement_times': 'cyan',
           'feedback_times': 'orange'}

    # === Load combined matrix first (so we can get pids/eids from r) ===
    r = regional_group('rm',cv=cv)

    # now build stats using only eids present in r
    d = _build_event_stats()

    if feat not in r:
        raise KeyError(f"Feature '{feat}' not found in combined data (try 'concat_z').")

    r['mean'] = np.mean(r[feat], axis=0)

    # get alignement event per PETH from example session
    tts = concat_PETHs('8d661567-49f3-4547-997d-a345c0ffe2dd',get_tts=True)
    align_dict = {k: [tts[k][0],tts[k][-1]] for k in tts.keys()}
    # anchors for events relative to stimOn (stimOn = 0)
    stim_anchor = 0.0
    move_anchor = float(d['av_times']['firstMovement_times'][0])
    fb_anchor   = float(d['av_times']['feedback_times'][0])

    # plotting
    fig, ax = plt.subplots(figsize=(7, 2.75))

    # segment order & labels
    ordered_segments = list(r['len'].keys())
    labels = r.get('peth_dict', {k: k for k in ordered_segments})

    # color handling
    import itertools
    if separate_cols:
        cmap = get_cmap('tab10')
        colors = itertools.cycle(cmap.colors)
    else:
        colors = itertools.cycle(['k'])

    yys = []
    st = 0  # index into the concatenated feature dimension

    for seg in ordered_segments:
        seg_len = int(r['len'][seg])

        # mean activity for this segment
        yy = r['mean'][st: st + seg_len]
        st += seg_len

        # skip if we don't know how this segment is aligned
        if seg not in align_dict:
            continue

        align_event, window = align_dict[seg]
        window = np.asarray(window, dtype=float).ravel()
        if window.size != 2:
            # malformed window spec
            continue

        pre_mag, offset_signed = float(window[0]), float(window[1])

        # window length in seconds, as you specified
        window_len_sec = pre_mag + offset_signed

        # time axis RELATIVE TO THE ALIGNMENT EVENT:
        # infer start/end relative to event from pre_mag and offset_signed
        start_rel_event = -pre_mag
        end_rel_event   = offset_signed

        # build segment time axis in seconds (seg_len bins)
        # (this ensures that the total span equals window_len_sec)
        t_rel = np.linspace(start_rel_event, end_rel_event, seg_len)

        # anchor in global time (stimOn = 0)
        if align_event == 'stimOn_times':
            t_anchor = stim_anchor
        elif align_event == 'firstMovement_times':
            t_anchor = move_anchor
        elif align_event == 'feedback_times':
            t_anchor = fb_anchor
        else:
            # nothing to do for unknown anchor
            continue

        # global time axis for this segment
        xx = t_rel + t_anchor

        # (defensive) shape consistency
        if xx.shape[0] != yy.shape[0]:
            L = min(xx.shape[0], yy.shape[0])
            xx = xx[:L]
            yy = yy[:L]

        color = next(colors)
        yys.append(np.nanmax(yy))
        ax.plot(xx, yy, label=seg, color=color)
        if anno:
            ax.annotate(
                labels.get(seg, seg),
                (xx[-1], yy[-1]),
                color=color,
                fontsize=8,
            )

    # event lines at their mean times
    ymax = max(yys) if len(yys) > 0 else 1.0
    for ev, col in evs.items():
        if ev not in d['av_times']:
            continue
        t_ev = float(d['av_times'][ev][0])
        ax.axvline(x=t_ev, label=ev, color=col, linestyle='-')
        if anno:
            ax.annotate(
                ev,
                (t_ev, 0.8 * ymax),
                color=col,
                rotation=90,
                textcoords='offset points',
                xytext=(-15, 0),
                fontsize=8,
            )

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('time [sec]')
    ax.set_ylabel('trial-averaged z')
    fig.canvas.manager.set_window_title('PETHs averaged across all BWM cells')
    fig.tight_layout()



def plot_xyz(mapping='Beryl', vers='concat', add_cents=False,
             restr=False, ax=None, axoff=True, exa=True,
             combine_mistake=False, ephys=False, nclus=7, cv=True):
    """
    3D scatter of cell features with optional example traces.
    Segment boundaries/labels (in example traces) come from r['len'] and r['peth_dict'].
    """

    r = regional_group(mapping, vers=vers, ephys=ephys, nclus=nclus,
                       cv=cv, combine_mistake=combine_mistake)

    # If mapping corresponds to a single PETH (or group), color by ranking if available
    if ((mapping in tts__) or (mapping in PETH_types_dict)) and ('rankings' in r):
        cmap = mpl.cm.get_cmap('Spectral')
        norm = mpl.colors.Normalize(vmin=min(r['rankings']), vmax=max(r['rankings']))
        r['cols'] = cmap(norm(r['rankings']))
    else:
        cmap = norm = None  # for colorbar logic below

    xyz = r['xyz'] * 1000  # convert to mm

    created_fig = False
    if ax is None:
        created_fig = True
        fig = plt.figure(figsize=(8.43, 7.26), label=mapping)
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.get_figure()

    if isinstance(restr, list) and len(restr):
        idcs = np.bitwise_or.reduce([r['acs'] == reg for reg in restr])
        xyz = xyz[idcs]
        r['cols'] = np.asarray(r['cols'])[idcs]
        r['acs']  = np.asarray(r['acs'])[idcs]

    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
               depthshade=False, marker='o',
               s=1 if created_fig else 0.5, c=r['cols'])

    if add_cents:
        if mapping != 'Beryl':
            print('add cents only for Beryl')
        else:
            regs = list(Counter(r['acs']))
            centsd = get_centroids()
            cents = np.array([centsd[x] for x in regs])
            volsd = get_volume(); vols = [volsd[x] for x in regs]
            scale = 5000; vols = scale * np.array(vols) / np.max(vols)
            cols = [pal[reg] for reg in regs]
            ax.scatter(cents[:, 0], cents[:, 1], cents[:, 2],
                       marker='*', s=vols, color=cols, depthshade=False)

    scalef = 1.2
    ax.view_init(elev=45.78, azim=-33.4)
    ax.set_xlim(min(xyz[:, 0]) / scalef, max(xyz[:, 0]) / scalef)
    ax.set_ylim(min(xyz[:, 1]) / scalef, max(xyz[:, 1]) / scalef)
    ax.set_zlim(min(xyz[:, 2]) / scalef, max(xyz[:, 2]) / scalef)
    ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False

    fontsize = 14
    ax.set_xlabel('x [mm]', fontsize=fontsize)
    ax.set_ylabel('y [mm]', fontsize=fontsize)
    ax.set_zlabel('z [mm]', fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(False)
    nbins = 3
    ax.xaxis.set_major_locator(MaxNLocator(nbins=nbins))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=nbins))

    if axoff:
        ax.axis('off')

    # Safe colorbar if we actually built a colormap/norm
    if (((mapping in tts__) or (mapping in PETH_types_dict)) and
        (cmap is not None) and (norm is not None) and ('rankings' in r)):
        mappable = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        mappable.set_array(r['rankings'])
        cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10)
        cbar.set_label(f'mean {mapping} rankings')

    if created_fig:
        ax.set_title(f'{mapping}')

    if exa:
        # Only supported for time-series mappings
        if (mapping not in tts__) and (mapping not in PETH_types_dict):
            print('not implemented for other mappings')
            return

        feat = 'concat_z'
        nrows = 10
        if 'rankings' not in r or len(r['rankings']) == 0:
            print('example traces require r["rankings"]')
            return

        rankings_s = sorted(r['rankings'])
        indices = [list(r['rankings']).index(x) for x in
                   np.concatenate([rankings_s[:nrows // 2],
                                   rankings_s[-nrows // 2:]])]

        fg, axx = plt.subplots(nrows=nrows, sharex=True, sharey=False, figsize=(7, 7))
        xx = np.arange(len(r[feat][0])) / c_sec

        # order and labels from file
        ordered_segments = list(r['len'].keys())
        labels = r.get('peth_dict', {k: k for k in ordered_segments})

        for kk, ind in enumerate(indices):
            yy = r[feat][ind]
            axx[kk].plot(xx, yy, color=r['cols'][ind], linewidth=2)
            sss = (r['acs'][ind] + '\n' + str(r['pid'][ind][:3]))
            axx[kk].set_ylabel(sss)

            if kk != (len(indices) - 1):
                axx[kk].spines['top'].set_visible(False)
                axx[kk].spines['right'].set_visible(False)
                axx[kk].spines['bottom'].set_visible(False)
            else:
                axx[kk].spines['top'].set_visible(False)
                axx[kk].spines['right'].set_visible(False)

            # segment boundaries/labels from data file order
            h = 0
            for seg in ordered_segments:
                seg_len = r['len'][seg]
                xv = (h + seg_len) / c_sec
                axx[kk].axvline(xv, linestyle='--', linewidth=1, color='grey')

                # highlight the plotted mapping if it's a composite listing that contains seg
                ccc = 'r' if seg == mapping else 'k'
                if mapping in PETH_types_dict and seg in PETH_types_dict[mapping]:
                    ccc = 'r'

                if kk == 0:
                    axx[kk].text((h + seg_len / 2) / c_sec, np.max(yy),
                                 '   ' + labels.get(seg, seg),
                                 rotation=90, color=ccc, fontsize=10, ha='center')
                h += seg_len

        axx[-1].set_xlabel('time [sec]')
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
                        reg='MOp', ephys=False, nclus=20, cv=True):
    """
    For a single cell, plot its feature vector with PETH labels.
    Segment order and labels are taken from r['len'] and r['peth_dict'].
    """

    feat = 'concat_z' if algo.endswith('z') else 'concat'
    r = regional_group(mapping, vers=vers, ephys=ephys, nclus=nclus,
                       cv=cv)

    # guard against empty region
    idxs = np.where(np.asarray(r['acs']) == reg)[0]
    if idxs.size == 0:
        raise ValueError(f"No cells found for region '{reg}' in mapping '{mapping}'.")

    fig, ax = plt.subplots(figsize=(6, 3.01))

    xx = np.arange(r[feat].shape[1]) / c_sec  # seconds
    samp = random.choice(list(idxs))
    print(reg, samp)
    yy = r[feat][samp]

    ax.plot(xx, yy, color=r['cols'][samp], linewidth=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # order and labels from file (not global dicts)
    ordered_segments = list(r['len'].keys())
    labels = r.get('peth_dict', {k: k for k in ordered_segments})

    # vertical boundaries + segment labels
    h = 0
    for seg in ordered_segments:
        seg_len = r['len'][seg]
        xv_bins = h + seg_len
        ax.axvline(xv_bins / c_sec, linestyle='--', linewidth=1, color='grey')
        ax.text((h + seg_len / 2) / c_sec, np.max(yy),
                '   ' + labels.get(seg, seg),
                rotation=90, color='k', fontsize=10, ha='center')
        h += seg_len

    ax.set_ylabel('z-scored firing rate')
    ax.set_xlabel('time [sec]')
    fig.tight_layout()

     

def _draw_peth_boundaries(ax, r, vers, yy_max, c_sec):
    """Add vertical window boundaries and labels, matching plot_single_feature."""
   

    d2 = {sec: r['len'][sec] for sec in r['ttypes']}
    h = 0
    for sec in d2:
        xv = d2[sec] + h
        ax.axvline(xv / c_sec, linestyle='--', linewidth=1, color='grey')
        ax.text(xv / c_sec - d2[sec] / (2 * c_sec), yy_max,
                '   ' + r['peth_dict'][sec], rotation=90, color='k',
                fontsize=10, ha='center', va='bottom')
        h += d2[sec]


def plot_example_neurons(
        n: int,
        vers: str = 'concat',
        mapping: str = 'rm',
        seed: Optional[int] = None,
        max_categories: int = 101,
        offset_scale: float = 4.0,
        linewidth: float = 1.3,
        savefig: bool = True,
        save_formats: tuple = ('png',),
        dpi: int = 200,
        show: bool = True,
        annotate: bool = True,
        label_key: str = 'Beryl',
        label_fontsize: int = 8,
        label_pad_frac: float = 0.01,
        nclus: int = 7,
        cv: bool = False,
        # new: choose one cluster or all
        sing_clus: Union[bool, int] = False,
        # --- NEW ARGUMENTS ---
        no_filts=False,
        min_max_fr: Optional[Tuple[float, float]] = (0.1, 100),
        min_max_lz: Optional[Tuple[float, float]] = (0.0, 0.6),
):
    """
    Plot example neurons per cluster (mapping categories).

    If sing_clus is an int, only that cluster ID is plotted.
    If sing_clus is False (default), all clusters are plotted.
    """



    if no_filts:
        min_max_fr = None
        min_max_lz = None

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    feat = 'concat_z'
    r = regional_group(mapping, vers=vers, cv=cv, nclus=nclus)
    if feat not in r:
        raise KeyError(f"Feature '{feat}' not in r. Available: {list(r.keys())}")

    if 'fr' not in r:
        raise KeyError("'fr' (firing rate) not found in r.")
    if 'lz' not in r:
        raise KeyError("'lz' (Lempel–Ziv complexity) not found in r.")

    acs_vals = np.asarray(r['acs'])
    cats = np.unique(acs_vals)
    if cats.size >= max_categories:
        print(f"[info] Found {cats.size} 'acs' categories (>= {max_categories}). Skipping.")
        return

    # restrict to single cluster, if requested
    if sing_clus is not False:
        sing_clus_int = int(sing_clus)
        if sing_clus_int not in cats:
            print(f"[info] sing_clus={sing_clus_int} not present in categories {cats}. Nothing to plot.")
            return
        cats = np.array([sing_clus_int])

    else:
        show = False  # only show when single cluster

    # saving
    if savefig:
        save_dir = Path(one.cache_dir, 'dmn', 'figs')
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"[info] Figures will be saved to {save_dir}")

    xx = np.arange(r[feat].shape[1]) / c_sec  # seconds

    for cat in cats:
        # -----------------------------
        # filtering by FR and LZ
        # -----------------------------
        idx_all = np.where(acs_vals == cat)[0]

        if min_max_fr is not None:
            lo, hi = min_max_fr
            idx_all = idx_all[(r['fr'][idx_all] >= lo) & (r['fr'][idx_all] <= hi)]

        if min_max_lz is not None:
            lo, hi = min_max_lz
            idx_all = idx_all[(r['lz'][idx_all] >= lo) & (r['lz'][idx_all] <= hi)]

        if idx_all.size == 0:
            continue

        k = min(n, idx_all.size)
        if idx_all.size >= k:
            samp = np.random.choice(idx_all, size=k, replace=False)
        else:
            samp = np.array(random.choices(idx_all, k=k))

        fig, ax = plt.subplots(figsize=(7.6, 10), constrained_layout=True)
        try:
            stds = [np.nanstd(r[feat][i]) for i in samp]
            base_off = 2.0 * (np.nanmedian(stds) if len(stds) else 1.0)
            off = base_off * offset_scale

            y_max_seen = -np.inf
            for j, i in enumerate(samp):
                yi = r[feat][i]
                yy = yi + j * off
                lbl = str(r[label_key][i])
                color = pal[lbl]

                ax.plot(xx, yy, linewidth=linewidth, color='black', alpha=0.9)
                y_max_seen = max(y_max_seen, np.nanmax(yy))

                if annotate:
                    x_offset = xx[0] - 0.02 * (xx[-1] - xx[0])
                    prefix = max(1, int(0.02 * yi.size))
                    y0 = np.nanmedian(yi[:prefix]) + j * off

                    ax.text(x_offset, y0, lbl,
                            fontsize=label_fontsize,
                            va='center', ha='right',
                            color=color,
                            alpha=0.95,
                            clip_on=False)

                    fr_val = float(r['fr'][i])
                    lz_val = float(r['lz'][i])
                    info_txt = f"{fr_val:.2f}, {lz_val:.2f}"
                    y1 = y0 - 0.3 * off
                    ax.text(x_offset, y1, info_txt,
                            fontsize=label_fontsize - 1,
                            va='center', ha='right',
                            color=color,
                            alpha=0.9,
                            clip_on=False)

            _draw_peth_boundaries(ax, r, vers, y_max_seen, c_sec)

            ax.set_xlabel("time [s]")
            ax.set_ylabel("z-scored firing rate (stacked)")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.yaxis.set_ticks([])

            ax.set_xlim(-0.5, xx[-1])

            # counts
            N0 = np.sum(acs_vals == cat)
            N1 = len(idx_all)
            if min_max_fr is None:
                N_fr = N0
            else:
                lo, hi = min_max_fr
                fr_mask = (r['fr'][acs_vals == cat] >= lo) & (r['fr'][acs_vals == cat] <= hi)
                N_fr = int(np.sum(fr_mask))

            title = f"{mapping} = {cat} of {nclus}, (n={k} of {N1} neurons)"
            if min_max_fr is not None:
                lo, hi = min_max_fr
                title += f"\nFR ∈ [{lo:.2f}, {hi:.2f}]   ({N0} → {N_fr})"
            if min_max_lz is not None:
                lo, hi = min_max_lz
                title += f"\nLZ ∈ [{lo:.2f}, {hi:.2f}]   ({N_fr} → {N1})"

            fig.suptitle(title, fontsize=12, weight='bold')
            fig.subplots_adjust(left=0.12, top=0.90)

            if savefig:
                for fmt in save_formats:
                    fn = save_dir / f"{mapping}_{cat}_of{nclus}_n{n}_cv{int(cv)}.{fmt}"
                    fig.savefig(fn, dpi=dpi, bbox_inches='tight')
                print(f"  saved: {mapping}_{cat}_of{nclus}_n{n}_cv{cv}.{save_formats[0]}")

            if show:
                plt.show(block=False)

        finally:
            gc.collect()

    
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
  

    
def clus_freqs(foc='rm', nmin=50, nclus=20, vers='concat', get_res=False,
               rerun=False, norm_=True, save_=True, single_regions=[],
               axs = None):

    '''
    foc: rm, Berly, dec
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
    r_k = regional_group('rm', vers=vers, nclus=nclus)

    if foc == 'rm':

        # --- cluster ids as integers (avoid '0' vs 0 mismatches) ---
        clus_ids = np.asarray(r_k['acs'])
        try:
            clus_ids = clus_ids.astype(int)
        except Exception:
            # robust fallback: map strings like '0' -> 0, else keep as-is then cast
            clus_ids = np.array([int(x) for x in clus_ids])

        cluss = np.unique(clus_ids)  # sorted unique cluster ids


        fig, axs = plt.subplots(nrows = len(cluss), ncols = 1,
                               figsize=(18.79,  15),
                               sharex=True, sharey=True if norm_ else False)
        
        fig.canvas.manager.set_window_title(
            f'Frequency of Beryl region label per'
            f' rm cluster ({nclus}); vers ={vers}')    



        # Build region -> color mapping from r_a once (first occurrence wins)
        cols_dict = {}
        for reg, col in zip(r_a['acs'], r_a['cols']):
            if reg not in cols_dict:
                cols_dict[reg] = col

        # order regions by canonical list
        p = (Path(iblatlas.__file__).parent / 'beryl.npy')
        regs_can = br.id2acronym(np.load(p), mapping='Beryl')
        regs_ = Counter(r_a['acs'])
        reg_ord = [reg for reg in regs_can if reg in regs_]

        d = {}
        for k, clus in enumerate(cluss):
            mask = (clus_ids == clus)
            # counts for this cluster across Beryl regions
            counts = Counter(np.asarray(r_a['acs'])[mask])

            reg_order = {reg: counts.get(reg, 0) for reg in reg_ord}
            labels = list(reg_order.keys())
            values = np.array(list(reg_order.values()), float)

            den = values.sum()
            if norm_ and den > 0:
                values = values / den

            x = np.arange(len(labels))
            colors = [cols_dict[lab] for lab in labels]
            bars = axs[k].bar(x, values, color=colors)
            axs[k].set_ylabel(f'clus {int(clus)}')
            axs[k].set_xticks(x)
            axs[k].set_xticklabels(labels, rotation=90, fontsize=6)

            for ticklabel, bar in zip(axs[k].get_xticklabels(), bars):
                ticklabel.set_color(bar.get_facecolor())

            axs[k].set_xlim(-0.5, len(labels) - 0.5)
            axs[k].spines['top'].set_visible(False)
            axs[k].spines['right'].set_visible(False)

            d[int(clus)] = values.tolist()
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
                                    
        clus_ids = np.array(r_k['acs'])
        clus_cols = np.array(r_k['cols'])
        clus_unique = np.unique(clus_ids)
        cols_dict = {int(cid): clus_cols[clus_ids == cid][0] for cid in clus_unique}
                    
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
            colors = ([cols_dict[int(label)] for label in labels] 
                if not single_regions else ['k'] * len(labels))

            d[reg] = values          

            bars = axs[k].bar(labels, values, color=colors)
            axs[k].text(
                0.98, 0.98, reg,
                transform=axs[k].transAxes,
                ha='right', va='top',
                fontsize=label_size * 2.5,
                color=pal[reg]
            )

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
                f'Frequency of rm cluster ({nclus}) per'
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
            # --- place region label like in foc='Beryl' ---
            axs[k].text(
                0.98, 0.98, reg,
                transform=axs[k].transAxes,
                ha='right', va='top',
                fontsize=label_size * 2.5,
                color=pal[reg] if reg in pal else 'k'
            )

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
        fig.tight_layout() 
        fig.savefig(Path(pth_dmn.parent, 'imgs',
                        f'{foc}_{nclus}_{vers}_nrm_{norm_}.svg'), dpi=150)
        fig.savefig(Path(pth_dmn.parent, 'imgs',
                        f'{foc}_{nclus}_{vers}_nrm_{norm_}.pdf'), dpi=150)
        np.save(pthres, d, allow_pickle=True)
    if get_res:
        return d


def plot_rm_cluster_profile(clus,
                            vers='concat',
                            nclus=20,
                            grid_upsample=0,
                            cv=True,
                            cv2=False,
                            ephys=False,
                            axs=None,
                            norm_reg_count: bool = False,
                            savefig: bool = True):
    """
    For a given Rastermap cluster (clus), plot:

    Top panel:
        - Average PETH-style feature vector (e.g. concat_z) for neurons in this
          rm cluster, with PETH boundaries drawn via _draw_peth_boundaries,
          using r['peth_dict'] as defined in regional_group.

    Bottom panel:
        - Bar plot of Beryl regions represented in this cluster.

          If norm_reg_count=False (default):
              bar height = # cells from that region in this cluster,
              sorted by descending count.

          If norm_reg_count=True:
              bar height = (# cells from that region in this cluster) /
                           (# cells from that region in the whole population),
              sorted by descending normalized value (enrichment).
    """
    # ---- load rm clusters and Beryl labels via regional_group ----
    r_rm = regional_group(
        mapping='rm',
        vers=vers,
        ephys=ephys,
        grid_upsample=grid_upsample,
        nclus=nclus,
        rerun=False,
        cv=cv,
        cv2=cv2,
    )
    r_B = regional_group(
        mapping='Beryl',
        vers=vers,
        ephys=ephys,
        grid_upsample=grid_upsample,
        nclus=nclus,
        rerun=False,
        cv=cv,
        cv2=cv2,
    )

    # ---- cluster mask (robust to str/int) ----
    clus_ids = np.asarray(r_rm['acs'])
    try:
        clus_ids = clus_ids.astype(int)
    except Exception:
        clus_ids = np.array([int(x) for x in clus_ids])

    clus = int(clus)
    mask = (clus_ids == clus)
    n_in_clus = int(mask.sum())
    if n_in_clus == 0:
        raise ValueError(f"No neurons found in rm cluster {clus}.")

    # ---- feature matrix and average PETH-like trace ----
    feat_key = 'concat_z' if 'concat_z' in r_rm else 'concat'
    feat_mat = np.asarray(r_rm[feat_key])  # shape (N_cells, T)
    if feat_mat.ndim != 2:
        raise ValueError(f"{feat_key} has shape {feat_mat.shape}, expected 2D (cells × time).")

    feat_cluster = feat_mat[mask, :]              # (n_in_clus, T)
    feat_mean = feat_cluster.mean(axis=0)         # (T,)

    # PETH-style time axis, consistent with plot_example_neurons
    xx = np.arange(feat_mat.shape[1]) / c_sec

    # ---- region distribution (Beryl) within this cluster ----
    acs_B = np.asarray(r_B['acs'])    # global Beryl labels (all cells)
    cols_B = np.asarray(r_B['cols'])

    regs_in_clus = acs_B[mask]        # Beryl labels only for this cluster
    # global base counts per region (all cells)
    global_counts = Counter(acs_B)

    # per-cluster counts
    counts = Counter(regs_in_clus)

    # consistent color mapping
    reg2col = {}
    for reg, col in zip(acs_B, cols_B):
        if reg not in reg2col:
            reg2col[reg] = col

    # --- compute raw and (optionally) normalized values per region ---
    regs = list(counts.keys())
    raw_vals = np.array([counts[r] for r in regs], dtype=float)

    if norm_reg_count:
        # normalize each region count by its global abundance
        vals = []
        for rname, v in zip(regs, raw_vals):
            base = global_counts.get(rname, 0)
            if base > 0:
                vals.append(v / float(base))
            else:
                vals.append(0.0)
        vals = np.array(vals, dtype=float)
    else:
        vals = raw_vals

    # --- sort by the values we actually plot (normalized or raw) ---
    sort_idx = sorted(
        range(len(regs)),
        key=lambda i: (-vals[i], str(regs[i]))
    )

    reg_sorted  = [regs[i] for i in sort_idx]
    vals_sorted = vals[sort_idx]
    cols_sorted = [reg2col[r] for r in reg_sorted]

    # ---- prepare axes ----
    alone = axs is None
    if alone:
        fig, (ax_top, ax_bottom) = plt.subplots(
            nrows=2, ncols=1, figsize=(20, 6),
            sharex=False
        )
    else:
        ax_top, ax_bottom = axs
        fig = ax_top.figure

    # ---- TOP PANEL: average PETH-like trace with PETH boundaries ----
    ax_top.plot(xx, feat_mean, linewidth=1.2, color='black', alpha=0.9)

    y_max_seen = float(np.nanmax(feat_mean))
    pad = 0.1 * (np.nanmax(feat_mean) - np.nanmin(feat_mean) + 1e-6)
    ax_top.set_ylim(np.nanmin(feat_mean) - pad, y_max_seen + pad)

    _draw_peth_boundaries(ax_top, r_rm, vers, y_max_seen, c_sec)

    ax_top.set_xlabel("time [s]")
    ax_top.set_ylabel("z-scored firing rate (mean over cluster)")
    ax_top.spines['top'].set_visible(False)
    ax_top.spines['right'].set_visible(False)
    ax_top.spines['left'].set_visible(False)
    ax_top.yaxis.set_ticks([])

    # ---- BOTTOM PANEL: region histogram ----
    x = np.arange(len(reg_sorted))
    bars = ax_bottom.bar(x, vals_sorted, color=cols_sorted)

    ax_bottom.set_xticks(x)
    ax_bottom.set_xticklabels(reg_sorted, rotation=90, fontsize=7)

    if norm_reg_count:
        ax_bottom.set_ylabel('fraction of region in cluster')
        title_norm = " (normalized by global region counts)"
    else:
        ax_bottom.set_ylabel('# cells in cluster')
        title_norm = ""

    ax_bottom.set_xlabel('Beryl region')
    ax_bottom.spines['top'].set_visible(False)
    ax_bottom.spines['right'].set_visible(False)
    ax_bottom.set_xlim(-0.5, len(reg_sorted) - 0.5)

    for ticklabel, bar in zip(ax_bottom.get_xticklabels(), bars):
        ticklabel.set_color(bar.get_facecolor())

    ax_bottom.set_title(
        f'RM cluster {clus} of {nclus} (n={n_in_clus} neurons)\n'
        f'vers={vers}, grid={grid_upsample}{title_norm}'
    )

    if alone:
        fig.tight_layout()
        fig.canvas.manager.set_window_title(
            f'RM cluster {clus}: mean PETH + Beryl composition'
        )

    if savefig:
        save_dir = Path(one.cache_dir, 'dmn', 'figs')
        save_dir.mkdir(parents=True, exist_ok=True)

        fstem = (
            f"rmclus_{clus}_of{nclus}"
            f"_vers{vers}"
            f"_grid{grid_upsample}"
            f"_cv{int(cv)}_cv2{int(cv2)}"
            f"_ephys{int(ephys)}"
            f"_normReg{int(norm_reg_count)}"
        )

        png_path = save_dir / f"{fstem}.png"
        fig.savefig(png_path, dpi=150, bbox_inches='tight')
        print(f"[saved] {png_path}")
        plt.close(fig)




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
    compare average feature vector for two groups of cells
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


def plot_rastermap(vers='concat', feat='concat_z', regex='ECT', 
                   exa = False, mapping='rm', bg=False, img_only=False,
                   interp='antialiased', single_reg=False, cv=True,
                   bg_bright = 0.99, vmax=2, rerun=False, sort_method='rastermap',nclus=100, clsfig=False, bounds=True, grid_upsample=0, locality=0.75, time_lag_window=5, symmetric=False):
    """
    Function to plot a rastermap with vertical segment boundaries 
    and labels positioned above the segments.

    Extra panel with colors of mapping.

    feat = 'single_reg' will show only cells in example region regex

    Most numerous regions for each Cosmos are:
    ['CP', 'MRN', 'PO', 'CA1', 'MOp', 'CUL4 5', 'IRN', 'PIR', 'ZI', 'BMA']
    ...
    sort_method : str, default 'rastermap'
        Sorting method for rows. One of:
        - 'rastermap' (default): use r['isort']
        - 'umap': sort by first UMAP dimension
        - 'pca': sort by first PCA dimension
    """
    r = regional_group(mapping, vers=vers, ephys=False, nclus=nclus, 
                       rerun=rerun, cv=cv, grid_upsample=grid_upsample,
                       locality=locality, time_lag_window=time_lag_window,
                       symmetric=symmetric)

    if exa:
        plot_cluster_mean_PETHs(r,mapping, feat)


    if clsfig:
        plt.ioff()

    spks = r[feat]
    # --- choose sorting algorithm ---
    if sort_method == 'rastermap':
        isort = r['isort' if feat != 'ephysTF' else 'isort_e']
    elif sort_method == 'umap':
        assert 'umap_z' in r, "r['umap_z'] not found in results."
        isort = np.argsort(r['umap_z'][:, 0])
    elif sort_method == 'pca':
        assert 'pca_z' in r, "r['pca_z'] not found in results."
        isort = np.argsort(r['pca_z'][:, 0])
    else:
        raise ValueError(f"Unknown sort_method: {sort_method}")

    data = spks[isort]
    row_colors = np.array(r['cols'])[isort]

    del spks
    gc.collect()        


    if single_reg:

        acs = np.array(r['Beryl'])[isort]

        # adjust background line width according to example region size
        n = len(acs)
        n_ex = sum(acs == regex)
        print('number of cells in example region:', n_ex)

        data = data[acs == regex]
        row_colors = row_colors[acs == regex]
        print(f'filtering rastermap for {regex} cells only')

        del acs
        gc.collect()        

    n_rows, n_cols = data.shape

    fig, ax = plt.subplots(figsize=(6, 8))

    vmin, vmax = 0, vmax
    data_clipped = np.clip(data, vmin, vmax)
    gray_scaled = (data_clipped - vmin) / (vmax - vmin)  # Normalized to 0-1

    # Manual alpha blending if bg is enabled
    if bg:
        # 1. Create color-coded background as full RGBA image
        rgba_bg = np.array([to_rgba(c) for c in row_colors], dtype=np.float32)
        rgba_bg = np.broadcast_to(rgba_bg[:, np.newaxis, :], (*data.shape, 4)).copy()

        rgba_bg[..., :3] = rgba_bg[..., :3] * bg_bright + (1 - bg_bright) 
        
        alpha_overlay = gray_scaled 

        # 3. Composite black (0,0,0) with alpha over the colored background
        for c in range(3):  # R, G, B channels
            rgba_bg[..., c] *= (1 - alpha_overlay)

        rgba_bg[..., 3] = 1  # Final alpha channel (opaque result)

        # 4. Display the blended image
        ax.imshow(rgba_bg, aspect="auto", interpolation=interp, zorder=1)

        del rgba_bg
        gc.collect()

    else:
        # No background: show signal as brightness (white = strong activity)
        rgba_overlay = np.zeros((*gray_scaled.shape, 4), dtype=np.float32)
        inv_gray = 1.0 - gray_scaled  # invert: high activity = bright

        # replicate across RGB channels
        rgba_overlay[..., :3] = inv_gray[..., np.newaxis]
        rgba_overlay[..., 3] = 1.0  # fully opaque

        ax.imshow(rgba_overlay, aspect="auto", interpolation=interp, zorder=2)

        del rgba_overlay
        gc.collect()

    if grid_upsample > 0:
        bounds = False  # disable boundaries for upsampled grids

    if mapping == 'rm' and bounds:
        # row_colors already reordered by isort (and possibly filtered by single_reg)
        rc = np.asarray(row_colors)

        # cluster labels after sorting (and same masking as data/row_colors)
        clus_sorted = np.asarray(r['acs'])[isort]
        if single_reg:
            # if you filtered by Beryl region, apply same mask to clusters
            acs_B = np.asarray(r['Beryl'])[isort]
            mask_reg = (acs_B == regex)
            clus_sorted = clus_sorted[mask_reg]

        if rc.ndim == 2 and rc.shape[0] > 1:
            # find indices where the color (cluster) changes between consecutive rows
            color_changes = np.any(np.diff(rc, axis=0) != 0, axis=1)
            boundaries = np.where(color_changes)[0] + 0.5  # between rows i and i+1

            # draw horizontal lines at cluster boundaries
            for y in boundaries:
                ax.axhline(
                    y,
                    color='k',
                    linewidth=0.6,
                    zorder=5
                )

            # --- NEW: add cluster numbers on the right, x-zoom independent ---
            # x in axes coordinates (0–1), y in data coordinates
            trans_right = mpl.transforms.blended_transform_factory(
                ax.transAxes,   # x in axes coords
                ax.transData    # y in data coords
            )

            n_rows = data.shape[0]
            # define edges of each cluster segment
            # start at row 0.5, then all boundaries, then last row + 0.5
            edges = np.concatenate(([0.5], boundaries, [n_rows - 0.5]))

            for i in range(len(edges) - 1):
                y0, y1 = edges[i], edges[i + 1]
                mid_y = 0.5 * (y0 + y1)

                # pick a representative row index for this segment
                row_idx = int(np.clip(np.floor(mid_y), 0, n_rows - 1))
                cid = int(clus_sorted[row_idx])

                ax.text(
                    1.01,          # just to the right of the axes
                    mid_y,
                    str(cid),
                    transform=trans_right,
                    va='center',
                    ha='left',
                    fontsize=8,
                    color='k',
                    clip_on=False
                )

    if feat != 'ephysTF':
        if 'len' not in r or not isinstance(r['len'], dict) or len(r['len']) == 0:
            raise KeyError("Segment lengths r['len'] missing or empty; cannot draw boundaries/labels.")

        ordered_segments = list(r['len'].keys())
        labels = r.get('peth_dict', {})

        if data.shape[1] != sum(r['len'].values()):
            print(f"[warn] data.shape[1] ({data.shape[1]}) != sum(len) ({sum(r['len'].values())})")

        # --- NEW: blended transform: x in data coords, y in axes coords ---
        # This keeps labels at a fixed height (e.g. 1.02 above top) regardless of y-zoom.
        trans_top = mpl.transforms.blended_transform_factory(
            ax.transData,      # x in data coordinates
            ax.transAxes       # y in axes coordinates (0–1)
        )

        h = 0
        for seg in ordered_segments:
            seg_len = r['len'][seg]
            xv = h + seg_len

            if xv > n_cols:
                break

            ax.axvline(xv, linestyle='--', linewidth=1, color='grey')

            midpoint = h + seg_len / 2.0
            if not img_only:
                ax.text(
                    midpoint,
                    1.02,  # just above top of axes; change to 1.01/1.05 if you prefer
                    labels.get(seg, seg),
                    rotation=90,
                    color='k',
                    fontsize=10,
                    ha='center',
                    va='bottom',
                    transform=trans_top,  # <-- crucial line
                    clip_on=False
                )
            h += seg_len

        x_ticks = np.arange(0, n_cols, c_sec)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f"{int(tick / c_sec)}" for tick in x_ticks])


    else:
        ax.set_xticks(range(data.shape[1]))
        ax.set_xticklabels(r['fts'], rotation=90)       

    ax.set_xlabel('time [sec]')    
    ax.set_ylabel(f'cells in {regex}' if feat == 'single_reg' else 'cells')

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if img_only:
        ax.axis('off')
        # print region name and number of neurons on top like title
        if single_reg:
            ax.text(0.5, 1.05, f'{regex} ({n_ex})', ha='center', va='bottom',
                    fontsize=12, color=pal[regex], transform=ax.transAxes)

    ax.set_xlim(0, n_cols)
    plt.tight_layout()  # Adjust the layout to prevent clipping
    # plt.show()

    # --- build descriptive filename and window title ---
    descriptor = (
        f"cv{int(cv)}"
        f"_nclus{nclus}"
        f"_locality{locality}"
        f"_timelag{time_lag_window}"
        f"_upsample{grid_upsample}"
        f"_symmetric{symmetric}")

    fname = "rastermap_" + descriptor.replace(" | ", "_").replace("=", "") + ".png"

    # Set figure window title (useful when many figures open)
    try:
        fig.canvas.manager.set_window_title(f"Rastermap: {descriptor}")
    except Exception:
        pass  # ignored in non-interactive backends

    out_path = pth_dmn.parent / "imgs" / fname
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')

    if clsfig:
        # close figure
        plt.close(fig)

    for v in ("sig","img_array","isort","r"):
        if v in locals(): 
            del locals()[v]



def raster_grid():


    grid_upsample_grid   = [0, 10]
    locality_grid        = [0.1, 0.5, 0.9]
    ncluss = [20,100]
    time_lag_window_grid = [0, 5, 15]
    symmetric_grid       = [True, False]
    cv_grid              = [True, False]

    for cv in cv_grid:
        for nclus in ncluss:
            for grid_upsample in grid_upsample_grid:
                for symmetric in symmetric_grid:
                    for locality in locality_grid:
                        for time_lag_window in time_lag_window_grid:
                            print(
                                f"[rm grid] cv={cv} grid={grid_upsample} sym={symmetric} "
                                f"loc={locality} tlag={time_lag_window}"
                            )
                            plot_rastermap(
                                cv=cv,
                                grid_upsample=grid_upsample,
                                symmetric=symmetric,
                                time_lag_window=time_lag_window,
                                locality=locality,clsfig=True,nclus=nclus
                            )
                            save_rastermap_pdf(
                                cv=cv,
                                grid_upsample=grid_upsample,
                                symmetric=symmetric,
                                locality=locality,
                                time_lag_window=time_lag_window,nclus=nclus
                            )





def plot_region_counts_svg():
    '''
    for rasterplot figure, get single reg rasterplot titles as svg
    '''
    ex_regs = ['CP', 'MRN', 'PO', 'CA1', 'MOp', 
               'CUL4 5', 'IRN', 'PIR', 'ZI', 'BMA']

    r = regional_group('Beryl')
    counts = Counter(r['acs'])
    fig, ax = plt.subplots()
    ax.axis('off')

    for i, (region) in enumerate(ex_regs):
        color = pal.get(region, 'black')
        ax.text(0.5, 1 - i * 0.1, f'{region} ({counts[region]})',
                ha='center', va='top', fontsize=8, color=color,
                transform=ax.transAxes)

    filename='beryl_region_counts.svg'
    fig.savefig(filename, format='svg', bbox_inches='tight')


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


def plot_histograms():
    '''
    Plot two histograms of the flatness scores across regions:
    one for decoding, one for clustering, as line outlines.
    '''

    n_bins = 20

    be = flatness_entropy_score(clus_freqs(foc='Beryl', get_res=True))
    print(f"# regions in clustering (be): {len(be)}")
    de = flatness_entropy_score(clus_freqs(foc='dec', get_res=True))
    print(f"# regions in decoding   (de): {len(de)}")

    regions = sorted(set(be.keys()) & set(de.keys()))
    print(f"# regions in intersection: {len(regions)}")

    be_values = [be[reg] for reg in regions]
    de_values = [de[reg] for reg in regions]

    # Shared bins
    all_values = be_values + de_values
    bins = np.histogram_bin_edges(all_values, bins=20)

    # Histogram counts
    be_counts, _ = np.histogram(be_values, bins=bins)
    de_counts, _ = np.histogram(de_values, bins=bins)

    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    fig, ax = plt.subplots(figsize=(3, 2.5))

    ax.step(bin_centers, be_counts, where='mid', label='Clustering', color='blue', linewidth=2)
    ax.step(bin_centers, de_counts, where='mid', label='Decoding', color='red', linewidth=2)

    ax.set_xlabel('Specialization', fontsize=10)
    ax.set_ylabel('# Regions', fontsize=10)

    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))

    plt.tight_layout()
    fig.savefig(Path(pth_dmn.parent, 'imgs', 'overleaf_pdf',
        'flatness_histograms.svg'), format='svg', bbox_inches='tight')
    plt.show()


def plot_three_swansons():
    '''
    For the cluster count figure, plot three Swansons:
    1. Cluster counts per region (flatness entropy from clustering)
    2. Specialization scores from decoding
    3. Cosmos colors
    '''

    # Load flatness scores independently
    be = flatness_entropy_score(clus_freqs(foc='Beryl', get_res=True))
    de = flatness_entropy_score(clus_freqs(foc='dec', get_res=True))

    acronyms_be = np.array(list(be.keys()))
    acronyms_de = np.array(list(de.keys()))

    log_be = np.log([be[k] for k in acronyms_be])
    log_de = np.log([de[k] for k in acronyms_de])

    fig, axs = plt.subplots(ncols=3, figsize=(4.5, 3))

    # Define colormaps and value ranges separately
    cmap = plt.get_cmap('magma')
    vmin_be, vmax_be = log_be.min(), log_be.max()
    vmin_de, vmax_de = log_de.min(), log_de.max()

    # Panel 1: Clustering specialization
    plot_swanson_vector(
        acronyms=acronyms_be,
        values=log_be,
        ax=axs[0],
        br=br,
        orientation='portrait',
        cmap=cmap,
        vmin=vmin_be,
        vmax=vmax_be,
        show_cbar=False,
    )
    axs[0].axis('off')

    # Add colorbar and grey "no data" box to first panel
    cax1 = fig.add_axes([0.05, 0.72, 0.015, 0.2])
    cb1 = plt.colorbar(ScalarMappable(norm=Normalize(vmin_be, vmax_be), cmap=cmap), cax=cax1)
    cb1.ax.tick_params(labelsize=5)
    cb1.set_label('log(specialization(clu))', fontsize=6, labelpad=2)
    grey_patch = mpatches.Patch(color='lightgrey', label='no data')
    axs[0].legend(handles=[grey_patch], loc='upper left', fontsize=5, frameon=False, handlelength=1, handleheight=0.8)

    # Panel 2: Decoding specialization
    plot_swanson_vector(
        acronyms=acronyms_de,
        values=log_de,
        ax=axs[1],
        br=br,
        orientation='portrait',
        cmap=cmap,
        vmin=vmin_de,
        vmax=vmax_de,
        show_cbar=False,
    )
    axs[1].axis('off')

    # Add colorbar to second panel
    cax2 = fig.add_axes([0.38, 0.72, 0.015, 0.2])
    cb2 = plt.colorbar(ScalarMappable(norm=Normalize(vmin_de, vmax_de), cmap=cmap), cax=cax2)
    cb2.ax.tick_params(labelsize=5)
    cb2.set_label('log(specialization(dec))', fontsize=6, labelpad=2)

    # Panel 3: Color overlay
    plot_swanson_vector(ax=axs[2], orientation='portrait')
    axs[2].axis('off')

    plt.tight_layout()
    fig.savefig(Path(pth_dmn.parent, 'imgs', 'overleaf_pdf', 'swanson_three_flatness.svg'),
                format='svg', bbox_inches='tight')



def scat_dec_clus(norm_=True, harris=False, nclus=10, corr_only=False,
                  log_scale=True, axs=None, compare='clu'):
    '''
    Scatter plots comparing specialization scores from clustering and decoding,
    and optionally Harris hierarchy scores.

    Parameters
    ----------
    norm_ : bool
        Placeholder; not used in current implementation.
    harris : bool
        If True, compare specialization scores to Harris hierarchy.
    log_scale : bool
        If True, use logarithmic axes.
    axs : matplotlib.axes.Axes
        Optional axis to plot on.
    compare : {'clu', 'dec'}
        In Harris mode, whether to compare clustering or decoding specialization to the hierarchy.
    '''

    # Load specialization scores
    be = flatness_entropy_score(clus_freqs(foc='Beryl', get_res=True,       
        nclus=nclus))
    de = flatness_entropy_score(clus_freqs(foc='dec', get_res=True))

    if axs is None:
        fig, axs = plt.subplots(figsize=(3, 3))

    def scatter_panel(ax, x, y, xlabel, ylabel, labels, colors):
        # Correlation
        corr, pval = pearsonr(x, y)
        print(f"Pearson r = {corr:.2f}, p = {pval:.4g}")

        # Linear fit
        slope, intercept, r_value, _, _ = linregress(x, y)
        xx = np.linspace(min(x), max(x), 100)
        yy = slope * xx + intercept
        ax.plot(xx, yy, '--', color='black', lw=1,
                label=f'$R^2$ = {r_value**2:.2f}')

        for i in range(len(x)):
            ax.scatter(x[i], y[i], color=colors[i], s=10)

        if log_scale:
            ax.set_xscale('log')
            ax.set_yscale('log')

        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(fontsize=8)

        if not log_scale:
            ax.xaxis.set_major_locator(MaxNLocator(3))
            ax.yaxis.set_major_locator(MaxNLocator(3))

        ax.text(0.98, 0.02, f'{len(x)} regions',
                transform=ax.transAxes, ha='right', va='bottom', color='k')

    if harris:
        harris_hierarchy_scores = {region: idx for idx, region in enumerate(harris_hierarchy)}

        if compare == 'clu':
            common = set(be) & set(harris_hierarchy_scores)
            x_vals = [be[r] for r in common]
            y_vals = [harris_hierarchy_scores[r] for r in common]
            labels = list(common)
            colors = [pal[r] for r in labels]
            xlabel = 'log(Specialization (clu))' if log_scale else 'Specialization (clu)'
            ylabel = 'Harris hierarchy score'
            save_name = 'scat_clu_vs_harris.svg'

        elif compare == 'dec':
            common = set(de) & set(harris_hierarchy_scores)
            x_vals = [de[r] for r in common]
            y_vals = [harris_hierarchy_scores[r] for r in common]
            labels = list(common)
            colors = [pal[r] for r in labels]
            xlabel = 'log(Specialization (dec))' if log_scale else 'Specialization (dec)'
            ylabel = 'Harris hierarchy score'
            save_name = 'scat_dec_vs_harris.svg'

        else:
            raise ValueError("compare must be 'clu' or 'dec'")

        if corr_only:
            # Just compute and print correlation
            plt.gcf().clear()
            corr, pval = pearsonr(x_vals, y_vals)
            print(f'{nclus} clusters; compare {compare} to Harris')
            print(f"Pearson r = {corr:.2f}, p = {pval:.4g}")
            return corr, pval

        scatter_panel(axs, x_vals, y_vals, xlabel, ylabel, labels, colors)

    else:
        common = set(be) & set(de)
        x_vals = [be[r] for r in common]
        y_vals = [de[r] for r in common]


        if corr_only:
            # Just compute and print correlation
            corr, pval = pearsonr(x_vals, y_vals)
            print(f'{nclus} clusters; compare to dec')
            print(f"Pearson r = {corr:.2f}, p = {pval:.4g}")
            return corr, pval


        labels = list(common)
        colors = [pal[r] for r in labels]
        xlabel = 'log(Specialization (clu))' if log_scale else 'Specialization (clu)'
        ylabel = (
            'log(Specialization (dec))' if log_scale else
            'Specialization (dec)'
        )
        save_name = 'scat_clu_vs_dec.svg'

        scatter_panel(axs, x_vals, y_vals, xlabel, ylabel, labels, colors)

    plt.tight_layout()
    plt.savefig(Path(pth_dmn.parent, 'imgs', 'overleaf_pdf', save_name),
                format='svg', bbox_inches='tight')
    plt.show()


def plot_cluster_pearson(ax=None, r_squared=False):
    """
    Plot Pearson's r as a function of k-means cluster number.

    Parameters
    ----------
    results : list
        List like:
        [
          [cluster_id, (np.float64(r_value), np.float64(p_value))],
          ...
        ]
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot into. If None, a new figure/axes is created.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the plot.
    """

    results = []
    for i in range(3,18):
        results.append([i,scat_dec_clus(nclus=i,corr_only=True)]) 

    # Extract cluster IDs and r-values
    clusters = []
    r_values = []
    for cluster_id, (r_val, p_val) in results:
        clusters.append(int(cluster_id))
        r_values.append(float(r_val) if r_squared == False else float(r_val**2))

    clusters = np.array(clusters)
    r_values = np.array(r_values)

    # Sort by cluster number, just in case input is unsorted
    order = np.argsort(clusters)
    clusters = clusters[order]
    r_values = r_values[order]

    # Prepare axis
    if ax is None:
        fig, ax = plt.subplots()

    # Plot
    ax.plot(clusters, r_values, marker="o")
    ax.axhline(0, linestyle="--", linewidth=1)  # zero reference

    ax.set_xlabel("k-means cluster")
    ax.set_ylabel("Pearson's r with dec specialisation score" if not r_squared else "R squared with dec specialisation score")
    ax.set_title("Cluster-wise correlation")


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
    """
    Plots decoding accuracies for normal vs. shuffled splits across different mappings and sources.
    Adds SEM error bars and connects means with lines (solid if p <= 0.05, dotted if p > 0.05).
    """
    # Load results
    re = np.load(Path(pth_dmn, 'decode.npy'), allow_pickle=True).flat[0]
    
    # Extract unique mappings and sources
    sources, mappings = [], []
    for key in re:
        src, mapp = key.split(' ')
        sources.append(src)
        mappings.append(mapp)
    sources = np.unique(sources)
    mappings = np.unique(mappings)
    palette = sns.color_palette("tab10", len(mappings))
    mapping_color = {m:c for m,c in zip(mappings, palette)}   

    # Build DataFrame of all accuracies
    records = []
    for key, (normal_results, shuffled_results) in re.items():
        src, mapp = key.split(' ')
        # Normal splits
        for split in normal_results:
            for acc in split[:, 1]:
                records.append({'Mapping': mapp, 'Source': src, 'Type': 'Test Normal', 'Accuracy': acc})
        # Shuffled splits
        for split in shuffled_results:
            for acc in split[:, 1]:
                records.append({'Mapping': mapp, 'Source': src, 'Type': 'Test Shuffled', 'Accuracy': acc})
    df = pd.DataFrame(records)
    
    # Compute statistics: mean, sem, and p-value for each Mapping+Source
    stats = []
    grouped = df.groupby(['Mapping', 'Source', 'Type'])['Accuracy']
    agg = grouped.agg(['mean', 'std', 'count']).reset_index()
    agg['sem'] = agg['std'] / np.sqrt(agg['count'])
    
    # Compute p-values per Mapping+Source
    pvals = {}
    for (mapp, src), subdf in df.groupby(['Mapping', 'Source']):
        normal_acc = subdf[subdf['Type']=='Test Normal']['Accuracy']
        shuffled_acc = subdf[subdf['Type']=='Test Shuffled']['Accuracy']
        t_stat, p_val = ttest_ind(normal_acc, shuffled_acc, equal_var=False)
        pvals[(mapp, src)] = p_val
        print(f"  Mapping: {mapp:<10} Source: {src:<8} p-value = {p_val:.2e}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    mapping_colors = sns.color_palette("tab10", len(mappings))
    mapping_palette = {mapp: color for mapp, color in zip(mappings, mapping_colors)}
    
    # Stripplot for all data
    sns.stripplot(
        data=df, x='Source', y='Accuracy', hue='Type',
        dodge=True, jitter=True, ax=ax, 
        palette={'Test Normal':'gray', 'Test Shuffled':'lightgray'},
        size=3, alpha=0.6
    )
    
    # Overlay SEM error bars and connect means
    x_positions = {src: i for i, src in enumerate(sources)}
    offset_normal = -0.15
    offset_shuffled = 0.15
    
    for idx, row in agg.iterrows():
        mapp = row['Mapping']
        src = row['Source']
        typ = row['Type']
        mean = row['mean']
        std_ = row['std']
        x = x_positions[src] + (offset_normal if typ=='Test Normal' else offset_shuffled)
        # Plot error bar
        ax.errorbar(x, mean, yerr=std_, fmt='o', color=mapping_palette[mapp], capsize=4)
    
    # Connect means for each mapping and source
    for mapp in mappings:
        for src in sources:
            # Retrieve means
            mean_norm = agg[(agg['Mapping']==mapp) & 
                             (agg['Source']==src) & 
                             (agg['Type']=='Test Normal')]['mean'].values[0]
            mean_shuf = agg[(agg['Mapping']==mapp) & 
                             (agg['Source']==src) & 
                             (agg['Type']=='Test Shuffled')]['mean'].values[0]
            x0 = x_positions[src] + offset_normal
            x1 = x_positions[src] + offset_shuffled
            p_val = pvals[(mapp, src)]
            linestyle = '-' if p_val <= 0.05 else '--'
            ax.plot([x0, x1], [mean_norm, mean_shuf], color=mapping_palette[mapp], linestyle=linestyle)
    
    # Final touches
    ax.axvline(x=0.5, color='k', linestyle='-', linewidth=1)
    ax.axvline(x=0,   color='k', linestyle='--', linewidth=1)
    ax.axvline(x=1,   color='k', linestyle='--', linewidth=1)
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('')
    #ax.legend(title='Type', loc='upper left')

    patches = [mpatches.Patch(color=mapping_color[m], label=m)
               for m in mappings]
    fig.legend(handles=patches, title='Mapping', loc='upper center',
               ncol=len(mappings), bbox_to_anchor=(0.5, 1.05))
    plt.tight_layout()
    plt.show()     


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


def save_rastermap_pdf(
    feat: str = "concat_z",
    mapping: str = "rm",
    bg: bool = False,
    cv: bool = False,
    vers: str = "concat",
    ephys: bool = False,
    nclus: int = 13,
    rerun: bool = False,
    bounds: bool = True,
    gamma: float | bool = False,
    *,
    # Rastermap-relevant args (must match regional_group signature)
    grid_upsample: int = 0,
    locality: float = 0.75,
    time_lag_window: int = 5,
    symmetric: bool = False,
):
    """
    Save a rastermap image as a PDF, optionally with colored row background.
    Relies on regional_group(...) for sorting and segment order.
    """

    # ---- load data with the requested switches ----
    r = regional_group(
        mapping,
        vers=vers,
        ephys=ephys,
        grid_upsample=grid_upsample,
        nclus=nclus,
        rerun=rerun,
        cv=cv,
        locality=locality,
        time_lag_window=time_lag_window,
        symmetric=symmetric,
    )

    spks = r[feat]
    isort = r["isort"]
    data = spks[isort]

    # ---- normalize to [0, 1] with safeguards ----
    data_min = float(np.min(data))
    data = data - data_min
    data_max = float(np.max(data))
    norm_data = (data / data_max) if data_max > 0 else np.zeros_like(data)

    if isinstance(gamma, (float, int)) and gamma is not False:
        g = norm_data ** float(gamma)
        gray = 1.0 - g
    else:
        gray = 1.0 - norm_data
    gray_u8 = (gray * 255).astype(np.uint8)

    # build RGBA image
    image_rgba = np.zeros((*gray_u8.shape, 4), dtype=np.uint8)
    image_rgba[..., 0] = gray_u8
    image_rgba[..., 1] = gray_u8
    image_rgba[..., 2] = gray_u8
    image_rgba[..., 3] = 255

    # ---- optional colored background per row ----
    if bg:
        row_colors = np.asarray(r["cols"])[isort]
        if row_colors.ndim == 2 and row_colors.shape[1] == 4:
            row_colors = row_colors[:, :3]
        row_colors_u8 = (np.clip(row_colors, 0, 1) * 255).astype(np.uint8)

        alpha_overlay = 0.20
        rgb = image_rgba[..., :3]
        for i in range(rgb.shape[0]):
            overlay = row_colors_u8[i]
            rgb[i, :, :] = (
                (1.0 - alpha_overlay) * rgb[i, :, :].astype(np.float32)
                + alpha_overlay * overlay[None, :].astype(np.float32)
            ).astype(np.uint8)

    # ---- cluster boundaries (Rastermap only) ----
    if mapping == "rm" and bounds:
        rc = np.asarray(r["cols"])[isort]
        if rc.ndim == 2 and rc.shape[0] > 1:
            color_changes = np.any(np.diff(rc, axis=0) != 0, axis=1)
            boundaries = np.where(color_changes)[0] + 1
            for y in boundaries:
                if 0 <= y < image_rgba.shape[0]:
                    image_rgba[y, :, 0] = 0
                    image_rgba[y, :, 1] = 0
                    image_rgba[y, :, 2] = 0

    # ---- PETH/segment boundaries (vertical dotted lines) ----
    if "len" in r and isinstance(r["len"], dict) and len(r["len"]) > 0:
        h = 0
        for seg, seg_len in r["len"].items():
            x = int(h + seg_len)
            if x >= image_rgba.shape[1]:
                break
            image_rgba[::2, x, 0] = 0
            image_rgba[::2, x, 1] = 0
            image_rgba[::2, x, 2] = 0
            h += seg_len

    # ---- write PDF ----
    img = Image.fromarray(image_rgba[..., :3], mode="RGB")

    # descriptive, compact filename (encode relevant switches)
    gamma_tag = (
        f"{float(gamma):.3g}" if isinstance(gamma, (float, int)) and gamma is not False else "False"
    )
    fname = (
        f"_cv{int(bool(cv))}"
        f"_bg{int(bool(bg))}"
        f"_nclus{int(nclus)}"
        f"_upsample{int(grid_upsample)}"
        f"_locality{float(locality):.3f}"
        f"_tlag{int(time_lag_window)}"
        f"_sym{int(bool(symmetric))}"
        f".pdf"
    )

    out_dir = pth_dmn.parent / "imgs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / fname
    img.save(out_path, "PDF")
    return out_path




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


        axs[row, 0].set_ylabel(vers.replace(' ', '\n'), fontsize=6)

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
         f'umap_si_norm{norm_}_smooth{smooth}.png'), dpi=180, bbox_inches='tight')



def plot_umap_SI_concat_cosmos(algo='umap_z', mapping='Cosmos', 
                            smooth=True, norm_=True):
    '''
    Plot 2D smoothed UMAP embeddings for all Cosmos regions and network "concat"
    '''

    # Assuming these variables and functions are available from the environment:
    # - PETH_types_dict
    # - regional_group()
    # - pal
    # - MM_TO_INCH
    # - pth_dmn

    vers = 'concat'  # fixed to concat network
    r = regional_group(mapping, vers=vers)

    regs = sorted(set(r['acs']))  # all unique regions in Cosmos mapping
    dim = 2

    fig, axs = plt.subplots(nrows=1, ncols=len(regs),
                figsize=(183 * MM_TO_INCH, 24 * MM_TO_INCH),
                sharex=True, sharey=True)

    mins, maxs = [], []
    for i in range(dim):
        mins.append(np.floor(np.min(r[algo][:, i])))
        maxs.append(np.ceil(np.max(r[algo][:, i])))

    for i in range(dim):
        maxs[i] = maxs[i] * 0.01 + maxs[i]

    coords = {}
    regcol = {reg: np.array(r['cols'])[r['acs'] == reg][0] for reg in regs}
    imgs = {}
    meshsize = 256

    for reg in regs:
        scaled_data = [
            (r[algo][np.array(r['acs']) == reg, i] - mins[i]) / (maxs[i] - mins[i])
            for i in range(dim)
        ]
        coords[reg] = scaled_data

        if smooth:
            data = np.array(scaled_data).T
            inds = (data * (meshsize - 1)).astype('uint')
            img = np.zeros([meshsize] * dim)
            for pt in inds:
                img[tuple(pt)] += 1
            imsm = ndi.gaussian_filter(img.T, [5] * dim)
            imgs[reg] = imsm / np.max(imsm) if norm_ else imsm

    for col, reg in enumerate(regs):
        ax = axs[col] if len(regs) > 1 else axs

        if smooth:
            ax.imshow(imgs[reg], origin='lower', cmap='viridis')
            ax.set_aspect('equal')
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            for spine in ['top', 'right', 'bottom']:
                ax.spines[spine].set_visible(False)
        else:    
            ax.scatter(coords[reg][0], coords[reg][1], color=regcol[reg], s=0.05)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.set_aspect('equal')

        ax.set_title(f'{reg}\n{sum(r["acs"] == reg)}', fontsize=6, color=pal[reg])

    fig.tight_layout()
    fig.savefig(Path(pth_dmn.parent, 'imgs', 'overleaf_pdf',
         f'umap_si_concat_cosmos_norm{norm_}_smooth{smooth}.png'), dpi=180, bbox_inches='tight')


def energy_distance_multivariate(X, Y):
    n, m = len(X), len(Y)
    A = cdist(X, X, metric='euclidean')  # shape (n, n)
    B = cdist(Y, Y, metric='euclidean')  # shape (m, m)
    C = cdist(X, Y, metric='euclidean')  # shape (n, m)
    
    term1 = 2.0 * np.sum(C) / (n * m)
    term2 = np.sum(A) / (n * n)
    term3 = np.sum(B) / (m * m)
    return term1 - term2 - term3


def compare_centroid_dists():

    '''
    For mappings in kmeans, Beryl, Cosmos and functional, plot the 
    distribution of distances of the centroids to that of all cells, 
    return mean and std
    '''

    fig, ax = plt.subplots()
    records = []

    for mapping in ['kmeans', 'Beryl', 'Cosmos', 'layers']:

        r = regional_group(mapping=mapping)

        data = r['xyz'] * 1000  # convert to mm
        labels = np.array(r['acs'])
        unique_labels = np.unique(labels)
        centroids = np.array([np.mean(data[labels == label], axis=0) 
                              for label in unique_labels])        
        cent0 = np.mean(data, axis=0)
        dists = np.linalg.norm(centroids - cent0, axis=1)        
        # Build list of records for long-form DataFrame
        for d in dists:
            records.append({'mapping': mapping, 'dist': d})       
    
    df_plot = pd.DataFrame(records)
    sns.stripplot(x='mapping', y='dist', data=df_plot, ax=ax, jitter=True)    
    ax.set_title('Centroid distances to that of all cells')
    # print for each mapping the number of classes, mean and std of the dist
    for mapping in df_plot['mapping'].unique():
        mean_dist = df_plot[df_plot['mapping'] == mapping]['dist'].mean()
        std_dist = df_plot[df_plot['mapping'] == mapping]['dist'].std()
        n_classes = len(df_plot[df_plot['mapping'] == mapping])
        print(f"{mapping}: {n_classes} classes, mean dist: {mean_dist:.2f}, std dist: {std_dist:.2f}")


def reaction_time_hist():

    from brainwidemap import download_aggregate_tables
    trr = pd.read_parquet(download_aggregate_tables(one, type='trials'))
    df =  trr.copy()

    # 1) Reaction time (seconds)
    df["rt"] = df["firstMovement_times"] - df["stimOn_times"]

    # Keep finite RTs in 0.08–1.0 s range
    df = df[np.isfinite(df["rt"]) & (df["rt"] > 0.08) & (df["rt"] < 1.0)]   

    # df['rt'].hist(bins=5000) has clear peak at 0.14 sec as in fig1 BWM

    df['congruent'] = (((~np.isnan(df['contrastLeft'])) & (df['probabilityLeft'] == 0.8)) | ((~df["contrastRight"].isna()) & (df["probabilityLeft"] == 0.2)))

    # In [75]: Counter(df["congruent"])
    # Out[75]: Counter({True: 128725, False: 58358})

    df["abs_contrast"] = df[["contrastLeft", "contrastRight"]].max(axis=1).abs()

    agg = (
        df.groupby(["abs_contrast", "congruent"], dropna=False)["rt"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg["sem"] = agg["std"] / np.sqrt(agg["count"].clip(lower=1))

    # --- Separate lines for plotting ---
    line_cong  = agg[agg["congruent"] == True ].sort_values("abs_contrast")
    line_incon = agg[agg["congruent"] == False].sort_values("abs_contrast")

    # --- Plot ---
    plt.figure(figsize=(6.4, 4.2))

    if not line_cong.empty:
        plt.errorbar(line_cong["abs_contrast"], line_cong["mean"], 
                    yerr=line_cong["sem"], fmt="-o", capsize=3, label="Congruent")
    if not line_incon.empty:
        plt.errorbar(line_incon["abs_contrast"], line_incon["mean"], 
                    yerr=line_incon["sem"], fmt="-o", capsize=3, label="Incongruent")

    plt.xlabel("Absolute contrast")
    plt.ylabel("Reaction time (s)")
    plt.title("Reaction time vs. contrast\n(congruent vs. incongruent blocks)")
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()


def make_contact_sheet_a4_sorted(
    in_dir,
    out_png="contact_sheet_A4.png",
    out_pdf="contact_sheet_A4.pdf",
    max_images=100,
    dpi=300,
    margin_mm=10,
):
    in_dir = Path(in_dir)

    def extract_nclus(path: Path) -> int:
        """
        Extract integer after 'nclus' in filename, e.g.
        'rastermap_rm_cv0_bg1_nclus7_concat.png' -> 7
        """
        m = re.search(r"nclus(\d+)", path.name)
        if not m:
            raise ValueError(f"No 'nclus' number found in filename: {path}")
        return int(m.group(1))


    # --- collect and sort PNG files by nclus ---
    files = sorted(in_dir.glob("*.png"), key=extract_nclus)
    if not files:
        raise FileNotFoundError(f"No PNG files found in: {in_dir}")

    files = files[:max_images]
    n_images = len(files)
    nclus_vals = [extract_nclus(p) for p in files]

    # --- A4 size in pixels ---
    MM_PER_INCH = 25.4
    a4_width_px  = int(210 / MM_PER_INCH * dpi)
    a4_height_px = int(297 / MM_PER_INCH * dpi)

    # --- margins in pixels ---
    margin_px = int(margin_mm / MM_PER_INCH * dpi)

    # --- grid layout ---
    cols = math.ceil(math.sqrt(n_images))
    rows = math.ceil(n_images / cols)

    usable_width  = a4_width_px  - 2 * margin_px
    usable_height = a4_height_px - 2 * margin_px
    cell_w = usable_width  // cols
    cell_h = usable_height // rows
    if cell_w <= 0 or cell_h <= 0:
        raise ValueError("Grid too dense for chosen DPI/margins.")

    # --- canvas ---
    sheet = Image.new("RGB", (a4_width_px, a4_height_px), "white")
    draw = ImageDraw.Draw(sheet)

    # font size relative to cell
    font_size = int(min(cell_w, cell_h) * 0.18)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    # --- paste images and draw labels ---
    for i, (path, nclus) in enumerate(zip(files, nclus_vals)):
        img = Image.open(path).convert("RGB")
        # fit into cell
        img.thumbnail((cell_w, cell_h), Image.LANCZOS)

        row = i // cols
        col = i % cols

        x0 = margin_px + col * cell_w
        y0 = margin_px + row * cell_h

        # center image in its cell
        x_img = x0 + (cell_w - img.width) // 2
        y_img = y0 + (cell_h - img.height) // 2
        sheet.paste(img, (x_img, y_img))

        # nclus label at top-left of the *cell*
        text = str(nclus)
        tx = x0 + int(cell_w * 0.03)
        ty = y0 + int(cell_h * 0.03)

        # pseudo-bold: draw several times with small offsets
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]:
            draw.text((tx + dx, ty + dy), text, fill="black", font=font)

    # --- save ---
    sheet.save(out_png, dpi=(dpi, dpi))
    sheet.save(out_pdf, "PDF", resolution=dpi)
    print(f"Saved {out_png} and {out_pdf}")



def plot_fr_lz_scatter_with_marginals(
    fr_key="fr",
    lz_key="lz",
    reg_key="Beryl",
    bins=120,
    s=3,
    alpha=0.25,
    rasterized=True):
    """
    Scatter: x=fr, y=lz, colored by Beryl region via `pal[reg]`.
    Marginals: top histogram of fr, right histogram of lz.
    """

    r = regional_group('rm')
    fr = np.asarray(r[fr_key]).astype(float)
    lz = np.asarray(r[lz_key]).astype(float)
    regs = np.asarray(r[reg_key])

    # Map region -> color (pal values can be RGB tuples in [0,1] or hex strings)
    cols = np.array([pal[reg] for reg in regs], dtype=object)

    # Filter invalid values (and keep colors aligned)
    m = np.isfinite(fr) & np.isfinite(lz)
    fr, lz, cols = fr[m], lz[m], cols[m]

    # Layout: scatter with marginals (no seaborn; pure matplotlib)
    fig = plt.figure(figsize=(9, 8), constrained_layout=True)
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=(4.0, 1.2),
        height_ratios=(1.2, 4.0),
        wspace=0.05, hspace=0.05
    )

    ax_histx = fig.add_subplot(gs[0, 0])
    ax_scatt = fig.add_subplot(gs[1, 0], sharex=ax_histx)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax_scatt)

    # Scatter
    ax_scatt.scatter(fr, lz, c=cols.tolist(), s=s, alpha=alpha, linewidths=0, rasterized=rasterized)
    ax_scatt.set_xlabel("Firing rate (fr)")
    ax_scatt.set_ylabel("Lempel-Ziv (lz)")

    # Marginals
    ax_histx.hist(fr, bins=bins)
    ax_histy.hist(lz, bins=bins, orientation="horizontal")

    # Clean up tick labels on marginals
    plt.setp(ax_histx.get_xticklabels(), visible=False)
    plt.setp(ax_histy.get_yticklabels(), visible=False)
    ax_histx.tick_params(axis="x", length=0)
    ax_histy.tick_params(axis="y", length=0)

    # Optional: remove spines for a cleaner marginal look
    for ax in (ax_histx, ax_histy):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.show()



















    
