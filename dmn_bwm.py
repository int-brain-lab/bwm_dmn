from one.api import ONE
from brainbox.singlecell import bin_spikes2D
from brainwidemap import (bwm_query, load_good_units, 
                          load_trials_and_mask, bwm_units)
from ibllib.atlas import AllenAtlas
from ibllib.atlas.regions import BrainRegions
import ibllib
from ibllib.atlas.flatmaps import plot_swanson_vector 
from brainbox.io.one import SessionLoader

from scipy import signal
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage
import gc
from pathlib import Path
import random
from copy import deepcopy
import time
import sys
import math
import string
import os
from scipy.stats import spearmanr
import umap

from matplotlib.axis import Axis
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import ListedColormap, LinearSegmentedColormap   
from matplotlib.gridspec import GridSpec   
from statsmodels.stats.multitest import multipletests
from matplotlib.lines import Line2D

import warnings
warnings.filterwarnings("ignore")

'''
script to process the BWM dataset for manifold analysis,
saving intermediate results (PETHs), computing
metrics and plotting them (plot_all, also supp figures)

A split is a variable, such as stim, where the trials are
averaged within a certain time window

To compute all from scratch, including data download, run:

##################
for split in align:
    get_all_PETHs(split)  # computes PETHs, squared firing rates
    stack(split)  # combine results across insertions
    
plot_all()  # plot main figure    
##################    
'''


np.set_printoptions(threshold=sys.maxsize)

plt.rcParams.update(plt.rcParamsDefault)
plt.ion()

f_size = 15  # font size

# canonical colors for left and right trial types
blue_left = [0.13850039, 0.41331206, 0.74052025]
red_right = [0.66080672, 0.21526712, 0.23069468]

b_size = 0.0125  # bin size [sec] for neural binning
sts = 0.002  # stride size in [sec] for overlapping bins
ntravis = 30  # #trajectories for vis, first 2 real, rest pseudo

one = ONE(base_url='https://openalyx.internationalbrainlab.org',
          password='international', silent=True)
          
ba = AllenAtlas()
br = BrainRegions()
#units_df = bwm_units(one)  # canonical set of cells


# save results here
pth_res = Path(one.cache_dir, 'dmn', 'res')
pth_res.mkdir(parents=True, exist_ok=True)

sigl = 0.05  # significance level (for stacking, plotting, fdr)

# trial split types, with string to define alignment
align = {'stim': 'stimOn_times',
         'stimL': 'stimOn_times',
         'stimR': 'stimOn_times',
         'choice': 'firstMovement_times',
         'choiceL': 'firstMovement_times',
         'choiceR': 'firstMovement_times',
         'fback1': 'feedback_times',
         'fback0': 'feedback_times',
         'block': 'stimOn_times',
         'blockL': 'stimOn_times',
         'blockR': 'stimOn_times',
         'end': 'intervals_1'}


def pre_post(split, con=False):
    '''
    [pre_time, post_time] relative to alignment event
    split could be contr or restr variant, then
    use base window
    
    ca: If true, use canonical time windows
    '''

    pre_post0 = {'stim': [0, 0.15],
                 'choice': [0.15, 0.15],
                 'fback1': [0, 0.7],
                 'fback0': [0, 0.7],
                 'block': [0.4, -0.1],
                 'end': [0,0.5]}

    # canonical windows
    pre_post_con =  {'stimL': [0, 0.15],
                    'stimR':  [0, 0.15],
                    'blockL':  [0.4, -0.1],
                    'blockR': [0.4, -0.1],
                    'choiceL': [0.15, 0],
                    'choiceR': [0.15, 0],
                    'fback1': [0, 0.3],
                    'fback0': [0, 0.3],
                    'end': [0, 0.3]}

    pp = pre_post_con if con else pre_post0

    if '_' in split:
        return pp[split.split('_')[0]]
    else:
        return pp[split]


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
    print(f'listing {t} per region')
    
    return Counter(units_df[units_df['eid'] == eid]['pid'])  


def get_PETHs(split, pid):
    '''
    for a given variable and insertion,
    cut neural data into trials, bin the activity,
    compute distances of trajectories per cell
    to be aggregated across insertions later
    Also save PETHs and cell numbers per region

    input
    split: trial split variable, about the window
    pid: insertion id

    returns:
    Dictionary D_ with entries
    acs: region acronyms per cell
    ws: PETHs for both trial types
    d_eucs: Euclidean distance between PETHs,
            summed across same reg
    '''

    eid, probe = one.pid2eid(pid)

    # load in spikes
    spikes, clusters = load_good_units(one, pid)

    # Load in trials data and mask bad trials (False if bad)
    trials, mask = load_trials_and_mask(one, eid) 
                                     
    events = []

    # window of interest (split in align) and baseline 
    if 'fback' in  split:
        if split[-1] == '1':
            events.append(trials[align[split]][np.bitwise_and.reduce([
                    mask, trials['feedbackType'] == 1])])
            events.append(trials[align['block']][np.bitwise_and.reduce([
                    mask, trials['feedbackType'] == 1])])
        else:
            events.append(trials[align[split]][np.bitwise_and.reduce([
                    mask, trials['feedbackType'] == -1])])         
            events.append(trials[align['block']][np.bitwise_and.reduce([
                    mask, trials['feedbackType'] == -1])])        
           
    else:
        events.append(trials[align[split]][mask])
        events.append(trials[align['block']][mask])


    print(len(events[0]), 'trials')

    assert len(
        spikes['times']) == len(
        spikes['clusters']), 'spikes != clusters'

    # bin and cut into trials
    bins = []
    c = 0
    for event in events:
        if c == 1:  # compare to inter trial interval baseline
            split = 'block'
        #  overlapping time bins, bin size = b_size, stride = sts
        bis = []
        st = int(b_size // sts)

        for ts in range(st):

            bi, _ = bin_spikes2D(
                spikes['times'],
                clusters['cluster_id'][spikes['clusters']],
                clusters['cluster_id'],
                np.array(event) + ts * sts,
                pre_post(split)[0], pre_post(split)[1],
                b_size)
            bis.append(bi)

        ntr, nn, nbin = bi.shape
        ar = np.zeros((ntr, nn, st * nbin))

        for ts in range(st):
            ar[:, :, ts::st] = bis[ts]

        bins.append(ar)
        c += 1
    
    ids = np.array(clusters['atlas_id'])
    xyz = np.array(clusters[['x','y','z']])

    # Discard cells with any nan or 0 for all bins
    goodcells = [k for k in range(bins[0].shape[1]) if
                 (not np.isnan(bins[0][:,k,:]).any()
                 and bins[0][:,k,:].any())]

    ids = ids[goodcells]
    xyz = xyz[goodcells]
    bins2 = [x[:, goodcells, :] for x in bins]

    D = {}
    D['ids'] = ids
    D['xyz'] = xyz
    
    # first subtract firing rates per trial, then square, average
    # time-average baseline, ad dim for subtraction
    v = np.expand_dims(bins2[1].mean(axis=-1),-1)        

    D['ws'] = np.mean(bins2[0]**2, axis=0)    
    D['base'] = np.mean((bins2[0] - v)**2, axis=0)

    return D


def concat_PETHs(pid):

    '''
    for each cell concat all possible PETHs
    '''
    
    eid, probe = one.pid2eid(pid)

    # load in spikes
    spikes, clusters = load_good_units(one, pid)        
    assert len(
            spikes['times']) == len(
            spikes['clusters']), 'spikes != clusters'
            
    D = {}
    D['ids'] = np.array(clusters['atlas_id'])
    D['xyz'] = np.array(clusters[['x','y','z']])

    # Load in trials data and mask bad trials (False if bad)
    trials, mask = load_trials_and_mask(one, eid)     
        
    # define align, trial type, window length   
    tts = {
        'stimL': ['stimOn_times', ~np.isnan(trials[f'contrastLeft']), [0, 0.15]],
        'stimR': ['stimOn_times', ~np.isnan(trials[f'contrastRight']), [0, 0.15]],
        'blockL': ['stimOn_times', trials['probabilityLeft'] == 0.8, [0.4, -0.1]],
        'blockR': ['stimOn_times', trials['probabilityLeft'] == 0.2, [0.4, -0.1]],
        'choiceL': ['firstMovement_times', trials['choice'] == 1, [0.15, 0]],
        'choiceR': ['firstMovement_times', trials['choice'] == -1, [0.15, 0]],
        'fback1': ['feedback_times', trials['feedbackType'] == 1, [0, 0.3]],
        'fback0': ['feedback_times', trials['feedbackType'] == -1, [0, 0.3]],
        'end': ['intervals_1', np.full(len(trials['choice']), True), [0, 0.3]]}


    tls = {}  # trial numbers for each type
    ws = []  # list of binned data
    
    for tt in tts:

        event = trials[tts[tt][0]][np.bitwise_and.reduce([mask, tts[tt][1]])]
        tls[tt] = len(event)

        # bin and cut into trials
        # overlapping time bins, bin size = b_size, stride = sts
        bis = []
        st = int(b_size // sts)

        for ts in range(st):

            bi, _ = bin_spikes2D(
                spikes['times'],
                clusters['cluster_id'][spikes['clusters']],
                clusters['cluster_id'],
                np.array(event) + ts * sts,
                tts[tt][-1][0], tts[tt][-1][1],
                b_size)
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


'''
###
### bulk processing
###
'''


def get_all_PETHs(split, eids_plus=None):

    '''
    for all BWM insertions, get the PSTHs and acronyms,
    i.e. run get_PETHs
    '''

    time00 = time.perf_counter()

    if eids_plus is None:
        df = bwm_query(one)
        eids_plus = df[['eid', 'probe_name', 'pid']].values

    # save results per insertion (eid_probe) in FlatIron folder
    pth = Path(one.cache_dir, 'dmn', split)
    pth.mkdir(parents=True, exist_ok=True)

    Fs = []
    k = 0
    print(f'Processing {len(eids_plus)} insertions')
    for i in eids_plus:
        eid, probe, pid = i

        time0 = time.perf_counter()
        try:
        
            if split == 'concat':
                D = concat_PETHs(pid)    
            else:
                D = get_PETHs(split, pid)
                            
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
    print((time11 - time00) / 60, f'min for the complete bwm set, {split}')
    print(f'{len(Fs)}, load failures:')
    print(Fs)


def check_for_load_errors(splits):

    s = [] 
    for split in splits:
        pth = Path(one.cache_dir, 'dmn', split)
        ss = os.listdir(pth)
        print(split, len(ss))
        s.append(ss)
           
    flat_list = [item for sublist in s for item in sublist]
    
    g = {}
    for split in splits:
        pth = Path(one.cache_dir, 'dmn', split)
        ss = os.listdir(pth)
        g[split] = list(set(flat_list)-set(ss))    
    
    # re-run with missing pids
    for split in g:
        h = [fn2_eid_probe_pid(u) for u in g[split]]
        get_all_PETHs(split, eids_plus=h)


    print('POST CORRECTION')
    for split in splits:
        pth = Path(one.cache_dir, 'dmn', split)
        ss = os.listdir(pth)
        print(split, len(ss))



def stack(split, min_reg=20, mapping='Beryl', layers=False, per_cell=False):

    time0 = time.perf_counter()

    '''
    pool across insertions,
    remove regions with too few neurons (min_reg)
    compute maxes, latencies and p-values
    
    eids_only: list of eids to restrict to (for licking)
    '''

    pth = Path(one.cache_dir, 'dmn', split)
    ss = os.listdir(pth)  # get insertions
    print(f'combining {len(ss)} insertions for split {split}') 
    
    # pool data for illustrative PCA
    ids = []
    xyz = []
    ws = []
    base = []
 
    if layers:
        mapping = 'Allen'        

    # group results across insertions
    for s in ss:
                   
        D_ = np.load(Path(pth, s),
                     allow_pickle=True).flat[0]

        ids.append(D_['ids'])
        ws.append(D_['ws'])
        base.append(D_['base'])
        xyz.append(D_['xyz'])

    ids = np.concatenate(ids)
    ws = np.concatenate(ws, axis=0)
    base = np.concatenate(base, axis=0)
    xyz = np.concatenate(xyz)

    acs = np.array(br.id2acronym(ids, mapping=mapping))

    # discard ill-defined regions
    goodcells = ~np.bitwise_or.reduce([acs == reg for
                                       reg in ['void', 'root']])

    ids = ids[goodcells]
    acs = acs[goodcells]
    ws = ws[goodcells] 
    base = base[goodcells]
    xyz = xyz[goodcells]

    if per_cell:
        return xyz, base
    
    print('computing grand average metrics ...')
    ncells, nt = ws.shape

    ga = {}
    ga['m0'] = np.mean(ws, axis=0)
    ga['ms'] = np.mean(base, axis=0)

    ga['v0'] = np.std(ws, axis=0) / (ncells**0.5)
    ga['vs'] = np.std(base, axis=0) / (ncells**0.5)

    ga['xyz'] = xyz
    ga['Beryl'] = acs
    ga['ids'] = ids
    ga['nclus'] = ncells

    # first is window, second is base window
    pca = PCA(n_components=3)
    wsc = pca.fit_transform(np.concatenate([ws, base],
                                            axis=1).T).T
    ga['pcs'] = wsc
    
    # temporal dim reduction
    pca = PCA(n_components=2)
    ga['pcst'] = pca.fit_transform(ws)
    ga['pcstb'] = pca.fit_transform(base)
    
    # differences of window and base, pooling all cells
    ga['d_euc'] = np.mean(ws, axis=0)**0.5                    
    ga['d_eucb'] = np.mean(base, axis=0)**0.5
    
    # per cell amplitudes
    ga['amp'] = np.max(ws, axis=1)
    ga['ampb'] = np.max(base, axis=1)
    
    ga['sum'] = np.sum(ws, axis=1)
    ga['sumb'] = np.sum(base, axis=1)              

    np.save(Path(pth_res, f'{split}_grand_averages.npy'), ga,
            allow_pickle=True)

    print('computing regional metrics ...')  
    
    regs0 = Counter(acs)

    if layers:

        # goup by last number
        regs = {reg: regs0[reg] for reg in regs0 
                if reg[-1].isdigit()}
        
        for reg in regs:        
            acs[acs == reg] = reg[-1]       
        
        # extra class of thalamic (and hypothalamic) regions 
        names = dict(zip(regs0,[get_name(reg) for reg in regs0]))
        thal = {x:names[x] for x in names if 'thala' in names[x]}
                                          
        for reg in thal: 
            acs[acs == reg] = 'thal'       
        
        mask = np.array([(x.isdigit() or x == 'thal') for x in acs])
        acs[~mask] = '0' 
        regsl = Counter(acs)
        regs = {reg:regsl[reg] for reg in regsl if regsl[reg] > min_reg}
        print('layers')
        print(regs)
        
                 
    else:
        regs = {reg: regs0[reg] for reg in regs0 
                if regs0[reg] > min_reg}
                
        print(len(regs), 'regions')    
    
    r = {}
    for reg in regs:
        res = {}

        # get PCA for 3d trajectories
        dat = ws[acs == reg, :]
        datb = base[acs == reg, :]
        
        res['ws'] = dat
        res['base'] = datb

        pca = PCA(n_components=3)
        wsc = pca.fit_transform(np.concatenate([dat, datb], axis=1).T).T

        res['pcs'] = wsc
        res['nclus'] = regs[reg]

        v = datb.mean(axis=1)
        res['d_euc'] = np.mean(dat, axis=0)**0.5
        res['d_eucb'] = np.mean(datb, axis=0)**0.5 
            
        # amplitude
        res['amp_euc'] = max(res['d_euc'])
        res['amp_eucb'] = max(res['d_eucb'])
        
        # sum
        res['sum_euc'] = np.sum(res['d_euc'])
        res['sum_eucb'] = np.sum(res['d_eucb'])        
        
        # latency
        loc = np.where((res['d_euc'] - min(res['d_euc'])) 
                        > 0.7 * (res['amp_euc'] - min(res['d_euc'])))[0]
        res['lat_euc'] = np.linspace(-pre_post(split)[0],
                                     pre_post(split)[1],
                                     len(res['d_euc']))[loc[0]]
                                     
        loc = np.where((res['d_eucb'] - min(res['d_eucb'])) 
                        > 0.7 * (res['amp_eucb'] - min(res['d_eucb'])))[0]
        res['lat_eucb'] = np.linspace(-pre_post(split)[0],
                                     pre_post(split)[1],
                                     len(res['d_eucb']))[loc[0]]
                                     
        r[reg] = res
        

    np.save(Path(pth_res, f'{split}.npy'),
            r, allow_pickle=True)

    time1 = time.perf_counter()
    print('total time:', np.round(time1 - time0, 0), 'sec')


def stack_concat():

    split = 'concat'
    pth = Path(one.cache_dir, 'dmn', split)
    ss = os.listdir(pth)  # get insertions
    print(f'combining {len(ss)} insertions for split {split}') 

    # pool data
    ids = []
    xyz = []
    ws = []   

    # group results across insertions
    for s in ss:
                   
        D_ = np.load(Path(pth, s),
                     allow_pickle=True).flat[0]

        ws.append(np.concatenate(D_['ws'],axis=1))
        
        ids.append(D_['ids'])
        xyz.append(D_['xyz'])
        
    r = {}  
    r['len'] = dict(zip(D_['trial_names'],
                        [x.shape[1] for x in D_['ws']]))  
    r['ids'] = np.concatenate(ids)
    r['xyz'] = np.concatenate(xyz)       
    cs = np.concatenate(ws, axis=0)
    
    # remove cells with nan entries
    goodcells = [~np.isnan(k).any() for k in cs]

    r['ids'] = r['ids'][goodcells]
    r['xyz'] = r['xyz'][goodcells]    
    cs = cs[goodcells] 
    
    r['mean'] = cs.mean(axis=0) 
    r['std'] = cs.std(axis=0)

    # various dim reduction to 2 dims
    ncomp = 2
    r['umap'] = umap.UMAP(n_components=ncomp).fit_transform(cs)
    r['tSNE'] = TSNE(n_components=ncomp).fit_transform(cs)
    r['PCA'] = PCA(n_components=ncomp).fit_transform(cs)
    r['ICA'] = FastICA(n_components=ncomp).fit_transform(cs) 

    np.save(Path(pth_res, f'{split}.npy'),
            r, allow_pickle=True)            
        

'''
#####################################################
### plotting
#####################################################
'''


def get_allen_info(rerun=False):
    '''
    Function to load Allen atlas info, like region colors
    '''
    
    pth_dmn = Path(one.cache_dir, 'dmn', 'alleninfo.npy')
    
    if (not pth_dmn.is_file() or rerun):
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
        bc = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        for i in range(6):
            palette[str(i)] = bc[i]
        
        palette['thal'] = 'k'    
        r = {}
        r['dfa'] = dfa
        r['palette'] = palette    
        np.save(pth_dmn, r, allow_pickle=True)   

    r = np.load(pth_dmn, allow_pickle=True).flat[0]
    return r['dfa'], r['palette']  


def put_panel_label(ax, k):
    ax.annotate(string.ascii_lowercase[k], (-0.05, 1.15),
                xycoords='axes fraction',
                fontsize=f_size * 1.5, va='top',
                ha='right', weight='bold')


def plot_all(splits=None, curve='euc', show_tra=True, axs=None,
             all_labs=False,ga_pcs=True, extra_3d=False, fig=None):
    '''
    main manifold figure:
    1. plot example 3D trajectories,
    2. plot lines for distance(t) (curve 'var' or 'euc')
       for select regions
    3. plot 2d scatter [amplitude, latency] of all regions

    sigl: significance level, default 0.01, p_min = 1/(nrand+1)
    ga_pcs: If true, plot 3d trajectories of all cells,
            else plot for a single region (first in exs list)

    all_labs: show all labels in scatters, else just examples

    '''
    if splits is None:
        splits = align

    # specify grid; scatter longer than other panels
    ncols = 12
    
    if not fig:
        alone = True
        axs = []
        if show_tra:
            fig = plt.figure(figsize=(20, 2.5*len(splits)))
            gs = fig.add_gridspec(len(splits), ncols)
        else:   
            fig = plt.figure(figsize=(20, 2.5*len(splits)),
                             layout='constrained')
            
            gs = fig.add_gridspec(len(splits), ncols)
        
    k = 0  # panel counter

    fsize = 12  # font size
    dsize = 13  # diamond marker size
    lw = 1  # linewidth       

    dfa, palette = get_allen_info()

    '''
    get significant regions
    '''
    tops = {}
    regsa = []

    for split in splits:

        d = np.load(Path(pth_res, f'{split}.npy'),
                    allow_pickle=True).flat[0]

        maxs = np.array([d[x][f'amp_{curve}'] for x in d])
        acronyms = np.array(list(d.keys()))
        order = list(reversed(np.argsort(maxs)))
        maxs = maxs[order]
        acronyms = acronyms[order]

        tops[split] = [acronyms, maxs]
        maxsf = [v for v in maxs if not (math.isinf(v) or math.isnan(v))]

        print(split, len(d), 'regions')

        regsa.append(list(d.keys()))
        print(' ')

    #  get Cosmos parent region for yellow color adjustment
    regsa = np.unique(np.concatenate(regsa))

    '''
    example regions per split for embedded space and line plots
    
    first in list is used for pca illustration
    '''
    
    exs0 = {'stim': ['LGd','VISp', 'PRNc','VISam','IRN', 'VISl',
                     'VISpm', 'VM', 'MS','VISli'],


            'choice': ['PRNc', 'VISal','PRNr', 'LSr', 'SIM', 'APN',
                       'MRN', 'RT', 'LGd', 'GRN','MV','ORBm'],

            'fback': ['IRN', 'SSp-n', 'PRNr', 'IC', 'MV', 'AUDp',
                      'CENT3', 'SSp-ul', 'GPe'],
            'block': ['Eth', 'IC'],

            'end': ['PRNc', 'VISal','PRNr', 'LSr', 'SIM', 'APN',
                       'MRN', 'RT', 'LGd', 'GRN','MV','ORBm']}

    exs1 = ['LGd','VISp', 'PRNc','VISam','IRN', 'VISl',
            'VISpm', 'VM', 'MS','VISli']

    # use same example regions for variant splits
    exs = exs0.copy()
    for split in splits:
        for split0 in exs0:
            if split0 in split:
                exs[split] = exs0[split0]
                
                
    if len(Counter(regsa)) < 10:
        print(f'{len(Counter(regsa))} regs detected, show all')  
        exs1 = list(Counter(regsa))
        for split in splits:
            exs[split] = exs1

    if show_tra:

        '''
        Trajectories for example regions in PCA embedded 3d space
        ''' 
            
        row = 0
        for split in splits:

            if ga_pcs:
                dd = np.load(Path(pth_res, f'{split}_grand_averages.npy'),
                            allow_pickle=True).flat[0]
            else:
                d = np.load(Path(pth_res, f'{split}.npy'),
                            allow_pickle=True).flat[0]

                # pick example region
                reg = exs[split][0]
                dd = d[reg]

            if extra_3d:
                axs.append(fig.add_subplot(gs[:,row*3: (row+1)*3],
                                           projection='3d'))        
            else:
                if alone:
                    axs.append(fig.add_subplot(gs[row, :3],
                               projection='3d'))            

            npcs, allnobs = dd['pcs'].shape
            lenw = len(dd['d_euc'])  # window length
            lbase = allnobs - len(dd['d_euc'])  # base window length
            nobs = [lenw, lbase]    
            st = [0,lenw] 
            en = [lenw, lbase + lenw]
            
            for j in range(2):

                # 3d trajectory
                cs = dd['pcs'][:,st[j]:en[j]].T

                if j == 0:
                    col = grad('Blues_r', nobs[j])
                elif j == 1:
                    col = grad('Reds_r', nobs[j])

                axs[k].plot(cs[:, 0], cs[:, 1], cs[:, 2],
                            color=col[len(col) // 2],
                            linewidth=5 if j in [0, 1] else 1, alpha=0.5)

                axs[k].scatter(cs[:, 0], cs[:, 1], cs[:, 2],
                               color=col,
                               edgecolors=col,
                               s=20 if j in [0, 1] else 1,
                               depthshade=False)

            if extra_3d:
                axs[k].set_title(split.split('_')[0])    

            else:
                axs[k].set_title(f"{split}, {reg} {d[reg]['nclus']}"
                                 if not ga_pcs else split)
            axs[k].grid(False)
            axs[k].axis('off')

            if not extra_3d:
                put_panel_label(axs[k], k)

            k += 1
            row += 1
            
        if extra_3d:
            return
        
    '''
    line plot per 5 example regions per split
    '''
    row = 0  # index

    for split in splits:

        if show_tra:
            axs.append(fig.add_subplot(gs[row, 3:6]))
        else:
            if alone:
                axs.append(fig.add_subplot(gs[0, :]))


        d = np.load(Path(pth_res, f'{split}.npy'),
                    allow_pickle=True).flat[0]

        # example regions to illustrate line plots
        regs = exs1  #exs[split]

        texts = []
        for reg in regs:
            if reg not in d:
                print(f'{reg} not in d:'
                       'revise example regions for line plots')
                continue
        
            if any(np.isinf(d[reg][f'd_{curve}'])):
                print(f'inf in {curve} of {reg}')
                continue

            xx = np.linspace(-pre_post(split)[0],
                             pre_post(split)[1],
                             len(d[reg][f'd_{curve}']))

            # get units in Hz
            yy = d[reg][f'd_{curve}']

            axs[k].plot(xx, yy, linewidth=lw,
                        color=palette[reg],
                        label=f"{reg} {d[reg]['nclus']}")

            # put region labels
            y = yy[-1]
            x = xx[-1]
            ss = f"{reg} {d[reg]['nclus']}"

            texts.append(axs[k].text(x, y, ss,
                                     color=palette[reg],
                                     fontsize=fsize))


        axs[k].axvline(x=0, lw=0.5, linestyle='--', c='k')

        if split in ['block', 'choice']:
            ha = 'left'
        else:
            ha = 'right'

        axs[k].spines['top'].set_visible(False)
        axs[k].spines['right'].set_visible(False)

        axs[k].set_ylabel('distance [Hz]')
        axs[k].set_xlabel('time [sec]')
        
        if show_tra:
            put_panel_label(axs[k], k)

        row += 1
        k += 1


    '''
    scatter latency versus max amplitude for significant regions
    '''

    row = 0  # row idx

    for split in splits:

        if show_tra:
            axs.append(fig.add_subplot(gs[row, 6:]))
        else:
            if alone:
                axs.append(fig.add_subplot(gs[1,:]))    


        d = np.load(Path(pth_res, f'{split}.npy'),
                    allow_pickle=True).flat[0]

        acronyms = list(d.keys())
        maxes = np.array([d[x][f'amp_{curve}'] for x in acronyms])
        lats = np.array([d[x][f'lat_{curve}'] for x in acronyms])
        cols = [palette[reg] for reg in acronyms]

        axs[k].errorbar(lats, maxes, yerr=None, fmt='None',
                        ecolor=cols, ls='None', elinewidth=0.5)

        # plot regions
        axs[k].scatter(np.array(lats),
                       np.array(maxes),
                       color=np.array(cols),
                       marker='D', s=dsize)

        texts = []
        for i in range(len(acronyms)):

            reg = acronyms[i]
            if reg not in exs[split]:
                if not all_labs: # restrict to example regions   
                    continue
            
            texts.append(
                axs[k].annotate(
                    '  ' + reg,
                    (lats[i], maxes[i]),
                    fontsize=fsize,
                    color=palette[acronyms[i]],
                    arrowprops=None))
                        

        axs[k].axvline(x=0, lw=0.5, linestyle='--', c='k')

        ha = 'left'


        axs[k].text(0, 0, align[split.split('_')[0]
                                   if '_' in split else split],
                    transform=axs[k].get_xaxis_transform(),
                    horizontalalignment=ha, rotation=90,
                    fontsize=f_size * 0.8)

        axs[k].spines['top'].set_visible(False)
        axs[k].spines['right'].set_visible(False)

        axs[k].set_ylabel('max dist. [Hz]')
        axs[k].set_xlabel('latency [sec]')
        
        if show_tra:
            put_panel_label(axs[k], k)


        row += 1
        k += 1

    if not show_tra:
        axs[-1].sharex(axs[-2])


def get_cmap(split):
    '''
    for each split, get a colormap defined by Yanliang
    '''
    dc = {'stim': ["#ffffff","#D5E1A0","#A3C968",
                   "#86AF40","#517146"],
          'choice': ["#ffffff","#F8E4AA","#F9D766",
                     "#E8AC22","#DA4727"],
          'fback': ["#ffffff","#F1D3D0","#F5968A",
                    "#E34335","#A23535"],
          'fback1': ["#ffffff","#F1D3D0","#F5968A",
                    "#E34335","#A23535"],                    
          'fback0': ["#ffffff","#F1D3D0","#F5968A",
                    "#E34335","#A23535"],                    
          'end': ["#ffffff","#D0CDE4","#998DC3",
                    "#6159A6","#42328E"],         
          'block': ["#ffffff","#D0CDE4","#998DC3",
                    "#6159A6","#42328E"]}

    if '_' in split:
        split = split.split('_')[0]

    return LinearSegmentedColormap.from_list("mycmap", dc[split])
   
    
def plot_swanson_supp(splits = align, curve = 'eucb',
                      show_legend = False, bina=False):
 
    '''
    swanson maps for maxes
    '''
    
    nrows = 2  # one for amplitudes, one for latencies
    ncols = len(splits)  # one per variable


    fig, axs = plt.subplots(nrows, ncols, figsize=(14, 11)) 
    
    if show_legend:
        '''
        plot Swanson flatmap with labels and colors
        '''
        fig0, ax0 = plt.subplots()
        plot_swanson_vector(annotate=True, ax=ax0)
        ax0.axis('off')
   
    k = 0  # panel counter
    c = 0  # column counter


    # normalize across variable values
    all_amps = []
    for split in splits:

        d = np.load(Path(pth_res, f'{split}.npy'),
                    allow_pickle=True).flat[0]    
        all_amps.append([d[x][f'amp_{curve}'] for x in d])
        
    aa = [item for sublist in all_amps for item in sublist]
    vmax0 = np.max(aa)
    vmin0 = np.min(aa)
    
    sws = []
    for split in splits:

        d = np.load(Path(pth_res, f'{split}.npy'),
                    allow_pickle=True).flat[0]
                    
        # get significant regions only
        acronyms = list(d.keys())


        # plot amplitudes
        if bina:
            # set all significant regions to 1, others zero
            amps = np.array([1 for x in acronyms])
        
        else:
            amps = np.array([d[x][f'amp_{curve}'] for x in acronyms])
            
        plot_swanson_vector(np.array(acronyms), np.array(amps)/vmax0, 
                            cmap=get_cmap(split), 
                            ax=axs[0,c], br=br, 
                            orientation='portrait',
                            linewidth=0.1,
                            vmin=vmin0,
                            vmax=vmax0,
                            annotate=True)
                            
        # add colorbar
        #clevels = (np.nanmin(amps), np.nanmax(amps))
        clevels = vmin0, vmax0
        norm = mpl.colors.Normalize(vmin=clevels[0], 
                                    vmax=clevels[1])
                                    
        cbar = fig.colorbar(mpl.cm.ScalarMappable(
                                norm=norm, 
                                cmap=get_cmap(split)), 
                                ax=axs[0,c])
    
        cbar.set_label('effect size [spikes/second]')

        axs[0,c].set_title(f'{split}')                    
        axs[0,c].axis('off')
        put_panel_label(axs[0,c], k)
        k += 1

        # plot latencies (cmap reversed, dark is early)    
        lats = np.array([d[x][f'lat_{curve}'] for x in acronyms]) 
        plot_swanson_vector(np.array(acronyms),np.array(lats), 
                     cmap=get_cmap(split).reversed(), 
                     ax=axs[1,c], br=br, orientation='portrait')

        clevels = (np.nanmin(lats), np.nanmax(lats))
        norm = mpl.colors.Normalize(vmin=clevels[0], 
                                    vmax=clevels[1])
                                    
        cbar = fig.colorbar(mpl.cm.ScalarMappable(
                                norm=norm, 
                                cmap=get_cmap(split).reversed()), 
                                ax=axs[1,c])
        cbar.set_label('latency [second]')


        axs[1,c].axis('off')
        axs[1,c].set_title(f'{split}')
        put_panel_label(axs[1,c], k)
        
        
        #print(split, acronyms, amps, lats)        
        
        k += 1
        c += 1

    fig.tight_layout()


def plot_traj_and_dist(split, reg='all', ga_pcs=False, curve='euc',
                       fig=None, axs=None):

    '''
    for a given region, plot 3d trajectory and 
    line plot below
    '''
    
    if 'can' in curve:
        print('using canonical time windows')
        can = True
    else:
        can = False   

    df, palette = get_allen_info()
    palette['all'] = (0.32156863, 0.74901961, 0.01568627,1)
     
    if not fig:
        alone = True     
        fig = plt.figure(figsize=(3,3.79))
        gs = fig.add_gridspec(5, 1)
        axs = [] 
        axs.append(fig.add_subplot(gs[:4, 0],
                               projection='3d'))
        axs.append(fig.add_subplot(gs[4:,0]))
       
    k = 0 
      
    # 3d trajectory plot
    if ga_pcs:
        dd = np.load(Path(pth_res, f'{split}_grand_averages.npy'),
                    allow_pickle=True).flat[0]            
    else:
        d = np.load(Path(pth_res, f'{split}.npy'),
                    allow_pickle=True).flat[0]

        # pick example region
        dd = d[reg]

    npcs, allnobs = dd['pcs'].shape
    lenw = len(dd['d_euc'])  # window length
    lbase = allnobs - len(dd['d_euc'])  # base window length
    nobs = [lenw, lbase]    
    st = [0,lenw] 
    en = [lenw, lbase + lenw]
    
    for j in range(2):

        # 3d trajectory
        cs = dd['pcs'][:,st[j]:en[j]].T

        if j == 0:
            col = grad('Blues_r', nobs[j])
        elif j == 1:
            col = grad('Reds_r', nobs[j])

        axs[k].plot(cs[:, 0], cs[:, 1], cs[:, 2],
                    color=col[len(col) // 2],
                    linewidth=5 if j in [0, 1] else 1, alpha=0.5)

        axs[k].scatter(cs[:, 0], cs[:, 1], cs[:, 2],
                       color=col,
                       edgecolors=col,
                       s=20 if j in [0, 1] else 1,
                       depthshade=False)

    if alone:
        axs[k].set_title(f"{split}, {reg} {dd['nclus']}")
                         
    axs[k].grid(False)
    axs[k].axis('off')

    #put_panel_label(axs[k], k)

    k += 1

    # line plot
    if reg != 'all':             
        if reg not in d:
            print(f'{reg} not in d:'
                   'revise example regions for line plots')
            return

    if any(np.isinf(dd[f'd_{curve}'].flatten())):
        print(f'inf in {curve} of {reg}')
        return

    xx = np.linspace(-pre_post(split)[0],
                     pre_post(split)[1],
                     len(dd[f'd_{curve}']))                   


    # get curve
    yy = dd[f'd_{curve}']

    axs[k].plot(xx, yy, linewidth=2,
                color=palette[reg],
                label=f"{reg} {dd['nclus']}")

    # put region labels
    y = yy[-1]
    x = xx[-1]
    ss = ' ' + reg

    axs[k].text(x, y, ss, color=palette[reg], fontsize=8)

    axs[k].axvline(x=0, lw=0.5, linestyle='--', c='k')

    if split in ['block', 'choice']:
        ha = 'left'
    else:
        ha = 'right'

    axs[k].spines['top'].set_visible(False)
    axs[k].spines['right'].set_visible(False)

    axs[k].set_ylabel('distance [Hz]')
    axs[k].set_xlabel('time [sec]')

    #put_panel_label(axs[k], k)
    fig.tight_layout() 
    fig.tight_layout()
    

def plot_single_cell(split, pid, recomp = True, curve = 'base'):

    '''
    for a given variable and insertion,
    plot trial-averaged activity per cell'
    colored by region
    '''    

    _,pa = get_allen_info()    
    
    eid, probe = one.pid2eid(pid)
    
    if recomp:
        D = get_PETHs(split, pid)   
    else:    
        s = pth_res.parent / split / f'{eid}_{probe}.npy'        
        D = np.load(s, allow_pickle=True).flat[0]
        
    acs = np.array(br.id2acronym(D['ids'], 
                                 mapping='Beryl'))
    
    fig, ax = plt.subplots(figsize=(5,5))
    
    
    xx = np.linspace(-pre_post(split)[0],
                 pre_post(split)[1],
                 len(D[curve][0])) 
                                 
    for i in range(len(acs)):
        if acs[i] in ['void', 'root']:
            continue
            
        ax.plot(xx, D[curve][i], color=pa[acs[i]])

    ax.set_title(f'single cell, (firing rate)**2, {curve}'
                 f' averaged across trials \n {split}, pid={pid}')
    ax.set_xlabel('time [sec]')
    

    regs = Counter(acs)
    
    for reg in ['root', 'void']:
        if reg in regs:
            del regs[reg]
        
    els = [Line2D([0], [0], color=pa[reg], 
           lw=4, label=f'{reg} {regs[reg]}')
           for reg in regs]               

    ax.legend(handles=els, ncols=1)
    
    

def variance_analysis(split):

    '''
    show activity distribution; one dot per cell
    2 dims being pca on PETH; 
    colored by region; by insertion; 
    '''
    
    ga = np.load(Path(pth_res, f'{split}_grand_averages.npy'),
                 allow_pickle=True).flat[0]    

    fig, ax = plt.subplots()
    _,pa = get_allen_info()
    cols = [pa[reg] for reg in ga['Beryl']]
    
    xl, yl  = 'sum' ,'sumb' #'amp', 'ampb'

    #x, y = ga[f'pcst{base}'][:, 0], ga[f'pcst{base}'][:, 1]
    # get amplitudes 
    x, y = ga[xl], ga[yl]
    

    ax.scatter(x, y, marker='o', c=cols, s=2)   
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.title(f'each point a cell, x,y is amps of PETH**2; {split}')


def plot_xyz(split):

    '''
    3d plot of feature per cell
    '''
    
    xyz, b = stack(split, per_cell=True)
    ft = np.max(b,axis=1)**3
    
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
        
#    ax.plot(xyz[:,0], xyz[:,1],xyz[:,2], marker='o', 
#            linestyle='', markersize=ft)

    ax.scatter(xyz[:,0], xyz[:,1],xyz[:,2], marker='o', s = ft)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(f'{split} max')


def plot_dim_reduction(algo='PCA', mapping='Beryl'):
    '''
    2 dims being pca on concat PETH; 
    colored by region
    '''
    
    r = np.load(Path(pth_res, 'concat.npy'),
                 allow_pickle=True).flat[0]  

    acs = np.array(br.id2acronym(r['ids'], 
                                 mapping=mapping))
                                 
    _,pa = get_allen_info()
    cols = [pa[reg] for reg in acs]

    fig, ax = plt.subplots()
    ax.scatter(r[algo][:,0], r[algo][:,1], marker='o', c=cols, s=2)
    
    ax.set_xlabel(f'{algo} dim1')
    ax.set_ylabel(f'{algo} dim2')
    ax.set_title(f'concat PETHs reduction, colors {mapping}')    
    
    
def plot_ave_PETHs():

    '''
    average PETHs across cells
    plot as lines within average trial times
    '''   
    evs = {'stimOn_times':'gray', 'firstMovement_times':'cyan',
           'feedback_times':'orange', 'intervals_1':'purple'}
           
    tts = {'stimL': blue_left,
         'stimR': red_right,
         'blockL': blue_left,
         'blockR': red_right,
         'choiceL': blue_left,
         'choiceR': red_right,
         'fback1': 'g',
         'fback0': 'k',
         'end': 'brown'}
            
    # get average temporal distances between events    
    pth_dmn = Path(one.cache_dir, 'dmn', 'mean_event_diffs.npy')
    
    if not pth_dmn.is_file():      

        eids = np.unique(bwm_query(one)['eid'])
 
        
        diffs = []
        for eid in eids:
            trials, mask = load_trials_and_mask(one, eid)    
            trials = trials[mask][:-100]
            diffs.append(np.mean(np.diff(trials[evs]),axis=0))
        
        d = {}
        d['mean'] = np.nanmean(diffs,axis=0) 
        d['std'] = np.nanstd(diffs,axis=0)
        d['diffs'] = diffs
        d['av_tr_times'] = [np.cumsum([0]+ list(x)) for x in d['diffs']]

        d['av_times'] = dict(zip(evs, 
                             zip(np.cumsum([0]+ list(d['mean'])),
                                 np.cumsum([0]+ list(d['std'])))))
        
        np.save(pth_dmn, d, allow_pickle=True)   

    d = np.load(pth_dmn, allow_pickle=True).flat[0]
    
    fig, ax = plt.subplots(figsize=(8.57, 4.8))
    r = np.load(Path(pth_res, 'concat.npy'),allow_pickle=True).flat[0]
    
    # plot trial averages
    yys = []  # to find maxes for annotation
    st = 0
    for tt in r['len']:
  
        xx = np.linspace(-pre_post(tt, con=True)[0],
                         pre_post(tt, con=True)[1],
                         r['len'][tt]) + d['av_times'][align[tt]][0]

        yy = r['mean'][st: st + r['len'][tt]]
        yys.append(max(yy))

        st += r['len'][tt]

        ax.plot(xx, yy, label=tt, color=tts[tt])
        ax.annotate(tt, (xx[-1], yy[-1]), color=tts[tt])
        
    
    for ev in d['av_times']:
        ax.axvline(x=d['av_times'][ev][0], label=ev,
                   color=evs[ev], linestyle='-')
        ax.annotate(ev, (d['av_times'][ev][0], 0.8*max(yys)), 
                    color=evs[ev], rotation=90, 
                    textcoords='offset points', xytext=(-15, 0))
    
                   
    d['av_tr_times'] = [np.cumsum([0]+ list(x)) for x in d['diffs']]               
    for s in d['av_tr_times']:
        k = 0
        for t in s:
            ax.axvline(x=t, color=evs[list(evs)[k]], 
                       linestyle='-', linewidth=0.01)
            k +=1            
    
    ax.set_xlabel('time [sec]')
    ax.set_ylabel('trial averaged fr [Hz]')
    ax.set_title('PETHs averaged across all BWM cells')
    ax.set_xlim(-0.5,3)
    fig.tight_layout()
         




