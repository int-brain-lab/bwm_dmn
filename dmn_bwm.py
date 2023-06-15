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
from sklearn.decomposition import PCA
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

from matplotlib.axis import Axis
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import ListedColormap, LinearSegmentedColormap   
from matplotlib.gridspec import GridSpec   
from statsmodels.stats.multitest import multipletests
from matplotlib.lines import Line2D


'''
script to process the BWM dataset for manifold analysis,
saving intermediate results (PETHs), computing
metrics and plotting them (plot_all, also supp figures)

A split is a variable, such as stim, where the trials are
disected by it - e.g. left stim side and right stim side

To compute all from scratch, including data download, run:

##################
for split in ['choice', 'stim','fback','block']:
    get_all_d_vars(split)  # computes PETHs, distance sums
    d_var_stacked(split)  # combine results across insertions
    
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
         'choice': 'firstMovement_times',
         'fback1': 'feedback_times',
         'fback0': 'feedback_times',
         'block': 'stimOn_times',
         'end': 'intervals_1'}


def pre_post(split, can=False):
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
    pre_post_can =  {'stim': [0, 0.1],
                     'choice': [0.1, 0],
                     'fback1': [0, 0.2],
                     'fback0': [0, 0.7],
                     'block': [0.4, -0.1],
                     'end': [0,0.5]}

    pp = pre_post_can if can else pre_post0

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
        get_all_PETH(split, eids_plus=h)


    print('POST CORRECTION')
    for split in splits:
        pth = Path(one.cache_dir, 'dmn', split)
        ss = os.listdir(pth)
        print(split, len(ss))



def stack(split, min_reg=20, mapping='Beryl'):

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
    ws = []
    base = []

    # group results across insertions
    for s in ss:
                   
        D_ = np.load(Path(pth, s),
                     allow_pickle=True).flat[0]

        ids.append(D_['ids'])
        ws.append(D_['ws'])
        base.append(D_['base'])

    ids = np.concatenate(ids)
    ws = np.concatenate(ws, axis=0)
    base = np.concatenate(base, axis=0)

    acs = np.array(br.id2acronym(ids, mapping=mapping))

    # discard ill-defined regions
    goodcells = ~np.bitwise_or.reduce([acs == reg for
                                       reg in ['void', 'root']])

    acs = acs[goodcells]
    ws = ws[goodcells] 
    base = base[goodcells]
    
    print('computing grand average metrics ...')
    ncells, nt = ws.shape

    ga = {}
    ga['m0'] = np.mean(ws, axis=0)
    ga['ms'] = np.mean(base, axis=0)

    ga['v0'] = np.std(ws, axis=0) / (ncells**0.5)
    ga['vs'] = np.std(base, axis=0) / (ncells**0.5)

    ga['nclus'] = ncells

    # first is window, second is base window
    pca = PCA(n_components=3)
    wsc = pca.fit_transform(np.concatenate([ws, base],
                                            axis=1).T).T
    ga['pcs'] = wsc
    
    # differences of window and base, pooling all cells

    ga['d_euc'] = np.mean(ws, axis=0)**0.5                    
    ga['d_eucb'] = np.mean(base, axis=0)**0.5                           

    np.save(Path(pth_res, f'{split}_grand_averages.npy'), ga,
            allow_pickle=True)

    print('computing regional metrics ...')  
    
    regs0 = Counter(acs)
    regs = {reg: regs0[reg] for reg in regs0 if regs0[reg] > min_reg}
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

        '''
        euc
        '''
        
        v = datb.mean(axis=1)
        res['d_euc'] = np.mean(dat, axis=0)**0.5
        res['d_eucb'] = np.mean(datb, axis=0)**0.5 
            
        # amplitude
        res['amp_euc'] = max(res['d_euc'])
        res['amp_eucb'] = max(res['d_eucb'])
        
        # latency, must be significant
        loc = np.where(res['d_euc'] > 0.7 * res['amp_euc'])[0]
        res['lat_euc'] = np.linspace(-pre_post(split)[0],
                                     pre_post(split)[1],
                                     len(res['d_euc']))[loc[0]]
                                     
        loc = np.where(res['d_eucb'] > 0.7 * res['amp_eucb'])[0]
        res['lat_eucb'] = np.linspace(-pre_post(split)[0],
                                     pre_post(split)[1],
                                     len(res['d_eucb']))[loc[0]]
                                     
        r[reg] = res
        

    np.save(Path(pth_res, f'{split}.npy'),
            r, allow_pickle=True)

    time1 = time.perf_counter()
    print('total time:', np.round(time1 - time0, 0), 'sec')


'''
#####################################################
### plotting
#####################################################
'''


def get_allen_info():
    '''
    Function to load Allen atlas info, like region colors
    '''

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

    return dfa, palette


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


    if not show_tra:
        fsize = 12 # font size
        dsize = 13  # diamond marker size
        lw = 1  # linewidth        

    else:
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
    cosregs_ = [
        dfa[dfa['id'] == int(dfa[dfa['acronym'] == reg][
            'structure_id_path'].values[0].split('/')[4])][
            'acronym'].values[0] for reg in regsa]

    cosregs = dict(zip(regsa, cosregs_))

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
          'block': ["#ffffff","#D0CDE4","#998DC3",
                    "#6159A6","#42328E"]}

    if '_' in split:
        split = split.split('_')[0]

    return LinearSegmentedColormap.from_list("mycmap", dc[split])
   
    
def plot_swanson_supp(splits = None, curve = 'euc',
                      show_legend = False, bina=False):
 
    '''
    swanson maps for maxes
    '''
    
    if splits is None:
        splits0 = ['stim', 'choice', 'fback','block']
        splits = [x+'_restr' for x in splits0]
    
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
   
    '''
    max dist_split onto swanson flat maps
    (only regs with p < sigl)
    '''
    
    k = 0  # panel counter
    c = 0  # column counter
    
    sws = []
    for split in splits:

        d = np.load(Path(pth_res, f'{split}.npy'),
                    allow_pickle=True).flat[0]
                    
        # get significant regions only
        acronyms = [reg for reg in d
                if d[reg][f'p_{curve}'] < sigl]


        # plot amplitudes
        if bina:
            # set all significant regions to 1, others zero
            amps = np.array([1 for x in acronyms])
        
        else:
            amps = np.array([d[x][f'amp_{curve}_can'] for x in acronyms])
            
        plot_swanson_vector(np.array(acronyms), np.array(amps), 
                            cmap=get_cmap(split), 
                            ax=axs[0,c], br=br, 
                            orientation='portrait',
                            linewidth=0.1)
                            
        # add colorbar
        clevels = (np.nanmin(amps), np.nanmax(amps))
        norm = mpl.colors.Normalize(vmin=clevels[0], 
                                    vmax=clevels[1])
                                    
        cbar = fig.colorbar(mpl.cm.ScalarMappable(
                                norm=norm, 
                                cmap=get_cmap(split)), 
                                ax=axs[0,c])
    
        cbar.set_label('effect size [spikes/second]')

                            
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
    
    





