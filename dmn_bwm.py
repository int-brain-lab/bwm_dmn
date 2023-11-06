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

from scipy.spatial import distance
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
import umap, trimap
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

import warnings
warnings.filterwarnings("ignore")


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
          
br = BrainRegions()
#units_df = bwm_units(one)  # canonical set of cells


# save results here
pth_res = Path(one.cache_dir, 'dmn', 'res')
pth_res.mkdir(parents=True, exist_ok=True)

sigl = 0.05  # significance level (for stacking, plotting, fdr)


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
    
    return Counter(units_df[units_df['eid'] == eid]['pid'])  


def concat_PETHs(pid, get_tts=False):

    '''
    for each cell concat all possible PETHs
    '''
    
    eid, probe = one.pid2eid(pid)



    # Load in trials data and mask bad trials (False if bad)
    trials, mask = load_trials_and_mask(one, eid)     
        
#    # define align, trial type, window length   
#    tts0 = {
#        'stimL': ['stimOn_times', ~np.isnan(trials[f'contrastLeft']), [0, 0.15]],
#        'stimR': ['stimOn_times', ~np.isnan(trials[f'contrastRight']), [0, 0.15]],
#        'blockL': ['stimOn_times', trials['probabilityLeft'] == 0.8, [0.4, -0.1]],
#        'blockR': ['stimOn_times', trials['probabilityLeft'] == 0.2, [0.4, -0.1]],
#        'choiceL': ['firstMovement_times', trials['choice'] == 1, [0.15, 0]],
#        'choiceR': ['firstMovement_times', trials['choice'] == -1, [0.15, 0]],
#        'fback1': ['feedback_times', trials['feedbackType'] == 1, [0, 0.3]],
#        'fback0': ['feedback_times', trials['feedbackType'] == -1, [0, 0.3]],
#        'end': ['intervals_1', np.full(len(trials['choice']), True), [0, 0.3]]}


    # define align, trial type, window length
    # including Ari splits
    tts = {
        'blockL': ['stimOn_times', 
                   trials['probabilityLeft'] == 0.8, [0.4, -0.1]],
        'blockR': ['stimOn_times', 
                   trials['probabilityLeft'] == 0.2, [0.4, -0.1]],
        
        'stimLbLcL': ['stimOn_times',
             np.bitwise_and.reduce([~np.isnan(trials[f'contrastLeft']), 
                                    trials['probabilityLeft'] == 0.8,
                                    trials['choice'] == 1]), 
                                    [0, 0.15]], 
        'stimLbRcL': ['stimOn_times',
             np.bitwise_and.reduce([~np.isnan(trials[f'contrastLeft']), 
                                    trials['probabilityLeft'] == 0.2,
                                    trials['choice'] == 1]), 
                                    [0, 0.15]],
        'stimLbRcR': ['stimOn_times',
             np.bitwise_and.reduce([~np.isnan(trials[f'contrastLeft']), 
                                    trials['probabilityLeft'] == 0.2,
                                    trials['choice'] == -1]), 
                                    [0, 0.15]],           
        'stimLbLcR': ['stimOn_times',
             np.bitwise_and.reduce([~np.isnan(trials[f'contrastLeft']), 
                                    trials['probabilityLeft'] == 0.8,
                                    trials['choice'] == -1]), 
                                    [0, 0.15]],
        'stimRbLcL': ['stimOn_times',
             np.bitwise_and.reduce([~np.isnan(trials[f'contrastRight']), 
                                    trials['probabilityLeft'] == 0.8,
                                    trials['choice'] == 1]), 
                                    [0, 0.15]], 
        'stimRbRcL': ['stimOn_times',
             np.bitwise_and.reduce([~np.isnan(trials[f'contrastRight']), 
                                    trials['probabilityLeft'] == 0.2,
                                    trials['choice'] == 1]), 
                                    [0, 0.15]],
        'stimRbRcR': ['stimOn_times',
             np.bitwise_and.reduce([~np.isnan(trials[f'contrastRight']), 
                                    trials['probabilityLeft'] == 0.2,
                                    trials['choice'] == -1]), 
                                    [0, 0.15]],        
        'stimRbLcR': ['stimOn_times',
             np.bitwise_and.reduce([~np.isnan(trials[f'contrastRight']), 
                                    trials['probabilityLeft'] == 0.8,
                                    trials['choice'] == -1]), 
                                    [0, 0.15]],
        'sLbLchoiceL': ['firstMovement_times',
             np.bitwise_and.reduce([~np.isnan(trials[f'contrastLeft']), 
                                    trials['probabilityLeft'] == 0.8,
                                    trials['choice'] == 1]), 
                                    [0.15, 0]], 
        'sLbRchoiceL': ['firstMovement_times',
             np.bitwise_and.reduce([~np.isnan(trials[f'contrastLeft']), 
                                    trials['probabilityLeft'] == 0.2,
                                    trials['choice'] == 1]), 
                                    [0.15, 0]],
        'sLbRchoiceR': ['firstMovement_times',
             np.bitwise_and.reduce([~np.isnan(trials[f'contrastLeft']), 
                                    trials['probabilityLeft'] == 0.2,
                                    trials['choice'] == -1]), 
                                    [0.15, 0]],           
        'sLbLchoiceR': ['firstMovement_times',
             np.bitwise_and.reduce([~np.isnan(trials[f'contrastLeft']), 
                                    trials['probabilityLeft'] == 0.8,
                                    trials['choice'] == -1]), 
                                    [0.15, 0]],
        'sRbLchoiceL': ['firstMovement_times',
             np.bitwise_and.reduce([~np.isnan(trials[f'contrastRight']), 
                                    trials['probabilityLeft'] == 0.8,
                                    trials['choice'] == 1]), 
                                    [0.15, 0]], 
        'sRbRchoiceL': ['firstMovement_times',
             np.bitwise_and.reduce([~np.isnan(trials[f'contrastRight']), 
                                    trials['probabilityLeft'] == 0.2,
                                    trials['choice'] == 1]), 
                                    [0.15, 0]],
        'sRbRchoiceR': ['firstMovement_times',
             np.bitwise_and.reduce([~np.isnan(trials[f'contrastRight']), 
                                    trials['probabilityLeft'] == 0.2,
                                    trials['choice'] == -1]), 
                                    [0.15, 0]],        
        'sRbLchoiceR': ['firstMovement_times',
             np.bitwise_and.reduce([~np.isnan(trials[f'contrastRight']), 
                                    trials['probabilityLeft'] == 0.8,
                                    trials['choice'] == -1]), 
                                    [0.15, 0]],        
        
        'choiceL': ['firstMovement_times', trials['choice'] == 1, 
                    [0, 0.15]],
        'choiceR': ['firstMovement_times', trials['choice'] == -1, 
                    [0, 0.15]],       

        'fback1': ['feedback_times', trials['feedbackType'] == 1, 
                   [0, 0.3]],
        'fback0': ['feedback_times', trials['feedbackType'] == -1, 
                   [0, 0.3]],
        'end': ['intervals_1', np.full(len(trials['choice']), True), 
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
                    local_path=LOCAL_DATA_PATH, one=one)                    
                    
     merged_df0 = D['df_raw_features'].merge(
                  D['df_channels'], 
                    on=['pid','channel'])               
                    
     merged_df = merged_df0.merge(
                    D['df_clusters'], 
                    on=['pid', 'axial_um', 'lateral_um'])                     
                    
    return merged_df       

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
        bc = ['b', 'g', 'r', 'c', 'm', 'y', 'brown', 'pink']
        for i in range(7):
            palette[str(i)] = bc[i]
        
        palette['thal'] = 'k'    
        r = {}
        r['dfa'] = dfa
        r['palette'] = palette    
        np.save(pth_dmn, r, allow_pickle=True)   

    r = np.load(pth_dmn, allow_pickle=True).flat[0]
    return r['dfa'], r['palette']  


def regional_group(mapping, algo, EAtlas=False):

    '''
    find group labels for all cells
    mapping: [Allen, Beryl, Cosmos, layers, clusters, clusters_xyz]
    '''

    r = np.load(Path(pth_res, 'concat.npy'),
                 allow_pickle=True).flat[0]               
    # add point names to dict
    r['nums'] = range(len(r[algo][:,0]))
                       
    if EAtlas:
        #  include ephys atlas info
        df = pd.DataFrame({'uuids':r['uuids']})
        merged_df = load_atlas_data()
        dfm = df.merge(merged_df, on=['uuids'])
        
        # remove cells that have no ephys info
        dfr = set(r['uuids']).difference(set(dfm['uuids']))
        rmv = [True if u in dfr else False for u in r['uuids']]

        for key in r:
            if type(r[key]) == np.ndarray and len(r[key]) == len(r[algo]):
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
        'rms_ap', 'rms_lf', 'rms_lf_csd', 'spike_count', 'spike_count_x', 
        'spike_count_y', 'tip_time_secs', 'tip_val', 
        'trough_time_secs', 
        'trough_val']
        
        # for u in 
        d2 = [
        
        
        
         
                         

    if mapping == 'clusters':
        # use clusters from hierarchical clustering to color
        nclus = 1000
        clusters = fcluster(r['linked'], t=nclus, criterion='maxclust')
        cmap = mpl.cm.get_cmap('Spectral')
        cols = cmap(clusters/nclus)
        acs = clusters
        # get average point and color per region
        av = {reg: [np.mean(r[algo][clusters == clus], axis=0), 
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
        
            zeros = np.arange(len(acs))[acs == '0']
            for key in r:
                if type(r[key]) == np.ndarray and len(r[key]) == len(acs):
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


def decode(mapping='Beryl', minreg=20, decoder='LDA', z_sco=True,
           n_runs = 1, confusion=False):
           
    print(mapping, f', minreg: {minreg},', decoder, 'z_score', z_sco)
        
    r = np.load(Path(pth_res, 'concat.npy'),
                allow_pickle=True).flat[0]

    if mapping == 'layers':
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
        
            zeros = np.arange(len(acs))[acs == '0']
            for key in r:
                if type(r[key]) == np.ndarray and len(r[key]) == len(acs):
                    r[key] = np.delete(r[key], zeros, axis=0)
                       
            acs = np.delete(acs, zeros)
    
    else:
        acs = np.array(br.id2acronym(r['ids'], 
                                     mapping=mapping))                         
        
    
    # get average points and color per region
    regs = Counter(acs)
    
    x = r['concat_z' if z_sco else 'concat']
    y = acs

    # restrict to regions with minreg cells
    regs2 = [reg for reg in regs if regs[reg]>minreg]
    mask = [True if ac in regs2 else False for ac in acs]

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


def stack_concat(get_concat=False):

    split = 'concat'
    pth = Path(one.cache_dir, 'dmn', split)
    ss = os.listdir(pth)  # get insertions
    print(f'combining {len(ss)} insertions for split {split}') 

    # pool data
    
    r = {}
    for ke in ['ids', 'xyz', 'uuids']:
        r[ke] = []   

    ws = []
    # group results across insertions
    for s in ss:
                   
        D_ = np.load(Path(pth, s),
                     allow_pickle=True).flat[0]

        ws.append(np.concatenate(D_['ws'],axis=1))
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
    r['len'] = dict(zip(D_['trial_names'],
                    [x.shape[1] for x in D_['ws']]))
    

    # hierarchical clustering on PETHs
    r['linked'] = linkage(cs, 'ward')   
    
    # hierarchical clustering on xys dist in histology space
    r['linked_xyz'] = linkage(r['xyz'], 'ward')
    
    # various dim reduction of PETHs to 2 dims
    ncomp = 2
    r['umap'] = umap.UMAP(n_components=ncomp).fit_transform(cs)
    r['umap_z'] = umap.UMAP(n_components=ncomp).fit_transform(cs_z)
    r['tSNE'] = TSNE(n_components=ncomp).fit_transform(cs)
    r['tSNE_z'] = TSNE(n_components=ncomp).fit_transform(cs_z)
    r['PCA'] = PCA(n_components=ncomp).fit_transform(cs)
    r['PCA_z'] = PCA(n_components=ncomp).fit_transform(cs_z)
#    r['ICA'] = FastICA(n_components=ncomp).fit_transform(cs) 
#    r['trimap'] = trimap.TRIMAP(n_dims=ncomp).fit_transform(cs)


    np.save(Path(pth_res, 'concat.npy'),
            r, allow_pickle=True)            



    

'''
#####################################################
### plotting
#####################################################
'''
        

def plot_dim_reduction(algo='umap_z', mapping='layers', 
                       means=False, exa=False, shuf=False,
                       exa_squ=False):
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
    '''
    
    feat = 'concat_z' if algo[-1] == 'z' else 'concat'
    
    r = regional_group(mapping, algo)

    fig, ax = plt.subplots()
    if shuf:
        shuffle(r['cols'])
    
    im = ax.scatter(r[algo][:,0], r[algo][:,1], marker='o', c=r['cols'], s=2)
    
    if means:
        # show means
        emb1 = [r['av'][reg][0][0] for reg in r['av']] 
        emb2 = [r['av'][reg][0][1] for reg in r['av']]
        cs = [r['av'][reg][1] for reg in r['av']]
        ax.scatter(emb1, emb2, marker='o', facecolors='none', 
                   edgecolors=cs, s=600, linewidths=4)
    
    ax.set_xlabel(f'{algo} dim1')
    ax.set_ylabel(f'{algo} dim2')
    ss = 'shuf' if shuf else ''
    ax.set_title(f'concat PETHs reduction, colors {mapping} {ss}')    
    
    if mapping == 'layers':
        ax.legend(handles=r['els'], ncols=1).set_draggable(True)

    elif 'clusters' in mapping:
        cax = fig.add_axes([0.27, 0.2, 0.5, 0.01])
        norm = mpl.colors.Normalize(vmin=0, 
                                    vmax=nclus)
                                    
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

                    line.set_data(b_size *np.arange(len(r[feat][key])),
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
                axx.plot(b_size * np.arange(len(r[feat][pt])),
                         r[feat][pt],color=r['cols'][pt], linewidth=0.5)
                maxys.append(np.max(r[feat][pt]))         
                         
                
            #square mean
            axx.plot(b_size * np.arange(len(r[feat][pt])),
                     np.mean(r[feat][pts],axis=0),
                color='k', linewidth=2)    

            axx.set_title(f'{s} \n {len(pts)} points in square')
            axx.set_xlabel('time [sec]')
            axx.set_ylabel(feat)
            
            # plot vertical boundaries for windows
            h = 0
            for i in r['len']:
            
                xv = r['len'][i] + h
                axx.axvline(b_size * xv, linestyle='--',
                            color='grey')
                            
                axx.text(b_size * xv, 0.8 * np.max(maxys), 
                         i, rotation=90, 
                         fontsize=12, color='k')
            
                h += r['len'][i]




def smooth_dist(algo='umap_z', mapping='layers', show_imgs=True,
                norm_=True, dendro=True):

    '''
    smooth 2d pointclouds, show per class
    norm_: normalize smoothed image by max brightness
    '''

    r = regional_group(mapping, algo)
    feat = 'concat_z' if algo[-1] == 'z' else 'concat'
    
    
    # Define grid size and density kernel size
    x_min = np.floor(np.min(r[algo][:,0]))
    x_max = np.ceil(np.max(r[algo][:,0]))
    y_min = np.floor(np.min(r[algo][:,1]))
    y_max = np.ceil(np.max(r[algo][:,1]))
    
    imgs = {}
    xys = {}
    
    regs = Counter(r['acs'])
    _,pa = get_allen_info()
    regcol = {reg: pa[reg] for reg in regs}    

    for reg in regcol:
    
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
        fig, axs = plt.subplots(nrows=3, ncols=len(regs))        
        axs = axs.flatten()    
        #[ax.set_axis_off() for ax in axs]

        vmin = np.min([np.min(imgs[reg].flatten()) for reg in imgs])
        vmax = np.max([np.max(imgs[reg].flatten()) for reg in imgs])
        
        k = 0 
        # row of panels showing smoothed point clouds
        for reg in imgs:
            axs[k].imshow(imgs[reg], origin='lower', vmin=vmin, vmax=vmax)
            axs[k].set_title(f'{reg}, ({regs[reg]})')
            axs[k].set_axis_off()
            k+=1 
            
        # row of images showing point clouds     
        for reg in imgs:
            axs[k].scatter(xys[reg][0], xys[reg][1], color=regcol[reg], s=2)
            axs[k].set_title(f'{reg}, ({regs[reg]})')
            axs[k].set_axis_off()
            k+=1               
            
        # row of images showing mean feature vector
        for reg in imgs:
            pts = np.arange(len(r['acs']))[r['acs'] == reg]
            
            xss = b_size * np.arange(len(np.mean(r[feat][pts],axis=0)))
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
                axs[k].axvline(b_size * xv, linestyle='--',
                            color='grey', linewidth=0.1)
                            
                axs[k].text(b_size * xv, 0.8 * np.max(maxys), 
                         i, rotation=90, 
                         fontsize=5, color='k')
            
                h += r['len'][i]
            
            k+=1
            
        
        fig.suptitle(f'algo: {algo}, mapping: {mapping}, norm:{norm_}')
        fig.tight_layout()    

    # show cosine similarity of density vectors
    
    def cosine_sim(v0, v1):
        # cosine similarity 
        return np.inner(v0,v1)/ (norm(v0) * norm(v1))
    
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

    fig0, ax0 = plt.subplots(figsize=(4,4))                     
    ims = ax0.imshow(res, origin='lower')
    ax0.set_xticks(np.arange(len(regs)), list(imgs.keys()))
    ax0.set_yticks(np.arange(len(regs)), list(imgs.keys()))
    ax0.set_title(f'cosine similarity of smooth images, norm:{norm_}')
    ax0.set_ylabel(mapping)
    plt.colorbar(ims,fraction=0.046, pad=0.04)
    fig0.tight_layout()
    
    # dendrogram of smoothed image vectors using cosine similarity
    fig1, ax1 = plt.subplots(figsize=(4,4))
    
    data = [imgs[reg].flatten() for reg in imgs]
    cl = linkage(data, method='single', metric=cosine_sim)
    dendrogram(cl, 
               labels=[reg for reg in imgs], 
               ax = ax1)    

    ax1.set_title(f'cosine similarity of smooth images, norm:{norm_}')
    ax1.set_xlabel(mapping)
    fig1.tight_layout()    
    
    
    
    
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
   

def plot_dec_confusion(mapping='Beryl', minreg=20, decoder='LDA', z_sco=True,
           n_runs = 1):
           
    '''
    For train and test, plot a confusion matrix
    '''       

    cm_train, cm_test, regs, r_train, r_test  = decode(
        mapping=mapping, minreg=minreg, 
        decoder=decoder, z_sco=z_sco,
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
    
    fig.suptitle(f'z-scored:{z_sco}; '
                 f'ac. train:{r_train}, ac. test:{r_test}')   
    fig.tight_layout()

    return cms                                         


def plot_ave_PETHs(feat = 'concat'):

    '''
    average PETHs across cells
    plot as lines within average trial times
    '''   
    evs = {'stimOn_times':'gray', 'firstMovement_times':'cyan',
           'feedback_times':'orange', 'intervals_1':'purple'}
           
    win_cols = {'stimL': blue_left,
         'stimR': red_right,
         'blockL': blue_left,
         'blockR': red_right,
         'choiceL': blue_left,
         'choiceR': red_right,
         'fback1': 'g',
         'fback0': 'k',
         'end': 'brown',
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
        if ('stim' in win) or ('block' in win):
            return 'stimOn_times'
        elif 'choice' in win:
            return 'firstMovement_times'
        elif 'fback' in win:
            return 'feedback_times'    
        elif 'end' in win:
            return 'intervals_1'


    def pre_post(win):
        '''
        [pre_time, post_time] relative to alignment event
        split could be contr or restr variant, then
        use base window
        '''

        pid = '1a60a6e1-da99-4d4e-a734-39b1d4544fad'
        tts = concat_PETHs(pid, get_tts=True)
        
        return tts[win][2]


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





