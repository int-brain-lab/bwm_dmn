import numpy as np
from collections import Counter
from pathlib import Path
from spectral_connectivity import Multitaper, Connectivity
import time
from itertools import product
from scipy.stats import norm

from brainwidemap import load_good_units, bwm_query
from iblutil.numerical import bincount2D
from one.api import ONE
from iblatlas.regions import BrainRegions
import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import gc as garbage
import matplotlib
#matplotlib.use('QtAgg')

one = ONE(base_url='https://openalyx.internationalbrainlab.org',
          password='international', silent=True)
br = BrainRegions()
T_BIN = 0.0125

# save results here
pth_res = Path(one.cache_dir, 'granger', 'res')
pth_res.mkdir(parents=True, exist_ok=True)


def get_allen_info():
    r = np.load(Path(one.cache_dir, 'dmn', 'alleninfo.npy'),
                allow_pickle=True).flat[0]
    return r['dfa'], r['palette']


def make_data(T=300000):

    x1 = np.random.normal(0, 1,T+3)
    x2 = np.random.normal(0, 1,T+3)

    for t in range(2,T+2):
        x2[t] = 0.55*x2[t-1] - 0.8*x2[t-2] + x2[t+1]
        x1[t] = 0.55*x1[t-1] - 0.8*x1[t-2] + 0.2 * x2[t-1] + x1[t+1]
    
    return np.array([x1[2:-1],x2[2:-1]])
    
    
def bin_average_neural(eid, mapping='Beryl', nmin=1):
    '''
    bin neural activity; bin, then average firing rates per region
    from both probes if available
    
    used to get session-wide time series, not cut into trials
    '''
    
    pids, probes = one.eid2pid(eid)
    
    if len(probes) == 1:
        spikes, clusters = load_good_units(one, pids[0])
        R, times, _ = bincount2D(spikes['times'], spikes['clusters'], T_BIN)
        acs = br.id2acronym(clusters['atlas_id'], mapping=mapping)
        regs = Counter(acs)
        regs2 = {x: regs[x] for x in regs if 
                 ((regs[x] >= nmin) and (x not in ['root','void']))}
                 
        R2 = np.array([np.mean(R[acs == reg],axis=0) for reg in regs2])
          
        return R2, times, regs2  
    
    else:
        sks = []
        clus = []
        for pid in pids:
            spikes, clusters = load_good_units(one, pid)    
            sks.append(spikes)
            clus.append(clusters)
    
        # add max cluster of p0 to p1, then concat, sort 
        max_cl0 = max(sks[0]['clusters'])
        sks[1]['clusters'] = sks[1]['clusters'] + max_cl0 + 1
         
        times_both = np.concatenate([sks[0]['times'],
                                     sks[1]['times']])
        clusters_both = np.concatenate([sks[0]['clusters'],
                                        sks[1]['clusters']])
                                        
        acs_both = np.concatenate([
                       br.id2acronym(clus[0]['atlas_id'],
                       mapping=mapping), 
                       br.id2acronym(clus[1]['atlas_id'],
                       mapping=mapping)])                                 
        
        t_sorted = np.sort(times_both)
        c_ordered = clusters_both[np.argsort(times_both)] 
        
        R, times, clus = bincount2D(t_sorted, c_ordered, T_BIN)  

        regs = Counter(acs_both)
        regs2 = {x: regs[x] for x in regs if 
                 ((regs[x] >= nmin) and (x not in ['root','void']))}
                 
        R2 = np.array([np.mean(R[acs_both == reg],axis=0) for reg in regs2])
          
        return R2, times, regs2    


def gc(r, segl=10):    

    '''
    chop up times series into segments of length segl [sec]
    Independent of trial-structure
    '''
    
    nchans, nobs = r.shape
    segment_length = int(segl / T_BIN)
    num_segments = nobs // segment_length
    
    # reshape into: n_signals x n_segments x n_time_samples
    r_segments = r[:, :num_segments * segment_length
                   ].reshape((nchans, num_segments, 
                   segment_length))
                   
    # reshape into:  n_time_samples x n_segments x n_signals               
    r_segments_reshaped = r_segments.transpose((2, 1, 0))


    m = Multitaper(
        r_segments_reshaped,
        sampling_frequency=1/T_BIN,
        time_halfbandwidth_product=2,
        start_time=0)
    
    c = Connectivity(
        fourier_coefficients=m.fft(), 
        frequencies=m.frequencies, 
        time=m.time)
 
    return c    


def get_gc(eid, segl=10, show_fig=False,
           metric='pairwise_spectral_granger_prediction'):

    '''
    For all regions, plot example segment time series, psg,
    matrix for max gc across freqs
    '''
    time00 = time.perf_counter()
    
    if eid == 'sim':
        r = make_data()
        regsd = {'dep':1, 'indep':1}
        ts = np.linspace(0, (r.shape[1] - 1) * T_BIN, r.shape[1])
    else:
        r, ts, regsd = bin_average_neural(eid)   
    
    c = gc(r, segl=segl)

    if not show_fig:
        return regsd, c    
           
    else:
        # freqs x chans x chans
        psg = getattr(c, metric)()[0]
        
        
        m = np.max(psg, axis=0)     
        
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(14,6))
        
        # plot example time series, first segment
        exdat = r[:,:int(segl/T_BIN)]/T_BIN
        extime = ts[:int(segl/T_BIN)]    
       
        _, pal = get_allen_info()  
        if eid == 'sim':
            pal['dep'] = 'b'
            pal['indep'] = 'r'
          
        regs = list(regsd)
         
        i = 0
        s = 0
        for y in exdat:       
            axs[0].plot(extime, y + s,c=pal[regs[i]])
            axs[0].text(extime[-1], s, regs[i], 
                        c=pal[regs[i]])
            s += np.max(y)
            i +=1
                  
        axs[0].set_xlabel('time [sec]')
        axs[0].set_ylabel('firing rate [Hz]')
        axs[0].spines['top'].set_visible(False)
        axs[0].spines['right'].set_visible(False)
        axs[0].set_title('example segment')

        # plot top five granger line plots with text next to it
        si = np.unravel_index(np.argsort(m,axis=None), m.shape)
        j = 10  # top j connections
        exes = [tup for tup in reversed(list(zip(*si))) 
                if ~np.isnan(m[tup])][:j]
        
        for tup in exes: 
            yy = psg[:,tup[0],tup[1]]
            # Order is inversed! Result is:   
            axs[1].plot(c.frequencies, yy, 
                        label =f'{regs[tup[1]]} --> {regs[tup[0]]}') 

        axs[1].legend()
        axs[1].set_xlabel('frequency [Hz]')
        axs[1].set_ylabel('directed Granger [a.u.]')   
        axs[1].set_title(f'top {j} tuples') 
        
        # plot directed granger matrix
        ims = axs[2].imshow(m, interpolation=None, cmap='gray_r', 
                      origin='lower')
                      
        # highlight max connections              
        for i, j in exes:
            rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1, 
                                      linewidth=2, 
                                      edgecolor='red', 
                                      facecolor='none')
            axs[2].add_patch(rect)              
                      
        axs[2].set_xticks(np.arange(m.shape[0]))
        axs[2].set_xticklabels(regs, rotation=90)
        axs[2].set_xlabel('source')
        axs[2].set_yticks(np.arange(m.shape[1]))
        axs[2].set_yticklabels(regs)
        axs[2].set_ylabel('target')     
        axs[2].set_title('max GC across freqs')
                      
        cb = plt.colorbar(ims,fraction=0.046, pad=0.04)              
    
    
        fig.suptitle(f'eid = {eid}')   
        fig.tight_layout()
        time11 = time.perf_counter()
        print('runtime [sec]: ', time11 - time00)                 
    

def get_all_granger(eids='all'):

    '''
    get spectral directed granger for all bwm sessions
    '''

    if isinstance(eids, str):
        df = bwm_query(one)
        eids = np.unique(df[['eid']].values)
                
    Fs = []
    k = 0
    print(f'Processing {len(eids)} sessions')
    time0 = time.perf_counter()
    for eid in eids:
               
        try:
            time00 = time.perf_counter()
            
            regsd, c = get_gc(eid)
            
            D = {'regsd': regsd,
                 'c': c,
                 'dtf': c.direct_directed_transfer_function()[0],
                 'psg': c.pairwise_spectral_granger_prediction()[0],
                 'pli': c.phase_lag_index()[0]}

                 
            np.save(Path(pth_res, f'{eid}.npy'), D, 
                    allow_pickle=True)

            garbage.collect()
            print(k + 1, 'of', len(eids), 'ok')
            time11 = time.perf_counter()
            print('runtime [sec]: ', time11 - time00)
            
        except BaseException:
            Fs.append(eid)
            garbage.collect()
            print(k + 1, 'of', len(eids), 'fail', eid)

        k += 1    

    time1 = time.perf_counter()
    print(time1 - time0, f'sec for {len(eids)} sessions')
    print(len(Fs), 'failures')
    return Fs



def plot_res(nmin=10, sessmin = 4):

    '''
    Group and plot granger results
    
    nmin: minimum number of neurons per region to be included
    sessmin: min number of sessions with region combi
    '''
    
    p = pth_res.glob('**/*')
    files = [x for x in p if x.is_file()]
    
    d = {}
    nd = []
    for sess in files:    
        D = np.load(sess, allow_pickle=True).flat[0]
        m = np.max(D['psg'], axis=0)
        regs = list(D['regsd'])
        
        if not isinstance(D['regsd'], dict):
            nd.append(sess)
            continue
        
        
        for i in range(len(regs)):
            for j in range(len(regs)):
                if i == j:
                    continue
                
                if ((D['regsd'][regs[i]] < nmin) or
                    (D['regsd'][regs[j]] < nmin)):
                    continue
                
                if f'{regs[i]} --> {regs[j]}' in d:
                    d[f'{regs[i]} --> {regs[j]}'].append(m[j, i])
                else:
                    d[f'{regs[i]} --> {regs[j]}'] = []
                    d[f'{regs[i]} --> {regs[j]}'].append(m[j, i])    


    dm = {x: np.mean(d[x]) for x in d if len(d[x]) > sessmin}
    dm_sorted = dict(sorted(dm.items(), key=lambda item: item[1]))

    exs = np.concatenate([list(dm_sorted.keys())[:10],
                          list(dm_sorted.keys())[-10:]])
    
    d_exs = {x:d[x] for x in exs}

    fig, ax = plt.subplots(figsize=(5,4))
    sns.stripplot(data=d_exs, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_title(f'10 min and 10 max directed region pairs \n'
                 f' with at least {sessmin} sessions out of all'
                 f' {len(files)} sessions \n'
                 f'only regs with at least {nmin} neurons'
                 f' \n one dot per session')
    ax.set_ylabel('max directed spectral Granger across freqs')
    fig.tight_layout()

    return nd















  
