#import os
#os.environ["SPECTRAL_CONNECTIVITY_ENABLE_GPU"] = "true"


import numpy as np
from collections import Counter
from pathlib import Path
from spectral_connectivity import Multitaper, Connectivity
import time
from itertools import combinations
from scipy.stats import norm
import pandas as pd
import umap
from copy import copy
#from sklearn.manifold import MDS
from scipy.stats import pearsonr, spearmanr

from brainwidemap import load_good_units, bwm_query
from iblutil.numerical import bincount2D
from one.api import ONE
from iblatlas.regions import BrainRegions
from iblatlas.atlas import AllenAtlas


import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import gc as garbage
import matplotlib
#matplotlib.use('QtAgg')
#matplotlib.use('tkagg')
sns.reset_defaults()
plt.ion()

one = ONE(base_url='https://openalyx.internationalbrainlab.org',
          password='international', silent=True)
          
ba = AllenAtlas()          
br = BrainRegions()

T_BIN = 0.0125  # 0.005

# save results here
pth_res = Path(one.cache_dir, 'granger', 'res')
pth_res.mkdir(parents=True, exist_ok=True)


def get_allen_info():
    r = np.load(Path(one.cache_dir, 'dmn', 'alleninfo.npy'),
                allow_pickle=True).flat[0]
    return r['dfa'], r['palette']


def get_structural(rerun=False):

    '''
    load structural connectivity matrix
    https://static-content.springer.com
    /esm/art%3A10.1038%2Fnature13186/MediaObjects/
    41586_2014_BFnature13186_MOESM70_ESM.xlsx
    '''
    
    pth_ = Path(one.cache_dir, 'granger', 'structural.npy')
    if (not pth_.is_file() or rerun):
        s=pd.read_excel('/home/mic/fig3.xlsx')
        cols = list(s.keys())[1:296]
        rows = s['Unnamed: 0'].array
        
        M = np.zeros((len(cols), len(rows)))
        for i in range(len(cols)):
            M[i] = s[cols[i]].array
            
        M = M.T

        # thresholding as in the paper
        M[M > 10**(-0.5)] = 1
        M[M < 10**(-3.5)] = 0

        cols1 = np.array([reg.strip().replace(",", "") for reg in cols])
        rows1 = np.array([reg.strip().replace(",", "") for reg in rows])
        
        # average across injections
        regsr = list(Counter(rows1))
        M2 = []
        for reg in regsr:
            M2.append(np.mean(M[rows1 == reg], axis=0))       

        M2 = np.array(M2)
        regs_source = regsr
        regs_target = cols1

        # turn into dict
        d0 = {}
        for i in range(len(regs_source)):
            for j in range(len(regs_target)):

                d0[' --> '.join([regs_source[i], 
                                 regs_target[j]])] = M2[i,j]

        np.save(pth_, d0,
                allow_pickle=True)
                
    else:
        d = np.load(pth_, allow_pickle=True).flat[0]        
        
    return d                                


def get_centroids(rerun=False):

    '''
    Beryl region centroids xyz
    '''
    
    pth_ = Path(one.cache_dir, 'granger', 'beryl_centroid.npy')
    if (not pth_.is_file() or rerun):
        beryl_vol = ba.regions.mappings['Beryl-lr'][ba.label]
        beryl_idx = np.unique(ba.regions.mappings['Beryl-lr'])

        d = {}
        k = 0
        for ridx in beryl_idx:
            idx = np.where(beryl_vol == ridx)
            ixiyiz = np.c_[idx[1], idx[0], idx[2]]
            xyz = ba.bc.i2xyz(ixiyiz)   
            d[br.index2acronym(ridx,
                mapping='Beryl')] = np.mean(xyz, axis=0)
            print(br.index2acronym(ridx,mapping='Beryl'),k, 'of',
                  len(beryl_idx))  
            k+=1

        np.save(pth_, d,
                allow_pickle=True)
                
    else:
        d = np.load(pth_, allow_pickle=True).flat[0]        
        
    return d
            

def get_volume(rerun=False):

    '''
    Beryl region volumina in mm^3
    '''

    pth_ = Path(one.cache_dir, 'granger', 'beryl_volumina.npy')
    if (not pth_.is_file() or rerun):
        ba.compute_regions_volume()  
        acs = np.unique(br.id2acronym(
                ba.regions.id, mapping='Beryl'))  
         
        d2 = {}  
        for ac in acs:
            d2[ac] = ba.regions.volume[
                        ba.regions.acronym2index(
                            ac, mapping='Beryl')[1]].sum()
                            
        np.save(pth_, d2,
                allow_pickle=True)                        
                          
    else:
        d2 = np.load(pth_, allow_pickle=True).flat[0]        
        
    return d2



def make_data(T=300000, vers='oscil', peak_freq_factor0=0.55,
              peak_freq_factor1=0.6, phase_lag_factor=0.2):
    
    '''
    auto-regressive data creation
    x2 dependend on x1, not vice versa
    '''
    
    x1 = np.random.normal(0, 1,T+3)
    x2 = np.random.normal(0, 1,T+3)

    if vers == 'dc': 
        x2 = x1 + 0.7* np.random.normal(0,1,T+3)     
    
    elif vers == 'oscil':
        for t in range(2,T+2):
            x2[t] = (peak_freq_factor1*x2[t-1] - 0.8*x2[t-2] + 
                     x2[t+1])
            x1[t] = (peak_freq_factor0*x1[t-1] - 0.8*x1[t-2] + 
                     phase_lag_factor * x2[t-1] + x1[t+1])
            
    elif vers == 'loopy':
        for t in range(2,T+2):
            x2[t] = (peak_freq_factor1 * x2[t - 1] - 0.8 * x2[t - 2] 
                     + phase_lag_factor * x1[t - 1] + x2[t + 1])
            x1[t] = (peak_freq_factor0 * x1[t - 1] - 0.8 * x1[t - 2] 
                     + phase_lag_factor * x2[t - 1] + x1[t + 1])        
    
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
        R, times, _ = bincount2D(spikes['times'], 
                        spikes['clusters'], T_BIN)
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
                 
        R2 = np.array([np.mean(R[acs_both == reg],axis=0) 
                       for reg in regs2])
          
        return R2, times, regs2    


def gc(r, segl=10, shuf=False, shuf_type = 'reg_shuffle'):    

    '''
    chop up times series into segments of length segl [sec]
    Independent of trial-structure, then compute metrics
    '''
    
    nchans, nobs = r.shape
    segment_length = int(segl / T_BIN)
    num_segments = nobs // segment_length
    
    # reshape into: n_signals x n_segments x n_time_samples
    r_segments = r[:, :num_segments * segment_length
                   ].reshape((nchans, num_segments, 
                   segment_length))

    if shuf:
        

        if shuf_type == 'reg_shuffle':
            # shuffle region order per trial        
            indices = np.arange(r_segments.shape[0])
            
            rs = np.zeros(r_segments.shape)
            for trial in range(r_segments.shape[1]):
                np.random.shuffle(indices)    
                rs[:,trial,:] = r_segments[indices, trial, :]
                
            r_segments = np.array(rs)    

        else:
            # shuffle segment order
            indices = np.arange(r_segments.shape[1])
            
            rs = np.zeros(r_segments.shape)
            for chan in range(r_segments.shape[0]):
                np.random.shuffle(indices)    
                rs[chan] = r_segments[chan, indices]
                
            r_segments = np.array(rs)    
            #print('segments channel-independently shuffled')
            
                   
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



def get_all_granger(eids='all', nshufs = 3, segl=10, nmin=10):

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
        print('eid:', eid)           
        try:
            time00 = time.perf_counter()
            
            r, ts, regsd = bin_average_neural(eid, nmin=nmin)
            if not bool(regsd):
                print(f'no data for {eid}') 
                continue
            
            nchans, nobs = r.shape
            segment_length = int(segl / T_BIN)
            num_segments = nobs // segment_length
            
            # reshape into: n_signals x n_segments x n_time_samples
            r_segments = r[:, :num_segments * segment_length
                           ].reshape((nchans, num_segments, 
                           segment_length))
                                       
            # reshape to n_time_samples x n_segments x n_signals               
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
            

            psg = c.pairwise_spectral_granger_prediction()[0]
            coh = c.coherence_magnitude()[0]
            score_g = np.mean(psg,axis=0)
            score_c = np.mean(coh,axis=0)
            
            # get scores after shuffling segments
            shuf_g = []
            shuf_c = []
            
            # shuffle pairs of regions separately
            pairs = np.array(list(combinations(range(nchans),2)))
                        
            for i in range(nshufs):
                if i%1000 == 0:
                    print('shuf', i, f'({nshufs})')
                
                mg = np.zeros([nchans,nchans])
                mc = np.zeros([nchans,nchans])
                
                for pair in pairs:                                      
                    rs = np.zeros([2, r_segments.shape[1],
                                      r_segments.shape[2]])
                                      
                    for trial in range(r_segments.shape[1]):
                        np.random.shuffle(pair)    
                        rs[:,trial,:] = r_segments[pair, trial, :]
                        
                    r_segments0 = np.array(rs)
                    
                    #into n_time_samples x n_segments x n_signals               
                    r_segments_reshaped0 = r_segments0.transpose((2, 1, 0))

                    m = Multitaper(
                        r_segments_reshaped0,
                        sampling_frequency=1/T_BIN,
                        time_halfbandwidth_product=2,
                        start_time=0)
                    
                    c0 = Connectivity(
                        fourier_coefficients=m.fft(), 
                        frequencies=m.frequencies, 
                        time=m.time)
                
                    mmg = np.mean(
                        c0.pairwise_spectral_granger_prediction()[0],
                            axis=0)
                    mmc = np.mean(
                        c0.coherence_magnitude()[0],
                            axis=0)        
                            
                    mg[pair[0], pair[1]] = mmg[0,1]
                    mg[pair[1], pair[0]] = mmg[1,0]
                    mc[pair[0], pair[1]] = mmc[0,1]
                    mc[pair[1], pair[0]] = mmc[1,0]
             
                shuf_g.append(mg)
                shuf_c.append(mc)                
                
            shuf_g = np.array(shuf_g)
            shuf_c = np.array(shuf_c)

            p_g = np.mean(shuf_g >= score_g, axis=0)
            p_c = np.mean(shuf_c >= score_c, axis=0)    
            
            D = {'regsd': regsd,
                 'freqs': c.frequencies,
                 'p_granger': p_g,
                 'p_coherence': p_c,
                 'coherence': score_c,
                 'granger': score_g,
                 'coherence_pks': c.frequencies[np.argmax(coh,axis=0)],  
                 'granger_pks': c.frequencies[np.argmax(psg,axis=0)]}

                 
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



def get_res(nmin=10, metric='coherence', sig_only=True, rerun=False):

    '''
    Group results
    
    nmin: minimum number of neurons per region to be included
    sessmin: min number of sessions with region combi
    
    metric in ['coherence', 'granger']    
    '''

    pth_ = Path(one.cache_dir, 'granger', f'{metric}.npy')
    if (not pth_.is_file() or rerun):

        
        p = pth_res.glob('**/*')
        files = [x for x in p if x.is_file()]
        
        d = {}
        nd = []
        k = 0
        
        insign = []
        
        for sess in files:    
            D = np.load(sess, allow_pickle=True).flat[0]
            m = D[metric]
            regs = list(D['regsd'])
            p_c = D[f'p_{metric}']
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
                    
                    if sig_only:
                        if p_c[i,j] > 0.0:
                            insign.append(f'{regs[i]} --> {regs[j]}')
                            continue
                        
                    if f'{regs[i]} --> {regs[j]}' in d:
                        d[f'{regs[i]} --> {regs[j]}'].append(m[j, i])

                    else:
                        d[f'{regs[i]} --> {regs[j]}'] = []
                        d[f'{regs[i]} --> {regs[j]}'].append(m[j, i])
                            
                             
            k+=1

        print(f'{len(nd)} failures')
        print(f'{len(insign)} pairs insign. and {len(d)} significant')
        u = '_all' if not sig_only else ''
        np.save(Path(one.cache_dir, 'granger', f'{metric}{u}.npy'), 
                d, allow_pickle=True)    
    else:
        d = np.load(pth_, allow_pickle=True).flat[0]        
        
    return d    
    
    
def get_meta_info(rerun=False):

    '''
    get neuron number and peak freq_s per region???
    '''
    

    pth_ = Path(one.cache_dir, 'granger', f'all_regs.npy')
    if (not pth_.is_file() or rerun):

        p = pth_res.glob('**/*')
        files = [x for x in p if x.is_file()]
        
        d = {}
        for sess in files:
            
            D = np.load(sess, allow_pickle=True).flat[0]

            if not isinstance(D['regsd'], dict):
                continue

            dd = {key: D[key] for key in ['regsd', 'granger_pks',
                                          'coherence_pks', 
                                          'p_granger', 'p_coherence',
                                          'granger', 'coherence']}

            d[str(sess).split('/')[-1].split('.')[0]] = dd

        np.save(pth_, d, allow_pickle=True)
        
    else:
        d = np.load(pth_, allow_pickle=True).flat[0]        
        
    return d
    

    
'''
#####################
plotting
#####################    
'''
    
def plot_gc(eid, segl=10, shuf=False,
            metric0='coherence', vers='oscil', 
            peak_freq_factor0=0.55, peak_freq_factor1=0.6,
            phase_lag_factor=0.2):

    '''
    For all regions, plot example segment time series, psg,
    matrix for max gc across freqs
    
    metric = 'pairwise_spectral_granger_prediction'
    or 'coherence_magnitude'
    
    '''
    time00 = time.perf_counter()
    
    if eid == 'sim':
        r = make_data(vers = vers, peak_freq_factor0=peak_freq_factor0,
                                   peak_freq_factor1=peak_freq_factor1,
                      phase_lag_factor=0.2)
        regsd = {'dep':1, 'indep':1}
        ts = np.linspace(0, (r.shape[1] - 1) * T_BIN, r.shape[1])
    else:
        r, ts, regsd = bin_average_neural(eid)   
    
    if metric0 == 'granger':
        metric = 'pairwise_spectral_granger_prediction'
        
    if metric0 == 'coherence':
        metric = 'coherence_magnitude'

    c = gc(r, segl=segl, shuf=shuf)
    # freqs x chans x chans
    psg = getattr(c, metric)()[0]
    
    # mean score across frequencies
    m = np.mean(psg, axis=0)
    
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
    si = np.unravel_index(np.argsort(np.abs(m),axis=None), m.shape)
    # top j connections
    j = 10  
    exes = [tup for tup in reversed(list(zip(*si))) 
            if (~np.isnan(m[tup]) and tup[0] != tup[1])][:j]
    
    for tup in exes: 
        yy = psg[:,tup[0],tup[1]]
        # Order is inversed! Result is:   
        axs[1].plot(c.frequencies, yy, 
                    label =f'{regs[tup[1]]} --> {regs[tup[0]]}') 

    axs[1].legend()
    axs[1].set_xlabel('frequency [Hz]')
    axs[1].set_ylabel(metric)   
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
    axs[2].set_title('mean across freqs')
                  
    cb = plt.colorbar(ims,fraction=0.046, pad=0.04)              


    fig.suptitle(f'eid = {eid} {"shuffled" if shuf else ""} ' 
                 f'vers={vers}, peak_freq_factor0 = {peak_freq_factor0}, '
                 f'peak_freq_factor1 = {peak_freq_factor1}, ' 
                 f'phase_lag_factor={phase_lag_factor},'
                 if eid == 'sim' else 
                 f'eid = {eid} {"shuffled" if shuf else ""} ')   
    fig.tight_layout()
    time11 = time.perf_counter()
    print('runtime [sec]: ', time11 - time00)     
    
    
    
def plot_strip_pairs(metric='coherence', sessmin = 2, 
             ptype='strip', shuf=False, expo=1):

    '''
    for spectral Granger, metric in ['granger', coherence']
    '''
    d0 = get_res(metric=metric)

                        
    regs = list(Counter(np.array([s.split(' --> ') for s in
                             d0]).flatten()))
                             
    _, palette = get_allen_info()
                                  
    if ptype == 'strip':                                             
        if metric == 'coherence':
            # remove directionality
            sep = ','
            
            d = {}
            for s in d0:
                a,b = s.split(' --> ')
                if ((sep.join([a,b]) in d) or (sep.join([b,a]) in d)):
                    continue
                else:
                    d[sep.join([a,b])] = d0[s]
            
  
        else:
            d = d0
            sep = ' --> '
                
                  
        dm = {x: np.mean(d[x]) for x in d if (len(d[x]) >= sessmin)}
        
        dm_sorted = dict(sorted(dm.items(), key=lambda item: item[1]))
                      
        exs = list(dm_sorted.keys())
        nrows = 3
        per_row = len(exs)//nrows
           
        d_exs = {x:d[x] for x in exs}
    
        fig, axs = plt.subplots(nrows=nrows, 
                                ncols=1, figsize=(10,20), sharey=True)

        for row in range(nrows):
            
            pairs = exs[per_row*row: per_row*(row+1)]
            
            # plot also bar with means
            ms = [np.mean(d[x]) for x in pairs]            
            axs[row].bar(np.arange(len(pairs)), ms, 
                         color='grey', alpha=0.5)
                         
            sns.stripplot(data={x:d[x] for x in pairs},
                          ax=axs[row], color='k', size=1)

            
            axs[row].set_xticklabels([x.split(sep)[0] for x in pairs], 
                                      rotation=90, 
                                      fontsize=5 if 
                                      (metric == 'psg') else 8)
            
            for label in axs[row].get_xticklabels():
                label.set_color(palette[label.get_text()])
                
            low_regs = [x.split(sep)[-1] for x in pairs]   
            for i, tick in enumerate(axs[row].get_xticklabels()):
                axs[row].text(tick.get_position()[0], -0.3, low_regs[i],
                    ha='center', va='center', rotation=90,
                    color=palette[low_regs[i]],
                    transform=axs[row].get_xaxis_transform(), 
                    fontsize=5 if (metric == 'psg') else 8)    
            
            axs[row].set_ylabel(metric)
                   

        fig.tight_layout()
        
        
    else:
        # plot matrix of all regions       
        M = np.zeros((len(regs),len(regs)))
        for i in range(len(regs)):
            for j in range(len(regs)):
                if (i == j) or (' --> '.join([regs[i], regs[j]]) not in d0):
                    M[i,j] = 0
                else:               
                    M[i,j] = np.mean(d0[' --> '.join([regs[i], regs[j]])])
       
        if shuf:
            # shuffle region list         
            np.random.shuffle(regs)
            print('region list shuffled')
        
        
        if ptype=='emb':        
        
        
            # incomplete embedding; distance matrix entries with 0        
            emb = MDS(n_components=2,metric=False, 
                      dissimilarity='precomputed').fit_transform(1 - M**expo)
                      
            cols = np.array([palette[reg] for reg in regs])
            
            
            fig, ax = plt.subplots(figsize=(10,10))
            
            ax.scatter(emb[:,0], emb[:,1],c=cols, s = 20)

                     
            for i in range(len(regs)):
                ax.annotate('  ' + regs[i], 
                    (emb[i][0], emb[i][1]),
                    fontsize=10,color=palette[regs[i]])   
                    
            ax.set_title(f'Dissimilarity: 1 - M^{expo}')                   

        else:            

               
                        
            fig, ax = plt.subplots(figsize=(10,10))
            
            
            # plot directed granger matrix
            ims = ax.imshow(M, interpolation=None, cmap='gray_r', 
                          origin='lower')
                          
            ax.set_xticks(np.arange(M.shape[0]))
            ax.set_xticklabels(regs, rotation=90)
            ax.set_xlabel('source')
            ax.set_yticks(np.arange(M.shape[1]))
            ax.set_yticklabels(regs)
            ax.set_ylabel('target')     
            ax.set_title(f'mean across sessions')
                          
            cb = plt.colorbar(ims,fraction=0.046, pad=0.04)              
        
        
            #fig.suptitle(f'eid = {eid} {"shuffled" if shuf else ""}')   
            fig.tight_layout()        

    
def scatter_psg_coh():

    '''
    scatter region pairs, granger and coherence
    '''
    
    dg = np.load(Path(one.cache_dir, 'granger', 
                        f'granger_all.npy'), 
                        allow_pickle=True).flat[0]    
    
    dc = np.load(Path(one.cache_dir, 'granger', 
                        f'coherence_all.npy'), 
                        allow_pickle=True).flat[0]
                        
    pairs = list(set(dg.keys()).intersection(set(dc.keys())))
    
    pts = []
    gs = []
    cs = []
    
    for p in pairs:
        for i in range(len(dg[p])):
            
            gs.append(dg[p][i])
            cs.append(dc[p][i])
            pts.append(p)
            
            
    fig, ax = plt.subplots()
    ax.scatter(gs, cs, color='k', s=0.5)
    
    for i in range(len(pts)):
        ax.annotate('  ' + pts[i], 
            (gs[i], cs[i]),
            fontsize=5,color='k')                   
    
            
    ax.set_xlabel('Granger')       
    ax.set_ylabel('coherence')

    cors,ps = spearmanr(gs, cs)
    corp,pp = pearsonr(gs, cs)

    ax.set_title(f'pearson: (r,p)=({np.round(corp,2)},{np.round(pp,2)}) \n'
                 f'spearman: (r,p)=({np.round(cors,2)},{np.round(ps,2)})')
    
    
def plot_dist_scat(dist_='centroids'):

    '''
    correlate granger and coherence with distance of pair
    '''
    
    
    if dist_ == 'centroids':
        dcent = get_centroids()
        
    elif dist_ == 'structural':
        dstru = get_structural()
    

    dg = np.load(Path(one.cache_dir, 'granger', 
                        f'granger_all.npy'), 
                        allow_pickle=True).flat[0]    
    
    dc = np.load(Path(one.cache_dir, 'granger', 
                        f'coherence_all.npy'), 
                        allow_pickle=True).flat[0]    
        
    pts = []
    gs = []
    cs = []
    dists = []

    
    for p in dg:
        a,b = p.split(' --> ')
        
        if dist_ == 'centroids':
            dist = sum((dcent[a] - dcent[b])**2)**0.5
            
        elif dist_=='structural':
            if p in dstru:
                dist = dstru[p]
            else:
                continue
            
        for i in range(len(dg[p])):
            gs.append(dg[p][i])
            cs.append(dc[p][i])
            pts.append(p)    
            dists.append(dist)
             
    
    fig, axs = plt.subplots(ncols=2, sharex=True)
    
    
    ylabs = ['granger', 'coherence']
    vals = [gs, cs]
    
    for k in range(len(vals)):
        
        axs[k].scatter(dists, vals[k], color='k', s=0.5)
        
        for i in range(len(pts)):
            axs[k].annotate('  ' + pts[i], 
                (dists[i], vals[k][i]),
                fontsize=5,color='k')                   
        
                
        axs[k].set_xlabel('structural connectivity' if 
                          dist_ == 'structural' else 'centroid distance')       
        axs[k].set_ylabel(ylabs[k])

        cors,ps = spearmanr(dists, vals[k])
        corp,pp = pearsonr(dists, vals[k])

        axs[k].set_title(f'({np.round(corp,2)},{np.round(pp,2)}), '
                     f'({np.round(cors,2)},{np.round(ps,2)})')    
    
    
def plot_hub(metric='coherence_magnitude'):

    '''
    Plot a hub stripplot, where for single regions
    their average scrore across all pairs it is part of is shown
    
    for spectral Granger, metric = 'psg'
    '''


    d0 = np.load(Path(one.cache_dir, 'granger', 
                        f'{metric}.npy'), 
                        allow_pickle=True).flat[0]
    regs = list(Counter(np.array([s.split(' --> ') for s in
                             d0]).flatten()))
                             
    _, palette = get_allen_info()
    sep = ' --> '                              
    d1 = {}
    for reg in regs:    
        for pair in d0:
            a,b = pair.split(sep)     
            if not reg in d1:
                d1[reg] = []
            if (a == reg or b == reg):
                d1[reg].append(d0[pair])    

    # get canonical region order
    df = pd.read_csv('/home/mic/bwm/'    
                 f'meta/region_info.csv')
    regscan = [reg for reg in df['Beryl'].tolist()]             
    d = {}
    regs = []
    for reg in regscan:
        if reg in d1:
            d[reg] = [it for sublist in d1[reg] for it in sublist]
            regs.append(reg)    

    nrows=4
    regsprow = len(regs)//nrows
    
    fig, axs = plt.subplots(nrows=nrows, ncols=1, 
                            figsize=(9,15), sharey=True)
    
    
    for row in range(nrows):
        
        regsrow = regs[regsprow*row: regsprow*(row+1)]
        cols = [palette[reg] for reg in regsrow]
        
        # plot also bar with means
        ms = [np.mean(d[x]) for x in regsrow]            
        axs[row].bar(np.arange(len(regsrow)), ms, 
                     color=[palette[reg] for reg in regsrow])
                     
        sns.stripplot(data={x:d[x] for x in regsrow},
                      ax=axs[row], color='k', size=1) 
                                                   
        axs[row].set_xticks(np.arange(len(regsrow)))
        axs[row].set_xticklabels([x+' '+f'({str(len(d[x]))})'
                                  for x in regsrow], 
                                    rotation=90)
        
        k = 0
        for label in axs[row].get_xticklabels():
            label.set_color(cols[k])
            k+=1           
             
        axs[row].set_ylabel(metric)

        
    fig.tight_layout()
    

def replot_struc():

    '''
    replot structural connectivity matrix from Allen data
    '''

    # get region order as in paper
    s=pd.read_excel('/home/mic/fig3.xlsx')
    cols = list(s.keys())[1:296]
    rows = s['Unnamed: 0'].array
    
    M = np.zeros((len(cols), len(rows)))
    for i in range(len(cols)):
        M[i] = s[cols[i]].array
        
    M = M.T

    # thresholding as in the paper
    M[M > 10**(-0.5)] = 1
    M[M < 10**(-3.5)] = 0

    cols1 = [reg.strip().replace(",", "") for reg in cols]
    rows1 = [reg.strip().replace(",", "") for reg in rows]
    
    fig, ax = plt.subplots()            
    ax.imshow(M)        
                                             
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols1, 
                       rotation=90, fontsize=5)
                                
    _, pal = get_allen_info()
    for label in ax.get_xticklabels():
        label.set_color(pal[label.get_text()]) 

    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows1, fontsize=5)
                                
    for label in ax.get_yticklabels():
        label.set_color(pal[label.get_text()]) 
 
    ax.set_title('structural connectivity, fig3 in Allen paper')
      
 
def freq_maxs_hists(perc = 95, freqlow=10):

    '''
    histograms of peak frequencies
    perc: threshold of percentile of scores to highlight 
        high freq interaction 
    '''

    d = get_meta_info()

    thr = {metric: np.nanpercentile(np.concatenate([d[eid][metric].flatten() 
                              for eid in d]), perc) for metric in 
                              ['granger', 'coherence']}

    fig, axs = plt.subplots(ncols=2, sharex=True)
    k = 0
    for metric in ['granger', 'coherence']:

        pks = []
        sess_high = []
        for eid in d:
            ks = d[eid][f'{metric}_pks'][d[eid][f'p_{metric}'] == 0.0]
            pks.append(ks)
            ks2 = d[eid][f'{metric}_pks'][np.bitwise_and.reduce([
                d[eid][f'p_{metric}'] == 0.0,
                d[eid][f'{metric}'] > thr[metric],
                d[eid][f'{metric}_pks'] > freqlow])]
            
            if np.all(ks2.size == 0):
                continue
            else:
                sess_high.append(eid)        
            
     
        pks = np.concatenate(pks) 
        axs[k].hist(pks,bins=600)
        axs[k].set_xlabel('peak frequency [Hz]')
        axs[k].set_ylabel('# region pairs')
        axs[k].set_title(metric)
        print(metric)
        print(sess_high) 
        k+=1

    fig.suptitle('only region-shuffle significant')
    fig.tight_layout()
    

def scatter_direction(only_sig=True):

    '''
    scatter plot for region pairs 
    Granger A --> B on x, B --> A on y
    source region colored
    '''
    
    _, pa = get_allen_info()    
    dg = np.load(Path(one.cache_dir, 'granger', 
                        f"granger{'' if only_sig else '_all'}.npy"), 
                        allow_pickle=True).flat[0]

    sep = ' --> '
    
    pairs = []
    for s in dg:
        a,b = s.split(sep)
        # check if both directions were significant
        if ((sep.join([a,b]) in dg) and (sep.join([b,a]) in dg)):
            if (([a,b] in pairs) or ([b,a] in pairs)):
                continue
            else:
                pairs.append([a,b])    
            
    # for each region pair get mean granger 
    dir0 = []
    dir1 = []
    pairs0 = []
    for pair in pairs:
        a,b = pair
        dir0.append(np.mean(dg[sep.join([a,b])]))
        dir1.append(np.mean(dg[sep.join([b,a])]))
        pairs0.append(', '.join([a,b]))
          
            
    fig, ax = plt.subplots()
    ax.scatter(dir0, dir1, color='k', s=0.5)

    for i in range(len(pairs)):
        ax.annotate('  ' + pairs0[i], 
            (dir0[i], dir1[i]),
            fontsize=5,color='k')  

            
    ax.set_xlabel('A --> B')       
    ax.set_ylabel('B --> A')

    cors,ps = spearmanr(dir0, dir1)
    corp,pp = pearsonr(dir0, dir1)

    ax.set_title(f'pearson: (r,p)=({np.round(corp,2)},{np.round(pp,2)}) \n'
                 f'spearman: (r,p)=({np.round(cors,2)},{np.round(ps,2)})')
        
    
    
    
    
    
    
    
    
    




 
  
  
