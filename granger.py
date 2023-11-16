import numpy as np
from collections import Counter
from pathlib import Path
from spectral_connectivity import Multitaper, Connectivity
import time

from brainwidemap import load_good_units
from iblutil.numerical import bincount2D
from one.api import ONE
from iblatlas.regions import BrainRegions

import matplotlib.pyplot as plt
import matplotlib.patches as patches


one = ONE(base_url='https://openalyx.internationalbrainlab.org',
          password='international', silent=True)
br = BrainRegions()
T_BIN=0.0125

def get_allen_info():
    r = np.load(Path(one.cache_dir, 'dmn', 'alleninfo.npy'),
                allow_pickle=True).flat[0]
    return r['dfa'], r['palette']
    
    
def bin_average_neural(eid, mapping='Beryl', nmin=1):
    '''
    bin neural activity; bin, then average firing rates per region
    from both probes if available
    
    used to get session wide time series, not cut into trials
    '''
    
    pids, probes = one.eid2pid(eid)
    
    if len(probes) == 1:
        spikes, clusters = load_good_units(one, pids[0])
        R, times, _ = bincount2D(spikes['times'], spikes['clusters'], T_BIN)
        acs = br.id2acronym(clusters['atlas_id'], mapping=mapping)
        regs = Counter(acs)
        regs2 = [x for x in regs if 
                 ((regs[x] >= nmin) and (x not in ['root','void']))]
                 
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
    

    r_segments = r[:, :num_segments * segment_length
                   ].reshape((nchans, num_segments, 
                   segment_length))
                   
    #  (n_time_samples, n_trials, n_signals)               
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

    # pairwise directed_spectral_granger; freqs x chans x chans
    psg = c.pairwise_spectral_granger_prediction()[0] 
    return psg, c.frequencies    


def plot_gc(eid, segl=10, show_fig=True):

    '''
    For all regions, plot example segment time series, psd,
    matrix for max gc across freqs
    '''
    time00 = time.perf_counter()
    
       
    r, ts, regsd = bin_average_neural(eid)
    psg, freqs = gc(r, segl=segl)
    
    m = np.max(psg, axis=0)

    time11 = time.perf_counter()
    
    print('runtime [min]: ', (time11 - time00) / 60)
    
    
    if show_fig:    
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(14,6))
        
        # plot example time series, first segment
        exdat = r[:,:int(segl/T_BIN)]/T_BIN
        extime = ts[:int(segl/T_BIN)]    
       
        _, pal = get_allen_info()    
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
            axs[1].plot(freqs, yy, label =f'{regs[tup[0]]} --> {regs[tup[1]]}') 

        axs[1].legend()
        axs[1].set_xlabel('frequency [Hz]')
        axs[1].set_ylabel('directed Granger [a.u.]')   
        axs[1].set_title(f'top {j} tuples') 
        
        # plot directed granger matrix
        ims = axs[2].imshow(m, interpolation=None, cmap='gray', 
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
        axs[2].set_xlabel('target')     
        axs[2].set_title('max GC')
                      
        cb = plt.colorbar(ims,fraction=0.046, pad=0.04)              
    
    
        fig.suptitle(f'eid = {eid}')   
        fig.tight_layout()

           
    else:
        return m, regs                   
    






  
