import numpy as np
from pathlib import Path
import pandas as pd
from scipy.stats import pearsonr
import time
import matplotlib.pyplot as plt
import gc as garbage
from collections import defaultdict
import seaborn as sns
from matplotlib.colors import PowerNorm
from matplotlib.ticker import MaxNLocator

from brainwidemap import (load_good_units, bwm_query, 
    download_aggregate_tables, load_trials_and_mask)
from iblatlas.regions import BrainRegions
from iblatlas.atlas import AllenAtlas
import iblatlas
from iblutil.numerical import bincount2D
from one.api import ONE


plt.ion()

one = ONE()
#bwmq = bwm_query(one)
ba = AllenAtlas()          
br = BrainRegions()

T_BIN = 0.0125  # 0.005
sigl=0.05  # alpha throughout

def get_allen_info():
    r = np.load(Path(one.cache_dir, 'dmn', 'alleninfo.npy'),
                allow_pickle=True).flat[0]
    return r['dfa'], r['palette']


pth_res = Path(one.cache_dir, 'cell_corr') 
pth_res.mkdir(parents=True, exist_ok=True)

# window names: [alignment times, segment length, gap, side]   
wins = {'feedback_plus1': ['feedback_times',1, 0, 'plus'],
        'stim_plus01': ['stimOn_times', 0.1, 0, 'plus'],
        'stim_minus06_minus02': ['stimOn_times', 0.4, 0.2, 'minus'],
        'move_minus01': ['firstMovement_times', 0.1, 0, 'minus']}    


def bin_neural(eid, mapping='Beryl'):
    '''
    bin neural activity; bin from both probes if available
    
    used to get session-wide time series, not cut into trials
    
    nmin: 
        minumum number of neurons per brain region to consider it
    returns: 
        R2: binned firing rate per region per time bin
        times: time stamps for all time bins
    '''
    
    pids0, probes = one.eid2pid(eid)
    
    pids = []
    

    for pid in pids0:
        if pid in bwmq['pid'].values:
            pids.append(pid)

    if len(pids) == 1:
        spikes, clusters = load_good_units(one, pids[0])
        R, times, _ = bincount2D(spikes['times'], 
                        spikes['clusters'], T_BIN)
        acs = br.id2acronym(clusters['atlas_id'], mapping=mapping)

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
                                        
        acs = np.concatenate([
                       br.id2acronym(clus[0]['atlas_id'],
                       mapping=mapping), 
                       br.id2acronym(clus[1]['atlas_id'],
                       mapping=mapping)])
                                                        
        t_sorted = np.sort(times_both)
        c_ordered = clusters_both[np.argsort(times_both)] 
        
        R, times, _ = bincount2D(t_sorted, c_ordered, T_BIN)  

    R = R[~np.isin(acs, ['void', 'root'])]
    acs = acs[~np.isin(acs, ['void', 'root'])]

    # remove cells that are always zero
    y = ~np.all(R == 0, axis=1)
    R = R[y]
    acs = acs[y]

    return R, times, acs


def cut_segments(r, ts, te, segment_length=100, side='plus', gap_length=0):

    '''
    r:
        binned activity time series
    ts:
        time stamps per bin
    te:
        event times where segments start
    segment_length:
        seg length in bins
    side: ['plus', 'minus']
        if segments start or end at alignement time
    gap_length:
        gap between segment and alignement event in bins
        
    Returns:
        A 3D array of segments with shape (n_regions, n_events, segment_length)

    ''' 

    r = np.array(r)
    ts = np.array(ts)
    te = np.array(te)
    
    # Ensure r is 2D, even if it's a single region
    if r.ndim == 1:
        r = r[np.newaxis, :]
        
    # Find indices of the nearest time stamps to event times
    event_indices = np.searchsorted(ts, te)  
      
    # Adjust start indices based on 'side' and gap_length
    if side == 'plus':
        # Start segment after the event time plus the gap
        start_indices = event_indices + gap_length
    elif side == 'minus':
        # End segment at event time minus the gap, so start earlier
        start_indices = event_indices - segment_length - gap_length
    else:
        raise ValueError("Invalid value for 'side'. Choose 'plus' or 'minus'.")
    
    # Create an array of indices for each segment
    indices = start_indices[:, np.newaxis] + np.arange(segment_length)
    
    # Clip indices to ensure they're within bounds
    indices = np.clip(indices, 0, r.shape[1] - 1)
    
    # Extract segments
    segments = r[:, indices]

    # Rearrange dimensions to (n_regions, n_events, segment_length)
    segments = np.transpose(segments, (0, 1, 2))
    
    # If original input was 1D, remove the singleton dimension
    if r.shape[0] == 1:
        segments = segments.squeeze(axis=1)
    
    return segments


def corr_cells(eid):

    '''
    for all windows of interest,
    get pearson correlation for all cell pairs
    after concatenating trials
    '''

    r, ts, acs = bin_neural(eid)

    d = {}
    d['acs'] = acs

    for win in wins:
        print(win, 'align|segl|gap|side', wins[win])

        segl = wins[win][1]  # in sec
        segment_length = int(segl / T_BIN)  # in bins
        gap = wins[win][2]  # in sec
        gap_length = int(gap / T_BIN)  # in bins
        side = wins[win][3]

        # only pick segments starting at "win" times
        # Load in trials data and mask bad trials (False if bad)
        trials, mask = load_trials_and_mask(one, eid,
            saturation_intervals=['saturation_stim_plus04',
                                    'saturation_feedback_plus04',
                                    'saturation_move_minus02',
                                    'saturation_stim_minus04_minus01',
                                    'saturation_stim_plus06',
                                    'saturation_stim_minus06_plus06'])
                                    
        te = trials[mask][wins[win][0]].values
            
        # n_cells x n_segments x n_time_samples
        r_segments = cut_segments(r, ts, te, gap_length=gap_length,
                        side=side, segment_length=segment_length)

        n_cells, n_segments, n_bins = r_segments.shape

        # concatenate windows               
        r_segments_reshaped = r_segments.reshape(n_cells, n_segments*n_bins)

        d[win] = np.corrcoef(r_segments_reshaped)

    return d


'''
################################
bulk processing
################################
'''

def get_all_corr(eids='all', wins=wins):

    if isinstance(eids, str):
        eids = np.unique(bwmq[['eid']].values)
                
    Fs = []
    k = 0
    print(f'Processing {len(eids)} sessions')

    time0 = time.perf_counter()
    for eid in eids:
        print('eid:', eid)

        try:
            time00 = time.perf_counter()
           
            d = corr_cells(eid)

            np.save(pth_res / f'{eid}.npy', d, allow_pickle=True)

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


def combine_res(full_distri=False, rerun=False, nmin=100):

    '''
    for each window and each region pair,
    group corr scores across sessions,
    return mean and std (unless full_distri=True)
    nmin: minimum number of values to keep region pair
    '''

    pth_ = Path(one.cache_dir, 'cell_corr_res', 
                f'full_distri_{full_distri}.npy')

    if (not pth_.is_file() or rerun):

        time0 = time.perf_counter()

        p = pth_res.glob('**/*')
        files = [x for x in p if x.is_file()]

        D = {}

        for win in wins:
            d = defaultdict(list)

            def process_session(corr_matrix, regions):
                n_regions = corr_matrix.shape[0]
                for i in range(n_regions):
                    # Only upper triangle, no need for j < i
                    for j in range(i + 1, n_regions):  
                        # Sort the pair to ensure non-directionality
                        pair = tuple(sorted([regions[i], regions[j]]))
                        # Append the correlation score to the dictionary
                        if not np.isnan(corr_matrix[i, j]):
                            d[pair].append(corr_matrix[i, j])

            for fi in files:
                res = np.load(fi, allow_pickle=True).flat[0]
                acs = res['acs']
                c = res[win]
                process_session(c, acs)

            d = {pair: values for pair, values in d.items() 
                 if len(values) >= nmin}

            D[win] = d

        D0 = {}
        for win, corr_dict in D.items():
            d0 = {}
            for pair, values in corr_dict.items():
                corr_values = [val for val in values if not np.isnan(val)]
                if len(corr_values) > nmin :
                    d0[pair] = [np.nanmean(corr_values), 
                                np.nanstd(corr_values), 
                                len(corr_values)]
            D0[win] = d0

        pth__ = Path(one.cache_dir, 'cell_corr_res', 
                f'full_distri_False.npy')
        np.save(pth__, D0, allow_pickle=True)
        pth__ = Path(one.cache_dir, 'cell_corr_res', 
                f'full_distri_True.npy')
        np.save(pth__, D, allow_pickle=True)

        time1 = time.perf_counter()
        print(time1 - time0, f'sec for {len(files)} sessions')

    if not rerun:
        return np.load(pth_, allow_pickle=True).flat[0]


'''
################################
plotting
################################
'''

def corr_eid(eid, clipped=False):
    '''
    For a given session (eid), load the correlation matrix for each window and plot it as subpanels using imshow.
    The matrix is labeled with region names in their respective colors, and each subplot is titled with the corresponding window.
    The main title is "per cell". The color scale is consistent across all windows.
    '''
    
    # Path to the correlation data for the given session
    p = list(pth_res.glob(f'**/*{eid}*.npy'))  # Assuming session files are saved in .npy format
    if not p:
        print(f"No data found for session {eid}")
        return
    
    # Load the session data
    res = np.load(p[0], allow_pickle=True).flat[0]  # Load session data assuming a single result file
    acs = res['acs']  # Region labels
    windows = list(res.keys())[1:]  # Get the different windows from the result
    
    # Load region color palette from Allen Brain Atlas info
    _, palette = get_allen_info()  # Assuming palette is a dictionary with region colors

    # Step 1: Find global min and max across all windows
    all_values = []
    for win in windows:
        all_values.extend(res[win].flatten())  # Flatten the matrix to collect all values
    
    if clipped:
        vmin = -0.01  # Global minimum correlation value
        vmax = 0.01
    else:
        vmin = np.nanmin(all_values)
        vmax = np.nanmax(all_values)

    # Create a 2x2 grid for subplots (one subplot for each window)
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    axs = axs.flatten()  # Flatten to iterate over it easily
    cmap = 'coolwarm'  # Colormap to use
    
    # Loop through each window and plot the correlation matrix
    for i, win in enumerate(windows):
        c = res[win]  # Get the correlation matrix for the current window
        
        # Plot the correlation matrix using imshow
        im = axs[i].imshow(c, cmap=cmap, aspect='equal', vmin=vmin, vmax=vmax)
        
        # Set x and y tick labels as region names and color them using the palette
        axs[i].set_xticks(np.arange(len(acs)))
        axs[i].set_xticklabels(acs, rotation=90, fontsize=8)

        axs[i].set_yticks(np.arange(len(acs)))
        axs[i].set_yticklabels(acs, fontsize=8)

        # Color the labels based on the palette
        for label in axs[i].get_xticklabels():
            region = label.get_text()
            label.set_color(palette.get(region, 'black'))  # Default to black if not found in the palette

        for label in axs[i].get_yticklabels():
            region = label.get_text()
            label.set_color(palette.get(region, 'black'))  # Default to black if not found in the palette
        
        # Set the title for the window
        axs[i].set_title(f'Window {win}')
    
    # Add a single colorbar on top, shared by all subplots
    cbar_ax = fig.add_axes([0.3, 0.92, 0.4, 0.02])  # Position of the colorbar (adjust as needed)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Correlation Value')

    # Set the main title for the entire figure
    fig.suptitle(f'Correlation Matrix for Session {eid} (per cell)', fontsize=16)
    
    # Adjust the layout to avoid overlap
    plt.tight_layout(rect=[0, 0, 1, 0.9])  # Leave space for the colorbar on top
    
    # Show the plot
    plt.show()


def plot_heatmap(clipped=True, return_adjacencies=False):

    '''
    For each of the windows, plot a heatmap of the means, using the canonical region sorting;
    using the same region axes for all panels, and displaying region labels in their respective colors.
    clipped: clip range to see differences
 
    '''

    # Get the combined results (means and std)
    d = combine_res()

    # Load the canonical order of regions (e.g., from a file)
    p = (Path(iblatlas.__file__).parent / 'beryl.npy')
    regs_ = br.id2acronym(np.load(p), mapping='Beryl')

    # Get region colors from the Allen Brain Atlas information
    _, palette = get_allen_info()

    # Filter the canonical regions to only those present in the data
    regions = [reg for reg in regs_ if 
                reg in set([pair[0] for win_dict in d.values() 
                for pair in win_dict.keys()] + [pair[1] 
                for win_dict in d.values() for pair in win_dict.keys()])]


    # Step 1: Find global min and max across all windows
    all_values = []
    for win in d.keys():
        for stats in d[win].values():
            all_values.append(stats[0])  # stats[0] is the mean correlation

    if clipped:
        vmin = -0.01  # Global minimum correlation value
        vmax = 0.01
    else:
        vmin = min(all_values)  # Global minimum correlation value
        vmax = max(all_values)        

    if not return_adjacencies:
        # Create a 2x2 grid of subplots (modify nrows and ncols based on the number of windows)
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 8), 
                                sharex=True, sharey=True)
        axs = axs.flatten()

    cmap = 'coolwarm'

    D = {}

    # Loop through each window and plot the heatmap
    for i, win in enumerate(d.keys()):
        # Initialize a matrix to store the mean correlation values for each region pair
        n_regions = len(regions)
        corr_matrix = np.full((n_regions, n_regions), np.nan)

        # Fill the correlation matrix with the mean values for each region pair
        for pair, stats in d[win].items():
            # stats[0] is the mean, stats[1] is the std, stats[2] is the number of entries
            reg1, reg2 = pair
            if reg1 in regions and reg2 in regions:
                idx1, idx2 = regions.index(reg1), regions.index(reg2)
                corr_matrix[idx1, idx2] = stats[0]
                corr_matrix[idx2, idx1] = stats[0]  # Ensure it's symmetric for non-directed pairs

        D[win] = [corr_matrix, regions]
        if return_adjacencies:
            continue

        # Plot the heatmap for the current window without colorbars (colorbars will be set outside)
        sns.heatmap(corr_matrix, ax=axs[i], xticklabels=regions, yticklabels=regions,
                    cmap=cmap, cbar=False, square=True, vmin=vmin, vmax=vmax)

        # Set the title for the subplot
        axs[i].set_title(f'Window {win}')
        axs[i].tick_params(length=0)

        # Set x and y tick labels with region-specific colors
        for label in axs[i].get_xticklabels():
            region = label.get_text()
            label.set_color(palette.get(region, 'black'))  # Default to black if not found in the palette
            label.set_fontsize(8)
            label.set_rotation(90)


        for label in axs[i].get_yticklabels():
            region = label.get_text()
            label.set_color(palette.get(region, 'black'))  # Default to black if not found in the palette
            label.set_fontsize(8)

    if return_adjacencies:
        return D

    # Add a single colorbar on top, shared by all subplots
    cbar_ax = fig.add_axes([0.3, 0.92, 0.4, 0.02])  # Position of the colorbar (adjust as needed)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Mean Correlation')

    # Adjust layout and display the plot
    fig.suptitle('Correlation Heatmap Across Windows', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.9])  # Leave space for the top colorbar
    plt.show()


def plot_3d_adjacency(label_reg=False):
    '''
    Plots 3D graphs, one per window, with regions as points and connections between them as lines.
    - Line thickness is proportional to the adjacency matrix values (weight).
    - Line color is based on the sign of the connection (red for negative, blue for positive).
    - Each region point is larger and colored based on the provided palette.
    - Axes are removed for a cleaner visualization.
    '''

    # Get the adjacency matrices and region names from the heatmap function
    D = plot_heatmap(return_adjacencies=True)

    # Load the 3D coordinates for regions (assumed to be available)
    pth_ = Path(one.cache_dir, 'granger', 'beryl_centroid.npy')
    coords = np.load(pth_, allow_pickle=True).flat[0]  # 3D coordinates for regions
    _, palette = get_allen_info()  # Get the color palette for regions

    n_windows = len(D)
    ncols = 2  # Number of columns (you can adjust based on your needs)
    nrows = int(np.ceil(n_windows / ncols))  # Number of rows needed to fit all windows

    # Create a figure with subplots for each window
    fig = plt.figure(figsize=(14, 10))

    # Loop through each window and plot the corresponding 3D graph
    for idx, (win, (adjacency_matrix, regions)) in enumerate(D.items()):
        n_regions = adjacency_matrix.shape[0]

        # Create a 3D subplot for the current window
        ax = fig.add_subplot(nrows, ncols, idx + 1, projection='3d')

        # Plot connections based on the adjacency matrix
        for i in range(n_regions):
            for j in range(i + 1, n_regions):  # Only upper triangle to avoid duplicating lines
                weight = adjacency_matrix[i, j]
                if weight != 0:  # Only plot if there's a connection
                    # Get the coordinates for the two regions
                    p1 = coords[regions[i]]
                    p2 = coords[regions[j]]

                    # Determine the color based on the sign of the weight
                    color = 'red' if weight > 0 else 'blue'

                    # Plot the line between the two points
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                            color=color, lw=np.abs(weight) * 7)  # Line width proportional to weight

        # Plot the region points in their respective colors and larger size

        xyz = []
        for i, region in enumerate(regions):
            region_color = palette.get(region, 'black')  # Get color from the palette
            ax.scatter(coords[region][0], coords[region][1], coords[region][2], 
                       color=region_color, s=5)  # Larger dots
            xyz.append([coords[region][0], coords[region][1], coords[region][2]])
            if label_reg:
                # Label each region
                ax.text(coords[region][0], coords[region][1], coords[region][2], 
                        region, color=region_color)

        # Set the title for the 3D plot
        ax.set_title(f'Window {win}')

        # Remove the axes (including ticks, grid, and labels)
        xyz = np.array(xyz)
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

        ax.grid(False)
        ax.set_axis_off()       


    # Adjust layout to avoid overlap between subplots
    plt.tight_layout()
    
    # Display the figure with all subplots
    plt.show()



def distri(reg_pair):
    '''
    For a given region pair reg_pair = (A, B), plot three distributions:
    - (A, A): Correlations within region A.
    - (B, B): Correlations within region B.
    - (A, B): Correlations between regions A and B.
    
    One panel (stripplot) per window, with means as thick black line segments.
    '''
    
    # Get the full distribution data
    d0 = combine_res(full_distri=True)
    d = {win: corr_dict for win, corr_dict in d0.items()
                 if reg_pair in corr_dict or tuple(reversed(reg_pair)) in corr_dict}

    # Separate the regions from the input pair
    reg1, reg2 = reg_pair

    # Create subplots: One per window
    n_windows = len(d)
    fig, axs = plt.subplots(nrows=1, ncols=n_windows, figsize=(3 * n_windows, 4), sharey=True)
    
    if n_windows == 1:  # If there is only one window, axs won't be an array
        axs = [axs]

    # Loop over each window and plot the stripplot for each
    for i, (win, corr_dict) in enumerate(d.items()):
        # Lists to store the correlation values for each distribution
        corr_AA = []
        corr_BB = []
        corr_AB = []

        # Collect the correlation values for (A, A), (B, B), and (A, B)
        for pair, values in corr_dict.items():
            if pair == (reg1, reg1):  # Correlation within region A (A, A)
                corr_AA.extend(values)  # Add all correlation values
            elif pair == (reg2, reg2):  # Correlation within region B (B, B)
                corr_BB.extend(values)  # Add all correlation values
            elif set(pair) == set([reg1, reg2]):  # Correlation between A and B (A, B)
                corr_AB.extend(values)  # Add all correlation values

        # Combine data and labels for stripplot
        data = corr_AA + corr_BB + corr_AB
        labels = (['(A, A)'] * len(corr_AA) + ['(B, B)'] * len(corr_BB) 
                + ['(A, B)'] * len(corr_AB))

        # Plot the stripplot in the corresponding subplot
        sns.stripplot(x=labels, y=data, jitter=True, ax=axs[i], size=1)
        axs[i].set_ylabel('Correlation Value')
        axs[i].set_title(f'Window {win}')
        xticks = axs[i].get_xticks()

        # Calculate means and plot them as black thick line segments at the correct x-tick positions
        if corr_AA:
            mean_AA = np.nanmean(corr_AA)
            axs[i].plot([xticks[0] - 0.2, xticks[0] + 0.2], 
            [mean_AA, mean_AA], color='black', lw=4)
        if corr_BB:
            mean_BB = np.nanmean(corr_BB)
            axs[i].plot([xticks[1] - 0.2, xticks[1] + 0.2], 
            [mean_BB, mean_BB], color='black', lw=4)
        if corr_AB:
            mean_AB = np.nanmean(corr_AB)
            axs[i].plot([xticks[2] - 0.2, xticks[2] + 0.2], 
            [mean_AB, mean_AB], color='black', lw=4)

    # Set a common title
    fig.suptitle(f'Distribution of Correlations for {reg1} and {reg2} Across Windows', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()



