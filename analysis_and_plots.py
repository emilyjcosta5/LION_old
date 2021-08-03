# Author: Emily Costa
# Created on: Jun 3, 2021
# Functions for analyzing the I/O behaviors of applications on an HPC system using the clusters generated from `clustering`.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import os
from os.path import join
from matplotlib import ticker
import seaborn as sns

def cluster_characteristics(cluster_info, save_path='./figures/cluster_characteristics.pdf', plot_log=False, verbose=False):
    '''
    Analyzes the number of clusters by application and number of runs by cluster.
    Plots the CDF of both the metrics.

    Parameters
    ----------
    path_to_clusters: string
        Path to the parquet file created by `clustering` of the cluster info.
    save_path: string, optional
        Where to save the figure.
    verbose: boolean, optional
        For debugging and info on amount of info being collected.

    Returns
    -------
    None
    '''
    # 'Application' 'Operation' 'Cluster Number' 'Cluster Size' 'Filename'
    # Initialize plot
    fig, axes = plt.subplots(2,1)
    # Read the data
    df = cluster_info
    # Collect Read Info
    operation = 'Read'
    if verbose:
        print('Analyzing %s clusters...'%operation)
    mask = df['Operation'] == operation 
    pos = np.flatnonzero(mask)
    tmp = df.iloc[pos]
    no_runs_by_cluster = []
    no_clusters_by_app = []
    apps = tmp['Application'].unique().tolist()
    for app in apps:
        mask = tmp['Application'] == app
        pos = np.flatnonzero(mask)
        tmp_app = tmp.iloc[pos]
        cluster_nos = tmp_app['Cluster Number'].unique().tolist()
        no_clusters_by_app.append(len(cluster_nos))
        for cluster_no in cluster_nos:
            no_runs_by_cluster.append(tmp_app.iloc[np.flatnonzero(tmp_app['Cluster Number'] == cluster_no)].shape[0])
    # Plot Read Info
    # First, number of cluster by applications
    if verbose:
        print('Plotting info of %s clusters...'%operation)
    median = np.median(no_clusters_by_app)-1
    if verbose:
        print("Median of %s clusters in the applications: %d"%(operation,median))
    bins = np.arange(0, int(math.ceil(max(no_clusters_by_app)))+1, 1)
    hist = np.histogram(no_clusters_by_app, bins=bins)[0]
    cdf = np.cumsum(hist)
    cdf = [x/cdf[-1] for x in cdf]
    axes[0].plot(bins[:-1], cdf, color='skyblue', linewidth=2, label=operation)
    axes[0].axvline(median, color='skyblue', zorder=0, linestyle='--', linewidth=2)
    # Second, number of runs by clusters
    median = np.median(no_runs_by_cluster)
    if verbose:
        print("Median of runs in the %s clusters: %d"%(operation,median))
    bins = np.arange(0, int(math.ceil(max(no_runs_by_cluster)))+1, 1)
    hist = np.histogram(no_runs_by_cluster, bins=bins)[0]
    cdf = np.cumsum(hist)
    cdf = [x/cdf[-1] for x in cdf]
    axes[1].plot(bins[:-1], cdf, color='skyblue', linewidth=2, label=operation)
    axes[1].axvline(median, color='skyblue', zorder=0, linestyle='--', linewidth=2)
    # Collect Write info
    operation = 'Write'
    if verbose:
        print('Analyzing %s clusters...'%operation)
    mask = df['Operation'] == operation
    pos = np.flatnonzero(mask)
    tmp = df.iloc[pos]
    no_runs_by_cluster = []
    no_clusters_by_app = []
    apps = tmp['Application'].unique().tolist()
    for app in apps:
        mask = tmp['Application'] == app
        pos = np.flatnonzero(mask)
        tmp_app = tmp.iloc[pos]
        cluster_nos = tmp_app['Cluster Number'].unique().tolist()
        no_clusters_by_app.append(len(cluster_nos))
        for cluster_no in cluster_nos:
            no_runs_by_cluster.append(tmp_app.iloc[np.flatnonzero(tmp_app['Cluster Number'] == cluster_no)].shape[0])
    # Plot Write Info
    # First, number of cluster by applications
    if verbose:
        print('Plotting info of %s clusters...'%operation)
    median = np.median(no_clusters_by_app)-1
    if verbose:
        print("Median of %s clusters in the applications: %d"%(operation,median))
    bins = np.arange(0, int(math.ceil(max(no_clusters_by_app)))+1, 1)
    hist = np.histogram(no_clusters_by_app, bins=bins)[0]
    cdf = np.cumsum(hist)
    cdf = [x/cdf[-1] for x in cdf]
    axes[0].plot(bins[:-1], cdf, color='maroon', linewidth=2, label=operation)
    axes[0].axvline(median, color='maroon', zorder=0, linestyle=':', linewidth=2)
    # Second, number of runs by clusters
    median = np.median(no_runs_by_cluster)
    if verbose:
        print("Median of runs in the %s clusters: %d"%(operation,median))
    bins = np.arange(0, int(math.ceil(max(no_runs_by_cluster)))+1, 1)
    hist = np.histogram(no_runs_by_cluster, bins=bins)[0]
    cdf = np.cumsum(hist)
    cdf = [x/cdf[-1] for x in cdf]
    axes[1].plot(bins[:-1], cdf, color='maroon', linewidth=2, label=operation)
    axes[1].axvline(median, color='maroon', zorder=0, linestyle=':', linewidth=2)
    # Overall figure aesthetics 
    axes[0].set_xlabel('CDF of Applications')
    axes[0].set_ylabel('Number of Clusters')
    axes[1].set_xlabel('CDF of Clusters')
    axes[1].set_ylabel('Number of Runs')
    axes[1].legend()
    axes[0].legend()
    axes[0].set_ylim(0,1)
    axes[1].set_ylim(0,1)
    vals = axes[0].get_yticks()
    axes[0].set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    vals = axes[1].get_yticks()
    axes[1].set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    fig.tight_layout()
    save_dir = "/".join(save_path.split('/')[:-1])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(save_path)
    return None

def general_temporal_trends(cluster_info, save_directory='./', verbose=False):
    '''
    overall cluster temporal overlap, inter-arrival times variability and time 
    span, overall time spans of clusters, 

    Parameters
    ----------
    path_to_clusters: string
        Path to the parquet file created by `clustering` of the cluster info.
    path_to_total: string
        Path to Darshan logs parsed in the 'total' format. The logs should be
        sorted by user ID and executable name.
    ranks: int, optional
        Parallize the data collection by increasing the number of processes
        collecting and saving the data.
    save_path: string, optional
        Where to save the figure.
    verbose: boolean, optional
        For debugging and info on amount of info being collected.

    Returns
    -------
    None
    '''
    cluster_info['End Time'] = cluster_info['End Time'].astype(int)
    cluster_info['Start Time'] = cluster_info['Start Time'].astype(int)
    # Read
    operation = 'Read'
    df = cluster_info[cluster_info['Operation']==operation]
    c_nos = df['Cluster Number'].unique()
    read_df = pd.DataFrame()
    for c_no in c_nos:
        mask = df['Cluster Number'] == c_no
        pos = np.flatnonzero(mask)
        tmp = df.iloc[pos].sort_index().reset_index()
        no_runs = tmp.shape[0]
        total_time = tmp['End Time'].max() - tmp['Start Time'].min()
        total_days              = total_time/86400
        time_differences = []
        for j in np.arange(0, no_runs-1):
            time_difference = abs(tmp.loc[j+1]['End Time']-tmp.loc[j]['Start Time'])
            time_differences.append(int(time_difference))
        time_differences_avg    = np.average(time_differences)
        time_differences_std    = np.std(time_differences) 
        time_differences_cov    = (time_differences_std/time_differences_avg)*100
        read_df = read_df.append({'Cluster Number': c_no, 'Total Time': total_time, 'Average Runs per Day': no_runs/total_days, 
            'Temporal Coefficient of Variation': time_differences_cov}, ignore_index=True)
    range = []
    for n in read_df['Total Time']:
        if(n<86400):
            range.append('<1d')
        elif(n<259200):
            range.append('1-\n3d')
        elif(n<604800):
            range.append('3d-\n1w')
        elif(n<(2592000/2)):
            range.append('1w-\n2w')
        elif(n<2592000):
            range.append('2w-\n1M')
        elif(n<7776000):
            range.append('1-\n3M')
        elif(n<15552000):
            range.append('3-\n6M')
        else:
            continue
    read_df['Range'] = range
    # Write
    operation = 'Write'
    df = cluster_info[cluster_info['Operation']==operation]
    c_nos = df['Cluster Number'].unique()
    write_df = pd.DataFrame()
    for c_no in c_nos:
        mask = df['Cluster Number'] == c_no
        pos = np.flatnonzero(mask)
        tmp = df.iloc[pos].sort_index().reset_index()
        no_runs = tmp.shape[0]
        total_time = tmp['End Time'].max() - tmp['Start Time'].min()
        total_days              = total_time/86400
        time_differences = []
        for j in np.arange(0, no_runs-1):
            time_difference = abs(tmp.loc[j+1]['End Time']-tmp.loc[j]['Start Time'])
            time_differences.append(int(time_difference))
        time_differences_avg    = np.average(time_differences)
        time_differences_std    = np.std(time_differences) 
        time_differences_cov    = (time_differences_std/time_differences_avg)*100
        write_df = write_df.append({'Cluster Number': c_no, 'Total Time': total_time, 'Average Runs per Day': no_runs/total_days, 
            'Temporal Coefficient of Variation': time_differences_cov}, ignore_index=True)
    range = []
    for n in write_df['Total Time']:
        if(n<86400):
            range.append('<1d')
        elif(n<259200):
            range.append('1-\n3d')
        elif(n<604800):
            range.append('3d-\n1w')
        elif(n<(2592000/2)):
            range.append('1w-\n2w')
        elif(n<2592000):
            range.append('2w-\n1M')
        elif(n<7776000):
            range.append('1-\n3M')
        elif(n<15552000):
            range.append('3-\n6M')
        else:
            continue
    write_df['Range'] = range
    ########################################################### CDF of time periods and frequency ###########################################################
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=[5, 2.2])
    fig.subplots_adjust(left=0.15, right=0.965, top=.94, bottom=0.34, wspace=0.05)
    read_info = read_df['Total Time']/86400
    write_info = write_df['Total Time']/86400
    read_median = np.median(read_info)
    write_median = np.median(write_info)
    read_info = np.log10(read_info)
    write_info = np.log10(write_info)
    read_median_plotting = np.median(read_info)
    write_median_plotting = np.median(write_info)
    read_bins = np.arange(0, int(math.ceil(max(read_info)))+1, 0.01)
    hist = np.histogram(read_info, bins=read_bins)[0]
    cdf_read = np.cumsum(hist)
    cdf_read = [x/cdf_read[-1] for x in cdf_read]
    write_bins = np.arange(0, int(math.ceil(max(write_info)))+1, 0.01)
    hist = np.histogram(write_info, bins=write_bins)[0]
    cdf_write = np.cumsum(hist)
    cdf_write = [x/cdf_write[-1] for x in cdf_write]
    axes[0].plot(read_bins[:-1], cdf_read, color='skyblue', linewidth=2, label='Read')
    axes[0].plot(write_bins[:-1], cdf_write, color='maroon', linewidth=2, label='Write')
    axes[0].set_ylabel('CDF of Clusters')
    axes[0].set_xlabel('(a) Cluster Time\nSpan (days)')
    axes[0].yaxis.grid(color='lightgrey', linestyle=':')
    axes[0].set_axisbelow(True)
    axes[0].set_ylim(0,1)
    axes[0].set_xlim(0,3)
    axes[0].set_yticks(np.arange(0,1.2,0.25))
    positions = [1, 2, 3]
    labels = ['$10^1$', '$10^2$', '$10^3$']
    axes[0].xaxis.set_major_locator(ticker.FixedLocator(positions))
    axes[0].xaxis.set_major_formatter(ticker.FixedFormatter(labels))
    vals = axes[0].get_yticks()
    axes[0].set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    # Add minor ticks
    ticks = [1,2,3,4,5,6,7,8,9] 
    f_ticks = []
    tmp_ticks = [np.log10(x) for x in ticks]
    f_ticks = f_ticks + tmp_ticks
    tmp_ticks = [np.log10(x)+1 for x in ticks]
    f_ticks = f_ticks + tmp_ticks
    tmp_ticks = [np.log10(x)+2 for x in ticks]
    f_ticks = f_ticks + tmp_ticks
    axes[0].set_xticks(f_ticks, minor=True)
    # Add vertical lines for medians
    axes[0].axvline(np.log10(4), color='skyblue', zorder=0, linestyle='--', linewidth=2)
    axes[0].axvline(write_median_plotting, color='maroon', zorder=0, linestyle=':', linewidth=2)
    # Add legend
    axes[0].legend(loc='lower right', fancybox=True)
    read_info = read_df['Average Runs per Day'].tolist()
    write_info = write_df['Average Runs per Day'].tolist()
    read_median = np.median(read_info)
    write_median = np.median(write_info)
    read_info = np.log10(read_info)
    write_info = np.log10(write_info)
    read_bins = np.arange(0, int(math.ceil(max(read_info)))+1, 0.01)
    hist = np.histogram(read_info, bins=read_bins)[0]
    cdf_read = np.cumsum(hist)
    cdf_read = [x/cdf_read[-1] for x in cdf_read]
    write_bins = np.arange(0, int(math.ceil(max(write_info)))+1, 0.01)
    hist = np.histogram(write_info, bins=write_bins)[0]
    cdf_write = np.cumsum(hist)
    cdf_write = [x/cdf_write[-1] for x in cdf_write]
    axes[1].plot(read_bins[:-1], cdf_read, color='skyblue', linewidth=2, label='Read')
    axes[1].plot(write_bins[:-1], cdf_write, color='maroon', linewidth=2, label='Write')
    axes[1].set_xlabel('(b) Run Frequency\n(runs/day)')
    axes[1].yaxis.grid(color='lightgrey', linestyle=':')
    axes[1].set_axisbelow(True)
    axes[1].set_ylim(0,1)
    axes[1].set_xlim(0,3)
    axes[1].set_yticks(np.arange(0,1.2,0.25))
    positions = [1, 2, 3]
    labels = ['$10^1$', '$10^2$', '$10^3$']
    axes[1].xaxis.set_major_locator(ticker.FixedLocator(positions))
    axes[1].xaxis.set_major_formatter(ticker.FixedFormatter(labels))
    vals = axes[0].get_yticks()
    axes[1].set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    # Add minor ticks
    ticks = [1,2,3,4,5,6,7,8,9] 
    f_ticks = []
    tmp_ticks = [np.log10(x) for x in ticks]
    f_ticks = f_ticks + tmp_ticks
    tmp_ticks = [np.log10(x)+1 for x in ticks]
    f_ticks = f_ticks + tmp_ticks
    tmp_ticks = [np.log10(x)+2 for x in ticks]
    f_ticks = f_ticks + tmp_ticks
    axes[1].set_xticks(f_ticks, minor=True)
    # Add vertical lines for medians
    axes[1].axvline(np.log10(read_median), color='skyblue', zorder=0, linestyle='--', linewidth=2)
    axes[1].axvline(np.log10(write_median), color='maroon', zorder=0, linestyle=':', linewidth=2)
    # Add legend
    axes[0].legend(loc='lower right', fancybox=True)
    #axes[1].get_legend().remove()
    plt.savefig(join(save_directory, 'time_periods_freq.pdf'))
    plt.close()
    plt.clf()
    ########################################################### Run Spread ###########################################################
    rm = np.median(read_df[read_df['Range']=='1w-\n2w']['Temporal Coefficient of Variation'])
    wm = np.median(write_df[write_df['Range']=='1w-\n2w']['Temporal Coefficient of Variation'])
    # Barplot of time periods to temporal CoV
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=[5,1.8])
    fig.subplots_adjust(left=0.19, right=0.990, top=0.96, bottom=0.48, wspace=0.03)
    order = ['<1d', '1-\n3d', '3d-\n1w', '1w-\n2w', '2w-\n1M', '1-\n3M', '3-\n6M']
    PROPS = {'boxprops':{'facecolor':'skyblue', 'edgecolor':'black'}, 'medianprops':{'color':'black'},
            'whiskerprops':{'color':'black'},'capprops':{'color':'black'}}
    sns.boxplot(ax=axes[0], x='Range', y='Temporal Coefficient of Variation', data=read_df, order=order, color='skyblue', fliersize=0, **PROPS)
    PROPS = {'boxprops':{'facecolor':'maroon', 'edgecolor':'black'}, 'medianprops':{'color':'white', 'linewidth': 1.25},
            'whiskerprops':{'color':'black'},'capprops':{'color':'black'}}
    sns.boxplot(ax=axes[1], x='Range', y='Temporal Coefficient of Variation', data=write_df, order=order,color='maroon', fliersize=0, **PROPS)
    # iterate over boxes
    for i,box in enumerate(axes[0].artists):
        box.set_edgecolor('black')
    axes[0].set_ylabel('')
    axes[1].set_ylabel('')
    fig.text(0.005, 0.45, 'Inter-arrival\nTimes CoV', rotation=90)
    axes[0].set_xlabel('')
    axes[1].set_xlabel('')
    fig.text(0.38, 0.13, '(a) Read', ha='center')
    fig.text(0.80, 0.13, '(b) Write', ha='center')
    fig.text(0.58, 0.03, 'Cluster Time Span', ha='center')
    #fig.text(0.001, 0.65, "Performance\nCoV (%)", rotation=90, va='center', multialignment='center')
    axes[0].yaxis.grid(color='lightgrey', linestyle=':')
    axes[1].yaxis.grid(color='lightgrey', linestyle=':')
    axes[0].set_axisbelow(True)
    axes[1].set_axisbelow(True)
    axes[0].set_yticks([0,1000,2000,3000])
    #axes[0].set_ylim(0,3000)
    plt.savefig(join(save_directory, 'time_period_v_temp_cov.pdf'))
    plt.close()
    plt.clf()
    return None

def io_performance_variability():
    '''
    overall I/O performance variability, day of week, time of day, time span, 
    I/O amount, check is cluster size is factor, temporal phasing.
    '''
    return None
