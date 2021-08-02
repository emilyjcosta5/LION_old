# Author: Emily Costa
# Created on: Jun 3, 2021
# Functions for analyzing the I/O behaviors of applications on an HPC system using the clusters generated from `clustering`.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import os 

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

def general_temporal_trends(path_to_clusters, path_to_total, save_path='./figures/general_temporal_trends.pdf', verbose=False):
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
    # Collect start and end times of all runs
    return None

def io_performance_variability():
    '''
    overall I/O performance variability, day of week, time of day, time span, 
    I/O amount, check is cluster size is factor, temporal phasing.
    '''
    return None
