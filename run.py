from data_collection import collect_darshan_data
from clustering import cluster_runs
from analysis_and_plots import cluster_characteristics, general_temporal_trends, io_performance_variability
import pandas as pd
from os.path import join

if __name__=='__main__':
    path_to_total    = './example/parsed_darshan_logs'
    path_to_data     = './example/outputs/data/run_info.parquet'
    path_to_clusters = './example/outputs/data/cluster_info.parquet' 
    
    # Collect info
    run_info = collect_darshan_data(path_to_total, save_path=path_to_data, verbose=True)
    print(run_info)

    # Cluster runs
    run_info = pd.read_parquet(path_to_data)
    cluster_info = cluster_runs(run_info, threshold=5, save_path=path_to_clusters, verbose=True)
     
    # Analysis and Plotting
    cluster_info    = pd.read_parquet(path_to_clusters)
    path_to_figures = './example/figures'
    path_to_cc      = join(path_to_figures, 'cluster_characteristics.pdf')
    cluster_characteristics(cluster_info, save_path=path_to_cc, plot_log=False, verbose=False)

