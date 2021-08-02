from data_collection import collect_darshan_data
from clustering import cluster_runs
from analysis_and_plots import cluster_characteristics, general_temporal_trends, io_performance_variability
import pandas as pd
if __name__=='__main__':
    path_to_total    = './sample_darshan_data/parsed'
    path_to_data     = './sample_darshan_data/run_info.parquet'
    path_to_clusters = './sample_darshan_data/cluster_info.parquet' 
    '''
    run_info = collect_darshan_data(path_to_total, save_path=path_to_data, verbose=True)
    print(run_info)
    run_info = pd.read_parquet(path_to_data)
    cluster_info = cluster_runs(run_info, threshold=5, save_path=path_to_clusters, verbose=True)
    '''
    cluster_info = pd.read_parquet(path_to_clusters)
    #print(cluster_info)
    cluster_characteristics(cluster_info, save_path='./figures/cluster_characteristics.pdf', plot_log=False, verbose=False)

