# LION

As I/O demand of scientific applications increases, identifying, predicting, and analyzing I/O behaviors is critical to ensure parallel storage systems are efficiently utilized. In order to address this issue, Library for I/O Nerds (LION) is a comprehensive library for analyzing Darshan logs collected on high-performance computing (HPC) systems and understand the I/O nature of the applications running on a production HPC system.

## Software and Dependencies

Darshan is available here: https://www.mcs.anl.gov/research/projects/darshan/download/
Darshan util installation instructions needed for parsing files: https://www.mcs.anl.gov/research/projects/darshan/docs/darshan-util.html
Python dependies: scikit-learn, scipy, seaborn, pyarrow, numpy, pandas, matplotlib

## Usage and Instructions
The three major steps in identifying I/O behaviors are (1) setting up the logs to be accessible, (2) collecting information from the runs for clustering, (3) clustering the runs, and (4) analysis. These steps must be completed in order and are provided below.
### 1. Setup Darshan Logs
To use the Darshan logs, we first untar and parse them in order to access the logs of each run. This should be done such that the output files are all in one directory so they may be used. Once untared, the Darshan logs should be parsed and placed as so, such that the `--total` flag is used:
`darshan-parser --total`. Example Darshan logs that have been extract are included in `example/parsed_darshan_logs`. An example script for extracting and parsing the Darshan logs into a new directory is as follows:

`
for FILE in *.darshan
do
	darshan-parser --total $FILE >> $FILE-parsed
done
`
### 2. Collect I/O Info
After parsing the Darshan logs, we can now begin running our Python scripts. Using `LION.data_collection.collect_darshan_data` we can now scrap the necessary data for clustering the files by run I/O characteristics. This function takes the following inputs:

`   path_to_total: string
        Path to Darshan logs parsed in the 'total' format. The logs should be
        sorted by user ID and executable name.


    ranks: int, optional
        Parallize the data collection by increasing the number of processes
        collecting and saving the data.


    save_path: string, optional
        Where to save the collected data.


    chunksize: int, optional
        In order to checkpoint data and continue if data collection is halted,
        set this to the number of runs to collect info on per "chunk". This will
        get the program to write the info to the output file with the final info.


    verbose: boolean, optional
        For debugging and info on amount of info being collected.`
  
For scraping massive data on HPC systems, it is useful to scale it to the number of cores of the node in use, which will automatically be done unless manually inputting the number of ranks. Additionally, the run information is automatically checkpointed by chunksize and will restart where left off if the function is abruptly interrupted. Setting verbose to true may help in tracking the number of runs which had information collected and the state of the program. 
### 3. Cluster Runs
Next, using the information collected in the previous step, we can cluster the runs based on application, user, and I/O characteristics using the function from `LION.clustering.cluster_runs`. This function outputs a parquet file with the cluster information, whether in a read or write cluster, for all the runs collected by Darshan. The following are the inputs for the `cluster_runs` function:

`
    run_info: pd.DataFrame
        Dataframe containing the run info for clustering, as collected and 
        returned from a function in data_collection. Needs application, time, 
        and cluster parameters.

    threshold: int, optional
        The threshold for how many times an application needs to be run in 
        order to be included in the data collection.


    ranks: int, optional
        Parallize the data collection by increasing the number of processes
        collecting and saving the data.


    save_path: string, optional
        Where to save the collected data.


    chunksize: int, optional
        In order to checkpoint data and continue if data collection is halted,
        set this to the number of runs to collect info on per "chunk". This will
        get the program to write the info to the output file with the final info.


    verbose: boolean, optional
        For debugging and info on amount of info being collected.`

The `run_info` input is a DataFrame with the information collected and saved in the `collect_darshan_data` function. Similarly to the previous function, this function can scale to the HPC node in use and implements tiling. The threshold is the minimum number of runs in a cluster needed for that cluster to be saved and can be adjusted based on the user's expectation during the analysis (ex. one might want to reduce the threshold in a smaller production system in order to include a higher ratio of applications ran). 
### 4. Analyze and Plot I/O Behavior
Finally, once our runs are clustered such that runs in a cluster use the same application, have the same I/O operation (read or write), and exhibit similar I/O characteristics, we can analyze the clusters and runs within them. We break this down in a few functions: characterizing the cluster, general I/O trends, and I/O performance variability. In order to generate figures, input these two parameters for any or all of the functions:

`
    clustered_runs: pd.DataFrame
        DataFrame containing run information, including how they are clustered, as
        generated from cluster_runs.


    save_directory: string, optional
        Where to save the output data or generated figures. Defaults to current
        working directory.
`

## Example
To run an example that uses a week of Darshan logs from a production HPC system, scripts are available in `/example`. Examples Darshan logs are included in `unparsed_darshan_logs` which are available for trying step (1). Tthe run script uses logs that are decompressed and parsed already in `parsed_darshan_logs` and therefore starts at step (2). Simple run 'source run.sh' in that directory. The clustering information will be saved to `example/data` and figures will be save to `example/figures`.
