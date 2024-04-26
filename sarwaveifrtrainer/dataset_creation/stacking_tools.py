import os
import glob
import pandas as pd

get_safename = lambda x: x.split(os.sep)[-2]


def select_csv_files(root_path, pol, burst_type):
    """
    Selects CSV files based on the specified polarization and burst type.

    Parameters:
    - root_path (str): The root path where the CSV files are located.
    - pol (str): The polarization value to filter CSV files.
    - burst_type (str): The burst type value to filter CSV files. 
    
    Returns:
    - list of str: A list of file paths matching the criteria.
    """
    
    csv_files = glob.glob(os.path.join(root_path, '*', burst_type, f'*-{pol.lower()}-*.csv'))
    return csv_files    


def create_dataframe(client, csv_files):
    """
    Creates a Pandas DataFrame by reading CSV files using Dask distributed computing.

    Parameters:
    - client: Dask distributed client.
    - csv_files (list of str): List of CSV file paths to read.

    Returns:
    - pd.DataFrame: Concatenated DataFrame from the CSV files.
    """
    dfs_future = client.map(pd.read_csv, csv_files)
    dfs = client.gather(dfs_future)
    
    df = pd.concat(dfs, ignore_index=True)
    return df