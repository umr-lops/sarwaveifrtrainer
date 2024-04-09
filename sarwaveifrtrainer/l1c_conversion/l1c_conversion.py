ASA_WVI_1PNP"""
Developped for the version 4.1 of l1c processor.
"""

import os
import glob
import pandas as pd
import xarray as xr


class L1CConverter:
    def __init__(self, path_l1c_subswath, root_savepath=None, selected_vars=['hs', 'phs0', 't0m1'], burst_type='intraburst'):
        """
        Initialize the L1C_Converter.

        Parameters:
        - path_l1c (str): Path to the l1c subswath file.
        - root_savepath (str): Root path where the converted file will be saved.
        - selected_vars (list): List of selected variables to extract from the dataset.
        - burst_type (str): Either 'interburst' or 'intraburst'.
        """
        self.path = path_l1c_subswath
        self.root_savepath = root_savepath
        self.selected_vars = selected_vars
        self.burst_type = burst_type

        
    def converter(self, save=True):
        """
        Convert the l1c subswath file to a dataframe.
        
        Parameters
        - save (boolean): If True the converted file is saved to a CSV. If False, the converted dataframe is returned. Defaults to True.

        Returns:
        - None if the file cannot be converted. Else, see save parameter.
        """

        ds = xr.open_dataset(self.path, group=self.burst_type)
        res = self.get_dataframe(ds)

        if res is not None:
            
            if save:
                self.save_dataframe(res)
                
            else:
                return res

    
    def get_dataframe(self, ds):
        """
        Convert Sentinel-1 Level-1C sub-swath data to a pandas DataFrame.

        Parameters:
        - ds (xarray.Dataset): Input Sentinel-1 Level-1C sub-swath dataset.
        - path (str): File path corresponding to the dataset.

        Returns:
        - pd.DataFrame : Returns a pandas DataFrame containing the extracted information from the input dataset.
        """
        if not set(self.selected_vars).issubset(ds.keys()):
            print(f'All Variables not found in {self.path}')
            return None
            
        ds = ds.drop_vars('crs')
        ds = ds[self.selected_vars]
        df_res = ds.drop_dims(['k_gp', 'phi_hf']).squeeze().to_dataframe().reset_index(drop=True).drop(columns=['spatial_ref', 'sample', 'line', 'pol'], errors='ignore')

        for _phi_hf in ds['phi_hf'].values:
            for _k_gp in ds['k_gp'].values:
                df = ds['cwave_params'].sel(k_gp=_k_gp, phi_hf=_phi_hf).to_dataframe().reset_index(drop=True).drop(columns=['k_gp', 'phi_hf', 'spatial_ref', 'sample', 'line', 'pol'], errors='ignore')
                df = df.rename(columns={'cwave_params': f'cwave_params_k_gp={_k_gp}_and_phi_hf={_phi_hf}'})
                df_res = df_res.merge(df, on=['longitude', 'latitude'])

        df_res = df_res.assign(file_path=self.path)
        return df_res
    
        
    def save_dataframe(self, df):
        """
        Save the converted data to a CSV file.

        Parameters:
        - df (pandas.DataFrame): Converted l1c subswath data.

        The CSV file is saved with the same base name as the corresponding NetCDF file.
        """
        
        safe_name, file_name = self.path.split(os.sep)[-2:]
        savepath = os.path.join(self.root_savepath, safe_name, self.burst_type, file_name).replace('.nc', '.csv')

        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        df.to_csv(savepath, index=False)