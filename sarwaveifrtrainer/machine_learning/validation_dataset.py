import numpy as np
import pandas as pd
import xarray as xr
import json
import os

class ValidationXarrayDataset:
    
    def __init__(self, keras_model, data_folder=None):
        """
        Initialize ModelValidation class.

        Args:
            keras_model: Keras model to validate.
            data_folder (str): Path to the folder containing data files.
        """
        data = read_json(os.path.join(data_folder, 'config.json'))
        
        self.data_folder = data_folder
        self.load_data()
        self.model = keras_model
        self.predictions = self.model.predict(self.X_val)

    def load_data(self):
        """
        Load data from files.
        """
        data = read_json(os.path.join(self.data_folder, 'config.json'))
        self.df_val = pd.read_csv(os.path.join(self.data_folder, 'df_val.csv'))
        self.X_val = np.load(os.path.join(self.data_folder, 'X_val.npy'))
        self.parameters = data.get('target_columns', [])
        self.class_bins = {f'bins_{param}': np.load(os.path.join(self.data_folder, f'bins_{param}.npy')) for param in self.parameters}

    def generate_validation_dataset(self):
        """
        Generate validation dataset.

        Returns:
            xr.Dataset: Validation dataset.
        """
        variables = {}
        coordinates = {}
        
        start = 0
        for param in self.parameters:
            param_bin = self.class_bins[f'bins_{param}']
            end = start + len(param_bin) -1
            coord_name = f"{param}_mid"
            var_name = f"{param}_logits"
            coordinates[coord_name] = ([f"{param}_mid"], (param_bin[:-1] + param_bin[1:])/2)
            variables[var_name] = (["nb_predictions", f"{param}_mid"], self.predictions[:, start:end])
            start = end
        coordinates["nb_predictions"] = (["nb_predictions"], list(range(self.predictions.shape[0])))
        
        ds = xr.Dataset(data_vars=variables, coords=coordinates)
        
        for param in self.parameters:
            ds[f'{param}_pdf'] = xr.apply_ufunc(lambda x: self.softmax(x, T=1),
                                          ds[f'{param}_logits'],
                                          input_core_dims=[[f'{param}_mid']],
                                          output_core_dims=[[f'{param}_mid']],
                                          vectorize=True)
            
            ds[f'{param}_mean'] = self.compute_values(ds, param, self.compute_mean)
            ds[f'{param}_most_likely'] = self.compute_values(ds, param, self.get_most_likely)
            ds[f'{param}_std'] = self.compute_values(ds, param, self.compute_std, True)
            ds[f'{param}_ww3'] = (('nb_predictions'), self.df_val[param].values) 
            
        return ds
    
    def add_variables(self, ds, variables):
        """
        Adds variables of interest for validation to the validation netCDF dataset.

        Args:
            ds (xarray.Dataset): The dataset to which variables will be added.
            variables (list): A list of variable names to be added.

        Returns:
            xarray.Dataset: The dataset with added variables.
        """
        for v in variables:
            ds[v] = (('nb_predictions'), self.df_val[v].values) 
        return ds
        
    @staticmethod
    def compute_values(ds, var, function, vectorize=False):
        """
        Compute values for the given variable using the given function.

        Args:
            ds (xr.Dataset): Dataset containing the data.
            var (str): Variable name.
            function (callable): Function to compute values.
            vectorize (bool): Whether to vectorize the computation.

        Returns:
            xr.DataArray: Computed values.
        """
        values = xr.apply_ufunc(function,
                                ds[f'{var}_mid'], ds[f'{var}_pdf'], 
                                input_core_dims=[[f'{var}_mid'],[f'{var}_mid']],
                                vectorize=vectorize)
        return values 
    
    @staticmethod
    def softmax(logits, T=1):      
        """
        Compute softmax values for logits.

        Args:
            logits (np.ndarray): Logits array.
            T (float): Temperature parameter.

        Returns:
            np.ndarray: Softmax values.
        """
        exp_logits = np.exp(logits/T)
        return exp_logits/np.sum(exp_logits)
    
    @staticmethod
    def get_most_likely(x, y):
        """
        Get the maximum of probability for each prediction.

        Args:
            x (np.ndarray): Input values.
            y (np.ndarray): Probabilities.

        Returns:
            np.ndarray: Most likely values.
        """
        i_max = np.argmax(y, axis=1)
        most_likely = x[i_max]
        return most_likely
    
    @staticmethod
    def compute_mean(x, y):
        """
        Compute the expected value.

        Args:
            x (np.ndarray): Input values.
            y (np.ndarray): Probabilities.

        Returns:
            np.ndarray: Expected value.
        """
        return np.sum(x * y, axis=1)
    
    @staticmethod
    def compute_std(x, y):
        """
        Compute the standard deviation.

        Args:
            x (np.ndarray): Input values.
            y (np.ndarray): Probabilities.

        Returns:
            np.ndarray: Standard deviation.
        """
        mean = np.sum(x * y)
        variance = np.sum(y * (x - mean) ** 2)
        return np.sqrt(variance)

def read_json(file):
    """
    Reads JSON data from a file and returns it as a Python dictionary.

    Args:
        file (str): The path to the JSON file to be read.

    Returns:
        dict: A dictionary containing the JSON data read from the file.
    """
    with open(file) as json_data:
        data = json.load(json_data)
    return data