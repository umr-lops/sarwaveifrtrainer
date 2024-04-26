import pandas as pd
import numpy as np
import datetime
import json
import pickle
import os

from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler

scaler_types = {
    'RobustScaler': RobustScaler,
    'MinMaxScaler': MinMaxScaler,
    'StandardScaler': StandardScaler
}


def generate_dataset_from_config(config):
    """
    Generate a dataset based on the provided configuration and filter condition.

    Parameters:
    - config (dict): A dictionary containing configuration parameters for generating the dataset.
    """

    # Load raw data and filter it
    try:
        print("Loading data...")
        df = pd.read_csv(config['raw_csv'])
        df = df.query(config['filter'])
    except Exception as e:
        raise ValueError(f"Error loading or filtering raw data: {e}")

    # Initialize scaler
    try:
        scaler = scaler_types[config['scaler']['name']](**config['scaler']['kwargs'])
    except KeyError:
        raise ValueError("Scaler configuration is missing or invalid.")

    # Create DatasetCreator instance
    try:
        dataset_creator = DatasetCreator(df, bin_width=config['bin_width'], transformer=scaler)
    except Exception as e:
        raise ValueError(f"Error initializing DatasetCreator: {e}")

    # Split dataframe and format input/target data
    try:
        print("Processing data...")
        train_safes = np.loadtxt(config['train_safes'], dtype=str)
        val_safes = np.loadtxt(config['val_safes'], dtype=str)
        dataset_creator.split_dataframe(train_safes, val_safes, config['kept_columns'])
        X_train, X_val = dataset_creator.format_input_data(config['target_columns'])
        y_train, y_val = dataset_creator.format_target_data(target_columns=config['target_columns'])
    except Exception as e:
        raise ValueError(f"Error processing data: {e}")

    # Save datasets and related files
    try:
        print(f"Saving data in {config['save_directory']}")
        os.makedirs(config['save_directory'], exist_ok=True)
        np.save(os.path.join(config['save_directory'], 'X_train.npy'), X_train)
        np.save(os.path.join(config['save_directory'], 'X_val.npy'), X_val)
        np.save(os.path.join(config['save_directory'], 'y_train.npy'), y_train)
        np.save(os.path.join(config['save_directory'], 'y_val.npy'), y_val)
        dataset_creator.df_train.to_csv(os.path.join(config['save_directory'], 'df_train.csv'), index=False)
        dataset_creator.df_val.to_csv(os.path.join(config['save_directory'], 'df_val.csv'), index=False)
        pickle.dump(dataset_creator.transformer, open(os.path.join(config['save_directory'], 'scaler.pkl'), 'wb'))
        with open(os.path.join(config['save_directory'], 'config.json'), 'w') as f:
            json.dump(config, f)
        for param in config['target_columns']:
            np.save(os.path.join(config['save_directory'], f'bins_{param}.npy'), dataset_creator.class_bins[param])
    except Exception as e:
        raise ValueError(f"Error saving dataset and related files: {e}")
        
def split_safes(safes, val_size=0.1, test_safes=None, seed=1):
    """
    Split safes into training and validation sets.

    Args:
        safes (str): List of safes to split.
        val_size (float): Proportion of safes to include in the validation set (between 0 and 1).
        test_safes (list): List of safes to exclude from the training and validation sets.
        seed (int): Seed for random number generation. Default is 1.

    Returns:
        tuple: A tuple containing:
            - train_safes (list): List of safes for training.
            - val_safes (list): List of safes for validation.
    """
    if test_safes is not None:
        safes = list(set(safes) - set(test_safes))

    rng = np.random.default_rng(seed=seed)
    val_safes = rng.choice(safes, int(len(safes) * val_size), replace=False)
    train_safes = list(set(safes) - set(val_safes))
    
    return train_safes, val_safes


class DatasetCreator:
    """
    A class to create datasets for machine learning tasks.

    Attributes:
    - df (pd.DataFrame): The input dataframe.
    - bin_width (float): The width of each bin for discretization (default is 0.2).
    - transformer: The transformer object for scaling the data.
    - df_train (pd.DataFrame): Training set dataframe.
    - df_val (pd.DataFrame): Validation set dataframe.
    - class_bins (dict): A dictionary to store the bins for discretizing continuous variables.

    Methods:
    - split_dataframe(train_safes, val_safes, kept_columns): Split the dataframe into training and validation sets based on the safes listing.
    - format_input_data(dropped_columns): Preprocess input data by dropping specified columns and scaling the data.
    - format_target_data(target_columns): Format target data by discretizing continuous variables into classes.
    - get_categorical(data, bins): Convert continuous data into categorical data using provided bins.
    """
    
    def __init__(self, df, bin_width=0.2, transformer=None):
        
        self.df = df
        self.bin_width = bin_width
        self.transformer = transformer
        
        self.df_train = pd.DataFrame()
        self.df_val = pd.DataFrame()
        
        self.class_bins = {}
        
        
    def split_dataframe(self, train_safes, val_safes, kept_columns):
        """
        Split a dataframe into training and validation sets based on the safes listing.

        Parameters:
        - train_safes (list): List of safes for the training set.
        - val_safes (list): List of safes for the validation set.
        - kept_columns (list): List of column names to keep in the resulting dataframes.

        Returns:
        - df_train (pd.DataFrame): Training set dataframe.
        - df_val (pd.DataFrame): Validation set dataframe.
        """
        df = self.df[kept_columns]

        df_val = df[df['safe'].isin(val_safes)]
        df_train = df[df['safe'].isin(train_safes)]

        self.df_val = df_val.reset_index(drop=True)
        self.df_train = df_train.reset_index(drop=True)
        
        
    def format_input_data(self, target_columns):
        """
        Preprocess input data for machine learning by performing the following steps:
        1. Drop specified columns from both training and validation sets.
        2. Scale the data using the given transformer.

        Parameters:
        - dropped_columns (list): List of column names to be dropped from the data.

        Returns:
        - X_train (array): Transformed training data after preprocessing.
        - X_val (array): Transformed validation data after preprocessing.
        """
        dropped_columns = target_columns + ['file_path', 'safe']
        
        X_train = self.df_train.drop(columns=dropped_columns).values
        X_val = self.df_val.drop(columns=dropped_columns).values

        if self.transformer is None:
            return X_train, X_val
        else:
            self.transformer.fit(X_train)
            X_train = self.transformer.transform(X_train)
            X_val = self.transformer.transform(X_val)
            return X_train, X_val

        
    def format_target_data(self, target_columns=['hs', 'phs0', 't0m1']):
        """
        Format target data by discretizing continuous variables into classes.

        Parameters:
        - target_columns (list of str, optional): The list of target column names to be formatted (default is ['hs', 'phs0', 't0m1']).
        """       
        for param in target_columns:
            p_min, p_max = (round_dec(self.df_train[param].min(), 1, 'down'), round_dec(self.df_train[param].max(), 1, 'up'))
            bins = np.arange(p_min, p_max+self.bin_width, self.bin_width)
            self.class_bins[param] = bins
            
        y_train = np.hstack([self.get_categorical(self.df_train[param], self.class_bins[param]) for param in target_columns])
        y_val = np.hstack([self.get_categorical(self.df_val[param], self.class_bins[param]) for param in target_columns])

        return y_train, y_val

    
    def get_categorical(self, data, bins):
        """
        Convert continuous data into categorical data using provided bins.

        Parameters:
        - data (pd.Series): The continuous data to be converted.
        - bins (array-like): The bins to use for discretization.

        Returns:
        - categorical_data (array): Categorical data after discretization.
        """
        nb_class = len(bins)-1
        targets_class = pd.cut(data, bins=bins, labels=range(nb_class), include_lowest=True).values
        return np.eye(nb_class, dtype=int)[targets_class]
        
                              
def round_dec(number, ndigits, method='round'):
    """
    Round the input number to the specified number of decimal places using different rounding methods.

    Parameters:
    - x (float): The number to be rounded.
    - decimal (int, optional): The number of decimal places to round to (default is 0).
    - method (str, optional): The rounding method to use. It can be one of the following:
        - 'nearest' (default): Round to the nearest integer or specified decimal place.
        - 'up': Round up towards positive infinity.
        - 'down': Round down towards negative infinity.

    Returns:
    float: The rounded number.

    Examples:
    >>> round_dec(3.14159)
    3.0

    >>> round_dec(3.14159, decimal=2, method='up')
    3.15

    >>> round_dec(3.14159, decimal=1, method='down')
    3.1
    """
    if method == 'down':
        return np.floor(number * 10**ndigits) / 10**ndigits
    elif method == 'up':
        return np.ceil(number * 10**ndigits) / 10**ndigits
    else:
        return round(number, ndigits)