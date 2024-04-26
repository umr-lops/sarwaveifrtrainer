import numpy as np
import datetime
import pytz
import json
import os

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Concatenate, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import CategoricalCrossentropy


class SeaParametersModel:
    """
    Sea parameters prediction model. The purpose of this class is to introduce all the necessary methods to facilitate the model training phase.

    Attributes:
        model: A Keras model for sea parameters prediction.
        X_train: Training data features.
        y_train: Training data labels.
        X_val: Validation data features.
        y_val: Validation data labels.
        nb_classes: A list of integers representing the number of classes for each output.
    """

    def __init__(self, data_folder=None,
                 X_train=None, y_train=None, X_val=None, y_val=None, nb_classes=None,
                 n_units=1024, n_layer=10, dropout_rate=0.2,
                 batch_size=128, epochs=50, patience=5
                ):
        """
        Initializes the SeaParametersModel.

        Args:
            X_train: Training data features. Default is None.
            y_train: Training data labels. Default is None.
            X_val: Validation data features. Default is None.
            y_val: Validation data labels. Default is None.
            nb_classes: A list of integers representing the number of classes for each output. Default is None.
            data_folder: Path to the folder containing data files. Default is None.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.nb_classes = nb_classes
        self.data_folder = data_folder
        
        if data_folder is not None:
            # Load data from folder
            self.load_data_from_folder(data_folder)
        
        self.n_units = n_units
        self.n_layer = n_layer
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        
        self.loss = None
        self.model = None
    
    def load_data_from_folder(self, data_folder):
        
        self.X_train = np.load(os.path.join(data_folder, 'X_train.npy'))
        self.X_val = np.load(os.path.join(data_folder, 'X_val.npy'))
        self.y_train = np.load(os.path.join(data_folder, 'y_train.npy'))
        self.y_val = np.load(os.path.join(data_folder, 'y_val.npy'))

        hs_bins = np.load(os.path.join(data_folder, 'bins_hs.npy'))
        phs0_bins = np.load(os.path.join(data_folder, 'bins_phs0.npy'))
        t0m1_bins = np.load(os.path.join(data_folder, 'bins_t0m1.npy'))
        
        self.nb_classes = [len(hs_bins)-1, len(phs0_bins)-1, len(t0m1_bins)-1]
        
    def build_model(self, loss):
        """
        Builds and compiles the Keras model for sea parameters prediction.
        """
        self.loss = loss
        
        input_shape = self.X_train.shape[1]
        X_input = Input(input_shape)

        X = Dense(self.n_units, activation='relu')(X_input)
        for i in range(self.n_layer-1):
            X = Dropout(self.dropout_rate)(X)
            X = Dense(self.n_units, activation='relu')(X)
        X_output = Concatenate()([Dense(n)(X) for n in self.nb_classes])

        model = Model(inputs=X_input, outputs=X_output)
        model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
        
        self.model = model

    def train_model(self):
        """
        Trains the sea parameters prediction model.

        Args:
            batch_size: Batch size for training.
            epochs: Number of epochs for training.
            patience: Number of epochs with no improvement after which training will be stopped.
        """
        # Add early stopping to prevent overfitting
        early_stopping = EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True)

        # Train the model
        self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=[early_stopping]
        )
        
    def save_model(self, save_directory, with_config=True, additional_informations=None):
        """
        Saves the Keras model to a file, along with the training configuration if with_config is True.

        Args:
            save_directory (str): The directory where the model will be saved.
            with_config (bool): Whether to save the configuration file. Default is True.
            additional_informations: Additional information to include in the configuration file.
        
        Returns:
            save_directory (str): The directory where the model has been saved.
        """
        paris_timezone = pytz.timezone('Europe/Paris')
        date = datetime.datetime.now(paris_timezone)
        save_directory = os.path.join(save_directory, date.strftime("%Y-%m-%d_%Hh%M"))
        os.makedirs(save_directory, exist_ok=True)
        
        self.model.save(os.path.join(save_directory, 'model.keras'))
        
        if with_config:
            if self.data_folder is None:
                print('It is impossible to save the training configuration used as data_folder was not provided as input.')
            else:
                config = {
                    'loss': str(self.loss),
                    'n_classes': self.nb_classes,
                    'training_data': self.data_folder,
                    'n_units': self.n_units,
                    'n_layer': self.n_layer,
                    'dropout_rate': self.dropout_rate,
                    'batch_size': self.batch_size,
                    'epochs': self.epochs,
                    'patience': self.patience,
                    'date': date.strftime('%d/%m/%Y %H:%M'),
                    'additional_informations': additional_informations,
                }
                with open(os.path.join(save_directory, 'config.json'), 'w') as f:
                    json.dump(config, f)
                    
        return save_directory
                    

def custom_loss(nb_classes, batch_size, h=0.01):
    """
    Returns a custom loss function that computes categorical crossentropy
    with a constraint on the cumulative probabilities over each batch.

    Parameters:
    - nb_classes (list): A list containing the number of classes for each task.
    - batch_size (int): Batch size used during training.
    - h (float): Parameter for controlling the smoothness of the constraint.

    Returns:
    - loss (function): Custom loss function.
    """
    crossentropy_loss = custom_categorical_crossentropy(nb_classes)
    cumulative_loss = cumulative_constraint_generalized(nb_classes, batch_size, h)
    
    def loss(y_true, y_pred):
        return crossentropy_loss(y_true, y_pred) + cumulative_loss(y_true, y_pred)
    
    return loss

def custom_categorical_crossentropy(nb_classes):
    """
    Returns a custom categorical crossentropy loss function for multi-parameters learning.

    Parameters:
    - nb_classes (list): A list containing the number of classes for each parameter.

    Returns:
    - loss (function): Custom categorical crossentropy loss function.
    """
    def loss(y_true, y_pred):
        loss = CategoricalCrossentropy(from_logits=True)
        total_loss = 0

        start = 0
        for num_classes in nb_classes:
            end = start + num_classes
            y_true_slice = y_true[:, start:end]
            y_pred_slice = y_pred[:, start:end]
            total_loss += loss(y_true_slice, y_pred_slice)
            start = end

        return total_loss
    
    return loss
    
def get_cumulative_probability(y_true, y_pred):
    """
    Computes the cumulative probability for a given true and predicted probability distribution.

    Parameters:
    - y_true (tensor): Targets.
    - y_pred (tensor): Predicted probability distribution.

    Returns:
    - cumulative_probabilities (tensor): Cumulative probabilities.
    """
    y_true_float = tf.cast(y_true, tf.float32)
    y_pred_softmax = tf.nn.softmax(y_pred, axis=1)
    cumsum = tf.cumsum(y_pred_softmax, axis=1)
    cumprob_vect = cumsum*y_true_float
    return tf.reduce_sum(cumprob_vect, axis=1)

def K(x):
    """
    Kernel function for the cumulative constraint.

    Parameters:
    - x (tensor): Input tensor.

    Returns:
    - The corresponding values of the input tensor.
    """
    return tf.exp(-tf.square(x) / 2) / tf.sqrt(2 * tf.constant(np.pi, dtype=tf.float32))

def cumulative_constraint_generalized(nb_classes, batch_size, h):
    """
    Parameters:
    - batch_size (int): Batch size used during training.
    - h (float): Parameter for controlling the smoothness of the constraint.

    Returns:
    - loss (function): Cumulative probabilities constraint loss function.
    """
    cumulative_loss = cumulative_constraint(batch_size, h)
    
    def loss(y_true, y_pred):
    
        total_loss = 0

        start = 0
        for num_classes in nb_classes:
            end = start + num_classes
            y_true_slice = y_true[:, start:end]
            y_pred_slice = y_pred[:, start:end]
            total_loss += cumulative_loss(y_true_slice, y_pred_slice)
            start = end
            
        return total_loss
    
    return loss

def cumulative_constraint(batch_size, h):
    
    kde_mean, kde_std = get_uniform_kde_vectors(batch_size, h, seed=1)
    def loss(y_true, y_pred):
    
        cumulative_probabilities = get_cumulative_probability(y_true, y_pred)
        
        x = tf.linspace(0.0, 1.0, 101)
        y = tf.reshape(cumulative_probabilities, (-1, 1))
        
        inputs = (x - y) / h
        y = tf.reduce_sum(K(inputs)/(h*batch_size), axis=0)

        # loss = tf.reduce_mean(tf.abs(y - tf.reduce_mean(y)))
        loss = tf.sqrt(tf.reduce_mean(tf.square((y - kde_mean)/kde_std))+1e-8)
        return loss/10
    
    return loss

def get_uniform_kde_vectors(batch_size, h, seed=1):
    rng = np.random.default_rng(seed=seed)
    
    kde = []
    n_bins = 100
    pts_per_bin = 50000
    n_run = int(n_bins*pts_per_bin/batch_size) + 1
    for i in range(n_run):
        cumulative_probabilities = rng.uniform(size=(batch_size))

        x = np.linspace(0, 1, n_bins+1)
        y = np.reshape(cumulative_probabilities, (-1, 1))

        inputs = (x - y) / h
        kde.append(np.sum(K_numpy(inputs)/(h*batch_size), axis=0))
        
    mean_kde = np.mean(kde, axis=0)
    std_kde = np.std(kde, axis=0)
    
    return mean_kde, std_kde

def K_numpy(x):
    return np.exp(-np.square(x) / 2) / np.sqrt(2 * np.pi)