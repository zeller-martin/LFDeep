import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from .layers import *


# model defaults

_default_layers = [
        layers.Conv1D(16, kernel_size= 64, padding = 'same', activation='linear'),
        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dense(2048, activation='relu'),]

_default_optimizer = keras.optimizers.Adam(learning_rate=0.001)


def circular_loss(y_true, y_pred):
    '''
    Computes the circular loss of two tensors.
    
    Positional arguments:
    y_true -- a tensor with shape (N, 1), where N is the number of ground-truth angles
    y_pred -- a tensor with shape (N, 1), where N is the number of predicted angles
    '''
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    return tf.sqrt(1.001 - tf.cos(y_pred - y_true)) # not using a flat 1 will prevent that tf.sqrt occassionaly return nan

def mean_abs_zscore_difference(y_true, y_pred):
    '''
    Performs z-scoring within a batch, and computes loss as the mean absolute difference.
    
    Positional arguments:
    y_true -- a tensor with shape (N, 1)
    y_pred -- a tensor with shape (N, 1)
    '''
    mean = tf.math.reduce_mean(y_true)
    std = tf.math.reduce_std(y_true)
    y_true = y_true - mean
    y_pred = y_pred - mean
    y_true = y_true / std
    y_pred = y_pred / std
    return tf.reduce_mean(tf.math.abs(y_true - y_pred))

# functions to create new models

def create_phase_model(input_shape = 1024,
                       middle_layers = _default_layers,
                       optimizer = _default_optimizer,
                       loss = circular_loss):
    '''
    Predict instantaneous phase of a band within a broadband signal. Uses sequential API.
    
    Keyword arguments:
    input_shape -- number of samples contained within a single input time series, default: 1024
    middle_layers -- model architecture after z-scoring and before circular output, default: see deep_loop/models.py
    optimizer -- training optimizer, default: keras.optimizers.Adam(learning_rate=0.001)
    loss -- a loss function that should be circular, default: sqrt(1 - cos(phase_predicted - phase_true ) )
    '''
    
    model_layers = [keras.Input(shape=( 1024,  1)), Zscore1D()]
    model_layers.extend(middle_layers)
    model_layers.extend([layers.Dense(2, activation='linear'), AngularOutput()])
                  
    model = keras.Sequential(model_layers)
    model.compile(optimizer = optimizer, loss = loss)
    
    return model


def create_amplitude_model(input_shape = 1024,
                           middle_layers = _default_layers,
                           optimizer = _default_optimizer,
                           loss = mean_abs_zscore_difference):
    '''
    Predict instantaneous phase of a band within a broadband signal. Uses functional API.
    
    Keyword arguments:
    input_shape -- number of samples contained within a single input time series, default: 1024
    middle_layers -- model architecture after z-scoring and before amplitude output, default: see deep_loop/models.py
    optimizer -- training optimizer, default: keras.optimizers.Adam(learning_rate=0.001)
    loss -- a loss function that should be circular, default: mean_abs_zscore_difference
    '''
    
    inputs = keras.Input(shape=( 1024,  1))
    zscoring = Zscore1D()
    std_layer = Std1D()

    graph = zscoring(inputs)
    std_res = std_layer(inputs)

    for middle_layer in middle_layers:
        graph = middle_layer(graph)

    z_amp = layers.Dense(1, activation='linear')(graph)

    output = layers.Multiply()([z_amp, std_res])

    model = keras.Model(inputs=inputs, outputs=output, name="amplitude_model")

    model.compile(optimizer= optimizer, loss = loss)
    return model
