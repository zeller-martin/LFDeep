import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from .layers import *
from .losses import *

# model defaults

_default_layers = [
        layers.Conv1D(16, kernel_size= 64, padding = 'same', activation='linear'),
        layers.Flatten(),
        layers.Dense(2048, activation='relu'),
        layers.Dense(512, activation='tanh'),]

_default_optimizer = keras.optimizers.Adam(learning_rate=0.001)
    

def _create_phase_branch(inputs, middle_layers = _default_layers):
    x = Zscore1D()(inputs)
    for layer in middle_layers:
        x = layer(x)
    x = layers.Dense(2, activation='linear')(x)
    x = AngularOutput()(x)
    return x
    
def _create_amplitude_branch(inputs, middle_layers = _default_layers):
    x = Zscore1D()(inputs)
    std_res = Std1D()(inputs)
    for layer in middle_layers:
        x = layer(x)
    x = layers.Dense(1, activation='linear')(x)
    x = layers.Multiply()([x, std_res])
    return x
    
def create_phase_model(input_shape,
                       middle_layers = _default_layers,
                       optimizer = _default_optimizer,
                       loss = circular_loss):
    '''
    Predict instantaneous phase of a band within a broadband signal.
    
    Keyword arguments:
    input_shape -- number of samples contained within a single input time series
    middle_layers -- model architecture after z-scoring and before circular output, default: see deep_loop/models.py
    optimizer -- training optimizer, default: keras.optimizers.Adam(learning_rate=0.001)
    loss -- a loss function that should be circular, default: sqrt(1 - cos(phase_predicted - phase_true ) )
    '''
    
    inputs = keras.Input(shape=( input_shape,  1))
    output = _create_phase_branch(inputs, middle_layers = middle_layers)
    model = keras.Model(inputs=inputs, outputs=output, name="phase_model")
    model.compile(optimizer = optimizer, loss = loss)
    return model


def create_amplitude_model(input_shape,
                           middle_layers = _default_layers,
                           optimizer = _default_optimizer,
                           loss = mean_abs_zscore_difference):
    '''
    Predict instantaneous amplitude of a band within a broadband signal.
    
    Keyword arguments:
    input_shape -- number of samples contained within a single input time series
    middle_layers -- model architecture after z-scoring and before amplitude output, default: see deep_loop/models.py
    optimizer -- training optimizer, default: keras.optimizers.Adam(learning_rate=0.001)
    loss -- a loss function that should be circular, default: mean_abs_zscore_difference
    '''
    
    inputs = keras.Input(shape=( input_shape,  1))
    output = _create_amplitude_branch(inputs, middle_layers = middle_layers)
    model = keras.Model(inputs=inputs, outputs=output, name="amplitude_model")
    model.compile(optimizer = optimizer, loss = loss)
    return model
    
    
def create_joint_model(input_shape,
                       middle_layers_amplitude = _default_layers,
                       middle_layers_phase = _default_layers,
                       optimizer = _default_optimizer,
                       loss_amplitude = mean_abs_zscore_difference,
                       loss_phase = circular_loss):
    '''
    Predict instantaneous amplitude and phase of a band within a broadband signal. Uses functional API.

    '''
    
    inputs = keras.Input(shape=( input_shape,  1))
    output_amplitude = _create_amplitude_branch(inputs, middle_layers = middle_layers_amplitude)
    output_phase = _create_phase_branch(inputs, middle_layers = middle_layers_phase)
    model = keras.Model(inputs=inputs, outputs= [output_amplitude, output_phase ], name="joint_model")
    model.compile(optimizer = optimizer, loss = [loss_amplitude, loss_phase])
    return model

def load_model(path):
    from .layers import Zscore1D, AngularOutput, Std1D
    custom_objects = {"Zscore1D": Zscore1D, "AngularOutput": AngularOutput, "Std1D": Std1D}
    return keras.models.load_model(path, custom_objects = custom_objects, compile = False)