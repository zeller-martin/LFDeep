import tensorflow as tf
from tensorflow import keras
from tf.keras import layers

_default_layers = []


'''
def create_phase_model(input_shape = 1024, middle_layers = _default_layers ):
    layers = [layers.Input,
              Zscore_Timeseries ...]
    
    layers.extend(middle_layers)
    
    layers.extend([layers.Dense(2),
                  AngularOutput])
                  
    model = keras.Sequential(layers)
    return model
    
def create_amplitude_model(input_shape = 1024, middle_layers = _default_layers ):
'''
