import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from .layers import Zscore1D, AngularOutput, ZRescale1D

_default_layers_phase = [
        layers.Conv1D(128, kernel_size= 128, padding = 'same', activation='linear'),
        #layers.Conv1D(128, kernel_size= 128, padding = 'same', activation='tanh'),
        layers.AveragePooling1D(pool_size = 16),
        layers.Flatten(),
        layers.Dense(4096, activation='tanh'),
        layers.Dense(2048, activation='tanh'),
        layers.Dense(512, activation='tanh'),
        layers.Dense(512, activation='tanh'),
        layers.Dense(2, activation='tanh'),]

_default_optimizer = keras.optimizers.Adam(learning_rate=0.0005)

def circular_loss(y_true, y_pred):
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    return tf.sqrt(1.01 - tf.cos(y_pred - y_true))

def create_phase_model(input_shape = 1024,
                       middle_layers = _default_layers_phase,
                       optimizer = _default_optimizer):
    
    layers = [keras.Input(shape=( 1024,  1)), Zscore1D()]
    layers.extend(middle_layers)
    layers.append(AngularOutput())
                  
    model = keras.Sequential(layers)
    model.compile(optimizer = optimizer, loss = circular_loss)
    
    return model


'''
def create_amplitude_model(input_shape = 1024, middle_layers = _default_layers ):
'''
