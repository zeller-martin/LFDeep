import tensorflow as tf
from tensorflow import keras
from tf.keras import layers


_default_layers = []


def circular_loss(y_true, y_pred):
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    return tf.sqrt(1 - tf.cos(angle - y_true))



'''
def create_phase_model(input_shape = 1024, middle_layers = _default_layers, learning_rate = 0.001):
    layers = [layers.Input,
              Zscore_Timeseries ...]
    
    layers.extend(middle_layers)
    
    layers.extend([layers.Dense(2),
                  AngularOutput])
                  
    model = keras.Sequential(layers)
    
    opt = ..
    
    model.compile(optimizer = opt, loss = circular_loss)
    
    return model
    
def create_amplitude_model(input_shape = 1024, middle_layers = _default_layers ):
'''
