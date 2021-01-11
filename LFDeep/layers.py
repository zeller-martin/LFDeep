import tensorflow as tf
from tensorflow import keras

    
class Zscore1D(keras.layers.Layer):
    '''
    A layer which individually z-scores time series contained within a tensor
    
    Positional arguments:
    inputs -- a tensor with shape (N, T, 1), where N is the number of individual timeseries and T the number of samples.
    '''
    
    
    def __init__(self):
        super(Zscore1D, self).__init__()
        
    def call(self, inputs):
        mean = tf.math.reduce_mean(inputs, axis = 1, keepdims = True)
        std = tf.math.reduce_std(inputs, axis = 1, keepdims = True)
        return (inputs - mean) / std

class AngularOutput(keras.layers.Layer):
    '''
    A layer which transforms sine-cosine encoded angles into a phase in the interval [-pi, pi].
    
    Positional arguments:
    inputs -- a tensor with shape (N, 2), where (:, 0) contains sines and (:, 1) the cosines of N angles.
    '''
    
    def __init__(self):
        super(AngularOutput, self).__init__()
    
    def call(self, inputs):
        angle = tf.atan2(inputs[:,0],inputs[:,1])
        return tf.reshape(angle, [-1]) 

class Std1D(keras.layers.Layer):
    '''
    A layer which computes the standard eviation of time series contained within a tensor.
    
    The intended purpose is to rescale the outputs of a model that starts with a Zscore1D layer.
    
    Positional arguments:
    inputs -- a tensor with shape (N, T, 1), where N is the number of individual timeseries and T the number of samples.
    '''
    def __init__(self):
        super(Std1D, self).__init__()
        
    def call(self, inputs):
        return tf.math.reduce_std(inputs, axis = 1)
