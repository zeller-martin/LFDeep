import tensorflow as tf
from tensorflow import keras


    
class ZscoreTimeseries(keras.layers.Layer):
    def __init__(self):
        super(ZscoreTimeseries, self).__init__()
        self.std = None
        
    def call(self, inputs):
        self.std =  ...
        zscored = (inputs - mean )/ std
        return zscored

class AngularOutput(keras.layers.Layer):
    def __init__(self):
        super(AngularOutput, self).__init__()
    
    def call(self, inputs):
        angle = tf.atan2(inputs[:,0],inputs[:,1])
        return tf.reshape(angle, [-1])
    
class AmplitudeRescalingOutput(keras.layers.Layer):
    def __init__(self, reference_layer):
        super(AngularOutput, self).__init__()
        self.scaling_factors = reference_layer.std
        
    def call(self, inputs):
        return inputs * self.scaling_factors

    

## z score preprocessing layer

## arctan2 layer

## z-score to actual amplitude transformation layer
