import deep_loop
import glob
from IPython import embed
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

batch_size = 1024
batches_per_epoch = 128
size = 1024

x_files = glob.glob('data/*_raw.float32')
y_files = glob.glob('data/*_theta_phase.float32')
             

x_train, y_train, x_val, y_val = deep_loop.split_data(x_files, y_files)

training_generator = deep_loop.DeepLoopGenerator(x_train, y_train, batch_size, batches_per_epoch, size)
validation_generator = deep_loop.DeepLoopGenerator(x_val, y_val, batch_size, batches_per_epoch, size)




class ZscoreTimeseries(keras.layers.Layer):
    '''
    A layer which individually z-scores time series contained within a tensor
    
    Positional arguments:
    inputs -- a tensor with shape (N, ...), where N is the number of individual timeseries ...
    '''
    
    
    def __init__(self):
        super(ZscoreTimeseries, self).__init__()
        self.std = None
        
    def call(self, inputs):
        mean = tf.math.reduce_mean(inputs, axis = 1)
        std = tf.math.reduce_std(inputs, axis = 1)
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

def circular_loss(y_true, y_pred):
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    return 1 - tf.cos(y_pred - y_true)

opt = keras.optimizers.Adam(learning_rate=0.0005)

model = keras.Sequential(
    [
        keras.Input(shape=( 1024,  1)),
        ZscoreTimeseries(),
        layers.Conv1D(32, kernel_size= 128, padding = 'same', activation='linear'),
        layers.AveragePooling1D(pool_size = 16),
        layers.Flatten(),
        layers.Dense(8192, activation='tanh'),
        layers.Dense(8192, activation='tanh'),
        layers.Dense(8192, activation='tanh'),
        layers.Dense(2, activation='tanh'),
        AngularOutput()
    ]
)

model.compile(optimizer= opt, loss = circular_loss)
model.fit(training_generator, validation_data = validation_generator,  validation_steps=32, steps_per_epoch = batches_per_epoch, epochs = 10, verbose = 1)


embed()