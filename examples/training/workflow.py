import deep_loop
import glob
from IPython import embed
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

batch_size = 8192
batches_per_epoch = 32
size = 1024

x_files = glob.glob('data/*CA1*_raw.float32')
y_files = glob.glob('data/*CA1*_gamma_hi_phase.float32')
             


test_layers = [
        layers.Conv1D(16, kernel_size= 64, padding = 'same', activation='linear'),
        #layers.Conv1D(128, kernel_size= 128, padding = 'same', activation='tanh'),
        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dense(2, activation='linear'),]

x_train, y_train, x_val, y_val = deep_loop.split_data(x_files, y_files)

training_generator = deep_loop.DeepLoopGenerator(x_train, y_train, batch_size, batches_per_epoch, size)
validation_generator = deep_loop.DeepLoopGenerator(x_val, y_val, batch_size, batches_per_epoch, size)


model = deep_loop.create_phase_model(middle_layers = test_layers)

model.summary()

model.fit(training_generator,
          validation_data = validation_generator,
          validation_steps = 32,
          steps_per_epoch = batches_per_epoch,
          epochs = 50,
          verbose = 1)


embed()