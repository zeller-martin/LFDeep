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
y_files = glob.glob('data/*CA1*_theta_amp.float32')
             

x_train, y_train, x_val, y_val = deep_loop.split_data(x_files, y_files)

training_generator = deep_loop.DeepLoopGenerator(x_train, y_train, batch_size, batches_per_epoch, size)
validation_generator = deep_loop.DeepLoopGenerator(x_val, y_val, batch_size, batches_per_epoch, size)


model = deep_loop.create_amplitude_model()

model.summary()

model.fit(training_generator,
          validation_data = validation_generator,
          validation_steps = 32,
          steps_per_epoch = batches_per_epoch,
          epochs = 100,
          verbose = 1)

deep_loop.evaluate_amplitude_model(model, validation_generator)

embed()
