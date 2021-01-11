import LFDeep
import glob
from IPython import embed
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

batch_size = 8192
batches_per_epoch = 32
size = 1024

x_files = glob.glob('data/*CA1*_raw.float32')
y_files = glob.glob('data/*CA1*_theta_phase.float32')
             

x_train, y_train, x_val, y_val = LFDeep.split_data(x_files, y_files)

training_generator = LFDeep.DataGenerator(x_train, y_train, batch_size, batches_per_epoch, size)
validation_generator = LFDeep.DataGenerator(x_val, y_val, batch_size, batches_per_epoch, size)


model = LFDeep.create_phase_model()

model.summary()

model.fit(training_generator,
          validation_data = validation_generator,
          validation_steps = 32,
          steps_per_epoch = batches_per_epoch,
          epochs = 2,
          verbose = 1)
LFDeep.evaluate_phase_model(model, validation_generator)


embed()
