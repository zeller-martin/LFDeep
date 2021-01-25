import LFPredict
import glob

model_name = 'example_phase_model'

batch_size = 8192
batches_per_epoch = 16
size = 1024

x_files = glob.glob('data/*CA1*_raw.float32')
y_files = glob.glob('data/*CA1*_theta_phase.float32')
             
x_train, y_train, x_val, y_val = LFPredict.split_data(x_files, y_files)

training_generator = LFPredict.DataGenerator(x_train, y_train, batch_size, batches_per_epoch, size)
validation_generator = LFPredict.DataGenerator(x_val, y_val, batch_size, batches_per_epoch, size)

model = LFPredict.create_phase_model(size)
model.summary()

model.fit(training_generator,
          validation_data = validation_generator,
          validation_steps = batches_per_epoch,
          steps_per_epoch = batches_per_epoch,
          epochs = 10,
          verbose = 1)
   
LFPredict.evaluate_phase_model(model, validation_generator)

model.save(model_name + '.h5')
