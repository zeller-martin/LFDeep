import LFPredict
import glob

model_name = 'example_joint_model'

batch_size = 8192
batches_per_epoch = 16
size = 1024

x_files = glob.glob('data/*CA1*_raw.float32')
y_files_phase = glob.glob('data/*CA1*_theta_phase.float32')
y_files_amplitude = glob.glob('data/*CA1*_theta_amp.float32')        

(x_train, y_train_amplitude, y_train_phase,
 x_val, y_val_amplitude, y_val_phase)        = LFPredict.split_data(x_files,
                                                                    y_files_amplitude,
                                                                    y_files_phase)

training_generator = LFPredict.DataGenerator(x_train,
                                            [y_train_amplitude, y_train_phase],
                                             batch_size,
                                             batches_per_epoch,
                                             size)
validation_generator = LFPredict.DataGenerator(x_train,
                                            [y_train_amplitude, y_train_phase],
                                             batch_size,
                                             batches_per_epoch,
                                             size)


model = LFPredict.create_joint_model(1024)
model.summary()

model.fit(training_generator,
          validation_data = validation_generator,
          validation_steps = batches_per_epoch,
          steps_per_epoch = batches_per_epoch,
          epochs = 5,
          verbose = 1)

LFPredict.evaluate_joint_model(model, validation_generator, amp_idx = 0, phase_idx = 1)

model.save(model_name + '.h5')