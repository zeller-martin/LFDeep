import deep_loop
import glob


batch_size = 1024
batches_per_epoch = 128
size = 1024

x_files = glob.glob('data/*_raw.float32')
y_files = glob.glob('data/*_phase.float32')
                    
x_train, y_train, x_val, y_val = deep_loop.split_data(x_files, y_files)

training_generator = deep_loop.DeepLoopGenerator(x_train, y_train, batch_size, batches_per_epoch, size)
validation_generator = deep_loop.DeepLoopGenerator(x_val, y_val, batch_size, batches_per_epoch, size)
