import time
import tensorflow as tf
from tensorflow import keras
import os
import numpy as np


class DeepLoopGenerator(keras.utils.Sequence) :
    '''
    Generator to supply DeepLoop models with training or validation data
    
    Positional arguments for __init__:
    x_files -- list, paths to binary files (float32), which contain input data
    y_files -- list, paths to binary files (float32), which contain output data
    batch_size -- int, number of training examples in a batch
    batches_per_epoch -- int, number of batches per epoch
    size -- int, number of samples to predict from
    
    Keyword arguments for __init__:
    memmap -- bool, whether to use memmap instead of array for large datasets (default: False)
    y_offset -- int, a known system delay to enhance instantaneous predictions (default: 0).
                Example: if new data is acquired with a constant delay of 10 ms at a sampling rate of 1 kHz, y_offset = 10 could improve real-time performance
    '''
    
    
    
    def _number_of_samples(self, file, bytes_per_sample = 4):
        return os.path.getsize(file) // 4
    
    def _create_index_map(self):
        self._file_lengths = np.zeros( self._n_files )
        
        for i, (x_f, y_f) in enumerate( zip(self.x_files, self.y_files) ):
            n_samples = self._number_of_samples(x_f)
            assert n_samples == self._number_of_samples(y_f), f"The files {x_f} and {y_f} do not have equal number of samples."
            assert n_samples > self.size, f"The files {x_f} and {y_f} are too short (minimum length: size = {self.size})."
            self._file_lengths[i] = n_samples
        
        self._total_samples = np.sum(self._file_lengths) - self._n_files * (self.size + self.y_offset)
        
        self._file_indices = np.zeros( self._total_samples )
        self._samples = np.zeros( self._total_samples )
        
        j = 0
        for i, file_len in enumerate(self.file_lengths):
            valid_indices = file_len - self.size - self.y_offset
            self._file_indices[j : j + valid_indices] = i
            self._samples[j : j + valid-indices] = np.arange(valid_indices)
            j += valid_indices
    
    def _load_np_array(self, file):
        return np.fromfile(file, dtype = np.float32)
    
    def _load_memmap(self, file):
        return np.memmap(file, dtype = np.float32)
    
    def _load_data(self):
        if self.memmap:
            loading_func = self._load_memmap
        else:
            loading_func - self._load_np_array
        
        for x_f, y_f in zip(self.x_files, self.y_files):
            self.x_data.append( loading_func(x_f) )
            self.y_data.append( loading_func(y_f) )
    
        
    
    def __init__(self,
                 x_files, y_files,
                 batch_size, batches_per_epoch, size,
                 memmap = False, y_offset = 0):
        
        self.x_files = x_files
        self.y_files = y_files
        
        self.x_data = list()
        self.y_data = list()
        
        assert len(self.x_files) == len(self.y_files), f"Length of x_files ({len(self.x_files)}) should be equal to length of y_files ({len(self.y_files)})!"
        self._n_files = len(self.x_files)
        
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.size = size
        self.memmap = memmap
        self.y_offset = y_offset
        
        self._create_index_map()
        self._load_data()
        
        
    def __len__(self) :
        return self.batches_per_epoch
      
      
    def __getitem__(self, idx) :
        idx_x = np.random.choice(self._total_samples, self.batch_size)
        idx_y = idx_x + self.size - 1 + self.y_offset
        
        batch_x = list()
        batch_y = list()
        
        for ix, iy in zip(idx_x, idx_y):
            f_idx = self._file_indices[ix]
            sample_x = self._samples[ix]
            sample_y = self._samples[iy]
            
            
            batch_x.append(np.array(
                            self.x_data[f_idx][sample_x : sample_x + self.size]).resize(( self.size,1 ))
                          )

            batch_x.append(np.array(
                            self.y_data[f_idx][sample_y])
                          )

        return np.array(batch_x), np.array(batch_y)

        
def split_data(x_files, y_files, split = .2):
    '''
    Helper function to produce training and validation datasets.
    
    Positional arguments for __init__:
    x_files -- list, paths to binary files (float32), which contain input data
    y_files -- list, paths to binary files (float32), which contain output data
    '''
    
    n_files = len(x_files)
    assert n_files == len(y_files), f"Length of x_files ({n_files}) should be equal to length of y_files ({len(y_files)})!"
    
    n_validation_files = int( n_files * split )
    
    indices = np.random.permutation(n_files)
    
    x_train = x_files[indices[n_validation_files:]]
    y_train = y_files[indices[n_validation_files:]]
    
    x_val = x_files[indices[:n_validation_files]]
    y_val = y_files[indices[:n_validation_files]]
    
    return x_train, y_train, x_val, y_val

def circular_loss(y_true, y_pred):
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    return tf.sqrt(1 - tf.cos(angle - y_true))
  
## training_dataset function

### data generators


