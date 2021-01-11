import time
import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
import pycircstat as pcs
from scipy import signal

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
    preprocessing_callback -- function, to be applied to training data snippets, to mimic processing steps such as filtering
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
        self._total_samples = int(self._total_samples)
        self._file_indices = np.zeros( self._total_samples )
        self._samples = np.zeros( self._total_samples )
        
        j = 0
        for i, file_len in enumerate(self._file_lengths):
            valid_indices = int(file_len - self.size - self.y_offset)
            self._file_indices[j : j + valid_indices] = i
            self._samples[j : j + valid_indices] = np.arange(valid_indices)
            j += valid_indices
            
        self._file_indices = self._file_indices.astype(int)
        self._samples = self._samples.astype(int)
    
    def _load_np_array(self, file):
        return np.fromfile(file, dtype = np.float32)
    
    def _load_memmap(self, file):
        return np.memmap(file, dtype = np.float32)
    
    def _load_data(self):
        if self.memmap:
            loading_func = self._load_memmap
        else:
            loading_func = self._load_np_array
        
        for x_f, y_f in zip(self.x_files, self.y_files):
            self.x_data.append( loading_func(x_f) )
            self.y_data.append( loading_func(y_f) )
    
    def _identity_preprocessing_callback(self, x):
        return x
    
    def __init__(self,
                 x_files, y_files,
                 batch_size, batches_per_epoch, size,
                 memmap = False,
                 y_offset = 0,
                 preprocessing_callback = None):
        
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
        
        if preprocessing_callback is None:
            self._preprocessing_callback = self._identity_preprocessing_callback
        else:
           self._preprocessing_callback = preprocessing_callback
        
        self._create_index_map()
        self._load_data()
        
        
    def __len__(self) :
        return self.batches_per_epoch
      
      
    def __getitem__(self, idx) :
        self.idx_x = np.random.choice(self._total_samples, self.batch_size)

        batch_x = list()
        batch_y = list()
        
        for ix in self.idx_x:
            f_idx = self._file_indices[ix]
            sample_x = self._samples[ix]
            sample_y = sample_x + self.size - 1 + self.y_offset
            
            x_example = np.array(self.x_data[f_idx][sample_x : sample_x + self.size]).flatten()
            x_example = self._preprocessing_callback(x_example)
            
            batch_x.append(x_example.reshape(( self.size,1 )) )

            batch_y.append(np.array(
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
    
    x_train = [x_files[i] for i in indices[n_validation_files:]]
    y_train = [y_files[i] for i in indices[n_validation_files:]]
    
    x_val = [x_files[i] for i in indices[:n_validation_files]]
    y_val = [y_files[i] for i in indices[:n_validation_files]]
    
    return x_train, y_train, x_val, y_val

def evaluate_phase_model(model, validation_generator):
    x, y = validation_generator[0]
    y_pred = model(x).numpy()
    cdiff = pcs.cdiff(y, y_pred)
    plt.hist(cdiff, bins = 20)
    mace = np.mean(np.abs(cdiff))
    print('MACE: {mace}')
    plt.show()


def evaluate_amplitude_model(model, validation_generator):
    pass


## training_dataset function

### data generators



def hilbert_phase(y):
    return np.angle(signal.hilbert(y))

def trough_to_trough_phase(y):
    peaks, _ = signal.find_peaks(y)
    troughs, _ = signal.find_troughs(-y)
    
    is_peak = np.zeros(peaks.shape[0] + troughs.shape[0], dtype = np.bool_)
    is_peak[:peaks.shape[0]] = True
    ind = np.zeros(peaks.shape[0] + troughs.shape[0],dtype=np.uint32)
    ind[:peaks.shape[0]] = peaks
    ind[peaks.shape[0]:] = troughs
    sorting = np.argsort(ind)
    ind = ind[sorting]
    is_peak = is_peak[sorting]
    phase = 0 * x
    
    for (i1, i2) , p in zip(zip(ind[:-1], ind[1:]), is_peak[:-1]):
        lin = (np.arange(i2 - i1) / (i2 - i1)) * np.pi
        if p:
            phase[i1:i2] = lin
        else:
            phase[i1:i2] = lin - np.pi
    
    return phase
    
def hilbert_amplitude(y):
    return np.abs(signal.hilbert(y))

def _make_gauss_kernel(dt, sigma, width = 5):
    n_timesteps = 2 * width * sigma / dt + 1
    time_axis = np.arange(n_timesteps) * dt
    time_axis -= np.median(time_axis)
    y = np.exp(-time_axis**2 / (2 * sigma**2))
    y /= np.sum(y)
    return y

def rms_amplitude(y,
                  sampling_rate = None,
                  kernel_sd_seconds = None):
    
    if None in (sampling_rate, kernel_sd_second):
        raise ValueError('sampling_rate and kernel_sd_seconds have to be specified!')

    kernel = _make_gauss_kernel(1 / sampling_rate, kernel_sd_seconds)
    return signal.oaconvolve(y**2, kernel, mode = 'same')**.5



def phase_training_files(src_file,
                         n_channels = 1,
                         selected_channel = 0,
                         corner_frequencies = None,
                         sampling_rate = None,
                         phase_extraction_function = hilbert_phase,
                         dest_folder = './',
                         tag = None, 
                         src_precision = np.int16,
                         src_order = 'row'):
    
    if None in (corner_frequencies, sampling_rate):
        raise ValueError('corner_frequencies and sampling_rate have to be specified!')
    
    data = np.fromfile(src_file, dtype = src_precision)
    
    if src_order == 'row':
        data.resize(data.shape[0] // n_channels, n_channels)
        y = data[:, selected_channel]
    elif src_order in ('col', 'column'):
        data.resize(n_channels, data.shape[0] // n_channels)
        x = data[selected_channel, :]
    else:
        raise ValueError(f'src_order not understood - "row", "col" and "column" permissible, given was "{src_order}".')
        
    sos = signal.butter(4, corner_frequencies, 'bandpass', fs = sampling_rate, output = 'sos')
    x_filtered = signal.sosfiltfilt(sos, x)
    
    phase = phase_extraction_function(x_filtered)
    fbase, _ = os.path.splitext(os.path.split(src_file)[1])
    
    if tag is None:
        x_filename = '_'.join(fbase, 'rawx') + '.float32'
        y_filename = '_'.join(fbase, 'phase') + '.float32'
    else:
        x_filename = '_'.join(fbase, tag, 'rawx') + '.float32'
        y_filename = '_'.join(fbase, tag, 'phase') + '.float32'
        
    x_filename = os.path.join(dest_folder, x_filename)
    y_filename = os.path.join(dest_folder, y_filename)
    x.astype(np.float32).tofile(x_filename)
    phase.astype(np.float32).tofile(y_filename)
    
    print(f'Preprocessed {src_file}.')
    print(f'Outputs saved to {x_filename} and {y_filename}.')

    
    
    
    
