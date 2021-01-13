import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from .layers import *
from .losses import *

_default_layers = [
        layers.Conv1D(16, kernel_size= 64, padding = 'same', activation='linear'),
        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dense(2048, activation='relu'),]

_default_optimizer = keras.optimizers.Adam(learning_rate=0.001)


def create_phase_branch(inputs, middle_layers = _default_layers):
    x = Zscore1D()(inputs)
    for layer in middle_layers:
        x = layer(x)
    x = layers.Dense(2, activation='linear')(x)
    x = AngularOutput()(x)
    return x
    
def create_amplitude_branch(inputs, middle_layers = _default_layers):
    x = Zscore1D()(inputs)
    std_res = Std1D()(inputs)
    for layer in middle_layers:
        x = layer(x)
    x = layers.Dense(1, activation='linear')(x)
    x = layers.Multiply()([x, std_res])
    return x
    
def create_phase_model(input_shape,
                       middle_layers = _default_layers,
                       optimizer = _default_optimizer,
                       loss = circular_loss):
    '''
    Predict instantaneous phase of a band within a broadband signal. Uses sequential API.
    
    Keyword arguments:
    input_shape -- number of samples contained within a single input time series, default: 1024
    middle_layers -- model architecture after z-scoring and before circular output, default: see deep_loop/models.py
    optimizer -- training optimizer, default: keras.optimizers.Adam(learning_rate=0.001)
    loss -- a loss function that should be circular, default: sqrt(1 - cos(phase_predicted - phase_true ) )
    '''
    
    input = keras.Input(shape=( input_shape,  1))
    output = create_phase_branch(input)
    model = keras.Model(inputs=input, outputs=output, name="phase_model")
    model.compile(optimizer = optimizer, loss = loss)
    return model


def create_amplitude_model(input_shape = 1024,
                           middle_layers = _default_layers,
                           optimizer = _default_optimizer,
                           loss = mean_abs_zscore_difference):
    '''
    Predict instantaneous amplitude of a band within a broadband signal. Uses functional API.
    
    Keyword arguments:
    input_shape -- number of samples contained within a single input time series, default: 1024
    middle_layers -- model architecture after z-scoring and before amplitude output, default: see deep_loop/models.py
    optimizer -- training optimizer, default: keras.optimizers.Adam(learning_rate=0.001)
    loss -- a loss function that should be circular, default: mean_abs_zscore_difference
    '''
    
    input = keras.Input(shape=( input_shape,  1))
    output = create_amplitude_branch(input)
    model = keras.Model(inputs=input, outputs=output, name="amplitude_model")
    model.compile(optimizer = optimizer, loss = loss)
    return model
    
def create_joint_model(input_shape,
                       middle_layers_amplitude = _default_layers,
                       middle_layers_amplitude = _default_layers,
                       optimizer = _default_optimizer,
                       loss_amplitude = mean_abs_zscore_difference,
                       loss_phase = circular_loss):
    '''
    Predict instantaneous amplitude and phase of a band within a broadband signal. Uses functional API.

    '''
    
    input = keras.Input(shape=( 1024,  1))
    output_amplitude = create_amplitude_branch(input)
    output_phase = create_amplitude_branch(input)
    model = keras.Model(inputs=inputs, outputs= [output_amplitude, output_phase ], name="joint_model")
    model.compile(optimizer = optimizer, loss = [loss_amplitude, loss_phase])
    return model





class DataGenerator(keras.utils.Sequence) :
    '''
    Generator to supply LFDeep models with training or validation data
    
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
        
        for i, (x_f, a_f, p_f) in enumerate( zip(self.x_files, self.amp_files, self.phase_files) ):
            n_samples = self._number_of_samples(x_f)
            assert n_samples == self._number_of_samples(a_f) == self._number_of_samples(p_f), f"The files {x_f}, {a_f} and {p_f} do not have equal number of samples."
            assert n_samples > self.size, f"The files {x_f}, {y_f} and {p_f} are too short (minimum length: size = {self.size})."
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
        
        for x_f, a_f, p_f in zip(self.x_files, self.amp_files, self.phase_files):
            self.x_data.append( loading_func(x_f) )
            self.amp_data.append( loading_func(a_f) )
            self.phase_data.append( loading_func(p_f) )
    
    def _identity_preprocessing_callback(self, x):
        return x
    
    def __init__(self,
                 x_files, amp_files, phase_files
                 batch_size, batches_per_epoch, size,
                 memmap = False,
                 y_offset = 0,
                 preprocessing_callback = None):
        
        self.x_files = x_files
        self.amp_files = amp_files
        self.phase_files = phase_files
        self.x_data = list()
        self.amp_data = list()
        self.phase_data = list()
        
        assert len(self.x_files) == len(self.y_files), f"Length of x_files ({len(self.x_files)}) should be equal to length of amp_files ({len(self.amp_files)}) and phase_files ({len(self.phase_files)}) !"
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
        batch_amp = list()
        batch_phase = list()
        for ix in self.idx_x:
            f_idx = self._file_indices[ix]
            sample_x = self._samples[ix]
            sample_y = sample_x + self.size - 1 + self.y_offset
            
            x_example = np.array(self.x_data[f_idx][sample_x : sample_x + self.size]).flatten()
            x_example = self._preprocessing_callback(x_example)
            
            batch_x.append(x_example.reshape(( self.size,1 )) )

            batch_amp.append(np.array(
                            self.amp_data[f_idx][sample_y])
                          )
            
            batch_phase.append(np.array(
                            self.phase_data[f_idx][sample_y])
                          )

        return np.array(batch_x), np.array(batch_y)
