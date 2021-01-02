import time

def _expand_to_iterable(x, n_iter):
    try:
        iter(x)
        return x
    except TypeError:
        out = [x for i in range(n_iter)]
        return out



def create_training_files(files, file_handler, bandpass_f, channel, n_channels, order = 4, filename = None):
    
    if filename is None:
        


    
class BasicFileHandler:
    
    def __init__(self, fs, order = 'row', dtype = np.int16):
        self.fs = fs
        self.order = order
        self.dtype = dtype
        
    def __call__(self, file, band, channel, n_channels):
        sos = signal.butter(self.order, band, 'bandpass', fs = self.fs, output = 'sos')
        
        dat_map = np.memmap(file, dtype = np.int16)
        
        if self.order = 'row':
            target_shape = (dat_map.shape[0] // n_channels, n_channels)
            dat_map.resize( target_shape )
        else:
            target_shape = (n_channels, dat_map.shape[0] // n_channels)
            dat_map.resize( target_shape )
            dat_map = dat_map.T
            
        data = np.array(dat_map[:, channel])
        
        
        
        
        


def circular_loss(y_true, y_pred):
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    return 1 - tf.cos(angle - y_true)
  
## training_dataset function

### data generators


