import numpy as np
from scipy import signal



def read_binary_channel(src_file,
                n_channels,
                selected_channel,
                src_precision = np.int16,
                src_order = 'row'):
    data = np.fromfile(src_file, dtype = src_precision)
    
    if src_order == 'row':
        data.resize(data.shape[0] // n_channels, n_channels)
        x = data[:, selected_channel]
    elif src_order in ('col', 'column'):
        data.resize(n_channels, data.shape[0] // n_channels)
        x = data[selected_channel, :]
    else:
        raise ValueError(f'src_order not understood - "row", "col" and "column" permissible, given was "{src_order}".')
 
    return x

def bandpass_filter(x, corner_frequencies, sampling_rate):
    sos = signal.butter(4, corner_frequencies, 'bandpass', fs = sampling_rate, output = 'sos')
    return signal.sosfiltfilt(sos, x)

def hilbert_phase(x):
    return np.angle(signal.hilbert(x))
    
def _make_gauss_kernel(dt, sigma, width = 5):
    n_timesteps = 2 * width * sigma / dt + 1
    time_axis = np.arange(n_timesteps) * dt
    time_axis -= np.median(time_axis)
    y = np.exp(-time_axis**2 / (2 * sigma**2))
    y /= np.sum(y)
    return y

def rms_amplitude(x,
                  sampling_rate = None,
                  kernel_sd_seconds = None):
    
    if None in (sampling_rate, kernel_sd_seconds):
        raise ValueError('sampling_rate and kernel_sd_seconds have to be specified!')

    kernel = _make_gauss_kernel(1 / sampling_rate, kernel_sd_seconds)
    return signal.oaconvolve(x**2, kernel, mode = 'same')**.5
