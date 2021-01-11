import numpy as np
from scipy import signal

def read_binary(src_file,
                n_channels = 1,
                selected_channel = 0,
                src_precision = np.int16):
 

def bandpass_filter(y, corner_frequencies, sampling_rate):
    sos = signal.butter(4, corner_frequencies, 'bandpass', fs = sampling_rate, output = 'sos')
    return signal.sosfiltfilt(sos, y)

def hilbert_phase(y):
    return np.angle(signal.hilbert(y))
    
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
    
    if None in (sampling_rate, kernel_sd_seconds):
        raise ValueError('sampling_rate and kernel_sd_seconds have to be specified!')

    kernel = _make_gauss_kernel(1 / sampling_rate, kernel_sd_seconds)
    return signal.oaconvolve(y**2, kernel, mode = 'same')**.5
