import numpy as np
from scipy import signal
import pycircstat as pcs

def hilbert_phase(y):
    return np.angle(signal.hilbert(y))

def bandpass_filter(y, corner_frequencies, sampling_rate):
    sos = signal.butter(4, corner_frequencies, 'bandpass', fs = sampling_rate, output = 'sos')
    return signal.sosfiltfilt(sos, y)


def instantaneous_hilbert(y, corner_frequencies, sampling_rate):
    y_filtered = bandpass_filter(y, corner_frequencies, sampling_rate)
    phase = hilbert_phase(y_filtered)
    return phase[-1]
    
def linear_extrapolation(y, corner_frequencies, sampling_rate):
    middle_index = y.shape[0] // 2
    extension = int(0.1 * y.shape[0])
    
    y_filtered = bandpass_filter(y, corner_frequencies, sampling_rate)
    phase = hilbert_phase(y_filtered)
    
    middle_phase = phase[middle_index]
    middle_frequency = np.mean(
                               pcs.cdiff(
                                         phase[middle_index - extension : middle_index + extension ]
                                         )
                               )
                               
    extrap_phase = (y.shape[0] - middle_index) * middle_frequency + middle_phase
    extrap_phase %= 2*np.pi
    return extrap_phase
               
