from scipy import signal
import numpy as np

def estimate_phase_hilbert(x, fs, corner_freq ):
    sos = signal.butter(2, corner_freq, 'bandpass', fs = fs, output = 'sos')
    filtered = signal.sosfiltfilt(sos, x)
    phase = np.angle(signal.hilbert(filtered))
    return phase[-1]
