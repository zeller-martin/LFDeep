from scipy import signal



def estimate_phase_hilbert(x, fs, corner_freq ):
    sos = signal.butter(4, corner_freq, 'bandpass', fs = fs, output = 'sos')
    filtered = signal.sosfiltfilt(sos, x)
    phase = np.angle(signal.hilbert(filtered))
    return phase[-1]
