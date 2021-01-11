from _helpers import *
import glob, os

your_data_folder = './'
target_folder = './'
n_channels = 32
selected_channel = 0

src_sampling_rate = 20000
target_sampling_rate = 1000

corner_freqs = (5, 12)
rms_kernel_width = 0.5


raw_files = glob.glob(os.path.join(your_data_folder, '*.dat'))

for file in raw_files:
    filename = os.path.split(file)[1]
    fbase = os.path.splitext(filename)[0]
    target_base = os.path.join(target_folder, fbase)
    
    x = read_binary_channel(file, n_channels, selected_channel)
    x = resample(x, src_sampling_rate, target_sampling_rate)
    x_filtered = bandpass_filter(x, corner_freqs, target_sampling_rate)
    
    amp = rms_amplitude(x, target_sampling_rate, rms_kernel_width)
    phase = hilbert_phase(x)
    
    x.astype(np.float32).tofile(target_base + '_raw.float32')
    amp.astype(np.float32).tofile(target_base + '_amplitude.float32')
    phase.astype(np.float32).tofile(target_base + '_phase.float32')
    
