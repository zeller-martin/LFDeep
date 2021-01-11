
from tensorflow import keras
from time import time, sleep
import numpy as np
model = keras.models.load_model('NET_7_predphase.keras',compile=False)
prediction = model(np.zeros((1,1024,1), dtype = np.float32)) # warm up the model

from netcom_wrap import *


import numpy as np
import matplotlib.pyplot as plt
from IPython import embed
import serial
from time import sleep
from scipy import signal
from numba import njit
import pycircstat as pcs


def zscore(y):
    return (y - np.mean(y)) / np.std(y)

def get_csc_data(stream_name):
    success, data, time_base, time_offset = mlab.csc_to_python(stream_name, nargout = 4) 
    if not success:
        return None

    data = np.array(data._data)
    try:
        time_offset = np.array(time_offset._data)
    except AttributeError:
        time_offset = np.array(time_offset)
    timestamps = time_base + time_offset

    return data, timestamps.flatten()


def estimate_phase(x):
    x_cos = np.diff(x)
    x_cos *= np.sum(np.abs(x)) / np.sum(np.abs(x_cos))
    return -np.angle(x[:-1] + 1j * x_cos) + np.pi

def estimate_frequency(phase, fs = 1250 / 8 ):
    pdiff = pcs.cdiff(phase[1:], phase[:-1])
    return np.median(pdiff) * fs / 2 / np.pi

     

 
class Stimulator:
    
    def __init__(self, serial_port, baud_rate, sync_bit):
        self._serial = serial.Serial(serial_port, baud_rate)
        self._sync_bit = sync_bit
        mlab.NlxGetNewEventData('Events', nargout = 0)
        
        self._serial.write(b'\xff')
        self._reference_time_ms = mlab.get_teensy_timestamp(self._sync_bit) / 1000
        
        self._is_on = False
        self._stim_params = dict()
        
    def synchronize(self):
        if not self._is_on:
            return
        mlab.NlxGetNewEventData('Events', nargout = 0)
        self._serial.write(b'\xf4')
        teensy_time = mlab.get_teensy_timestamp(self._sync_bit) / 1000
        self._serial.write(np.uint32(teensy_time - self._reference_time_ms).tobytes())
        
    def update_stimulus(self, timestamp_ms, phase, frequency, phase_start, phase_end):
        if not self._is_on:
            self._serial.write(b'\xfe')
            self._is_on = True
        else:
            self._serial.write(b'\xfd')
        cycle_ms_passed = 1000 * (1 / frequency) * phase / (2 * np.pi)
        self._stim_params['reset'] = timestamp_ms - cycle_ms_passed - self._reference_time_ms
        self._stim_params['cycle_length_ms'] = 1000 / frequency
        self._stim_params['stim_start_ms'] = self._stim_params['cycle_length_ms'] * phase_start / ( 2*np.pi )
        self._stim_params['stim_end_ms'] = self._stim_params['cycle_length_ms'] * phase_end / ( 2*np.pi )
        
        for param in self._stim_params.values():
            self._serial.write(np.uint32(param).tobytes())
            
    def turn_off_stimulus(self):
        self._serial.write(b'\xc8')
        self._is_on = False
        
    def __del__(self):
        self.turn_off_stimulus()
        self._serial.close()

fs = 30000
stimulus_freq = 3.5
stim_start = 0.8 * np.pi
stim_end = 1.2* np.pi

min_f = 2
max_f = 5


buffer_size = 2**16
sync_bit = 4
stream_chan = 'S1R1'
serial_port = 'COM5'
baud_rate = 115200

phase_start = 0.9 * np.pi
phase_end = 1.1 * np.pi



ring_buffer = NlxRingBuffer(buffer_size, stream_chan)
stimulator = Stimulator(serial_port, baud_rate, sync_bit)


sos = signal.butter(4, 20, 'lowpass', fs = 30000, output = 'sos')


j = 0
ring_buffer.update()
sleep(.5)
ring_buffer.update()
print('start')
ps = list()
freqs = list()

looptime = np.zeros(100)

while True:
    
    
    if ring_buffer.update():
        
        a = time()
        
        # preprocess recent ring buffer content
        preprocessed = signal.sosfiltfilt(sos, ring_buffer[:])[::24]
        preprocessed = zscore(preprocessed[-1024:])
        preprocessed.resize(1, 1024, 1 )
        
        #predict instantaneous phase and frequency using a shallow CNN
        prediction = ( model(preprocessed).numpy().flatten() + 2*np.pi ) % (2*np.pi)
        
        phase = prediction[-1]
        #ps.append(phase)
        
        frequency = estimate_frequency(prediction[-10:-1])
        
        
        #freqs.append(frequency)
        timestamp = ring_buffer.timestamp_ms_at(-5 * 192 )
        stimulator.update_stimulus(timestamp, phase, frequency, phase_start, phase_end)
        looptime[j % 100] = time() - a
        j += 1
        sleep(.02)
    if j % 100 == 0:
        print(j)
        print(f"Average evaluation time (ms): {np.round(np.mean(looptime * 1000), 2)}")
        stimulator.synchronize()
    
    if j == 20000:
        break
    
stimulator.turn_off_stimulus()

embed()
