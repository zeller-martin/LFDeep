import matlab.engine
import numpy as np

print('Setting up the matlab wrapper ...')
mlab = matlab.engine.start_matlab()
mlab.addpath(mlab.genpath('C:\\Martin\\code\\NetCom')) 
mlab.addpath('C:\\Martin\\code\\LFPredict\\examples\\implementation\\MATLAB_helpers')
mlab.NlxConnectToServer('localhost')  
mlab.NlxSetApplicationName('test')
mlab.NlxOpenStream('Events') 
print('Finished matlab setup.')

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
    
    
class NlxRingBuffer:

  def __init__(self, buffer_size, stream_name, fs = 30000):
      if buffer_size % 512 != 0:
          Warning('Buffer size not divisible by 512!')

      mlab.NlxOpenStream(stream_name)

      self._n_records = buffer_size // 512
      self._buffer_size = buffer_size

      self._data = np.zeros(self._buffer_size, dtype = np.int16)
      self._timestamps = np.zeros(self._n_records, dtype = np.int64)
      self._stream_name = stream_name
      self._fs = fs

  def update(self):
      csc_buffer = get_csc_data(self._stream_name)

      if csc_buffer is None:
          return False

      samples_returned = csc_buffer[0].shape[0]
      records_returned = csc_buffer[1].shape[0]

      if samples_returned == 0:
          return False
      elif samples_returned > self._buffer_size:
          self._data = csc_buffer[0][-self._buffer_size:]
          self._timestamps = csc_buffer[1][-self._n_records:]
      elif samples_returned == self._buffer_size:
          self._data = csc_buffer[0]
          self._timestamps = csc_buffer[1]
      else:
          self._data[:-samples_returned] = self._data[samples_returned:]
          self._data[-samples_returned:] = csc_buffer[0]

          self._timestamps[:-records_returned] = self._timestamps[records_returned:]
          self._timestamps[-records_returned:] = csc_buffer[1]

      return True

  def __getitem__(self, idx):
      return self._data[idx].astype(np.float32)

  def timestamp_ms_at(self, idx):
      if idx > self._buffer_size:
          print('Warining bad idx')

      if idx < 0:
          if -idx > self._buffer_size:
              print('Warining bad idx')
          idx = self._buffer_size + idx


      record = idx // 512
      sample = idx % 512
      record_timestamp_ms = self._timestamps[record] / 1000
      idx_offset_ms = 1000 * sample / self._fs
      return record_timestamp_ms + idx_offset_ms
