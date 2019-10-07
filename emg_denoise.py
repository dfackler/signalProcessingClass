# denoise emg signal with TKEO
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.signal
from scipy import *
import copy

# import data
emgdata = sio.loadmat('/Users/david/Documents/sigprocMXC-TimeSeriesDenoising/emg4TKEO.mat')

# extract needed variables
emgtime = emgdata['emgtime'][0]
emg = emgdata['emg'][0]

# initialize filtered signal
emgf = copy.deepcopy(emg)

# the loop version
for i in range(1, len(emgf)-1):
    emgf[i] = emg[i]**2 - emg[i-1]*emg[i+1]

# the vectorized version for speed and elegance
emgf2 = copy.deepcopy(emg)
emgf2[1:-1] = emg[1:-1]**2 - emg[0:-2]*emg[2:]

# convert both signals to zscore

# find timepoint zero (time started in negatives)
time0 = np.argmin(emgtime**2)

# convert both signals to zscore
emgZ = (emg-np.mean(emg[0:time0])) / np.std(emg[0:time0])
emgZf = (emgf-np.mean(emgf[0:time0])) / std(emgf[0:time0])

# plot
# plot "raw" (normalized to max.1)
plt.plot(emgtime, emg/np.max(emg), 'b', label='EMG')
plt.plot(emgtime, emgf/np.max(emgf), 'm', label='TKEO energy')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude or energy')
plt.legend()

plt.show()

# plot zscored
plt.plot(emgtime, emgZ, 'b', label='EMG')
plt.plot(emgtime, emgZf, 'm', label='TKEO energy')

plt.xlabel('Time (ms)')
plt.ylabel('Zscore relative to pre-stimulus')
plt.legend()
