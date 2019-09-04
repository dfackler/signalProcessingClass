# scratch sheet for stuff
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.signal
from scipy import *
import copy


# create signal
srate = 1000  # Hz
time = np.arange(0, 3, 1/srate)
n = len(time)
p = 15  # poles for random interpolation

# noise level, measured in standard deviations
noiseamp = 5

# amplitude, noise, and signal
ampl = np.interp(np.linspace(0, p, n), np.arange(0, p), np.random.rand(p)*30)
noise = noiseamp * np.random.rand(n)
signal = ampl + noise

# initialize filtered signal
filtsig = np.zeros(n)

# implement running mean filter
k = 20  # filter window is actually k*2+1 to include point and k back and k up
for i in range(k+1, n-k-1):
    filtsig[i] = np.mean(signal[i-k:i+k])

# compute windowsize in ms
windowsize = 1000*(k*2+1) / srate

# plot noisy and filtered signals
plt.plot(time, signal, label='orig')
plt.plot(time, filtsig, label='filtered')

plt.legend()
plt.xlabel('Time (sec.)')
plt.ylabel('Amplitude')
plt.title('Running mean filter with a k=%d-ms filter' % windowsize)

plt.show()
