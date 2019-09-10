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

"""
START FILTER TYPES
MEAN
GAUSSIAN
"""
# initialize filtered signal
filtsig = np.zeros(n)
# implement running mean filter
k = 20  # filter window is actually k*2+1 to include point and k back and k up
for i in range(k+1, n-k-1):
    filtsig[i] = np.mean(signal[i-k:i+k])

# implement gaussian filter
# full-width half-maximum parameter for gaussian function
fwhm = 25  # in ms
k = 40
gtime = 1000*np.arange(-k, k)/srate

# create gaussian window
gauswin = np.exp(-(4*np.log(2)*gtime**2)/fwhm**2)

# compute empirical FWHM
pstPeakHalf = k + np.argmin((gauswin[k:]-.5)**2)
prePeakHalf = np.argmin((gauswin-.5)**2)

empFWHM = gtime[pstPeakHalf] - gtime[prePeakHalf]

# show gaussian
plt.plot(gtime, gauswin, 'bo-')
plt.plot([gtime[prePeakHalf], gtime[pstPeakHalf]],
         [gauswin[prePeakHalf], gauswin[pstPeakHalf]])

# normalize gaussian to unit energy
gauswin = gauswin/np.sum(gauswin)
plt.xlabel('Time (ms)')
plt.ylabel('Gain')
plt.show()

# apply gaussian filter
filtsigG = copy.deepcopy(signal)
for i in range(k+1, n-k-1):
    filtsigG[i] = sum(signal[i-k:i+k]*gauswin)

# compute windowsize in ms
windowsize = 1000*(k*2+1) / srate

# plot noisy and filtered signals
plt.plot(time, signal, 'r', label='orig')
plt.plot(time, filtsigG, 'b', label='filteredG')
plt.plot(time, filtsig, 'g', label='filteredM')

plt.legend()
plt.xlabel('Time (sec.)')
plt.ylabel('Amplitude')
plt.title('Running mean filter with a k=%d-ms filter' % windowsize)

plt.show()
