# fft_test.py
# simple test on the use of the fft

def bandpass_resp(fmin,fmax,dt):
   """
   Create a bandpass filter and plot the response
   Uses scipy, numpy
   No data input necesary
   Assumes data is a numpy array, column vector
   """
   import math
   import scipy.signal as signal
   import numpy as np
   import matplotlib.pyplot as plt

   # Define nyquist and scale filter
   fs   = 1/dt
   fnyq = 0.5*fs
   f0   = fmin/fnyq
   f1   = fmax/fnyq

   wn   = [f0,f1] 
   b, a = signal.butter(4, wn,'bandpass')
   w, h = signal.freqz(b, a)
   freq = fnyq*w/np.pi
 
   plt.plot(freq, (abs(h)))
   plt.plot(fmin, 0.5*np.sqrt(2), 'ko')
   plt.plot(fmax, 0.5*np.sqrt(2), 'ko')
   plt.axvline(fmin, color='k')
   plt.axvline(fmax, color='k')
   plt.title('Butterworth filter frequency response')
   plt.xlabel('Frequency (Hz)')
   plt.ylabel('Amplitude ')
   plt.grid(which='both', axis='both')
   plt.show()

def bandpass(data,fmin,fmax,dt):
   """
   Filter the signal in data using a Butterworth bandpass 
   filter, and a two pass filter. Similar to Matlab's approach
   Uses scipy, numpy
   Assumes data is a numpy array, column vector
   """
   import scipy.signal as signal
   import numpy as np

   ndim = data.ndim
   if ndim==1 :
      x = data 
      x = x[:,np.newaxis]
  
   # Define nyquist and scale filter
   fnyq = 0.5/dt
   f0   = fmin/fnyq
   f1   = fmax/fnyq

   print(fnyq,fmin,fmax,f0,f1)

   wn   = [f0,f1] 
   b, a = signal.butter(4, wn,'bandpass')

   y    = signal.filtfilt(b, a, x[:,0])
   y    = y[:,np.newaxis]

   return y

import math
import numpy as np
import matplotlib.pyplot as plt
import random

import math
import numpy as np
import matplotlib.pyplot as plt
import random

npts = 50
f1   = 0.2
f2   = 0.1
f3   = 0.3
dt   = 1.0
t    = np.arange(npts)

xnoise = np.random.normal(0.0,0.1,npts) 
xsin1  = 0.5*np.sin(2*math.pi*f2*t) 
xsin2  = 0.5*np.sin(2*math.pi*f3*t) 
data2  = xsin1 + xsin2 + 2.*xnoise
data3  = xsin1 + xsin2
data   = np.sin(2*math.pi*f1*t)

bandpass_resp(f2,f3,1.0)
bandpass_resp(0.05,0.25,1.0)
bandpass_resp(0.25,0.40,1.0)

fdata   = bandpass(data,f2,f3,1)
fdata2  = bandpass(data2,0.05,0.25,1)
fdata2b = bandpass(data2,0.25,0.4,1)


# Plot the data
plt.subplot(211)
plt.plot(t,data)
plt.plot(t,data2)

plt.subplot(212)
plt.plot(t, fdata,'r')
plt.plot(t, fdata2,'m')
plt.plot(t, fdata2b,'k')
plt.show()

