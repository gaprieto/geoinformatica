def amp_spec(data,dt):
   """
   Compute the power spectrum of a given signal
   Uses numpy
   Assumes data is a numpy array, column vector
   """
   import scipy.signal as sci
   import numpy as np

   ndim = data.ndim
   if ndim==1 :
      x = data 
      x = x[:,np.newaxis]
   
   # Detrend signal
   x = sci.detrend(x,axis=0)

   npts = x.shape[0]
   nfft = npts

   if np.all(np.isreal(data)):
      if (npts%2 == 0):
         nf = round(nfft/2.) + 1
         fvec_p  = np.arange(0,nfft/2+1)/(nfft*dt)
         fvec_n  = np.arange(-(nfft/2),0)/(nfft*dt)
      else:
         nf = round((nfft+1)/2)
         fvec_p  = np.arange(0,(nfft-1)/2+1)/(nfft*dt)
         fvec_n  = np.arange(-(nfft-1)/2,0)/(nfft*dt)
    
      # Create vector as in FFT sampling 
      # except with nyquist in positive freq
      # (matlab and Python numpy.fft)
      fvec = np.concatenate((fvec_p,fvec_n))
      freq = np.zeros((nf,1))
      spec = np.zeros((nf,1))

      fdata = np.fft.fft(x,nfft,0)
      spec  = abs(fdata[0:nf])
      freq  = fvec[0:nf]

   #print(npts, nfft, nf)
   #print('freq ', freq)

   return freq,spec

def amp_spec_taper(data,dt):
   """
   Compute the power spectrum of a given signal
   Uses numpy
   Assumes data is a numpy array, column vector
   Applies a hanning taper before FFT
   """
   import scipy.signal as sci
   import numpy as np

   ndim = data.ndim
   if ndim==1 :
      x = data 
      x = x[:,np.newaxis]
   
   # Detrend signal
   x = sci.detrend(x,axis=0)

   npts = x.shape[0]
   hwin = np.hanning(npts)
   hwin = hwin[:,np.newaxis]
   nfft = npts

   # Apply taper
   x    = x*hwin

   if np.all(np.isreal(data)):
      if (npts%2 == 0):
         nf = round(nfft/2.) + 1
         fvec_p  = np.arange(0,nfft/2+1)/(nfft*dt)
         fvec_n  = np.arange(-(nfft/2),0)/(nfft*dt)
      else:
         nf = round((nfft+1)/2)
         fvec_p  = np.arange(0,(nfft-1)/2+1)/(nfft*dt)
         fvec_n  = np.arange(-(nfft-1)/2,0)/(nfft*dt)
    
      # Create vector as in FFT sampling 
      # except with nyquist in positive freq
      # (matlab and Python numpy.fft)
      fvec = np.concatenate((fvec_p,fvec_n))
      freq = np.zeros((nf,1))
      spec = np.zeros((nf,1))

      fdata = np.fft.fft(x,nfft,0)
      spec  = abs(fdata[0:nf])
      freq  = fvec[0:nf]

   return freq,spec

