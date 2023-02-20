import os
import sys
sys.path.insert(0,"/home/zkader/coderepo/RWLensPy/")
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm,SymLogNorm
from time import time
from scipy.fft import rfft,irfft,fft,ifft,fftfreq,fftshift,rfftfreq,fftn,ifftn
from scipy.ndimage import gaussian_filter
import matplotlib as mpl

mpl.rcParams['figure.figsize'] = [8.0, 6.0]
mpl.rcParams['figure.dpi'] = 80
mpl.rcParams['savefig.dpi'] = 100

mpl.rcParams['font.size'] = 12
mpl.rcParams['legend.fontsize'] = 'large'
mpl.rcParams['figure.titlesize'] = 'large'

mpl.rcParams['agg.path.chunksize'] = 10000

from astropy import units as u
from astropy import constants as c
from astropy import cosmology

arr = np.load('test_sim_wfall.npy')

print(arr.shape[0])

bb_mf = np.sqrt(np.abs(arr[-1,:,:])**2 / (np.abs(arr[-1,:,:])**2).sum())
for i in range(bb_mf.shape[0]):
    bb_mf[i,:] = gaussian_filter(bb_mf[i,:],2)

bb_mf = np.sqrt(np.abs(bb_mf)**2 / (np.abs(bb_mf)**2).sum())

plt.figure()
plt.plot(np.arange(bb_mf.shape[-1])*1.25e-9, np.sum(bb_mf,axis=0))
plt.axhline(y=1e-3)
plt.xlabel('Time [sec]')
plt.ylabel('Power [arb.]')
plt.yscale('log')
plt.savefig('scatterlens_mf.png')

start_bin = np.where(np.sum(bb_mf, axis=0) > 1e-3)[0][0]
end_bin = np.where(np.sum(bb_mf, axis=0) > 1e-3)[0][-1] 
print(start_bin,end_bin)

wid = (end_bin - start_bin) // 2
tres = 2.56e-6 / 1e-3

x_cen = np.argmax(np.sum(bb_mf**2, axis=0))

l_wid = start_bin#x_cen - start_bin
r_wid = start_bin+end_bin #- x_cen

bbfreqs = (800 - rfftfreq(2048,d=1/800))[:-1]
    
scatter_wid = ((wid) * ( bbfreqs/bbfreqs[-1] )**(-4)  + start_bin)*tres
scatter_kwid = ( (wid) * ( bbfreqs/bbfreqs[-1] )**(-4.4)  + start_bin)*tres


plt.figure()
plt.imshow( (np.abs(arr)**2).sum(axis=1),
            extent=[0, arr.shape[-1]*2.56e-6/1e-3, 400, 800],
            aspect="auto",
            norm=LogNorm(),
            cmap="Greys",
)
plt.ylabel("Frequency [MHz]", size=18)
plt.xlabel("Time [ms]", size=18)
plt.plot(scatter_wid,bbfreqs,c='r',label='Gaussian. ($\\nu^{-4}$)')
plt.plot(scatter_kwid,bbfreqs,c='b',label='Kolmogorov. ($\\nu^{-4.4}$)')
plt.axvline(x=start_bin*tres,c='k',ls='--')
plt.legend(loc=0,prop={'size':12})
plt.savefig('scatterlens_wfall.png')
