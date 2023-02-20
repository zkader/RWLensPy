import os
import sys
sys.path.insert(0,"/home/zkader/coderepo/RWLensPy/")
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm,SymLogNorm
from time import time
from scipy.fft import rfft,irfft,fft,ifft,fftfreq,fftshift,rfftfreq,fftn,ifftn

import matplotlib as mpl

GREYMAP = mpl.cm.__dict__["Greys"]

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

import rwlenspy.lensing as rwl
from rwlenspy.utils import *

cosmo = cosmology.Planck18

############################################################
# |              |               |              |
# |              |               |              |
# |              |               |              |
# |              |               |              |
# obs            r1             r2              src
############################################################

# Comoving
# D_from_to
D_obs_src = cosmo.comoving_distance(1)
D_obs_r1 = cosmo.comoving_distance(1)/2
D_obs_r2 = D_obs_src - 1*u.kpc

z_obs_r1 = cosmology.z_at_value(cosmo.comoving_distance,D_obs_r1)
z_obs_r2 = cosmology.z_at_value(cosmo.comoving_distance,D_obs_r2)
z_obs_src = cosmology.z_at_value(cosmo.comoving_distance,D_obs_src)

Eins_time_const = 4*c.G*c.M_sun/c.c**3

# Ang. Diam. Distance
D_obs_r1 = cosmo.angular_diameter_distance(z_obs_r1)
D_obs_r2 = cosmo.angular_diameter_distance(z_obs_r2)
D_r1_r2 = cosmo.angular_diameter_distance_z1z2(z_obs_r1,z_obs_r2)

const_r1 = D_r1_r2 / (D_obs_r1 * D_obs_r2)

D_obs_src = cosmo.angular_diameter_distance(z_obs_src)
D_r2_src = cosmo.angular_diameter_distance_z1z2(z_obs_r2,z_obs_src)


const_r2 = D_r2_src / (D_obs_r2 * D_obs_src)

#r_inner = 1E7*u.cm#1e10*u.m
r_inner = 1e8*u.cm
r_outer = 100*u.pc #1E13*u.cm
freq0 = 400E6 * u.Hz
mass = 10 #solar mass
r_e = c.alpha**2 * c.a0 # classical electron radius
p_scale =  (1*u.AU/D_obs_r2).to(u.m/u.m)
#n_e = 0.06* u.cm**-3
DM_0 = 0.03*u.pc*u.cm**-3

theta_inner = (r_inner/D_obs_r2).to(u.m/u.m)
theta_outer = (r_outer/D_obs_r2).to(u.m/u.m)
theta_fres0 = np.sqrt(c.c/(2*np.pi*freq0) * const_r2).to(u.m/u.m)
theta_p0 = (theta_fres0 * np.sqrt(r_e* c.c * DM_0 / freq0 )).to(u.m/u.m)
print(f'p scale : {p_scale} | theta inner: {theta_inner} | theta fres: {theta_fres0} | theta p0: {theta_p0}')
print(f'p scale / theta fres: {p_scale/theta_fres0} | p scale /  theta p0: {p_scale /theta_p0}')

#theta_char = p_scale
#max_fres = 5*theta_char

theta_char = theta_fres0
max_fres = theta_char * 5000

theta_min = -max_fres.value
theta_max = max_fres.value
theta_N = 1001

dump_frames = 1
freqs = 800e6 - rfftfreq(2048*dump_frames, d=1/800e6)
#freqs = np.linspace(800e6,400e6,1024)
freq_min = freqs[-1]
freq_max = freqs[0]
freq_N = freqs.size

freq_ii = 400 * u.MHz
freq_ii = freq_ii.to(u.Hz).value

beta_x = 0.0
#beta_y = 3.0*theta_char.value
beta_y = 0

geom_const = ((1/(const_r2*c.c)).to(u.s)).value
lens_const = ((r_e * c.c  /( 2 * np.pi)).to(u.cm**2/u.s) * ((1.0*u.pc/u.cm).to(u.m/u.m)).value).value # k_DM
print(geom_const)
print(lens_const)
print(lens_const / (400e6**2))

x1 = np.arange(theta_N)* (theta_max - theta_min ) / (theta_N - 1) + theta_min
#lens_arr = 0.7 * np.ones((theta_N*theta_N)) #np. np.exp(-0.5*()/p_scale.value )#np.random.normal(size=(theta_N*theta_N))

de1 = np.abs(x1[1] - x1[0])
def1 = np.abs(freqs[1] - freqs[0])
#posx = 0*theta_char.value
#posy = 0*theta_char.value

#r = np.sqrt(x1[:,None]**2 + x1[None,:]**2)
#r0 = np.sqrt(posx**2 + posy**2)

seed=np.random.randint(1,10000)
print('seed: ',seed) #6339

ne = get_plasma_Ne(x1.size,x1.size,de1,theta_inner,theta_outer,\
                   C2_n=2e-11,freq=freq0,D_eff=1/const_r2,seed=seed,plot=False)

lens_arr = ne

#lens_arr = 0.03 * np.exp(-0.5*( (x1[:,None])**2 + (x1[None,:])**2)/p_scale.value**2)

#lens_arr = 0.03 * np.exp(-0.5*((r-r0)**2)/p_scale.value**2)
geom_arr = 0.5*( (x1[:,None]-beta_x)**2 + (x1[None,:] - beta_y)**2)

#lens_arr = 1/lens_const * 4*
#lens_arr = 50 *np.ones((theta_N,theta_N))
#lens_arr = 1e-2*np.random.normal(loc=0,scale=1,size=(theta_N,theta_N))

lens_arr = lens_arr.astype(np.double).ravel()
geom_arr = geom_arr.astype(np.double).ravel()

print('Getting the transfer function')
t1 = time()
xinds,yinds,finds = rwl.GetFreqStationaryPoints( geom_arr,
                                  lens_arr,
                                  theta_min,
                                  theta_max,
                                  theta_N,
                                  beta_x,
                                  beta_y,
                                  freq_min,
                                  freq_max,
                                  freq_N,
                                  geom_const,
                                  lens_const)
tv = time() - t1
print('Total Time :',tv,'s',' | ',tv/60,'min',tv/3600,'hr')

xinds = np.asarray(xinds)
yinds = np.asarray(yinds)
finds = np.asarray(finds)

txvals = (theta_max - theta_min)/ (theta_N -1 ) * xinds + theta_min
tyvals = (theta_max - theta_min)/ (theta_N -1 ) * yinds + theta_min
fvals = (freq_max - freq_min)/ (freq_N -1 ) * finds + freq_min

delayvals = geom_const*geom_arr[yinds + theta_N*xinds ] +  lens_const*fvals**(-2) * lens_arr[yinds + theta_N*xinds]

#magvals = #GetPntMag()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(fvals/1e6, tyvals*206264.806247/1e-9, txvals*206264.806247/1e-9,s=4)
ax.set_xlabel('Freq [MHz]')
ax.set_ylabel('$\\theta_X$ [nanoarcsec]')
ax.set_zlabel('$\\theta_Y$ [nanoarcsec]')
plt.savefig('test_statpoints.png')

fig = plt.figure()
ax = plt.axes(projection='3d')
axsc = ax.scatter(fvals/1e6, tyvals*206264.806247/1e-9, txvals*206264.806247/1e-9, c=delayvals, s=4)
fig.colorbar(axsc)
ax.set_xlabel('Freq [MHz]')
ax.set_ylabel('$\\theta_X$ [nanoarcsec]')
ax.set_zlabel('$\\theta_Y$ [nanoarcsec]')
plt.savefig('test_delay_statpoints.png')
