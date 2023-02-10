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


cosmo = cosmology.Planck18

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

theta_char = p_scale.value
max_fres = 5*theta_char

theta_min = -max_fres
theta_max = max_fres
theta_N = 1001

xvv = np.linspace(theta_min,theta_max,theta_N)
yvv = 3*p_scale.value
avv = p_scale.value

lensvv = lambda x,a: (1 - np.exp(-0.5*x*x/(a*a))/(a*a))*x
lenstvv = lambda x,y,a: 0.5*(x-y)**2 + np.exp(-0.5*x**2/a**2)   
lensmvv = lambda x,a: ( 1+0j - (2 - x*x/a*a + (1 -  x*x/a*a)*np.exp(-0.5*x*x/(a*a))/(a*a) )* np.exp(-0.5*x*x/(a*a))/(a*a) )**-0.5


r_e = c.alpha**2 * c.a0 # classical electron radius
plasma_const = r_e * c.c  /( 2 * np.pi * ((400* u.MHz ).to(1/u.s))**2)
plasma_const = plasma_const.to(u.pc**-1*u.cm**3 * u.s)    
DM_0 = 0.03* u.pc *u.cm**-3 
plasma_time_factor =  (plasma_const * DM_0 ).to(u.s)

#freqsvv = np.linspace(800,400,1025)*1e6 * u.Hz
freqsvv = (800e6 - rfftfreq(2048,d=1/800e6)[:-1]) * u.Hz

timeconst = (r_e * c.c *DM_0  /( 2 * np.pi * ((freqsvv).to(1/u.s))**2) ).to(u.s)
theta_plasmavv = np.sqrt( timeconst * c.c * const_r2 ).to(u.m/u.m)    
tf = p_scale / theta_plasmavv  

#plt.figure()
#plt.scatter(np.zeros_like(xvv),xvv*206264.806247/1e-9\
#            ,c=lenstvv(xvv/theta_plasmavv[-1],yvv/theta_plasmavv[-1],avv/theta_plasmavv[-1] ) )
#plt.colorbar()
#plt.xlabel('Freq [MHz]')
#plt.ylabel('Image Position [nanoarcsec]')
#plt.title('delay [s]')
#plt.savefig('plens_pos.png')

freqvv = np.array([])
xposvv = np.array([])
i_delvv = np.array([])
i_magvv = np.array([])
for ii,tfi in enumerate(tf):    
    idx = np.argwhere(np.diff(np.sign(lensvv(xvv / theta_plasmavv[ii],\
                                             tfi ) \
                                      - yvv / theta_plasmavv[ii]) \
                                     )).flatten()
    
    freqvv = np.append(freqvv,freqsvv[ii].value*np.ones(idx.size))
    xposvv = np.append(xposvv,xvv[idx])
    i_delvv = np.append(i_delvv,\
                        timeconst[ii].value\
                        *lenstvv(xvv[idx]/theta_plasmavv[ii]\
                                 ,yvv/theta_plasmavv[ii]\
                                 ,avv/theta_plasmavv[ii]))
    i_magvv = np.append(i_magvv,lensmvv(xvv[idx]/theta_plasmavv[ii]\
                                        ,avv/theta_plasmavv[ii]))


plt.figure()
plt.scatter(freqvv/1e6,xposvv*206264.806247/1e-9,c=i_delvv)
plt.colorbar()
plt.xlabel('Freq [MHz]')
plt.ylabel('Image Position [nanoarcsec]')
plt.title('delay [s]')
plt.savefig('plens_delay.png')

plt.figure()
plt.scatter(freqvv/1e6,xposvv*206264.806247/1e-9,c=np.abs(i_magvv),norm=LogNorm())
plt.colorbar()
plt.xlabel('Freq [MHz]')
plt.ylabel('Image Position [nanoarcsec]')
plt.title('mag [ul]')
plt.savefig('plens_mag.png')


dumpframes = 1000
plot_bbd = np.zeros((freqsvv.shape[0],dumpframes),dtype=np.complex128)

impulse = np.zeros(dumpframes)
impulse = np.exp(-0.5*(np.arange(dumpframes) - dumpframes//8)/(10)**2)
impulse[:dumpframes//8-10] = 0
impulse[dumpframes//8+10:] = 0

imgfreqs = freqvv.reshape((freqvv.size//3,3) ) 
imgdelays = i_delvv.reshape((freqvv.size//3,3) ) 
imgmags = i_magvv.reshape((freqvv.size//3,3))

for img_i in range(imgfreqs.shape[-1]):
    img_bbmf = impulse[None,:] * imgmags[:,img_i][:,None]
    
    for freq_i in range(plot_bbd.shape[0]):
        img_bbmf[freq_i,:] = np.roll(img_bbmf[freq_i,:] , np.round((imgdelays[freq_i,img_i] - imgdelays[0,0])/2.56e-6).astype(int) ,axis=-1)
        plot_bbd += img_bbmf


plt.figure()
plt.imshow(np.abs(plot_bbd)**2,aspect='auto',extent=[0,dumpframes*2.56e-6,400,800],norm=LogNorm())
plt.colorbar()
plt.ylabel('Freq [MHz]')
plt.xlabel('Time [sec]')
plt.title('Power [arb.]')
plt.savefig('plens_wfall.png')
