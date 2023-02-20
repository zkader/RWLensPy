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
from rwlenspy.baseband_sim import *

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

step = np.abs(x1[1] -x1[0])
x1v,x2v = np.meshgrid(x1,x1)

for freqv in [400e6,500e6,600e6,700e6,800e6]:
#freqv = 800e6
    ferm = geom_const*geom_arr + lens_const/(freqv**2)*lens_arr
    #ferm = ferm.astype(np.double)

    crit_freq = np.sqrt( (lens_const * lens_arr) / (geom_const * geom_arr) )

    #crit_freq[(crit_freq < 400e6)] = np.nan
    #crit_freq[(crit_freq > 800e6)] = np.nan

    critcut = np.abs(crit_freq -freqv) < def1*dump_frames
    critpntsx1 = x1v[critcut]
    critpntsx2 = x2v[critcut]
    critmag = []
    for indexs in np.argwhere(critcut):
        critmag.append(GetPntMag(indexs[0],indexs[1],step,ferm))
    critdel = ferm[critcut]

    statpntsx1 = []
    statpntsx2 = []
    statmag = []
    statdel = []

    for i in range(2,ferm.shape[0]-2):
        for j in range(2,ferm.shape[1]-2):
            if GetStatPnt(ferm,i,j):
                statpntsx1.append(x1v[i,j])
                statpntsx2.append(x2v[i,j])
                statmag.append(GetPntMag(i,j,step,ferm))
                statdel.append(ferm[i,j])

        #if GetStatPnt(crit_freq - freqv,i,j):
        #    critpntsx1.append(x1v[i,j])
        #    critpntsx2.append(x2v[i,j])

    #print(critpntsx1.shape)
    
    statpntsx1 = np.asarray(statpntsx1)
    statpntsx2 = np.asarray(statpntsx2)
    critpntsx1 = np.asarray(critpntsx1)
    critpntsx2 = np.asarray(critpntsx2)

    critmag = np.asarray(critmag)
    critdel = np.asarray(critdel)
    statmag = np.asarray(statmag)
    statdel = np.asarray(statdel)

    plt.figure()
    plt.pcolormesh(x1*206264.806247/1e-9,x1*206264.806247/1e-9,ferm*2*np.pi*freqv)
    plt.colorbar()
    #plt.scatter(critpntsx1,critpntsx2,c='r')
    plt.scatter(statpntsx1*206264.806247/1e-9,statpntsx2*206264.806247/1e-9,c='k',s=4)
    plt.ylabel('Image Position [nanoarcsec]')
    plt.xlabel('Image Position [nanoarcsec]')
    plt.savefig(f'ferm_full_{freqv/1e6}.png',transparent=True)

    plt.figure()
    plt.scatter(statpntsx1*206264.806247/1e-9,statpntsx2*206264.806247/1e-9,c=statdel*2*np.pi*freqv,s=5)
    plt.scatter(critpntsx1*206264.806247/1e-9,critpntsx2*206264.806247/1e-9,c=critdel*2*np.pi*freqv,s=5)
    plt.colorbar()
    plt.ylabel('Image Position [nanoarcsec]')
    plt.xlabel('Image Position [nanoarcsec]')
    plt.savefig(f'fermdel_full_{freqv/1e6}.png',transparent=True)

    norm = 2*np.pi *freqv* geom_const/(lens_const/(freqv**2))

    plt.figure()
    plt.scatter(statpntsx1*206264.806247/1e-9,statpntsx2*206264.806247/1e-9,c=np.abs(statmag)*np.sqrt(norm),s=5)
    plt.scatter(critpntsx1*206264.806247/1e-9,critpntsx2*206264.806247/1e-9,c=np.abs(critmag)*np.sqrt(norm),s=5)
    plt.colorbar()
    plt.ylabel('Image Position [nanoarcsec]')
    plt.xlabel('Image Position [nanoarcsec]')
    plt.savefig(f'fermmag_full_{freqv/1e6}.png',transparent=True)
    print('saved ')


plt.figure()
plt.pcolormesh(x1*206264.806247/1e-9,x1*206264.806247/1e-9,crit_freq/1e6,vmin=400,vmax=800)
plt.ylabel('Image Position [nanoarcsec]')
plt.xlabel('Image Position [nanoarcsec]')
plt.colorbar()
plt.savefig('ferm_critfreq.png',transparent=True)    
