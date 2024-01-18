import numpy as np
from time import time
from scipy.fft import rfftfreq

from astropy import units as u
from astropy import constants as c
from astropy import cosmology

import rwlenspy.lensing as rwl


cosmo = cosmology.Planck18

############################################################
# |              |               |
# |              |               |
# |              |               |
# |              |               |
# obs            r1             src  
############################################################

# Comoving
# D_from_to
D_obs_src = 1*u.Gpc 
D_obs_r1 = 1*u.kpc

print(D_obs_src,D_obs_r1)
z_obs_r1 = 0 
z_obs_src = cosmology.z_at_value(cosmo.comoving_distance,D_obs_src)

print(z_obs_src,z_obs_r1)
Eins_time_const = 4*c.G*c.M_sun/c.c**3

# Ang. Diam. Distance
D_obs_r1 = 1*u.kpc
D_obs_src = cosmo.angular_diameter_distance(z_obs_src) - 1*u.pc
D_r2_src =  cosmo.angular_diameter_distance(z_obs_src) - 1*u.pc - 1*u.kpc

const_r1 = D_r2_src / (D_obs_r1 * D_obs_src)

freq0 = 400E6 * u.Hz
r_e = c.alpha**2 * c.a0 # classical electron radius
kdm = ((r_e * c.c  /( 2 * np.pi)).to(u.cm**2/u.s) * ((1.0*u.pc/u.cm).to(u.m/u.m)).value).value

mass = 10 #solar mass

p_scale = ( 5.670047618847658*u.AU / D_obs_r1).to(u.m/u.m) 

#theta_fres0 = np.sqrt(c.c/(2*np.pi*freq0) * const_r2).to(u.m/u.m)
#theta_E = np.sqrt( mass*Eins_time_const*c.c*const_r1 ).to(u.m/u.m)

theta_char = 1.0
max_fres = 7.0

theta_min = -max_fres#.value
theta_max = max_fres#.value
theta_N = 201

dump_frames = 5000 # 5 hrs?
#dump_frames = 500 # 5 hrs?
freqs = np.asarray([800e6,400e6])# 800e6 - rfftfreq(2048*dump_frames, d=1/(800e6)) #MHz

freq_ref = 800e6

beta_x = 0.0
beta_y = 0.0

kdm = ((r_e * c.c  /( 2 * np.pi)).to(u.cm**2/u.s) * ((1.0*u.pc/u.cm).to(u.m/u.m)).value).value

scale = p_scale.value
sig_DM = 0.0006
geom_const = ((1/(const_r1*c.c)).to(u.s)).value
geom_const = geom_const * scale**2
lens_const = kdm * sig_DM
freq_power = -2.0

print(f'Geom. const 1: {geom_const}')
print(f'Lens. const 1: {lens_const}')
print(f'Lens. param 800 MHz: {lens_const/geom_const/(freq_ref**2) }')
print(f'Lens. param 400 MHz: {lens_const/geom_const/((freq_ref/2)**2) }')

#scale2 = np.sqrt(  (lens_const2/400e6**2) / ((1/(const_r1*c.c)).to(u.s)).value /0.001 )
#kp2 = 0.002
#print('Scale for R1:', np.sqrt(  (lens_const2/400e6**2) / ((1/(const_r1*c.c)).to(u.s)).value /kp2 ), (np.sqrt(  (lens_const2/400e6**2) / ((1/(const_r1*c.c)).to(u.s)).value /kp2 ) * D_obs_r1 ).to(u.AU)) 

#kp1 = 96.0801906066323 #plasma
#kp1 = 21 #plasma
kp1 = 0.20003266123359079
print('Scale:', np.sqrt(  (lens_const/400e6**2) / ((1/(const_r1*c.c)).to(u.s)).value /kp1 ), (np.sqrt(  (lens_const/400e6**2) / ((1/(const_r1*c.c)).to(u.s)).value /kp1 ) * D_obs_r1 ).to(u.AU)) 
print('DM',sig_DM)
x1 = np.arange(theta_N)* (theta_max - theta_min ) / (theta_N - 1) + theta_min

seed = 1348
rdmstate = np.random.RandomState(seed)
lens_arr = rdmstate.normal(loc=0,scale=1,size=(theta_N,theta_N))

lens_arr = lens_arr.astype(np.double).ravel(order='C')
#lens_arr2 = lens_arr2.astype(np.double).ravel(order='C')

freqs = freqs.astype(np.double).ravel(order='C')

freqn = 1024*1 + 1
freqmax = 800e6
freqmin = 400e6
freqss = [800e6,400e6]#800e6 - rfftfreq(2048,d=1/800e6)
freq_ref = 800e6

print('Getting the transfer function')
t1 = time()
txvals,tyvals,fvals,delayvals,magvals = rwl.GetUnitlessFreqStationaryPoints(
                                       theta_min,
                                       theta_max,
                                       theta_N,
                                       lens_arr,    
                                       freqss,    
                                       beta_x,
                                       beta_y,
                                       geom_const,
                                       lens_const,
                                       freq_power)
tv = time() - t1
print('Total Time :',tv,'s',' | ',tv/60,'min',tv/3600,'hr')

txvals = np.asarray(txvals)
tyvals = np.asarray(tyvals)
fvals = np.asarray(fvals)
delayvals = np.asarray(delayvals)
magvals = np.asarray(magvals)

