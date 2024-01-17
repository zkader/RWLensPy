from time import time

import numpy as np
from astropy import constants as c
from astropy import cosmology
from astropy import units as u

import rwlenspy.lensing as rwl
from rwlenspy.utils import AnalyticPointMassGrav, LogLens

cosmo = cosmology.Planck18
"""
Diagram of Lensing System setup.
####################################
  |              |               |
  |              |               |
  |              |               |
  |              |               |
  obs            r1             src
####################################
"""
# Comoving
D_obs_src = cosmo.comoving_distance(1)
D_obs_len = cosmo.comoving_distance(1) / 2

# Redshift
z_obs_src = cosmology.z_at_value(cosmo.comoving_distance, D_obs_src)
z_obs_len = cosmology.z_at_value(cosmo.comoving_distance, D_obs_len)

# Ang. Diam. Dist
D_obs_src = cosmo.angular_diameter_distance(z_obs_src)
D_obs_len = cosmo.angular_diameter_distance(z_obs_len)
D_len_src = cosmo.angular_diameter_distance_z1z2(z_obs_len, z_obs_src)

# Physical Lens Params.
Eins_time_const = 4 * c.G * c.M_sun / c.c**3
const_D = D_len_src / (D_obs_len * D_obs_src)
freq_ref = 800e6
mass = 10  # solar mass
theta_E = np.sqrt(mass * Eins_time_const * c.c * const_D).to(u.m / u.m)
dump_frames = 1
freqs = np.array([freq_ref / 2, freq_ref])
freqs = freqs.astype(np.double).ravel(order="C")
beta_x = 2.5
beta_y = 0.0+1e-6

# Grid parameters
max_fres = 10
theta_min = -max_fres
theta_max = max_fres
theta_N = 501

# Lens Parameters
geom_const = ((1 / (const_D * c.c)).to(u.s)).value
geom_const = geom_const * theta_E.value**2
lens_const = mass * Eins_time_const.to(u.s).value
freq_power = 0

# Spatial Grid
x1 = np.arange(theta_N) * (theta_max - theta_min) / (theta_N - 1) + theta_min
de1 = np.abs(x1[1] - x1[0])

# Lens Functions
lens_arr = -LogLens(x1[None, :], x1[:, None])
lens_arr = lens_arr.astype(np.double).ravel(order="C")

# Solutions from Algorithm
#print("Getting the Images with Algorithm...")
t1 = time()
txvals, tyvals, fvals, delayvals, magvals = rwl.GetUnitlessFreqStationaryPoints(
    theta_min,
    theta_max,
    theta_N,
    lens_arr,
    freqs,
    beta_x,
    beta_y,
    geom_const,
    lens_const,
    freq_power,
)
tv = time() - t1
print("Images obtained in:", tv, "s", " | ", tv / 60, "min", tv / 3600, "hr")

txvals = np.asarray(txvals)
tyvals = np.asarray(tyvals)
fvals = np.asarray(fvals)
delayvals = np.asarray(delayvals)
magvals = np.asarray(magvals)


# Analytic Solutions
xpos_analytic, delay_analytic, mag_analytic = AnalyticPointMassGrav(
    beta_x, geom_const, lens_const
)
print(xpos_analytic,delay_analytic,mag_analytic)