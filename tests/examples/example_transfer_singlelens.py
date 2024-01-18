from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import numpy as np
from astropy import constants as c
from astropy import cosmology
from astropy import units as u
from scipy.fft import rfftfreq, rfft, irfft

import rwlenspy.lensing as rwl
from rwlenspy.utils import RandomGaussianLens

"""
####################################
#### Lensing Transfer Function #####
####################################
"""

"""
Diagram of Lensing System setup.
####################################
 |             |               |
 |             |               |
 |             |               |
 |             |               |
obs           lens            src
####################################
"""
cosmo = cosmology.Planck18

# Comoving
D_obs_src = cosmo.comoving_distance(1)
D_obs_len = 1 * u.kpc

# Redshift
z_obs_len = 0
z_obs_src = cosmology.z_at_value(cosmo.comoving_distance, D_obs_src)

# Ang. Diam. Distance
D_obs_len = 1 * u.kpc
D_obs_src = cosmo.angular_diameter_distance(z_obs_src)
D_len_src = cosmo.angular_diameter_distance(z_obs_src) - D_obs_len

# Physical Lens Params
const_D = D_len_src / (D_obs_len * D_obs_src)
r_e = c.alpha**2 * c.a0  # classical electron radius
kdm = (
    (r_e * c.c / (2 * np.pi)).to(u.cm**2 / u.s)
    * ((1.0 * u.pc / u.cm).to(u.m / u.m)).value
).value
lens_scale = (5.670047618847658 * u.AU / D_obs_len).to(u.m / u.m)
scale = lens_scale.value
sig_DM = 0.0006
geom_const = ((1 / (const_D * c.c)).to(u.s)).value
geom_const = geom_const * scale**2
lens_const = kdm * sig_DM
freq_power = -2.0
beta_x = 0.0
beta_y = 0.0

# Grid Parameters
max_fres = 7.0
theta_min = -max_fres
theta_max = max_fres
theta_N = 201

# Lens Parameters
freq_ref = 800e6
bb_frames = 500
freqs = 800e6 - rfftfreq(2048 * bb_frames, d=1 / (800e6))  # MHz
freqs = freqs.astype(np.double).ravel(order="C")
nyqalias = True

# Spatial Grid
x1 = np.arange(theta_N) * (theta_max - theta_min) / (theta_N - 1) + theta_min

# Lens Functions
seed = 1234
lens_arr = RandomGaussianLens(theta_N, theta_N, 1, seed=seed)
lens_arr = lens_arr.astype(np.double).ravel(order="C")

print("Getting the transfer function with Algorithm...")
t1 = time()
transferfunc = rwl.RunUnitlessTransferFunc(
    theta_min,
    theta_max,
    theta_N,
    freqs,
    freq_ref,
    lens_arr,
    beta_x,
    beta_y,
    geom_const,
    lens_const,
    freq_power,
    nyqalias,
)
tv = time() - t1
print("Tranfer function obtained in:", tv, "s", " | ", tv / 60, "min", tv / 3600, "hr")
transferfunc = np.array(transferfunc).astype(np.cdouble)


"""
##############################
#### Pseudo-Baseband Sim #####
##############################
"""
tres = 1.25e-9  # s
times = np.arange(2048 * bb_frames) * tres
sig_ = np.random.normal(loc=0, scale=1, size=times.size) * np.sqrt(
    9
    * np.exp(
        -0.5
        * (times - 2048 * bb_frames // 4) ** 2
        / (2 * (2048 * bb_frames // 12) ** 2)
    )
)

sig_ = irfft(rfft(sig_) * transferfunc)
noise_ = np.random.normal(loc=0, scale=1, size=sig_.size)

vstream_ = sig_ + noise_

baseband_ = vstream_.reshape((vstream_ // 2048, 2048))
baseband_ = rfft(baseband_, axis=-1).T

plt.figure()
plt.imshow(
    np.abs(baseband_) ** 2,
    aspect="auto",
    extent=[freqs.amin(), freqs.amax(), 0, times[-1] / 1e-3],
)
plt.ylabel("Freq [MHz]", size=14)
plt.xlabel("Time [ms]", size=14)

save_path = Path.cwd()
save_path = save_path / "singlelens_baseband.png"
plt.savefig(str(save_path))