from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import numpy as np
from astropy import constants as c
from astropy import cosmology
from astropy import units as u
from matplotlib.colors import LogNorm
from scipy.fft import irfft, rfft, rfftfreq

import rwlenspy.lensing as rwl
from rwlenspy.utils import LogLens, RandomGaussianLens

"""
############################
#### Lensing Ray Trace #####
############################
"""

"""
Diagram of Lensing System setup.
############################################################
# |              |               |              |
# |              |               |              |
# |              |               |              |
# |              |               |              |
# obs            r1             r2              src
############################################################
"""
cosmo = cosmology.Planck18

# Comoving
D_obs_src = 1 * u.Gpc
D_obs_r1 = D_obs_src / 2
D_obs_r2 = 3 * D_obs_src / 4

# redshift
z_obs_r1 = cosmology.z_at_value(cosmo.comoving_distance, D_obs_r1)
z_obs_r2 = cosmology.z_at_value(cosmo.comoving_distance, D_obs_r2)
z_obs_src = cosmology.z_at_value(cosmo.comoving_distance, D_obs_src)

# Ang. Diam. Distance
D_obs_r1 = cosmo.angular_diameter_distance(z_obs_r1)
D_obs_r2 = cosmo.angular_diameter_distance(z_obs_r2)
D_r1_r2 = cosmo.angular_diameter_distance_z1z2(z_obs_r1, z_obs_r2)
D_obs_src = cosmo.angular_diameter_distance(z_obs_src)
D_r2_src = cosmo.angular_diameter_distance_z1z2(z_obs_r2, z_obs_src)

# Physical Lens (r2) Params
r_e = c.alpha**2 * c.a0  # classical electron radius
kdm = (
    (r_e * c.c / (2 * np.pi)).to(u.cm**2 / u.s)
    * ((1.0 * u.pc / u.cm).to(u.m / u.m)).value
).value
const_Dr2 = D_r2_src / (D_obs_r2 * D_obs_src)
lens_r2_scale = (1000 * u.AU / D_obs_r2).to(u.m / u.m)
scale_r2 = lens_r2_scale.value
sig_DM = 0.0005
geom_const_r2 = ((1 / (const_Dr2 * c.c)).to(u.s)).value
geom_const_r2 = geom_const_r2 * scale_r2**2
lens_const_r2 = kdm * sig_DM
freq_power_r2 = -2.0
beta_r2_x = 0.0
beta_r2_y = 0.0

# Physical Lens (r1) Params
Eins_time_const = 4 * c.G * c.M_sun / c.c**3
const_Dr1 = D_r1_r2 / (D_obs_r1 * D_obs_r2)
mass = 1  # solar mass
lens_r1_scale = np.sqrt(mass * Eins_time_const * c.c * const_Dr1).to(u.m / u.m)
scale_r1 = lens_r1_scale.value
geom_const_r1 = ((1 / (const_Dr1 * c.c)).to(u.s)).value
geom_const_r1 = geom_const_r1 * scale_r1**2
lens_const_r1 = mass * Eins_time_const.to(u.s).value
freq_power_r1 = 0
beta_r1_x = 1.5
beta_r1_y = 0.0

# Sim Parameters
freq_ref = 800e6
bb_frames = 300
freqs = 800e6 - rfftfreq(2048 * bb_frames, d=1 / (800e6))  # MHz
nyqalias = True

# Grid Parameters
max_fres = 5
theta_min = -max_fres
theta_max = max_fres
theta_N = 251

# Spatial Grid
x1 = np.arange(theta_N) * (theta_max - theta_min) / (theta_N - 1) + theta_min

# Lens functions
seed = 4321
lens_arr_r2 = np.ones((theta_N, theta_N)) * RandomGaussianLens(theta_N, 1, 1, seed=seed)  # 1D lens
lens_arr_r1 = -1.0 * LogLens(x1[:, None], x1[None, :])

lens_arr_r2 = lens_arr_r2.astype(np.double).ravel(order="C")
lens_arr_r1 = lens_arr_r1.astype(np.double).ravel(order="C")
freqs = freqs.astype(np.double).ravel(order="C")

# Get Transfer
print("Getting the Transfer Function")
t1 = time()
transferfunc = rwl.RunMultiplaneTransferFunc(
    theta_min,
    theta_max,
    theta_N,
    freqs,
    freq_ref,
    lens_arr_r2,
    lens_r2_scale,
    beta_r2_x,
    beta_r2_y,
    geom_const_r2,
    lens_const_r2,
    freq_power_r2,
    lens_arr_r1,
    lens_r1_scale,
    beta_r1_x,
    beta_r1_y,
    geom_const_r1,
    lens_const_r1,
    freq_power_r1,
    nyqalias,
)
tv = time() - t1
print("Total Time :", tv, "s", " | ", tv / 60, "min", tv / 3600, "hr")
transferfunc = np.array(transferfunc).astype(np.cdouble)

# Save numpy file
save_path = Path.cwd()
save_path = save_path / "multilens_tranferfunc.npy"
np.save(str(save_path), transferfunc)


"""
##############################
#### Pseudo-Baseband Sim #####
##############################
"""
tres = 1.25e-9  # s
times = np.arange(2048 * bb_frames) * tres
sig_ = np.random.normal(loc=0, scale=1, size=times.size) * np.sqrt(
    30
    * np.exp(
        -0.5 * (times - (2048 * bb_frames // 4) * tres) ** 2 / ((2048 * 4 * tres) ** 2)
    )
)

sig_ = irfft(rfft(sig_) * transferfunc)

noise_ = np.random.normal(loc=0, scale=1, size=sig_.size)

vstream_ = sig_ + noise_

# plot figure
fig, ax = plt.subplots()
ax.plot(times / 1e-3, vstream_)
ax.set_ylabel("Voltage [V]", size=14)
ax.set_xlabel("Time [ms]", size=14)

# save png
save_path = Path.cwd()
save_path = save_path / "multilens_voltstream.png"
fig.savefig(str(save_path))

# FFT the voltage
vstream_ = vstream_.reshape((vstream_.shape[-1] // 2048, 2048))
baseband_ = rfft(vstream_, axis=-1).T

# Simple masking of caustic frequencies
mask = np.mean(np.abs(baseband_) ** 2, axis=-1)
mask = mask / np.median(mask)
mask_mean = np.median(mask)
mask_std = 1.4826 * np.median(np.abs(mask - mask_mean))
mask = mask > 8
baseband_[mask, :] = 0 + 0j

# plot baseband
fig, ax = plt.subplots()
im1 = ax.imshow(
    np.abs(baseband_) ** 2,
    aspect="auto",
    norm=LogNorm(),
    extent=[0, vstream_.shape[-1] // 2048 * 2.56e-6 / 1e-3, 400, 800],
)
ax.set_ylabel("Freq [MHz]", size=14)
ax.set_xlabel("Time [ms]", size=14)
fig.colorbar(im1)

# save png
save_path = Path.cwd()
save_path = save_path / "multilens_baseband.png"
fig.savefig(str(save_path))
