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

"""
scipy rfft convention is sum G(t) e^{-i 2 pi f t }
and irfft is sum G(f) e^{i 2 pi f t } / 2 pi.
"""
nyqalias = True

# Spatial Grid
x1 = np.arange(theta_N) * (theta_max - theta_min) / (theta_N - 1) + theta_min

# Lens Functions
seed = 1234
lens_arr = RandomGaussianLens(theta_N, theta_N, 1, seed=seed)
lens_arr = lens_arr.astype(np.double).ravel(order="C")


# Get Transfer Function
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

# Save numpy file
save_path = Path.cwd()
save_path = save_path / "singlelens_tranferfunc.npy"
np.save(str(save_path), transferfunc)


"""
##############################
#### Pseudo-Baseband Sim #####
##############################
"""
tres = 1.25e-9  # s
times = np.arange(2048 * bb_frames) * tres
sig_ = np.random.normal(loc=0, scale=1, size=times.size) * np.sqrt(
    20
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
save_path = save_path / "singlelens_voltstream.png"
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
save_path = save_path / "singlelens_baseband.png"
fig.savefig(str(save_path))
