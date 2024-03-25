from pathlib import Path
from time import time

import matplotlib as mpl
import matplotlib.animation
import matplotlib.cm as cmaps
import matplotlib.pyplot as plt
import numpy as np
from astropy import constants as c
from astropy import cosmology
from astropy import units as u
from matplotlib.colors import LogNorm
from scipy.fft import rfftfreq

import rwlenspy.lensing as rwl
from rwlenspy.utils import RandomGaussianLens

# Matplotlib setup
GREYMAP = mpl.cm.__dict__["Greys"]
mpl.rcParams["figure.figsize"] = [8.0, 6.0]
mpl.rcParams["figure.dpi"] = 80
mpl.rcParams["savefig.dpi"] = 100
mpl.rcParams["font.size"] = 12
mpl.rcParams["legend.fontsize"] = "large"
mpl.rcParams["figure.titlesize"] = "large"
mpl.rcParams["agg.path.chunksize"] = 10000
plt.rcParams["savefig.dpi"] = 70

"""
############################
#### Lensing Ray Trace #####
############################
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
# Memory in bytes
max_memory = 4e9

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
freqs = 800e6 - rfftfreq(2048, d=1 / (800e6))  # MHz
freqs = freqs.astype(np.double).ravel(order="C")

# Spatial Grid
x1 = np.arange(theta_N) * (theta_max - theta_min) / (theta_N - 1) + theta_min

# Lens Functions
seed = 1234
lens_arr = np.ones((theta_N, theta_N)) * RandomGaussianLens(1, theta_N, 1, seed=seed)
lens_arr = lens_arr.astype(np.double).ravel(order="C")

# Solutions from Algorithm
print("Getting the Images with Algorithm...")
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
    max_memory
)
tv = time() - t1
print("Images obtained in:", tv, "s", " | ", tv / 60, "min", tv / 3600, "hr")

# Numpy the arrays for ease
txvals = np.asarray(txvals)
tyvals = np.asarray(tyvals)
fvals = np.asarray(fvals)
delayvals = np.asarray(delayvals)
magvals = np.asarray(magvals)


"""
##########################
#### Animate Spatial #####
##########################
"""
# Spatial Animation setup
num_frames = freqs.size

# Setup plot
fig = plt.figure()
ax = fig.add_subplot(111)
axsc = ax.scatter([], [], s=4)
axisscale = 1e0
axisstr = ""
framescale = 1e6
framestr = "M"
scaling = scale * 206264.806247 / axisscale


# Select largest spatial extent
cut1 = fvals == np.amin(fvals)  # lowest freq for scattering
maxv_ = (
    max(np.amax(np.abs(txvals[cut1])), np.amax(np.abs(tyvals[cut1]))) * scaling * 1.1
)

# Set axes and plot
ax.set_ylim(-maxv_, maxv_)
ax.set_xlim(-maxv_, maxv_)
ax.set_ylabel(f"$\\theta_Y$ [{axisstr}arcsec]", size=14)
ax.set_xlabel(f"$\\theta_X$ [{axisstr}arcsec]", size=14)
ax.set_facecolor("black")
cmap = cmaps.gray
norm = LogNorm(vmin=1e-3, vmax=1)
axsc.set_cmap(cmap)
axsc.set_norm(norm)
cb = fig.colorbar(cmaps.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
cb.ax.set_title("Img. Mag.", y=1.02)


# frame animation
def update(i):
    tcutt = fvals == freqs[i]
    data = np.stack([txvals[tcutt] * scaling, tyvals[tcutt] * scaling]).T
    axsc.set_offsets(data)
    axsc.set_array(np.abs(magvals[tcutt]))
    ax.set_title(f"Freq: {freqs[i]/framescale:.0f}  [{framestr}Hz]", size=14)
    return (axsc,)


# image framing
image_duration = 2  # seconds
frame_interval = 30e-3  # seconds between frames
total_aniframes = image_duration / frame_interval
stepsize = np.ceil(freqs.size / total_aniframes).astype(int)

if stepsize == 0:
    stepsize = 1

frame_numbers = np.arange(0, freqs.size, step=stepsize)

ani = matplotlib.animation.FuncAnimation(
    fig, update, frames=frame_numbers, interval=30, blit=True
)

# save
save_path = Path.cwd()
save_path = save_path / "singelens_spatial_freqslice.gif"
ani.save(filename=str(save_path), writer="pillow")


"""
###########################
#### Animate Temporal #####
###########################
"""
# setup time
total_frames = 500
left_edge = -total_frames // 4
right_edge = total_frames + left_edge
time_res = 2.56e-6  # s
trange = np.arange(-1, total_frames + 2) * time_res + left_edge * time_res

# setup figure
fig = plt.figure()
ax = fig.add_subplot(111)
axsc = ax.scatter([], [], s=4)
time_axis_scale = 1e-3
time_axis_str = "m"
freq_axis_scale = 1e6
freq_axis_str = "M"
ax.set_ylim(400, 800)
ax.set_xlim(
    left_edge * time_res / time_axis_scale, right_edge * time_res / time_axis_scale
)
ax.set_ylabel(f"Freq. [{freq_axis_str}Hz]")
ax.set_xlabel(f"Time [{time_axis_str}s]")
cmap = cmaps.binary
norm = LogNorm(vmin=1e-3, vmax=1)
axsc.set_cmap(cmap)
axsc.set_norm(norm)
cb = fig.colorbar(cmaps.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
cb.ax.set_title("Img. Mag.")

# image framing
image_duration = 2  # seconds
frame_interval = 30e-3  # seconds between frames
total_aniframes = image_duration / frame_interval
stepsize = np.ceil((trange.size - 1) / total_aniframes).astype(int)

if stepsize == 0:
    stepsize = 1

trange_inds = np.arange(0, trange.size - 1, step=stepsize)


# animate frame
def update(i):
    tcutt = (delayvals > trange[trange_inds[i]]) * (
        delayvals <= trange[trange_inds[i + 1]]
    )
    data = np.stack(
        [delayvals[tcutt] / time_axis_scale, fvals[tcutt] / freq_axis_scale]
    ).T
    axsc.set_offsets(data)
    axsc.set_array(np.abs(magvals[tcutt]))
    return (axsc,)


# animate
ani = matplotlib.animation.FuncAnimation(
    fig, update, frames=trange_inds.size - 1, interval=30, blit=True
)

# save
save_path = Path.cwd()
save_path = save_path / "singelens_baseband_arrival.gif"
ani.save(filename=str(save_path), writer="pillow")

"""
#######################################
#### Animate Temporal and Spatial #####
#######################################
"""
# setup time
total_frames = 100
left_edge = -total_frames // 4
right_edge = total_frames + left_edge
time_res = 2.56e-6  # s
trange = np.arange(-1, total_frames + 2) * time_res + left_edge * time_res
fmin = np.amin(fvals)

# Setup plot
fig = plt.figure()
ax = fig.add_subplot(111)
axsc = ax.scatter([], [], s=4)
axisscale = 1e0
axisstr = ""
framescale = 1e6
framestr = "M"
scaling = scale * 206264.806247 / axisscale

# Select largest spatial extent
cut1 = fvals == np.amin(fmin)  # lowest freq for scattering
maxv_ = (
    max(np.amax(np.abs(txvals[cut1])), np.amax(np.abs(tyvals[cut1]))) * scaling * 1.1
)

# Set axes and plot
ax.set_ylim(-maxv_, maxv_)
ax.set_xlim(-maxv_, maxv_)
ax.set_ylabel(f"$\\theta_Y$ [{axisstr}arcsec]", size=14)
ax.set_xlabel(f"$\\theta_X$ [{axisstr}arcsec]", size=14)
ax.set_facecolor("black")
cmap = cmaps.gray
norm = LogNorm(vmin=1e-3, vmax=1)
axsc.set_cmap(cmap)
axsc.set_norm(norm)
cb = fig.colorbar(cmaps.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
cb.ax.set_title("Img. Mag.", y=1.02)

# image framing
image_duration = 2  # seconds
frame_interval = 30e-3  # seconds between frames
total_aniframes = image_duration / frame_interval
stepsize = np.ceil((trange.size - 1) / total_aniframes).astype(int)

if stepsize == 0:
    stepsize = 1

trange_inds = np.arange(0, trange.size - 1, step=stepsize)


# frame animation
def update(i):
    tcutt = (delayvals[cut1] > trange[trange_inds[i]]) * (
        delayvals[cut1] <= trange[trange_inds[i + 1]]
    )

    data = np.stack([txvals[cut1][tcutt] * scaling, tyvals[cut1][tcutt] * scaling]).T
    axsc.set_offsets(data)
    axsc.set_array(np.abs(magvals[cut1][tcutt]))
    ax.set_title(f"Freq: {fmin/framescale:.0f} [{framestr}Hz] ", size=14)
    return (axsc,)


# animate
ani = matplotlib.animation.FuncAnimation(
    fig, update, frames=trange_inds.size - 1, interval=30, blit=True
)

# save
save_path = Path.cwd()
save_path = save_path / "singlelens_baseband_spatial_arrival.gif"
ani.save(filename=str(save_path), writer="pillow")
