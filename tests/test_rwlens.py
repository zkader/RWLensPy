from time import time

import numpy as np
from astropy import constants as c
from astropy import cosmology
from astropy import units as u
from scipy.fft import rfftfreq,rfft,irfft

import rwlenspy.lensing as rwl
import rwlenspy.utils as utils


def test_analyticgrav():
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
    cosmo = cosmology.Planck18
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
    mass = 1  # solar mass
    theta_E = np.sqrt(mass * Eins_time_const * c.c * const_D).to(u.m / u.m)
    freqs = [freq_ref]
    beta_x = 2.5
    beta_y = 0.0

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

    # Lens Functions
    lens_arr = -utils.LogLens(x1[None, :], x1[:, None])
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
    )
    tv = time() - t1
    print("Images obtained in:", tv, "s", " | ", tv / 60, "min", tv / 3600, "hr")

    txvals = np.asarray(txvals)
    tyvals = np.asarray(tyvals)
    fvals = np.asarray(fvals)
    delayvals = np.asarray(delayvals)
    magvals = np.asarray(magvals)

    # Analytic Solutions
    xpos_analytic, delay_analytic, mag_analytic = utils.AnalyticPointMassGrav(
        beta_x, geom_const, lens_const
    )

    # check for two solutions
    assert fvals.size == delay_analytic.size

    # check for delay accuracy
    delay_diff = (
        (delayvals[0] - delayvals[1]) - (delay_analytic[1] - delay_analytic[0])
    ) / (delay_analytic[1] - delay_analytic[0])

    assert np.abs(delay_diff) < 1e-4  # Frac. error for 501 x 501

    # check for mag. accuracy
    mag_diff = (
        np.abs(magvals[0] / magvals[1]) - np.abs(mag_analytic[1] / mag_analytic[0])
    ) / np.abs(mag_analytic[1] / mag_analytic[0])
    assert np.abs(mag_diff) < 1e-1  # Frac. error for 501 x 501

    return


def test_transferfunc():
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
    cosmo = cosmology.Planck18
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
    mass = 0.01  # solar mass
    theta_E = np.sqrt(mass * Eins_time_const * c.c * const_D).to(u.m / u.m)
    dumpframes = 2048
    freqs = freq_ref - rfftfreq(dumpframes,d=1/freq_ref)
    beta_x = 2.5
    beta_y = 0.0

    # Grid parameters
    max_fres = 1
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

    # Lens Functions
    lens_arr = -utils.LogLens(x1[None, :], x1[:, None])
    lens_arr = lens_arr.astype(np.double).ravel(order="C")

    nyqalias = True
    # Solutions from Algorithm
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
        nyqalias
    )
    tv = time() - t1
    print(
        "Tranfer function obtained in:", tv, "s", " | ", tv / 60, "min", tv / 3600, "hr"
    )
    transferfunc = np.array(transferfunc).astype(np.cdouble)

    # Analytic Solutions
    _, delay_analytic, mag_analytic = utils.AnalyticPointMassGrav(
        beta_x, geom_const, lens_const
    )

    delta_func = np.zeros(dumpframes)
    delta_func[1] = 1
    delta_func = rfft(delta_func)
   
    analytic_tf = mag_analytic[0]*np.exp(1j*2*np.pi*freqs*delay_analytic[0])\
        + mag_analytic[1]*np.exp(1j*2*np.pi*freqs*delay_analytic[1])

    delta_analytic = irfft(delta_func*analytic_tf)

    delta_transfer = irfft(delta_func*transferfunc)
    
    assert np.argmax(delta_analytic) == np.argmax(delta_transfer)
    
    return
