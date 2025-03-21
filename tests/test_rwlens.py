from time import time

import numpy as np
from astropy import constants as c
from astropy import cosmology
from astropy import units as u
from scipy.fft import rfftfreq, rfft, irfft, fftfreq, ifft, fft, fftshift

import rwlenspy.lensing as rwl
import rwlenspy.utils as utils


def test_imageobservables():
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
    # Memory in bytes
    maxmemory = 2E9

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
        maxmemory
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


def test_analytictransferfunc():
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
    mass = 1e-2  # solar mass
    theta_E = np.sqrt(mass * Eins_time_const * c.c * const_D).to(u.m / u.m)
    freq_ref = 800e6
    dumpframes = 2048
    freqs = freq_ref - rfftfreq(dumpframes, d=1 / freq_ref)
    beta_x = 2.5
    beta_y = 0.0

    # Lens Parameters
    geom_const = ((1 / (const_D * c.c)).to(u.s)).value
    geom_const = geom_const * theta_E.value**2
    lens_const = mass * Eins_time_const.to(u.s).value

    nyqzone = True

    # grav 2
    # Solutions from Algorithm
    print("Getting the transfer function with Algorithm...")
    t1 = time()
    transferfunc = rwl.RunGravTransferFunc(freqs, beta_x, beta_y, mass, nyqzone)
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

    analytic_tf = mag_analytic[0] * np.exp(
        1j * 2 * np.pi * freqs * delay_analytic[0]
    ) + mag_analytic[1] * np.exp(1j * 2 * np.pi * freqs * delay_analytic[1])

    delta_analytic = irfft(analytic_tf)

    delta_transfer = irfft(transferfunc)

    assert np.argmax(delta_analytic) == np.argmax(delta_transfer)

    assert (np.abs(delta_analytic - delta_transfer) < 1e-10).all()
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
    mass = 1e-3  # solar mass
    theta_E = np.sqrt(mass * Eins_time_const * c.c * const_D).to(u.m / u.m)
    dumpframes = 2048
    tres = 1 / 800e6
    freqs = 800e6 - rfftfreq(dumpframes, d=tres)
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

    nyqzone = True

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
        nyqzone,
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

    max_mag_arg = np.argmax(np.abs(mag_analytic))

    analytic_tf = mag_analytic[0] * np.exp(
        1j * 2 * np.pi * freqs * delay_analytic[0]
    ) + mag_analytic[1] * np.exp(1j * 2 * np.pi * freqs * delay_analytic[1])

    analytic_tf = analytic_tf / ( 
        mag_analytic[max_mag_arg] 
        * np.exp(1j * 2 * np.pi * freqs * delay_analytic[max_mag_arg])
    )

    delta_analytic = irfft(analytic_tf)

    delta_transfer = irfft(transferfunc)

    # check peaks line up in time
    assert np.argmax(np.abs(delta_analytic)) == np.argmax(np.abs(delta_transfer))

    corr_transfer = irfft(np.abs(transferfunc) ** 2)
    expected_delay_ind = np.round(
        np.abs(delay_analytic[0] - delay_analytic[1]) / (tres)
    ).astype(int)

    # check delayed signal is at expected delay
    assert np.argmax(np.abs(corr_transfer[1:1024])) == expected_delay_ind

    corr_zero_peak = np.sum(np.abs(mag_analytic) ** 2)

    # Check if total image magnification is within error expectations
    assert (np.abs(corr_transfer[0]) - corr_zero_peak) / corr_zero_peak < 1e-2

    return


def test_dedisperion():
    cosmo = cosmology.Planck18

    # Comoving
    D_obs_src = cosmo.comoving_distance(1)
    D_obs_len = 1 * u.kpc

    # Redshift
    # z_obs_len = 0
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
    lens_scale = (10 * u.AU / D_obs_len).to(u.m / u.m)
    scale = lens_scale.value
    geom_const = ((1 / (const_D * c.c)).to(u.s)).value
    geom_const = geom_const * scale**2
    lens_const = kdm
    DM = 0.001
    freq_power = -2.0
    beta_x = 0.0
    beta_y = 0.0

    # Grid Parameters
    max_fres = 5.0
    theta_min = -max_fres
    theta_max = max_fres
    theta_N = 201

    # Lens Parameters
    freq_ref = 0
    bb_frames = 5
    freqs = 800e6 - rfftfreq(2048 * bb_frames, d=1 / (800e6))  # MHz
    freqs = freqs.astype(np.double).ravel(order="C")

    nyqalias = True

    # Spatial Grid
    x1 = np.arange(theta_N) * (theta_max - theta_min) / (theta_N - 1) + theta_min

    # Lens Functions
    lens_arr = utils.ConstantLens(x1[:, None], x1[None, :], DM)
    lens_arr = lens_arr.astype(np.double).ravel(order="C")

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
        nyqalias,
    )
    tv = time() - t1
    print(
        "Tranfer function obtained in:", tv, "s", " | ", tv / 60, "min", tv / 3600, "hr"
    )
    transferfunc = np.array(transferfunc).astype(np.cdouble)

    # Analytic dispersion for alias sampled freq
    dedispersion_tf = np.exp(
        2j * np.pi * kdm * DM * freqs**(-1)
    )

    # dedispersion removes the phase
    assert (np.abs(np.angle(transferfunc * dedispersion_tf)) < 1e-10).all()

    return

def test_phasegeneration():
    # Memory in bytes
    maxmemory = 2E9

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
    mass = 1e0  # solar mass
    theta_E = np.sqrt(mass * Eins_time_const * c.c * const_D).to(u.m / u.m)
    windowsamples = 2048
    bb_frames = 500
    dumpframes = windowsamples * bb_frames
    tres = 1 / 800e6
    freqs = 800e6 - rfftfreq(dumpframes, d=tres)
    freqs = freqs.astype(np.double).ravel(order="C")
    beta_x = 0.5
    beta_y = 0.0

    new_freqs = freqs[:bb_frames+1]

    # Grid parameters
    max_fres = 5
    theta_min = -max_fres
    theta_max = max_fres
    theta_N = 251
    nyqalias = True
    verbose=False

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

    # Get Transfer Function
    print("Getting the transfer function with Algorithm...")
    t1 = time()
    transferfunc = rwl.RunUnitlessTransferFunc(
        theta_min,
        theta_max,
        theta_N,
        new_freqs,
        freq_ref,
        lens_arr,
        beta_x,
        beta_y,
        geom_const,
        lens_const,
        freq_power,
        nyqalias,
        verbose=verbose
    )
    tv = time() - t1
    print("Tranfer function obtained in:", tv, "s", " | ", tv / 60, "min", tv / 3600, "hr")
    transferfunc = np.array(transferfunc).astype(np.cdouble)

    freqss = 800e6 - rfftfreq(windowsamples*1,d=tres)
    freqn = freqss.size
    # Solutions from Algorithm
    print("Getting the Images with Algorithm...")
    t1 = time()
    txvals, tyvals, fvals, delayvals, magvals = rwl.GetUnitlessFreqStationaryPoints(
        theta_min,
        theta_max,
        theta_N,
        lens_arr,
        new_freqs,
        beta_x,
        beta_y,
        geom_const,
        lens_const,
        freq_power,
        maxmemory,
        verbose=verbose
    )
    tv = time() - t1
    print("Images obtained in:", tv, "s", " | ", tv / 60, "min", tv / 3600, "hr")

    txvals = np.asarray(txvals)
    tyvals = np.asarray(tyvals)
    fvals = np.asarray(fvals)
    delayvals = np.asarray(delayvals)
    magvals = np.asarray(magvals)


    ref_delay_f = np.where(fvals == freq_ref)
    ref_arg = np.argmax(magvals[ref_delay_f])  
    ref_mag = magvals[ref_delay_f][ref_arg]
    ref_delay = delayvals[ref_delay_f][ref_arg]

    print(ref_mag,ref_delay)

    delayvals = delayvals - ref_delay
    magvals = magvals / np.abs(ref_mag)

    subchan_tf = np.zeros(transferfunc.shape) + 0j

    for i in range(new_freqs.size):
        fcut = np.where(new_freqs[i] == fvals)

        img_mags = magvals[fcut] 
        img_dels = delayvals[fcut]

        subchan_tf[i] = np.sum( img_mags*np.exp(1j*2*np.pi*(new_freqs[i])*img_dels),axis=-1)


    chan_center = freqss[[0,1]]

    subchan_tf2 = np.zeros((chan_center.size,bb_frames)) + 0j

    subchan_freqs = fftfreq(bb_frames,d=windowsamples*tres)

    all_chans = chan_center[:,None] - subchan_freqs[None,:]

    print(all_chans.shape)
    for i in range(chan_center.size):
        fcut = np.where(chan_center[i] == fvals)

        img_mags = magvals[fcut] 
        img_dels = delayvals[fcut]

        subchan_tf2[i,:] = np.sum( img_mags[None,:]*np.exp(1j*2*np.pi*(all_chans[i,:][:,None])*img_dels[None,:]),axis=-1)

    all_chans = fftshift(all_chans,axes=-1).ravel()
    subchan_tf2 = fftshift(subchan_tf2,axes=-1).ravel()

    fcut = (all_chans <= chan_center[0] ) * (all_chans >= chan_center[1])

    all_chans = all_chans[fcut]
    subchan_tf2 = subchan_tf2[fcut] 

    assert (np.abs( np.abs(transferfunc) - np.abs(subchan_tf2)) < 1e-10).all()
    assert (np.abs( np.angle(transferfunc) - np.angle(subchan_tf2)) < 1e-10).all()

    assert (np.abs( np.abs(transferfunc) - np.abs(subchan_tf)) < 1e-10).all()
    assert (np.abs( np.angle(transferfunc) - np.angle(subchan_tf)) < 1e-10).all()

    assert (np.abs( np.abs(subchan_tf2) - np.abs(subchan_tf)) < 1e-10).all()
    assert (np.abs( np.angle(subchan_tf2) - np.angle(subchan_tf)) < 1e-10).all()

    return

def test_morphologygeneration():
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
    maxmemory = 2E9

    cosmo = cosmology.Planck18

    # Comoving
    D_obs_src = cosmo.comoving_distance(1)
    D_obs_len = D_obs_src - 1*u.kpc

    # Redshift
    z_obs_len = cosmology.z_at_value(cosmo.comoving_distance,D_obs_src)
    z_obs_src = cosmology.z_at_value(cosmo.comoving_distance,D_obs_src)

    # Ang. Diam. Distance
    D_obs_len = cosmo.angular_diameter_distance(z_obs_len) - 1*u.kpc
    D_obs_src = cosmo.angular_diameter_distance(z_obs_src)
    D_len_src = 1*u.kpc

    # Physical Lens Params
    const_D = D_len_src / (D_obs_len * D_obs_src)
    r_e = c.alpha**2 * c.a0  # classical electron radius
    kdm = (
        (r_e * c.c / (2 * np.pi)).to(u.cm**2 / u.s)
        * ((1.0 * u.pc / u.cm).to(u.m / u.m)).value
    ).value
    lens_scale = (1 * u.AU / D_obs_len).to(u.m / u.m)
    scale = lens_scale.value
    geom_const = ((1 / (const_D * c.c)).to(u.s)).value
    geom_const = geom_const * scale**2
    DM = 0.02
    lens_const = kdm * DM
    freq_power = -2.0
    beta_x = 0.75
    beta_y = 0.0

    # Grid Parameters
    max_fres = 7
    theta_min = -max_fres
    theta_max = max_fres
    theta_N = 201
    nyqalias = True
    verbose=False

    chan_samp = 256

    # Lens Parameters
    freq_ref = 800e6
    bb_frames = 500
    freqs = 800e6 - rfftfreq(chan_samp * bb_frames, d=1 / (800e6))  # MHz
    freqs = freqs.astype(np.double).ravel(order="C")

    # Spatial Grid
    x1 = np.arange(theta_N) * (theta_max - theta_min) / (theta_N - 1) + theta_min

    # Lens Functions
    lens_arr = utils.RationalLens(x1[:, None], x1[None, :])
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
        verbose=verbose
    )
    tv = time() - t1
    print("Tranfer function obtained in:", tv, "s", " | ", tv / 60, "min", tv / 3600, "hr")
    transferfunc = np.array(transferfunc).astype(np.cdouble)

    freqss = 800e6 - rfftfreq(chan_samp*1,d=1/800e6)
    freqn = freqss.size
    # Solutions from Algorithm
    print("Getting the Images with Algorithm...")
    t1 = time()
    txvals, tyvals, fvals, delayvals, magvals = rwl.GetUnitlessFreqStationaryPoints(
        theta_min,
        theta_max,
        theta_N,
        lens_arr,
        freqss,
        beta_x,
        beta_y,
        geom_const,
        lens_const,
        freq_power,
        maxmemory,
        verbose=verbose    
    )
    tv = time() - t1
    print("Images obtained in:", tv, "s", " | ", tv / 60, "min", tv / 3600, "hr")

    txvals = np.asarray(txvals)
    tyvals = np.asarray(tyvals)
    fvals = np.asarray(fvals)
    delayvals = np.asarray(delayvals)
    magvals = np.asarray(magvals)


    tres = 1.25e-9  # s
    times = np.arange(chan_samp * bb_frames) * tres

    sig_ = np.zeros(times.size)
    sig_[0] = 1

    vstream_1 = sig_.copy()

    # full tf
    sig_ = irfft(rfft(sig_) * transferfunc)
    vstream_2 = sig_

    vstream_1 = vstream_1.reshape((vstream_1.shape[-1] // chan_samp, chan_samp))
    vstream_2 = vstream_2.reshape((vstream_2.shape[-1] // chan_samp, chan_samp))

    baseband_chan_to_transf = rfft(vstream_1, axis=-1).T

    baseband_transf_to_chan = rfft(vstream_2, axis=-1).T

    ref_delay_f = np.where(fvals == freq_ref)
    ref_arg = np.argmax(magvals[ref_delay_f])  
    ref_mag = magvals[ref_delay_f][ref_arg]
    ref_delay = delayvals[ref_delay_f][ref_arg]

    delayvals = delayvals - ref_delay
    magvals = magvals / ref_mag

    subchan_freqs = fftfreq(bb_frames, d=1.25e-9 * chan_samp)

    baseband_chan_to_transf = fft(baseband_chan_to_transf,axis=-1)
    subchan_tf = np.zeros_like(baseband_chan_to_transf)

    for i in range(freqss.size):
        fcut = np.where(freqss[i] == fvals)

        img_mags = magvals[fcut] 
        img_dels = delayvals[fcut]
        subchan_tf[i,:] =  np.sum( img_mags[None,:]*np.exp(-1j*2*np.pi*(subchan_freqs[:,None])*img_dels[None,:] ),axis=-1)

    baseband_chan_to_transf = baseband_chan_to_transf * subchan_tf
    baseband_chan_to_transf = ifft(baseband_chan_to_transf,axis=-1)

    t1 = np.sum(np.abs(baseband_chan_to_transf) ** 2,axis = 0) 
    t1 = t1 / np.amax(t1)
    t2 = np.sum(np.abs(baseband_transf_to_chan) ** 2,axis = 0)
    t2 = t2 / np.amax(t2)

    assert np.sum(t1 * t2)/ np.sqrt(np.sum(t1**2) * np.sum(t2**2)) > 0.9

    return