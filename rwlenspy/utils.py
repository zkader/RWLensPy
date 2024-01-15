"""Utility Functions for generating lensing functions."""
import typing as T

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from astropy import constants as const
from astropy import units as u
from astropy.units import Quantity
from scipy.fft import fftfreq, fftn, ifftn


def FermatPotential(
    rx: npt.ArrayLike,
    ry: npt.ArrayLike,
    sx: float,
    sy: float,
    D_eff: float,
    lens_func: T.Callable,
    **funcargs
) -> npt.ArrayLike:
    """Get the Fermat potential.

    Get the Fermat potential for a single lens plane. A given lensing
    function will generate the lensing array for the Fermat potential.
    The function must be of the form func(rx,ry,**funcargs) where the
    positions on the plane will map to the geometric delay and lensing
    delay at that point.

    Args:
        rx (array[float]): Array of X positions on the lens plane.
        ry (array[float]): Array of Y positions on the lens plane.
        sx (float): X position of the source point.
        sy (float): Y position of the source point.
        D_eff (float): The effective distance of the lensing system.
        lens_func (func): The function must be of the form
                          func(rx,ry,**funcargs).
        **funcargs: Arguments for the lensing function.

    Returns:
        array[float]: The Fermat potential
    """
    Geom_del = (D_eff / const.c).to(u.s).value * 0.5 * ((rx - sx) ** 2 + (ry - sy) ** 2)

    Lens_del = lens_func(rx, ry, **funcargs)

    return Geom_del + Lens_del


def DMLens(
    rx: npt.ArrayLike,
    ry: npt.ArrayLike,
    freq: Quantity = 1 * u.Hz,
    DM: Quantity = 1 * u.pc * u.cm**-3,
    **funcargs
):
    """Get the lensing delay from a lens of constant DM.

    Args:
        rx (array[float]): Array of X positions on the lens plane.
        ry (array[float]): Array of Y positions on the lens plane.
        freq (float, optional): Frequency of the observation.
                                [Hz] Defaults to 1.
        DM (float, optional): The DM of the lens. [pc cm^-3] Defaults to 1.

    Returns:
        float: The lensing delay.
    """
    r_e = const.alpha**2 * const.a0  # classical electron radius
    plasma_const = r_e * const.c / (2 * np.pi * (freq.to(1 / u.s)) ** 2)
    plasma_const = plasma_const.to(u.pc**-1 * u.cm**3 * u.s)

    return plasma_const.value * DM


def PlasmaPhaseLens(
    rx: npt.ArrayLike,
    ry: npt.ArrayLike,
    freq: Quantity = 1 * u.Hz,
    ne_mat: T.Optional[npt.ArrayLike] = None,
    **funcargs
):
    """Get the lensing delay from a given electron column density array.

    Args:
        rx (array[float]): Array of X positions on the lens plane.
        ry (array[float]): Array of Y positions on the lens plane.
        freq (float, optional): Frequency of the observation.
                                [Hz] Defaults to 1.
        ne_mat (array[float], optional): _description_. Defaults to None.
        **funcargs: Arguments for the lensing function.

    Returns:
        array[float]: The lensing delay.
    """
    r_e = const.alpha**2 * const.a0  # classical electron radius
    plasma_const = r_e * const.c / (2 * np.pi * (freq.to(1 / u.s)) ** 2)
    plasma_const = plasma_const.to(u.pc**-1 * u.cm**3 * u.s)

    ne_mat = plasma_const.value * ne_mat

    return ne_mat


def GaussianPlasmaLens(
    rx: npt.ArrayLike,
    ry: npt.ArrayLike,
    freq: Quantity = 1 * u.Hz,
    scale: float = 1,
    DM_: float = 1,
    **funcargs
):
    """_summary_

    Args:
        rx (array[float]): Array of X positions on the lens plane.
        ry (array[float]): Array of Y positions on the lens plane.
        freq (float, optional): Frequency of the observation.
                                [Hz] Defaults to 1.
        ne_mat (array[float], optional): _description_. Defaults to None.
        **funcargs: Arguments for the lensing function.
        scale (int, optional): _description_. Defaults to 1.
        n_e (int, optional): _description_. [pc cm^-3] Defaults to 1.

    Returns:
        _type_: _description_
    """
    r_e = const.alpha**2 * const.a0  # classical electron radius
    plasma_const = DM_ * r_e * const.c / (2 * np.pi * (freq) ** 2)
    plasma_const = plasma_const.to(u.s).value

    Lens_del = plasma_const * np.exp(-0.5 * ((rx) ** 2 + (ry) ** 2) / (scale**2))
    return Lens_del


def MultiGaussianPlasmaLens(
    rx: npt.ArrayLike,
    ry: npt.ArrayLike,
    freq: Quantity = 1*u.Hz,
    scale: np.ndarray = np.array([]),
    N_e: np.ndarray = np.array([]),
    posx: np.ndarray = np.array([]),
    posy: np.ndarray = np.array([]),
    **funcargs
) -> npt.ArrayLike:
    """
    _summary_

    _extended_summary_

    Parameters
    ----------
    rx : npt.ArrayLike
        _description_
    ry : npt.ArrayLike
        _description_
    freq : Quantity, optional
        _description_, by default 1
    scale : np.ndarray, optional
        _description_, by default np.array([])
    N_e : np.ndarray, optional
        _description_, by default np.array([])
    posx : np.ndarray, optional
        _description_, by default np.array([])
    posy : np.ndarray, optional
        _description_, by default np.array([])

    Returns
    -------
    npt.ArrayLike
        _description_
    """
    assert posx.shape == posy.shape
    assert posx.shape == N_e.shape
    assert N_e.shape == scale.shape

    r_e = const.alpha**2 * const.a0  # classical electron radius
    Lens_del = 0
    plasma_const = r_e * const.c / (2 * np.pi * (freq) ** 2)

    for ii in range(scale.shape[0]):
        plasma_const_ii = (N_e[ii] * plasma_const).to(u.s).value
        Lens_del = Lens_del + plasma_const_ii * np.exp(
            -0.5 * ((rx - posx[ii]) ** 2 + (ry - posy[ii]) ** 2) / (scale[ii] ** 2)
        )
    return Lens_del


def GravitationalPMLens(rx, ry, mass=1, **funcargs):
    """
    mass = solar mass
    """
    Eins_time_const = (4 * const.G * mass * const.M_sun / const.c**3).to(u.s).value
    Lens_del = -Eins_time_const * np.log(np.sqrt((rx) ** 2 + (ry) ** 2))
    return Lens_del


def get_plasma_Ne(
    rx_size: int,
    ry_size: int,
    dr: float,
    theta_in: float,
    theta_out: float,
    C2_n: float = 1,
    freq: float = 1,
    D_eff: float = 1,
    seed: int = None,
    plot: bool = False,
) -> npt.ArrayLike:
    theta_fres = np.sqrt(const.c / (2 * np.pi * freq * D_eff)).to(u.m / u.m)
    t_inn = theta_in / theta_fres
    t_out = theta_out / theta_fres

    dtheta = dr / theta_fres

    k_1 = fftfreq(rx_size, d=dtheta)
    k_2 = fftfreq(ry_size, d=dtheta)
    k1v, k2v = np.meshgrid(k_1, k_2)

    if type(seed) is not None:
        np.random.seed(seed)

    n_e = np.random.normal(loc=0, scale=1, size=(rx_size, ry_size))
    n_e = fftn(n_e, axes=(-2, -1))

    lmin = t_inn.value
    lmax = t_out.value
    P_ne = (
        C2_n
        * (k1v**2 + k2v**2 + (1 / lmax) ** 2) ** (-11 / (2 * 3))
        * np.exp(-0.5 * (k1v**2 + k2v**2) / (1 / lmin) ** 2)
        / np.sqrt(2 * np.pi * (1 / lmin) ** 2)
    )

    n_e = n_e * np.sqrt(P_ne)
    n_e = ifftn(n_e, axes=(-2, -1)).real
    n_e = n_e - np.mean(n_e)

    if plot:
        plt.figure()
        plt.plot(np.sqrt(k1v**2 + k2v**2).ravel(), P_ne.ravel())
        plt.yscale("log")
        plt.xscale("log")
        plt.show()
    return n_e


def AnalyticPointMassGrav(
    y: float,
    geo_par: float,
    lens_par: float,
) -> T.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get the observables for a point mass gravitational lens.

    Get the observables for a point mass gravitational lens using the
    analytic solution for a point mass lens.

    Parameters
    ----------
    y : float
        _description_
    geo_par : float
        _description_
    lens_par : float
        _description_

    Returns
    -------
    T.Tuple[np.ndarray, np.ndarray, np.ndarray]
        _description_
    """

    # stationary point func
    def _lensgrad(y):
        im_p = 0.5 * (y + np.sqrt(y * y + 4))
        im_m = 0.5 * (y - np.sqrt(y * y + 4))
        return im_p, im_m

    # lens delay func
    def _lensdel(x, y, mu_g, mu_l):
        return mu_g * 0.5 * (x - y) ** 2 - mu_l * np.log(np.abs(x))

    # lens mag func
    def _lensmag(x):
        return 1 / np.sqrt((1 - 1 / x**4) + 0j)

    xpos1, xpos2 = _lensgrad(y)
    xposvv = np.array([xpos1, xpos2])

    i_delvv = _lensdel(xposvv, y, geo_par, lens_par)
    i_magvv = _lensmag(xposvv)

    return xposvv, i_delvv, i_magvv


def AnalyticGaussPlasma(
    y: float,
    geo_par: float,
    lens_par: float,
    freqvals: npt.ArrayLike,
    xm: float = 10,
    N: int = 20001,
) -> T.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    _summary_

    _extended_summary_

    Parameters
    ----------
    y : float
        _description_
    geo_par : float
        _description_
    lens_par : float
        _description_
    freqvals : npt.ArrayLike
        _description_
    xm : float, optional
        _description_, by default 10
    N : int, optional
        _description_, by default 20001

    Returns
    -------
    T.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        _description_
    """
    xvv = np.linspace(-xm, xm, N)  # in lens scale units

    # stationary point func
    def _lensgrad(x, mu):
        return x * (1 - mu * np.exp(-0.5 * x * x))

    # lens delay func
    def _lensdel(x, y, mu1, mu2):
        return mu1 * 0.5 * (x - y) ** 2 + mu2 * np.exp(-0.5 * x * x)

    # lens mag func
    def _lensmag(x, mu):
        return (
            (1 + 0 * 1j - mu * np.exp(-0.5 * x * x))
            * (1 - mu * np.exp(-0.5 * x * x) * (1 - x * x))
        ) ** (-0.5)

    freqvv = np.array([])
    xposvv = np.array([])
    i_delvv = np.array([])
    i_magvv = np.array([])
    for ii, tfi in enumerate(freqvals):
        mu_geom = geo_par
        mu_lens = lens_par / tfi**2

        mu_ = mu_lens / mu_geom

        idx = np.argwhere(np.diff(np.sign(_lensgrad(xvv, mu_) - y))).flatten()

        freqvv = np.append(freqvv, freqvals[ii] * np.ones(idx.size))

        xposvv = np.append(xposvv, xvv[idx])

        i_delvv = np.append(i_delvv, _lensdel(xvv[idx], y, mu_geom, mu_lens))

        i_magvv = np.append(i_magvv, _lensmag(xvv[idx], mu_))

    return freqvv, xposvv, i_delvv, i_magvv
