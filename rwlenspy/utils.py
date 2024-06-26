"""Utility Functions for generating lensing functions."""
import typing as T

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.fft import fftfreq, fftn, ifftn


def FermatPotential(
    rx: npt.ArrayLike,
    ry: npt.ArrayLike,
    sx: float,
    sy: float,
    freq: float,
    geom_par: float,
    lens_par: float,
    lens_freq_scaling: float,
    lens_func: T.Callable,
    **funcargs
) -> npt.ArrayLike:
    """
    Get the Fermat potential with a given lensing function.

    Get the Fermat potential for a single lens plane. A given lensing
    function will generate the lensing array for the Fermat potential.
    The function must be of the form func(rx,ry,**funcargs) where the
    positions on the plane will map to the geometric delay and lensing
    delay at that point.


    Parameters
    ----------
    rx : ArrayLike
        ArrayLike of X positions on the lens plane.
    ry : ArrayLike
        ArrayLike of Y positions on the lens plane.
    sx : float
        X position of the source point.
    sy : float
        Y position of the source point.
    freq : float
        Frequency of light being propagated.
    geom_par : float
        The geometric parameter of the Fermat potential.
    lens_par : float
        The lensing parameter of the Fermat potential.
    lens_freq_scaling : float
        The frequency power scaling of the lens.
    lens_func : Callable
        The function must be of the form func(rx,ry,**funcargs).
    **funcargs : dict, optional
        Extra arguments to `lens_func`.

    Returns
    -------
    ArrayLike
        The Fermat potential time delay
    """
    Geom_del = geom_par * 0.5 * ((rx - sx) ** 2 + (ry - sy) ** 2)
    Lens_del = lens_par * freq**lens_freq_scaling * lens_func(rx, ry, **funcargs)

    return Geom_del + Lens_del


def ConstantLens(
    rx: npt.ArrayLike, ry: npt.ArrayLike, constant: float = 1
) -> npt.ArrayLike:
    """
    Get the lensing function of a constant lens.

    Parameters
    ----------
    rx : ArrayLike
        ArrayLike of X positions on the lens plane.
    ry : ArrayLike
        ArrayLike of Y positions on the lens plane.
    constant : float, optional
        The constant value of the lens, by default 1

    Returns
    -------
    ArrayLike
        The value of the lensing function.
    """
    return 0 * rx + 0 * ry + constant


def NumericGridLens(
    rx: npt.ArrayLike, ry: npt.ArrayLike, lens_del: T.Optional[npt.ArrayLike] = None
) -> npt.ArrayLike:
    """
    Get a numerical lens array in a compatible format for FermatPotential().

    Parameters
    ----------
    rx : ArrayLike
        ArrayLike of X positions on the lens plane.
    ry : ArrayLike
        ArrayLike of Y positions on the lens plane.
    lens_del : ArrayLike or None, optional
        A ArrayLike of lensing delay functions, by default None

    Returns
    -------
    ArrayLike
        The value of the lensing function.
    """
    if lens_del is None:
        return 0 * rx + 0 * ry
    else:
        return 0 * rx + 0 * ry + lens_del


def GaussianLens(
    rx: npt.ArrayLike, ry: npt.ArrayLike, scale: float = 1, amp: float = 1
) -> npt.ArrayLike:
    """
    Get a Gaussian lens.

    Parameters
    ----------
    rx : ArrayLike
        ArrayLike of X positions on the lens plane.
    ry : ArrayLike
        ArrayLike of Y positions on the lens plane.
    scale : float, optional
        The width of the Gaussian, by default 1
    amp : float, optional
        The amplitude of the Gaussian, by default 1

    Returns
    -------
    ArrayLike
        The value of the lensing function.
    """
    Lens_del = amp * np.exp(-0.5 * ((rx) ** 2 + (ry) ** 2) / (scale**2))
    return Lens_del


def RationalLens(
    rx: npt.ArrayLike, ry: npt.ArrayLike, scale: float = 1, amp: float = 1
) -> npt.ArrayLike:
    """
    Get a Rational lens.

    Parameters
    ----------
    rx : ArrayLike
        ArrayLike of X positions on the lens plane.
    ry : ArrayLike
        ArrayLike of Y positions on the lens plane.
    scale : float, optional
        The width of the lens, by default 1
    amp : float, optional
        The amplitude of the lens, by default 1

    Returns
    -------
    ArrayLike
        The value of the lensing function.
    """
    Lens_del = amp / (1 + 0.5 * ((rx) ** 2 + (ry) ** 2) / (scale**2))
    return Lens_del


def MultiGaussianLens(
    rx: npt.ArrayLike,
    ry: npt.ArrayLike,
    scales: np.ndarray = np.array([1]),
    amps: np.ndarray = np.array([1]),
    posx: np.ndarray = np.array([0]),
    posy: np.ndarray = np.array([0]),
) -> npt.ArrayLike:
    """
    Get a lens consisting of multiple Gaussian lenses.

    Parameters
    ----------
    rx : ArrayLike
        ArrayLike of X positions on the lens plane.
    ry : ArrayLike
        ArrayLike of Y positions on the lens plane.
    scales : ndarray, optional
        Array of widths for the Gaussians, by default np.array([1])
    amps : ndarray, optional
        Array of amplitudes for the Gaussians, by default np.array([1])
    posx : ndarray, optional
        Array of X positions for the Gaussians, by default np.array([0])
    posy : ndarray, optional
        Array of Y positions for the Gaussians, by default np.array([0])

    Returns
    -------
    ArrayLike
        The value of the lensing function.
    """
    assert posx.shape == posy.shape
    assert posx.shape == amps.shape
    assert amps.shape == scales.shape

    Lens_del = 0
    for ii in range(scales.shape[0]):
        Lens_del = Lens_del + amps[ii] * np.exp(
            -0.5 * ((rx - posx[ii]) ** 2 + (ry - posy[ii]) ** 2) / (scales[ii] ** 2)
        )
    return Lens_del


def LogLens(rx: npt.ArrayLike, ry: npt.ArrayLike) -> npt.ArrayLike:
    """
    Get a radially symmetric natural log lens.

    Parameters
    ----------
    rx : ArrayLike
        ArrayLike of X positions on the lens plane.
    ry : ArrayLike
        ArrayLike of Y positions on the lens plane.

    Returns
    -------
    ArrayLike
        The value of the lensing function.
    """
    return np.log(np.sqrt((rx) ** 2 + (ry) ** 2))


def RandomPowerLawLens(
    rx_size: int,
    ry_size: int,
    dr: float,
    powerscaling: float = -11 / 3,
    amp: float = 1,
    seed: int = None,
    plot: bool = False,
) -> npt.ArrayLike:
    """
    Get a random realization for a power law lens.

    The lens array generated here is based on a power spectrum scaling.
    The power scaling relation is given by,
        P(k) = amp * k ^ powerscaling
    The random state can be seeded.

    Parameters
    ----------
    rx_size : int
        Number of grid points in the X directions.
    ry_size : int
        Number of grid points in the X directions.
    dr : float
        The spatial width of one grid cell.
    powerscaling : float, optional
        The power scaling of the power spectrum, by default -11/3
    amp : float, optional
        The amplitude of the power spectrum, by default 1
    seed : int, optional
        The seed of the random state, by default None
    plot : bool, optional
        Plot a diagnostic plot, by default False

    Returns
    -------
    ArrayLike
        The value of the lensing function.
    """
    k_1 = fftfreq(rx_size, d=dr)
    k_2 = fftfreq(ry_size, d=dr)
    k1v, k2v = np.meshgrid(k_1, k_2)

    if type(seed) is not None:
        rdmstate = np.random.RandomState(seed)
        lens_del = rdmstate.normal(loc=0, scale=1, size=(rx_size, ry_size))
    else:
        lens_del = np.random.normal(loc=0, scale=1, size=(rx_size, ry_size))

    lens_del = fftn(lens_del, axes=(-2, -1))

    Powerspec = amp * (k1v**2 + k2v**2) ** (powerscaling / 2)

    lens_del = lens_del * np.sqrt(Powerspec)
    lens_del = ifftn(lens_del, axes=(-2, -1)).real
    lens_del = lens_del - np.mean(lens_del)

    if plot:
        plt.figure()
        plt.plot(np.sqrt(k1v**2 + k2v**2).ravel(), Powerspec.ravel())
        plt.yscale("log")
        plt.xscale("log")
        plt.show()
    return lens_del


def RandomGaussianLens(
    rx_size: int, ry_size: int, sigma: float = 1, seed: int = None
) -> npt.ArrayLike:
    """
    Get a random zero-mean Gaussian realization for a lens.

    The lens array generated here is based on Gaussian statistics.
    The random state can be seeded.

    Parameters
    ----------
    rx_size : int
        Number of grid points in the X directions.
    ry_size : int
        Number of grid points in the Y directions.
    sigma : float, optional
        The standard deviation of the Gaussian probabilty distribution, by default 1
    seed : int, optional
        The seed of the random state, by default None

    Returns
    -------
    ArrayLike
        The value of the lensing function.
    """
    if type(seed) is not None:
        rdmstate = np.random.RandomState(seed)
        lens_del = rdmstate.normal(loc=0, scale=sigma, size=(rx_size, ry_size))
    else:
        lens_del = np.random.normal(loc=0, scale=sigma, size=(rx_size, ry_size))

    return lens_del


def AnalyticPointMassGrav(
    y: float,
    geom_par: float,
    lens_par: float,
) -> T.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get the observables for a point mass gravitational lens.

    Get the observables for a point mass gravitational lens using the
    analytic solution for a point mass lens. Note this is solved along
    the axis of the source and the origin of the lens plane.

    Parameters
    ----------
    y : float
        Source position on the lens plane.
    geom_par : float
        The geometric parameter of the Fermat potential.
    lens_par : float
        The lensing parameter of the Fermat potential.

    Returns
    -------
    xposvv : ndarray
        Array of the image positions on the lens plane.
    i_delvv : ndarray
        Array of the image delays.
    i_magvv : ndarray
        Array of the image magnifications.
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

    i_delvv = _lensdel(xposvv, y, geom_par, lens_par)
    i_magvv = _lensmag(xposvv)

    return xposvv, i_delvv, i_magvv


def AnalyticGaussPlasma(
    y: float,
    geom_par: float,
    lens_par: float,
    freqvals: npt.ArrayLike,
    xm: float = 10,
    N: int = 20001,
) -> T.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Get the observables for a Gaussian plasma lens.

    Get the observables for a Gaussian plasma lens using the
    analytic solutions. The image solutions do not always have an
    analytic solution so the solutions are found by numerical root
    finding.Note this is solved along the axis of the source and
    the origin of the lens plane, assuming the lens is radially
    symmetric.

    Parameters
    ----------
    y : float
        Source position on the lens plane.
    geom_par : float
        The geometric parameter of the Fermat potential.
    lens_par : float
        The lensing parameter of the Fermat potential.
    freqvals : ArrayLike
        Array of frequency values to search for images.
    xm : float, optional
        The maximum spatial grid position on the lens plane, by default 10
    N : int, optional
        The total number of points on the grid, by default 20001

    Returns
    -------
    freqvv : ndarray
        Array of the frequencies for the image.
    xposvv : ndarray
        Array of the image positions on the lens plane.
    i_delvv : ndarray
        Array of the image delays.
    i_magvv : ndarray
        Array of the image magnifications.
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
        mu_lens = lens_par / tfi**2

        mu_ = mu_lens / geom_par

        idx = np.argwhere(np.diff(np.sign(_lensgrad(xvv, mu_) - y))).flatten()

        freqvv = np.append(freqvv, freqvals[ii] * np.ones(idx.size))

        xposvv = np.append(xposvv, xvv[idx])

        i_delvv = np.append(i_delvv, _lensdel(xvv[idx], y, geom_par, mu_lens))

        i_magvv = np.append(i_magvv, _lensmag(xvv[idx], mu_))

    return freqvv, xposvv, i_delvv, i_magvv


def AnalyticRationalLens(
    y: float,
    geom_par: float,
    lens_par: float,
    freqvals: npt.ArrayLike
) -> T.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Get the observables for a rational plasma lens.

    Get the observables for a rational plasma lens using the
    analytic solutions. The image solutions do not always have an
    analytic solution so the solutions are found by numerical root
    finding. Note this is solved along the axis of the source and
    the origin of the lens plane, assuming the lens is radially
    symmetric.

    Parameters
    ----------
    y : float
        Source position on the lens plane.
    geom_par : float
        The geometric parameter of the Fermat potential.
    lens_par : float
        The lensing parameter of the Fermat potential.
    freqvals : ArrayLike
        Array of frequency values to search for images.

    Returns
    -------
    freqvv : ndarray
        Array of the frequencies for the image.
    xposvv : ndarray
        Array of the image positions on the lens plane.
    i_delvv : ndarray
        Array of the image delays.
    i_magvv : ndarray
        Array of the image magnifications.
    """
    # stationary point func
    def _lenspos(y, kappa, sigma):
        coeffs = [1 / (4 * sigma ** 4),
                  -y / (4 * sigma ** 4),
                  1 / (sigma ** 2),
                  -y / (sigma ** 2),
                  1 - kappa / sigma ** 2,
                  -y]

        images = np.roots(coeffs)
        real_imgs = np.array([])
        for im_ in images:
            if np.imag(im_) == 0:
                xci = np.real(im_)
                real_imgs = np.append(real_imgs, xci)

        return real_imgs

    # lens delay func
    def _lensdel(x, y, geom_const, lens_const, sigma):
        return (geom_const * 0.5 * (x - y) ** 2
                + lens_const * 1 / (1 + 0.5 * x ** 2 / sigma ** 2))

    # lens mag func
    def _lensmag(x, kappa, sigma):
        x_a = x * x / sigma**2
        pre_fac = (1 + 0.5 * x_a)

        eig1 = 1 + kappa / pre_fac**3 * (3 / 2 * x * x - 1) + 0j
        eig2 = 1 - kappa / pre_fac**2 + 0j

        return 1 / np.sqrt(eig1 * eig2)

    imgfreqs = []
    imgdels = []
    imgmags = []
    imgxs = []

    for f_ in freqvals:
        kap_ = lens_par / geom_par / f_ ** 2

        imgs = _lenspos(y, kap_, 1)
        for im_ in imgs:
            imgfreqs.append(f_)
            imgxs.append(im_)
            imgdels.append(_lensdel(im_, y, geom_par, lens_par / f_ ** 2, 1))
            imgmags.append(_lensmag(im_, kap_, 1))

    imgfreqs = np.array(imgfreqs)
    imgdels = np.array(imgdels)
    imgmags = np.array(imgmags)
    imgxs = np.array(imgxs)

    return imgfreqs, imgxs, imgdels, imgmags
