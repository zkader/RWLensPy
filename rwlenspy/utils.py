import sys
sys.path.insert(0,"/home/zkader/coderepo/RWLensPy/")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm,SymLogNorm
from scipy.fft import rfft,irfft,fft,ifft,fftfreq,fftshift,rfftfreq,fftn,ifftn

from astropy import units as u
from astropy import constants as c
from astropy import cosmology

def FermatPotential(rx,ry,sx,sy,D_eff,lens_func,**funcargs):
    """

    """
    Geom_del = (D_eff/c.c ).to(u.s).value*0.5*((rx-sx)**2 + (ry-sy)**2)
    Lens_del = lens_func(rx,ry,**funcargs)
    return Geom_del + Lens_del

def DM_lens(rx,ry,freq=1,DM=1,**funcargs):
    r_e = c.alpha**2 * c.a0 # classical electron radius
    plasma_const = r_e * c.c  /( 2 * np.pi * (freq.to(1/u.s))**2)
    plasma_const = plasma_const.to(u.pc**-1*u.cm**3 * u.s)

    return plasma_const.value*DM

def plasma_phase_lens(rx,ry,freq=1,Ne_mat=None,**funcargs):
    r_e = c.alpha**2 * c.a0 # classical electron radius
    plasma_const = r_e * c.c  /( 2 * np.pi * (freq.to(1/u.s))**2)
    plasma_const = plasma_const.to(u.pc**-1*u.cm**3 * u.s)

    Ne_mat = plasma_const.value*Ne_mat

    return Ne_mat

def gaussian_plasma_lens(rx,ry,freq=1,scale=1,N_e=1,**funcargs):
    """
    freq = Hz
    N_e = pc cm^-3

    """
    r_e = c.alpha**2 * c.a0 # classical electron radius
    plasma_const = N_e*r_e * c.c  /( 2 * np.pi * (freq)**2)
    plasma_const = plasma_const.to(u.s).value

    Lens_del = plasma_const*np.exp(-0.5*((rx)**2 + (ry)**2)/(scale**2) )
    return Lens_del

def gaussiancircle_plasma_lens(rx,ry,freq=1,scale=1,N_e=1,posx=0,posy=0,**funcargs):
    """
    freq = Hz
    N_e = pc cm^-3

    """
    r_e = c.alpha**2 * c.a0 # classical electron radius
    plasma_const = N_e*r_e * c.c  /( 2 * np.pi * (freq)**2)
    plasma_const = plasma_const.to(u.s).value

    r = np.sqrt(rx**2 + ry**2)
    r0 = np.sqrt(posx**2 + posy**2)

    Lens_del = plasma_const*np.exp(-0.5*((r - r0)**2)/(scale**2) )
    return Lens_del

def multi_gaussian_plasma_lens(rx,
                         ry,
                         freq=1,
                         scale=np.array([]),
                         N_e=np.array([]),
                         posx=np.array([]),
                         posy=np.array([]),
                         **funcargs):
    """
    freq = Hz
    N_e = pc cm^-3

    """
    assert posx.shape == posy.shape
    assert posx.shape == N_e.shape
    assert N_e.shape == scale.shape

    r_e = c.alpha**2 * c.a0 # classical electron radius
    Lens_del = 0
    plasma_const =  r_e * c.c  /( 2 * np.pi * (freq)**2)

    for ii in range(scale.shape[0]):
        plasma_const_ii = (N_e[ii]*plasma_const).to(u.s).value
        Lens_del = Lens_del + plasma_const_ii*np.exp(\
                                                  -0.5*((rx-posx[ii])**2 + (ry-posy[ii])**2)\
                                                     /(scale[ii]**2) )
    return Lens_del

def gravitational_lens(rx,ry,mass=1,**funcargs):
    """
    mass = solar mass
    """
    Eins_time_const = (4*c.G*mass*c.M_sun/c.c**3).to(u.s).value
    Lens_del = -Eins_time_const*np.log(np.sqrt((rx)**2 + (ry)**2) )
    return Lens_del

def map_grav_p(vec_x,vec_y,eins):
    # map to lensing plane
    ang = np.mod( np.arctan2(vec_y, vec_x) + 2 * np.pi, 2 * np.pi)
    r = np.sqrt(vec_x**2 + vec_y**2)
    mag = 0.5 * ( r + np.sqrt(r**2 + 4 * eins**2 ) )

    return mag*np.cos(ang), mag*np.sin(ang)
    
def map_grav_m(vec_x,vec_y,eins):
    # map to lensing plane
    ang = np.mod( np.arctan2(vec_y, vec_x) + 2 * np.pi, 2 * np.pi)
    r = np.sqrt(vec_x**2 + vec_y**2)
    mag = 0.5 * ( r - np.sqrt(r**2 + 4 * eins**2 ) )

    return mag*np.cos(ang), mag*np.sin(ang)

def grav_mag(vec_x,vec_y,eins):
    # lensing plane
    r = np.sqrt(vec_x**2 + vec_y**2)
    mag = np.zeros_like(r)
    mag[r==0] = 0
    mag[r==eins] = np.inf
    mag[(r!=0)*(r!=eins) ] = ( 1 - (eins/r[(r!=0)*(r!=eins) ] )**4 )**(-1)
    return mag

def grav_delay(vec_x,vec_y,imp_x,imp_y,eins,mass):
    # lensing plane
    Eins_time_const = 4*c.G*c.M_sun/c.c**3
    r1 = np.sqrt((vec_x-imp_x)**2 + (vec_y-imp_y)**2)/eins
    r2 = np.sqrt((vec_x)**2 + (vec_y)**2)/eins
    return Eins_time_const*mass*(0.5*r1**2 - np.log(r2) )

def get_plasma_Ne(rx_size,ry_size,dr,theta_in,theta_out,C2_n=1,freq=1,D_eff=1,seed=None,plot=False):

    theta_fres = np.sqrt(c.c/(2*np.pi*freq* D_eff)).to(u.m/u.m)
    t_inn = theta_in/theta_fres
    t_out = theta_out/theta_fres

    dtheta = dr/theta_fres

    k_1 = fftfreq(rx_size,d=dtheta)
    k_2 = fftfreq(ry_size,d=dtheta)
    k1v, k2v = np.meshgrid(k_1,k_2)

    kv_max = np.amax(np.sqrt(k1v**2 + k2v**2))

    if type(seed) is not None:
        np.random.seed(seed)

    n_e = np.random.normal(loc=0,scale=1,size=(rx_size,ry_size))
    n_e = fftn(n_e,axes=(-2,-1))

    lmin = t_inn.value
    lmax = t_out.value
    P_ne = C2_n*(k1v**2 + k2v**2 + (1/lmax)**2 )**(-11/(2*3))*np.exp(-0.5*(k1v**2 + k2v**2)/(1/lmin)**2)/np.sqrt(2*np.pi*(1/lmin)**2)

    n_e = n_e*np.sqrt(P_ne)
    n_e = ifftn(n_e,axes=(-2,-1)).real
    n_e = n_e - np.mean(n_e)

    #n_e = ifftn( fftn(n_e)*np.exp(-0.5*(k1v**2 + k2v**2)/(kv_max)**2)/np.sqrt(2*np.pi*(kv_max)**2)).real

    if plot:
        plt.figure()
        plt.plot(np.sqrt(k1v**2 + k2v**2).ravel(),P_ne.ravel())
        plt.yscale('log')
        plt.xscale('log')
        plt.show()
    return n_e

def GetStatPnt(fermat_pot,i,j):
    F_cntr = fermat_pot[i,j]

    F_fx = fermat_pot[i+1,j] - F_cntr
    F_fy = fermat_pot[i,j+1] - F_cntr

    F_bx = F_cntr - fermat_pot[i - 1,j]
    F_by = F_cntr - fermat_pot[i,j - 1]

    if ((np.sign(F_fx) * np.sign(F_bx)  < 0) * (np.sign(F_fy) * np.sign(F_by) < 0))\
       | ( F_fx + F_bx + F_fy + F_by == 0):
        return True
    else:
        return False

def GetPntMag(itheta, jtheta, theta_step, fermat_pot):
        magval = 0.0+0j

        fxx = ( -fermat_pot[itheta+2,jtheta]
                        +16.0*fermat_pot[itheta+1,jtheta]
                        -30.0*fermat_pot[itheta,jtheta]
                        +16.0*fermat_pot[itheta-1,jtheta]
                        -fermat_pot[itheta-2,jtheta]
                )/ (12.0 * theta_step * theta_step)

        fyy = ( -fermat_pot[itheta,jtheta+2]
                        +16.0*fermat_pot[itheta,jtheta+1]
                        -30.0*fermat_pot[itheta,jtheta]
                        +16.0*fermat_pot[itheta,jtheta-1]
                        -fermat_pot[itheta,jtheta-2]
                )/ (12.0 * theta_step * theta_step)
        fxy = ( -fermat_pot[itheta+2,jtheta+2]
                        +fermat_pot[itheta+2,jtheta-2]
                        +fermat_pot[itheta-2,jtheta+2]
                        -fermat_pot[itheta-2,jtheta-2]
                        +16.0*fermat_pot[itheta+1, jtheta+1]
                        -16.0*fermat_pot[itheta+1, jtheta-1]
                        -16.0*fermat_pot[itheta-1,jtheta+1]
                        +16.0*fermat_pot[itheta-1,jtheta-1]
                )/ (48.0 * theta_step * theta_step)

        magval += fxx*fyy - fxy*fxy;
        if magval == 0.0+0j:
            return 0.0+0j
        else:
            magval = 1/np.sqrt(magval)
            return magval

def PntMagVal(xinds,
              yinds,
              theta_step,
              theta_N,
              geom_arr,
              lens_arr,
              geom_const,
              lens_const,
              freqvals,
              get_eigs=False
             ):
    
        def _fermat_pot(xi,yi):
            return geom_const*geom_arr[yi + theta_N*xi ] + lens_const*fvals**(-2) * lens_arr[yi + theta_N*xi]            
                
        fxx = ( -_fermat_pot(xinds+2,yinds)
                        +16.0*_fermat_pot(xinds+1,yinds)
                        -30.0*_fermat_pot(xinds,yinds)
                        +16.0*_fermat_pot(xinds-1,yinds)
                        -_fermat_pot(xinds-2,yinds)
                )/ (12.0 * theta_step * theta_step)

        fyy = ( -_fermat_pot(xinds,yinds+2)
                        +16.0*_fermat_pot(xinds,yinds+1)
                        -30.0*_fermat_pot(xinds,yinds)
                        +16.0*_fermat_pot(xinds,yinds-1)
                        -_fermat_pot(xinds,yinds-2)
                )/ (12.0 * theta_step * theta_step)
        fxy = ( -_fermat_pot(xinds+2,yinds+2)
                        +_fermat_pot(xinds+2,yinds-2)
                        +_fermat_pot(xinds-2,yinds+2)
                        -_fermat_pot(xinds-2,yinds-2)
                        +16.0*_fermat_pot(xinds+1,yinds+1)
                        -16.0*_fermat_pot(xinds+1,yinds-1)
                        -16.0*_fermat_pot(xinds-1,yinds+1)
                        +16.0*_fermat_pot(xinds-1,yinds-1)
                )/ (48.0 * theta_step * theta_step)

        if get_eigs:
            return fxx * geom_const,fyy * geom_const,fxy * geom_const
        else:
            magval = fxx*fyy - fxy*fxy + 0j
            magval[magval == 0.0+0j] = 0.0+0j
            magval[magval != 0.0+0j] = 1/np.sqrt(magval[magval != 0.0+0j])

            return magval * geom_const
