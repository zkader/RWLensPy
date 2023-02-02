import os
import sys
sys.path.insert(0,"/home/zkader/coderepo/RWLensPy/")
sys.path.insert(0,"/home/zkader/coderepo/baseband-analysis/")
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm,SymLogNorm
from time import time
from scipy.fft import rfft,irfft,fft,ifft,fftfreq,fftshift

import matplotlib as mpl
from baseband_analysis.dev.lensing_sim import *

GREYMAP = mpl.cm.__dict__["Greys"]

mpl.rcParams['figure.figsize'] = [8.0, 6.0]
mpl.rcParams['figure.dpi'] = 80
mpl.rcParams['savefig.dpi'] = 100

mpl.rcParams['font.size'] = 12
mpl.rcParams['legend.fontsize'] = 'large'
mpl.rcParams['figure.titlesize'] = 'large'

mpl.rcParams['agg.path.chunksize'] = 10000

from astropy import units as u
from astropy import constants as c
from astropy import cosmology


import rwlenspy.lensing as rwl

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

def mag_grav(vec_x,vec_y,eins): 
    # lensing plane
    r = np.sqrt(vec_x**2 + vec_y**2)
    mag = np.zeros_like(r)
    mag[r==0] = 0
    mag[r==eins] = np.inf
    mag[(r!=0)*(r!=eins) ] = ( 1 - (eins/r[(r!=0)*(r!=eins) ] )**4 )**(-1)
    return mag

def grav_delay(vec_x,vec_y,eins,mass):
    # lensing plane    
    Eins_time_const = 4*c.G*c.M_sun/c.c**3    
    r = np.sqrt(vec_x**2 + vec_y**2)/eins
    return Eins_time_const*mass*(0.5*r**2 - np.log(r) )


def get_plasma_Ne(rx,ry,dr,theta_in,theta_out,C2_n=1,freq=1,D_eff=1,seed=None,plot=False):
    
    theta_fres = np.sqrt(c.c/(2*np.pi*freq* D_eff)).to(u.m/u.m)
    t_inn = theta_in/theta_fres
    t_out = theta_out/theta_fres
    
    eta1v = rx/theta_fres
    eta2v = ry/theta_fres
    dtheta = dr/theta_fres
    
    k_1 = fftfreq(eta_1.size,d=dtheta)
    k_2 = fftfreq(eta_2.size,d=dtheta)
    k1v, k2v = np.meshgrid(k_1,k_2)

    kv_max = np.amax(np.sqrt(k1v**2 + k2v**2))
    
    if type(seed) is not None:
        np.random.seed(seed)
    
    n_e = np.random.normal(loc=0,scale=1,size=eta1v.shape)
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
    

def GetArrZeroPixels(arr,darrx):    
    dfx,dfy = np.gradient(arr,darrx)
        
    #PMzeros = np.argwhere(
    #   (np.diff(np.sign(dfx),axis=0)[:,:-1]*np.diff(np.sign(dfy),axis=1)[:-1,:] !=0 )
    #   | (np.diff(np.sign(dfx),axis=0)[:,:-1]*np.diff(np.sign(dfy),axis=1)[1:,:] !=0 )
    #)
    
    cellx = PMzeros[:,0]    
    celly = PMzeros[:,1] 

    return cellx,celly


def GetArrMagPhase(x,y,cellx,celly,arr):
    #dfx,dfy = np.gradient(arr,darrx)
    #dfxx,dfyx = np.gradient(dfx,darrx)
    #dfxx,dfyx = np.gradient(dfx,darrx)
    
    #cell_cntr = np.append( (xl + xr)[:,None],(yl + yr)[:,None],axis=-1) * 0.5    

    mag_val = np.zeros(cellx.shape[0])
    delay_val = np.zeros(cellx.shape[0])    
    nx,ny = arr.shape
    darrx = np.abs(x[1] - x[0])
    
    for ii,xi in enumerate(cellx):
        yi = celly[ii]
        
        if xi - 2 == 0 or xi + 2 == nx:
            mag_val[ii] = np.nan
            delay_val[ii] = np.nan
            continue
        
        if yi - 2 == 0 or yi + 2 == ny:
            mag_val[ii] = np.nan
            delay_val[ii] = np.nan            
            continue
        
        dfxx = (arr[yi+2,xi] + arr[yi-2,xi]) / (4 * darrx**2)
        dfyy = (arr[yi,xi-2] + arr[yi,xi-2]) / (4 * darrx**2)        
        dfxy = (arr[yi+1,xi+1] - arr[yi+1,xi-1]  - arr[yi-1,xi+1]  + arr[yi-1,xi-1]) / (4 * darrx**2)
        
        mag_val[ii] = 1/(dfxx*dfyy - dfxy)        
        delay_val[ii] = arr[xi,yi]
        
    return x[celly],y[cellx],delay_val, mag_val

def GetArrTransfer(x,y,delay_val, mag_val,freq=1):
    cut = np.isfinite(delay_val) |  np.isfinite(mag_val)    
    E = (mag_val[cut]+0j)**(0.5) *np.exp(2j*np.pi*freq*delay_val[cut]) 

    return x[cut],y[cut],E
    
def GetZeroPixels(spline_func,x,y):
    fxy = spline_func(x,y,dx=0,dy=0,grid=True)
    dfx = spline_func(x,y,dx=1,dy=0,grid=True)
    dfy = spline_func(x,y,dx=0,dy=1,grid=True)

    fsgnx = np.sign(dfx)
    fsgny = np.sign(dfy)
    
    PMzeros = np.argwhere(
        (np.diff(np.sign(dfx),axis=0)[:,:-1]*np.diff(np.sign(dfy),axis=1)[:-1,:] !=0)
        | (np.diff(np.sign(dfx),axis=0)[:,:-1]*np.diff(np.sign(dfy),axis=1)[1:,:] !=0)
    )
    
    cellx = PMzeros[:,0]
    celly = PMzeros[:,1] 

    return cellx,celly

def FlowPoint(x,y,cellx,celly,spline_func,Niters=500,full=True):    
    xl = x[celly]
    xr = x[celly+1]
    yl = y[cellx]
    yr = y[cellx+1]

    cell_cntr = np.append( (xl + xr)[:,None],(yl + yr)[:,None],axis=-1) * 0.5
    #print(cell_cntr.shape)

    #print(np.unique(cell_cntr,axis=0).shape)
    
    if full:
        for i in range(Niters):
            M = spline_func(cell_cntr[:,1],cell_cntr[:,0],dx=0,dy=1,grid=False)
            N = spline_func(cell_cntr[:,1],cell_cntr[:,0],dx=1,dy=0,grid=False)   
            old_mag = np.sqrt(M**2 + N**2)

            itercut = old_mag < 1e-9

            if itercut.all():
                break

            dM_dx = f2(cell_cntr[:,1],cell_cntr[:,0],dx=0,dy=2,grid=False)
            dM_dy = f2(cell_cntr[:,1],cell_cntr[:,0],dx=1,dy=1,grid=False)
            dN_dy = f2(cell_cntr[:,1],cell_cntr[:,0],dx=2,dy=0,grid=False)

            new_cntr = cell_cntr - 0.0001 *np.append(( (M*dM_dx + N*dM_dy)/np.sqrt(M**2 + N**2) )[:,None],\
                                     ( (M*dM_dy + N*dN_dy)/np.sqrt(M**2 + N**2) )[:,None],axis=-1)

            pixelcut = (new_cntr[:,0] > xr)|(new_cntr[:,0] < xl)|(new_cntr[:,1] > yr)| (new_cntr[:,1] < yl)

            new_cntr[pixelcut] = cell_cntr[pixelcut]

            M = spline_func(new_cntr[:,1],new_cntr[:,0],dx=0,dy=1,grid=False)
            N = spline_func(new_cntr[:,1],new_cntr[:,0],dx=1,dy=0,grid=False)   
            new_mag = np.sqrt(M**2 + N**2)


            cell_cntr[new_mag < old_mag] =  new_cntr[new_mag < old_mag] 
    else:
        p0 = np.append( xl[:,None],yl[:,None],axis=-1)
        p1 = np.append( xr[:,None],yl[:,None],axis=-1)
        p2 = np.append( xl[:,None],yr[:,None],axis=-1)
        p3 = np.append( xr[:,None],yr[:,None],axis=-1)
        points = np.array((p0,p1,p2,p3)) 
        dermag = np.zeros(points.shape[0:2])       
        for i in range(points.shape[0]):
            M = spline_func(points[i][:,1],points[i][:,0],dx=0,dy=1,grid=False)
            N = spline_func(points[i][:,1],points[i][:,0],dx=1,dy=0,grid=False)   
            dermag[i,:] = np.sqrt(M**2 + N**2)
            
        argcut = np.argmin(dermag,axis=0)  
        cell_cntr = np.zeros(points.shape[1:]) 
        for ii in range(argcut.shape[0]):
            cell_cntr[ii,:] = points[argcut[ii],ii,:]
            
        cell_cntr = np.unique(cell_cntr,axis=0)
    return cell_cntr*u.m/u.m

def get_transfer_func(points,spline_func,freq=1):
    mag = spline_func(points[:,1],points[:,0],dx=2,dy=0,grid=False) \
        * spline_func(points[:,1],points[:,0],dx=0,dy=2,grid=False) \
        - spline_func(points[:,1],points[:,0],dx=1,dy=1,grid=False)**2
    
    mag = 1/mag
    
    E = (mag+0j)**(0.5) *np.exp(2j*np.pi*freq*spline_func(points[:,1],points[:,0],dx=0,dy=0,grid=False)) 

    return E

def get_mag_and_phase(points,spline_func,freq=1):
    mag = spline_func(points[:,1],points[:,0],dx=2,dy=0,grid=False) \
        * spline_func(points[:,1],points[:,0],dx=0,dy=2,grid=False) \
        - spline_func(points[:,1],points[:,0],dx=1,dy=1,grid=False)**2
    
    mag = 1/mag

    return mag,spline_func(points[:,1],points[:,0],dx=0,dy=0,grid=False)



cosmo = cosmology.Planck18

############################################################
# |              |               |              |
# |              |               |              |
# |              |               |              |
# |              |               |              |
# obs            r1             r2              src  
############################################################

# Comoving
# D_from_to
D_obs_src = cosmo.comoving_distance(1)
D_obs_r1 = cosmo.comoving_distance(1)/2
D_obs_r2 = D_obs_src - 1*u.kpc

z_obs_r1 = cosmology.z_at_value(cosmo.comoving_distance,D_obs_r1)
z_obs_r2 = cosmology.z_at_value(cosmo.comoving_distance,D_obs_r2)
z_obs_src = cosmology.z_at_value(cosmo.comoving_distance,D_obs_src)

Eins_time_const = 4*c.G*c.M_sun/c.c**3

# Ang. Diam. Distance
D_obs_r1 = cosmo.angular_diameter_distance(z_obs_r1)
D_obs_r2 = cosmo.angular_diameter_distance(z_obs_r2)
D_r1_r2 = cosmo.angular_diameter_distance_z1z2(z_obs_r1,z_obs_r2)

const_r1 = D_r1_r2 / (D_obs_r1 * D_obs_r2)

D_obs_src = cosmo.angular_diameter_distance(z_obs_src)
D_r2_src = cosmo.angular_diameter_distance_z1z2(z_obs_r2,z_obs_src)


const_r2 = D_r2_src / (D_obs_r2 * D_obs_src)

#r_inner = 1E7*u.cm#1e10*u.m
r_inner = 1e8*u.cm
r_outer = 100*u.pc #1E13*u.cm
freq0 = 400E6 * u.Hz
mass = 10 #solar mass
r_e = c.alpha**2 * c.a0 # classical electron radius
p_scale =  (1*u.AU/D_obs_r2).to(u.m/u.m)
#n_e = 0.06* u.cm**-3
DM_0 = 0.03*u.pc*u.cm**-3

theta_inner = (r_inner/D_obs_r2).to(u.m/u.m)
theta_outer = (r_outer/D_obs_r2).to(u.m/u.m)
theta_fres0 = np.sqrt(c.c/(2*np.pi*freq0) * const_r2).to(u.m/u.m)
theta_p0 = (theta_fres0 * np.sqrt(r_e* c.c * DM_0 / freq0 )).to(u.m/u.m)
print(f'p scale : {p_scale} | theta inner: {theta_inner} | theta fres: {theta_fres0} | theta p0: {theta_p0}')
print(f'p scale / theta fres: {p_scale/theta_fres0} | p scale /  theta p0: {p_scale /theta_p0}')

theta_char = theta_p0
max_fres = 1.5*theta_char
theta_min = -max_fres.value
theta_max = max_fres.value
theta_N = 1001
freq_min = 400E6
freq_max = 800E6
freq_N = 1025*100

freq_ii = 400 * u.MHz
freq_ii = freq_ii.to(u.Hz).value

beta_x = 0.0
beta_y = 0.0

geom_const = ((1/(const_r2*c.c)).to(u.s)).value
lens_const = ((r_e * c.c  /( 2 * np.pi)).to(u.cm**2/u.s) * ((1.0*u.pc/u.cm).to(u.m/u.m)).value).value # k_DM
print(geom_const)
print(lens_const)

x1 = np.arange(theta_N)* (theta_max - theta_min ) / (theta_N - 1) + theta_min
print(x1)
#lens_arr = 0.7 * np.ones((theta_N*theta_N)) #np. np.exp(-0.5*()/p_scale.value )#np.random.normal(size=(theta_N*theta_N))

lens_arr = 0.3 * np.exp(-0.5*( x1[:,None]**2 + x1[None,:]**2)/p_scale.value**2)

plt.figure()
plt.imshow( geom_const*0.5*( x1[:,None]**2 + x1[None,:]**2) + lens_const*lens_arr,aspect='auto')
plt.colorbar()
plt.savefig('ferm_full.png')

lens_arr = lens_arr.ravel()
t1 = time()
transferfunc = rwl.RunPlasmaTransferFunc( lens_arr,
                                       theta_min,
                                       theta_max,
                                       theta_N,
                                       beta_x,
                                       beta_y,
                                       freq_min,
                                       freq_max,
                                       freq_N,
                                       geom_const,
                                       lens_const)

tv = time() - t1
print('Total Time :',tv,'s',' | ',tv/60,'min',tv/3600,'hr')

transferfunc = np.asarray(transferfunc)
print(transferfunc)
impulse = fftshift(irfft(transferfunc))

snr_inj=9

sim = BasebandSim(W=dump_frames,diagnostic=False,upsample=1)
sim.FRBSignal(2.56e-6*10,snr=snr_inj*(1*2048),polratio=0.66)
sim.CreateVoltageStream(addnoise=False)
vr = sim.v_stream[0,:].copy()
sim.v_stream[0,:] = sim.v_stream[0,:] + np.random.normal(scale=0.1,size=vr.shape[0])
sim.CreateWaterfall(plot=True,sig_freqs=[400e6,801e6],apply_rfimask=False,addnoise=False)

vr = rfft(vr,axis=-1)
vr = vr*transferfunc/np.sqrt(np.mean(np.abs(transferfunc)**2))    
vr = irfft(vr,axis=-1)

sim.v_stream[0,:] = vr #+ np.random.normal(scale=0.1,size=vr.shape[0])
sim.v_stream[1,:] = vr #+ np.random.normal(scale=0.1,size=vr.shape[0])

sim.CreateWaterfall(plot=True,sig_freqs=[400e6,801e6],apply_rfimask=False,addnoise=False)

smfilt = get_smooth_matched_filter(sim.v_fall[:,:,:],10)
plot_wfall(sim.v_fall,smfilt,10,plotloc='test_full_wfall.png')

plt.figure()
plt.plot(np.linspace(freq_min,freq_max,freq_N)/1e6,np.abs(transferfunc))
plt.xlabel('Freq [MHz]')
plt.ylabel('Transfer Func [arb.]')
plt.savefig('test_rwlens_fig1.png')

plt.figure()
plt.plot(np.arange(impulse.shape[0])*1.25e-9,np.abs(impulse))
plt.xlabel('time [s]')
plt.ylabel('Impulse Response [arb.]')
plt.savefig('test_rwlens_fig2.png')


#tv = time() - t1
#print('Total Time :',tv,'s',' | ',tv/60,'min',tv/3600,'hr')            
