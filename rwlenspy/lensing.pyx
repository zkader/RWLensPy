# distutils: language = c++
# distutils: sources = rwlenspy/rwlens.cpp

import numpy as np
cimport numpy as np
cimport cython
from cython.parallel cimport parallel, prange, threadid
cimport openmp

np.import_array()

cdef double PI_ = 3.14159265358979323846


"""
===================================================================
                 Transfer Functions
===================================================================

"""
cpdef vector[complex] RunUnitlessTransferFunc(
                                       double theta_min,
                                       double theta_max,
                                       int theta_N,
                                       vector[double] freq_arr,
                                       double freq_ref,    
                                       vector[double] lens_arr,    
                                       double beta_x,
                                       double beta_y,
                                       double geom_const,
                                       double lens_const,
                                       double freq_power
                                       ):
    # T(theta) = geom_const*geom_arr(theta,beta) + lens_const*freq^-2*lens_arr(theta)
    cdef int freq_N = freq_arr.size()
    cdef vector[complex] tfunc = vector[complex](freq_N)
    cdef vector[physpoint] grad_lens_arr = vector[physpoint](theta_N*theta_N)
    cdef vector[physpoint] hess_lens_arr = vector[physpoint](theta_N*theta_N)
	
    cdef physpoint beta_vec
    
    beta_vec.valx = beta_x
    beta_vec.valy = beta_y
	
    cdef int freq_ii
    cdef double theta_step, freq_val

    theta_step = (theta_max - theta_min) /  (theta_N - 1)

    SetGradientArrs( theta_N, theta_step, lens_arr, grad_lens_arr, hess_lens_arr)
    
    cdef int freq_mod = freq_N//10

    reset() # reset counter and init lock        
    with nogil, parallel():
        for freq_ii in prange(freq_N):
            freq_val = freq_arr[freq_ii]

            tfunc[freq_ii] = GetTransferFuncVal( theta_step, theta_N, theta_min,\
                                                freq_val, freq_ref, freq_power, lens_arr, grad_lens_arr,\
                                               hess_lens_arr, geom_const, lens_const, \
                                               beta_vec)
            report(freq_mod,freq_N)
    destroy() # release lock
    
    return tfunc

cpdef vector[complex] RunPlasmaGravTransferFunc(
                                       double theta_min,
                                       double theta_max,
                                       int theta_N,
                                       vector[double] freq_arr,
                                       double freq_ref,
                                       vector[double] lens_arr, 
                                       double lens_scale,             
                                       double beta_x,
                                       double beta_y,    
                                       double geom_const,
                                       double lens_const,
                                       double freq_power,
                                       double eins,
                                       double beta_E_x,
                                       double beta_E_y,                                                    
                                       double mass
                                       ):
    # T(theta) = geom_const*geom_arr(theta,beta) + lens_const*freq^-2*lens_arr(theta)
    # mass in M_sol
    cdef int freq_N = freq_arr.size()
    cdef vector[complex] tfunc = vector[complex](freq_N)
    cdef vector[physpoint] grad_lens_arr = vector[physpoint](theta_N*theta_N)
    cdef vector[physpoint] hess_lens_arr = vector[physpoint](theta_N*theta_N)
	
    cdef physpoint beta_vec, beta_E_vec
    
    beta_vec.valx = beta_x
    beta_vec.valy = beta_y
    
    beta_E_vec.valx = beta_E_x
    beta_E_vec.valy = beta_E_y
	
    cdef int freq_ii
    cdef double theta_step, freq_val

    theta_step = (theta_max - theta_min) /  (theta_N - 1)

    SetGradientArrs( theta_N, theta_step, lens_arr, grad_lens_arr, hess_lens_arr)
    
    cdef double lens_scaling = lens_scale / eins
    
    cdef int freq_mod = freq_N//10
    
    reset() # reset counter and init lock            
    with nogil, parallel():
        for freq_ii in prange(freq_N):
            freq_val = freq_arr[freq_ii]
            
            tfunc[freq_ii] = GetPlanePMGravTransferFuncVal( theta_step, theta_N, theta_min, freq_val,\
                                                      freq_ref, freq_power, lens_arr, grad_lens_arr,\
                                                      hess_lens_arr, geom_const, lens_const, mass,\
                                                      beta_E_vec, beta_vec, lens_scaling)
            report(freq_mod,freq_N)
    destroy() # release lock            
    return tfunc

cpdef vector[complex] RunMultiplaneTransferFunc(
                                       double theta_min,
                                       double theta_max,
                                       int theta_N,    
                                       vector[double] freq_arr,
                                       double freq_ref,    
                                       vector[double] lens_arr_1,
                                       double lens_scale_1,
                                       double beta_1_x,
                                       double beta_1_y,
                                       double geom_const_1,
                                       double lens_const_1,
                                       double freq_power_1,
                                       vector[double] lens_arr_2,
                                       double lens_scale_2,
                                       double beta_2_x,
                                       double beta_2_y,
                                       double geom_const_2,
                                       double lens_const_2,
                                       double freq_power_2    
                                       ):
    # T(theta) = geom_const*geom_arr(theta,beta) + lens_const*freq^-2*lens_arr(theta)
    cdef int freq_N = freq_arr.size()
    cdef vector[complex] tfunc = vector[complex](freq_N)
    cdef vector[physpoint] grad_lens_arr_1 = vector[physpoint](theta_N*theta_N)
    cdef vector[physpoint] hess_lens_arr_1 = vector[physpoint](theta_N*theta_N)
    cdef vector[physpoint] grad_lens_arr_2 = vector[physpoint](theta_N*theta_N)
    cdef vector[physpoint] hess_lens_arr_2 = vector[physpoint](theta_N*theta_N)
	
    cdef physpoint beta_1_vec, beta_2_vec
    
    beta_1_vec.valx = beta_1_x
    beta_1_vec.valy = beta_1_y
    
    beta_2_vec.valx = beta_2_x
    beta_2_vec.valy = beta_2_y
	
    cdef int freq_ii
    cdef double theta_step, freq_val

    theta_step = (theta_max - theta_min) /  (theta_N - 1)

    SetGradientArrs( theta_N, theta_step, lens_arr_1, grad_lens_arr_1, hess_lens_arr_1)
    SetGradientArrs( theta_N, theta_step, lens_arr_2, grad_lens_arr_2, hess_lens_arr_2)
    
    cdef double lens_scaling = lens_scale_1 / lens_scale_2
    
    cdef int freq_mod = freq_N//10
    
    reset() # reset counter and init lock            
    with nogil, parallel():
        for freq_ii in prange(freq_N):
            freq_val = freq_arr[freq_ii]
            
            tfunc[freq_ii] = GetTwoPlaneTransferFuncVal( theta_step, theta_N, theta_min,\
                                freq_val, freq_ref, freq_power_1, lens_arr_1, grad_lens_arr_1,\
                                hess_lens_arr_1, geom_const_1, lens_const_1, freq_power_2,\
                                lens_arr_2, grad_lens_arr_2, hess_lens_arr_2, geom_const_2,\
                                lens_const_2, lens_scaling, beta_1_vec, beta_2_vec) 
            
            report(freq_mod,freq_N)
    destroy() # release lock            
    return tfunc

cpdef vector[complex] RunGravTransferFunc(
                                       vector[double] freq_arr,
                                       double beta_x,
                                       double beta_y,    
                                       double mass
                                       ):
    # T(theta) = geom_const*geom_arr(theta,beta) + lens_const*freq^-2*lens_arr(theta)
    # mass in M_sol
    cdef int freq_N = freq_arr.size()
    cdef vector[complex] tfunc = vector[complex](freq_N)
    cdef physpoint beta_vec
    
    beta_vec.valx = beta_x
    beta_vec.valy = beta_y
    
    cdef int freq_ii
    cdef double freq_val
    
    cdef int freq_mod = freq_N//10
    
    reset() # reset counter and init lock            
    with nogil, parallel():
        for freq_ii in prange(freq_N):
            freq_val = freq_arr[freq_ii]
            
            tfunc[freq_ii] = GetPMGravTransferFuncVal( freq_val, mass, beta_vec)

            report(freq_mod,freq_N)
    destroy() # release lock
            
    return tfunc

"""
===================================================================
                      Ray Tracing
===================================================================

"""
cpdef GetUnitlessFreqStationaryPoints(vector[double] lens_arr,
                                       double theta_min,
                                       double theta_max,
                                       int theta_N,
                                       double beta_x,
                                       double beta_y,
                                       vector[double] freq_arr,
                                       double geom_const,
                                       double lens_const,
                                       double freq_power
                                       ):
    # T(theta) = geom_const*geom_arr(theta,beta) + lens_const*freq^-2*lens_arr(theta)    
    cdef int freq_N = freq_arr.size()
    cdef vector[vector[imagepoint]] freqpnts = vector[vector[imagepoint]](freq_N)
    cdef vector[physpoint] grad_lens_arr = vector[physpoint](theta_N*theta_N)
    cdef vector[physpoint] hess_lens_arr = vector[physpoint](theta_N*theta_N)
	
    cdef physpoint beta_vec
    
    beta_vec.valx = beta_x
    beta_vec.valy = beta_y
	
    cdef int freq_ii
    cdef double theta_step, freq_step, freq_val

    theta_step = (theta_max - theta_min) /  (theta_N - 1)

    SetGradientArrs( theta_N, theta_step, lens_arr, grad_lens_arr, hess_lens_arr)
    
    cdef int freq_mod = freq_N//10
    cdef size_t vsize 
    
    reset() # reset counter,time,memory and init lock            
    with nogil, parallel():
        for freq_ii in prange(freq_N):
            freq_val = freq_arr[freq_ii]

            GetFreqImage(theta_step, theta_N, theta_min, freq_val, freq_power, lens_arr, grad_lens_arr,\
                         hess_lens_arr,	geom_const, lens_const, beta_vec, freqpnts[freq_ii])

            vsize = sizeof(imagepoint) * freqpnts[freq_ii].capacity() + sizeof(freqpnts[freq_ii])
            report_withsize(freq_mod,freq_N,vsize)
            
    destroy() # release lock
   
    cdef vector[double] thetaxs_,thetays_,freqs_
    cdef vector[complex] magarr
    cdef vector[double] delayarr
    
    thetaxs_,thetays_,freqs_,delayarr,magarr = ConvertFreqStatPnts(freqpnts)

    return thetaxs_,thetays_,freqs_,delayarr,magarr

cpdef GetMultiplaneFreqStationaryPoints( double theta_min,
                                       double theta_max,
                                       int theta_N,    
                                       vector[double] freq_arr,
                                       double freq_ref,    
                                       vector[double] lens_arr_1,
                                       double lens_scale_1,
                                       double beta_1_x,
                                       double beta_1_y,
                                       double geom_const_1,
                                       double lens_const_1,
                                       double freq_power_1,
                                       vector[double] lens_arr_2,
                                       double lens_scale_2,
                                       double beta_2_x,
                                       double beta_2_y,
                                       double geom_const_2,
                                       double lens_const_2,
                                       double freq_power_2 
                                     ):
    # T(theta) = geom_const*geom_arr(theta,beta) + lens_const*freq^-2*lens_arr(theta)    
    cdef int freq_N = freq_arr.size()
    cdef vector[complex] tfunc = vector[complex](freq_N)
    cdef vector[physpoint] grad_lens_arr_1 = vector[physpoint](theta_N*theta_N)
    cdef vector[physpoint] hess_lens_arr_1 = vector[physpoint](theta_N*theta_N)
    cdef vector[physpoint] grad_lens_arr_2 = vector[physpoint](theta_N*theta_N)
    cdef vector[physpoint] hess_lens_arr_2 = vector[physpoint](theta_N*theta_N)
    cdef vector[vector[imagepoint]] freqpnts = vector[vector[imagepoint]](freq_N)
	
    cdef physpoint beta_1_vec, beta_2_vec
    
    beta_1_vec.valx = beta_1_x
    beta_1_vec.valy = beta_1_y
    
    beta_2_vec.valx = beta_2_x
    beta_2_vec.valy = beta_2_y
	
    cdef int freq_ii
    cdef double theta_step, freq_val

    theta_step = (theta_max - theta_min) /  (theta_N - 1)
    SetGradientArrs( theta_N, theta_step, lens_arr_1, grad_lens_arr_1, hess_lens_arr_1)

    cdef double lens_scaling = lens_scale_1 / lens_scale_2
    
    SetGradientArrs( theta_N, theta_step, lens_arr_2, grad_lens_arr_2, hess_lens_arr_2)
        
    cdef int freq_mod = freq_N//10
    cdef size_t vsize 
    
    reset() # reset counter and init lock                    
    with nogil, parallel():
        for freq_ii in prange(freq_N):
            freq_val = freq_arr[freq_ii]

            GetMultiplaneFreqImage(theta_step, theta_N, theta_min, lens_scaling, freq_val, freq_power_1,\
                                   lens_arr_1, grad_lens_arr_1, hess_lens_arr_1, geom_const_1, lens_const_1,\
                                   beta_1_vec, freq_power_2, lens_arr_2, grad_lens_arr_2, hess_lens_arr_2,\
                                   geom_const_2, lens_const_2, beta_2_vec, freqpnts[freq_ii])  
            
            vsize = sizeof(imagepoint) * freqpnts[freq_ii].capacity() + sizeof(freqpnts[freq_ii])  
            report_withsize(freq_mod,freq_N,vsize)
    destroy() # release lock
               
    cdef vector[double] thetaxs_,thetays_,freqs_
    cdef vector[complex] magarr
    cdef vector[double] delayarr
    
    thetaxs_,thetays_,freqs_,delayarr,magarr = ConvertFreqStatPnts(freqpnts)

    return thetaxs_,thetays_,freqs_,delayarr,magarr


cpdef GetPlaneToPMGravFreqStationaryPoints( double theta_min,
                                       double theta_max,
                                       int theta_N,
                                       vector[double] freq_arr,
                                       double freq_ref,
                                       vector[double] lens_arr_1,
                                       double lens_scale_1,
                                       double beta_1_x,
                                       double beta_1_y,
                                       double geom_const_1,
                                       double lens_const_1,
                                       double freq_power_1,
                                       double mass,
                                       double lens_scale_2,
                                       double beta_2_x,
                                       double beta_2_y
                                     ):
    # T(theta) = geom_const*geom_arr(theta,beta) + lens_const*freq^-2*lens_arr(theta)    
    cdef int freq_N = freq_arr.size()
    cdef vector[physpoint] grad_lens_arr_1 = vector[physpoint](theta_N*theta_N)
    cdef vector[physpoint] hess_lens_arr_1 = vector[physpoint](theta_N*theta_N)
    cdef vector[vector[imagepoint]] freqpnts = vector[vector[imagepoint]](freq_N)
	
    cdef physpoint beta_1_vec, beta_2_vec
    
    beta_1_vec.valx = beta_1_x
    beta_1_vec.valy = beta_1_y
    
    beta_2_vec.valx = beta_2_x
    beta_2_vec.valy = beta_2_y
	
    cdef int freq_ii
    cdef double theta_step, freq_val
    cdef double lens_scaling = lens_scale_1 / lens_scale_2

    theta_step = (theta_max - theta_min) /  (theta_N - 1)
    SetGradientArrs( theta_N, theta_step, lens_arr_1, grad_lens_arr_1, hess_lens_arr_1)
            
    cdef int freq_mod = freq_N//10
    cdef size_t vsize     
    
    reset() # reset counter and init lock            
        
    with nogil, parallel():
        for freq_ii in prange(freq_N):
            freq_val = freq_arr[freq_ii]

            GetPlaneToPMGravFreqImage(theta_step,\
                                      theta_N,\
                                      theta_min,\
                                      lens_scaling,\
                                      freq_val,\
                                      freq_power_1,\
                                      lens_arr_1,\
                                      grad_lens_arr_1,\
                                      hess_lens_arr_1,\
                                      geom_const_1,\
                                      lens_const_1,\
                                      beta_1_vec,\
                                      mass,\
                                      beta_2_vec,\
                                      freqpnts[freq_ii])

            vsize = sizeof(imagepoint) * freqpnts[freq_ii].capacity() + sizeof(freqpnts[freq_ii])
            report_withsize(freq_mod,freq_N,vsize)
    destroy() # release lock
   
    cdef vector[double] thetaxs_,thetays_,freqs_
    cdef vector[complex] magarr
    cdef vector[double] delayarr
    
    thetaxs_,thetays_,freqs_,delayarr,magarr = ConvertFreqStatPnts(freqpnts)

    return thetaxs_,thetays_,freqs_,delayarr,magarr


"""
===================================================================
                   Utility Functions
===================================================================

"""
cpdef GetLensGradArrs(vector[double] lens_arr,
                                       double theta_min,
                                       double theta_max,
                                       int theta_N):
    """
    Get the gradient and trace and determinant of the hessian of the lens array.
    
    This is a wrapper function for the C++ functions. 
    
    """  
    # T(theta) = geom_const*geom_arr(theta,beta) + lens_const*freq^-2*lens_arr(theta)
    cdef vector[physpoint] grad_lens_arr = vector[physpoint](theta_N*theta_N)
    cdef vector[physpoint] hess_lens_arr = vector[physpoint](theta_N*theta_N)		
    cdef double theta_step = (theta_max - theta_min) /  (theta_N - 1)

    SetGradientArrs( theta_N, theta_step, lens_arr, grad_lens_arr, hess_lens_arr)

    cdef vector[double] gradx_lens, grady_lens 
    cdef vector[double] TrHess_lens, DetHess_lens 
    
    gradx_lens, grady_lens = ConvertPhyspointVec(grad_lens_arr)
    TrHess_lens, DetHess_lens = ConvertPhyspointVec(hess_lens_arr)

    return gradx_lens, grady_lens, TrHess_lens, DetHess_lens

# Data Conversion to python
cpdef ConvertFreqStatPnts(vector[vector[imagepoint]] freqpnts):
    "Convert the C++ vector of the 5 image parameters into 5 python compatible arrays."
    cdef int iteri, iterj
    cdef int Npnts = freqpnts.size()
    cdef int Nimages 
    cdef vector[imagepoint] temppnts
    cdef imagepoint tempimgpnt
    cdef vector[double] xvals,yvals,fvals
    cdef vector[complex] magv
    cdef vector[double] delayv

    for iteri in range(Npnts):
        temppnts = freqpnts[iteri]
        Nimages = temppnts.size()
        for iterj in range(Nimages):
            tempimgpnt = temppnts[iterj]
            xvals.push_back(tempimgpnt.valx)
            yvals.push_back(tempimgpnt.valy)
            fvals.push_back(tempimgpnt.valf)
            delayv.push_back(tempimgpnt.delay)
            magv.push_back(tempimgpnt.mag)
            
    return xvals,yvals,fvals,delayv,magv


cpdef ConvertPhyspointVec(vector[physpoint] vec_):
    "Convert the C++ vector into 2 python compatible arrays."
    cdef int iteri
    cdef int Npnts = vec_.size()
    cdef physpoint temppnt
    
    cdef vector[double] xvals
    cdef vector[double] yvals

    for iteri in range(Npnts):
        temppnt = vec_[iteri]
        xvals.push_back(temppnt.valx)
        yvals.push_back(temppnt.valy)

    return xvals,yvals

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.double_t, ndim=2] GetPoints(double theta_min,
                double theta_max,
                int theta_N,
                ):
    """Get the coordinate grid point values of the array.

    Get the X and Y values of a square NxN grid. The points 
    are returned in an array of shape (N*N,2), where 0 and 1
    index X and Y, respectively. 

    Args:
        theta_min (double): The minimum X and Y value.
        theta_max (double): The maximum X and Y value.
        theta_N (int): The number of point along one axis.

    Return:
        np.ndarray: An array containing grid values of size (N*N)
    """    
    cdef int sizeNN = theta_N*theta_N
    cdef np.ndarray[np.double_t, ndim=2] arr = np.zeros([sizeNN,2], dtype=np.double)
    cdef int ii, jj, thetaind

    with nogil, parallel():
        for thetaind in prange(sizeNN):
            jj = thetaind % theta_N
            ii = (thetaind - jj)//theta_N

            arr[thetaind,0] = (theta_max - theta_min) / (theta_N - 1) * ii + theta_min
            arr[thetaind,1] = (theta_max - theta_min) / (theta_N - 1) * jj + theta_min

    return arr
