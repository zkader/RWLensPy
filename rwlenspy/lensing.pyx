# distutils: language = c++
# distutils: sources = rwlenspy/rwlens.cpp

import numpy as np
from time import time

cimport numpy as np
cimport cython

from cython.parallel cimport parallel, prange, threadid

cimport openmp

np.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.double_t, ndim=2] GetPoints(double muMin,
                double muMax,
                int NM,
                ):
    
    cdef int sizeMu = NM*NM
    cdef np.ndarray[np.double_t, ndim=2] arr = np.zeros([sizeMu,2], dtype=np.double)
    cdef int iMu, jMu, indMu

    with nogil, parallel():
        for indMu in prange(sizeMu):
            jMu = indMu % NM
            iMu = (indMu - jMu)//NM

            arr[indMu,0] = (muMax - muMin) / (NM - 1) * iMu + muMin
            arr[indMu,1] = (muMax - muMin) / (NM - 1) * jMu + muMin

    return arr

cpdef vector[complex] RunPlasmaTransferFunc(vector[double] geom_arr,
                                       vector[double] lens_arr,
                                       double theta_min,
                                       double theta_max,
                                       int theta_N,
				       double beta_x,
				       double beta_y,
                                       double freq_min,
                                       double freq_max,
                                       int freq_N,
				       double geom_const,
				       double lens_const
                                       ):
    # T(theta) = geom_const*geom_arr(theta,beta) + lens_const*freq^-2*lens_arr(theta)
    cdef vector[complex] tfunc = vector[complex](freq_N)
    cdef vector[double] fermat_pot
    
    cdef int freq_ii
    cdef double theta_step, freq_step, freq_val
    cdef double lens_factor

    theta_step = (theta_max - theta_min) /  (theta_N - 1)
    freq_step = (freq_max - freq_min) /  (freq_N - 1)

    with nogil, parallel():
        for freq_ii in prange(freq_N):
            freq_val = freq_step * freq_ii + freq_min
            fermat_pot = vector[double](theta_N*theta_N)
            lens_factor = lens_const / (freq_val * freq_val)

            SetFermatPotential(geom_const, lens_factor, geom_arr, lens_arr, fermat_pot)
            tfunc[freq_ii] = GetTransferFuncVal(theta_step, theta_N, freq_val, fermat_pot)

    return tfunc

cpdef int GetFreqStationaryPoints(vector[double] lens_arr,
                                       double theta_min,
                                       double theta_max,
                                       int theta_N,
                                       double beta_x,
                                       double beta_y,
                                       double freq_min,
                                       double freq_max,
                                       int freq_N,
                                       double geom_const,
                                       double lens_const
                                       ):
    # T(theta) = geom_const*geom_arr(theta,beta) + lens_const*freq^-2*lens_arr(theta)
    cdef vector[complex] tfunc = vector[complex](freq_N)
    cdef vector[double] geom_arr = vector[double](theta_N*theta_N)
    cdef vector[double] fermat_pot
    cdef vector[imagepoint] temppnts
    cdef vector[vector[imagepoint]] freqpnts

    SetGeometricDelayArr(theta_min, theta_max, theta_N, beta_x, beta_y, geom_arr)

    cdef int freq_ii, thrid
    cdef double theta_step, freq_step, freq_val
    cdef double lens_factor

    theta_step = (theta_max - theta_min) /  (theta_N - 1)
    freq_step = (freq_max - freq_min) /  (freq_N - 1)

    freqpnts.resize(openmp.omp_get_max_threads())
    with nogil, parallel():
        for freq_ii in prange(freq_N):
            thrid = threadid()

            freq_val = freq_step * freq_ii + freq_min
            fermat_pot = vector[double](theta_N*theta_N)
            lens_factor = lens_const / (freq_val * freq_val)

            SetFermatPotential(geom_const, lens_factor, geom_arr, lens_arr, fermat_pot)
            GetFreqImage(theta_N, freq_ii, fermat_pot, freqpnts[thrid])

    return 0