
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.complex cimport complex

ctypedef unsigned int size_t

cdef extern from "rwlens.h":
    cdef struct imagepoint:
        int xind
        int yind
        int find

    double complex GetTransferFuncVal(\
                   double theta_step, int theta_NM,\
                   double freq, vector[double]& fermat_pot,\
                   double geom_factor) nogil

    void GetFreqImage(int theta_NM, int freqind,\
                      vector[double] &fermat_pot,\
                      vector[imagepoint] &freq_images) nogil

    double complex GetMag(int itheta, int jtheta, int theta_NM,\
                          double theta_step, vector[double]& fermat_pot,\
                          double geom_factor) nogil

    bint IsStationary(int itheta, int jtheta, int theta_NM, vector[double]& fermat_pot) nogil

    void SetGeometricDelayArr(double theta_min, double theta_max, int theta_NM, \
                             double beta_x, double beta_y, vector[double]& geom_arr) nogil

    void SetFermatPotential(double time_scale, double theta_scale, \
                            vector[double]& geom_arr, vector[double]& lens_arr,\
                            vector[double]& fermat_pot) nogil