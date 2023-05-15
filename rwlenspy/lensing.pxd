
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.complex cimport complex

ctypedef unsigned int size_t

cdef extern from "rwlens.h":
    cdef struct imagepoint:
        int xind
        int yind
        int find

    cdef struct physpoint: 
        double valx
        double valy

    double complex GetTransferFuncVal(\
                   double theta_step, int theta_NM,\
                   double freq, vector[double]& fermat_pot,\
                   double geom_factor) nogil

    double complex GetTransferFuncVal(\
    double theta_step, int theta_NM, \
    double theta_min, double freq, \
    vector[double]& lens_arr, \
    vector[physpoint]& dlens_arr, \
    vector[physpoint]& ddlens_arr, \
    double geom_fac, double lens_fac, \
    physpoint betav) nogil

    double complex GetGravTransferFuncVal(
	double theta_step,
	int theta_NM, 
	double theta_min,
	double freq,
	vector[double]& fermat_pot,
        double geom_factor,
	double eins,
	double mass,
        double beta_x,
        double beta_y) nogil

    void GetFreqImage(int theta_NM, int freqind,\
                      vector[double] &fermat_pot,\
                      vector[imagepoint] &freq_images) nogil

    double complex GetMag(int itheta, int jtheta, int theta_NM,\
                          double theta_step, vector[double]& fermat_pot,\
                          double geom_factor) nogil

    bint IsStationary(int itheta, int jtheta, int theta_NM, vector[double]& fermat_pot) nogil

    void SetGeometricDelayArr(double theta_min, double theta_max, int theta_NM, \
                             double beta_x, double beta_y, vector[double]& geom_arr) nogil

    void SetFermatPotential(double geom_factor, double lens_factor, \
                            int theta_NM, double freq,\
                            vector[double]& geom_arr, vector[double]& lens_arr,\
                            vector[double]& fermat_pot) nogil

    void SetGradientArrs( int theta_NM, double theta_step,    \
                        vector[double]& lens_arr, vector[physpoint]& dlens_arr, \
                        vector[physpoint]& ddlens_arr)