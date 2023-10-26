
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.complex cimport complex

ctypedef unsigned int size_t

cdef extern from "rwlens.h":
    cdef struct imagepoint:
        double valx
        double valy
        double valf
        double delay
        double complex mag

    cdef struct physpoint: 
        double valx
        double valy

    double complex GetTransferFuncVal(\
    double theta_step, int theta_NM, \
    double theta_min, double freq, \
    double freq_ref, double freq_power,\
    vector[double]& lens_arr, \
    vector[physpoint]& dlens_arr, \
    vector[physpoint]& ddlens_arr, \
    double geom_fac, double lens_fac, \
    physpoint betav) nogil

    double complex GetTwoPlaneTransferFuncVal(
	double theta_step,
	int theta_NM, 
	double theta_min,		
	double freq,
	double freq_ref,    
	double freq_power1,        
	vector[double]& lens_arr1,
	vector[physpoint]& dlens_arr1,
	vector[physpoint]& ddlens_arr1,
    double geom_fac1,    
    double lens_fac1,
	double freq_power2,
	vector[double]& lens_arr2,
	vector[physpoint]& dlens_arr2,
	vector[physpoint]& ddlens_arr2,
    double geom_fac2,    
    double lens_fac2,    
    double multilens_12_scale,        
	physpoint beta1,
	physpoint lens12_offset) nogil

    double complex GetPlanePMGravTransferFuncVal(
	double theta_step,
	int theta_NM, 
	double theta_min,		
	double freq,
	double freq_ref,    
	double freq_power,        
	vector[double]& lens_arr,
	vector[physpoint]& dlens_arr,
	vector[physpoint]& ddlens_arr,
    double geom_fac,    
    double lens_fac,
	double mass,
	physpoint betaE_v,
	physpoint betav,
    double multilens_scale) nogil
    
    double complex GetPMGravTransferFuncVal(
    double freq,
    double mass,
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
    
    void GetFreqImage( double theta_step, int theta_N, double theta_min,\
                       double freq,  double freq_power,\
                       vector[double]& lens_arr, vector[physpoint]& dlens_arr,\
                       vector[physpoint]& ddlens_arr, double geom_fac, double lens_fac,\
                       physpoint betav, vector[imagepoint]& freq_images) nogil    
    
    void GetMultiplaneFreqImage(
        double theta_step,
        int theta_N,
        double theta_min,
        double scaling_factor,
        double freq,
        double freq_power1,
        vector[double]& lens_arr1,
        vector[physpoint]& dlens_arr1,
        vector[physpoint]& ddlens_arr1,
        double geom_fac1,
        double lens_fac1,
        physpoint beta1,
        double freq_power2,
        vector[double]& lens_arr2,
        vector[physpoint]& dlens_arr2,
        vector[physpoint]& ddlens_arr2,
        double geom_fac2,
        double lens_fac2,
        physpoint beta2,
        vector[imagepoint]& freq_images) nogil
    
    void GetPlaneToPMGravFreqImage(
        double theta_step,
        int theta_N,
        double theta_min,
        double scaling_factor,
        double freq,
        double freq_power1,
        vector[double]& lens_arr1,
        vector[physpoint]& dlens_arr1,
        vector[physpoint]& ddlens_arr1,
        double geom_fac1,
        double lens_fac1,
        physpoint beta1,
        double mass,
        physpoint betaE_v,
        vector[imagepoint]& freq_images) nogil
    
    double complex GetMag(int itheta, int jtheta, int theta_NM,\
                          double theta_step, vector[double]& fermat_pot,\
                          double geom_factor) nogil

    bint IsStationary(int itheta, int jtheta, int theta_NM,\
                      double theta_step, double theta_min, \
                      vector[physpoint]& dlens_arr, double lens_param,\
                      physpoint betav ) nogil
    
    double complex GetImgVal( int itheta, int jtheta, double theta_step, int theta_N,\
                             double theta_min, double freq, double freq_ref,\
                             double freq_power, vector[double]& lens_arr, \
                             vector[physpoint]& ddlens_arr, double geom_fac,\
                             double lens_fac, physpoint betav ) nogil

    void SetGeometricDelayArr(double theta_min, double theta_max, int theta_NM, \
                             double beta_x, double beta_y, vector[double]& geom_arr) nogil

    void SetFermatPotential(double geom_factor, double lens_factor, \
                            int theta_NM, double freq,\
                            vector[double]& geom_arr, vector[double]& lens_arr,\
                            vector[double]& fermat_pot) nogil

    void SetGradientArrs( int theta_NM, double theta_step,    \
                        vector[double]& lens_arr, vector[physpoint]& dlens_arr, \
                        vector[physpoint]& ddlens_arr)
