from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.complex cimport complex

ctypedef unsigned int size_t

cdef extern from * nogil:
    r"""
    #include <iostream>    
    #include <omp.h>

    static omp_lock_t cnt_lock;
    static int cnt = 0;
    static double start_time = omp_get_wtime();
    static size_t total_bytes = 0;
    static bool memcheck = false;    
    
    void reset(){
        omp_init_lock(&cnt_lock);
        cnt = 0;
        start_time = omp_get_wtime();
        total_bytes = 0;
        memcheck = false;        
    }
    
    void destroy(){
        omp_destroy_lock(&cnt_lock);
    }

    void report(int mod, int totalcnts){
        omp_set_lock(&cnt_lock);
        // start protected code:
        cnt++;
        
        //std::cout << "debug| " << progress << std::endl;
        if(cnt % mod == 0){
            double progress = (double)cnt / (double)totalcnts;    
            double check_time = omp_get_wtime() - start_time;
            std::cout << "Progress : " << progress*100.0 << " % ";
            std::cout << "| Time Elapsed : " << check_time << " s " << std::endl;
        }
        // end protected code block
        omp_unset_lock(&cnt_lock);
    }
    
    void report_withsize(int mod, int totalcnts,size_t v_size, size_t maxmem){
        omp_set_lock(&cnt_lock);
        // start protected code:
        cnt++;
        total_bytes = total_bytes + v_size;

        if(total_bytes > maxmem){memcheck = true;}
        
        if(cnt % mod == 0){
            double progress = (double)cnt / (double)totalcnts;    
            double check_time = omp_get_wtime() - start_time;
            std::cout << "Progress : " << progress*100.0 << " % ";
            std::cout << "| Time Elapsed : " << check_time << " s ";
            std::cout << "| Memory Used : "<< 100.0 * (double)total_bytes / (double) maxmem << " %" << std::endl;            
        }
        // end protected code block
        omp_unset_lock(&cnt_lock);
    }

    bool check_mem(){ return memcheck;}
    """
    void reset()
    void destroy()
    void report(int mod,int totalcnts)
    void report_withsize(int mod, int totalcnts,size_t v_size, size_t maxmem)
    bint check_mem()

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
    double theta_step, int theta_N, \
    double theta_min, double freq, \
    double freq_ref, double freq_power,\
    vector[double]& lens_arr, \
    vector[physpoint]& dlens_arr, \
    vector[physpoint]& ddlens_arr, \
    double geom_fac, double lens_fac, \
    physpoint betav, bint nyqzone_aliased) nogil

    double complex GetTwoPlaneTransferFuncVal(
	double theta_step,
	int theta_N, 
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
	physpoint lens12_offset,
    bint nyqzone_aliased) nogil

    double complex GetPlanePMGravTransferFuncVal(
	double theta_step,
	int theta_N, 
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
    double multilens_scale,
    bint nyqzone_aliased) nogil
    
    double complex GetPMGravTransferFuncVal(
    double freq,
    double mass,
    physpoint betav,
    bint nyqzone_aliased) nogil
    
    double complex GetGravTransferFuncVal(
	double theta_step,
	int theta_N, 
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
    
    double complex GetMag(int itheta, int jtheta, int theta_N,\
                          double theta_step, vector[double]& fermat_pot,\
                          double geom_factor) nogil

    bint IsStationary(int itheta, int jtheta, int theta_N,\
                      double theta_step, double theta_min, \
                      vector[physpoint]& dlens_arr, double lens_param,\
                      physpoint betav ) nogil
    
    double complex GetImgVal( int itheta, int jtheta, double theta_step, int theta_N,\
                             double theta_min, double freq, double freq_ref,\
                             double freq_power, vector[double]& lens_arr, \
                             vector[physpoint]& ddlens_arr, double geom_fac,\
                             double lens_fac, physpoint betav ) nogil

    void SetGeometricDelayArr(double theta_min, double theta_max, int theta_N, \
                             double beta_x, double beta_y, vector[double]& geom_arr) nogil

    void SetFermatPotential(double geom_factor, double lens_factor, \
                            int theta_N, double freq,\
                            vector[double]& geom_arr, vector[double]& lens_arr,\
                            vector[double]& fermat_pot) nogil

    void SetGradientArrs( int theta_N, double theta_step,    \
                        vector[double]& lens_arr, vector[physpoint]& dlens_arr, \
                        vector[physpoint]& ddlens_arr)
