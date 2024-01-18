// Imports
#include <iostream>
#include <vector>
#include <complex>
#include <string>
#include <fstream>
#include <cmath>
#include <cfloat>
#include <assert.h>

// Global Constants Definition
const double pi = 3.14159265358979323846;
const std::complex<double> I (0,1.0);

// Data Types Definition
typedef struct imagepoint {
	double valx;
	double valy;
	double valf;
    double delay;
    std::complex<double> mag;
} imagepoint;

typedef struct physpoint {
	double valx = 0.0;
	double valy = 0.0;
} physpoint;

// Functions
std::complex<double> GetTransferFuncVal(
	const double theta_step,
	const int theta_N, 
	const double theta_min,		
	const double freq,
	const double freq_ref,    
	const double freq_power,       
	const std::vector<double> &lens_arr,
	const std::vector<physpoint> &dlens_arr,
	const std::vector<physpoint> &ddlens_arr,	
    const double geom_fac,    
    const double lens_fac,
	const physpoint betav,
    const bool nyqzone_aliased    
);

std::complex<double> GetTwoPlaneTransferFuncVal(
	const double theta_step,
	const int theta_N, 
	const double theta_min,		
	const double freq,
	const double freq_ref,    
	const double freq_power1,        
	const std::vector<double> &lens_arr1,
	const std::vector<physpoint> &dlens_arr1,
	const std::vector<physpoint> &ddlens_arr1,
    const double geom_fac1,    
    const double lens_fac1,
	const double freq_power2,
	const std::vector<double> &lens_arr2,
	const std::vector<physpoint> &dlens_arr2,
	const std::vector<physpoint> &ddlens_arr2,
    const double geom_fac2,    
    const double lens_fac2,    
    const double multilens_12_scale,        
	const physpoint beta1,
	const physpoint lens12_offset,
    const bool nyqzone_aliased    
);

std::complex<double> GetPlanePMGravTransferFuncVal(
	const double theta_step,
	const int theta_N, 
	const double theta_min,		
	const double freq,
	const double freq_ref,    
	const double freq_power,        
	const std::vector<double> &lens_arr,
	const std::vector<physpoint> &dlens_arr,
	const std::vector<physpoint> &ddlens_arr,
    const double geom_fac,    
    const double lens_fac,
	const double mass,
	const physpoint betaE_v,
	const physpoint betav,
    const double multilens_scale,
    const bool nyqzone_aliased    
);

std::complex<double> GetPMGravTransferFuncVal(
	const double freq,
	const double mass,
	const physpoint betav,
    const bool nyqzone_aliased    
);

void GetFreqImage(
	const double theta_step,
	const int theta_N, 
	const double theta_min,		
	const double freq,
	const double freq_power, 
	const std::vector<double> &lens_arr,
	const std::vector<physpoint> &dlens_arr,
	const std::vector<physpoint> &ddlens_arr,	
    const double geom_fac,    
    const double lens_fac,
	const physpoint betav,        
    std::vector<imagepoint> &freq_images
);

void GetMultiplaneFreqImage(
	const double theta_step,
	const int theta_N, 
	const double theta_min,		
    const double scaling_factor,                
	const double freq,
	const double freq_power1,        
	const std::vector<double> &lens_arr1,
	const std::vector<physpoint> &dlens_arr1,
	const std::vector<physpoint> &ddlens_arr1,
    const double geom_fac1,    
    const double lens_fac1,
	const physpoint beta1,    
	const double freq_power2,
	const std::vector<double> &lens_arr2,
	const std::vector<physpoint> &dlens_arr2,
	const std::vector<physpoint> &ddlens_arr2,
    const double geom_fac2,    
    const double lens_fac2,    
	const physpoint beta2,
    std::vector<imagepoint> &freq_images
); 

void GetPlaneToPMGravFreqImage(
	const double theta_step,
	const int theta_N, 
	const double theta_min,		
    const double scaling_factor,                
	const double freq,
	const double freq_power1,        
	const std::vector<double> &lens_arr1,
	const std::vector<physpoint> &dlens_arr1,
	const std::vector<physpoint> &ddlens_arr1,
    const double geom_fac1,    
    const double lens_fac1,
	const physpoint beta1,    
	const double mass,
	const physpoint betaE_v,
    std::vector<imagepoint> &freq_images
);

bool IsStationary(
	const int itheta,
	const int jtheta,
	const int theta_N,
	const double theta_step,
	const double theta_min,	
	const std::vector<physpoint> &dlens_arr,
    const double lens_param,    
	const physpoint betav 
); 

std::complex<double> GetMag(
	const int itheta,
	const int jtheta,
	const int theta_N,
	const std::vector<physpoint> &mag_arr,
	const double lens_param
);

double GetLensDelay(
    const double lens_fac,
    const double freq,
    const double freq_ref,
    const double freq_power,
    const double lens_arr_val
);

std::complex<double> GetImgVal(
    const int itheta,
    const int jtheta,
	const double theta_step,
	const int theta_N, 
	const double theta_min,		
	const double freq,
	const double freq_ref,    
	const double freq_power,        
	const std::vector<double> &lens_arr,
	const std::vector<physpoint> &ddlens_arr,
    const double geom_fac,    
    const double lens_fac,
	const physpoint betav,
    const bool nyqzone_aliased    
);

void SetGeometricDelayArr(
	const double theta_min,
	const double theta_max,
	const int theta_N,
	const double beta_x,
	const double beta_y,
	std::vector<double> &geom_arr
);

void SetFermatPotential(
	const double geom_factor,
	const double lens_factor,
    const int theta_N,
    const double freq,    
	const std::vector<double> &geom_arr,
	const std::vector<double> &lens_arr,
	std::vector<double> &fermat_pot 
);

void SetGradientArrs(
    const int theta_N,
    const double theta_step,    
	const std::vector<double> &lens_arr,
	std::vector<physpoint> &dlens_arr ,
	std::vector<physpoint> &ddlens_arr 	
);

int Sign(const double val);

bool StatCellCheck(
	const double a,
	const double b
); 

bool IVPCheck(
	const double fa,
	const double fc,
	const double fb
);

bool CellCheck(
	const double left,
	const double center,
	const double right,
	const double val
);

physpoint map_grav_p(
	const physpoint srcpos,
	const double eins
);

physpoint map_grav_m(
	const physpoint srcpos,
	const double eins
);

std::complex<double> grav_magval(
	const physpoint imgpos,
	const double eins
);

double grav_delayval(
	const physpoint imgpos,
	const physpoint srcpos,
	const double eins,
	const double mass
);

