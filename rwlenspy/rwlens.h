#include <iostream>
#include <vector>
#include <complex>
#include <string>
#include <fstream>
#include <cmath>
#include <cfloat>
#include <assert.h>

const double pi = 3.14159265358979323846;

const std::complex<double> I (0,1.0);

typedef struct imagepoint {
	int xind;
	int yind;
	int find;
    double delay;
    std::complex<double> mag;
} imagepoint;

typedef struct physpoint {
	double valx = 0.0;
	double valy = 0.0;
} physpoint;

// 
std::complex<double> GetTransferFuncVal(
	const double theta_step,
	const int theta_NM, 
	const double freq,                    // [Hz]
	const std::vector<double> &fermat_pot, // [s]
	const double geom_factor // [1/s]
);

std::complex<double> GetTransferFuncVal(
	const double theta_step,
	const int theta_NM, 
	const double theta_min,		
	const double freq,
	const std::vector<double> &lens_arr,
	const std::vector<physpoint> &dlens_arr,
	const std::vector<physpoint> &ddlens_arr,	
    const double geom_fac,    
    const double lens_fac,
	const physpoint betav
);

std::complex<double> GetGravTransferFuncVal(
	const double theta_step,
	const int theta_NM, 
	const double theta_min,
	const double freq,
	const std::vector<double> &fermat_pot,
    const double geom_factor,
	const double eins,
	const double mass,
	const double beta_x,
	const double beta_y
);

void GetFreqImage(
	const double theta_step,        
	const int theta_NM,
	const int freqind,
	const std::vector<double> &fermat_pot, // [s]
    std::vector<imagepoint> &freq_images,
    const double geom_factor
); 

bool IsStationary(
	const int itheta,
	const int jtheta,
	const int theta_NM,
	const std::vector<double> &fermat_pot
); 

bool IsStationary(
	const int itheta,
	const int jtheta,
	const int theta_NM,
	const double theta_step,
	const double theta_min,	
	const std::vector<physpoint> &dlens_arr,
    const double lens_param,    
	const physpoint betav 
); 

std::complex<double> GetMag(
	const int itheta,
	const int jtheta,
	const int theta_NM,
	const double theta_step, 
	const std::vector<double> &fermat_pot,
    const double geom_factor
);

std::complex<double> GetMag(
	const int itheta,
	const int jtheta,
	const int theta_NM,
	const std::vector<physpoint> &mag_arr,
	const double lens_param
);

void SetGeometricDelayArr(
	const double theta_min,
	const double theta_max,
	const int theta_NM,
	const double beta_x,
	const double beta_y,
	std::vector<double> &geom_arr
);

void SetFermatPotential(
	const double geom_factor,
	const double lens_factor,
    const int theta_NM,
    const double freq,    
	const std::vector<double> &geom_arr,
	const std::vector<double> &lens_arr,
	std::vector<double> &fermat_pot 
);

void SetGradientArrs(
    const int theta_NM,
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

