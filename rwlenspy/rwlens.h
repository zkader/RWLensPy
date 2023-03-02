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
} imagepoint;

// 
std::complex<double> GetTransferFuncVal(
	const double theta_step,
	const int theta_NM, 
	double freq,                    // [Hz]
	std::vector<double> &fermat_pot, // [s]
	const double geom_factor // [1/s]
);

void GetFreqImage(
	const int theta_NM,
	const int freqind,             
	std::vector<double> &fermat_pot, // [s]		
    std::vector<imagepoint> &freq_images
); 

std::complex<double> GetMag(
	const int itheta,
	const int jtheta,
	const int theta_NM,
	const double theta_step, 
	std::vector<double> &fermat_pot,
    const double geom_factor
);

bool IsStationary(
	const int itheta,
	const int jtheta,
	const int theta_NM,
	std::vector<double> &fermat_pot
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
	double geom_factor,
	double lens_factor,
	std::vector<double> &geom_arr,
	std::vector<double> &lens_arr,
	std::vector<double> &fermat_pot 
);

int Sign(double val);