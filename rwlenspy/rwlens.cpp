#include "rwlens.h"

std::complex<double> GetTransferFuncVal(
	const double theta_step,
	const int theta_NM, 
	double freq,
	std::vector<double> &fermat_pot )
{
	std::complex<double> tfunc_val = 0.0 + I*0.0;
	
    for(int itheta = 2; itheta < theta_NM - 2; itheta++)
    {
        for(int jtheta = 2; jtheta < theta_NM - 2; jtheta++)
        {        	
			if( IsStationary(itheta, jtheta, theta_NM, fermat_pot) )
			{
				std::complex<double> mag = GetMag(itheta, jtheta, theta_NM, theta_step, fermat_pot);
				tfunc_val = tfunc_val + mag*std::exp(I*2.0*pi*freq*fermat_pot[jtheta + theta_NM * itheta] );				
			}
        }
    }		
    return tfunc_val;
} 

bool IsStationary(const int itheta, const int jtheta, const int theta_NM, std::vector<double> &fermat_pot) 
{
	double F_fx, F_fy, F_bx, F_by;
	double F_cntr = fermat_pot[jtheta + theta_NM * itheta];

	F_fx = fermat_pot[jtheta  + theta_NM * (itheta + 1)] - F_cntr;
	F_fy = fermat_pot[(jtheta + 1) + theta_NM * itheta] - F_cntr;	

	F_bx = F_cntr - fermat_pot[jtheta  + theta_NM * (itheta - 1)];		
	F_by = F_cntr - fermat_pot[(jtheta - 1) + theta_NM * itheta];

	if( (F_fx + F_bx == 0)&&(F_fy + F_by == 0) )
	{
	  return true;
	}else
	{
		if(F_fx/F_bx < 0.0)
		{
			if(F_fy/F_by < 0.0)
			{
				return true;
		}else{
			return false;
		}
	}else
	{
		return false;
	}	
	}
}

std::complex<double> GetMag(const int itheta, const int jtheta, const int theta_NM, const double theta_step, std::vector<double> &fermat_pot)
{
	std::complex<double> magval, fxx,fxy,fyy;	
	magval = 0.0+I*0.0;
	
	fxx = (fermat_pot[jtheta + theta_NM * (itheta+1)] - 2*fermat_pot[jtheta + theta_NM * itheta] + fermat_pot[jtheta + theta_NM * (itheta-1)])/(theta_step * theta_step) ;
	fyy = (fermat_pot[(jtheta+1) + theta_NM * itheta] - 2*fermat_pot[jtheta + theta_NM * itheta] + fermat_pot[(jtheta-1) + theta_NM * itheta])/(theta_step * theta_step) ;	
	fxy = (fermat_pot[(jtheta+1) + theta_NM * (itheta+1)] - fermat_pot[(jtheta+1) + theta_NM * (itheta-1)] - fermat_pot[(jtheta-1) + theta_NM * (itheta+1)] + fermat_pot[(jtheta-1) + theta_NM * (itheta-1)])/( 4.0*theta_step * theta_step) ;
		
	magval = fxx*fyy - fxy*fxy;
	if(magval == 0.0 + I*0.0 )
		{
			return 0.0+I*0.0;
		}
	else
	{
		magval = pow(magval,-0.5);
		return magval;	
	}
}

void SetGeometricDelayArr(
	const double theta_min,
	const double theta_max,
	const int theta_NM,
	const double beta_x,
	const double beta_y,
	std::vector<double> &geom_arr
)
{
	assert ( geom_arr.size() == theta_NM * theta_NM);	
	
	for(int arr_ii; arr_ii < theta_NM; arr_ii++)
	{
		const double theta_x = (theta_max - theta_min) / (theta_NM - 1) * arr_ii + theta_min;				
		for(int arr_jj; arr_jj < theta_NM; arr_jj++)
		{
			const double theta_y = (theta_max - theta_min) / (theta_NM - 1) * arr_jj + theta_min;				
			geom_arr[arr_ii] = 0.5*(pow( ( theta_x - beta_x ) ,2.0) + pow(theta_y - beta_y,2.0) );
		}
	}
}

void SetFermatPotential(
	double geom_factor,
	double lens_factor,
	std::vector<double> &geom_arr,
	std::vector<double> &lens_arr,
	std::vector<double> &fermat_pot )
{
	assert ( geom_arr.size() == lens_arr.size());
	assert ( geom_arr.size() == fermat_pot.size());
	
	for(int arr_ii; arr_ii < fermat_pot.size(); arr_ii++)
	{
		fermat_pot[arr_ii] = geom_factor*geom_arr[arr_ii] + lens_factor*lens_arr[arr_ii];
	}
} 
