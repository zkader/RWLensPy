#include "rwlens.h"

std::complex<double> GetTransferFuncVal(
	const double theta_step,
	const int theta_NM, 
	double freq,
	std::vector<double> &fermat_pot,
    const double geom_factor)
{
	std::complex<double> tfunc_val = 0.0 + I*0.0;
	
    for(int itheta = 2; itheta < theta_NM - 2; itheta++)
    {
        for(int jtheta = 2; jtheta < theta_NM - 2; jtheta++)
        {
			if( IsStationary(itheta, jtheta, theta_NM, fermat_pot) )
			{
				std::complex<double> mag = GetMag(itheta, jtheta, theta_NM, theta_step, fermat_pot, geom_factor);
				//std::cout << "Stationary i,j: " << itheta << "," << jtheta << "| freq: " << freq <<"\n" ;
				//std::cout << "Stationary mag: " << mag << "| freq: " << freq <<"\n" ;
				//std::cout << "Stationary delay: " << fermat_pot[jtheta + theta_NM * itheta] << "| freq: " << freq <<"\n"  ;				
				double phase = 2 * pi * freq * fermat_pot[jtheta + theta_NM * itheta] ;
				tfunc_val = tfunc_val + mag*std::exp(I*phase);
				//std::cout << "Stationary tfunc: " << tfunc_val << "| freq: " << freq  <<"\n" ;								
			}
        }
    }		
    return tfunc_val;
} 

std::complex<double> GetGravTransferFuncVal(
	const double theta_step,
	const int theta_NM, 
	const double theta_min,
	double freq,
	std::vector<double> &fermat_pot,
    const double geom_factor,
	const double eins,
	const double mass,
	const double beta_x,
	const double beta_y	
)
{
	std::complex<double> tfunc_val = 0.0 + I*0.0;
	
    for(int itheta = 2; itheta < theta_NM - 2; itheta++)
    {
        for(int jtheta = 2; jtheta < theta_NM - 2; jtheta++)
        {
			if( IsStationary(itheta, jtheta, theta_NM, fermat_pot) )
			{
				double imgphase,phase;
				physpoint sourcepos,imagepos;
				std::complex<double> mag, imgmag;
				
				mag = GetMag(itheta, jtheta, theta_NM, theta_step, fermat_pot, geom_factor);				
				phase = 2 * pi * freq * fermat_pot[jtheta + theta_NM * itheta];
				
				sourcepos.thetax = theta_min + theta_step * itheta + beta_x;				
				sourcepos.thetay = theta_min + theta_step * jtheta + beta_y;

				imagepos = map_grav_p(sourcepos, eins);
				imgmag = grav_magval(imagepos, eins);
				imgphase = grav_delayval(imagepos, sourcepos, eins, mass);
				imgphase = 2 * pi * freq * imgphase;
				tfunc_val = tfunc_val + imgmag*mag*std::exp(I*(phase+imgphase));
				
				imagepos = map_grav_m(sourcepos, eins);
				imgmag = grav_magval(imagepos, eins);
				imgphase = grav_delayval(imagepos, sourcepos, eins, mass);								
				imgphase = 2 * pi * freq * imgphase;				
				tfunc_val = tfunc_val + imgmag*mag*std::exp(I*(phase+imgphase));
			}
        }
    }		
    return tfunc_val;
} 

void GetFreqImage(
	const int theta_NM,
	const int freqind,
	std::vector<double> &fermat_pot,
    std::vector<imagepoint> &freq_images	
	)
{		
    for(int itheta = 2; itheta < theta_NM - 2; itheta++)
    {
        for(int jtheta = 2; jtheta < theta_NM - 2; jtheta++)
        {        	
			if( IsStationary(itheta, jtheta, theta_NM, fermat_pot) )
			{
				imagepoint stationary_image;
				stationary_image.xind = itheta;
				stationary_image.yind = jtheta;
				stationary_image.find = freqind;
				freq_images.push_back(stationary_image);
			}
        }
    }
} 

bool IsStationary(const int itheta, const int jtheta, const int theta_NM, std::vector<double> &fermat_pot) 
{
	double F_fx, F_fy, F_bx, F_by;
	double F_cntr = fermat_pot[jtheta + theta_NM * itheta];

	F_fx = fermat_pot[jtheta  + theta_NM * (itheta + 1)] - F_cntr;
	F_fy = fermat_pot[(jtheta + 1) + theta_NM * itheta] - F_cntr;	

	F_bx = F_cntr - fermat_pot[jtheta  + theta_NM * (itheta - 1)];		
	F_by = F_cntr - fermat_pot[(jtheta - 1) + theta_NM * itheta];
    
	if( ((Sign(F_fx) *  Sign(F_bx)  < 0) && (Sign(F_fy) * Sign(F_by) < 0)) || ( F_fx + F_bx + F_fy + F_by == 0) ) 
	{
		//std::cout << "c Stationary 2: " << itheta << "," << jtheta <<"\n" ;
		//std::cout << "c Stationary fx,fy,bx,by: " << F_fx  << "," << F_fy  << "," << F_fx  << "," << F_fy  << "," <<"\n" ;
		//std::cout << "c Stationary fxs,fys : " << Fx_sign  << "," << Fy_sign <<"\n" ;						
		return true;
	}else
	{
		return false;
	}	

}

std::complex<double> GetMag(const int itheta, const int jtheta, const int theta_NM, const double theta_step, std::vector<double> &fermat_pot, const double geom_factor)
{
	std::complex<double> magval, fxx,fxy,fyy;	
	magval = 0.0+I*0.0;
	
	fxx = ( -fermat_pot[(jtheta) + theta_NM * (itheta+2)]
			+16.0*fermat_pot[(jtheta) + theta_NM * (itheta+1)]
			-30.0*fermat_pot[(jtheta) + theta_NM * (itheta)]
			+16.0*fermat_pot[(jtheta) + theta_NM * (itheta-1)]
			-fermat_pot[(jtheta) + theta_NM * (itheta-2)]	
		)/ (12.0 * theta_step * theta_step);

	fyy = ( -fermat_pot[(jtheta+2) + theta_NM * (itheta)]
			+16.0*fermat_pot[(jtheta+1) + theta_NM * (itheta)]
			-30.0*fermat_pot[(jtheta) + theta_NM * (itheta)]
			+16.0*fermat_pot[(jtheta-1) + theta_NM * (itheta)]
			-fermat_pot[(jtheta-2) + theta_NM * (itheta)]	
		)/ (12.0 * theta_step * theta_step);

	fxy = ( -fermat_pot[(jtheta+2) + theta_NM * (itheta+2)]
			+fermat_pot[(jtheta-2) + theta_NM * (itheta+2)]
			+fermat_pot[(jtheta+2) + theta_NM * (itheta-2)]
			-fermat_pot[(jtheta-2) + theta_NM * (itheta-2)]	
			+16.0*fermat_pot[(jtheta+1) + theta_NM * (itheta+1)]
			-16.0*fermat_pot[(jtheta-1) + theta_NM * (itheta+1)]
			-16.0*fermat_pot[(jtheta+1) + theta_NM * (itheta-1)]
			+16.0*fermat_pot[(jtheta-1) + theta_NM * (itheta-1)]
		)/ (48.0 * theta_step * theta_step);
		
	//fxx = (fermat_pot[jtheta + theta_NM * (itheta+1)] - 2*fermat_pot[jtheta + theta_NM * itheta] + fermat_pot[jtheta + theta_NM * (itheta-1)])/(theta_step * theta_step) ;
	//fyy = (fermat_pot[(jtheta+1) + theta_NM * itheta] - 2*fermat_pot[jtheta + theta_NM * itheta] + fermat_pot[(jtheta-1) + theta_NM * itheta])/(theta_step * theta_step) ;	
	//fxy = (fermat_pot[(jtheta+1) + theta_NM * (itheta+1)] - fermat_pot[(jtheta+1) + theta_NM * (itheta-1)] - fermat_pot[(jtheta-1) + theta_NM * (itheta+1)] + fermat_pot[(jtheta-1) + theta_NM * (itheta-1)])/( 4.0*theta_step * theta_step) ;
		
	magval = fxx*fyy - fxy*fxy;	
	if(magval == 0.0 + I*0.0 )
		{
			return 0.0+I*0.0;
		}
	else
	{
		magval = pow(magval,-0.5);
		magval *= geom_factor; // normalization to unlensed case 
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
		double theta_x = (theta_max - theta_min) / (theta_NM - 1) * arr_ii + theta_min;				
		for(int arr_jj; arr_jj < theta_NM; arr_jj++)
		{
			double theta_y = (theta_max - theta_min) / (theta_NM - 1) * arr_jj + theta_min;				
			geom_arr[arr_ii] = 0.5*( ( theta_x - beta_x ) * (theta_x - beta_x)  + (theta_y - beta_y) * (theta_y - beta_y) );
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

int Sign(double val){
	return (val > 0) - (val < 0);
}

physpoint map_grav_p(const physpoint srcpos, const double eins){
    double ang = fmod( atan2(srcpos.thetay, srcpos.thetax) + 2 * pi, 2 * pi);
    double norm = sqrt( srcpos.thetax*srcpos.thetax + srcpos.thetay * srcpos.thetay);
    double mag = 0.5 * ( norm + sqrt(norm*norm + 4 * eins * eins ) );    
	physpoint imgpos;
	imgpos.thetax = mag*cos(ang);
	imgpos.thetay = mag*sin(ang);

    return imgpos;
}

physpoint map_grav_m(const physpoint srcpos, const double eins){
    double ang = fmod( atan2(srcpos.thetay, srcpos.thetax) + 2 * pi, 2 * pi);
    double norm = sqrt( srcpos.thetax*srcpos.thetax + srcpos.thetay * srcpos.thetay);
    double mag = 0.5 * ( norm - sqrt(norm*norm + 4 * eins * eins ) );    
	physpoint imgpos;
	imgpos.thetax = mag*cos(ang);
	imgpos.thetay = mag*sin(ang);

    return imgpos;
}

std::complex<double> grav_magval(const physpoint imgpos, const double eins){
	double norm = sqrt( imgpos.thetax*imgpos.thetax + imgpos.thetay * imgpos.thetay);

    if(norm == 0)
    {
    	return 0.0;
    } else if(norm == eins)
    {
        return -1.0;
    } else
    {
        return pow( 1.0+0.0*I - pow((eins+0.0*I)/norm,4),-0.5);
    }
}

double grav_delayval(const physpoint imgpos,const physpoint srcpos,const double eins,const double mass){
    double Eins_time_const =  1.97E-5;//4*G*M_sun/c^3 in s/M_sun
    
    double rnorm = sqrt( (imgpos.thetax - srcpos.thetax)*(imgpos.thetax - srcpos.thetax) \
	                   + (imgpos.thetay - srcpos.thetay)*(imgpos.thetay - srcpos.thetay) )/eins;
	                   
    double inorm = sqrt( imgpos.thetax*imgpos.thetax \
	                   + imgpos.thetay*imgpos.thetay )/eins;

    return Eins_time_const*mass*(0.5*rnorm*rnorm - log(inorm) );
}