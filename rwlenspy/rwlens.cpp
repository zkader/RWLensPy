#include "rwlens.h"

std::complex<double> GetTransferFuncVal(
	const double theta_step,
	const int theta_NM, 
	const double freq,
	const std::vector<double> &fermat_pot,
    const double mag_norm)
{
	std::complex<double> tfunc_val = 0.0 + I*0.0;    
    for(int itheta = 2; itheta < theta_NM - 2; itheta++)
    {
        for(int jtheta = 2; jtheta < theta_NM - 2; jtheta++)
        {
			if( IsStationary(itheta, jtheta, theta_NM, fermat_pot) )
			{
                //bool b_up = IsStationary(itheta + 1, jtheta, theta_NM, fermat_pot) ;
                //bool b_down = IsStationary(itheta - 1, jtheta, theta_NM, fermat_pot) ;
                //bool b_left = IsStationary(itheta, jtheta-1, theta_NM, fermat_pot) ;
                //bool b_right = IsStationary(itheta, jtheta+1, theta_NM, fermat_pot) ;
                
                //if 
				double phase = 2 * pi * freq * fermat_pot[jtheta + theta_NM * itheta] ;
                
				std::complex<double> mag = GetMag(itheta, jtheta, theta_NM, theta_step, fermat_pot, mag_norm);
				//std::cout << "Stationary i,j: " << itheta << "," << jtheta << "| freq: " << freq <<"\n" ;
				//std::cout << "Stationary mag: " << mag << "| freq: " << freq <<"\n" ;
				//std::cout << "Stationary delay: " << fermat_pot[jtheta + theta_NM * itheta] << "| freq: " << freq <<"\n"  ;				
				//std::cout << "Stationary delay: " << fermat_pot[jtheta + theta_NM * itheta] << "| freq: " << freq <<"\n"  ;				                
                //phase = fmod( phase, 2 * pi);
				//std::cout << "Stationary delay: " << fermat_pot[jtheta + theta_NM * itheta] << "| freq: " << freq <<"\n" 
                std::complex<double> tempval = mag*std::complex<double>( cos(phase), sin(phase));       
				tfunc_val += tempval;
				//std::cout << "Stationary tfunc: " << phase << "| freq: " << freq  <<"\n" ;								
			}
        }
    }		
    return tfunc_val;
} 

std::complex<double> GetGravTransferFuncVal(
	const double theta_step,
	const int theta_NM, 
	const double theta_min,
	const double freq,
	const std::vector<double> &fermat_pot,
    const double mag_norm,
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
				
				mag = GetMag(itheta, jtheta, theta_NM, theta_step, fermat_pot, mag_norm);				
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
	const double theta_step,        
	const int theta_NM,
	const int freqind,
	const std::vector<double> &fermat_pot,
    std::vector<imagepoint> &freq_images,
    const double mag_norm    
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
                stationary_image.delay = fermat_pot[jtheta + theta_NM * itheta] ;                                
                stationary_image.mag = GetMag(itheta, jtheta, theta_NM, theta_step, fermat_pot, mag_norm);	
                
				freq_images.push_back(stationary_image);
			}
        }
    }
}   

bool IsStationary(const int itheta, const int jtheta, const int theta_NM, const std::vector<double> &fermat_pot) 
{
	double F_cntr = fermat_pot[jtheta + theta_NM * itheta];
    if(std::isinf(F_cntr)){return false;}
    
    // central difference
	double F_cntr_dy = fermat_pot[jtheta  + theta_NM * (itheta + 1)] - fermat_pot[jtheta  + theta_NM * (itheta - 1)];
	double F_cntr_dx = fermat_pot[(jtheta + 1) + theta_NM * itheta] - fermat_pot[(jtheta - 1) + theta_NM * itheta];

    if((F_cntr_dy == 0) && (F_cntr_dx == 0)){return true;}
    
    // y difference
	double F_up_dy = fermat_pot[jtheta  + theta_NM * (itheta + 2)] - F_cntr;
	double F_down_dy = F_cntr - fermat_pot[jtheta  + theta_NM * (itheta - 2)];
        
    if(StatCellCheck(F_cntr_dy,F_down_dy) || StatCellCheck(F_cntr_dy,F_up_dy) )
    {
        // x difference 
        double F_right_dx = fermat_pot[(jtheta + 2) + theta_NM * itheta] - F_cntr;
        double F_left_dx = F_cntr - fermat_pot[(jtheta - 2) + theta_NM * itheta];            
        
        if(StatCellCheck(F_cntr_dx,F_right_dx) || StatCellCheck(F_cntr_dx,F_left_dx) )
        {
            return true;         
         }else
        {
            return false;
        }        
    }else
    {
        return false;
    }

    
    //if((Sign(F_right_dx - F_cntr_dx) !=  Sign(F_left_dx - F_cntr_dx)) && (Sign(F_fy) != Sign(F_by) ))

	//F_fx = fermat_pot[jtheta  + theta_NM * (itheta + 1)] - F_cntr;
	//F_fy = fermat_pot[(jtheta + 1) + theta_NM * itheta] - F_cntr;	

	//F_bx = F_cntr - fermat_pot[jtheta  + theta_NM * (itheta - 1)];		
	//F_by = F_cntr - fermat_pot[(jtheta - 1) + theta_NM * itheta];
    
	//if( ((Sign(F_fx) *  Sign(F_bx)  < 0) && (Sign(F_fy) * Sign(F_by) < 0)) || ( F_fx + F_bx + F_fy + F_by == 0) )
    //if((Sign(F_fx) !=  Sign(F_bx)) && (Sign(F_fy) != Sign(F_by) ))
	//{
		//std::cout << "c Stationary 2: " << itheta << "," << jtheta <<"\n" ;
		//std::cout << "c Stationary fx,fy,bx,by: " << F_fx  << "," << F_fy  << "," << F_fx  << "," << F_fy  << "," <<"\n" ;
		//std::cout << "c Stationary fxs,fys : " << Fx_sign  << "," << Fy_sign <<"\n" ;						
	//	return true;
	//}else
	//{
	//	return false;
	//}	

}

std::complex<double> GetMag(const int itheta, const int jtheta, const int theta_NM, const double theta_step, const std::vector<double> &fermat_pot, const double geom_factor)
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
            int arr_ind = arr_jj + arr_ii * theta_NM;
			geom_arr[arr_ind] = 0.5*( ( theta_x - beta_x ) * (theta_x - beta_x)  + (theta_y - beta_y) * (theta_y - beta_y) );
		}
	}
}

void SetFermatPotential(
	const double geom_factor,
	const double lens_factor,
    const int theta_NM,
    const double freq,
	const std::vector<double> &geom_arr,
	const std::vector<double> &lens_arr,
	std::vector<double> &fermat_pot )
{
	assert ( geom_arr.size() == theta_NM*theta_NM);    
	assert ( geom_arr.size() == lens_arr.size());
	assert ( geom_arr.size() == fermat_pot.size());
	    
    for(int itheta = 0; itheta < theta_NM ; itheta++)
    {
        for(int jtheta = 0; jtheta < theta_NM ; jtheta++)
        {
            int arr_ind = jtheta + theta_NM * itheta;
            fermat_pot[arr_ind] = geom_factor*geom_arr[arr_ind] + lens_factor*lens_arr[arr_ind];                
        }
    }
} 

int Sign(const double val){
	return (val > 0) - (val < 0);
}

bool StatCellCheck(const double a, const double b){
    if((Sign(a) != Sign(b)) && fabs(a) < fabs(b)){
        return true;
    }else{
        return false;        
    }    
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
    double Eins_time_const =  1.970196379056507E-05;//4*G*M_sun/c^3 in s/M_sun
    
    double rnorm = sqrt( (imgpos.thetax - srcpos.thetax)*(imgpos.thetax - srcpos.thetax) \
	                   + (imgpos.thetay - srcpos.thetay)*(imgpos.thetay - srcpos.thetay) )/eins;
	                   
    double inorm = sqrt( imgpos.thetax*imgpos.thetax \
	                   + imgpos.thetay*imgpos.thetay )/eins;

    return Eins_time_const*mass*(0.5*rnorm*rnorm - log(inorm) );
}