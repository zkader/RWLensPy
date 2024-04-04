#include "rwlens.h"

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
	)
{
	std::complex<double> tfunc_val = 0.0 + I*0.0;    
    double lens_param = lens_fac * pow(freq,freq_power) / geom_fac ;
    std::complex<double> tempval;
    
	for(int itheta = 2; itheta < theta_N - 2; itheta++) 
    {
        for(int jtheta = 2; jtheta < theta_N - 2; jtheta++)
        {
            // check if there exists a stationary point in the grid cell
			if( IsStationary( itheta, jtheta, theta_N,
							theta_step, theta_min,	
							dlens_arr, lens_param, betav ))
            {
                // Get the phase delay and magnification of the image
                tempval = GetImgVal( itheta, jtheta, theta_step, theta_N, 
                                    theta_min, freq, freq_ref, freq_power,        
                                    lens_arr, ddlens_arr, geom_fac, lens_fac,
                                    betav, nyqzone_aliased);
                                    
                tfunc_val += tempval; //add image to frequency
            }
        }
    }
    return tfunc_val;
} 

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
    )
{
	std::complex<double> tfunc_val = 0.0 + I*0.0;      
    std::complex<double> tempval;
    double lens_param1 = lens_fac1 * pow(freq,freq_power1) / geom_fac1 ;
    double lens_param2 = lens_fac2 * pow(freq,freq_power2) / geom_fac2 ;
    
    double theta1_x, theta1_y, theta2_x, theta2_y;
    double geomdelay1, lensdelay1, geomdelay2, lensdelay2, phase;
    std::complex<double> mag1, mag2;
    
    physpoint beta2;
    
	for(int itheta = 2; itheta < theta_N - 2; itheta++) 
    {
        for(int jtheta = 2; jtheta < theta_N - 2; jtheta++)
        {
            // check if there exists a stationary point in the grid cell        
			if( IsStationary( itheta, jtheta, theta_N,
							theta_step, theta_min,	
							dlens_arr1, lens_param1, beta1 ))
            {
                phase = 0;

				theta1_x  = theta_step * jtheta + theta_min;				    			
				theta1_y  = theta_step * itheta + theta_min;
                
                beta2.valx = multilens_12_scale*theta1_x + lens12_offset.valx;
                beta2.valy = multilens_12_scale*theta1_y + lens12_offset.valy;
                
                for(int ktheta = 2; ktheta < theta_N - 2; ktheta++) 
                {
                    for(int ltheta = 2; ltheta < theta_N - 2; ltheta++)
                    {
                        if( IsStationary( ktheta, ltheta, theta_N,
                                        theta_step, theta_min,	
                                        dlens_arr2, lens_param2, beta2 ))
                        {
                            theta2_x  = theta_step * ltheta + theta_min;				    			
                            theta2_y  = theta_step * ktheta + theta_min;					

                            // plane 1 delays and mag
                            geomdelay1 = geom_fac1*0.5*( pow( theta1_x - beta1.valx ,2.0)\
                                        + pow( theta1_y - beta1.valy ,2.0));

                            lensdelay1 = GetLensDelay(lens_fac1, freq,\
                                                    freq_ref, freq_power1,\
                                                    lens_arr1[jtheta + theta_N * itheta]);

                            mag1 = GetMag(itheta, jtheta, theta_N, ddlens_arr1, lens_param1);

                            // plane 2 delays and mag
                            geomdelay2 = geom_fac2*0.5*( pow( theta2_x - beta2.valx ,2.0)\
                                        + pow( theta2_y - beta2.valy ,2.0));

                            lensdelay2 = GetLensDelay(lens_fac2, freq,\
                                            freq_ref, freq_power2,\
                                            lens_arr2[ktheta + theta_N * ltheta]);

                            mag2 = GetMag(ktheta, ltheta, theta_N, ddlens_arr2, lens_param2);

                            // scipy rfft phase convention is exp(-i 2 pi f t)
                            phase = -2 * pi * freq * ( geomdelay1 + lensdelay1\
                                                     + geomdelay2 + lensdelay2 );        

                            // conj if evaluating in aliased nyq zone
                            if(nyqzone_aliased){phase = -phase;}
              
                            tempval = mag1*mag2*\
                                            std::complex<double>(
                                            cos(phase),
                                            sin(phase));      
                        
                            tfunc_val += tempval;
                        }
                    }
                }
			}
        }
    }
    return tfunc_val;
}


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
	)
{
	double eins = 1.0; //unitless
	std::complex<double> tfunc_val = 0.0 + I*0.0;   
    double lens_param = lens_fac * pow(freq,freq_power) / geom_fac ;
    double theta_x,theta_y;
    double geomdelay, lensdelay, imgphase, phase;
	physpoint sourcepos,imagepos;
    std::complex<double> mag, imgmag, tempval;
    
	for(int itheta = 2; itheta < theta_N - 2; itheta++) 
    {
        for(int jtheta = 2; jtheta < theta_N - 2; jtheta++)
        {
			if( IsStationary( itheta, jtheta, theta_N,
							theta_step, theta_min,	
							dlens_arr, lens_param, betav ))
            {
				theta_x  = theta_step * jtheta + theta_min;				    			
				theta_y  = theta_step * itheta + theta_min;					
					
				geomdelay = geom_fac*0.5*( pow( theta_x - betav.valx ,2.0) + pow( theta_y - betav.valy ,2.0));
				
                lensdelay = GetLensDelay(lens_fac, freq,\
                                        freq_ref, freq_power,\
                                        lens_arr[jtheta + theta_N * itheta]);

                phase = -2 * pi * freq * ( geomdelay + lensdelay );
                // conj if evaluating in aliased nyq zone
                if(nyqzone_aliased){phase = -phase;}
                
				mag = GetMag(itheta, jtheta, theta_N, ddlens_arr, lens_param);
												
				sourcepos.valx = multilens_scale*theta_x + betaE_v.valx;				
				sourcepos.valy = multilens_scale*theta_y + betaE_v.valy;

                // first pm grav solution
				imagepos = map_grav_p(sourcepos, eins);
				imgmag = grav_magval(imagepos, eins);
				imgphase = grav_delayval(imagepos, sourcepos, eins, mass);
				imgphase = -imgphase * 2 * pi * freq;
                if(nyqzone_aliased){imgphase = -imgphase;}

                tempval = imgmag * mag *std::complex<double>(
                            cos(phase+imgphase), sin(phase+imgphase));

                tfunc_val += tempval;       
            	
                // second pm grav solution
				imagepos = map_grav_m(sourcepos, eins);
				imgmag = grav_magval(imagepos, eins);
				imgphase = grav_delayval(imagepos, sourcepos, eins, mass);								
				imgphase = -imgphase * 2 * pi * freq;
                if(nyqzone_aliased){imgphase = -imgphase;}

                tempval = imgmag * mag *std::complex<double>(
                            cos(phase+imgphase), sin(phase+imgphase));
                
                tfunc_val += tempval;
			}
        }
    }

    return tfunc_val ;
}

std::complex<double> GetPMGravTransferFuncVal(
	const double freq,
	const double mass,
	const physpoint betav,
    const bool nyqzone_aliased    
	)
{
	double eins = 1.0; //unitless
	std::complex<double> tfunc_val = 0.0 + I*0.0;
    double imgphase;
	physpoint sourcepos,imagepos;
    std::complex<double> imgmag, tempval;
    
    imagepos = map_grav_p(betav, eins);
    imgmag = grav_magval(imagepos, eins);
    imgphase = grav_delayval(imagepos, betav, eins, mass);    
	imgphase = -imgphase * 2 * pi * freq;
    if(nyqzone_aliased){imgphase = -imgphase;}

    tempval = imgmag * std::complex<double>(
                cos(imgphase), sin(imgphase));
    tfunc_val += tempval;
    
    imagepos = map_grav_m(betav, eins);
    imgmag = grav_magval(imagepos, eins);
    imgphase = grav_delayval(imagepos, betav, eins, mass);								
    imgphase = -imgphase * 2 * pi * freq;
    if(nyqzone_aliased){imgphase = -imgphase;}

    tempval = imgmag * std::complex<double>(
                cos(imgphase), sin(imgphase));
    tfunc_val += tempval;
	
    return tfunc_val;
}

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
    std::vector<imagepoint> &freq_images)
{
    double lens_param = lens_fac * pow(freq,freq_power) / geom_fac ;
    double theta_x,theta_y;
    double geomdelay,delayv;
    std::complex<double> magv;    
    imagepoint stationary_image;
        
	for(int itheta = 2; itheta < theta_N - 2; itheta++) 
    {
        for(int jtheta = 2; jtheta < theta_N - 2; jtheta++)
        {
			if( IsStationary( itheta, jtheta, theta_N,
							theta_step, theta_min,	
							dlens_arr, lens_param, betav ))
            {
				theta_x = theta_step * jtheta + theta_min;				    			
				theta_y = theta_step * itheta + theta_min;					
					
				geomdelay =  geom_fac*0.5*( pow( theta_x - betav.valx ,2.0) + pow( theta_y - betav.valy ,2.0));
				
                delayv = geomdelay + lens_fac * pow(freq,freq_power) * lens_arr[jtheta + theta_N * itheta];                   
                
				magv = GetMag(itheta, jtheta, theta_N, ddlens_arr, lens_param);
				                
				stationary_image.valx = theta_x;
				stationary_image.valy = theta_y;
				stationary_image.valf = freq;
                stationary_image.delay = delayv;                                
                stationary_image.mag = magv;	
                
				freq_images.push_back(stationary_image);                
			}
        }
    }
} 


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
    std::vector<imagepoint> &freq_images)    
{
    double lens_param1 = lens_fac1 * pow(freq,freq_power1) / geom_fac1 ;
    double lens_param2 = lens_fac2 * pow(freq,freq_power2) / geom_fac2 ;
    
    double theta1_x, theta1_y, theta2_x, theta2_y;
    double delay1, delay2;    
    std::complex<double> mag1, mag2;

    physpoint beta_stat;
    imagepoint stationary_image;    
    
	for(int itheta = 2; itheta < theta_N - 2; itheta++) 
    {
        for(int jtheta = 2; jtheta < theta_N - 2; jtheta++)
        {
			if( IsStationary( itheta, jtheta, theta_N,
							theta_step, theta_min,	
							dlens_arr1, lens_param1, beta1 ))
            {
				theta1_x  = theta_step * jtheta + theta_min;				    			
				theta1_y  = theta_step * itheta + theta_min;					
			                                
                beta_stat.valx = scaling_factor*theta1_x + beta2.valx;
                beta_stat.valy = scaling_factor*theta1_y + beta2.valy;
                
                for(int ktheta = 2; ktheta < theta_N - 2; ktheta++) 
                {
                    for(int ltheta = 2; ltheta < theta_N - 2; ltheta++)
                    {
                        if( IsStationary( ktheta, ltheta, theta_N,
                                        theta_step, theta_min,	
                                        dlens_arr2, lens_param2, beta_stat ))
                        {                            
                            theta2_x  = theta_step * ltheta + theta_min;				    			
                            theta2_y  = theta_step * ktheta + theta_min;	
                            
                            delay1 = geom_fac1*0.5*( pow( theta1_x - beta1.valx ,2.0)\
                                    + pow( theta1_y - beta1.valy ,2.0))\
                                    + lens_fac1 * pow( freq,freq_power1)\
                                    * lens_arr1[jtheta + theta_N * itheta];
                            
                            mag1 = GetMag(itheta, jtheta, theta_N, ddlens_arr1, lens_param1);				
                            
                            delay2 = geom_fac2*0.5*( pow( theta2_x - beta_stat.valx ,2.0)\
                                    + pow( theta2_y - beta_stat.valy ,2.0))\
                                    + lens_fac2 * pow( freq,freq_power2)\
                                    * lens_arr2[ltheta + theta_N * ktheta];
                            
                            mag2 = GetMag(ktheta, ltheta, theta_N, ddlens_arr2, lens_param2);				
                                                        
                            stationary_image.valx = theta2_x;
                            stationary_image.valy = theta2_y;
                            stationary_image.valf = freq;
                            stationary_image.delay = delay2 + delay1;                                
                            stationary_image.mag = mag2*mag1;	

                            freq_images.push_back(stationary_image);                

                        }
                    }
                }
			}
        }
    }
}

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
    std::vector<imagepoint> &freq_images)    
{
    double lens_param1 = lens_fac1 * pow(freq,freq_power1) / geom_fac1 ;
	double eins = 1.0; //unitless
    
    double theta1_x, theta1_y;
    double delay1, delay2;    
    std::complex<double> mag1, mag2;

    physpoint beta_stat,imagepos;
    imagepoint grav_image;    
    
	for(int itheta = 2; itheta < theta_N - 2; itheta++) 
    {
        for(int jtheta = 2; jtheta < theta_N - 2; jtheta++)
        {
			if( IsStationary( itheta, jtheta, theta_N,
							theta_step, theta_min,	
							dlens_arr1, lens_param1, beta1 ))
            {
				theta1_x  = theta_step * jtheta + theta_min;				    			
				theta1_y  = theta_step * itheta + theta_min;					
					
				delay1 = geom_fac1*0.5*( pow( theta1_x - beta1.valx ,2.0) + pow( theta1_y - beta1.valy ,2.0)) +  lens_fac1 * pow( freq,freq_power1) * lens_arr1[jtheta + theta_N * itheta];
				mag1 = GetMag(itheta, jtheta, theta_N, ddlens_arr1, lens_param1);				
                                
                beta_stat.valx = scaling_factor*theta1_x + betaE_v.valx;
                beta_stat.valy = scaling_factor*theta1_y + betaE_v.valy;
                
				imagepos = map_grav_p(beta_stat, eins);
                mag2 = grav_magval(imagepos, eins);
				delay2 = grav_delayval(imagepos, beta_stat, eins, mass);

                grav_image.valx = imagepos.valx;
                grav_image.valy = imagepos.valy;
                grav_image.valf = freq;
                grav_image.delay = delay2 + delay1;                                
                grav_image.mag = mag2*mag1;	

                freq_images.push_back(grav_image);                                
                
				imagepos = map_grav_m(beta_stat, eins);
                mag2 = grav_magval(imagepos, eins);
				delay2 = grav_delayval(imagepos, beta_stat, eins, mass);
                
                grav_image.valx = imagepos.valx;
                grav_image.valy = imagepos.valy;
                grav_image.valf = freq;
                grav_image.delay = delay2 + delay1;                                
                grav_image.mag = mag2*mag1;	

                freq_images.push_back(grav_image);                                
                
			}
        }
    }
}


bool IsStationary(
	const int itheta,
	const int jtheta,
	const int theta_N,
	const double theta_step,
	const double theta_min,	
	const std::vector<physpoint> &dlens_arr,
    const double lens_param,    
	const physpoint betav ) 
{
    // determine if there is a stationary point within the grid cell
	physpoint pnt_cntr = dlens_arr[jtheta + theta_N * itheta];
    
    // check for infinites
    if(std::isinf(pnt_cntr.valx) || std::isinf(pnt_cntr.valy) ) {return false;}
	
	double fx_cntr = theta_step * jtheta + theta_min + lens_param * pnt_cntr.valx - betav.valx;
	double fy_cntr = theta_step * itheta + theta_min + lens_param * pnt_cntr.valy - betav.valy;

    // check if both the points are zero    
	if((fx_cntr == 0.0) && (fy_cntr == 0.0)){return true;}	

    // check for if gradient is 0 along x direction first
	physpoint pnt_left = dlens_arr[jtheta - 1 + theta_N * itheta];	
    if(std::isinf(pnt_left.valx) || std::isinf(pnt_left.valy) ) {return false;}   	    
    
	physpoint pnt_right = dlens_arr[jtheta + 1 + theta_N * itheta];    
    if(std::isinf(pnt_right.valx) || std::isinf(pnt_right.valy) ) {return false;}
    	
	double fx_left = theta_step * (jtheta - 1) + theta_min + lens_param * pnt_left.valx - betav.valx;	
	double fx_right = theta_step * (jtheta + 1) + theta_min + lens_param * pnt_right.valx - betav.valx;

	if( StatCellCheck(fx_cntr,fx_left) || StatCellCheck(fx_cntr,fx_right))
	{
        // check for if gradient is 0 along y direction second        
		physpoint pnt_up = dlens_arr[jtheta + theta_N * (itheta - 1)];
	    if(std::isinf(pnt_up.valx) || std::isinf(pnt_up.valy) ) {return false;}
		
		physpoint pnt_down = dlens_arr[jtheta + theta_N * (itheta + 1 )];
	    if(std::isinf(pnt_down.valx) || std::isinf(pnt_down.valy) ) {return false;}		
	
		double fy_up = theta_step * (itheta - 1) + theta_min + lens_param * pnt_up.valy - betav.valy;	
		double fy_down = theta_step * (itheta + 1) + theta_min + lens_param * pnt_down.valy - betav.valy;
	
        // if both directions have a 0 in the gradient, there is a stationary point
		if(StatCellCheck(fy_cntr,fy_up) || StatCellCheck(fy_cntr,fy_down)){
			return true;
		}else{return false;}   
	}else{
        return false;
    }
    return false;
}

std::complex<double> GetMag(
    const int itheta,
    const int jtheta,
    const int theta_N,
    const std::vector<physpoint> &mag_arr,
    const double lens_param)
{
    // Get the magnification of the lens through the hessian
	std::complex<double> magvalcntr, magvalup, magvaldown, magvalleft, magvalright;	
	
	physpoint maghesscntr = mag_arr[jtheta + theta_N * itheta];
	physpoint maghessup = mag_arr[jtheta + theta_N * (itheta-1)];
	physpoint maghessdown = mag_arr[jtheta + theta_N * (itheta+1)];
	physpoint maghessleft = mag_arr[(jtheta-1) + theta_N * itheta];
	physpoint maghessright = mag_arr[(jtheta+1) + theta_N * itheta];
    	
    // valx is positive eigvalue and valy is negative eigvalue
	// calculate the mag at all grid boundaries
	magvalcntr = 0.25 * (2 + lens_param * maghesscntr.valx) * (2 + lens_param * maghesscntr.valy) ;
	magvalup = 0.25 * (2 + lens_param * maghessup.valx) * (2 + lens_param * maghessup.valx);
	magvaldown = 0.25 * (2 + lens_param * maghessdown.valx) * (2 + lens_param * maghessdown.valx) ;
	magvalleft = 0.25 * (2 + lens_param * maghessleft.valx) * (2 + lens_param * maghessleft.valx) ;
	magvalright = 0.25 * (2 + lens_param * maghessright.valx) * (2 + lens_param * maghessright.valx) ;

    // Check for 0 in the hessian i.e. a caustic point within the grid cell
    if( (StatCellCheck(fabs(magvalcntr),fabs(magvalleft)) || StatCellCheck(fabs(magvalcntr),fabs(magvalright))) 
       && (StatCellCheck(fabs(magvalcntr),fabs(magvalup)) || StatCellCheck(fabs(magvalcntr),fabs(magvaldown))) ){
        std::cout << "Warning: No curvature / Caustic at " << itheta << "," << jtheta << std::endl;			
		return 0.0+I*0.0;                
    }else
	{
		magvalcntr = pow(magvalcntr,-0.5);
		return magvalcntr;	
	}
}

double GetLensDelay(
    const double lens_fac,
    const double freq,
    const double freq_ref,
    const double freq_power,
    const double lens_arr_val
){
    double lens_delay = 0;    
    if(freq_power != 0.0)
    {
        lens_delay = lens_fac * ( pow( freq_ref, freq_power)
                            -  pow( freq, freq_power) )\
                            * lens_arr_val ;        
    }else
    {
        lens_delay = lens_fac * lens_arr_val ;
    }  
    return lens_delay;
}

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
){
    // Set the values of the geometric delay on a grid
    double phase, geomdelay, lensdelay;
    double theta_x  = theta_step * jtheta + theta_min;				    			
    double theta_y  = theta_step * itheta + theta_min;					
    double lens_param = lens_fac * pow(freq,freq_power) / geom_fac ;
    std::complex<double> mag;

    // geometric delay
    geomdelay = geom_fac*0.5*( pow( theta_x - betav.valx ,2.0) + pow( theta_y - betav.valy ,2.0));
    
    // lensing delay
    lensdelay = GetLensDelay(lens_fac, freq,\
                            freq_ref, freq_power,\
                            lens_arr[jtheta + theta_N * itheta]);
    
    // scipy rfft phase convention is exp(-i 2 pi f t)
    phase = -2 * pi * freq * ( geomdelay + lensdelay );

    // conj if evaluating in aliased nyq zone
    if(nyqzone_aliased){phase = -phase;}

    // get the image magnification
    mag = GetMag(itheta, jtheta, theta_N, ddlens_arr, lens_param);

    // get image phase
    mag = mag*std::complex<double>( cos(phase), sin(phase));
    
    return mag;
}

void SetGeometricDelayArr(
	const double theta_min,
	const double theta_max,
	const int theta_N,
	const double beta_x,
	const double beta_y,
	std::vector<double> &geom_arr
)
{
    // Set the values of the geometric delay on a grid
	assert ( geom_arr.size() == theta_N * theta_N);	
	
	for(int arr_ii = 0; arr_ii < theta_N; arr_ii++)
	{
		double theta_x = (theta_max - theta_min) / (theta_N - 1) * arr_ii + theta_min;				
		for(int arr_jj = 0; arr_jj < theta_N; arr_jj++)
		{
			double theta_y = (theta_max - theta_min) / (theta_N - 1) * arr_jj + theta_min;
            int arr_ind = arr_jj + arr_ii * theta_N;
			geom_arr[arr_ind] = 0.5*( ( theta_x - beta_x ) * (theta_x - beta_x)  + (theta_y - beta_y) * (theta_y - beta_y) );
		}
	}
}

void SetFermatPotential(
	const double geom_factor,
	const double lens_factor,
    const int theta_N,
    const double freq,
	const std::vector<double> &geom_arr,
	const std::vector<double> &lens_arr,
	std::vector<double> &fermat_pot )
{
    // Set the values of the Fermat potential
	assert ( geom_arr.size() == theta_N*theta_N);    
	assert ( geom_arr.size() == lens_arr.size());
	assert ( geom_arr.size() == fermat_pot.size());
	    
    for(int itheta = 0; itheta < theta_N ; itheta++)
    {
        for(int jtheta = 0; jtheta < theta_N ; jtheta++)
        {
            int arr_ind = jtheta + theta_N * itheta;
            fermat_pot[arr_ind] = geom_factor*geom_arr[arr_ind] + lens_factor*lens_arr[arr_ind];                
        }
    }
} 

void SetGradientArrs(
    const int theta_N,
    const double theta_step,    
	const std::vector<double> &lens_arr,
	std::vector<physpoint> &dlens_arr ,
	std::vector<physpoint> &ddlens_arr 	
){  
    // Calculate the gradient and eigenvalues of the hessian for a lens 
	assert ( lens_arr.size() == dlens_arr.size());
	assert ( lens_arr.size() == theta_N*theta_N);
	    
	// note: 2 points lost for hessian for finite difference calcuation
    for(int itheta = 2; itheta < theta_N - 2; itheta++)
    {
        for(int jtheta = 2; jtheta < theta_N - 2; jtheta++)
        {
            int arr_ind = jtheta + theta_N * itheta;
            
		    // central difference
			double Ldy = (lens_arr[jtheta  + theta_N * (itheta + 1)] - lens_arr[jtheta  + theta_N * (itheta - 1)]
						)/ (2.0 * theta_step);
			double Ldx = (lens_arr[(jtheta + 1) + theta_N * itheta] - lens_arr[(jtheta - 1) + theta_N * itheta]
						)/ (2.0 * theta_step);
						
			physpoint gradpnt;
			
			gradpnt.valx = Ldx;
			gradpnt.valy = Ldy;			
			
            dlens_arr[arr_ind] = gradpnt;            
            
            // Hessian calculation
            double fyy = ( -lens_arr[(jtheta) + theta_N * (itheta+2)]
				+16.0*lens_arr[(jtheta) + theta_N * (itheta+1)]
				-30.0*lens_arr[(jtheta) + theta_N * (itheta)]
				+16.0*lens_arr[(jtheta) + theta_N * (itheta-1)]
				-lens_arr[(jtheta) + theta_N * (itheta-2)]	
				)/ (12.0 * theta_step * theta_step);

			double fxx = ( -lens_arr[(jtheta+2) + theta_N * (itheta)]
				+16.0*lens_arr[(jtheta+1) + theta_N * (itheta)]
				-30.0*lens_arr[(jtheta) + theta_N * (itheta)]
				+16.0*lens_arr[(jtheta-1) + theta_N * (itheta)]
				-lens_arr[(jtheta-2) + theta_N * (itheta)]	
				)/ (12.0 * theta_step * theta_step);

			double fxy = ( -lens_arr[(jtheta+2) + theta_N * (itheta+2)]
				+lens_arr[(jtheta-2) + theta_N * (itheta+2)]
				+lens_arr[(jtheta+2) + theta_N * (itheta-2)]
				-lens_arr[(jtheta-2) + theta_N * (itheta-2)]	
				+16.0*lens_arr[(jtheta+1) + theta_N * (itheta+1)]
				-16.0*lens_arr[(jtheta-1) + theta_N * (itheta+1)]
				-16.0*lens_arr[(jtheta+1) + theta_N * (itheta-1)]
				+16.0*lens_arr[(jtheta-1) + theta_N * (itheta-1)]
			)/ (48.0 * theta_step * theta_step);
            
            
            physpoint magpnt;
			
			// first eigenvalue
            magpnt.valx = fxx + fyy + pow( pow(fxx - fyy,2) + 4 * fxy* fxy ,0.5);
			
			// second eigenvalue
            magpnt.valy = fxx + fyy - pow( pow(fxx - fyy,2) + 4 * fxy* fxy ,0.5);
			
            ddlens_arr[arr_ind] = magpnt;                
        }
    }	
}

int Sign(const double val){
    // Get the sign of a number
	return (val > 0) - (val < 0);
}

bool StatCellCheck(const double a, const double b){
    // Check if zero is crossed between two values
    if(( (Sign(a) != Sign(b)) || (a == 0.0) ) && (fabs(a) < fabs(b)) ){
        return true;
    }else{
        return false;        
    }    
}

bool IVPCheck(const double fa, const double fc, const double fb){
	// Check if f(a) <= fc < f(b) 
    if((fc > fa) && (fc < fb) )
	{
    	return true;
    }else{
        return false;        
    }    
}	

bool CellCheck(const double left, const double center, const double right, const double val){
	// check if in cell and put to nearest neighbour
	// IVP function bounds and closest neighbour is grid point
    if( (IVPCheck(left,val,center) && (fabs(center - val) <  fabs(left - val) )) 
	 || (IVPCheck(center,val,right) && (fabs(center - val) <  fabs(right - val) )) )
	{
    	return true;
	}else{
		return false;
	}
}

physpoint map_grav_p(const physpoint srcpos, const double eins){
    // Get the positive image plane solution to the point mass grav lens
    double ang = fmod( atan2(srcpos.valy, srcpos.valx) + 2 * pi, 2 * pi);
    double norm = sqrt( srcpos.valx*srcpos.valx + srcpos.valy * srcpos.valy);
    double mag = 0.5 * ( norm + sqrt(norm*norm + 4 * eins * eins ) );    
    
	physpoint imgpos;
	imgpos.valx = mag*cos(ang);
	imgpos.valy = mag*sin(ang);

    return imgpos;
}

physpoint map_grav_m(const physpoint srcpos, const double eins){
    // Get the negative image plane solution to the point mass grav lens    
    double ang = fmod( atan2(srcpos.valy, srcpos.valx) + 2 * pi, 2 * pi);
    double norm = sqrt( srcpos.valx*srcpos.valx + srcpos.valy * srcpos.valy);
    double mag = 0.5 * ( norm - sqrt(norm*norm + 4 * eins * eins ) );    

	physpoint imgpos;
	imgpos.valx = mag*cos(ang);
	imgpos.valy = mag*sin(ang);

    return imgpos;
}

std::complex<double> grav_magval(const physpoint imgpos, const double eins){
    // Get the magnification of a point mass grav lens    
	double norm = sqrt( imgpos.valx*imgpos.valx + imgpos.valy * imgpos.valy);

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
    // Get the time delay of a point mass grav lens    
    double Eins_time_const =  1.970196379056507E-05;//4*G*M_sun/c^3 in s/M_sun
    
    double rnorm = sqrt( (imgpos.valx - srcpos.valx)*(imgpos.valx - srcpos.valx) \
	                   + (imgpos.valy - srcpos.valy)*(imgpos.valy - srcpos.valy) )/eins;
	                   
    double inorm = sqrt( imgpos.valx*imgpos.valx \
	                   + imgpos.valy*imgpos.valy )/eins;

    return Eins_time_const*mass*(0.5*rnorm*rnorm - log(inorm) );
}
