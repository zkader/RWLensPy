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
	)
{
	std::complex<double> tfunc_val = 0.0 + I*0.0;    
    double lens_param = lens_fac / geom_fac ;
    
	for(int itheta = 2; itheta < theta_NM - 2; itheta++) 
    {
        for(int jtheta = 2; jtheta < theta_NM - 2; jtheta++)
        {
			if( IsStationary( itheta, jtheta, theta_NM,
							theta_step, theta_min,	
							dlens_arr, lens_param, betav )){
				double theta_x = theta_step * itheta + theta_min;				    			
				double theta_y = theta_step * jtheta + theta_min;					
					
				double geomdelay =  geom_fac*0.5*( pow( theta_x - betav.valx ,2.0) + pow( theta_y - betav.valy ,2.0));
				
				double phase = 2 * pi * freq * (geomdelay + lens_fac * lens_arr[jtheta + theta_NM * itheta] );
			
				std::complex<double> mag = GetMag(itheta, jtheta, theta_NM, ddlens_arr, lens_param);
				
                std::complex<double> tempval = mag*std::complex<double>( cos(phase), sin(phase));       
				tfunc_val += tempval;				
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
				
				sourcepos.valx = theta_min + theta_step * itheta + beta_x;				
				sourcepos.valy = theta_min + theta_step * jtheta + beta_y;

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
    if(std::isinf(F_cntr) 
	|| std::isinf(fermat_pot[jtheta + 1 + theta_NM * itheta])
	|| std::isinf(fermat_pot[jtheta - 1 + theta_NM * itheta])
	|| std::isinf(fermat_pot[jtheta + theta_NM * (itheta + 1)])
	|| std::isinf(fermat_pot[jtheta + theta_NM * (itheta + 1)])   ){return false;}

    
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

bool IsStationary(
	const int itheta,
	const int jtheta,
	const int theta_NM,
	const double theta_step,
	const double theta_min,	
	const std::vector<physpoint> &dlens_arr,
    const double lens_param,    
	const physpoint betav ) 
{
	physpoint pnt_cntr = dlens_arr[jtheta + theta_NM * itheta];
    if(std::isinf(pnt_cntr.valx) || std::isinf(pnt_cntr.valy) ) {return false;}
	
	double fx_cntr = theta_step * jtheta + theta_min + lens_param * pnt_cntr.valx;
	double fy_cntr = theta_step * itheta + theta_min + lens_param * pnt_cntr.valy;	
	
	if((betav.valx == fx_cntr) && (betav.valy == fy_cntr)){return true;}	
    	
	physpoint pnt_left = dlens_arr[jtheta - 1 + theta_NM * itheta];	
    if(std::isinf(pnt_left.valx) || std::isinf(pnt_left.valy) ) {return false;}
   	    
	physpoint pnt_right = dlens_arr[jtheta + 1 + theta_NM * itheta];
    if(std::isinf(pnt_right.valx) || std::isinf(pnt_right.valy) ) {return false;}
	
	double fx_left = theta_step * (jtheta - 1) + theta_min + lens_param * pnt_left.valx;	
	double fx_right = theta_step * (jtheta + 1) + theta_min + lens_param * pnt_right.valx;

	if( CellCheck(fx_left, fx_cntr, fx_right, betav.valx))
	{
		physpoint pnt_up = dlens_arr[jtheta + theta_NM * (itheta - 1)];
	    if(std::isinf(pnt_up.valx) || std::isinf(pnt_up.valy) ) {return false;}
		
		physpoint pnt_down = dlens_arr[jtheta + theta_NM * (itheta + 1 )];
	    if(std::isinf(pnt_down.valx) || std::isinf(pnt_down.valy) ) {return false;}		
	
		double fy_up = theta_step * (itheta - 1) + theta_min + lens_param * pnt_up.valy;	
		double fy_down = theta_step * (itheta + 1) + theta_min + lens_param * pnt_down.valy;
	
		if( CellCheck(fy_up, fy_cntr, fy_down, betav.valy)){
			return true;
		}		
	}else
	{
		return false;
	}
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
	
	magval += fxx*fyy - fxy*fxy ;			
	
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

std::complex<double> GetMag(const int itheta, const int jtheta, const int theta_NM, const std::vector<physpoint> &mag_arr, const double lens_param)
{
	std::complex<double> magval;	
	
	physpoint maghess = mag_arr[jtheta + theta_NM * itheta];
	
	//                                            Det(Hess)                   Tr(Hess)
	magval = 1.+I*0.0 + lens_param * lens_param * maghess.valy + lens_param * maghess.valx ; 
	
	if(magval == 0.0 + I*0.0 )
		{
			std::cout << "Warning: No curvature / Caustic at " << itheta << "," << jtheta <<"\n" ;			
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

void SetGradientArrs(
    const int theta_NM,
    const double theta_step,    
	const std::vector<double> &lens_arr,
	std::vector<physpoint> &dlens_arr ,
	std::vector<physpoint> &ddlens_arr 	
){  
	assert ( lens_arr.size() == dlens_arr.size());
	assert ( lens_arr.size() == theta_NM*theta_NM);
	    
	// points lost for hessian
    for(int itheta = 2; itheta < theta_NM - 2; itheta++)
    {
        for(int jtheta = 2; jtheta < theta_NM - 2; jtheta++)
        {
            int arr_ind = jtheta + theta_NM * itheta;
            
		    // central difference
			double Ldy = (lens_arr[jtheta  + theta_NM * (itheta + 1)] - lens_arr[jtheta  + theta_NM * (itheta - 1)]
						)/ (2.0 * theta_step);
			double Ldx = (lens_arr[(jtheta + 1) + theta_NM * itheta] - lens_arr[(jtheta - 1) + theta_NM * itheta]
						)/ (2.0 * theta_step);
						
			physpoint gradpnt;
			
			gradpnt.valx = Ldx;
			gradpnt.valy = Ldy;			
			
            dlens_arr[arr_ind] = gradpnt;            
            
            // Hessian
            double fyy = ( -lens_arr[(jtheta) + theta_NM * (itheta+2)]
				+16.0*lens_arr[(jtheta) + theta_NM * (itheta+1)]
				-30.0*lens_arr[(jtheta) + theta_NM * (itheta)]
				+16.0*lens_arr[(jtheta) + theta_NM * (itheta-1)]
				-lens_arr[(jtheta) + theta_NM * (itheta-2)]	
				)/ (12.0 * theta_step * theta_step);

			double fxx = ( -lens_arr[(jtheta+2) + theta_NM * (itheta)]
				+16.0*lens_arr[(jtheta+1) + theta_NM * (itheta)]
				-30.0*lens_arr[(jtheta) + theta_NM * (itheta)]
				+16.0*lens_arr[(jtheta-1) + theta_NM * (itheta)]
				-lens_arr[(jtheta-2) + theta_NM * (itheta)]	
				)/ (12.0 * theta_step * theta_step);

			double fxy = ( -lens_arr[(jtheta+2) + theta_NM * (itheta+2)]
				+lens_arr[(jtheta-2) + theta_NM * (itheta+2)]
				+lens_arr[(jtheta+2) + theta_NM * (itheta-2)]
				-lens_arr[(jtheta-2) + theta_NM * (itheta-2)]	
				+16.0*lens_arr[(jtheta+1) + theta_NM * (itheta+1)]
				-16.0*lens_arr[(jtheta-1) + theta_NM * (itheta+1)]
				-16.0*lens_arr[(jtheta+1) + theta_NM * (itheta-1)]
				+16.0*lens_arr[(jtheta-1) + theta_NM * (itheta-1)]
			)/ (48.0 * theta_step * theta_step);
            
            
            physpoint magpnt;
			
			// Trace of Hessian
			magpnt.valx = fxx + fyy;
			
			// Determinant of Hessian
			magpnt.valy = fxx*fyy - fxy*fxy;			
			
            ddlens_arr[arr_ind] = magpnt;                
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

bool IVPCheck(const double fa, const double fc, const double fb){
	// f(a) <= fc < f(b) ?
    if((fc > fa) && (fc < fb) )
	{
    	return true;
    }else{
        return false;        
    }    
}	

bool CellCheck(const double left, const double center, const double right, const double val){
	// check if in cell and put to nearest neighbour
	// IVP function bounds  and closest neighbour is grid point
    if( (IVPCheck(left,val,center) && (fabs(center - val) <  fabs(left - val) )) 
	 || (IVPCheck(center,val,right) && (fabs(center - val) <  fabs(right - val) )) )
	{
    	return true;
	}else{
		return false;
	}
}

physpoint map_grav_p(const physpoint srcpos, const double eins){
    double ang = fmod( atan2(srcpos.valy, srcpos.valx) + 2 * pi, 2 * pi);
    double norm = sqrt( srcpos.valx*srcpos.valx + srcpos.valy * srcpos.valy);
    double mag = 0.5 * ( norm + sqrt(norm*norm + 4 * eins * eins ) );    
	physpoint imgpos;
	imgpos.valx = mag*cos(ang);
	imgpos.valy = mag*sin(ang);

    return imgpos;
}

physpoint map_grav_m(const physpoint srcpos, const double eins){
    double ang = fmod( atan2(srcpos.valy, srcpos.valx) + 2 * pi, 2 * pi);
    double norm = sqrt( srcpos.valx*srcpos.valx + srcpos.valy * srcpos.valy);
    double mag = 0.5 * ( norm - sqrt(norm*norm + 4 * eins * eins ) );    
	physpoint imgpos;
	imgpos.valx = mag*cos(ang);
	imgpos.valy = mag*sin(ang);

    return imgpos;
}

std::complex<double> grav_magval(const physpoint imgpos, const double eins){
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
    double Eins_time_const =  1.970196379056507E-05;//4*G*M_sun/c^3 in s/M_sun
    
    double rnorm = sqrt( (imgpos.valx - srcpos.valx)*(imgpos.valx - srcpos.valx) \
	                   + (imgpos.valy - srcpos.valy)*(imgpos.valy - srcpos.valy) )/eins;
	                   
    double inorm = sqrt( imgpos.valx*imgpos.valx \
	                   + imgpos.valy*imgpos.valy )/eins;

    return Eins_time_const*mass*(0.5*rnorm*rnorm - log(inorm) );
}