#!/usr/bin/env python

import os
import sys
import pdb
import numpy as np
import pandas as pd
import datetime as dt
import scipy.stats as stats
from scipy.interpolate import interp1d
from scipy import integrate
import matplotlib
import matplotlib.pyplot as plt
from plot import color_list

def linrest(x,y):
    n = len(y)
    dofreedom = n-2
    z, _ = np.polyfit(x,y,1,cov=True)
    p = np.poly1d(z)
    yp = p(x) #predicted y values based on fit
    slope = z[0]
    intercept = z[1]
    r2 = np.corrcoef(x,y)[0][1]**2
    #regression_ss = np.sum( (yp-np.mean(y))**2)
    residual_ss = np.sum( (y-yp)**2 )
    slope_pm = np.sqrt(residual_ss / (dofreedom*np.sum((x-np.mean(x))**2)))
    intercept_pm = slope_pm*np.sqrt(np.sum(x**2)/n)
    #s = np.sqrt(residual_ss/dofreedom)
    #F = regression_ss/s**2
    
    return slope, slope_pm, intercept, intercept_pm, r2 

def conversion_factor(df, x = 'lnAOD440', y = 'ln_n50'):
    
    #Linear fit    
    if len(df)>0:
        m, error_m, n, error_n, R2 = linrest(df.loc[:, x], df.loc[:, y])
    else: 
        m, error_m, n, error_n, R2 = np.nan*np.ones(5)
    #exponent
    exponent = m
    exponent_error = error_m

    #intercept
    intercept = np.exp(n)
    intercept_error = intercept*error_n

    return [exponent, exponent_error, intercept, intercept_error, R2]

def filter_df(df,conditions_dict):
    conditions = True*np.ones(len(df.index))
    for key_ in conditions_dict.keys():
        condition_ = df[key_] == conditions_dict[key_]
        conditions =  np.logical_and(conditions,condition_)    
    df_ = df[conditions]
    return df_

def resample_logradius_distribution(distribution, factor=10, interpolation_kind='quadratic'):
    radius = np.asarray([0.050000,0.065604,0.086077,0.112939,0.148184,0.194429,0.255105,0.334716,0.439173,0.576227,0.756052,0.991996,1.301571,1.707757,2.240702,2.939966,3.857452,5.061260,6.640745,8.713145,11.432287,15.000000]        )
    lnr = np.log(radius)
    resol_lnr = np.diff(lnr).mean()
    resample_resol_lnr = resol_lnr/factor
    resample_lnr = np.arange(lnr[0], lnr[-1]+resol_lnr/factor,resol_lnr/factor)
    resample_radius = np.exp(resample_lnr)

    #Increase resolution to improve the fitting      
    resample_function = interp1d(radius, distribution, kind=interpolation_kind, bounds_error=True)
    resample_distribution_scipy = resample_function(resample_radius)

    resample_distribution_numpy = np.interp(resample_radius, radius, distribution)

    resample_distribution = np.concatenate((resample_distribution_scipy[resample_radius<11.432287], 
                    resample_distribution_numpy[resample_radius>=11.432287]))
    return resample_distribution, resample_radius, resample_lnr, resample_resol_lnr    

def reader_all(filepath):
    '''
    Input:
    filepath: File path of AERONET file type *.all
    Output:
    df: dataframe
    '''
    if os.path.isfile(filepath):
        df = pd.read_csv(filepath,skiprows=6)
        df['Datetime'] = pd.to_datetime(df['Date(dd:mm:yyyy)'] + ' ' + df['Time(hh:mm:ss)'], format='%d:%m:%Y %H:%M:%S')
        df.set_index('Datetime',inplace=True)
        df.drop(['Date(dd:mm:yyyy)', 'Time(hh:mm:ss)', 'Day_of_Year', 'Day_of_Year(Fraction)'], axis=1)
        df['date'] = df.index.date
        df['time'] = df.index.time
    else:
        df =[]
    return df

def lidar_wavelengths_lev15(df):
    #AOD @ 355, 550, 532, 1064 nm
    df['AOD_355nm'] = df['AOD_440nm'].values*(355/440)**(-df['380-500_Angstrom_Exponent'].values)
    df['AOD_550nm'] = df['AOD_675nm'].values*(550./675.)**(-df['500-870_Angstrom_Exponent'].values)
    df['AOD_532nm'] = df['AOD_550nm'].values*(532/550)**(-df['500-870_Angstrom_Exponent'].values)

    df['675-1020nm_Angstrom_Exponent'] = - (np.log(df['AOD_675nm'].values/df['AOD_1020nm'].values))/(np.log(675/1020))
    df['AOD_1064nm'] = df['AOD_1020nm'].values*(1064/1020)**(-df['675-1020nm_Angstrom_Exponent'].values)

    return df

def lidar_wavelengths_all(df):
    #AOD @ 355, 550, 532, 1064 nm
    df['AOD_Coincident_Input[355nm]'] = df['AOD_Coincident_Input[440nm]'].values*(355/440)**(-df['Angstrom_Exponent_440-870nm_from_Coincident_Input_AOD'].values)
    df['AOD_Coincident_Input[550nm]'] = df['AOD_Coincident_Input[675nm]'].values*(550./675.)**(-df['Angstrom_Exponent_440-870nm_from_Coincident_Input_AOD'].values)
    df['AOD_Coincident_Input[532nm]'] = df['AOD_Coincident_Input[550nm]'].values*(532/550)**(-df['Angstrom_Exponent_440-870nm_from_Coincident_Input_AOD'].values)
    df['AOD[550nm]_extinction'] = df['AOD_Coincident_Input[675nm]'].values*(550./675.)**(-df['Extinction_Angstrom_Exponent_440-870nm-Total'].values)
    
    df['Angstrom_Exponent_675-1020nm_from_Coincident_Input_AOD'] = - (np.log(df['AOD_Coincident_Input[675nm]'].values/df['AOD_Coincident_Input[1020nm]'].values))/(np.log(675/1020))
    df['AOD_Coincident_Input[1064nm]'] = df['AOD_Coincident_Input[1020nm]'].values*(1064/1020)**(-df['Angstrom_Exponent_675-1020nm_from_Coincident_Input_AOD'].values)    
    return df

def retrieval4CCN_resample(filepath):
    '''
    Input:
    filepath: File path of AERONET file type *.all
    Output:
    df: dataframe including new variables ['lnAOD440', 'N', 'lnN', 'lnN100', 'lnN_fine', 'lnN_coarse', 'N_fine', 'N_coarse', 'FMF', 'AOD[550nm]_coincident', 'AOD[550nm]_extinction']
    '''
    df = reader_all(filepath)

    if len(df) > 0:

        #Find radius in df header 
        radius = np.asarray([0.050000,0.065604,0.086077,0.112939,0.148184,0.194429,0.255105,0.334716,0.439173,0.576227,0.756052,0.991996,1.301571,1.707757,2.240702,2.939966,3.857452,5.061260,6.640745,8.713145,11.432287,15.000000])        

        dV_dlnr = df.iloc[:,np.arange(53,75,1)  ] #VSD um^3/um^2
        dV_dlnr = dV_dlnr.to_numpy()

        Vradius = (4./3.)*np.pi*np.power(radius,3) #i-bin volume        
        dN_dlnr = (dV_dlnr / Vradius) #NSD #/um^2
        n50 = np.zeros(len(df))
        s50 = np.zeros(len(df))
        n60 = np.zeros(len(df))
        s60 = np.zeros(len(df))
        n100 = np.zeros(len(df))
        n120 = np.zeros(len(df))
        s100 = np.zeros(len(df))
        s120 = np.zeros(len(df))
        n250 = np.zeros(len(df))
        s250 = np.zeros(len(df))
        n290 = np.zeros(len(df))
        s290 = np.zeros(len(df))        
        n500 = np.zeros(len(df))
        s500 = np.zeros(len(df))        
        n_coarse = np.zeros(len(df))
        s_coarse = np.zeros(len(df))
        n_fine = np.zeros(len(df))
        s_fine = np.zeros(len(df))        
        for idx in np.arange(len(df)):
            #Resample
            resample_dN_dlnr, resample_radius,  _, resample_resol_lnr = resample_logradius_distribution(dN_dlnr[idx,:], factor=10)            
            #n50
            n50[idx] = np.sum(resample_dN_dlnr[resample_radius>=0.05]*resample_resol_lnr)#Total N #/microm^2                        
            #s50
            S50 = 4*np.pi*resample_radius[resample_radius>=0.05]**2
            s50[idx] = np.sum(resample_dN_dlnr[resample_radius>=0.05]*S50*resample_resol_lnr)#Total N #/microm^2

            #N60        
            n60[idx] = np.sum(resample_dN_dlnr[resample_radius>=0.06]*resample_resol_lnr)#Total N #/microm^2
            #s60
            S60 = 4*np.pi*resample_radius[resample_radius>=0.06]**2
            s60[idx] = np.sum(resample_dN_dlnr[resample_radius>=0.06]*S60*resample_resol_lnr)#Total N #/microm^2

            #N100
            n100[idx] = np.sum(resample_dN_dlnr[resample_radius>=0.1]*resample_resol_lnr)#Total N #/microm^2
            #s100
            S100 = 4*np.pi*resample_radius[resample_radius>=0.1]**2
            s100[idx] = np.sum(resample_dN_dlnr[resample_radius>=0.1]*S100*resample_resol_lnr)#Total N #/microm^2

            #N120
            n120[idx] = np.sum(resample_dN_dlnr[resample_radius>=0.12]*resample_resol_lnr)#Total N #/microm^2
            #s120
            S120 = 4*np.pi*resample_radius[resample_radius>=0.12]**2
            s120[idx] = np.sum(resample_dN_dlnr[resample_radius>=0.12]*S120*resample_resol_lnr)#Total N #/microm^2

            #N250
            n250[idx] = np.sum(resample_dN_dlnr[resample_radius>=0.25]*resample_resol_lnr)#Total N #/microm^2
            #s250
            S250 = 4*np.pi*resample_radius[resample_radius>=0.25]**2
            s250[idx] = np.sum(resample_dN_dlnr[resample_radius>=0.25]*S250*resample_resol_lnr)#Total N #/microm^2

            #N290
            n290[idx] = np.sum(resample_dN_dlnr[resample_radius>=0.29]*resample_resol_lnr)#Total N #/microm^2

            #s290
            S290 = 4*np.pi*resample_radius[resample_radius>=0.29]**2
            s290[idx] = np.sum(resample_dN_dlnr[resample_radius>=0.29]*S290*resample_resol_lnr)#Total N #/microm^2

            #N500
            n500[idx] = np.sum(resample_dN_dlnr[resample_radius>=0.5]*resample_resol_lnr)#Total N #/microm^2

            #s500
            S500 = 4*np.pi*resample_radius[resample_radius>=0.5]**2
            s500[idx] = np.sum(resample_dN_dlnr[resample_radius>=0.5]*S500*resample_resol_lnr)#Total N #/microm^2

            #n Coarse         
            inflection_radius = df['Inflection_Radius_of_Size_Distribution(um)'].iloc[idx]
            n_coarse[idx] = np.sum(resample_dN_dlnr[resample_radius>=inflection_radius]*resample_resol_lnr)#Total N #/microm^2
            #s Coarse            
            Scoarse = 4*np.pi*resample_radius[resample_radius>=inflection_radius]**2
            s_coarse[idx] = np.sum(resample_dN_dlnr[resample_radius>=inflection_radius]*Scoarse*resample_resol_lnr)#Total N #/microm^2                       

            #Fine            
            n_fine[idx] = np.sum(resample_dN_dlnr[resample_radius<inflection_radius]*resample_resol_lnr)#Total N #/microm^2
            #s Fine            
            Sfine = 4*np.pi*resample_radius[resample_radius<inflection_radius]**2
            s_fine[idx] = np.sum(resample_dN_dlnr[resample_radius<inflection_radius]*Sfine*resample_resol_lnr)#Total N #/microm^2                       

        #Save arrays in DATAFRAME
        df['n50'] = n50
        df['ln_n50'] = np.log(n50)
        df['s50'] = s50
        df['ln_s50'] = np.log(s50)

        df['n60'] = n60
        df['ln_n60'] = np.log(n60)
        df['s60'] = s60
        df['ln_s60'] = np.log(s60)

        df['n100'] = n100
        df['ln_n100'] = np.log(n100)
        df['s100'] = s100
        df['ln_s100'] = np.log(s100)

        df['n120'] = n120
        df['ln_n120'] = np.log(n120)
        df['s120'] = s120
        df['ln_s120'] = np.log(s120)

        df['n250'] = n250
        df['ln_n250'] = np.log(n250)
        df['s250'] = s250
        df['ln_s250'] = np.log(s250)

        df['n290'] = n290
        df['ln_n290'] = np.log(n290)
        df['s290'] = s290
        df['ln_s290'] = np.log(s290)

        df['n_coarse'] = n_coarse
        df['ln_n_coarse'] = np.log(n_coarse)
        df['s_coarse'] = s_coarse
        df['ln_s_coarse'] = np.log(s_coarse)

        df['n_fine'] = n_fine
        df['ln_n_fine'] = np.log(n_fine)
        df['s_fine'] = s_fine
        df['ln_s_fine'] = np.log(s_fine)

        #AOD @ 355, 550, 532, 1064 nm
        df['AOD_Coincident_Input[355nm]'] = df['AOD_Coincident_Input[440nm]'].values*(355/440)**(-df['Angstrom_Exponent_440-870nm_from_Coincident_Input_AOD'].values)
        df['AOD_Coincident_Input[550nm]'] = df['AOD_Coincident_Input[675nm]'].values*(550./675.)**(-df['Angstrom_Exponent_440-870nm_from_Coincident_Input_AOD'].values)
        df['AOD_Coincident_Input[532nm]'] = df['AOD_Coincident_Input[550nm]'].values*(532/550)**(-df['Angstrom_Exponent_440-870nm_from_Coincident_Input_AOD'].values)
        df['AOD[550nm]_extinction'] = df['AOD_Coincident_Input[675nm]'].values*(550./675.)**(-df['Extinction_Angstrom_Exponent_440-870nm-Total'].values)
        
        df['Angstrom_Exponent_675-1020nm_from_Coincident_Input_AOD'] = - (np.log(df['AOD_Coincident_Input[675nm]'].values/df['AOD_Coincident_Input[1020nm]'].values))/(np.log(675/1020))
        df['AOD_Coincident_Input[1064nm]'] = df['AOD_Coincident_Input[1020nm]'].values*(1064/1020)**(-df['Angstrom_Exponent_675-1020nm_from_Coincident_Input_AOD'].values)

        #FMF
        df['FMF'] = df['AOD_Extinction-Fine[675nm]']/df['AOD_Extinction-Total[675nm]']

        #Ln applied to AOD
        waves = [355, 440, 550, 532, 675, 870, 1020, 1064]
        for wave_ in waves:
            df['lnAOD%d' % wave_] = np.log(df['AOD_Coincident_Input[%dnm]' % wave_])
    return df


def aerosol_type_classification(df):    
    df['aerosol_type'] = 'MIXTURE'
    
    #DUST
    fmfmin_threshold, fmfmax_threshold, ssamin_threshold, ssamax_threshold = 0, 0.4, 0.0, 0.95
    dust_condition = np.logical_and.reduce((df['Single_Scattering_Albedo[440nm]']>ssamin_threshold,
                                            df['Single_Scattering_Albedo[440nm]']<=ssamax_threshold,
                                            df['FMF']>fmfmin_threshold,
                                            df['FMF']<fmfmax_threshold))
    df.loc[dust_condition, 'aerosol_type'] = 'DUST'
    
    #NA
    fmfmin_threshold, fmfmax_threshold, ssamin_threshold, ssamax_threshold = 0.6, 1., 0.95, 1.
    na_condition = np.logical_and.reduce((df['Single_Scattering_Albedo[440nm]']>ssamin_threshold,
                                            df['Single_Scattering_Albedo[440nm]']<ssamax_threshold,
                                            df['FMF']>fmfmin_threshold,
                                            df['FMF']<fmfmax_threshold))
    df.loc[na_condition, 'aerosol_type'] = 'NA'

    #HA
    fmfmin_threshold, fmfmax_threshold, ssamin_threshold, ssamax_threshold = 0.6, 1., 0., 0.85
    ha_condition = np.logical_and.reduce((df['Single_Scattering_Albedo[440nm]']>ssamin_threshold,
                                            df['Single_Scattering_Albedo[440nm]']<ssamax_threshold,
                                            df['FMF']>fmfmin_threshold,
                                            df['FMF']<fmfmax_threshold))
    df.loc[ha_condition, 'aerosol_type'] = 'HA'
    #SA
    fmfmin_threshold, fmfmax_threshold, ssamin_threshold, ssamax_threshold = 0.6, 1., 0.9, 0.95
    sa_condition = np.logical_and.reduce((df['Single_Scattering_Albedo[440nm]']>ssamin_threshold,
                                            df['Single_Scattering_Albedo[440nm]']<=ssamax_threshold,
                                            df['FMF']>fmfmin_threshold,
                                            df['FMF']<fmfmax_threshold))
    df.loc[sa_condition, 'aerosol_type'] = 'SA'

    df['mode_predominance'] = 'MIXTURE'
    
    #COARSE
    aemax_threshold = 0.6
    coarse_condition = df['Angstrom_Exponent_440-870nm_from_Coincident_Input_AOD']<=aemax_threshold
    df.loc[coarse_condition, 'mode_predominance'] = 'COARSE'

    #HIGHLY COARSE
    aemax_threshold = 0.2
    hcoarse_condition = df['Angstrom_Exponent_440-870nm_from_Coincident_Input_AOD']<=aemax_threshold
    df.loc[hcoarse_condition, 'mode_predominance'] = 'HIGHLY_COARSE'

    #NONE
    aemin_threshold, aemax_threshold = 0.6, 1.4 
    none_condition = np.logical_and(df['Angstrom_Exponent_440-870nm_from_Coincident_Input_AOD']>aemin_threshold, df['Angstrom_Exponent_440-870nm_from_Coincident_Input_AOD']<aemax_threshold)
    df.loc[none_condition, 'mode_predominance'] = 'NONE'

    #FINE
    aemin_threshold = 1.4
    fine_condition = df['Angstrom_Exponent_440-870nm_from_Coincident_Input_AOD']>=aemin_threshold
    df.loc[fine_condition, 'mode_predominance'] = 'FINE'

    #HIGHLY_FINE
    aemin_threshold = 1.8
    hfine_condition = df['Angstrom_Exponent_440-870nm_from_Coincident_Input_AOD']>=aemin_threshold
    df.loc[fine_condition, 'mode_predominance'] = 'HIGHLY_FINE'
    return df

def retrieval4CCN_simpson(filepath):
    '''
    Input:
    filepath: File path of AERONET file type *.all
    Output:
    df: dataframe including new variables ['lnAOD440', 'N', 'lnN', 'lnN100', 'lnN_fine', 'lnN_coarse', 'N_fine', 'N_coarse', 'FMF', 'AOD[550nm]_coincident', 'AOD[550nm]_extinction']
    '''
    df = reader_all(filepath)

    if len(df) > 0:

        #Find radius in df header 
        radius = np.asarray([0.050000,0.065604,0.086077,0.112939,0.148184,0.194429,0.255105,0.334716,0.439173,0.576227,0.756052,0.991996,1.301571,1.707757,2.240702,2.939966,3.857452,5.061260,6.640745,8.713145,11.432287,15.000000]) #um

        dV_dlnr = df.iloc[:,np.arange(53,75,1)  ] #VSD um^3/um^2
        dV_dlnr = dV_dlnr.to_numpy()

        Vradius = (4./3.)*np.pi*np.power(radius,3) #i-bin volume  um^3
        delta_lnr = np.diff(np.log(radius)).mean() #resolution um
        dN_dlnr_serie = (dV_dlnr / Vradius) #NSD #/um^2
        dN_dlnr_serie_m = (dV_dlnr / Vradius) #NSD #/um^2
        n50 = np.zeros(len(df))
        s50 = np.zeros(len(df))
        n60 = np.zeros(len(df))
        s60 = np.zeros(len(df))
        n100 = np.zeros(len(df))
        n120 = np.zeros(len(df))
        s100 = np.zeros(len(df))
        s120 = np.zeros(len(df))
        n250 = np.zeros(len(df))
        s250 = np.zeros(len(df))
        n290 = np.zeros(len(df))
        s290 = np.zeros(len(df))        
        n500 = np.zeros(len(df))
        s500 = np.zeros(len(df))        
        n_coarse = np.zeros(len(df))
        s_coarse = np.zeros(len(df))
        n_fine = np.zeros(len(df))
        s_fine = np.zeros(len(df))        
        for idx in np.arange(len(df)):
            #Increase resolution to improve the fitting                  
            dN_dlnr = dN_dlnr_serie_m[idx,:]

            #n50
            n50[idx] = integrate.simps(dN_dlnr[radius>=0.05],dx=delta_lnr) #Total N #/microm^2                        
            #s50
            S50 = 4*np.pi*radius[radius>=0.05]**2
            
            s50[idx] = integrate.simps(dN_dlnr[radius>=0.05]*S50,dx=delta_lnr) #Total N #/microm^2

            #N60        
            n60[idx] = integrate.simps(dN_dlnr[radius>=0.06],dx=delta_lnr)#Total N #/microm^2
            #s60
            S60 = 4*np.pi*radius[radius>=0.06]**2
            s60[idx] = integrate.simps(dN_dlnr[radius>=0.06]*S60,dx=delta_lnr)#Total N #/microm^2

            #N100
            n100[idx] = integrate.simps(dN_dlnr[radius>=0.1],dx=delta_lnr)#Total N #/microm^2
            #s100
            S100 = 4*np.pi*radius[radius>=0.1]**2
            s100[idx] = integrate.simps(dN_dlnr[radius>=0.1]*S100,dx=delta_lnr)#Total N #/microm^2

            #N120
            n120[idx] = integrate.simps(dN_dlnr[radius>=0.12],dx=delta_lnr)#Total N #/microm^2
            #s120
            S120 = 4*np.pi*radius[radius>=0.12]**2
            s120[idx] = integrate.simps(dN_dlnr[radius>=0.12]*S120,dx=delta_lnr)#Total N #/microm^2

            #N250
            n250[idx] = integrate.simps(dN_dlnr[radius>=0.25],dx=delta_lnr)#Total N #/microm^2
            #s250
            S250 = 4*np.pi*radius[radius>=0.25]**2
            s250[idx] = integrate.simps(dN_dlnr[radius>=0.25]*S250,dx=delta_lnr)#Total N #/microm^2

            #N290
            n290[idx] = integrate.simps(dN_dlnr[radius>=0.29],dx=delta_lnr)#Total N #/microm^2

            #s290
            S290 = 4*np.pi*radius[radius>=0.29]**2
            s290[idx] = integrate.simps(dN_dlnr[radius>=0.29]*S290,dx=delta_lnr)#Total N #/microm^2

            #N500
            n500[idx] = integrate.simps(dN_dlnr[radius>=0.5],dx=delta_lnr)#Total N #/microm^2

            #s500
            S500 = 4*np.pi*radius[radius>=0.5]**2
            s500[idx] = integrate.simps(dN_dlnr[radius>=0.5]*S500,dx=delta_lnr)#Total N #/microm^2

            #n Coarse    
            inflection_radius = df['Inflection_Radius_of_Size_Distribution(um)'].iloc[idx]     
            n_coarse[idx] = integrate.simps(dN_dlnr[radius>=inflection_radius],dx=delta_lnr)#Total N #/microm^2
            #s Coarse            
            Scoarse = 4*np.pi*radius[radius>=inflection_radius]**2
            s_coarse[idx] = integrate.simps(dN_dlnr[radius>=inflection_radius]*Scoarse,dx=delta_lnr)#Total N #/microm^2                       

            #Fine            
            n_fine[idx] = integrate.simps(dN_dlnr[radius<inflection_radius],dx=delta_lnr) #Total N #/microm^2
            #s Fine            
            Sfine = 4*np.pi*radius[radius<inflection_radius]**2
            s_fine[idx] =  integrate.simps(dN_dlnr[radius<inflection_radius]*Sfine,dx=delta_lnr)#Total N #/microm^2                       

        #Save arrays in DATAFRAME
        df['n50'] = n50
        df['ln_n50'] = np.log(n50)
        df['s50'] = s50
        df['ln_s50'] = np.log(s50)

        df['n60'] = n60
        df['ln_n60'] = np.log(n60)
        df['s60'] = s60
        df['ln_s60'] = np.log(s60)

        df['n100'] = n100
        df['ln_n100'] = np.log(n100)
        df['s100'] = s100
        df['ln_s100'] = np.log(s100)

        df['n120'] = n120
        df['ln_n120'] = np.log(n120)
        df['s120'] = s120
        df['ln_s120'] = np.log(s120)

        df['n250'] = n250
        df['ln_n250'] = np.log(n250)
        df['s250'] = s250
        df['ln_s250'] = np.log(s250)

        df['n290'] = n290
        df['ln_n290'] = np.log(n290)
        df['s290'] = s290
        df['ln_s290'] = np.log(s290)

        df['n_coarse'] = n_coarse
        df['ln_n_coarse'] = np.log(n_coarse)
        df['s_coarse'] = s_coarse
        df['ln_s_coarse'] = np.log(s_coarse)

        df['n_fine'] = n_fine
        df['ln_n_fine'] = np.log(n_fine)
        df['s_fine'] = s_fine
        df['ln_s_fine'] = np.log(s_fine)

        df =lidar_wavelengths_all(df)
        
        #FMF
        df['FMF'] = df['AOD_Extinction-Fine[675nm]']/df['AOD_Extinction-Total[675nm]']

        #Ln applied to AOD
        waves = [355, 440, 550, 532, 675, 870, 1020, 1064]
        for wave_ in waves:
            df['lnAOD%d' % wave_] = np.log(df['AOD_Coincident_Input[%dnm]' % wave_])

        #Aerosol type classification
        df = aerosol_type_classification(df)
    return df

def plot_distribution(df, date_string, type_distribution='volume', figurepath='', xlims = [], ylims = [], clims = [], dpi = 400):
    '''
    plt_distribution() plots all the volume/surface/number size distributions on the date 'date_string'
    Input:
    df: AERONET dataframe loaded from reader_all.     
    date_string: date string in format '%Y-%m-%d'
    type_distribution (optional): type of distribution 'volume', 'surface', 'number'
    figurepath (optional): file path to save the figure. Directory must exist.
    xlims (optional): 2D-array with x-axis limits.
    ylims (optional): 2D-array with y-axis limits.
    clims (optional): 2D-array with z-axis limits.
    dpi (optional): dots-per-inch figure (higher means larger quality and file size)
    Output:
    figure handle
    axes handles
    Figure file
    '''
    
    time_condition = df['date'].values == dt.datetime.strptime(date_string, '%Y-%m-%d').date()
    
    #Find radius in df header 
    radius = np.asarray([0.050000,0.065604,0.086077,0.112939,0.148184,0.194429,0.255105,0.334716,0.439173,0.576227,0.756052,0.991996,1.301571,1.707757,2.240702,2.939966,3.857452,5.061260,6.640745,8.713145,11.432287,15.000000])                
    df_ = df.loc[time_condition] #VSD um^3/um^2
    dV_dlnr = df_.iloc[:,53:75] #VSD um^3/um^2    
    if len(dV_dlnr)>0:       
        #Plot volumne size Distribution
        SMALL_SIZE = 12
        MEDIUM_SIZE = 14
        BIGGER_SIZE = 18

        plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
        #Figure
        fig, axes = plt.subplots(1, 1)
        fig.set_figwidth(7)
        fig.set_figheight(5)       
        colors = color_list(len(dV_dlnr))
        for i_ in np.arange(len(dV_dlnr)):
            if type_distribution == 'volume':
                axes.plot(radius, dV_dlnr.iloc[i_,:], label = df_['time'][i_], c=colors[i_], linewidth=3) 
                axes.set_ylabel(r'dV/dlnr, [$\mu$$m^3$/$\mu$$m^2$]')
            elif type_distribution == 'surface':
                Vradius = (4./3.)*np.pi*np.power(radius,3) #i-bin volume
                dN_dlnr = (dV_dlnr.iloc[i_,:] / Vradius) #NSD #/um^2
                Sradius = 4.*np.pi*np.power(radius,2) #i-bin surface
                #dSlnr_dlnr = (3 * dV_dlnr.iloc[i_,:] / radius ) #SSD #um^2/um^2
                dS_dlnr = dN_dlnr * Sradius
                axes.plot(radius, dS_dlnr, label = df_['time'][i_], c=colors[i_], linewidth=3) 
                axes.set_ylabel(r'dS/dlnr, [$\mu$$m^2$/$\mu$$m^2$]')
            elif type_distribution == 'number':
                Vradius = (4./3.)*np.pi*np.power(radius,3) #i-bin volume
                dNlnr_dlnr = (dV_dlnr.iloc[i_,:] / Vradius) #NSD #/um^2
                axes.plot(radius, dNlnr_dlnr, label = df_['time'][i_], c=colors[i_], linewidth=3) 
                axes.set_ylabel(r'dN/dlnr, [#/$\mu$$m^2$]')
        axes.set_title('%s | %s' % (df_['Site'][0], str(df_['date'][0])))
        axes.set_xlabel(r'Radius, [$\mu$m]')
        axes.set_xscale('log')
        
        if len(xlims)>0:
            axes.set_xlim(xlims)
        if len(ylims)>0:
            axes.set_ylim(ylims)
        axes.legend(loc='best')
        axes.grid(axis = 'y')
        if len(figurepath)>0:
            if os.path.exists(os.path.dirname(figurepath)):
                plt.savefig(figurepath, dpi = dpi, bbox_inches = 'tight')
            else:
                print('Error: figure directory not found.')
    return fig, axes

def plot_single_scatter(df, cf_info, wavelength = 440, min_radius = 50, colorbar_property = 'Angstrom_Exponent_440-870nm_from_Coincident_Input_AOD', filter_name = '', xlims = [], ylims = [], clims = []):
    '''
    apc_scatter makes scatter plots of up to three dataframes readed with function reader_all. 
    Input:
    wavelength: AOD wavelength in nm (e.g., 440)
    minimum radius: minimum radius to integrate the NSD in nm (e.g., 50)
    colorbar_property: variable in dataframe for colorbar (e.g., 'Angstrom_Exponent_440-870nm_from_Coincident_Input_AOD' [default])
    filter_name: Allows to indicate in the figure title in the data are filtered.
    xlims: set x limits
    ylims: set y limits
    clims: set colorbar limits
    '''      

    #Define colorbar label
    colorbar_label_dict = {'Angstrom_Exponent_440-870nm_from_Coincident_Input_AOD': 'AE[440-870nm]', 
                           'Sphericity_Factor(%)': 'Sphericity Factor(%)',
                           'Depolarization_Ratio[440nm]': 'Depol. ratio[440nm]',
                           'Depolarization_Ratio[675nm]': 'Depol. ratio[675nm]',                           
                           'FMF[675nm]': 'FMF][675nm]',
                           'Lidar_Ratio[440nm]': 'Lidar ratio[440nm]',
                           'Single_Scattering_Albedo[440nm]': 'SSA[440nm]'}
    
    if not (colorbar_property in colorbar_label_dict.keys()):
        colorbar_label_dict[colorbar_property] = colorbar_property

    #Define y label
    lnAOD = 'lnAOD%d' % wavelength
    lnN = 'ln_n%d' % min_radius
    ylabel_string = '$ln(N_{%d})$' % min_radius
    xlabel_string = '$ln(AOD[%d nm])$' % wavelength    

    #Figure    
    fig,axes = plt.subplots(1, 1)
    fig.set_figwidth(10)
    fig.set_figheight(6)        

    #Scatter
    ax_mappable = axes.scatter(df[lnAOD],df[lnN],c=df[colorbar_property])

    #Linear fit
    exponent, exponent_error, intercept, intercept_error, R2 = cf_info    
    m, error_m, n, error_n, R2 = exponent, exponent_error, np.log(intercept), intercept_error/intercept, R2
    plt.plot(df[lnAOD], m*df[lnAOD]+n, 'r',label='y=(%.4f $\pm$ %.4f)x + (%.4f $\pm$ %.4f) [R=%.2f]' %(m, error_m, n, error_n, np.sqrt(R2)))

    #Axes format
    if len(filter_name)>0:            
        axes.set_title('%s | Filter: %s | data[#]: %d ' % (df['Site'][0], filter_name, len(df)))
    else:
        axes.set_title('%s | data[#]: %d ' % (df['Site'][0], len(df)))
    axes.legend()    
    axes.set_xlabel(xlabel_string)    
    axes.set_ylabel(ylabel_string)
    if len(xlims)>0:
        ax_mappable.axes.set_xlim(xlims)
    if len(ylims)>0:
        ax_mappable.axes.set_ylim(ylims)              
    if len(clims)>0:
        ax_mappable.set_clim(clims)    
    fig.colorbar(ax_mappable, label=colorbar_label_dict[colorbar_property], ax=axes)    
    fig.tight_layout()
    return fig, axes

def plot_scatter(df_tuple, x = 'lnAOD440', y = 'ln_n50', colorbar_property = 'Angstrom_Exponent_440-870nm_from_Coincident_Input_AOD', filter_name = '', xlims = [], ylims = [], clims = []):
    '''
    apc_scatter makes scatter plots of up to three dataframes readed with function reader_all. 
    Input:
    x: variable in dataframe (e.g., 'Depolarization_Ratio[440nm]')
    y: variable in dataframe (e.g., 'Lidar_Ratio[440nm]')
    colorbar_property: variable in dataframe for colorbar (e.g., 'Angstrom_Exponent_440-870nm_from_Coincident_Input_AOD' [default])
    filter_name: Allows to indicate in the figure title in the data are filtered.
    xlims: set x limits
    ylims: set y limits
    clims: set colorbar limits
    '''
    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 18

    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    #Define colorbar label
    colorbar_label_dict = {'Angstrom_Exponent_440-870nm_from_Coincident_Input_AOD': 'AE[440-870nm]', 
                           'Sphericity_Factor(%)': 'Sphericity Factor(%)',
                           'Depolarization_Ratio[440nm]': 'Depol. ratio[440nm]',
                           'Depolarization_Ratio[675nm]': 'Depol. ratio[675nm]',                           
                           'FMF[675nm]': 'FMF][675nm]',
                           'Lidar_Ratio[440nm]': 'Lidar ratio[440nm]',
                           'Single_Scattering_Albedo[440nm]': 'SSA[440nm]'}
    
    if not (colorbar_property in colorbar_label_dict.keys()):
        colorbar_label_dict[colorbar_property] = colorbar_property

    #Define y label
    ylabel_dic = {'lnN_fine': '$ln(N_{fine})$','lnN_coarse': '$ln(N_{coarse}$)', 'lnN': '$ln(N)$'}
    if not y in ylabel_dic.keys():
        ylabel_dic[y] = y

    xlabel_dic = {'lnAOD440': '$ln(AOD[440nm])$'}
    if not x in xlabel_dic.keys():
        xlabel_dic[x] = x

    #Figure    
    fig,axes = plt.subplots(1, len(df_tuple))
    fig.set_figwidth(15)
    fig.set_figheight(3)
    
    for axes_idx in np.arange(len(df_tuple)):
        df = df_tuple[axes_idx]        
        #Scatter
        ax_mappable = axes[axes_idx].scatter(df[x],df[y],c=df[colorbar_property],cmap='YlOrRd')    
        
        #Linear fit
        m, error_m, n, error_n, R2 = linrest(df[x], df[y])        
        axes[axes_idx].plot(df[x], m*df[x]+n, 'r',lw=3, label='AERONET | R=%.2f | m=%.3f$\pm$%.3f | n=%.3f$\pm$%.3f' % (np.sqrt(R2), m, error_m, n, error_n))
        #Axes format
        if len(filter_name)>0:            
            # axes[axes_idx].set_title('%s | data[#]: %d | %s\n R=%.2f | m=%.3f$\pm$%.3f | n=%.3f$\pm$%.3f' % (filter_name, len(df), df['Site'][0], np.sqrt(R2), m, error_m, n, error_n))
            axes[axes_idx].set_title('%s | data[#]: %d | %s' % (filter_name, len(df), df['Site'][0]))
        else:
            axes[axes_idx].set_title('%s' % (df['Site'][0]))
            # axes[axes_idx].set_title('%s\n R=%.2f | m=%.2f$\pm$%.2f | n=%.2f$\pm$%.2f' % (df['Site'][0], np.sqrt(R2), m, error_m, n, error_n))
            #axes[axes_idx].set_title('%s\n $R$=%.2f |alpha=%.2f | C=%.2f' % (df['Site'][0], R, m, np.exp(n)))
        axes[axes_idx].set_facecolor('gainsboro')
        axes[axes_idx].set_xlabel(xlabel_dic[x])
        axes[axes_idx].legend()
        if axes_idx == 0:
            axes[axes_idx].set_ylabel(ylabel_dic[y])

        if len(clims)>1:
            ax_mappable.set_clim(clims)
        else:            
            if axes_idx == 0:
                clims = ax_mappable.get_clim()
        if len(xlims)>0:
            ax_mappable.axes.set_xlim(xlims)
        if len(ylims)>0:
            ax_mappable.axes.set_ylim(ylims)              
        fig.colorbar(ax_mappable, label=colorbar_label_dict[colorbar_property], ax=axes[axes_idx])
    return fig, axes

    # def retrieval4CCN_riemmann(filepath):
#     '''
#     Input:
#     filepath: File path of AERONET file type *.all
#     Output:
#     df: dataframe including new variables ['lnAOD440', 'N', 'lnN', 'lnN100', 'lnN_fine', 'lnN_coarse', 'N_fine', 'N_coarse', 'FMF', 'AOD[550nm]_coincident', 'AOD[550nm]_extinction']
#     '''
#     df = reader_all(filepath)

#     if len(df) > 0:

#         #Find radius in df header 
#         radius = np.asarray([0.050000,0.065604,0.086077,0.112939,0.148184,0.194429,0.255105,0.334716,0.439173,0.576227,0.756052,0.991996,1.301571,1.707757,2.240702,2.939966,3.857452,5.061260,6.640745,8.713145,11.432287,15.000000])

#         dV_dlnr = df.iloc[:,np.arange(53,75,1)  ] #VSD um^3/um^2
#         dV_dlnr = dV_dlnr.to_numpy()

#         Vradius = (4./3.)*np.pi*np.power(radius,3) #i-bin volume
#         delta_lnr = np.diff(np.log(radius)).mean() #resolution
#         dN_dlnr_serie = (dV_dlnr / Vradius) #NSD #/um^2
#         n50 = np.zeros(len(df))
#         s50 = np.zeros(len(df))
#         n60 = np.zeros(len(df))
#         s60 = np.zeros(len(df))
#         n100 = np.zeros(len(df))
#         n120 = np.zeros(len(df))
#         s100 = np.zeros(len(df))
#         s120 = np.zeros(len(df))
#         n250 = np.zeros(len(df))
#         s250 = np.zeros(len(df))
#         n290 = np.zeros(len(df))
#         s290 = np.zeros(len(df))        
#         n500 = np.zeros(len(df))
#         s500 = np.zeros(len(df))        
#         n_coarse = np.zeros(len(df))
#         s_coarse = np.zeros(len(df))
#         n_fine = np.zeros(len(df))
#         s_fine = np.zeros(len(df))        
#         for idx in np.arange(len(df)):
#             #Increase resolution to improve the fitting                  
#             dN_dlnr = dN_dlnr_serie[idx,:]

#             #n50
#             n50[idx] = np.sum(dN_dlnr[radius>=0.05]*delta_lnr)#Total N #/microm^2                        
#             #s50
#             S50 = 4*np.pi*radius[radius>=0.05]**2
#             s50[idx] = np.sum(dN_dlnr[radius>=0.05]*S50*delta_lnr)#Total N #/microm^2

#             #N60        
#             n60[idx] = np.sum(dN_dlnr[radius>=0.06]*delta_lnr)#Total N #/microm^2
#             #s60
#             S60 = 4*np.pi*radius[radius>=0.06]**2
#             s60[idx] = np.sum(dN_dlnr[radius>=0.06]*S60*delta_lnr)#Total N #/microm^2

#             #N100
#             n100[idx] = np.sum(dN_dlnr[radius>=0.1]*delta_lnr)#Total N #/microm^2
#             #s100
#             S100 = 4*np.pi*radius[radius>=0.1]**2
#             s100[idx] = np.sum(dN_dlnr[radius>=0.1]*S100*delta_lnr)#Total N #/microm^2

#             #N120
#             n120[idx] = np.sum(dN_dlnr[radius>=0.12]*delta_lnr)#Total N #/microm^2
#             #s120
#             S120 = 4*np.pi*radius[radius>=0.12]**2
#             s120[idx] = np.sum(dN_dlnr[radius>=0.12]*S120*delta_lnr)#Total N #/microm^2

#             #N250
#             n250[idx] = np.sum(dN_dlnr[radius>=0.25]*delta_lnr)#Total N #/microm^2
#             #s250
#             S250 = 4*np.pi*radius[radius>=0.25]**2
#             s250[idx] = np.sum(dN_dlnr[radius>=0.25]*S250*delta_lnr)#Total N #/microm^2

#             #N290
#             n290[idx] = np.sum(dN_dlnr[radius>=0.29]*delta_lnr)#Total N #/microm^2

#             #s290
#             S290 = 4*np.pi*radius[radius>=0.29]**2
#             s290[idx] = np.sum(dN_dlnr[radius>=0.29]*S290*delta_lnr)#Total N #/microm^2

#             #N500
#             n500[idx] = np.sum(dN_dlnr[radius>=0.5]*delta_lnr)#Total N #/microm^2

#             #s500
#             S500 = 4*np.pi*radius[radius>=0.5]**2
#             s500[idx] = np.sum(dN_dlnr[radius>=0.5]*S500*delta_lnr)#Total N #/microm^2

#             #n Coarse    
#             inflection_radius = df['Inflection_Radius_of_Size_Distribution(um)'].iloc[idx]     
#             n_coarse[idx] = np.sum(dN_dlnr[radius>=inflection_radius]*delta_lnr)#Total N #/microm^2
#             #s Coarse            
#             Scoarse = 4*np.pi*radius[radius>=inflection_radius]**2
#             s_coarse[idx] = np.sum(dN_dlnr[radius>=inflection_radius]*Scoarse*delta_lnr)#Total N #/microm^2                       

#             #Fine            
#             n_fine[idx] = np.sum(dN_dlnr[radius<inflection_radius]*delta_lnr)#Total N #/microm^2
#             #s Fine            
#             Sfine = 4*np.pi*radius[radius<inflection_radius]**2
#             s_fine[idx] = np.sum(dN_dlnr[radius<inflection_radius]*Sfine*delta_lnr)#Total N #/microm^2                       

#         #Save arrays in DATAFRAME
#         df['n50'] = n50
#         df['ln_n50'] = np.log(n50)
#         df['s50'] = s50
#         df['ln_s50'] = np.log(s50)

#         df['n60'] = n60
#         df['ln_n60'] = np.log(n60)
#         df['s60'] = s60
#         df['ln_s60'] = np.log(s60)

#         df['n100'] = n100
#         df['ln_n100'] = np.log(n100)
#         df['s100'] = s100
#         df['ln_s100'] = np.log(s100)

#         df['n120'] = n120
#         df['ln_n120'] = np.log(n120)
#         df['s120'] = s120
#         df['ln_s120'] = np.log(s120)

#         df['n250'] = n250
#         df['ln_n250'] = np.log(n250)
#         df['s250'] = s250
#         df['ln_s250'] = np.log(s250)

#         df['n290'] = n290
#         df['ln_n290'] = np.log(n290)
#         df['s290'] = s290
#         df['ln_s290'] = np.log(s290)

#         df['n_coarse'] = n_coarse
#         df['ln_n_coarse'] = np.log(n_coarse)
#         df['s_coarse'] = s_coarse
#         df['ln_s_coarse'] = np.log(s_coarse)

#         df['n_fine'] = n_fine
#         df['ln_n_fine'] = np.log(n_fine)
#         df['s_fine'] = s_fine
#         df['ln_s_fine'] = np.log(s_fine)

#         #AOD @ 355, 550, 532, 1064 nm
#         df['AOD_Coincident_Input[355nm]'] = df['AOD_Coincident_Input[440nm]'].values*(355/440)**(-df['Angstrom_Exponent_440-870nm_from_Coincident_Input_AOD'].values)
#         df['AOD_Coincident_Input[550nm]'] = df['AOD_Coincident_Input[675nm]'].values*(550./675.)**(-df['Angstrom_Exponent_440-870nm_from_Coincident_Input_AOD'].values)
#         df['AOD_Coincident_Input[532nm]'] = df['AOD[550nm]_coincident'].values*(532/550)**(-df['Angstrom_Exponent_440-870nm_from_Coincident_Input_AOD'].values)
#         df['AOD[550nm]_extinction'] = df['AOD_Coincident_Input[675nm]'].values*(550./675.)**(-df['Extinction_Angstrom_Exponent_440-870nm-Total'].values)
        
#         df['Angstrom_Exponent_675-1020nm_from_Coincident_Input_AOD'] = - (np.log(df['AOD_Coincident_Input[675nm]'].values/df['AOD_Coincident_Input[1020nm]'].values))/(np.log(675/1020))
#         df['AOD_Coincident_Input[1064nm]'] = df['AOD_Coincident_Input[1020nm]'].values*(1064/1020)**(-df['Angstrom_Exponent_675-1020nm_from_Coincident_Input_AOD'].values)

#         #FMF
#         df['FMF'] = df['AOD_Extinction-Fine[675nm]']/df['AOD_Extinction-Total[675nm]']

#         #Ln applied to AOD
#         waves = [355, 440, 550, 532, 675, 870, 1020, 1064]
#         for wave_ in waves:
#             df['lnAOD%d' % wave_] = np.log(df['AOD_Coincident_Input[%dnm]' % wave_])
#     return df
