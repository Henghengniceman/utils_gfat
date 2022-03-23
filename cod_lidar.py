# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 19:45:12 2020

@author: Marta
"""

import os
import sys
import gzip
import shutil
import tempfile
import math
import glob
import scipy
import sympy
from sympy.abc import x
import datetime as dt
import xarray as xr
import numpy as np
from scipy import stats
from scipy.signal import correlate
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
# import radar_calibration as rc
# sys.path.append('G:\Mi unidad\03.PROGRAMAS\utils_gfat')
# from readers import backscatter_lidar
sys.path.append(r'G:\Mi unidad\03.PROGRAMAS')
#sys.path.append(r'/home/mopsmap/Escritorio/Marta/03.PROGRAMAS')
import lidar
sys.path.append(r'G:\Mi unidad\03.PROGRAMAS\utils_gfat\lidar_processing')
#sys.path.append(r'/home/mopsmap/Escritorio/Marta/03.PROGRAMAS/utils_gfat/lidar_processing')
import atmo
#import lidar
import netCDF4 as nc
import glob
from scipy.signal import savgol_filter


__author__ = "Jiménez-Martín, Marta María"
__license__ = "GPL"
__version__ = "1.1"
__maintainer__ = "Jiménez-Martín, Marta María"
__email__ = "mmjimenez@ugr.es"
__status__ = "Production"


def COD_calculation(maindir, day, month, year, number_of_channel):
    """
    Function to calculate the Cloud Optical Depth (COD) and cloud base level (CBH) in terms of LIDAR data.
    Input :
    LIDAR file (mhc_1a_Prs_*.nc)
    Output:  
    COD and CBH data string for one specific day, by periods of 10 minutes.
    """

    # 7.Se cargan los datos del LIDAR
    # maindir_LIDAR = '/home/mopsmap/Escritorio/Marta/00.INVESTIGACION/DATOS/MULHACEN/1a/2020/01/15'
    #             maindir_LIDAR = os.path.join(maindir_root,Time_month[0:4], Time_month[5:7], str(_day))
    LIDARfile = 'mhc_1a_Prs_*.nc'
    LIDARfileDC = 'mhc_1a_Pdc_*.nc'
    filedir = os.path.join(maindir, year, month, day)
    LIDARpath = os.path.join(filedir, LIDARfile)
    LIDARpathDC = os.path.join(filedir, LIDARfileDC)
    
    hours = np.arange(0,24)
    minutes = ['00','10','20','30','40','50']
    
    Time_month = year + '-' + month

    # Corregir medidas LIDAR con la corriente oscura
    
    if (os.path.isdir(filedir)==True):

        correct_LIDAR = lidar.preprocessing(LIDARpath, LIDARpathDC)
        print(correct_LIDAR)
        print('Longitudes de onda:', correct_LIDAR['wavelength'].values)

        COD_serie = []
        CBH_serie = []

        for j,_hour in enumerate(hours):
            for i,_minute in enumerate(minutes[0:len(minutes)-1]):

                #set height range in meters
                zref1 = 0
                zref2 = 20000

                startTime = Time_month + '-' + day + ' ' + str(hours[j]) + ':' + minutes[i]
                stopTime = Time_month + '-' + day + ' ' + str(hours[j]) + ':' + minutes[i+1]

                initime = pd.to_datetime(startTime,format='%Y-%m-%d %H:%M')
                endtime = pd.to_datetime(stopTime,format='%Y-%m-%d %H:%M')
    #             signal_to_profile = 'correct_LIDAR.corrected_rcs_' + number_of_channel
                signal_to_profile = 'corrected_rcs_' + number_of_channel
                profile = correct_LIDAR[signal_to_profile].sel(time=slice(initime, endtime), range=slice(zref1, zref2))

                if (str(profile.mean().values) == 'nan'):

                    COD = np.nan;
                    zref1 = zref2 = np.nan
    #                 print(COD)

    #                 f_CBH.write(str(zref1) + '\n')
    #                 f_COD.write(str(COD) + '\n')

                    COD_serie = np.append(COD_serie,COD)
                    CBH_serie = np.append(CBH_serie,zref1)

                else:

                    # Alturas sobre el nivel del mar del LIDAR

                    heights = profile.range.values + 680

                    # Parámetros de atmósfera standard

                    T = np.ones(heights.size)*np.nan
                    P = np.ones(heights.size)*np.nan
                    for i,_height in enumerate(heights):
                        sa = atmo.standard_atmosphere(_height)
                        P[i]  = sa[0]
                        T[i]  = sa[1]

                    # Alpha molecular y extinción teóricas

                    betamol = np.ones(heights.size)*np.nan
                    alfamol = np.ones(heights.size)*np.nan
                    for i,_height in enumerate(heights):
                        betamol[i] = atmo.molecular_backscatter(532, P[i], T[i])  
                        alfamol[i] = atmo.molecular_extinction(532, P[i], T[i])  


                    # Transmitancia: Integral de la definición por la resolución en altura.

                    transmittance = np.exp(-2*scipy.integrate.cumtrapz(alfamol, initial = alfamol[0])*(heights[1] - heights[0]))

                    # Cumptrapz
                    # scipy.integrate.cumtrapz(y, x=None, dx=1.0, axis=-1, initial=None)
                    # Beta molecular atenuada teórica

                    beta_molecular_att = betamol*np.resize(transmittance, heights.shape) 

                    zref1 = 3000.
                    zref2 = 12000.

                    mprofile = profile.mean(axis=0)
                    smprofile = savgol_filter(mprofile, 51, 3) # window size 51, polynomial order 3


                    # Cálculo de los límites de la nube. Implementación automática de los límites de integración

                    stat_diff = [];
                    for i,_height in enumerate(heights):        
                        # Antes de llegar a la nube
                        idx = np.logical_and(heights >= heights[i], heights <= heights[i] + 500)   
                        stat_beta=scipy.stats.linregress(heights[idx],beta_molecular_att[idx])
                        stat_mprofile=scipy.stats.linregress(heights[idx],mprofile[idx])
                    #     print('Altura:',heights[i])
                    #     print('Pendiente b_att:',stat_beta.slope)
                    #     print('Pendiente R_fit:',stat_rayleigh.slope)
                        stat_diff = np.append(stat_diff,(stat_beta.slope - stat_mprofile.slope))

                    stat_diff_sel_idx = np.logical_and(heights>=zref1, heights<=zref2)
                    stat_diff_sel = stat_diff[stat_diff_sel_idx]

                    der_vector = []
                    cloud_range = []
                    for i,_ejemplo in enumerate(stat_diff_sel[0:(len(stat_diff_sel)-1)]):
            #             der_vector = np.append(der_vector,stat_diff_sel[i]-stat_diff_sel.mean())
                        der_vector = np.append(der_vector,stat_diff_sel[i]-np.nanmedian(stat_diff_sel))
                        der = stat_diff_sel[i]-np.nanmedian(stat_diff_sel)
                        if (math.fabs(der) >= 2000):
                            cloud_range = np.append(cloud_range, heights[stat_diff_sel_idx][i])

                    if (cloud_range == []):

                        COD = np.nan;
                        zref1 = zref2 = np.nan
    #                     print(COD)

                        COD_serie = np.append(COD_serie,COD)
                        CBH_serie = np.append(CBH_serie,zref1)

                    else:

                        zref1 = cloud_range[0]
                        zref2 = cloud_range[len(cloud_range)-1]

        #                         plt.figure(figsize=[15,10])
        #                         plt.plot(der_vector)


        #                         print('CBH:',zref1)
        #                         print('CTH:',zref2)
        #                         print('STD:',np.std(stat_diff_sel))

                        # Antes de llegar a la nube
                        idx_bf = np.logical_and(heights >= (zref1 - 500), heights <= zref1)
                        beta_molecular_att_bf = beta_molecular_att[idx_bf].mean()
                        profile_rayleigh_bf = smprofile[idx_bf].mean()
                        ratio_bf  = beta_molecular_att_bf/profile_rayleigh_bf
                        rayleigh_fit_bf = smprofile*ratio_bf

                        # Después de pasar la nube
                        idx_af = np.logical_and(heights >= (zref2), heights <= zref2 + 500)
                        beta_molecular_att_af = beta_molecular_att[idx_af].mean()
                        profile_rayleigh_af = smprofile[idx_af].mean()
                        ratio_af = beta_molecular_att_af/profile_rayleigh_af
                        rayleigh_fit_af = smprofile*ratio_af

                        COD = 2*np.log(ratio_af/ratio_bf)

                        if (COD<0):
                            COD = np.nan
    #                         print(COD)

                        COD_serie = np.append(COD_serie,COD)
                        CBH_serie = np.append(CBH_serie,zref1)
    else:
        print('File not found:',day+month+year)
        COD_serie = []
        CBH_serie = []
    
    
    return COD_serie, CBH_serie




def COD_calculation_monthly(maindir, month, year, number_of_channel):
    """
    Function to calculate monthly COD and monthly CBH series in terms of LIDAR data.
    Input :
    LIDAR file (mhc_1a_Prs_*.nc)
    Output:  
    COD and CBH data string by periods of 10 minutes.
    """
    
    COD_monthly_serie = []
    CBH_monthly_serie = []
    
    day = np.arange(1,32)

    for _day in day:
        
        if (_day<10):
            
            _day = '0' + str(_day)
            COD_day = []
            CBH_day = []
            COD_day, CBH_day = COD_calculation(maindir, _day, month, year, number_of_channel)
            COD_monthly_serie = np.append(COD_monthly_serie,COD_day)
            CBH_monthly_serie = np.append(CBH_monthly_serie,CBH_day)
            print('Done:',_day)
            
        elif (_day>=10):
            
            _day = str(_day)
            COD_day = []
            CBH_day = []
            COD_day, CBH_day = COD_calculation(maindir, _day, month, year, number_of_channel)
            COD_monthly_serie = np.append(COD_monthly_serie,COD_day)
            CBH_monthly_serie = np.append(CBH_monthly_serie,CBH_day)
            print('Done:',_day)
            
    return COD_monthly_serie, CBH_monthly_serie


def date_COD_calculation(maindir, dateini, dateend, number_of_channel):
    """
    Function to calculate COD and CBH series in a specific period in terms of LIDAR data.
    Inputs:
    - maindir: path where 1a-level LIDAR data are located.
    - Initial date [yyyy-mm-dd] (str). 
    - Final date [yyyy-mm-dd] (str).
        
    Outputs: 
    - COD and CBH data string by periods of 10 minutes.
    """        

    if maindir == 'GFATserver':
        maindir = '/mnt/NASGFAT/datos/RPG-HATPRO/Data'       

    inidate = dt.datetime.strptime(dateini, '%Y%m%d')
    enddate = dt.datetime.strptime(dateend, '%Y%m%d')

    period = enddate - inidate
    
    COD_date_serie = []
    CBH_date_serie = []
    
    for _day in range(period.days + 1):
        current_date = inidate + dt.timedelta(days=_day)  
        COD_day = []
        CBH_day = []
        COD_day, CBH_day = COD_calculation(maindir,'%02d' % current_date.day, '%02d' % current_date.month, '%d' % current_date.year, number_of_channel)
        COD_date_serie = np.append(COD_date_serie,COD_day)
        CBH_date_serie = np.append(CBH_date_serie,CBH_day)
        print('Done:',current_date.day+current_date.month+current_date.year)
        
    return COD_date_serie, CBH_date_serie

