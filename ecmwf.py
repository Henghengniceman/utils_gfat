#!/usr/bin/env python

import os
import sys
import pandas as pd
import numpy as np
import xarray as xr
import datetime as dt
from scipy import integrate

import pdb

MODULE_DIR = os.path.dirname(sys.modules[__name__].__file__)
sys.path.insert(0, MODULE_DIR)
import logs
import config

logger = logs.create_logger()

def meteo_ecmwf(date_ini_dt, date_end_dt, ranges, ecmwf_dn='GFATserver'):
    """

    Parameters
    ----------
    date_ini_dt: datetime
    date_end_dt: datetime
    ranges: array
        lidar ranges
    ecmwf_dn: str
        ecmwf data dir

    Returns
    -------

    """
    # pdb.set_trace()
    def nearest(items, pivot):
        """
        items: numpy.datetime64
        pivot: numpy.datetime64
        """
        return min(items, key=lambda x: abs(x - pivot))

    # ECMWF Data Directory:
    if ecmwf_dn == 'GFATserver':
        ecmwf_dn = os.path.join(config.DATA_DN, "ECMWF")

    # Read Necessary ecmwf data: loop over days
    # Expected: ecmwf_dn/yyyy/yyyymmdd_granada_ecmwf.nc
    date_days = pd.date_range(date_ini_dt.date(), date_end_dt.date(), freq='D')
    ecmwf_ds = None
    for _day in date_days:
        ecmwf_fn = os.path.join(ecmwf_dn, _day.strftime('%Y'),
                                "%s_%s_ecmwf.nc" % (_day.strftime('%Y%m%d'), "granada"))
        if os.path.exists(ecmwf_fn):
            with xr.open_dataset(ecmwf_fn) as ds:
                ds = ds[['height', 'temperature', 'pressure']]
                ds = ds.transpose()
            if ecmwf_ds is None:
                ecmwf_ds = ds
            else:
                ecmwf_ds = xr.concat([ecmwf_ds, ds], 'time')

    if ecmwf_ds is not None:
        # Extract Representative Profile
        # Time Interpolation (LINEAR)
        option = 'jaba'    
        if option == 'dbp': 
            ecmwf_ds = ecmwf_ds.sel(time=slice((date_ini_dt - dt.timedelta(minutes=30)).strftime("%Y%m%dT%H:%M:%S.0"),
                (date_end_dt + dt.timedelta(minutes=30)).strftime("%Y%m%dT%H:%M:%S.0")))
            ecmwf_avg = ecmwf_ds.interp(time=date_ini_dt + (date_end_dt - date_ini_dt)/2)
        elif option == 'jaba':
            ndate1 = nearest(ecmwf_ds.time.values, np.datetime64(date_ini_dt))
            ndate2 = nearest(ecmwf_ds.time.values, np.datetime64(date_end_dt))
            ecmwf_avg = ecmwf_ds.sel(time=slice(ndate1, ndate2)).mean(dim='time')
            
        ecmwf_int = xr.Dataset({'temperature': (['range'], ecmwf_avg.temperature.values),
                                'pressure': (['range'], ecmwf_avg.pressure.values)},
                               coords={'range': ecmwf_avg.height.values})
        # Height Interpolation (LINEAR)
        ecmwf_int = ecmwf_int.interp(range=ranges, kwargs={'fill_value': 'extrapolate'})

        # Set attributes
        ecmwf_int['pressure'].attrs['units'] = 'Pa'
        ecmwf_int['pressure'].attrs['long_name'] = 'Pressure'
        ecmwf_int['pressure'].attrs['standard_name'] = 'air_pressure'
        ecmwf_int['temperature'].attrs['units'] = 'K'
        ecmwf_int['temperature'].attrs['long_name'] = 'Temperature'
        ecmwf_int['temperature'].attrs['standard_name'] = 'air_temperature'
        ecmwf_int['range'].attrs['units'] = 'm'
        ecmwf_int['range'].attrs['long_name'] = 'Height above ground'
        ecmwf_int['range'].attrs['standard_name'] = 'height'
    else:
        ecmwf_int = None
        logger.warning("Error. No ecmwf data")

    return ecmwf_int


def nearest_datetime(items, pivot):
        return min(items, key=lambda x: abs(x - pivot))

def reader(filepath):    
    if os.path.isfile(filepath):
        ecmwf = xr.open_dataset(filepath)
    else:      
        ecmwf = []  
        print('ECMWF file not found.')
    return ecmwf

def level2height(ecmwf_profile, range_array):
    '''
    Exchange level by height and interpolate height to the given range array.    
    '''
    if not 'time' in ecmwf_profile.dims.keys():
        ecmwf_profile_ = ecmwf_profile.swap_dims({'level': 'height'})
        pressure_prf = ecmwf_profile_['pressure'].interp({'height': range_array})
        temperature_prf = ecmwf_profile_['temperature'].interp({'height': range_array})
        try:
            pressure_prf = pressure_prf.interpolate_na(dim='height', method="linear", fill_value="extrapolate")
            temperature_prf = temperature_prf.interpolate_na(dim='height', method="linear", fill_value="extrapolate")
            pressure_prf.rename({'height':'range'})
            temperature_prf.rename({'height':'range'})
        except:
            pressure_prf = pressure_prf.interpolate_na(dim='range', method="linear", fill_value="extrapolate")
            temperature_prf = temperature_prf.interpolate_na(dim='range', method="linear", fill_value="extrapolate")
    else:
        print('Remove time before applying level2range')
        ecmwf_profile = []
    return pressure_prf, temperature_prf

def to_profile(ecmwf, ini_date, end_date, ymin, ymax, interpol_resolution=7.5):    
    '''
    Input:
    ecmwf: xarray.dataset from reader. File type: ECMWF from CLOUDNET.
    Output:
    P, T: profiles (Pa, K)
    '''    

    if ecmwf:
        ini_date_dt = pd.to_datetime(ini_date, format='%Y%m%dT%H%M%S').to_numpy()
        end_date_dt = pd.to_datetime(end_date, format='%Y%m%dT%H%M%S').to_numpy()
        if np.logical_or(ini_date_dt == end_date_dt, (end_date_dt - ini_date_dt) < np.timedelta64(1,'h')):            
            nearest_date = nearest_datetime(ecmwf['time'], ini_date_dt)            
            mprof = ecmwf.sel(time=slice(nearest_date, nearest_date)).mean('time')
        else:
            mprof = ecmwf.sel(time=slice(ini_date, end_date)).mean('time')
        
        if mprof:
            level_max = mprof['level'][np.squeeze(np.abs(mprof['height']-ymax) == np.min(np.abs(mprof['height']-ymax)))].values[0]
            mprof = mprof.sel(level=slice(137,level_max)).swap_dims({'level': 'height'})
            if interpol_resolution != None:
                height = np.arange(ymin,ymax, interpol_resolution)
                improf = mprof.interp(height=height).interpolate_na(dim='height', method="linear", fill_value="extrapolate").rename_dims({'height': 'range'})
            else: 
                improf = mprof
            T = improf['temperature']
            P = improf['pressure']
        else:
            P, T = None, None
            print('No data in the given period.')
    else:
        P, T = None, None
        print('ECMWF xarray.Dataset not supported.')
    return P, T