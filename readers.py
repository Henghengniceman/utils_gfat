#!/usr/bin/env python

"""
function for reading files of the GFAT instrumentation

noaaRT: line 17
backscatter_lidar: line 38
doppler_lidar: line 98
rpg: line 176
cimel: 208
ceilometer: 221
disdrometer:

"""
import os
import sys
import glob
import xarray as xr
import numpy as np
import pandas as pd
import netCDF4 as nc
import datetime as dt

__author__ = "Bravo-Aranda, Juan Antonio"
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Bravo-Aranda, Juan Antonio"
__email__ = "jabravo@ugr.es"
__status__ = "Production"

def noaaRT(filepath):
    """This program can be used for reading NOAA-RT files. Example:
       readers.noaaRT(Y:\\datos\\IN-SITU\\Data\\noaaRT\\20191801.csv)
       Output is in pandas format
    """
    # Reading file
    # ----------------------------------------------------------------------------
    if os.path.isfile(filepath):
        dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M')
        df=pd.read_csv(filepath, names=['DateTime', 'pm25', 'eb'], date_parser=dateparse, index_col='DateTime')
    else:
        df=''
        print('File not found!')
    
    # cleaning file
    # ----------------------------------------------------------------------------
    for key in df.keys():
        df[key][df[key]>9000.]=np.nan        
    return df

def backscatter_lidar(list_files, date, channel):
    """read data from netCDF files

    """
    print('Reading netCDF files')

    print(list_files)

    data = {}

    # open all files
    nc_ids = [nc.Dataset(file_) for file_ in list_files]

    # localization of instrument
    data['lat'] = nc_ids[0].variables['lat'][:]
    data['lon'] = nc_ids[0].variables['lon'][:]
    data['alt'] = nc_ids[0].variables['altitude'][:]
    data['location'] = nc_ids[0].site_location.split(',')[0]
    data['instr'] = nc_ids[0].system

    # read alt (no need to concantenate)
    data['range'] = nc_ids[0].variables['range'][:]

    # wavelength (no need to concantenate)
    tmp = nc_ids[0].variables['wavelength'][:]
    data['wavelength'] = tmp 
    data['wavelength_units'] = nc_ids[0].variables['wavelength'].units  

    # detection_mode (no need to concantenate)
    tmp = nc_ids[0].variables['detection_mode'][:]
    data['detection_mode'] = tmp

    # polarization (no need to concantenate)
    tmp = nc_ids[0].variables['polarization'][:]
    data['polarization'] = tmp

    # read time 
    units = nc_ids[0].variables['time'].units #'days since %s' % dt.datetime.strftime(date, '%Y-%m-%d %H:%M:%S')
    tmp = [nc.num2date(nc_id.variables['time'][:], units) for nc_id in nc_ids]
    data['raw_time'] = np.concatenate(tmp)

    # check if any data available
    print(date)
    time_filter = (data['raw_time'] >= date) & (data['raw_time'] < date + dt.timedelta(days=1))
    if not np.any(time_filter):
        print('No data in user-defined time.')
        return None

    # RCS
    tmp = [nc_id.variables['rcs_%02d' % channel][:] for nc_id in nc_ids]
    data['rcs'] = np.ma.filled(np.concatenate(tmp, axis=0))    

    # Background
    tmp = [nc_id.variables['bckgrd_rcs_%02d' % channel][:] for nc_id in nc_ids]
    data['background'] = np.ma.filled(np.concatenate(tmp, axis=0), fill_value=np.nan)       
    
    # close all files
    [nc_id.close() for nc_id in nc_ids]

    return data

def doppler_lidar(list_files, date):
    """read data from netCDF files of the Doppler lidar
    
    """

    def mean_data_2d(time, range_, var, freq='1H'):
        """mean 2d data along time"""

        # convert numpy array to dataframe
        n_time = time.size
        n_range = range_.size

        df = pd.DataFrame(var.flatten(),
                        index=[np.repeat(time, n_range), np.tile(range_, n_time)])

        # mean data along time
        mean_var = df.unstack(level=1).resample(freq).mean()
        mean_time = mean_var.index.to_pydatetime()

        return mean_time, mean_var

    data = {}

    # open all files
    nc_ids = [nc.Dataset(file_) for file_ in list_files]

    # localization of instrument
    data['lat'] = nc_ids[0].variables['latitude'][:]
    data['lon'] = nc_ids[0].variables['longitude'][:]
    data['alt'] = nc_ids[0].variables['altitude'][:]
    data['instr'] = nc_ids[0].system.split()[1]

    # read alt (no need to concantenate)
    data['range'] = nc_ids[0].variables['height'][:]

    # read time
    units = 'hours since %s' % dt.datetime.strftime(date, '%Y-%m-%d %H:%M:%S')
    tmp = [nc.num2date(nc_id.variables['time'][:], units) for nc_id in nc_ids]
    data['raw_time'] = np.concatenate(tmp)

    # check if any data available
    time_filter = (data['raw_time'] >= date) & (data['raw_time'] < date + dt.timedelta(days=1))
    if not np.any(time_filter):
        print('No data in user-defined time.')
        return None

    # data availability
#     tmp = [nc_id.variables['data_availability'][:] for nc_id in nc_ids]
#     data['avail'] = np.concatenate(tmp, axis=0)
#     avail_filter = data['avail'] < LIMIT_AVAIL

    # U
    tmp = [nc_id.variables['u'][:] for nc_id in nc_ids]
    data['u'] = np.ma.filled(np.concatenate(tmp, axis=0))
    data['time'], data['u'] = mean_data_2d(data['raw_time'], data['range'], data['u'])

    # Wind speed error
    tmp = [nc_id.variables['wind_speed_error'][:] for nc_id in nc_ids]
    data['wind_speed_error'] = np.ma.filled(np.concatenate(tmp, axis=0), fill_value=np.nan)   
    data['time'], data['wind_speed_error'] = mean_data_2d(data['raw_time'], data['range'], data['wind_speed_error'])
    
    # V
    tmp = [nc_id.variables['v'][:] for nc_id in nc_ids]
    data['v'] = np.ma.filled(np.concatenate(tmp, axis=0))
    data['time'], data['v'] = mean_data_2d(data['raw_time'], data['range'], data['v'])

    # W
    tmp = [nc_id.variables['w'][:] for nc_id in nc_ids]
    data['w'] = np.ma.filled(np.concatenate(tmp, axis=0), fill_value=np.nan)

    # Vertical Wind speed error
    tmp = [nc_id.variables['w_error'][:] for nc_id in nc_ids]
    data['w_error'] = np.ma.filled(np.concatenate(tmp, axis=0), fill_value=np.nan)       

    # close all files
    [nc_id.close() for nc_id in nc_ids]

    return data    

def rpg(mainPath, radarNick, stationNick, datestr):
    #Date format conversion
    datestr = dt.datetime.strptime(datestr, '%Y%m%d')
    date2name = dt.datetime.strftime(datestr, '%y%m%d')
    year = dt.datetime.strftime(datestr, '%Y')
    month = dt.datetime.strftime(datestr, '%m')
    day = dt.datetime.strftime(datestr, '%d')    
    fileName='%s_%s_%s*_ZEN_compact.nc' % (radarNick, stationNick, date2name)    
    print(fileName)
    filePath = os.path.join(mainPath, fileName)
    print('Filepath: %s' % filePath)
    fileList = glob.glob(filePath)
    control = 1    
    rpg = []
    print(fileList)
    for tempFilePath in fileList:
        temp_rpg = xr.open_dataset(tempFilePath)       
        #Round values to avoid conflicts during merging
        temp_rpg['Lat'] = np.round(temp_rpg.Lat,2)
        temp_rpg['Lon'] = np.round(temp_rpg.Lon,2)
        if control:
            rpg = temp_rpg            
            control = 0
        else:                         
            rpg = xr.merge([rpg, temp_rpg])            
        rpg.attrs = temp_rpg.attrs

        #Encoding time as python datetime
        rpg['time'].attrs['units'] = 'seconds since 2000-01-01'
        xr.decode_cf(rpg)
        temp_rpg.close()
    return rpg    

def cimel(filename):
    """
    Reader for cimel files type *.lev15.
    Output in pandas format.      
    """    
    if os.path.isfile(filename):
        dateparse = lambda x: pd.datetime.strptime(x, '%d:%m:%Y %H:%M:%S')
        data = pd.read_csv(filename, parse_dates={'datetime': ['Date(dd-mm-yy)', 'Time(hh:mm:ss)']}, date_parser=dateparse, index_col='datetime')
    else:
        data = None
        print('File not found!')
    return data

def ceilometer(filename):
    a=0
    return a

def disdrometer(filename):
    a=0
    return a

