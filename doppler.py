#!/usr/bin/env python

import os
import sys
import glob
import argparse
import matplotlib
import scipy as sp
import numpy as np
import pandas as pd
import xarray as xr
import scipy.ndimage
import netCDF4 as nc
import datetime as dt
import plot
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from distutils.dir_util import mkpath
from matplotlib.dates import DateFormatter

__author__ = "Bravo-Aranda, Juan Antonio"
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Bravo-Aranda, Juan Antonio"
__email__ = "jabravo@ugr.es"
__status__ = "Production"


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
