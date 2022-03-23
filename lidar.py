#!/usr/bin/env python
import os
import sys
import glob
import platform
from distutils.dir_util import mkpath
import argparse
import warnings
import collections
import netCDF4 as nc
import numpy as np
import xarray as xr
import dask.array as da
import datetime as dt
import matplotlib
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import matplotlib.pyplot as plt
import pdb
import time

warnings.filterwarnings("ignore")

""" import utils_gfat modules """
MODULE_DIR = os.path.dirname(sys.modules[__name__].__file__)
sys.path.insert(0, MODULE_DIR)
import logs
import utils
import plot
import solar
from lidar_trigger_delay import get_bin_zero
from lidar_dead_time import get_dead_time
from lidar_preprocessing import *
from utils_gfat import config

__author__ = "Bravo-Aranda, Juan Antonio"
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Bravo-Aranda, Juan Antonio"
__email__ = "jabravo@ugr.es"
__status__ = "Production"

logger = logs.create_logger()

""" DEFAULT AUXILIAR INFO
"""
# Root Directory (in NASGFAT)  according to operative system
DATA_DN = config.DATA_DN

#this is some change

# LIDAR SYSTEM INFO
# Correspondence with raw2l1
LIDAR_SYSTEM = {
    'MULHACEN': {
        'LIDARNICK': 'mhc',
        'CHANNELS': ['532xpa', '532xpp', '532xsa', '532xsp', 
                     '355xta', '355xtp', '1064xta', '607xtp', 
                     '387xtp', '408xtp'],
        'MODULES': ['xf']
    },
    'VELETA': {
        'LIDARNICK': 'vlt',
        'CHANNELS': [''],
        'MODULES': ['xf']
        },
    'KASCAL': {
        'LIDARNICK': 'kal',
        'CHANNELS': [''],
        'MODULES': ['xf']
        },
    'ALHAMBRA': {
        'LIDARNICK': 'alh',
        'CHANNELS': ['1064fta', '1061fta',  
                     '532fta',  '532ftp',  
                     '531fta',  '531ftp',  
                     '355fpa',  '355fpp',  '355fsa',  '355fsp',  
                     '354fta',  '354ftp',  
                     '408fta',  '408ftp',  
                     '1064nta', '1064nta',
                     '532npa',  '532npa',  '532npp', '532npp',
                     '532nsa',  '532nsa',  '532nsp', '532nsp',
                     '355npa',  '355npa',  '355npp', '355npp',
                     '355nsa',  '355nsa',  '355nsp', '355nsp',
                     '387nta',  '387nta',  '387ntp', '387ntp',
                     '607nta',  '607nta'],
        'MODULES': ['nf', 'ff']
    }
}


""" READ DATA 
"""
def reader_netcdf(filelist, channels):
    """
    Lidar data reader. 
    Inputs:
    - filelist: List of radar files (i.e, '/drives/c/*.nc') (str)
    - channels: tuple of channel numbers (e.g., (0,1,2) ) (tuple)
    Output:
    - lidar: dictionary or 'None' in case of error.
    """

    # Date format conversion
    files2load = glob.glob(filelist)

    if files2load:
        lidar = {}

        # open all files
        nc_ids = [nc.Dataset(file_) for file_ in files2load]

        # localization of instrument
        lidar['lat'] = nc_ids[0].variables['lat'][:]
        lidar['lon'] = nc_ids[0].variables['lon'][:]
        lidar['alt'] = nc_ids[0].variables['altitude'][:]
        lidar['location'] = nc_ids[0].site_location.split(',')[0]
        lidar['instr'] = nc_ids[0].system

        # read alt (no need to concatenate)
        lidar['range'] = nc_ids[0].variables['range'][:]

        # wavelength (no need to concatenate)
        tmp = nc_ids[0].variables['wavelength'][:]
        lidar['wavelength'] = tmp
        lidar['wavelength_units'] = nc_ids[0].variables['wavelength'].units

        # detection_mode (no need to concatenate)
        tmp = nc_ids[0].variables['detection_mode'][:]
        lidar['detection_mode'] = tmp

        # polarization (no need to concatenate)
        tmp = nc_ids[0].variables['polarization'][:]
        lidar['polarization'] = tmp

        # read time 
        units = nc_ids[0].variables[
            'time'].units  # 'days since %s' % dt.datetime.strftime(date, '%Y-%m-%d %H:%M:%S')
        tmp = [nc.num2date(nc_id.variables['time'][:], units) for nc_id in
               nc_ids]
        lidar['raw_time'] = np.concatenate(tmp)

        # check if any data available        
        # time_filter = (lidar['raw_time'] >= date) & (lidar['raw_time'] < date + dt.timedelta(days=1))
        # if not np.any(time_filter):
        #     print('No data in user-defined time.')
        #     return None

        # RCS
        for channel in channels:
            tmp = [nc_id.variables['rcs_%02d' % channel][:] for nc_id in nc_ids]
            lidar['rcs_%02d' % channel] = np.ma.filled(
                np.concatenate(tmp, axis=0))

            # Background
            tmp = [nc_id.variables['bckgrd_rcs_%02d' % channel][:] for nc_id in
                   nc_ids]
            lidar['background_%02d' % channel] = np.ma.filled(
                np.concatenate(tmp, axis=0), fill_value=np.nan)

            # close all files
        [nc_id.close() for nc_id in nc_ids]
    else:
        lidar = None
    return lidar


def reader_xarray(filelist, date_ini=None, date_end=None, ini_range=None, end_range=None, \
    percentage_required=80, channels='all'):
    """
    Lidar data reader using xarray module. 
    Inputs:
    - filelist: List of lidar files (i.e, '/drives/c/*.nc') (str)    
    - date_ini: 'yyyymmddThhmmss'
    - date_end: 'yyyymmddThhmmss'
    - ini_range: int/float (m)
    - end_range: int/float (m)
    - percentage_required= percentage of the time period required to continue the process. Default 80%  (int)
    - channels: list of channel number (e.g., [0, 1, 5]) or 'all' to load all of them
    Output:
    - lidar: dictionary or 'None' in case of error.
    """

    """ Aux Functions
    """
    def select_channels(dataset, channels):
        """
        select_channels function creates a new dataset with 'signal_CHANNEL' defined in 'channels' (list).
        Input:
        dataset: xarray dataset
        channels: list of lidar channel names
        Output:
        dataset: xarray datset
        """
        # pdb.set_trace()
        if channels != 'all':
            if not isinstance(channels, collections.abc.Iterable):
                channels = [channels]
            if isinstance(channels, np.ndarray):
                channels = channels.tolist()
            _vars = ['signal']
            for _channel in dataset["channel"]:
                if _channel not in channels:
                    for _var in _vars:
                        varname = '%s_%s' % (_var, _channel)
                        dataset = dataset.drop_vars(varname)
            dataset = dataset.sel(channel=channels)
            dataset = dataset.assign_coords(channel=channels)
        return dataset


    def check_minimum_profiles(times, date_ini, date_end, percentage_required):
        """ Check Lidar Data has enough profiles

        Args:
            times ([type]): [description]
            date_ini ([type]): [description]
            date_end ([type]): [description]
            percentage_required ([type]): [description]
        """
        # pdb.set_trace()

        check = True
        time_resolution = float(np.median(np.diff(times))/np.timedelta64(1, 's'))                    
        interval_duration = (date_end - date_ini).total_seconds()
        Nt = np.round(interval_duration / time_resolution)  # Theoretical Number of profiles
        Nm = (percentage_required/100)*Nt  # Minimum number of profiles
        Np = len(times) # Number of actual profiles  
        if Np > Nm:
            logger.info('Data loaded from %s to %s' % (date_ini.strftime('%Y%m%dT%H%M%S'),
            date_end.strftime('%Y%m%dT%H%M%S')))
        else:
            logger.warning('Not enough data found (%s<%s) in the user-required period (%s s.)' %
            (Np, Nm, interval_duration))
            check = False

        return check


    def get_lidar_system_from_filename(fn):
        """ Get Lidar System Name from L1a File Name
        Args:
            fn (function): [description]
        """
       

        try:
            lidar_nick = os.path.basename(fn).split('_')[0]
            for lidar in LIDAR_SYSTEM:
                if LIDAR_SYSTEM[lidar]['LIDARNICK'] == lidar_nick:
                    return lidar
        except Exception as e:
            logger.critical(str(e))
            return None


    """ The Reader
        The Reader does:
        1. concatenate along time dimension 
        2. merge channels comming from different telescopes (ALHAMBRA), assuming same range coordinate 
    """
    # pdb.set_trace()

    logger.info("Start Reader ...")
    # pdb.set_trace()

    # Find Files to Read
    try:
        files2load = glob.glob(filelist)
    except Exception as e:
        files2load = []
        logger.warning(str(e))
        logger.warning("Files in %s not found." % filelist)

    if len(files2load) > 0:
        logger.info(files2load)
        lidartemp = None
        try:
            # Get Lidar System Name
            lidar_system = get_lidar_system_from_filename(files2load[0])
            # Loop over modules: 1) concat time; 2) merge module
            for module in LIDAR_SYSTEM[lidar_system]['MODULES']:
                module_fns = [x for x in files2load if module in x]
                lidarmod = None
                if len(module_fns) > 0:
                    for fn in module_fns:
                        with xr.open_dataset(fn, chunks={}) as _dx: #chunks={"time": 600, "range": 1000})
                            _dx = select_channels(_dx, channels)
                        if not lidarmod:
                            lidarmod = _dx
                        else:
                            # concat only variables that have "time" dimension.
                            # rest of variables keep values from first dataset
                            try:
                                lidarmod = xr.concat([lidarmod, _dx], dim='time', data_vars="minimal", coords="minimal", compat="override")
                            except Exception as e:
                                logger.critical(str(e))
                                logger.critical("Dataset in %s not concatenated" % fn)
                    # Sort Dataset by Time
                    lidarmod = lidarmod.sortby(lidarmod['time'])
                    # Merge Module
                    if not lidartemp:
                        lidartemp = lidarmod
                    else:
                        try:
                            lidartemp = xr.merge([lidartemp, lidarmod])
                        except Exception as e:
                            logger.critical(str(e))
                            logger.critical("Dataset from module %s not merged" % module)
                del lidarmod
        except Exception as e:
            logger.critical(str(e))
            logger.critical("Files not concatenated")
            
        if lidartemp:
            # Selection time window and Check Enough Profiles
            if np.logical_and(date_ini is not None, date_end is not None):
                if np.logical_and(isinstance(date_ini, str), isinstance(date_end, str)):                    
                    # Times Formatting
                    date_ini_dt = utils.str_to_datetime(date_ini)
                    date_end_dt = utils.str_to_datetime(date_end)

                    # Time Selection
                    # pdb.set_trace()

                    min_time_resol = dt.timedelta(seconds=0.1)     
                    lidar = lidartemp.sel(time=slice(date_ini_dt - min_time_resol, date_end_dt + min_time_resol))
                    # Check selection
                    ok = check_minimum_profiles(lidar['time'], date_ini_dt, date_end_dt, percentage_required)
                    if not ok:
                        lidar = None
                else:
                    lidar = lidartemp
            else:
                lidar = lidartemp
            del lidartemp

            # Complete lidar dataset
            if lidar:
                # Range Clip
                if np.logical_and(ini_range is not None, end_range is not None):
                    if end_range > ini_range:
                        lidar = lidar.sel(range=slice(ini_range, end_range))

                # add background ranges
                if 'BCK_MIN_ALT' not in lidar.attrs.keys(): 
                    lidar.attrs['BCK_MIN_ALT'] = 75000
                if 'BCK_MAX_ALT' not in lidar.attrs.keys(): 
                    lidar.attrs['BCK_MAX_ALT'] = 105000

                # Extract information from filename                
                try:
                    lidar.attrs['lidarNick'] = os.path.basename(files2load[0]).split('_')[0]
                    lidar.attrs['dataversion'] = os.path.basename(files2load[0]).split('_')[1]
                except:
                    lidar.attrs['lidarNick'] = 'Unknown'
                    lidar.attrs['dataversion'] = 'Unknown'

            else:
                lidar = None    
        else:
            lidar = None
    else:
        lidar = None

    if lidar is None:
        logger.warning('No lidar dataset created')

    logger.info("End Reader")

    return lidar


""" LIDAR PREPROCESSING
"""
def preprocessing(rs_fl, channels='all', \
    ini_date=None, end_date=None, percentage_required=80, \
        ini_range=0, end_range=20000, bg_window=None, \
            darkcurrent_flag=True, dc_fl=None, \
                deadtime_flag=True, dead_time_fn=None, \
                    zerobin_flag=True, bin_zero_fn=None, \
                        merge_flag=True, \
                            data_dn=None):
     
    """
    Preprocessing lidar signals including: dead time, dark measurement,
    background, and bin shift.

    Parameters
    ----------
    rs_fl: str
        Wildcard List of lidar files (i.e, '/drives/c/*.nc') (str)....
    dc_fl: str
        Wildcard List of DC lidar files (i.e, '/drives/c/*.nc') (str)....
    ini_date: str
        yyyymmddThhmmss
    end_date: str
        yyyymmddThhmmss
    ini_range: int, float
        min range [m]
    end_range: int, float
        max range [m]
    bg_window: tuple
        range window limits to calculate background
    percentage_required: int, float
        percentage of the time period required to continue the process. Default 80%
    channels: str, list(str)
        list of channel number (e.g., [0, 1, 5]) or 'all' to load all of them
    bin_zero_fn: str
        bin zero file
    dead_time_fn: str
        dead time file
    data_dn: str
        full path of directory of data where bin zero file should be
    darkcurrent_flag: bool
        active/desactive the dark-current correction.
    deadtime_flag: bool
        active/desactive the dead time correction.
    zerobin_flag: bool
        active/desactive the zero-bin and trigger delay corrections.
    merge_flag: bool
        active/desactive the merge of polarizing components.

    Returns
    -------
    ps_ds: xarray.Dataset
        dataset with pre-processed signal

    """

    """ Auxiliary Functions
    """
    def mulhacen_fix_channels(dataset):
        """ CHANGE OF WAVELENGTH VALUES """
        DATE_ROTATIONAL_RAMAN355 = '15/12/2016 00:00:00'
        DATE_ROTATIONAL_RAMAN532 = '04/05/2017 00:00:00'
        if dataset.time.max() > np.datetime64(dt.datetime.strptime(DATE_ROTATIONAL_RAMAN532, '%d/%m/%Y %H:%M:%S')):
            if 607 in dataset.wavelength.values:
                wave = dataset.wavelength.values
                wave[wave == 607] = 530
                dataset['wavelength'].values = wave
        if dataset.time.max() > np.datetime64(dt.datetime.strptime(DATE_ROTATIONAL_RAMAN355, '%d/%m/%Y %H:%M:%S')):
            if 387 in dataset.wavelength.values:
                wave = dataset.wavelength.values
                wave[wave == 387] = 354
                dataset['wavelength'].values = wave                
        return dataset

    """ The Preprocessing
    """
    # pdb.set_trace()
    info_flag = False #to be converted to logging
    debug_flag =  True #to be converted to logging
    warning_flag = False #to be converted to logging
    error_flag = True #to be converted to logging
    logger.info('Start Lidar Preprocessing ...')

    """ Read Raw Measurement for the time period given. """
    logger.info("Read Raw Signal")
    # Reader Xarray
    rs_ds = reader_xarray(rs_fl, date_ini=ini_date, date_end=end_date, percentage_required=percentage_required, channels=channels)    
    # pdb.set_trace() 

    if rs_ds is not None and len(rs_ds['time'])>0:  # There are measurements
        """ Initialize Preprocessed Dataset """
        ps_ds = rs_ds.copy(deep=True)

        """ Get Relevant Info from RS Measurement """
        lidar_name = rs_ds.system
        # Define time, ranges, channels
        times = rs_ds['time'].values
        ranges = rs_ds['range'].values
        channel_names = rs_ds['channel'].values
        n_channels = len(channel_names)
        ref_time = utils.numpy_to_datetime(times[0]).strftime("%Y%m%d")

        # lat, lon, alt
        lat = float(rs_ds.lat.values)
        lon = float(rs_ds.lon.values)
        alt = float(rs_ds.altitude.values)

        """ Background Range Index Filter """
        # Need numerical indices
        if bg_window is None:
            bg_window = (rs_ds.attrs['BCK_MIN_ALT'], rs_ds.attrs['BCK_MAX_ALT'])
        #idx_bg = np.logical_and(ranges >= bg_window[0], ranges <= bg_window[1])
        # bg_window = [45000,60000] # just for KASCAL
        idx_bg = np.squeeze(np.argwhere(np.logical_and(ranges >= bg_window[0], ranges <= bg_window[1])))
        
        # pdb.set_trace()
        """ day/ngt RS measurements """
        sun_rs = solar.SUN(times, lon, lat, elev=alt)
        csza_rs = sun_rs.get_csza()
        idx_rs_day = np.where(csza_rs >= 0)[0]
        idx_rs_ngt = np.where(csza_rs < 0)[0]

        """ Read Associated Dark Current Measurements, if wanted """
        if darkcurrent_flag:
            if dc_fl:
                try:
                    logger.info("Read DC Signal")
                    dc_ds = reader_xarray(dc_fl, channels=channels)
                except Exception as e:  # Read From Lidar Measurement
                    dc_ds = None
                    logger.warning(str(e))
                    logger.warning("DC measurements cannot be read. DC set to None")
                if dc_ds is not None:
                    # day/ngt DC measurements
                    times_dc = dc_ds['time']
                    sun_dc = solar.SUN(times_dc, lon, lat, elev=alt)
                    csza_dc = sun_dc.get_csza()
                    idx_dc_day = np.where(csza_dc > 0.01)[0]
                    idx_dc_ngt = np.where(csza_dc < -0.01)[0]
                    
                    # Check proper DC
                    if rs_ds.dims['channel'] != dc_ds.dims['channel']:  # DC Properly Exists
                        dc_ds = None
                        logger.warning('dark measurement files do not match number of measurement channels: %s' % dc_fl)
            else:
                # TODO: ¿TIENE SENTIDO LEER EL DC DE OTRO DIA?
                dc_ds = None
                logger.warning('dark measurement files not provided: %s' % dc_fl)
                logger.warning('Process continues without dark measurement correction.')
        else:
            dc_ds = None

        """ Loop over Channels: (wv, pol, det_mode) """
        # Arrays for storing Bin Zero and Dead Time
        bin_zero_arr = np.zeros(n_channels)*np.nan
        tau_arr = np.zeros(n_channels)*np.nan
        # pdb.set_trace()
        for i_chan, channel_ in enumerate(channel_names):
            logger.info("Channel %s" % channel_)
            try:
                # channel wavelength, polarization, detection mode
                wv = rs_ds.wavelength[i_chan].values
                pol = rs_ds.polarization[i_chan].values
                mod = rs_ds.detection_mode[i_chan].values
                if debug_flag:
                    logger.debug("%d, %d, %d" % (wv, pol, mod))

                """ Raw Signal """
                rs = rs_ds['signal_%s' % channel_]                    
            
                """ Bin Zero """ 
                if zerobin_flag:
                    bz = get_bin_zero(lidar_name, wv, pol, mod, ref_time=ref_time, bin_zero_fn=bin_zero_fn, data_dn=data_dn)
                else:
                    bz = 0
                bin_zero_arr[i_chan] = bz

                # Initialize Common Attributes for channel dataarray
                binzero_corrected = zerobin_flag
                background_corrected = True
                # pdb.set_trace()

                """ Pre-processing """
                if mod == 0:  # ANALOG SIGNAL
                    """ Dark Signal 
                    + Compute averaged profile for day and night conditions.
                    + 0-like by default
                    + preprocessed Dark Signal with time dimension as raw signal.
                    """
                    # Initialize Specific Attributes
                    dark_corrected = False
                    
                    # Initialize DC
                    dc_avg_day = xr.zeros_like(rs[0, :], dtype=float)
                    dc_avg_ngt = xr.zeros_like(rs[0, :], dtype=float)
                    if darkcurrent_flag:  # If there is DC
                        if dc_ds is not None:
                            try:
                                dc = dc_ds['signal_%s' % channel_]
                                if np.logical_and(len(idx_dc_day) > 0, len(idx_dc_ngt) > 0):
                                    dc_avg_day = xr.apply_ufunc(average_dc_signal, dc[idx_dc_day, :], dask='allowed', input_core_dims=[['time', 'range']], output_core_dims=[['range']])
                                    dc_avg_ngt = xr.apply_ufunc(average_dc_signal, dc[idx_dc_ngt, :], dask='allowed', input_core_dims=[['time', 'range']], output_core_dims=[['range']])
                                else:
                                    dc_avg_day = xr.apply_ufunc(average_dc_signal, dc, dask='allowed', input_core_dims=[['time', 'range']], output_core_dims=[['range']])
                                    dc_avg_ngt = dc_avg_day
                                dark_corrected = True
                            except Exception as e:
                                if warning_flag:
                                    logger.warning(str(e))
                                    logger.warning("Error averaging DC. Use 0.")
                    """ Analog Pre-processing
                    + compute for day and night conditions
                    """
                    if np.logical_and(len(idx_rs_day) > 0, len(idx_rs_ngt) > 0):
                        ps_day = xr.apply_ufunc(preprocessing_analog_signal, rs[idx_rs_day, :], dc_avg_day, bz, idx_bg[0], idx_bg[-1],
                            kwargs={'zerobin_flag': zerobin_flag}, dask='allowed', \
                                input_core_dims=[['time', 'range'], ['range'], [], [], []], output_core_dims=[['time', 'range']])
                        ps_ngt = xr.apply_ufunc(preprocessing_analog_signal, rs[idx_rs_ngt, :], dc_avg_ngt, bz, idx_bg[0], idx_bg[-1],
                            kwargs={'zerobin_flag': zerobin_flag}, dask='allowed', \
                                input_core_dims=[['time', 'range'], ['range'], [], [], []], output_core_dims=[['time', 'range']])
                        ps = xr.concat([ps_day, ps_ngt], dim='time').sortby('time')
                    else:
                        ps = xr.apply_ufunc(preprocessing_analog_signal, rs, dc_avg_day, bz, idx_bg[0], idx_bg[-1],
                            kwargs={'zerobin_flag': zerobin_flag}, dask='allowed', \
                                input_core_dims=[['time', 'range'], ['range'], [], [], []], output_core_dims=[['time', 'range']])

                    """ Current Long Name """
                    current_long_name = ''
                    if dark_corrected:
                        current_long_name += 'DM-, '
                    if binzero_corrected:
                        current_long_name += 'BZ-, '
                    if background_corrected:
                        current_long_name += 'background-, '
                    current_long_name += 'corrected signal'

                    """ Create Attributes Dictionary """
                    new_attrs = {'long_name': current_long_name, 'units': 'a.u.', \
                        'dark_corrected': dark_corrected, 'binzero_corrected': binzero_corrected, 
                        'background_corrected': background_corrected}
                elif mod == 1:  # PHOTONCOUNTING SIGNAL
                    """ Dark Signal
                    + DC = 0.0
                    """
                    # Initialize Specific Attributes
                    deadtime_corrected = deadtime_flag

                    """ Dead Time """
                    if deadtime_flag:
                        tau = get_dead_time(lidar_name, wv, pol, mod, ref_time=ref_time, dead_time_fn=dead_time_fn, data_dn=data_dn)
                    else:
                        tau = 0.0
                    # tau = 0.0
                    tau_arr[i_chan] = tau

                    """ Photoncounting Pre-processing
                    + pc peak correction if MULHACEN
                    + compute for day and night conditions
                    + rs is loaded in memory to speed-up pc_peak_correction computation
                    """
                    if lidar_name == 'MULHACEN':
                        rs2 = mulhacen_pc_peak_correction(rs.values)
                        rs = rs.copy(data=rs2)
                    if np.logical_and(len(idx_rs_day) > 0, len(idx_rs_ngt) > 0):
                        """ Preprocessing """
                        ps_day = xr.apply_ufunc(preprocessing_photoncounting_signal, rs[idx_rs_day, :], tau, bz, idx_bg[0], idx_bg[-1],
                            kwargs={'deadtime_flag':deadtime_flag, 'zerobin_flag': zerobin_flag}, \
                            dask='allowed', input_core_dims=[['time', 'range'], [], [], [], []], output_core_dims=[['time', 'range']])                            
                        ps_ngt = xr.apply_ufunc(preprocessing_photoncounting_signal, rs[idx_rs_ngt, :], tau, bz, idx_bg[0], idx_bg[-1],
                            kwargs={'deadtime_flag':deadtime_flag, 'zerobin_flag': zerobin_flag}, \
                            dask='allowed', input_core_dims=[['time', 'range'], [], [], [], []], output_core_dims=[['time', 'range']])                            
                        ps = xr.concat([ps_day, ps_ngt], dim='time').sortby('time')
                    else:
                        """ Preprocessing """
                        ps = xr.apply_ufunc(preprocessing_photoncounting_signal, rs, tau, bz, idx_bg[0], idx_bg[-1],
                            kwargs={'deadtime_flag':deadtime_flag, 'zerobin_flag': zerobin_flag}, \
                            dask='allowed', input_core_dims=[['time', 'range'], [], [], [], []], output_core_dims=[['time', 'range']])                            

                    """ Current Long Name """
                    current_long_name = ''
                    if deadtime_flag:
                        current_long_name += 'DT-, '
                    if zerobin_flag:
                        current_long_name += 'BZ-, '
                    if background_corrected:
                        current_long_name += 'background-, '
                    current_long_name += '- corrected signal'

                    """ Create Attributes Dictionary """
                    new_attrs = {'long_name': current_long_name, 'units': 'a.u.', \
                        'deadtime_corrected': deadtime_corrected, 'binzero_corrected': binzero_corrected, 
                        'background_corrected': background_corrected}
                else:  # not AN nor PC
                    ps = xr.full_like(rs, np.nan)
                    current_long_name = 'not corrected signal'
                    new_attrs = {'long_name': current_long_name, 'units': 'a.u.'}
                    if error_flag:
                        logger.critical("ERROR. Channel %s not Analog nor Photoncounting" % channel_)
            except Exception as e:
                ps = xr.full_like(rs, np.nan)
                current_long_name = 'not corrected signal'
                new_attrs = {'long_name': current_long_name, 'units': 'a.u.'}
                if error_flag:
                    logger.critical(str(e))
                    logger.critical("Error in channel %s. Preprocessing not performed." % channel_)

            """ Add Corrected Signal to Dataset """
            ps_ds['signal_%s' % channel_] = ps.assign_attrs(new_attrs)

        """ Clip Height Range """
        idx_range = np.logical_and(ranges >= ini_range, ranges <= end_range)
        ps_ds = ps_ds.sel(range=ranges[idx_range])
        
        """" Mulhacen fix channels (This is attempted already in raw2l1, but you never know...) """
        if lidar_name == 'MULHACEN':
            ps_ds = mulhacen_fix_channels(ps_ds)

        """ Bin Zero """
        pbz = xr.ones_like(rs_ds['wavelength'])*bin_zero_arr
        ps_ds['bin_zero'] = pbz.assign_attrs({'units': '', 'long_name': 'bin zero'})
     
        """ Dead Time """
        ptau = xr.ones_like(rs_ds['wavelength'])*tau_arr
        ps_ds['dead_time'] = ptau.assign_attrs({'units': '', 'long_name': 'dead time'})

        """ Add Default Depolarization Channels Info """   
        # pdb.set_trace()
        if merge_flag:
            ps_ds = merge_polarized_channels(ps_ds, channels)

        if info_flag:
            print("End Preprocessing Succesfully")
    else:
        ps_ds = None
        if error_flag:
            print("ERROR. Preprocessing not performed for File(s) %s" % rs_fl)

    logger.info('End Lidar Preprocessing.')
    return ps_ds


""" MERGE POLARIZED CHANNELS TO DATASET
"""
def merge_polarized_channels(lxarray, channels):
    """
    It merges the polarized channels and retrieve the Linear Volume Depolarization Ratio

    Parameters
    ----------
    lxarray: xarray.Dataset from lidar.preprocessing() (xarray.Dataset)
    channels: str, list(str)
        list of channels (e.g., ['532xpa', '1064xta']) or 'all' to load all of them

    Returns
    -------
    lxarray: xarray.Dataset with new varaibles ('signal%d_total' % _wavelength ; 'LVDR%d' % _wavelength)
    """

    logger.info('Start Merge Polarized Channels')

    """ Check Input """
    # pdb.set_trace()
    if not isinstance(channels, collections.abc.Iterable):
        channels = [channels]
        if isinstance(channels, np.ndarray):
            channels = channels.tolist()
    if channels == 'all':
        channels = lxarray['channel'].values

    """ Relevant Info """
    polarization_value = {'R': 1, 'T':2}  # Reflected , Transmitted
    detection_value = {'an': 0, 'pc':1}   #analog = 0, photoncounting = 1
    telescope_value = {'xf': 1, 'ff': 1, 'nf': 2}

    # Dictionary organized by wavelengths
    depoCalib = {'mulhacen': {'xf':{}}, 'alhambra': {'ff':{}, 'nf':{}}, 'veleta': {'xf':{}}, 'kascal': {'xf':{}}}
    # Each wavelength has a depolarization calibration DataFrame
    depoCalib['mulhacen']['xf']['532'] = pd.DataFrame(
        columns=['date', 'eta_an', 'eta_pc', 'GR', 'GT', 'HR', 'HT', 'K'])

    dict0 = {'date': dt.datetime(2015, 6, 7), 'eta_an': 0.0757,
             'eta_pc': 0.1163, 'GR': 1., 'GT': 1, 'HR': 1., 'HT': -1., 'K': 1.}
    dict1 = {'date': dt.datetime(2016, 6, 7), 'eta_an': 0.0757,
             'eta_pc': 0.1163, 'GR': 1., 'GT': 1, 'HR': 1., 'HT': -1., 'K': 1.}
    # dict1 = {'date': dt.datetime(2017,9,1), 'eta_an': 0.041, 'eta_pc': 0.0717, 'GR':1.31651, 'GT':0.65366, 'HR':1.22341, 'HT':-0.62378, 'K': 1.007}
    # DATE_ROTATIONAL_RAMAN532 = '04/05/2017 00:00:00'   
    # dict2 = {'date': dt.datetime(2017,5,5), 'eta_an': 0.041, 'eta_pc': 0.0717, 'GR':1.31651, 'GT':0.65366, 'HR':1.22341, 'HT':-0.62378, 'K': 1.007} 
    # dict3 = {'date': dt.datetime(2017,5,5), 'eta_an': 0.13, 'eta_pc': 0.21, 'GR':1.58726, 'GT':0.41269, 'HR':1.50151, 'HT':-0.33235, 'K': 1.0000}

    # Caso A: Rotator epsilon = 0. alpha = 0.
    dictA = {'date': dt.datetime(2017, 5, 5), 'eta_an': 0.13, 'eta_pc': 0.21,
             'GR': 1.59994, 'GT': 0.40001, 'HR': 1.59990, 'HT': -0.39999,
             'K': 1.000}
    # ===========================================================================================================
    #  GR     , GT     , HR     , HT     ,  K(0.000),  K(0.004), K(0.02) ,  K(0.1) ,  K(0.2) ,  K(0.3) ,  K(0.45)
    #  1.59994, 0.40001, 1.59990,-0.39999,  1.00000,  1.00000,  1.00000,  1.00000,  1.00000,  1.00000,  1.00000
    # ===========================================================================================================

    # Caso B: Rotator epsilon = -5.9. alpha = 0.
    dictB = {'date': dt.datetime(2017, 5, 5), 'eta_an': 0.13, 'eta_pc': 0.21,
             'GR': 1.58726, 'GT': 0.41269, 'HR': 1.57877, 'HT': -0.37886,
             'K': 1.0000}
    # ===========================================================================================================
    # GR     , GT     , HR     , HT     ,  K(0.000),  K(0.004), K(0.02) ,  K(0.1) ,  K(0.2) ,  K(0.3) ,  K(0.45)
    # 1.58726, 0.41269, 1.57877,-0.37886,  1.00000,  1.00000,  1.00000,  1.00000,  1.00000,  1.00000,  1.00000
    # ===========================================================================================================

    # Caso C: Rotator epsilon = 0. alpha = -5.9.
    dictC = {'date': dt.datetime(2017, 5, 5), 'eta_an': 0.13, 'eta_pc': 0.21,
             'GR': 1.59994, 'GT': 0.40001, 'HR': 1.56609, 'HT': -0.39154,
             'K': 1.0000}
    # ===========================================================================================================
    # GR     , GT     , HR     , HT     ,  K(0.000),  K(0.004), K(0.02) ,  K(0.1) ,  K(0.2) ,  K(0.3) ,  K(0.45)
    # 1.59994, 0.40001, 1.56609,-0.39154,  1.00000,  1.00000,  1.00000,  1.00000,  1.00000,  1.00000,  1.00000
    # ===========================================================================================================

    # Caso D: Rotator Alvarez's method alpha+epsilon = -13º --> epsilon = -5.9. alpha = -7.1
    dictD = {'date': dt.datetime(2017, 5, 5), 'eta_an': 0.13, 'eta_pc': 0.21,
             'GR': 1.58726, 'GT': 0.41269, 'HR': 1.57066, 'HT': -0.40741,
             'K': 1.0000}
    # ===========================================================================================================
    # GR     , GT     , HR     , HT     ,  K(0.000),  K(0.004), K(0.02) ,  K(0.1) ,  K(0.2) ,  K(0.3) ,  K(0.45)
    # 1.58726, 0.41269, 1.57066,-0.40741,  1.00000,  1.00000,  1.00000,  1.00000,  1.00000,  1.00000,  1.00000
    # ===========================================================================================================

    # Caso E: Rotator Alvarez's method alpha-epsilon = -13º --> epsilon = -5.9. alpha = -20.9
    dictE = {'date': dt.datetime(2017, 5, 5), 'eta_an': 0.13, 'eta_pc': 0.21,
             'GR': 1.58726, 'GT': 0.41269, 'HR': 1.28597, 'HT': -0.39147,
             'K': 1.0000}
    # ===========================================================================================================
    # GR     , GT     , HR     , HT     ,  K(0.000),  K(0.004), K(0.02) ,  K(0.1) ,  K(0.2) ,  K(0.3) ,  K(0.45)
    # 1.58726, 0.41269, 1.28597,-0.39147,  1.00000,  1.00000,  1.00000,  1.00000,  1.00000,  1.00000,  1.00000
    # ===========================================================================================================

    # Caso F: Rotator Alvarez's method alpha+epsilon = +13º --> epsilon = -5.9. alpha = +20.9
    dictF = {'date': dt.datetime(2017, 5, 5), 'eta_an': 0.13, 'eta_pc': 0.21,
             'GR': 1.58726, 'GT': 0.41269, 'HR': 1.06790, 'HT': -0.17339,
             'K': 1.00004}
    # ===========================================================================================================
    # GR     , GT     , HR     , HT     ,  K(0.000),  K(0.004), K(0.02) ,  K(0.1) ,  K(0.2) ,  K(0.3) ,  K(0.45)
    # 1.58726, 0.41269, 1.06790,-0.17339,  1.00004,  1.00004,  1.00004,  1.00003,  1.00002,  1.00002,  1.00001
    # ===========================================================================================================

    # Caso G: Rotator Alvarez's method alpha+epsilon = +13º --> epsilon = -5.9. alpha = +7.1
    dictG = {'date': dt.datetime(2017, 5, 5), 'eta_an': 0.13, 'eta_pc': 0.21,
             'GR': 1.58726, 'GT': 0.41269, 'HR': 1.49040, 'HT': -0.32715,
             'K': 1.00001}
    # ===========================================================================================================
    # GR     , GT     , HR     , HT     ,  K(0.000),  K(0.004), K(0.02) ,  K(0.1) ,  K(0.2) ,  K(0.3) ,  K(0.45)
    # 1.58726, 0.41269, 1.49040,-0.32715,  1.00001,  1.00001,  1.00001,  1.00001,  1.00001,  1.00001,  1.00000
    # ===========================================================================================================

    # Caso H: Rotator Alvarez's method alpha+epsilon = +13º --> epsilon = +5.9. alpha = +7.1
    dictH = {'date': dt.datetime(2017, 5, 5), 'eta_an': 0.13, 'eta_pc': 0.21,
             'GR': 1.58726, 'GT': 0.41269, 'HR': 1.57066, 'HT': -0.40741,
             'K': 1.0000}
    # ===========================================================================================================
    # GR     , GT     , HR     , HT     ,  K(0.000),  K(0.004), K(0.02) ,  K(0.1) ,  K(0.2) ,  K(0.3) ,  K(0.45)
    # 1.58726, 0.41269, 1.57066,-0.40741,  1.00000,  1.00000,  1.00000,  1.00000,  1.00000,  1.00000,  1.00000
    # ===========================================================================================================

    # Caso I: Rotator Alvarez's method alpha+epsilon = -13º --> epsilon = +5.9. alpha = -20.9
    dictI = {'date': dt.datetime(2017, 5, 5), 'eta_an': 0.13, 'eta_pc': 0.21,
             'GR': 1.58726, 'GT': 0.41269, 'HR': 1.06790, 'HT': -0.17339,
             'K': 1.00004}
    # ===========================================================================================================
    # GR     , GT     , HR     , HT     ,  K(0.000),  K(0.004), K(0.02) ,  K(0.1) ,  K(0.2) ,  K(0.3) ,  K(0.45)
    # 1.58726, 0.41269, 1.06790,-0.17339,  1.00004,  1.00004,  1.00004,  1.00003,  1.00002,  1.00002,  1.00001
    # ===========================================================================================================

    # Caso J: Polarizer epsilon = 15. alpha = 7.1
    dictJ = {'date': dt.datetime(2017, 5, 5), 'eta_an': 0.13, 'eta_pc': 0.21,
             'GR': 3.19952, 'GT': 0.00010, 'HR': 3.10176, 'HT': -0.00006,
             'K': 4.16828}
    # ===========================================================================================================
    # GR     , GT     , HR     , HT     ,  K(0.000),  K(0.004), K(0.02) ,  K(0.1) ,  K(0.2) ,  K(0.3) ,  K(0.45)
    # 2.98522, 0.05367, 2.15126, 0.03558,  4.26295,  4.25698,  4.23541,  4.16358,  4.11450,  4.08424,  4.05465
    # ===========================================================================================================
    # no epsilon
    # ===========================================================================================================
    # GR     , GT     , HR     , HT     ,  K(0.000),  K(0.004), K(0.02) ,  K(0.1) ,  K(0.2) ,  K(0.3) ,  K(0.45)
    # 3.19952, 0.00010, 3.10176,-0.00006,  4.16828,  4.16674,  4.16073,  4.13423,  4.10745,  4.08569,  4.05957
    # ===========================================================================================================

    dict2 = dictG

    depoCalib['mulhacen']['xf']['532'].loc[0] = pd.Series(dict0)
    depoCalib['mulhacen']['xf']['532'].loc[1] = pd.Series(dict1)
    depoCalib['mulhacen']['xf']['532'].loc[2] = pd.Series(dict2)
    depoCalib['mulhacen']['xf']['532'] = depoCalib['mulhacen']['xf']['532'].set_index('date')
        
    #ALHAMBRA
    depoCalib['alhambra']['ff']['355'] = pd.DataFrame(columns=['date', 'eta_an', 'eta_pc', 'GR', 'GT', 'HR', 'HT', 'K'])
    depoCalib['alhambra']['nf']['355'] = pd.DataFrame(columns=['date', 'eta_an', 'eta_pc', 'GR', 'GT', 'HR', 'HT', 'K'])
    depoCalib['alhambra']['nf']['532'] = pd.DataFrame(columns=['date', 'eta_an', 'eta_pc', 'GR', 'GT', 'HR', 'HT', 'K'])
    dict0 = {'date': dt.datetime(2021, 12, 1), 'eta_an': 1., 'eta_pc': 1., 'GR': 1., 'GT': 1, 'HR': -1., 'HT': 1., 'K': 1.} #Considering y = + 1 i.e., parallel in T. 
    dict1 = {'date': dt.datetime(2021, 12, 1), 'eta_an': 1., 'eta_pc': 1., 'GR': 1., 'GT': 1, 'HR': 1., 'HT': -1., 'K': 1.} #Considering y = - 1 i.e., parallel in R. 
    depoCalib['alhambra']['ff']['355'].loc[0] = pd.Series(dict1)
    depoCalib['alhambra']['nf']['355'].loc[0] = pd.Series(dict1)
    depoCalib['alhambra']['nf']['532'].loc[0] = pd.Series(dict0)
    depoCalib['alhambra']['ff']['355'] = depoCalib['alhambra']['ff']['355'].set_index('date')
    depoCalib['alhambra']['nf']['355'] = depoCalib['alhambra']['nf']['355'].set_index('date')
    depoCalib['alhambra']['nf']['532'] = depoCalib['alhambra']['nf']['532'].set_index('date')
    
    #VELETA
    depoCalib['veleta']['xf']['355'] = pd.DataFrame(columns=['date', 'eta_an', 'eta_pc', 'GR', 'GT', 'HR', 'HT', 'K'])
    depoCalib['kascal']['xf']['355'] = pd.DataFrame(columns=['date', 'eta_an', 'eta_pc', 'GR', 'GT', 'HR', 'HT', 'K'])

    
    #dict0 = {'date': dt.datetime(2021, 12, 1), 'eta_an': 1., 'eta_pc': 1., 'GR': 1., 'GT': 1, 'HR': -1., 'HT': 1., 'K': 1.} #Considering y = + 1 i.e., parallel in T. 
    dict1 = {'date': dt.datetime(2021, 12, 1), 'eta_an': 1., 'eta_pc': 1., 'GR': 1., 'GT': 1, 'HR': 1., 'HT': -1., 'K': 1.} #Considering y = - 1 i.e., parallel in R. 
    depoCalib['veleta']['xf']['355'].loc[0] = pd.Series(dict0)        
    depoCalib['veleta']['xf']['355'] = depoCalib['veleta']['xf']['355'].set_index('date')

    dict2 = {'date': dt.datetime(2018, 2, 5), 'eta_an': 0.0339, 'eta_pc': 0.0350, 'GR': 1.00000, 'GT': 1.00000, 'HR': 0.95621, 'HT': -0.95629, 'K': 1.00001} #Considering y = - 1 i.e., parallel in R. 

    depoCalib['kascal']['xf']['355'].loc[0] = pd.Series(dict2)        
    depoCalib['kascal']['xf']['355'] = depoCalib['kascal']['xf']['355'].set_index('date')

    
    # TODO: leer netcdf de /mnt/NASGFAT/datos/MULHACEN/QA/depolarization_calibration/YYYY/MM/DD/rotator_YYYYMMDD_HHMM/*rot*.nc
    # de momento, solo calibracion del rotador

    # INCLUIR LA INFO DE G, H, K EN lxarray

    # Date of the current measurement
    current_date = lxarray.time[0].min().values
    # pdb.set_trace()
    #Telescope
    lidar_name = lxarray.attrs['system'].lower()
    polchannels = {}
    if lidar_name == 'alhambra':
        polchannels['ff'] = {355: {'Tan': '355fpa', 'Tpc': '355fpp', 'Ran': '355fsa', 'Rpc': '355fsp'}}
        polchannels['nf'] = {355: {'Tan': '355npa', 'Tpc': '355npp', 'Ran': '355nsa', 'Rpc': '355nsp'}, 
                            532: {'Tan': '532npa', 'Tpc': '532npp', 'Ran': '532nsa', 'Rpc': '532nsp'}}
    elif lidar_name == 'mulhacen':
        polchannels['xf'] = {532: {'Tan': '532xpa', 'Tpc': '532xpp', 'Ran': '532xsa', 'Rpc': '532xsp'}} 
    elif lidar_name == 'veleta': 
        polchannels['xf'] = {355: {'Tan': '355xpa', 'Tpc': '355xpp', 'Ran': '355xsa', 'Rpc': '355xsp'}} 
    elif lidar_name == 'kascal': 
        polchannels['xf'] = {355: {'Tan': '355xpa', 'Tpc': '355xpp', 'Ran': '355xsa', 'Rpc': '355xsp'}} 
    else:
        pass
    
    # loop over channels/wavelengths    
    for field_ in polchannels.keys():         
        for _wavelength in polchannels[field_].keys():
            # pdb.set_trace()
            # Search the last calibration performed the current measurement
            idx = depoCalib['%s' % lidar_name][field_]['%0d' % _wavelength].index.get_loc(current_date, method='pad')  # 'pad': search the nearest lower; 'nearest': search the absolute nearest.
            calib = depoCalib['%s' % lidar_name][field_]['%0d' % _wavelength].iloc[idx]

            LVDR = {}            
            # ANALOG DEPOLARIZATION CHANNELS
            if  np.logical_and(polchannels[field_][_wavelength]['Tan'] in channels, polchannels[field_][_wavelength]['Ran'] in channels):
                # Find the signals to be used: wavelength | reflected/transmitted | analog/photoncounting                
                idx_T_an = np.argwhere(np.squeeze(np.logical_and.reduce((lxarray['polarization'].values == polarization_value['T'],
                                                            lxarray['detection_mode'].values == detection_value['an'],
                                                            lxarray['telescope'].values == telescope_value[field_],
                                                            lxarray['wavelength'].values == _wavelength)) )).item()
                wave_T_an = np.squeeze(lxarray['channel'].values[idx_T_an])
                idx_R_an = np.argwhere(np.squeeze(np.logical_and.reduce((lxarray['polarization'].values== polarization_value['R'],
                                                            lxarray['detection_mode'].values== detection_value['an'],
                                                            lxarray['telescope'].values == telescope_value[field_],
                                                            lxarray['wavelength'].values == _wavelength)) )).item()
                wave_R_an = np.squeeze(lxarray['channel'].values[idx_R_an])                        
                
                # Merge ANALOG signals and LVDR retrieval:
                if np.logical_and(np.size(wave_T_an) > 0, np.size(wave_R_an) > 0):                    
                    signal_Tan = lxarray['signal_%s' % wave_T_an] 
                    signal_Ran = lxarray['signal_%s' % wave_R_an]
                    channel_total_an = '%d%sta' % (_wavelength, field_[0])
                    # Calculate Total signal
                    signal_total_an = np.abs(calib['eta_an'] * calib['HR'] * signal_Tan - calib['HT'] * signal_Ran)
                    new_attrs = {'long_name': 'signal', 'wavelength': _wavelength, 'detection_mode': 0, 'units': 'a.u.', 'eta_GHK': str(calib)}
                    lxarray['signal_%s' % channel_total_an] = signal_total_an.assign_attrs(new_attrs)
                    # LVD
                    eta_an = calib['eta_an'] / calib['K']
                    ratio_an = (signal_Ran / signal_Tan) / eta_an
                    lvdr_an = ((calib['GT'] + calib['HT']) * ratio_an - (calib['GR'] + calib['HR'])) / ((calib['GR'] - calib['HR']) - (calib['GT'] - calib['HT']) * ratio_an)
                    new_attrs = {'long_name': 'Linear Volume Depolarization Ratio', 'detection_mode': 0, 'wavelength': _wavelength, 'units': '$\#$'}
                    LVDR['an'] = lvdr_an.assign_attrs(new_attrs)
                    channel_lvd_an = '%d%sa' % (_wavelength, field_[0])
                    lxarray['lvd_%s' % channel_lvd_an] = LVDR['an']
                else:
                    print("WARNING: analog signals not merged for wavelength %.1f. R and/or T are missing" % _wavelength)
            
            # PHOTONCOUNTING DEPOLARIZATION CHANNELS
            if np.logical_and(polchannels[field_][_wavelength]['Tpc'] in channels, polchannels[field_][_wavelength]['Rpc'] in channels):        
                # Find the signals to be used: wavelength | reflected/transmitted | analog/photoncounting
                idx_T_pc = np.argwhere(np.squeeze(np.logical_and.reduce((lxarray['polarization'].values == polarization_value['T'],
                                                            lxarray['detection_mode'].values == detection_value['pc'],
                                                            lxarray['telescope'].values == telescope_value[field_],
                                                            lxarray['wavelength'].values == _wavelength)) )).item()        
                wave_T_pc = np.squeeze(lxarray['channel'].values[idx_T_pc])
                idx_R_pc = np.argwhere(np.squeeze(np.logical_and.reduce((lxarray['polarization'].values == polarization_value['R'],
                                                            lxarray['detection_mode'].values == detection_value['pc'],
                                                            lxarray['telescope'].values == telescope_value[field_],
                                                            lxarray['wavelength'].values == _wavelength)) )).item()
                wave_R_pc = np.squeeze(lxarray['channel'].values[idx_R_pc])

                # Merge PHOTONCOUNTING signals:            
                if np.logical_and(np.size(wave_T_pc) > 0, np.size(wave_R_pc) > 0):
                    signal_Tpc = lxarray['signal_%s' % wave_T_pc]
                    signal_Rpc = lxarray['signal_%s' % wave_R_pc]
                    channel_total_pc = '%d%stp' % (_wavelength, field_[0])
                    # Total signal
                    signal_total_pc = np.abs(calib['eta_pc'] * calib['HR'] * signal_Tpc - calib['HT'] * signal_Rpc)
                    new_attrs = {'long_name': 'signal', 'wavelength': _wavelength, 'detection_mode': 1, 'units': 'a.u.', 'eta_GHK': str(calib)}
                    lxarray['signal_%s' % channel_total_pc] = signal_total_pc.assign_attrs(new_attrs)
                    # LVDR
                    eta_pc = calib['eta_pc'] / calib['K']
                    ratio_pc = (signal_Rpc / signal_Tpc) / eta_pc
                    lvdr_pc = ((calib['GT'] + calib['HT']) * ratio_pc - (calib['GR'] + calib['HR'])) / ((calib['GR'] - calib['HR']) - (calib['GT'] - calib['HT']) * ratio_pc)
                    new_attrs = {'long_name': 'Linear Volume Depolarization Ratio', 'detection_mode': 1, 'wavelength': _wavelength, 'units': '$\#$'}
                    LVDR['pc'] = lvdr_pc.assign_attrs(new_attrs)
                    channel_lvd_pc = '%d%sp' % (_wavelength, field_[0])
                    lxarray['lvd_%s' % channel_lvd_pc] = LVDR['pc']
                else:
                    print("WARNING: photoncounting signals not merged for wavelength %.1f. R and/or T are missing" % _wavelength)        

    logger.info('End Merge Polarized Channels')
    return lxarray


""" GLUING
"""
def gluing(rcs_an, rcs_pc, height, range_min=1500, range_max=5000, half_window=375, debug=False):    
    """_summary_

    Args:
        rcs_an (_type_): _description_
        rcs_pc (_type_): _description_
        height (_type_): _description_
        range_min (int, optional): _description_. Defaults to 1500.
        range_max (int, optional): _description_. Defaults to 5000.
        half_window (int, optional): _description_. Defaults to 375.
        debug (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    
    intecerpt_value = {True: 1, False: 0}
    idx_sel = np.logical_and(height>range_min, height<range_max)
    height_sel = height[idx_sel]        
    rcs_an_sm = sp.signal.savgol_filter(rcs_an[idx_sel]/np.nanmean(rcs_an[idx_sel]), 11, 3) 
    rcs_pc_sm = sp.signal.savgol_filter(rcs_pc[idx_sel]/np.nanmean(rcs_pc[idx_sel]), 11, 3) 
    nrcs_sel={'an': rcs_an_sm,'pc':rcs_pc_sm}
        
    nan_array = np.nan*np.ones(len(height_sel))
    slope={'an':nan_array, 'pc': nan_array} 
    intercept={'an':nan_array, 'pc': nan_array} 
    r={'an':nan_array, 'pc': nan_array} 
    
    for mode in ['an','pc']:
        slope[mode] = np.ones([len(height_sel)])*np.nan                
        for idx_h, h_ in enumerate(height_sel):   
            idx_slide = np.logical_and(height_sel> (h_ - half_window), height_sel<(h_ + half_window))            
            slope[mode][idx_h], _, _, _, _ = sp.stats.linregress(nrcs_sel[mode][idx_slide],height_sel[idx_slide])
    
    slope_diff = np.abs(slope['pc']-slope['an'])    
    idx_min = np.abs(slope_diff-slope_diff.min()).argmin()
    height_gluing = height_sel[idx_min]
    
    if isinstance(height_gluing, collections.abc.Iterable):
        if len(height_gluing)>1:
            height_gluing = height_gluing[0]
    
    idxs_gluing = np.logical_and(height> (height_gluing - half_window), height<(height_gluing + half_window))
    slope_gluing, intercept_gluing, r_gluing, _, _ = sp.stats.linregress(rcs_an[idxs_gluing],rcs_pc[idxs_gluing])
    if r_gluing > 0.9:
        rcs_gluing_an = rcs_an*slope_gluing + intercept_gluing
        rcs_gl = np.concatenate([rcs_gluing_an[height<height_gluing], rcs_pc[height>=height_gluing]])
    else:
        rcs_gluing = []
        print('Error: linear fit not good enough for gluing: r= %0.3f' % r_gluing)
    
    return rcs_gl    


def glue_channels(dataset):
    """
    It merges the analog and photoncounting channels of a given dataset

    Parameters
    ----------
    dataset: xarray.Dataset from lidar.preprocessing() (xarray.Dataset)    

    Returns
    -------
    dataset: xarray.Dataset with new varaibles
    """

    telescope_number = {'x': 0, 'f': 1, 'n': 2}
    mode_number = {'a': 0, 'p': 1, 'g': 2}
    polarization_number = {'t': 0, 'p': 1, 's': 2}

    if not 'time' in dataset.dims.keys():
        dsgs = {}
        channels_pc = [channel_ for channel_ in dataset.channel_id.values if channel_[-1]=='p']
        wavelength, telescope, polarization, mode, channel_name = [], [], [], [], []
        for channel_pc in channels_pc:    
            nchan_pc = dataset['channel'][dataset.channel_id== channel_pc].item()
            channel_an = '%sa' % channel_pc[0:-1]
            print(channel_pc, channel_an)
            if channel_an in dataset.channel_id:
                nchan_an = dataset['channel'][dataset.channel_id == channel_an].item()        
                rcs_an = dataset['corrected_rcs_%02d' % nchan_an]
                rcs_pc = dataset['corrected_rcs_%02d' % nchan_pc]
                dsgs['rcs_%sg' % channel_pc[0:-1]] = xr.apply_ufunc(gluing, rcs_an, rcs_pc, dataset['range'], dask='allowed',
                                    input_core_dims=[['range'],['range'],['range']], output_core_dims=[['range']])
                wavelength.append(int(channel_pc[0:-3]))
                channel_name.append('rcs_%sg' % channel_pc[0:-1])
                telescope.append(telescope_number[channel_pc[-3]])
                polarization.append(polarization_number[channel_pc[-2]])
                mode.append(mode_number['g'])        
                
        #FALTA CONVERTIR ESTO EN UN DATASET Y PEGARLO AL INPUT
        new_dataset = xr.Dataset({'wavelength': (['channel_name'], wavelength),
                            'detection_mode': (['channel_name'], mode),
                            'polarization': (['channel_name'], polarization)},
                            coords={'channel_name': channel_name, 'range': dataset['range']})
        for key_ in dsgs.keys():
            new_dataset[key_] = dsgs[key_]
    return new_dataset


""" PLOT LIDAR
"""
def plot_lidar_channels(filelist, dcfilelist, channels2plot, plt_conf, figdirectory):
    """
    Quicklook maker of lidar measurements.
    Inputs:
    - filelist: List of radar files (i.e, '/drives/c/*ZEN*.LC?') (str).
    - dcfilelist: List of dark measurement radar files (i.e, '/drives/c/*ZEN*.LC?') (str).
    - channels2plot: Array of numbers corresponding to the lidar channels (integer).
    - plt_conf: dictionary with plot configuration (dict).
    - figdirectory: directory to save the figure. Date tree will be created (str).
    Outputs:
    - None
    """

    #Channels linked to depolarization technique
    depo_channels = {'mulh': ['rcs_532xta', 'lvd_532xa'], 'alh': ['rcs_355fta', 'lvd_355fa', 'rcs_355nta', 'lvd_355na', 'rcs_532nta', 'lvd_532na'], 'vlt': ['rcs_355xta', 'lvd_355xa']}

    # Dictionaries
    # Detection mode (analog, photoncounting)
    mode = {0: 'a', 1: 'p'}
    # Polarization component (total, parallel, cross)
    pol = {0: 't', 1: 'p', 2: 'c'}

    # Font size of the letters in the figure
    matplotlib.rcParams.update({'font.size': 16})

    # Read the list of files to plot
    # --------------------------------------------------------------------
    lidarnick = os.path.basename(filelist).split('_')[0]
    
    #Check if merge polarizing channels is necessary
    # pdb.set_trace()
    # set depo_required False  
    depo_required = False
    
    # for channel_ in channels2plot:
    #     if (channel_ == depo_channels[lidarnick]).any():
    #         depo_required = True 
    lxarray = preprocessing(rs_fl=filelist, dc_fl=dcfilelist, deadtime_flag=False, channels=channels2plot, merge_flag=depo_required)
    lidar_name = lxarray.system.lower()  
    if channels2plot == 'all':
        channels2plot = lxarray.channel.values
    else:
        channels2plot = channels2plot
      

    if lxarray != None:
        # if ldate < np.datetime64('2019-10-15'):
        #     #Mininum value on the colorbar
        #     Vmin = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
        #     #Maximum value on the colorbar
        #     Vmax = {0: 1e9, 1: 5e10, 2: 5e8, 3: 5e8, 4: 1e9, 5: 5e8, 6: 1e9, 7: 5e8, 8: 1e7, 9: 5e8}        
        # else:            
        # Mininum value on the colorbar
        Vmin_mulhacen = {'532xpc': 0, '532xpa': 0, '532xca': 0, '532xcp': 0, '355xta': 0, '355xtp': 0, '1064xta': 0}
        Vmin_veleta = {'355xpp': 0, '355xpa': 0, '355xsa': 0, '355xsp': 0, '387xta': 0, '387xtp': 0}

        Vmin_alhambra = {'1064fta': 0, '1061fta': 0, '532fta': 0, '532ftp': 0, '531fta': 0, '531ftp': 0, 
                                '355fpa': 0, '355fpp': 0, '355fsa': 0, '355fsp': 0,'354fta': 0, 
                                '354ftp': 0, '408fta': 0, '408ftp': 0, '1064nta': 0, '532npa': 0, 
                                '532npp': 0, '532nsa': 0, '532nsp': 0, '355npa': 0, '355npp': 0, 
                                '355nsa': 0, '355nsp': 0, '387nta': 0, '387ntp': 0, '607nta': 0, '607ntp': 0}
        Vmax_mulhacen = {'532xpc': 1e7, '532xpa': 1e7, '532xca': 1e7, '532xcp': 1e7, '355xta': 1e7, '355xtp': 1e7, '1064xta': 1e7}
        Vmax_veleta = {'355xpp': 1e7, '355xpa': 1e7, '355xsa': 1e7, '355xsp': 1e7, '387xta': 1e7, '387xtp': 1e7}
        Vmax_alhambra = {'1064fta': 1e7, '1061fta': 1e7, '532fta': 1e7, '532ftp': 1e7, '531fta': 1e7, '531ftp': 1e7, 
                                '355fpa': 1e7, '355fpp': 1e7, '355fsa': 1e7, '355fsp': 1e7,'354fta': 1e7, 
                                '354ftp': 1e7, '408fta': 1e7, '408ftp': 1e7, '1064nta': 1e7, '532npa': 1e7, 
                                '532npp': 1e7, '532nsa': 1e7, '532nsp': 1e7, '355npa': 1e7, '355npp': 1e7, 
                                '355nsa': 1e7, '355nsp': 1e7, '387nta': 1e7, '387ntp': 1e7, '607nta': 1e7, '607ntp': 1e7}
        Vmin_dict = {'mulhacen': Vmin_mulhacen, 'alhambra':Vmin_alhambra,'veleta':Vmin_veleta,'kascal':Vmin_veleta}
        Vmax_dict = {'mulhacen': Vmax_mulhacen, 'alhambra':Vmax_alhambra,'veleta':Vmax_veleta,'kascal':Vmin_veleta}

        # Minimum value on the colorbar
        Vmin = Vmin_dict[lidar_name]
        
        # Maximum value on the colorbar
        if 'vmax' in plt_conf.keys():
            Vmax = plt_conf['vmax']
        else:
            Vmax = Vmax_dict[lidar_name]

        # One figure per variable
        # pdb.set_trace()
        # --------------------------------------------------------------------
        for channel_ in channels2plot:
            print(channel_)
            # channel_ = '355xsa'
            print('Colorbar range: %f - %f' % (Vmin[channel_], Vmax[channel_]))
            # Create channel string
            channelstr = channel_

            # Create Figure
            fig, axes = plt.subplots(1,1,figsize=(15, 5))
            
            # Plot
            cmap = matplotlib.cm.jet
            bounds = np.linspace(Vmin[channel_], Vmax[channel_], 128)
            norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
            range_km = lxarray.range / 1000.
            # RCSP  = (lxarray['signal_'+channel_]*(lxarray['range'].values**2)).T.values
            # Range = range_km.values
            # DateTimes = lxarray.time.values
            # pdb.set_trace()
            # LIDARDatas = xr.Dataset(
            #                       {
            #                         "RCSP":(("range",'deltatime'),RCSP[0:1333,:]),
    
            #                       },
            #                         coords={"range": list(Range[0:1333]),"deltatime": list(DateTimes)}, 
            #                     )        
            # RCSPda = LIDARDatas.RCSP
            # plt.semilogy(Range,RCSP)
            
            # q = RCSPda.plot.contourf(
            #             ax=axes,
            #             # vmin = 0,
            #             # vmax = 1.5,

            #             # y="Range",
            #             levels=13,
            #             robust=True,
            #             cmap = 'jet',
            #             add_colorbar=False,
            #             extend='both'
            #                   )
            # pdb.set_trace()
            q = axes.pcolormesh(lxarray.time, range_km,
                                (lxarray['signal_'+channel_]*(lxarray['range'].values**2)).T,
                                cmap=cmap,
                                # norm=norm,
                                # vmin=Vmin[channel_],
                                # vmax=Vmax[channel_]
                                )
            q.cmap.set_over('white')
            # pdb.set_trace()

            cb = plt.colorbar(q, ax=axes,
                              # ticks=bounds,
                              extend='max',
                              format='%.1e')
            axes.set_ylim(plt_conf['y_min'], plt_conf['y_max'])


            
            # search for gaps in data
            # --------------------------------------------------------------------     
            if plt_conf['gapsize'] == 'default':
                dif_time = lxarray['time'].values[1:] - lxarray['time'].values[0:-1]
                GAP_SIZE = 2 * int(np.ceil((np.median(dif_time).astype(
                    'timedelta64[s]').astype('float') / 60)))  # GAP_SIZE is defined as the median of the resolution fo the time array (in minutes)
                print('GAP_SIZE parameter automatically retrieved to be %d.' % GAP_SIZE)
            else:
                GAP_SIZE = int(plt_conf['gapsize'])
                print('GAP_SIZE set by the user: %d (in minutes)' % GAP_SIZE)
            dttime = np.asarray([dt.datetime.utcfromtimestamp((
                time_ - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's'))
                for time_ in lxarray['time'].values])
            # pdb.set_trace()

            plot.gapsizer(plt.gca(), dttime, lxarray['range'], GAP_SIZE, '#c7c7c7')
            # pdb.set_trace()

            # Setting axes 
            # --------------------------------------------------------------------
            mf = matplotlib.ticker.FuncFormatter(plot.tmp_f)
            axes.xaxis.set_major_formatter(mf)
            hours = mdates.HourLocator(range(0, 25, 3))
            date_fmt = mdates.DateFormatter('%H')
            axes.xaxis.set_major_locator(hours)
            axes.xaxis.set_major_formatter(date_fmt)
            min_date = lxarray['time'].values.min()
            max_date = lxarray['time'].values.max()
            axes.set_xlim(min_date.astype('datetime64[D]'),
                          max_date.astype('datetime64[D]') + np.timedelta64(1, 'D'))
            axes.set_ylim(plt_conf['y_min'], plt_conf['y_max'])
            plt.grid(True)
            axes.set_xlabel('Time, $[UTC]$')
            axes.set_ylabel('Height, $[km agl]$')
            cb.ax.set_ylabel('Range corrected signal, $[a.u.]$')
            # pdb.set_trace()


            # title
            # ----------------------------------------------------------------------------
            datestr = lxarray.time[0].min().values.astype('str').split('T')[0]
            plt_conf['title1'] = '%s %s' % (lxarray.attrs['instrument_id'], channelstr)
            plot.title1(plt_conf['title1'], plt_conf['coeff'])
            plot.title2(datestr, plt_conf['coeff'])
            plot.title3('{} ({:.1f}N, {:.1f}E)'.format(lxarray.attrs['site_location'],
                                                       float(lxarray.attrs['geospatial_lat_min']),
                                                       float(lxarray.attrs['geospatial_lon_min'])),
                plt_conf['coeff'])

            # logo
            # ----------------------------------------------------------------------------
            plot.watermark(fig, axes, alpha=0.5, scale=15, ypos=315)

            debugging = False

            if debugging:
                plt.show()
            else:
                # create output folder
                # --------------------------------------------------------------------
                year = datestr[0:4]
                fulldirpath = os.path.join(figdirectory, channelstr, year)
                if np.logical_not(os.path.exists(fulldirpath)):
                    mkpath(fulldirpath)
                    print('fulldirpath created: %s' % fulldirpath)
                figstr = '%s_%s_rcs-%s_%s.png' % (
                    lxarray.attrs['lidarNick'], lxarray.attrs['dataversion'],
                    channelstr, datestr.replace('-', ''))
                finalpath = os.path.join(fulldirpath, figstr)
                print('Saving %s' % finalpath)
                plt.savefig(finalpath, dpi=100, bbox_inches='tight')
                if os.path.exists(finalpath):
                    print('Saving %s...DONE!' % finalpath)
                else:
                    print('Saving %s... error!' % finalpath)
                plt.close()

        # plot depolarization
        if depo_required:
            for _key in depo_channels[lidar_name]:
                if  _key in lxarray.keys():
                    print('Current plot %s' % _key)
                    Vmin = 0. #{'rcs_532xta': 0., 'lvd_532xa': 0.}
                    Vmax = {'rcs_532xta': 2.5e6, 'lvd_532xa': 0.4, 'rcs_355fta': 2.5e6, 'lvd_355fa': 0.4, 'rcs_355nta': 2.5e6, 'lvd_355na': 0.4, 'rcs_532nta': 2.5e6, 'lvd_532na': 0.4}
                    signal_type = _key.split('_')[0]
                    channel_name = _key.split('_')[-1]

                    # Create Figure
                    fig = plt.figure(figsize=(15, 5))
                    axes = fig.add_subplot(111)
                    # Plot
                    cmap = matplotlib.cm.jet
                    bounds = np.linspace(Vmin, Vmax[_key], 128)
                    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
                    range_km = lxarray.range / 1000.
                    q = axes.pcolormesh(lxarray.time, range_km, lxarray[_key].T,
                                        cmap=cmap,
                                        vmin=Vmin[_key],
                                        vmax=Vmax[_key])
                    q.cmap.set_over('white')
                    cb = plt.colorbar(q, ax=axes,
                                    # ticks=bounds,
                                    extend='max',
                                    format='%.1e')

                    # search for gaps in data
                    # --------------------------------------------------------------------     
                    if plt_conf['gapsize'] == 'default':
                        dif_time = lxarray['time'].values[1:] - lxarray['time'].values[0:-1]
                        GAP_SIZE = 2 * int(np.ceil((np.median(dif_time).astype(
                            'timedelta64[s]').astype('float') / 60)))  # GAP_SIZE is defined as the median of the resolution fo the time array (in minutes)
                        print('GAP_SIZE parameter automatically retrieved to be %d.' % GAP_SIZE)
                    else:
                        GAP_SIZE = int(plt_conf['gapsize'])
                        print('GAP_SIZE set by the user: %d (in minutes)' % GAP_SIZE)
                    dttime = np.asarray([dt.datetime.utcfromtimestamp(
                        (time_ - np.datetime64('1970-01-01T00:00:00Z')) /
                        np.timedelta64(1, 's')) for time_ in lxarray['time'].values])
                    plot.gapsizer(plt.gca(), dttime, lxarray['range'], GAP_SIZE, '#c7c7c7')

                    # Setting axes 
                    # --------------------------------------------------------------------
                    mf = matplotlib.ticker.FuncFormatter(plot.tmp_f)
                    axes.xaxis.set_major_formatter(mf)
                    hours = mdates.HourLocator(range(0, 25, 3))
                    date_fmt = mdates.DateFormatter('%H')
                    axes.xaxis.set_major_locator(hours)
                    axes.xaxis.set_major_formatter(date_fmt)
                    min_date = lxarray['time'].values.min()
                    max_date = lxarray['time'].values.max()
                    axes.set_xlim(min_date.astype('datetime64[D]'), max_date.astype('datetime64[D]') + np.timedelta64(1, 'D'))
                    axes.set_ylim(plt_conf['y_min'], plt_conf['y_max'])
                    plt.grid(True)
                    axes.set_xlabel('Time, $[UTC]$')
                    axes.set_ylabel('Height, $[km agl]$')
                    cb.ax.set_ylabel('%s, %s' % (lxarray[_key].attrs['long_name'], lxarray[_key].attrs['units']))

                    # title
                    # ----------------------------------------------------------------------------
                    datestr = lxarray.time[0].min().values.astype('str').split('T')[0]
                    plt_conf['title1'] = '%s %s' % (lxarray.attrs['instrument_id'], _key)
                    plot.title1(plt_conf['title1'], plt_conf['coeff'])
                    plot.title2(datestr, plt_conf['coeff'])
                    plot.title3('{} ({:.1f}N, {:.1f}E)'.format(lxarray.attrs['site_location'],
                                                            float(lxarray.attrs['geospatial_lat_min']),
                                                            float(lxarray.attrs['geospatial_lon_min'])),
                                plt_conf['coeff'])

                    # logo
                    # ----------------------------------------------------------------------------
                    plot.watermark(fig, axes, alpha=0.5, scale=15, ypos=315)
                    
                    debugging = False

                    if debugging:
                        plt.show()
                    else:
                        # create output folder
                        # --------------------------------------------------------------------
                        year = datestr[0:4]
                        fulldirpath = os.path.join(figdirectory, year)
                        if np.logical_not(os.path.exists(fulldirpath)):
                            mkpath(fulldirpath)
                            print('fulldirpath created: %s' % fulldirpath)
                        figstr = '%s_%s_%s-%s_%s.png' % (
                            lxarray.attrs['lidarNick'],
                            lxarray.attrs['dataversion'],
                            signal_type, channel_name, datestr.replace('-', ''))
                        finalpath = os.path.join(fulldirpath, figstr)
                        print('Saving %s' % finalpath)
                        plt.savefig(finalpath, dpi=100, bbox_inches='tight')
                        if os.path.exists(finalpath):
                            print('Saving %s...DONE!' % finalpath)
                        else:
                            print('Saving %s... error!' % finalpath)
                        plt.close()
    return


def daily_quicklook(filelist, dcfilelist, figdirectory, channels2plot='all', plot_depo=True, **kwargs):
    """
    Formatted daily quicklook of RPG Cloud Radar measurements.
    Inputs:
    - filelist: List of radar files (i.e, '/drives/c/*ZEN*.LC?') (str)
    - figdirectory: Array of numbers corresponding to the moment of the Doppler spectra. (integer)
    - kwargs:
        + gapsize
        + y_min
        + y_max
        + coeff
        + Vmaxn where n is the channel (e.g., Vmax0)        
    Outputs:
    - None
    """
    """ Get Input Arguments """
    lidar_nick = os.path.basename(filelist).split('_')[0]

    gapsize = kwargs.get("gapsize", 'default')
    y_min, y_max = kwargs.get("y_min", 0), kwargs.get("y_max", 14)
    coeff = kwargs.get("coeff", 2)    
    Vmax_dict = {}
    Vmax_dict['mulh'] = {'532xpc': kwargs.get('vmax_532xpc', 4e6), '532xpa': kwargs.get("vmax_532xpa", 5e10), '532xca': kwargs.get("vmax_532xca", 5e10), 
                        '532xcp': kwargs.get('vmax_532xcp', 1e7), '355xta': kwargs.get("vmax_355xta", 1e7), '355xtp': kwargs.get("vmax_355xtp", 1e7), 
                        '1064xta': kwargs.get("vmax_1064xta", 1e7)}
    Vmax_dict['alh'] = {'1064fta': kwargs.get('vmax_1064fta', 1e7), '1061fta': kwargs.get('vmax_1061fta', 1e7), '532fta': kwargs.get('vmax_532fta', 1e7), 
                            '532ftp': kwargs.get('vmax_532ftp', 1e7), '531fta': kwargs.get('vmax_531fta', 1e7), '531ftp': kwargs.get('vmax_531ftp', 1e7), 
                            '355fpa': kwargs.get('vmax_355fpa', 1e7), '355fpp': kwargs.get('vmax_355fpp', 1e7), '355fsa': kwargs.get('vmax_355fsa', 1e7), 
                            '355fsp': kwargs.get('vmax_355fsp', 1e7), '354fta': kwargs.get('vmax_354fta', 1e7), '354ftp': kwargs.get('vmax_354ftp', 1e7),
                            '408fta': kwargs.get('vmax_408fta', 1e7), '408ftp': kwargs.get('vmax_408ftp', 1e7), '1064nta': kwargs.get('vmax_1064nta', 1e7), 
                            '532npa': kwargs.get('vmax_532npa', 1e7), '532npp': kwargs.get('vmax_532npp', 1e7), '532nsa': kwargs.get('vmax_532nsa', 1e7), 
                            '532nsp': kwargs.get('vmax_532nsp', 1e7), '355npa': kwargs.get('vmax_355npa', 1e7), '355npp': kwargs.get('vmax_355npp', 1e7), 
                            '355nsa': kwargs.get('vmax_355nsa', 1e7), '355nsp': kwargs.get('vmax_355nsp', 1e7), '387nta': kwargs.get('vmax_387nta', 1e7), 
                            '387ntp': kwargs.get('vmax_387ntp', 1e7), '607nta': kwargs.get('vmax_607nta', 1e7), '607ntp': kwargs.get('vmax_607ntp', 1e7)}
    Vmax_dict['vlt'] = {'355xpp': kwargs.get('vmax_355xpp', 4e6), '355xpa': kwargs.get("vmax_355xpa", 5e10), '355xsa': kwargs.get("vmax_532xsa", 5e10), 
                        '355xsp': kwargs.get('vmax_355xsp', 1e7), '387xta': kwargs.get("vmax_387xta", 1e7), '387xtp': kwargs.get("vmax_387xtp", 1e7),}
    
    Vmax_dict['kal'] = {'355xpp': kwargs.get('vmax_355xpp', 4e6), '355xpa': kwargs.get("vmax_355xpa", 5e10), '355xsa': kwargs.get("vmax_532xsa", 5e10), 
                        '355xsp': kwargs.get('vmax_355xsp', 1e7), '387xta': kwargs.get("vmax_387xta", 1e7), '387xtp': kwargs.get("vmax_387xtp", 1e7),}
    Vmax = Vmax_dict[lidar_nick]

    plt_conf = {'gapsize': gapsize, 'y_min': y_min, 'y_max': y_max, 'coeff': coeff, 'vmax': Vmax}        
    channels = kwargs.get("channels", "all")
    if channels is None:
        channels = "all"
    # plot_lidar_channels(filelist, dcfilelist, channels2plot, plt_conf, figdirectory, plot_depo=plot_depo)
    plot_lidar_channels(filelist, dcfilelist, channels2plot, plt_conf, figdirectory)


def date_quicklook(dateini, dateend, lidar='mulhacen', channels2plot='default', path1a='GFATserver', figpath='GFATserver', **kwargs):
    """
    Formatted daily quicklook of lidar measurements for hierarchy GFAT data.
    Inputs:
    - path1a: path where 1a-level data are located.
    - figpath: path where figures are saved.
    - Initial date [yyyy-mm-dd] (str). 
    - Final date [yyyy-mm-dd] (str).

    Outputs: 
    - None
    """

    channels_DEFAULT={'alhambra': ['532fta', '532npa'], 'mulhacen': ['532xpa', '1064xta'], 'veleta': ['355xpa']}

    def filepattern(lidar, ftype, current_date):
        """
        lidar: lidar name
        ftype: rs, dc
        """
        filename = '%s_1a_P%s_rcs*%s*.nc' % (LIDAR_SYSTEM[lidar.upper()]['LIDARNICK'], \
            ftype, dt.datetime.strftime(current_date, '%y%m%d'))
        return filename
    
    if channels2plot == 'default':        
        channels2plot = np.asarray(channels_DEFAULT[lidar])
    else:
        if type(channels2plot) is list:
            channels2plot = np.asarray(channels2plot)    
            
    if path1a == 'GFATserver':
        path1a = '/mnt/NASGFAT/datos/%s/1a' % lidar.upper()

    if figpath == 'GFATserver':
        figpath = '/mnt/NASGFAT/datos/%s/quicklooks' % lidar.upper()

    inidate = dt.datetime.strptime(dateini, '%Y%m%d')
    enddate = dt.datetime.strptime(dateend, '%Y%m%d')

    period = enddate - inidate

    for _day in range(period.days + 1):
        current_date = inidate + dt.timedelta(days=_day)
        filename = filepattern(lidar, 'rs', current_date)        
        filelist = os.path.join(path1a, '%d' % current_date.year, '%02d' % current_date.month,'%02d' % current_date.day, filename)
        dcfilename = filepattern(lidar, 'dc', current_date)       
        dcfilelist = os.path.join(path1a, '%d' % current_date.year,'%02d' % current_date.month,'%02d' % current_date.day, dcfilename)
        daily_quicklook(filelist, dcfilelist, figpath, channels2plot = channels2plot, lidar=lidar, **kwargs)


""" DEAD TIME ESTIMATION
"""
def dead_time_estimation(rs_fl, dc_fl):
    """

    """

    # TODO: Implementar
    try:
        print("To be implemented")
        # Read RS and DC data
        #rs_ds = reader_xarray(rs_fl)
        #dc_ds = reader_xarray(dc_fl)
    except Exception as e:
        print("ERROR in dead_time_estimation. %s" % str(e))


def main():
    parser = argparse.ArgumentParser(description="usage %prog [arguments]")
    parser.add_argument("-i", "--initial_date",
                        action="store",
                        dest="dateini",
                        required=True,
                        help="Initial date [example: '20190131'].")
    parser.add_argument("-e", "--final_date",
                        action="store",
                        dest="dateend",
                        default=".",
                        help="Final date [example: '20190131'].")
    parser.add_argument("-d", "--datadir",
                        action="store",
                        dest="path1a",
                        default="GFATserver",
                        help="Path where date-hierarchy files are located [example: '~/data/1a'].")
    parser.add_argument("-f", "--figuredir",
                        action="store",
                        dest="figpath",
                        default="GFATserver",
                        help="Path where figures will be saved [example: '~/radar/quicklooks'].")
    args = parser.parse_args()

    date_quicklook(args.dateini, args.dateend, path1a=args.path1a,
                   figpath=args.figpath)


if __name__ == "__main__":
    main()
