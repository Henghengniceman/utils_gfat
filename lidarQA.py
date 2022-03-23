import os
import sys
import glob
import platform
import gzip
import shutil
from distutils.dir_util import mkpath
import itertools
import tempfile
import math
import re
import scipy
import time
import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da
import datetime as dt
from scipy import stats
from scipy.signal import correlate
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
import matplotlib
import argparse
import fileinput
import logging
import pdb

MODULE_DIR = os.path.dirname(sys.modules[__name__].__file__)
sys.path.insert(0, MODULE_DIR)
import lidar
import utils
from grawmet import reader_typeA, reader_typeB  
import atmo
from utils_gfat import config
from utils_gfat.ecmwf import meteo_ecmwf
#from lidar_processing.lidar_processing import helper_functions
#from . import lidar
#from . import utils
#from .grawmet import grawmet_reader
#from .lidar_processing.lidar_processing import helper_functions

""" add GHK folder to path """
ghk_dn = os.path.join(MODULE_DIR, "GHK_0.9.8e5_Py3.7")
ghk_sett_dn = os.path.join(ghk_dn, "system_settings")
ghk_inp_fn = os.path.join(ghk_sett_dn, "mulhacen_run.py")
sys.path.append(ghk_dn)

"""
DESCRIPCION DEL MODULO
"""

"""
44 extend_meteo_profile
99 meteo_ecmwf
152 mol_attenuated_backscatter
200 rayleigh_fit
478 setup_rayleigh_fit
523 telecover
675 setup_telecover
732 main
"""

LOG_DATE_FMT = '%Y-%m-%d %H:%M:%S'
LOG_FMT= '%(funcName)s(). %(lineno)s: %(message)s'
#LOG_DIR = 'logs'
#LOG_FILENAME = 'raw2l1.log'

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))
#logging.basicConfig(format='%(funcName)s(). %(lineno)s: %(message)s', level=logging.INFO)

"""
Directories
"""
# Root Directory (in NASGFAT)  according to operative system
# GFATNAS Data Dir:
nas_data_dn = config.DATA_DN

# Run Dir
# TODO: revisar este run_dir
run_dir = os.getcwd()

"""
General Info
"""
""" Lidar Systems """
lidar_system = {'MULHACEN': 'mhc', 'VELETA': 'vlt'}
lidar_id = "gr"
lidar_location = "Granada"

""" Types of Lidar Measurements """
measurement_types = {"RS": "Prs", "DC": "Pdc", "OT": "Pot", "DP-P45": "Pdp-45",
                     "DP-N45": "Pdp-N45"}

""" Polarization and Detection_modes """
pol_name = {0: 'total', 1: 'parallel', 2: 'perpendicular'}
pol_id = {0: 't', 1: 'p', 2: 's'}
det_mod_name = {0: 'analog', 1: 'photoncounting', 2: 'gluing'}
det_mod_id = {0: 'a', 1: 'p', 2: 'g'}


"""
Helper Functions
"""


def extend_meteo_profile(P, T, heights):
    """
    If our Pressure and Temperature vectors are not the same size as our lidar data, then we include standard atmosphere values
    And if they are bigger than our data, we shorten them until they are the same size
    Inputs:
    - T: temperature vector (numpy array)
    - P: pressure vector (numpy array)
    - heights: heights vector (numpy array)
    Outputs:
    - atmospheric_profiles: dataset with pressure and temperature data (xarray dataset)
    """
    # standard atmosphere profile:
    Tsa = np.ones(heights.size)*np.nan
    Psa = np.ones(heights.size)*np.nan
    for i, _height in enumerate(heights):
        sa = atmo.standard_atmosphere(_height)
        Psa[i] = sa[0]
        Tsa[i] = sa[1]
        
    if P.size == Psa.size:  # if they are the same size, we leave them be
        extended_P = P

    if T.size == Tsa.size:  # if they are the same size, we leave them be
        extended_T = T

    if P.size > Psa.size:  # If our pressure vector is bigger than 'heights', we make it the same size
        maxsa = Psa.size
        extended_P = np.ones(heights.size)*np.nan
        for i in range(maxsa):
            extended_P[i] = P[i]

    if T.size > Tsa.size:  # If our temperature vector is bigger than 'heights', we make it the same size
        maxsa = Tsa.size
        extended_T = np.ones(heights.size)*np.nan
        for i in range(maxsa):
            extended_T[i] = T[i]
    
    if P.size < Psa.size:  # if we don't have enough data to make a full pressure profile
        extended_P = P
        maxh = P.size  # number of the last data
        for i in range(Psa.size-maxh):
            extended_P = np.append(extended_P, Psa[maxh+i])  # we use standard atmosphere as our pressure profile

    if T.size < Tsa.size:  # if we don't have enough data to make a full temperature profile
        extended_T = T
        maxh = T.size  # number of the last data
        for i in range(Tsa.size-maxh):
            extended_T = np.append(extended_T, Tsa[maxh+i])  # we use standard atmosphere as our temperature profile

    atmospheric_profiles = xr.Dataset({'pressure': (['range'], extended_P),
                                       'temperature': (['range'], extended_T)},
                                      coords={'range': heights})

    return atmospheric_profiles

def extend_scaled_meteo_profiles(radiosonde, heights):
    """
    If maximum radiosonde height is below the lidar range,
    scaled standard atmosphere is used to fulfill the profiles.

    Parameters
    ----------
    radiosonde: xarray.Dataset
        from grawmet_reader (xarray.Dataset)
    heights: array
        lidar ranges (m)

    Returns
    -------
    atmospheric_profiles: xarray.Dataset
        pressure and temperature data (xarray dataset)
    """
    
    # standard atmosphere profile:
    Tsa = np.ones(heights.size)*np.nan
    Psa = np.ones(heights.size)*np.nan
    for i, _height in enumerate(heights):
        sa = atmo.standard_atmosphere(_height)
        Psa[i] = sa[0]
        Tsa[i] = sa[1]    
    max_sonde_idx = np.squeeze(np.where(np.max(radiosonde['range'].values) == radiosonde['range'].values))

    # Reindex Sonde Heights to Lidar Ranges
    meteo_profiles = radiosonde.interp(range=heights, kwargs={'fill_value': 'extrapolate'})
    interp_sonde_range = meteo_profiles['range'].values
    top_pressure_idx = np.squeeze(np.where(abs(interp_sonde_range-radiosonde['range'].values[max_sonde_idx])==abs(interp_sonde_range-radiosonde['range'].values[max_sonde_idx]).min())[0])
    top_pressure_val = np.squeeze(interp_sonde_range[abs(interp_sonde_range-radiosonde['range'].values[max_sonde_idx])==abs(interp_sonde_range-radiosonde['range'].values[max_sonde_idx]).min()])
    top_temperature_val = meteo_profiles['temperature'].values[top_pressure_idx]

    #Radiosonde data:
    bottom_T = meteo_profiles['temperature'].values[0:top_pressure_idx]
    bottom_P = meteo_profiles['pressure'].values[0:top_pressure_idx]*100 #Convert to Pa

    #Extended profile with scaled standard atmosphere:
    extension_Tsa = Tsa[top_pressure_idx:]
    extension_Psa = Psa[top_pressure_idx:]    
    up_T = bottom_T[-1]*(extension_Tsa/Tsa[top_pressure_idx])
    up_P = bottom_P[-1]*(extension_Psa/Psa[top_pressure_idx])

    extended_T = np.concatenate([bottom_T, up_T])
    extended_P = np.concatenate([bottom_P, up_P])

    atmospheric_profiles = xr.Dataset({'pressure': (['range'], extended_P),
                                       'temperature': (['range'], extended_T)},
                                      coords={'range': heights})
    return atmospheric_profiles


def molecular_properties(wavelength, pressure, temperature, heights, component='total'):
    """
    Molecular Attenuated  Backscatter: beta_mol_att = beta_mol * Transmittance**2

    Parameters
    ----------
    wavelength: int, float
        wavelength of our desired beta molecular attenuated
    pressure: array
        pressure profile
    temperature: array
        temperature profile
    heights: array
        heights profile

    Returns
    -------
    beta_molecular_att: array
        molecular attenuated backscatter profile
    """

    """ molecular backscatter and extinction """
    beta_mol = atmo.molecular_backscatter(wavelength, pressure, temperature, component=component)
    alfa_mol = atmo.molecular_extinction(wavelength, pressure, temperature)
    lr_mol = atmo.molecular_lidar_ratio(wavelength)
    
    """ transmittance """
    transmittance = atmo.transmittance(alfa_mol, heights)

    """ attenuated molecular backscatter """
    att_beta_mol = atmo.attenuated_backscatter(beta_mol, transmittance)

    mol_properties = xr.Dataset({'molecular_beta': (['range'], beta_mol),
                 'molecular_alpha': (['range'], alfa_mol), 
                 'attenuated_molecular_beta': (['range'], att_beta_mol),
                 'molecular_lidar_ratio': lr_mol},
                 coords={'range': heights})

    return mol_properties

def ghk_output_reader(filepath):
    GHK = {'GR': None, 'GT': None, 'HR': None, 'HT': None, 'K1': None, 
           'K2': None, 'K3': None, 'K4': None, 'K5': None, 'K6': None, 'K7': None}
    if os.path.isfile(filepath):
        f = open(filepath, 'r')
        for line in f:
            line = line.strip()
            if line[0:2] == 'GR':
                break
        line = f.readline().replace(' ', '').split(',')
        print(line)
        GHK['GR'], GHK['GT'], GHK['HR'], GHK['HT'], GHK['K1'], GHK['K2'], \
        GHK['K3'], GHK['K4'], GHK['K5'], GHK['K6'], GHK['K7'] = np.array(line, 'float')
        #print(GR, GT, HR, HT, K1, K2, K3, K4, K5, K6, K7)
    else:
        logger.warning('File not found: %s' % filepath)
            
    return GHK


""" QUALITY ANALYSIS FUNCTIONS
"""

def rayleigh_fit(**kwargs):
    """
    Thought for measurements taken in a given day. No more than one day should
    be processed

    Parameters
    ----------
    kwargs: dict, all optional
        rs_fl: str
            wildcard list of lidar raw signal level 1a files (*.nc)
        dc_fl: str
            wildcard list of lidar dark current signal level 1a files (*.nc)
        date_str: str
            date in format yyyymmdd
        hour_ini: str
            hh, hhmm, hhmmss
        hour_end: str
            hh, hhmm, hhmmss
        duration: int, float
            duration in minutes for RF
        lidar_name: str
            lidar name (VELETA, MULHACEN)
        meas_type: str
            measurement type (RS, OT, ...)
        channels: str
            if we want all of them we put 'all' (str),
            if we only want some of them we put a list (ex.: 0,1,3,8) (comma separated)
        z_min: int, float
            minimum altitude we use as base for our rayleigh fit
        z_max: int, float
            maximum altitude we use as base for our rayleigh fit
        smooth_range: int, float
            range resolution window for smoothing profiles
        range_min: int, float
            minimum range
        range_max: int, float
            maximum range
        meteo: str
            identifier for hydrostatic (P, T) data source (ecmwf (default), user, lidar, grawmet)
        pressure_prf: array
            pressure profile
        temperature_prf: array
            temperature profile
        data_dn: str
            absolute path for data directory: "data_dn"/LIDAR/1a
        level_1a_dn: str
            level 1a data directory
        ecmwf_dn: str
            ecmwf data directory (ecmwf_dn/yyyy/)
        output_dn: str
            directory to store rayleigh fit results
        save_fig: bool
            save RF figures in png format
        save_earlinet bool
            save earlinet files in ascii format
        verbose: str
            level of message to print ('debug','info','warning','error','critical')
    Returns
    -------

    """ 

    """ Logging """
    # psb
    verbose = kwargs.get("verbose", 'info')
    log_dict = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "simple": {
                "datefmt": LOG_DATE_FMT,
                "format": LOG_FMT,
            }
        },

        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": verbose.upper(),
                "formatter": "simple",
                "stream": "ext://sys.stdout"
            },            
        },
        "root": {
            "level": 'DEBUG',
            "handlers": ["console"]
        }
    }    
    logging.config.dictConfig(log_dict)

    logger.info("Start Rayleigh Fit")

    """ Get Input Arguments """
    rs_fl = kwargs.get("rs_fl", None)
    dc_fl = kwargs.get("dc_fl", None)

    date_str = kwargs.get("date_str", None)
    hour_ini = kwargs.get("hour_ini", None)
    hour_end = kwargs.get("hour_end", None)
    duration = kwargs.get("duration", None)

    lidar_name = kwargs.get("lidar_name", "MULHACEN")
    if lidar_name is None:
        lidar_name = "MULHACEN"

    meas_type = kwargs.get("meas_type", "RS")
    if meas_type is None:
        meas_type = "RS"

    channels = kwargs.get("channels", "all")
    # if channels is None:
    #     channels = "all"
    # if channels != "all":
    #     if isinstance(channels, list):
    #         channels = [int(i) for i in channels]
    #     else:
    #         channels = [channels]

    z_min = kwargs.get("z_min", 5000)
    if z_min is None:
        z_min = 5000
    z_max = kwargs.get("z_max", 6000)
    if z_max is None:
        z_max = z_min + 1000
    smooth_range = kwargs.get("smooth_range", 250)
    if smooth_range is None:
        smooth_range = 250

    range_min = kwargs.get("range_min", 15)
    if range_min is None:
        range_min = 15
    range_max = kwargs.get("range_max", 30000)
    if range_max is None:
        range_max = 30000

    meteo = kwargs.get("meteo", "ecmwf")
    if meteo is None:
        meteo = "ecmwf"
    pressure_prf = kwargs.get("pressure_prf", None)
    temperature_prf = kwargs.get("temperature_prf", None)
    if np.logical_and(temperature_prf is not None, pressure_prf is not None):
        meteo = "user"

    data_dn = kwargs.get("data_dn", None)
    # pdb.set_trace()
    if np.logical_or(data_dn is None, data_dn == "GFATserver"):
        data_dn = nas_data_dn
    ecmwf_dn = kwargs.get("ecmwf_dn", None)
    if np.logical_or(ecmwf_dn is None, ecmwf_dn == "GFATserver"):
        ecmwf_dn = os.path.join(data_dn, "ECMWF")
    level_1a_dn = kwargs.get("level_1a_dn", None)
    if np.logical_or(level_1a_dn is None, level_1a_dn == "GFATserver"):
        level_1a_dn = os.path.join(data_dn, lidar_name, "1a")
    output_dn = kwargs.get("output_dn", None)
    
    darkcurrent_flag = kwargs.get("darkcurrent_flag", True)
    if darkcurrent_flag is None:
        darkcurrent_flag = True

    deadtime_flag = kwargs.get("deadtime_flag", True)
    if deadtime_flag is None:
        deadtime_flag = True

    zerobin_flag = kwargs.get("zerobin_flag", True)
    if zerobin_flag is None:
        zerobin_flag = True

    merge_flag = kwargs.get("merge_flag", True)
    if merge_flag is None:
        merge_flag = True

    save_fig = kwargs.get("save_fig", True)
    if save_fig is None:
        save_fig = True

    save_earlinet = kwargs.get("save_earlinet", True)
    if save_earlinet is None:
        save_earlinet = True

    datetime_fmt = "%Y%m%dT%H%M%S"

    """ Check Input Arguments """
    # TODO
    """
    if rs_fl is not None:
        assert isinstance(rs_fl, str), "rs_fl must be String Type"
    if dc_fl is not None:
        assert isinstance(dc_fl, str), "dc_fl must be String Type"
    assert isinstance(date_str, str), "date_str must be String Type"
    assert isinstance(hour_ini, str), "hour_ini must be String Type"
    assert isinstance(hour_end, str), "hour_end must be String Type"
    """

    """ Lidar Files, Date, Lidar Name, Measurement Type """
    if rs_fl is not None:  # Derive Date, Lidar Name, Measure Type
        if glob.glob(rs_fl):
            # lidar name
            fn_0 = glob.glob(rs_fl)[0]
            lidar_name = "LIDAR"
            for k, v in lidar_system.items():
                if v in fn_0:
                    lidar_name = k
            # measurement type
            meas_type = "XX"
            for k, v in measurement_types.items():
                if v in fn_0:
                    meas_type = k
            # date
            try:
                xxx = re.search(r"\d{4}\d{2}\d{2}", os.path.basename(fn_0))
                if xxx:
                    date_str = xxx.group()
                else:
                    with xr.open_dataset(fn_0) as ds:
                        t0 = ds.time[0].values
                        t0_dt = utils.numpy_to_datetime(t0)
                        date_str = t0_dt.strftime("%Y%m%d")
            except Exception as e:
                logger.error("no date found in file %s. exit" % fn_0)
                return

            # Check DC filelist
            if dc_fl is None:
                dc_fl = os.path.join(os.path.dirname(fn_0), "*%s*.nc" % measurement_types["DC"])

            # date dt
            date_dt = utils.str_to_datetime(date_str)
            year_str, month_str, day_str = "%04d" % date_dt.year, \
                                           "%02d" % date_dt.month, \
                                           "%02d" % date_dt.day
        else:
            logger.error("%s not found. Exit" % rs_fl)
            return
    else:
        if date_str is None:
            date_dt = dt.datetime.utcnow()
            date_str = date_dt.strftime("%Y%m%d")
        else:
            date_dt = utils.str_to_datetime(date_str)
            date_str = date_dt.strftime("%Y%m%d")  # just in case
        year_str, month_str, day_str = "%04d" % date_dt.year, \
                                       "%02d" % date_dt.month, \
                                       "%02d" % date_dt.day
        rs_fl = os.path.join(level_1a_dn, year_str, month_str, day_str, '*%s_%s.nc'
                             % (measurement_types[meas_type], date_str))
        dc_fl = os.path.join(level_1a_dn, year_str, month_str, day_str, '*%s_%s_*.nc'
                             % (measurement_types["DC"], date_str))
        if not glob.glob(rs_fl):
            logger.error("%s not found. Exit" % rs_fl)
            return

    """ Period of Measurements to Process """
    if hour_ini is None:
        fn_0 = glob.glob(rs_fl)[0]
        with xr.open_dataset(fn_0) as ds:
            t0 = ds.time[0].values
            t0_dt = utils.numpy_to_datetime(t0)
        date_ini_dt = t0_dt
        date_ini_str = date_ini_dt.strftime(datetime_fmt)
    else:
        date_ini_str = "%sT%s" % (date_str, hour_ini)
        date_ini_dt = utils.str_to_datetime(date_ini_str)
        date_ini_str = date_ini_dt.strftime(datetime_fmt)
    if hour_end is None:
        if duration is None:
            duration = 30  # Default Duration in minutes
        date_end_dt = date_ini_dt + dt.timedelta(minutes=duration)
        date_end_str = date_end_dt.strftime(datetime_fmt)
    else:
        date_end_str = "%sT%s" % (date_str, hour_end)
        date_end_dt = utils.str_to_datetime(date_end_str)
        date_end_str = date_end_dt.strftime(datetime_fmt)
    duration_secs = float((date_end_dt - date_ini_dt).total_seconds())

    """ LIDAR preprocessing """        
    lidar_ds = lidar.preprocessing(rs_fl, dc_fl=dc_fl, ini_date=date_ini_str,
                                   end_date=date_end_str, channels=channels,
                                   ini_range=range_min, end_range=range_max, 
                                   data_dn=data_dn, darkcurrent_flag=darkcurrent_flag,
                                   deadtime_flag=deadtime_flag, zerobin_flag=zerobin_flag,
                                   merge_flag=merge_flag)    

    # times, ranges in array
    times = lidar_ds['time'].values
    times = np.array([utils.numpy_to_datetime(xx) for xx in times])
    ranges = lidar_ds['range'].values

    """ Meteo Profile: P, T """
    # Radiosonde, ECMWF, user-input (P,T)
    if meteo == 'user':
        if np.logical_and(pressure_prf is not None, temperature_prf is not None):
            pressure_prf = np.array(pressure_prf)
            temperature_prf = np.array(temperature_prf)
            meteo_profiles = extend_meteo_profile(pressure_prf, temperature_prf, ranges)
            radiosonde_wmo_id = "user"
        else:
            raise ValueError("T or P must be not None")
    elif meteo == 'lidar':
        # TODO: resolver primero raw2L1 para que incluya P,T en los netcdf 1a
        # meteo_profiles = meteo_ecmwf(dateini_dt, dateend_dt, ranges)
        logger.error('Error: option lidar not available yet.')
        return
        radiosonde_wmo_id = "lidar"
    elif meteo == 'ecmwf':
        meteo_profiles = meteo_ecmwf(date_ini_dt, date_end_dt, ranges, ecmwf_dn=ecmwf_dn)
        if meteo_profiles is None:
            logger.warning("Set Default Values for T,P: (25, 938)")
            pressure_prf = np.array(938.0)
            temperature_prf = np.array(25.0)
            meteo_profiles = extend_meteo_profile(pressure_prf, temperature_prf, ranges)
        radiosonde_wmo_id = "ecmwf"
    elif meteo == 'grawmet':
        #First try:        
        meteo_fl = glob.glob(os.path.join(nas_data_dn, "radiosondes_spain", "RS*",
                                          '*', 'SOUNDING DATA', '%s*.txt' % date_str))
        if len(meteo_fl) == 0:
            meteo_fl = glob.glob(os.path.join(nas_data_dn, "radiosondes_spain", "RS*",
                                                '%s*.txt' % date_str))            
        # if len(meteo_fl) == 0:
        #     meteo_fl = glob.glob(os.path.join(nas_data_dn, "radiosondes_spain", "RS*",
        #                                       '*', 'SOUNDING DATA', '%s*.txt' % year_str))
        if len(meteo_fl) > 0:
            try:
                rs_ds = reader_typeA(meteo_fl[0])
            except:
                rs_ds = reader_typeB(meteo_fl[0])
            if rs_ds:                
                meteo_profiles = extend_scaled_meteo_profiles(rs_ds, ranges)
                radiosonde_wmo_id = "local radiosonde"
        else:
            logger.error('No radiosonde file found.')
            return
    else:
        meteo_profiles = meteo_ecmwf(date_ini_dt, date_end_dt, ranges, ecmwf_dn=ecmwf_dn)
        radiosonde_wmo_id = "ecmwf"
    radiosonde_location = 'Granada'
    radiosonde_datetime = date_str
    pressure_prf = meteo_profiles["pressure"].values
    temperature_prf = meteo_profiles["temperature"].values

    """ Lidar Resolution, Smoothing Bins """
    resolution = np.median(np.diff(ranges))
    smooth_bins = np.round((smooth_range/resolution)).astype(int)  # (33)

    """ Output Directory """
    # pdb.set_trace()

    if np.logical_or(output_dn is None, output_dn == "GFATserver"):
        output_dn = os.path.join(data_dn, lidar_name, "QA", "rayleigh_fit")
    output_dn = os.path.join(output_dn, year_str, month_str, day_str,
                             "rayleighfit_%s" % date_ini_str)
    if not os.path.exists(output_dn):
        try:
            mkpath(output_dn)
        except Exception as e:
            logger.error('Output Directory Not Created. %s' % output_dn)
            # output_dn = os.getcwd()
    # pdb.set_trace()

    """ Rayleigh Fit Along Channels """
    # list of channels
    if channels == 'all':
        chan_num = lidar_ds.channel.values
    else:
        chan_num = channels
    # indices for averaging and normalizing
    idx_avg_t = np.logical_and(times >= date_ini_dt, times <= date_end_dt)
    idx_norm_r = np.logical_and(ranges >= z_min, ranges <= z_max)
    # output dataset will have height info in km
    ranges_km = ranges*1e-3
    z_min_km = z_min*1e-3
    z_max_km = z_max*1e-3
    # loop over channels
    # pdb.set_trace() 
    for channel_ in chan_num:
        if lidar_ds.active_channel.sel(channel=channel_).values.all():
            """ Get Channel Data, info: """
            # Wavelength, Detection Mode, Polarization
            wavelength = int(lidar_ds.wavelength.sel(channel=channel_).values)
            wavelength_str = str(wavelength)
            detection_mode = int(lidar_ds.detection_mode.sel(channel=channel_).values)
            polarization = int(lidar_ds.polarization.sel(channel=channel_).values)
            logger.info('Channel %s' % channel_)
            

            # RCS, DC profiles
            # pdb.set_trace() 
            
            rcs = lidar_ds['signal_'+channel_].values*(lidar_ds['range'].values**2)
            # rcs = lidar_ds['signal_'+channel_].values

            # pdb.set_trace() 

            # dc = lidar_ds['dc_smoothed_%02d' % channel_].values
            if detection_mode == 0:  # (if analog)
                dark_subtracted = lidar_ds['signal_'+channel_].dark_corrected
            else:
                dark_subtracted = False
            # pdb.set_trace() 
            # Dark Subtraction info
            if dark_subtracted == 'yes':
                dark_subtracted = 'dark-subtracted'
            elif dark_subtracted == 'no':
                dark_subtracted = 'not-dark-subtracted'
            else:
                dark_subtracted = "unknown"

            # Molecular Attenuated Backscatter
            beta_mol_att = molecular_properties(wavelength, pressure_prf,
                                                            temperature_prf, ranges)

            # Time Average
            rcs_avg_time = np.nanmean(rcs[idx_avg_t, :], axis=0)
            # dc_avg_time = np.nanmean(dc[idx_avg_t, :], axis=0)

            # Smooth Time Averaged RCS
            idx_finite = np.isfinite(rcs_avg_time)
            rcs_avg_time_smooth = rcs_avg_time.copy()
            rcs_avg_time_smooth[idx_finite] = scipy.signal.savgol_filter(
                rcs_avg_time[idx_finite], smooth_bins, 3)

            # Normalize time-averaged RCS, BCS nd Smoothed Time-Averaged RCS
            rcs_avg_time_norm = rcs_avg_time / np.nanmean(rcs_avg_time[idx_norm_r])
            # pdb.set_trace() 

            # dc_avg_time_norm = dc_avg_time / np.nanmean(dc_avg_time[idx_norm_r])
            beta_mol_att_norm = beta_mol_att['attenuated_molecular_beta'].values / np.nanmean(beta_mol_att['attenuated_molecular_beta'].values[idx_norm_r])
            rcs_avg_time_smooth_norm = rcs_avg_time_smooth / np.nanmean(rcs_avg_time_smooth[idx_norm_r])

            """ Build a Dataset with Rayleigh Fit Useful Data: EARLINET FORMAT """
            rf_ds = xr.Dataset(
                {'wavelength': xr.DataArray(data=wavelength,
                                            attrs={'str': wavelength_str}),
                 'polarization': xr.DataArray(data=polarization,
                                  attrs={'meaning': pol_name[polarization],
                                         'id': pol_id[polarization]}),
                 'detection_mode': xr.DataArray(data=detection_mode,
                                              attrs={'meaning':det_mod_name[detection_mode],
                                                     'id': det_mod_id[detection_mode]}),
                 'RCS': xr.DataArray(data=rcs_avg_time,
                                     dims=['range'],
                                     coords={'range': ranges_km},
                                     attrs={'name': 'RangeCorrectedSignal',
                                            'long_name': 'range corrected signal avg',
                                            'units': 'a.u.'}),
                 'RCS_smooth': xr.DataArray(data=rcs_avg_time_smooth,
                                      dims=['range'],
                                      coords={'range': ranges_km},
                                      attrs={'name': 'RangeCorrectedSignal',
                                             'long_name': 'range corrected signal smoothed avg',
                                             'units': 'a.u.'}),
                 # 'DC': xr.DataArray(data=dc_avg_time,
                 #                      dims=['range'],
                 #                      coords={'range': ranges_km},
                 #                      attrs={'name': 'D',
                 #                             'long_name': 'dark current avg',
                 #                             'units': 'a.u.'}),
                 'BCS': xr.DataArray(data=beta_mol_att['attenuated_molecular_beta'].values,
                                      dims=['range'],
                                      coords={'range': ranges_km},
                                      attrs={'name': 'attnRayleighBSC',
                                             'long_name': 'attenuated molecular backscatter',
                                             'units': 'a.u.'}),
                 'RCS_norm': xr.DataArray(data=rcs_avg_time_norm,
                                          dims=['range'],
                                          coords={'range': ranges_km},
                                          attrs={'name': 'RangeCorrectedSignal',
                                                 'long_name': 'range corrected signal avg norm',
                                                 'units': 'a.u.'}),
                 'RCS_smooth_norm': xr.DataArray(data=rcs_avg_time_smooth_norm,
                                                 dims=['range'],
                                                 coords={'range': ranges_km},
                                                 attrs={'name': 'RangeCorrectedSignal',
                                                        'long_name': 'range corrected signal smoothed avg norm',
                                                        'units': 'a.u.'}),
                 # 'DC_norm': xr.DataArray(data=dc_avg_time_norm,
                 #                         dims=['range'],
                 #                         coords={'range': ranges_km},
                 #                         attrs={'name': 'D',
                 #                                'long_name': 'dark current avg norm',
                 #                                'units': 'a.u.'}),
                 'BCS_norm': xr.DataArray(data=beta_mol_att_norm,
                                          dims=['range'],
                                          coords={'range': ranges_km},
                                          attrs={'name': 'attnRayleighBSC',
                                                 'long_name': 'attenuated molecular backscatter norm',
                                                 'units': 'a.u.'})
                 },
                attrs={
                    'lidar_location': lidar_location,
                    'lidar_id': lidar_id,
                    'lidar_system': lidar_name,
                    'radiosonde_location': radiosonde_location,
                    'radiosonde_wmo_id': radiosonde_wmo_id,
                    'radiosonde_datetime': radiosonde_datetime,
                    'datetime_ini': date_ini_str,
                    'datetime_end': date_end_str,
                    'date_format': datetime_fmt,
                    'timestamp': date_str,
                    'duration': duration_secs,
                    'duration_units': 'seconds',
                    'rayleigh_height_limits': [z_min_km, z_max_km],
                    'dark_subtracted': dark_subtracted
                }
            )
            rf_ds['range'].attrs['units'] = 'km'
            rf_ds['range'].attrs['long_name'] = 'height'

            """ Save Rayleigh Fit in File """    
            # pdb.set_trace()
            if save_earlinet:                
                # to netcdf
                rf_nc_fn = os.path.join(output_dn, "%slvdayleighFit%sx%s%s.nc"
                                        % (lidar_id, wavelength_str, pol_id[polarization],
                                        det_mod_id[detection_mode]))
                rf_ds.to_netcdf(rf_nc_fn)

                # Filename
                rf_fn = os.path.join(output_dn, "%sRayleighFit%sx%s%s.csv"
                                    % (lidar_id, wavelength_str, pol_id[polarization],
                                        det_mod_id[detection_mode]))

                # Select Columns to write
                
                cols = ['BCS', 'RCS']
                if detection_mode == 0:  # (if analog)
                    pass
                    # cols.append('DC') # Dark current value not exit 
                rf_df = rf_ds[cols].to_dataframe()
                rf_df.columns = [rf_ds[col].attrs['name'] for col in cols]

                # Write File Earlinet Format             
                with open(rf_fn, 'w') as f:
                    f.write("station ID = %s (%s)\n" % (lidar_id, lidar_location))
                    f.write("system = %s\n" % lidar_name)
                    f.write("signal = %s, %s, %s, %s\n" % (wavelength_str, pol_name[polarization],
                                                        det_mod_name[detection_mode],
                                                        dark_subtracted))
                    f.write("date of measurement, time, duration of measurement= %s, %s s\n"
                            % (date_ini_dt.strftime("%d.%m.%Y, %HUTC"), duration_secs))
                    f.write("location, WMO radiosonde station ID, date of radiosonde = %s, %s, %s\n"
                            % (radiosonde_location, radiosonde_wmo_id,
                            date_ini_dt.strftime("%d.%m.%Y, %HUTC")))
                    f.write("lower and upper Rayleigh height limits = %i, %i\n"
                            % (np.round(z_min_km), np.round(z_max_km)))
                f.close()
                rf_df.index = rf_df.index.map(lambda x: '%.4f' % x)
                rf_df.to_csv(rf_fn, mode='a', header=True, na_rep='NaN', float_format='%.4e')

            """ FIGURE """
            fig_title = '%s Rayleigh fit - channel %sx%s%s | %s from %s to %s UTC | Reference height: %d-%d km' \
                        % (lidar_name, wavelength_str, pol_id[polarization],
                           det_mod_id[detection_mode], date_str,
                           date_ini_dt.strftime("%H:%M"), date_end_dt.strftime("%H:%M"),
                           z_min_km, z_max_km)
            fig_y_label = 'Normalized attenuated backscatter, #'
            x_lim = (range_min*1e-3, range_max*1e-3)
            y_lim = (1e-3, 1e2)

            raw_colors = {'355': mcolors.CSS4_COLORS['aliceblue'],
                          '532': mcolors.CSS4_COLORS['honeydew'],
                          '1064': mcolors.CSS4_COLORS['seashell']}
            smooth_colors = {'355': 'b', '532': 'g', '1064': 'r'}
            if wavelength_str in smooth_colors:
                raw_color = raw_colors[wavelength_str]
                smooth_color = smooth_colors[wavelength_str]
            else:
                raw_color = mcolors.CSS4_COLORS['aliceblue']
                smooth_color = 'b'
            # dfData = pd.read_excel(r'C:\datos\VELETA\QA\rayleigh_fit\2022\02\21\rayleighfit_20220221T203000\VELETA_RF_355xpp_20220221.xlsx',index_col=0)
            
          
            fig = plt.figure(figsize=(15, 6))
            ax = fig.add_subplot(111)
            ax.grid()
            rf_ds["RCS_norm"].plot(ax=ax, x='range', label='raw', color=raw_color)
            rf_ds["BCS_norm"].plot(ax=ax, x='range', label=r'$\beta_{att}^{mol}$', color='k', linewidth=2)
            rf_ds["RCS_smooth_norm"].plot(ax=ax, x='range', label='smoothed', color=smooth_color)
            # ax.plot(dfData.index,dfData['RCS_smooth_norm'],color='red',label='Veleta_smoothed')
            ax.set_title(fig_title, fontsize='x-large')
            ax.xaxis.get_label().set_fontsize('large')
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
            ax.set_ylabel(fig_y_label, fontsize='large')
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            ax.set_yscale('log')
            leg = ax.legend(fontsize='medium')
            frame = leg.get_frame()
            frame.set_edgecolor('black')
            frame.set_facecolor('silver')
        else:
            logger.info('Channel not activated:'+channel_)
        if save_fig:
            fig_fn = os.path.join(output_dn, '%s_RF_%sx%s%s_%s.png'
                                  % (lidar_name, wavelength_str,
                                     pol_id[polarization],
                                     det_mod_id[detection_mode], date_str))
           
            plt.savefig(fig_fn, dpi=200, bbox_inches="tight")
            # pdb.set_trace()
            # Data = np.vstack((np.array(rf_ds["RCS_norm"]),np.array(rf_ds["RCS_smooth_norm"]))).T
            # dfData = pd.DataFrame(data=Data, index=np.array(rf_ds["range"])[0:3999],columns=['RCS_norm','RCS_smooth_norm'])
            # dfData.to_excel(Data_an)
        lidar_ds.close()
        rf_ds.close()
        # pdb.set_trace()

    logger.info("End Rayleigh Fit")
    return rf_ds


def telecover(dateini, zmin=2000, zmax=2500, input_directory='GFATserver',
              output_directory='GFATserver', savefig=True):
    """
    # TODO: Reformular como se ha hecho con rayleigh fit y depolarization calibration
    telecover
    Inputs:
    - date_str: date ('yyyymmdd') (str)
    - zmin: minimum altitude we use as base for our rayleigh fit (float)
    - zmax: max altitude we use as base for our rayleigh fit (float)
    - savefig: it determines if we will save our plots (True) or not (False) (bool)
    - level_1a_dn: folder where telecover files are located.
    - rayleigh_fit_dn: folder where we want to save our figures.
    Outputs:
    - ascii files
    - figures
    """

    def plot_telecover_channel(tc_type, rf_ds, zmin, zmax, output_directory, savefig):
        logger.info('Creating figure: %s' % rf_ds.channel_code)
        """
        Inputs:
        - tc_type: Telecover for ultriviolet or visible/near-infrarred (uv or vnir) (str)
        - rf_dc: xarray dataset (xarray.Dataset)        
        - savefig: it determines if we will save our plots (True) or not (False) (bool)        
        - rayleigh_fit_dn: folder where we want to save our figures.
        Outputs:        
        - figures
        """
        font = {'size' : 12}
        matplotlib.rc('font', **font)
        # FIGURE
        ydict_rcs = {'355xpa': (0, 2e6), '355xpp': (0, 5e7),'355xsa': (0, 3.5e6), '355xsp': (0, 5e7),
                     '387xta': (0, 2e6), '387xtp': (0, 3e7),
                     '355xta': (0, 2e6), '355xtp': (0, 3e7), '532xpa': (0, 3e6), '532xpp': (0, 6e7),
                '532xcp': (0, 2e7), '532xca': (0, 1e6), '1064xta': (0, 7e6),
                '353xtp': (0, 3e6), '530xtp': (0, 5e7),'408xtp': (0, 2e8)}
        ydict_norm_rcs = {'355xpa': (0, 3), '355xpp': (0, 2),'355xsa': (0, 3), '355xsp': (0, 2),
                     '387xta': (0, 4), '387xtp': (0, 4),
                '355xta': (0, 10), '355xtp': (0, 10), '532xpa': (0, 8), '532xpp': (0, 5),
                '532xcp': (0, 4), '532xca': (0, 10), '1064xta': (0, 20),
                '353xtp': (0, 5), '530xtp': (0, 2),'408xtp': (0, 5)}                
        lidar_system = {'MULHACEN': 'mhc','VELETA':'vlt'}
        x_lim = (0, 3000)
        colorbar = matplotlib.cm.get_cmap('jet', len(rf_ds.sectors))
        colors = colorbar(np.linspace(0, 1, len(rf_ds.sectors)))
        fig = plt.figure(figsize=(15,10))
        fig_title = '%s telecover - channel %s | %s | Reference height: %3.1f-%3.1f km' % (rf_ds.attrs['lidar_system'], rf_ds.channel_code, dt.datetime.strftime(rf_ds.datetime_ini, "%d.%m.%Y, %H:%MUTC"), zmin/1000., zmax/1000.)
        #MEAN        
        sectors = [sector_ for sector_ in rf_ds.sectors if sector_.find('2') == -1]
        sum_rcs = np.zeros(rf_ds.range.size)
        for sector_ in sectors:
            rcs = rf_ds[sector_]
            try:
                sum_rcs += rcs
            except:
                sum_rcs += rcs.compute()
        mean_rcs = sum_rcs/len(sectors)
        mean_rcs.name = 'M'

        # Normalized
        for sector_ in rf_ds.sectors:    
            rf_ds['n%s' % sector_] = rf_ds[sector_]/rf_ds[sector_].sel(range=slice(zmin,zmax)).mean()
        norm_mean_rcs = mean_rcs/mean_rcs.sel(range=slice(zmin,zmax)).mean()
        norm_mean_rcs.name = 'nM'

        # RAW RCS
        ax = fig.add_subplot(311)
        fig_y_label = 'RCS, a.u.' 
        for iter_ in zip(rf_ds.sectors, colors):          
            sector_ = iter_[0]
            color_ = iter_[1]    
            rf_ds[sector_].plot(ax=ax, x='range',  linewidth=2, label=sector_, color=color_)
        mean_rcs.plot(ax=ax, x='range',  linewidth=2, label='M', color='k')
        ax.set_title(fig_title, fontsize='x-large', verticalalignment='baseline')
        ax.set_ylim(ydict_rcs[rf_ds.channel_code])
        plt.ticklabel_format(style='sci',axis='y', scilimits=(0,1))
        ax.set_ylabel(fig_y_label, fontsize='large')
        plt.legend(loc=1,fontsize='large')

        # Normalized RCS
        ax = fig.add_subplot(312)
        fig_y_label = 'Normalized RCS, a.u.' 
        for iter_ in zip(rf_ds.sectors, colors):          
            sector_ = iter_[0]
            color_ = iter_[1]    
            rf_ds['n%s' % sector_].plot(ax=ax, x='range',  linewidth=2, label=sector_, color=color_)
        norm_mean_rcs.plot(ax=ax, x='range',  linewidth=2, label='nM', color='k')
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,1))
        ax.set_ylim(ydict_norm_rcs[rf_ds.channel_code])
        ax.set_ylabel(fig_y_label, fontsize='large')
        plt.legend(loc=1, fontsize='large')

        # Diference
        ax = fig.add_subplot(313)
        fig_y_label = u'normalized RCS\nrelative difference, %' 

        for iter_ in zip(rf_ds.sectors, colors):          
            sector_ = iter_[0]
            color_ = iter_[1]    
            rf_ds['diff_%s' % sector_] = 100*(rf_ds['n%s' % sector_] - norm_mean_rcs)/norm_mean_rcs
            rf_ds['diff_%s' % sector_].plot(ax=ax, x='range',  linewidth=2, label=sector_, color=color_)
        ax.xaxis.get_label().set_fontsize('large')
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(125))
        ax.set_ylabel(fig_y_label, fontsize='large')
        ax.set_ylim(-100, 100)
        plt.legend(loc=1, fontsize='large')

        for ax in fig.get_axes():
            ax.tick_params(axis='both', labelsize=14) 
            ax.set_xlim(x_lim)
            ax.grid()
            ax.label_outer()
        # pdb.set_trace()
        if savefig:        
            fig_fn = os.path.join(output_directory,'%s_TC%s_%s_%s.png' % (lidar_system[rf_ds.lidar_system], tc_type, rf_ds.channel_code, dt.datetime.strftime(rf_ds.datetime_ini, "%Y%m%d_%H%M")))
            plt.savefig(fig_fn, dpi=200, bbox_inches="tight")
            if os.path.isfile(fig_fn):
                logger.info(' %s telecover Figure succesfully saved!' % rf_ds.channel_code)
        
    # Dictionary telecover type
    # tc_type = ('uv', 'vnir')
    pdb.set_trace()

    tc_type = ['uv']

    
    # Dictionary channels
    channels = {'uv': ('355xpa', '355xpp','355xsa', '355xsp','387xta','387xtp'),
                'vnir': ('532xpa', '532xpp', '532xca', '532xcp', '530xtp', '1064xta')}
    telecover={}

    # Date info for processing:
    _year = dateini[0:4]
    _month = dateini[4:6]
    _day = dateini[6:8]

    # Input Directory
    # pdb.set_trace()

    if input_directory == "GFATserver":        
        # input_directory = os.path.join(nas_data_dn, "MULHACEN", "1a", _year, _month, _day)
        input_directory = os.path.join(nas_data_dn, "VELETA", "1a", _year, _month, _day)
        # mkpath(input_directory)

    if output_directory == "GFATserver":        
        # main_output_directory = os.path.join(nas_data_dn, "MULHACEN", "QA", "telecover")
        main_output_directory = os.path.join(nas_data_dn, "VELETA", "QA", "telecover")
        
        if not os.path.exists(main_output_directory):
            try:
                mkpath(main_output_directory)
            except Exception as e:
                logger.error('Output Directory Not Created. %s' % main_output_directory)
        # if not os.path.isdir(main_output_directory):
        #     mkpath(main_output_directory)
        #     # logger.warning('ERROR: NAS server not found.')
        #     return
    else:
        if not os.path.isdir(output_directory):
            main_output_directory = output_directory
            mkpath(main_output_directory)

    # Type of telecover: uv or vnir
    # pdb.set_trace()
    for tc_ in tc_type:
        filelist = glob.glob(os.path.join(input_directory, '*Ptc-E*.nc'))            
        hours = list()  # to list different telecovers in the same day
        for file_ in filelist:        
            # hour_ = file_.split('_')[4].split('.')[0]
            hour_ = file_.split('_')[6][0:4]

            hours.append(hour_)    
        hours = np.unique(hours)  # Number of telecovers of the same type
        for hour_ in hours:
            try:
                datetime_ini = dt.datetime.strptime('%s%s' % (dateini, hour_), '%Y%m%d%H%M')
            except:
                logger.error('Datetime conversion error.')
                return 

            # Search DC files at hour_
            DCfilelist = os.path.join(input_directory, 'vlt_1a_Pdc_rs_xf_%s_%s.nc' % (dateini, hour_))
            if not os.path.isfile(DCfilelist):
                DCfilelist = []
                dark_subtracted = 0
            else:
                dark_subtracted = 1

            # Create output folder
            output_directory = os.path.join(main_output_directory, 'telecover_%s_%s_%s' % (tc_, dateini, hour_))
            if not os.path.isdir(output_directory):
                os.makedirs(output_directory, exist_ok=True)
            # Search files of a given telecover tests, i.e., same tc_ and same hour_
            filepathformat = os.path.join(input_directory, '*Ptc-*%s*.nc' % (hour_)) #TC files 
            filelist = glob.glob(filepathformat)
            telecover[tc_]={} #dictionary to save Datasets with all channels separated by sectors
            # pdb.set_trace()
            for file_ in filelist:
                sector_ = file_.split('_')[2].split('-')[-1]    
                logger.info('Loading sector %s' % sector_)
                # filepathformat = os.path.join(input_directory, '*Ptc-%s-%s_*%s.nc' % (tc_, sector_, hour_)) #TC files 
                xr_tmp = lidar.preprocessing(rs_fl=file_, dc_fl=DCfilelist)
                telecover[tc_][sector_] = xr_tmp.mean(dim='time')                
            # pdb.set_trace()
            # Creating a Dataset [rf_ds] for each channel with all the sectors
            for channel_ in telecover[tc_][sector_].channel.values:
                # Wavelength, Detection Mode, Polarization
                wavelength = np.floor(telecover[tc_][sector_].wavelength.sel(channel=channel_).values).astype('int')
                detection_mode = int(telecover[tc_][sector_].detection_mode.sel(channel=channel_).values)
                polarization = int(telecover[tc_][sector_].polarization.sel(channel=channel_).values)                
                # channel_code = '%dx%s%s' % (wavelength, pol_id[polarization], det_mod_id[detection_mode])
                # pdb.set_trace()

                if bool(telecover[tc_][sector_].active_channel.sel(channel=channel_).values):
                    if channel_ in channels[tc_]:
                        rf_ds=[]
                        for idx, sector_ in enumerate(telecover[tc_].keys()):
                            # TELECOVER EARLINET DATA FORMAT
                            # TODO: sacar a una funcion                                            
                            if idx == 0:
                                signal = telecover[tc_][sector_]['signal_'+channel_]*(telecover[tc_][sector_]['range'].values**2)
                                # rcs = lidar_ds['signal_'+channel_].values*(lidar_ds['range'].values**2)

                                signal.name = sector_ 
                                rf_ds = signal.copy()
                            else:
                                # signal = telecover[tc_][sector_]['signal_'+channel_]
                                signal = telecover[tc_][sector_]['signal_'+channel_]*(telecover[tc_][sector_]['range'].values**2)

                                signal.name = sector_                     
                                rf_ds = xr.merge([rf_ds, signal])
                            # TODO rf_ds = rf_ds.assign_coords({"range": rf_ds.range/1e3})
                        if dark_subtracted:
                            # DC #Take the DC measurement from the last sector since all have the same DC
                            try:
                                dc = xr_tmp['dc_smoothed_%02d' % channel_]
                                dc.name = "D"
                                rf_ds = xr.merge([rf_ds, dc])
                            except:
                                dark_subtracted = 0                        
                        # pdb.set_trace()

                        # Average over time
                        # rf_ds = rf_ds.mean(dim='time')
                        
                        # Attributes
                        rf_ds['range'].attrs['units'] = 'm'
                        rf_ds['range'].attrs['long_name'] = 'Height'
                        rf_ds = rf_ds.sel(range=slice(0,30000))                   
                        rf_ds = xr.merge([rf_ds, telecover[tc_][sector_].detection_mode.sel(channel=channel_), telecover[tc_][sector_].polarization.sel(channel=channel_), telecover[tc_][sector_].wavelength.sel(channel=channel_)])
                        rf_ds['wavelength'].attrs['value_str'] = str(wavelength)

                        # TODO: sacar la creacion de un dataset especifico para RF a una funcion
                        # Polarization
                        rf_ds['polarization'].attrs['meaning'] = pol_name[polarization]
                        rf_ds['polarization'].attrs['id'] = pol_id[polarization]

                        # TODO: sacar esto a una funcion
                        # Detection mode
                        rf_ds['detection_mode'].attrs['meaning'] = det_mod_name[detection_mode]
                        rf_ds['detection_mode'].attrs['id'] = det_mod_id[detection_mode]

                        # Dark Measurement
                        dark_sub = {1: 'dark-subtracted', 2: 'not-dark-subtracted'}

                        rf_ds.attrs['sectors'] = list(telecover[tc_].keys())
                        rf_ds.attrs['lidar_location'] = 'Granada'
                        rf_ds.attrs['lidar_id'] = 'gr'
                        rf_ds.attrs['lidar_system'] = 'VELETA'
                        rf_ds.attrs['datetime_ini'] = datetime_ini            
                        rf_ds.attrs['date_format'] = "%Y%m%dT%H:%M:%S.%f"
                        rf_ds.attrs['timestamp'] = dateini[0:8]
                        dark_subtracted = 1
                        rf_ds.attrs['dark_subtracted'] = dark_sub[dark_subtracted]
                        rf_ds.attrs['channel_code'] = channel_
                        
                        # SAVE RF FILE
                        # TODO: sacar a una funcion el escribir este archivo con formato
                        cols = list(telecover[tc_].keys())
                        if detection_mode==0:  # (if analog)
                            pass
                            # cols.append('D')

                        # Convert to pandas
                        rf_df = []
                        rf_df = rf_ds[cols].to_dataframe()

                        # Create file
                        rf_fn = os.path.join(output_directory, "%sTelecover%sx%s%s.csv" % (rf_ds.lidar_id, rf_ds.wavelength.value_str, rf_ds.polarization.id, rf_ds.detection_mode.id))                    
                        with open(rf_fn, 'w') as f: 
                            f.write("station ID = %s (%s)\n" % (rf_ds.lidar_id, rf_ds.lidar_location))
                            f.write("system = %s\n"%rf_ds.lidar_system)
                            f.write("signal = %s, %s, %s, %s\n"%(rf_ds.wavelength.value_str, rf_ds.polarization.meaning, rf_ds.detection_mode.meaning, rf_ds.dark_subtracted))
                            f.write("date, time= %s\n" % dt.datetime.strftime(rf_ds.datetime_ini, "%d.%m.%Y, %HUTC"))
                        f.close()
                        rf_df.index = rf_df.index.map(lambda x: '%.4f' % x)
                        rf_df.to_csv(rf_fn, mode='a', header=True, na_rep='NaN', float_format='%.4e')
                        if os.path.isfile(rf_fn):
                            logger.info('%s-telecover %s file done!' % (tc_, rf_ds.channel_code))
                            # Plot Telecover for current channel
                            plot_telecover_channel(tc_, rf_ds, zmin, zmax, output_directory, savefig)
                else:
                    logger.warning('Telecover-%s: channel %s not found in measurements.' % (tc_, channel_))


def depolarization_cal(**kwargs):
    """
    Calibration of Depolarization based on Delta90-calibration method

    Parameters
    ----------
    kwargs
        p45_fn: str
            file name for P45 measurement (netcdf)
        n45_fn: str
            file name for N45 measurement (netcdf)
        rs_fn: str
            file name for RS complementary measurement (netcdf)
        dc_fn: str
            filename of lidar dark current signal level 1a files (*.nc)
        cal_height_an: tuple, list (2-el)
            height range for averaging calibrations for analog signals
        cal_height_pc: tuple, list (2-el)
            height range for averaging calibrations for photoncounting signals
        pbs_orientation_parameter: int
            polarising beam splitter orientation parameter, y [Freudenthaler, 2016, sec3.3]
            y = -1: reflected = parallel
            y = +1: reflected = perpendicular
        cal_type: str
            type of calibration: rotator (rot), polarizer (pol)
        channels: str ('all'), list()
            list of channels to process
            if we want all of them we put 'all' (str),
            if we only want some of them we put a list (ex.: [0,1,3,8]) (list)
        range_min: int, float
            minimum range
        range_max: int, float
            maximum range
        alpha: float
            rotational misalignment of the polarizing plane of the laser light respect to the incident plane of the PBS
        epsilon: float
            misalignment angle of the rotator
        ghk_tpl_fn: str
            ghk file template for running GHK
        output_dn: str
            directory to store depolarization calibration results

    Returns
    -------

    """

    logger.info("Start Calibration of Depolarization")

    """ Get Input Arguments """
    p45_fn = kwargs.get("p45_fn", None)
    if p45_fn is None:
        logger.error("p45_fn cannot be None")
        return
    n45_fn = kwargs.get("n45_fn", None)
    if n45_fn is None:
        logger.error("n45_fn cannot be None")
        return
    rs_fn = kwargs.get("rs_fn", None)
    if rs_fn is None:
        logger.error("rs_fn cannot be None")
        return
    dc_fn = kwargs.get("dc_fn", None)
    if dc_fn is None:
        logger.error("dc_fn cannot be None")
        return

    channels = kwargs.get("channels", "all")
    if channels is None:
        channels = "all"
    range_min = kwargs.get("range_min", 0)
    if range_min is None:
        range_min = 0
    range_max = kwargs.get("range_max", 30000)
    if range_max is None:
        range_max = 30000

    cal_height_an = kwargs.get("cal_height_an", None)
    if cal_height_an is None:
        ha_min, ha_max = 1500, 3000
    else:
        ha_min, ha_max = cal_height_an
    cal_height_pc = kwargs.get("cal_height_pc", None)
    if cal_height_pc is None:
        hp_min, hp_max = 2500, 4000
    else:
        hp_min, hp_max = cal_height_pc

    pbs_orientation_parameter = kwargs.get("pbs_orientation_parameter", -1)
    if pbs_orientation_parameter is None:
        pbs_orientation_parameter = - 1
    if not np.logical_or(pbs_orientation_parameter == -1, pbs_orientation_parameter == 1):
        logger.error("PBS orientation parameter must be -1 or +1. Exit")
        return
    # pdb.set_trace()

    cal_type = kwargs.get("cal_type", "rot")
    if cal_type is None:
        cal_type = "rot"
    if not np.logical_or(cal_type == "rot", cal_type == "pol"):
        logger.error("calibration type must be rot or pol. Exit")
        return

    epsilon = kwargs.get("epsilon", None)
    alpha = kwargs.get("alpha", None)
    if alpha is None:
        alpha = 7.1
    alpha_err = 0.3
    ghk_tpl_fn = kwargs.get("ghk_tpl_fn", None)

    data_dn = kwargs.get("data_dn", None)
    if np.logical_or(data_dn is None, data_dn == "GFATserver"):
        data_dn = nas_data_dn
    output_dn = kwargs.get("output_dn", None)
    if output_dn is None:
        output_dn = "GFATserver"

    """ Confirm Files Exist """
    if not np.logical_and.reduce((os.path.isfile(p45_fn), os.path.isfile(n45_fn),
                                  os.path.isfile(rs_fn), os.path.isfile(dc_fn))):
        logger.error("Some file(s) missing: P45, N45, RS, DC. Exit")
        return

    """ Get Info from File (P45) """
    fn_0 = p45_fn
    # lidar name
    lidar_name = "LIDAR"
    for k, v in lidar_system.items():
        if v in fn_0:
            lidar_name = k
    # date
    try:
        xxx = re.search(r"\d{4}\d{2}\d{2}_\d{2}\d{2}", os.path.basename(fn_0))
        if xxx:
            date_str = xxx.group()
        else:
            with xr.open_dataset(fn_0) as ds:
                t0 = ds.time[0].values
                t0_dt = utils.numpy_to_datetime(t0)
                date_str = t0_dt.strftime("%Y%m%d_%H%M")
    except Exception as e:
        logger.error("no date found in file %s. exit" % fn_0)
        logger.error(str(e))
        return
    # date dt
    date_dt = utils.str_to_datetime(date_str)
    year_str, month_str, day_str = "%04d" % date_dt.year, \
                                   "%02d" % date_dt.month, \
                                   "%02d" % date_dt.day

    # Calibration Label
    cal_label = "%s_%s" % (date_str, cal_type)

    """ Output Directory """
    if np.logical_or(output_dn is None, output_dn == "GFATserver"):
        output_dn = os.path.join(data_dn, lidar_name, "QA", "depolarization_calibration")
    output_dn = os.path.join(output_dn, year_str, month_str, day_str, "depolcal_%s" % cal_label)
    if not os.path.exists(output_dn):
        mkpath(output_dn)

    """ Prepare Data: P45, N45, RS """
    # lidar preprocessing: P45, N45, RS
    # pdb.set_trace()
    try:
        n45_ds = lidar.preprocessing(rs_fl=n45_fn, dc_fl=dc_fn,channels=channels, ini_range=range_min, end_range=range_max)
        p45_ds = lidar.preprocessing(rs_fl=p45_fn, dc_fl=dc_fn,channels=channels, ini_range=range_min, end_range=range_max)
        rs_ds = lidar.preprocessing(rs_fl=rs_fn, dc_fl=dc_fn, channels=channels, ini_range=range_min, end_range=range_max)
    except Exception as e:
        logger.error("Error: Reading P45, N45 files. Exit")
        logger.error(str(e))
        return

    # times array
    times_dp = np.sort(np.append(n45_ds.time.values, p45_ds.time.values))
    times_rs = rs_ds.time.values
    time_start_np, time_end_np = np.min(times_dp), np.max(times_dp)
    time_start_dt, time_end_dt = utils.numpy_to_datetime(time_start_np), utils.numpy_to_datetime(time_end_np)

    # ranges
    ranges = rs_ds.range.values
    ranges_km = ranges*1e-3

    # N45, P45 time average
    n45_avg_ds = n45_ds.mean(dim="time", skipna=True)
    p45_avg_ds = p45_ds.mean(dim="time", skipna=True)

    # Find 30-min average of RS
    times_rs_closest = min(times_rs, key=lambda x: abs(x - np.min(times_dp)))
    if times_rs[-1] == times_rs_closest:  # if RS is measured AFTER DP
        times_rs_end = times_rs_closest
        times_rs_ini = pd.to_datetime(times_rs_closest) - dt.timedelta(minutes=30)
    else:  # otherwise
        times_rs_ini = times_rs_closest
        times_rs_end = pd.to_datetime(times_rs_closest) + dt.timedelta(minutes=30)
    rs_avg_ds = rs_ds.sel(time=slice(times_rs_ini, times_rs_end)).mean(dim="time", skipna=True)

    """ Calibration of Depolarization """
    logger.info("Compute Calibration of depolarization")

    # assign polarization id for Reflected/Transmitted signals
    if pbs_orientation_parameter == -1:  # Reflected == Parallel (1), Transmitted == Perpendicular (2)
        R_pol, T_pol = 1, 2
        R_pol_str, T_pol_str = 'p','s' 
    else:  # Reflected == Perpendicular (2), Transmitted == Parallel (1)
        R_pol, T_pol = 2, 1
        R_pol_str, T_pol_str = 's','p' 

    # pdb.set_trace()
    # Define Dataset to Store Depol.Cal. Calculations
    dp_ds = xr.Dataset(
        {'time_start': xr.DataArray(data=time_start_np),
         'time_end': xr.DataArray(data=time_end_np),
         'cal_height_an': xr.DataArray(data=[ha_min, ha_max],
                                       attrs={'long_name': 'calibration height range for analog signals',
                                              'units': 'm'}),
         'cal_height_pc': xr.DataArray(data=[hp_min, hp_max],
                                       attrs={'long_name': 'calibration height range for photoncounting signals',
                                              'units': 'm'}),
         'cal_type': xr.DataArray(data=cal_type,
                                  attrs={'name': 'calibration type',
                                         'comment': 'rot: rotator; pol: polarizer'}),
         'pbs_orientation': xr.DataArray(data=pbs_orientation_parameter,
                                         attrs={'long_name': 'polarising beam splitter orientation parameter y',
                                                'comment': '-1: Reflected == Parallel (1), Transmitted == Perpendicular (2); '
                                                           '1: Reflected == Perpendicular (2), Transmitted == Parallel (1)'})
         },
        coords={'range': ranges_km},
        attrs=rs_ds.attrs)
    dp_ds.attrs['location'] = 'Granada'
    dp_ds.attrs['location_id'] = 'GR'
    dp_ds.attrs['lidar_name'] = lidar_name

    # loop over (wavelength, det_mode)
    wavelengths = utils.unique(n45_ds.wavelength.values)
    detection_modes = utils.unique(n45_ds.detection_mode.values)
    channel_count = 0
    channel_id = []
    wv_channel = []
    det_mod_channel = []
    # pdb.set_trace()

    for wv, mo in itertools.product(wavelengths, detection_modes):
        # channel specific info
        wv_str = str(int(wv))
        mode_id_str = det_mod_id[mo]
        mode_name_str = det_mod_name[mo]
        channel_id_str ="%s%s" % (wv_str, mode_id_str)

        logger.info("Channel: %s" % channel_id_str)
        # Find channels (wv, mo)

        channel_R = np.where(np.logical_and.reduce((
            n45_ds.wavelength == wv, n45_ds.detection_mode == mo,
            n45_ds.polarization == R_pol)))[0]
        channel_T = np.where(np.logical_and.reduce((
            n45_ds.wavelength == wv, n45_ds.detection_mode == mo,
            n45_ds.polarization == T_pol)))[0]
        if np.logical_and(len(channel_R) > 0, len(channel_T) > 0):  # Channels exist. Calculate
            # Channel Name: WV, MODE
            wv_channel.append(wv)
            channel_id.append(channel_id_str)
            det_mod_channel.append(mode_name_str)

            # Calibration Height Interval
            if mode_id_str == 'a':  # ranges for analog channel
                h_min, h_max = ha_min, ha_max
            elif mode_id_str == 'p':  # ranges for photoncounting channel
                h_min, h_max = hp_min, hp_max
            else:  # default = photoncounting
                h_min, h_max = hp_min, hp_max
            idx_av_ranges = np.logical_and(ranges >= h_min, ranges <= h_max)

            # ID R, T channels
            # pdb.set_trace()
            # id_R = n45_ds.n_chan[channel_R]
            # id_T = n45_ds.n_chan[channel_T]
            # where is R, and where is T 
            Channel_R = 'signal_'+wv_str+'x'+R_pol_str+mode_id_str
            Channel_T = 'signal_'+wv_str+'x'+T_pol_str+mode_id_str


            # time averaged signal profiles
            
            n45_R = n45_avg_ds[Channel_R].values
            n45_T = n45_avg_ds[Channel_T].values
            p45_R = p45_avg_ds[Channel_R].values
            p45_T = p45_avg_ds[Channel_T].values
            rs_R = rs_avg_ds[Channel_R].values
            rs_T = rs_avg_ds[Channel_T].values

            # Gain ratio (x45) profile, eta^*(x45) [Freudenthaler, 2016, Eq.80]
            gain_ratio_n45 = n45_R / n45_T
            gain_ratio_p45 = p45_R / p45_T

            # Gain ratio Delta90 profile, eta^*_{Delta90} [Freudenthaler, 2016, Eq.85]
            gain_ratio_Delta90 = np.sqrt(gain_ratio_n45 * gain_ratio_p45)
            gain_ratio_Delta90_avg = np.nanmean(gain_ratio_Delta90[idx_av_ranges])
            gain_ratio_Delta90_std = np.nanstd(gain_ratio_Delta90[idx_av_ranges])

            # Calibrator rotation, epsilon [Freudenthaler, V. (2016)., Eq. 194, 195]
            # average over calibration height interval
            Y = (gain_ratio_p45 - gain_ratio_n45) / (gain_ratio_p45 + gain_ratio_n45)
            Y_avg = np.nanmean(Y[idx_av_ranges])
            Y_std = np.nanstd(Y[idx_av_ranges])
            if epsilon is None:
                epsilon_iter = (180/np.pi)*0.5*np.arcsin(np.tan(0.5*np.arcsin(Y_avg)))
            else:
                epsilon_iter = epsilon
            epsilon_err = (180/np.pi)*0.5*abs(0.5*np.arcsin(np.tan(0.5*np.arcsin(Y_avg + Y_std))) -
                                              0.5*np.arcsin(np.tan(0.5*np.arcsin(Y_avg - Y_std))))
            # Rotation of plane of polarisation of the emitted laser beam, alpha [assumed]
            # pdb.set_trace()
            """ Calculate GHK parameters"""
            # Prepare GHK input file
            if ghk_tpl_fn is None:  # If not given, use a template
                # Define a GHK file template
                if cal_type == "rot":
                    ghk_tpl_fn_iter = os.path.join(ghk_sett_dn, "mulhacen_rotator_casoG_template.py")
                elif cal_type == "pol":
                    ghk_tpl_fn_iter = os.path.join(ghk_sett_dn, "mulhacen_polarizer_template.py")
                else:  # default is rotator
                    ghk_tpl_fn_iter = os.path.join(ghk_sett_dn, "mulhacen_rotator_casoG_template.py")
            else:  # use that given in input
                ghk_tpl_fn_iter = ghk_tpl_fn
            # pdb.set_trace()

            # Create Input file from template. Remove first if exists.
            if os.path.exists(ghk_inp_fn):
                os.remove(ghk_inp_fn)
            shutil.copyfile(ghk_tpl_fn_iter, ghk_inp_fn)
            # Substitute calculated epsilon, alpha values
            for line in fileinput.input(ghk_inp_fn, inplace=True):
                print(line.replace("epsilon_value", "%6.2f" % epsilon_iter), end="")
            for line in fileinput.input(ghk_inp_fn, inplace=True):
                print(line.replace("epsilon_error", "%6.2f" % epsilon_err), end="")
            for line in fileinput.input(ghk_inp_fn, inplace=True):
                print(line.replace("alpha_value", "%6.2f" % alpha), end="")
            for line in fileinput.input(ghk_inp_fn, inplace=True):
                print(line.replace("alpha_error", "%6.2f" % alpha_err), end="")
            # pdb.set_trace()

            # Save Particularized Input File
            ghk_in_particular_fn = os.path.join(output_dn, "mulhacen_run_%sx%s.py"
                                                % (wv_str, mode_id_str))
            if os.path.exists(ghk_in_particular_fn):
                os.remove(ghk_in_particular_fn)
            shutil.copyfile(ghk_inp_fn, ghk_in_particular_fn)

            # run GHK: uses ghk_inp_fn as Input. Generates ghk_param_fn
            os.system("python %s" % os.path.join(ghk_dn, "GHK.py"))
            os.chdir(run_dir)  # GHK cambia el directorio
            ghk_param_fn = os.path.join(ghk_dn, "output_files", "output_mulhacen_run_GHK.dat")
            ghk_param_particular_fn = os.path.join(output_dn, "output_mulhacen_run_GHK_%sx%s.dat" %
                                                   (wv_str, mode_id_str))
            if os.path.exists(ghk_param_particular_fn):
                os.remove(ghk_param_particular_fn)
            # pdb.set_trace()
            shutil.copyfile(ghk_param_fn, ghk_param_particular_fn)

            # ghk output (Dictionary):
            ghk_param = ghk_output_reader(ghk_param_fn)

            #ghk_param = {
            #        'GR':  1.78,
            #        'HR':  1.35,
            #        'GT':  0.64,
            #        'HT': -0.54,
            #        'K2': 1
            #        }
            #print("GHK NUESTRO")
            #print(ghk_param)

            """ Calibration Factor, eta """
            # Calibration factor, eta [Freudenthaler, 2016, Eq.84]
            # average over calibration height interval
            cal_factor = gain_ratio_Delta90 / ghk_param["K2"]
            # cal_factor  = cal_factor*0.0439
            cal_factor_avg = np.nanmean(cal_factor[idx_av_ranges])
            cal_factor_std = np.nanstd(cal_factor[idx_av_ranges])
            # pdb.set_trace()


            """ Application of GHK parameters to a RS signal: VLDR """
            # Signal Ratio, delta^* [Freudenthaler, 2016, Eq.60]
            # pdb.set_trace()
            # aa = (rs_T / rs_R) 
            # plt.plot(aa)
            # aa = (rs_R / rs_T)
            signal_ratio = (rs_R / rs_T) / cal_factor_avg
            
            # with plt.style.context(['science','no-latex']):
            #     fig = plt.figure(figsize=(8,6),dpi=120)

            #     plt.subplots_adjust(hspace=0,right=0.95,top=0.95,left=0.15)
            #     ax = fig.add_subplot(1,1,1)
            #     ax.plot(aa,n45_ds.coords['range'].values/1000,linewidth=2, label='signal_ratio') 
            #     ax.plot(gain_ratio_n45,n45_ds.coords['range'].values/1000,linewidth=2, label='gain_ratio_n45') 
            #     ax.plot(gain_ratio_p45,n45_ds.coords['range'].values/1000,linewidth=2, label='gain_ratio_p45') 
            #     ax.set_xlim(0,10)
            #     ax.legend()
            
            # pdb.set_trace()
            # signal_ratio = signal_ratio
            # signal_ratio[:] = 2000
            # Polarisation Parameter, a [Freudenthaler, 2016, Eq.61]
            pol_param = (signal_ratio*ghk_param["GT"] - ghk_param["GR"]) / \
                        (ghk_param["HR"] - signal_ratio*ghk_param["HT"])

            # pol_param = (160*ghk_param["GT"] - ghk_param["GR"]) / \
            #              (ghk_param["HR"] - 160*ghk_param["HT"])
            # Volume Linear Depolarization Ratio, delta [Freudenthaler, 2018, Eq.62]
            vldr = (1 - pol_param) / (1 + pol_param)
            
            vldr_avg = np.nanmean(vldr[np.logical_and(ranges >= 5000, ranges <= 7000)])
            vldr_std = np.nanstd(vldr[np.logical_and(ranges >= 5000, ranges <= 7000)])

            """ Save in Dataset DP_DS """
            dp_ds["rs_T_%02d" % channel_count] = xr.DataArray(data=da.from_array(rs_T),
                                                              dims='range',
                                                              attrs={'name': 'RS_T',
                                                                     'long_name': 'Rayleigh signal Transmitted',
                                                                     'units': 'a.u.'})
            dp_ds["rs_R_%02d" % channel_count] = xr.DataArray(data=da.from_array(rs_R),
                                                              dims='range',
                                                              attrs={'name': 'RS_R',
                                                                     'long_name': 'Rayleigh signal Reflected',
                                                                     'units': 'a.u.'})
            dp_ds["n45_T_%02d" % channel_count] = xr.DataArray(data=da.from_array(n45_T),
                                                               dims='range',
                                                               attrs={'name': 'N45_T',
                                                                     'long_name': 'N45 Transmitted',
                                                                     'units': 'a.u.'})
            dp_ds["n45_R_%02d" % channel_count] = xr.DataArray(data=da.from_array(n45_R),
                                                               dims='range',
                                                               attrs={'name': 'N45_R',
                                                                      'long_name': 'N45 Reflected',
                                                                      'units': 'a.u.'})
            dp_ds["p45_T_%02d" % channel_count] = xr.DataArray(data=da.from_array(p45_T),
                                                               dims='range',
                                                               attrs={'name': 'P45_T',
                                                                      'long_name': 'P45 Transmitted',
                                                                      'units': 'a.u.'})
            dp_ds["p45_R_%02d" % channel_count] = xr.DataArray(data=da.from_array(p45_R),
                                                               dims='range',
                                                               attrs={'name': 'P45_R',
                                                                      'long_name': 'P45 Reflected',
                                                                      'units': 'a.u.'})
            dp_ds["vldr_%02d" % channel_count] = xr.DataArray(data=da.from_array(vldr),
                                                              dims='range',
                                                              attrs={'name': 'VLDR',
                                                                     'long_name': 'Volume Linear Depolarization Ratio',
                                                                     'units': '#'})
            dp_ds["vldr_avg_%02d" % channel_count] = xr.DataArray(data=vldr_avg,
                                                                  attrs={'name': 'VLDR Avg',
                                                                         'long_name': 'Volume Linear Depolarization Ratio Avg',
                                                                         'units': '#'})
            dp_ds["vldr_std_%02d" % channel_count] = xr.DataArray(data=vldr_std,
                                                                  attrs={'name': 'VLDR Std',
                                                                         'long_name': 'Volume Linear Depolarization Ratio Std',
                                                                         'units': '#'})
            dp_ds["gain_ratio_n45_%02d" % channel_count] = xr.DataArray(data=da.from_array(gain_ratio_n45),
                                                                        dims='range',
                                                                        attrs={'name': 'gain_ratio_N45',
                                                                               'long_name': 'gain ratio N45',
                                                                               'units': 'a.u.'})
            dp_ds["gain_ratio_p45_%02d" % channel_count] = xr.DataArray(data=da.from_array(gain_ratio_p45),
                                                                        dims='range',
                                                                        attrs={'name': 'gain_ratio_P45',
                                                                               'long_name': 'gain ratio P45',
                                                                               'units': 'a.u.'})
            dp_ds["gain_ratio_Delta90_%02d" % channel_count] = xr.DataArray(data=da.from_array(gain_ratio_Delta90),
                                                                            dims='range',
                                                                            attrs={'name': 'gain_ratio_D90',
                                                                                   'long_name': 'gain ratio D90',
                                                                                   'units': ''})
            dp_ds["gain_ratio_Delta90_avg_%02d" % channel_count] = xr.DataArray(data=gain_ratio_Delta90_avg,
                                                                                attrs={'name': 'gain_ratio_D90 avg',
                                                                                       'long_name': 'gain ratio D90 avg',
                                                                                       'units': ''})
            dp_ds["gain_ratio_Delta90_std_%02d" % channel_count] = xr.DataArray(data=gain_ratio_Delta90_std,
                                                                                attrs={'name': 'gain_ratio_D90 std',
                                                                                       'long_name': 'gain ratio D90 std',
                                                                                       'units': ''})
            dp_ds["Y_%02d" % channel_count] = xr.DataArray(data=da.from_array(Y),
                                                           dims='range',
                                                           attrs={'name': 'Y',
                                                                  'long_name': 'Y',
                                                                  'units': 'a.u.'})
            dp_ds["Y_avg_%02d" % channel_count] = xr.DataArray(data=Y_avg,
                                                               attrs={'name': 'Y avg',
                                                                      'long_name': 'Y avg over calibration height range',
                                                                      'units': 'a.u.'})
            dp_ds["Y_std_%02d" % channel_count] = xr.DataArray(data=Y_std,
                                                               attrs={'name': 'Y std',
                                                                      'long_name': 'Y std over calibration height range',
                                                                      'units': 'a.u.'})
            dp_ds["epsilon_%02d" % channel_count] = xr.DataArray(data=epsilon_iter,
                                                                 attrs={'name': 'epsilon',
                                                                        'long_name': 'epsilon',
                                                                        'units': 'deg'})
            dp_ds["epsilon_error_%02d" % channel_count] = xr.DataArray(data=epsilon_err,
                                                                       attrs={'name': 'epsilon error',
                                                                              'long_name': 'epsilon error',
                                                                              'units': 'deg'})
            dp_ds["alpha_%02d" % channel_count] = xr.DataArray(data=alpha,
                                                               attrs={'name': 'alpha',
                                                                      'long_name': 'alpha',
                                                                      'units': 'deg'})
            dp_ds["alpha_error_%02d" % channel_count] = xr.DataArray(data=alpha_err,
                                                                     attrs={'name': 'alpha error',
                                                                            'long_name': 'alpha error',
                                                                            'units': 'deg'})
            for i_ghk in ghk_param:
                dp_ds["%s_%02d" % (i_ghk, channel_count)] = xr.DataArray(data=ghk_param[i_ghk],
                                                                         attrs={'name': i_ghk,
                                                                                'long_name': i_ghk,
                                                                                'units': 'a.u.'})
            dp_ds["signal_ratio_%02d" % channel_count] = xr.DataArray(data=da.from_array(signal_ratio),
                                                                      dims='range',
                                                                      attrs={'name': 'signal ratio',
                                                                             'long_name': 'signal ratio',
                                                                             'units': ''})
            dp_ds["polarization_parameter_%02d" % channel_count] = xr.DataArray(data=da.from_array(pol_param),
                                                                                dims='range',
                                                                                attrs={'name': 'polarisation parameter',
                                                                                       'long_name': 'polarisation parameter',
                                                                                       'units': ''})
            dp_ds["calibration_factor_%02d" % channel_count] = xr.DataArray(data=da.from_array(cal_factor),
                                                                            dims='range',
                                                                            attrs={'name': 'calibration factor',
                                                                                   'long_name': 'calibration factor',
                                                                                   'units': ''})
            dp_ds["calibration_factor_avg_%02d" % channel_count] = xr.DataArray(data=cal_factor_avg,
                                                                                attrs={'name': 'calibration factor avg',
                                                                                       'long_name': 'calibration factor avg',
                                                                                       'units': ''})
            dp_ds["calibration_factor_std_%02d" % channel_count] = xr.DataArray(data=cal_factor_std,
                                                                                attrs={'name': 'calibration factor std',
                                                                                       'long_name': 'calibration factor std',
                                                                                       'units': ''})
            # Next Channel (WV, MODE)
            channel_count += 1

            """ Save Channel Depol.Cal. in CSV in EARLINET format """
            df_cols = ["ITplus45", "IRplus45", "ITminus45", "IRminus45", "ITRayleigh", "IRRayleigh"]
            dp_df = pd.DataFrame(data=np.array([p45_T, p45_R, n45_T, n45_R, rs_T, rs_R]).T, columns=df_cols, index=ranges)
            dp_df.index.name = "range"
            dp_fn = os.path.join(output_dn, "depolarization_%s_%sx%s.csv" % (cal_label, wv_str, mode_id_str))
            with open(dp_fn, 'w') as f:
                f.write("station ID = %s (%s)\n" % (dp_ds.location_id, dp_ds.location))
                f.write("system = %s\n" % dp_ds.lidar_name)
                f.write("signal = %s, %s\n" % (wv_str, mode_name_str))
                f.write("date of the calibration, time = %s\n" % date_dt.strftime("%d.%m.%Y, %HUTC"))
                f.write("date of the Rayleigh measurement = %s\n" % date_dt.strftime("%d.%m.%Y"))
                f.write("GR, GT, HR, HT, K, laser polarisation rotation, system rotation y "
                        "= %5.3f, %5.3f, %5.3f, %5.3f, %5.3f, %5.3f, %i\n"
                        % (ghk_param['GR'], ghk_param['GT'], ghk_param['HR'],
                           ghk_param['HT'], ghk_param['K2'], alpha, pbs_orientation_parameter))
                f.close()
            dp_df.index = dp_df.index.map(lambda x: '%.4f' % x)
            dp_df.to_csv(dp_fn, mode='a', header=True, na_rep='NaN', float_format='%.4e')

    # Once Loop is over
    # Save Channel id, associated wavelength and detection mode
    channel_coords = np.arange(channel_count)
    dp_ds["channel_id"] = xr.DataArray(data=channel_id, dims='channels',
                                       coords={'channels': channel_coords})
    dp_ds["wavelength"] = xr.DataArray(data=wv_channel, dims='channels',
                                       coords={'channels': channel_coords})
    dp_ds["detection_mode"] = xr.DataArray(data=det_mod_channel, dims='channels',
                                           coords={'channels': channel_coords})
    # Attributes for coord range at the end
    dp_ds['range'].attrs['name'] = "Height"
    dp_ds['range'].attrs['units'] = "km, agl"

    # Save Depolarization Calibration Data in NetCDF
    dp_nc_fn = os.path.join(output_dn, "depolarization_%s.nc" % cal_label)
    dp_ds.to_netcdf(dp_nc_fn)

    """ Plot Depol. Cal. for every wavelength """
    # loop over unique wavelengths
    wvs = np.unique(dp_ds.wavelength)
    for wv in wvs:
        wv_str = str(int(wv))
        # subset variables for wavelength
        idx_wv = dp_ds.wavelength == wv
        dw = dp_ds.sel(channels=idx_wv)
        # id for analog, photoncounting signals
        ch_an = "%02d" % dw.channels[dw.detection_mode == "analog"]
        ch_pc = "%02d" % dw.channels[dw.detection_mode == "photoncounting"]

        # Plot Options
        font = {'size': 12}
        matplotlib.rc('font', **font)
        y_lim = (0, 12)
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15, 10), sharey=True)

        # Title
        fig_title_1 = r"Calibration depolarization analysis: %s. %s - %s" \
                      % (time_start_dt.strftime("%Y%m%d"), time_start_dt.strftime("%H%M"),
                         time_end_dt.strftime("%H%M"))
        fig_title_2 = r"Channel %s. $\alpha_{misalign.}$ ($\circ$)= %4.2f $\pm$ %4.2f" \
                      % (wv_str, dw["alpha_%s" % ch_an], dw["alpha_error_%s" % ch_an])
        fig_title_3 = r"Analog: Calib. factor [%3.1f - %3.1f km] = %6.4f $\pm$ %6.4f ; " \
                      r"$\epsilon_{misalig.}$ ($^\circ$) = %4.2f $\pm$ %4.2f" \
                      % (dw["cal_height_an"][0]*1e-3, dw["cal_height_an"][1]*1e-3,
                         dw["gain_ratio_Delta90_avg_%s" % ch_an], dw["gain_ratio_Delta90_std_%s" % ch_an],
                         dw["epsilon_%s" % ch_an], dw["epsilon_error_%s" % ch_an])
        fig_title_4 = r"Photoncounting: Calib. factor [%3.1f - %3.1f km] = %6.4f $\pm$ %6.4f ; " \
                      r"$\epsilon_{misalig.}$ ($^\circ$) = %4.2f $\pm$ %4.2f" \
                      % (dw["cal_height_pc"][0]*1e-3, dw["cal_height_pc"][1]*1e-3,
                         dw["gain_ratio_Delta90_avg_%s" % ch_pc], dw["gain_ratio_Delta90_std_%s" % ch_pc],
                         dw["epsilon_%s" % ch_pc], dw["epsilon_error_%s" % ch_pc])
        plt.suptitle("%s \n %s \n %s \n %s" % (fig_title_1, fig_title_2, fig_title_3, fig_title_4))

        # Plot RCS: N45, P45, Rayleigh
        dw["p45_T_%s" % ch_an].plot(ax=ax1, y='range', lw=2, c='g', label=r"RCS$^T_{+45}$. AN")
        dw["p45_R_%s" % ch_an].plot(ax=ax1, y='range', lw=1, c='g', label=r"RCS$^R_{+45}$. AN")
        dw["n45_T_%s" % ch_an].plot(ax=ax1, y='range', lw=2, c='r', label=r"RCS$^T_{-45}$. AN")
        dw["n45_R_%s" % ch_an].plot(ax=ax1, y='range', lw=1, c='r', label=r"RCS$^R_{-45}$. AN")
        dw["p45_T_%s" % ch_pc].plot(ax=ax1, y='range', lw=2, c='b', label=r"RCS$^T_{+45}$. PC")
        dw["p45_R_%s" % ch_pc].plot(ax=ax1, y='range', lw=1, c='b', label=r"RCS$^R_{+45}$. PC")
        dw["n45_T_%s" % ch_pc].plot(ax=ax1, y='range', lw=2, c='m', label=r"RCS$^T_{-45}$. PC")
        dw["n45_R_%s" % ch_pc].plot(ax=ax1, y='range', lw=1, c='m', label=r"RCS$^R_{-45}$. PC")
        dw["rs_T_%s" % ch_an].plot(ax=ax1, y='range', lw=2, c='k', label=r"RCS$^T_{Rayleigh}$. AN")
        dw["rs_R_%s" % ch_an].plot(ax=ax1, y='range', lw=1, c='k', label=r"RCS$^R_{Rayleigh}$. AN")
        dw["rs_T_%s" % ch_pc].plot(ax=ax1, y='range', lw=2, c='c', label=r"RCS$^T_{Rayleigh}$. PC")
        dw["rs_R_%s" % ch_pc].plot(ax=ax1, y='range', lw=1, c='c', label=r"RCS$^R_{Rayleigh}$. PC")
        ax1.grid()
        ax1.axes.set_xlabel(r"RCS [a.u.]")
        ax1.axes.set_ylabel(r"Height [km, agl]")
        ax1.axes.set_ylim(y_lim)
        ax1.axes.set_xscale('log')
        ax1.legend(fontsize='small', loc=2)

        # Plot Gain Ratio (eta^* x45)
        dw["gain_ratio_p45_%s" % ch_an].plot(ax=ax2, y='range', c='b', label=r"AN at +45")
        dw["gain_ratio_n45_%s" % ch_an].plot(ax=ax2, y='range', c='r', label=r"AN at -45")
        dw["gain_ratio_p45_%s" % ch_pc].plot(ax=ax2, y='range', c='g', label=r"PC at +45")
        dw["gain_ratio_n45_%s" % ch_pc].plot(ax=ax2, y='range', c='k', label=r"PC at -45")
        ax2.grid()
        ax2.axes.set_xlabel(r"Signal Ratio [R / T]")
        ax2.axes.set_ylabel(r"")
        ax2.legend(fontsize='small', loc=7)
        ax2.axes.set_xlim((0, 1))

        # Plot Gain Ratio (eta^* Delta90)
        dw["gain_ratio_Delta90_%s" % ch_an].plot(ax=ax3, y='range', c='g', label=r"AN")
        dw["gain_ratio_Delta90_%s" % ch_pc].plot(ax=ax3, y='range', c='m', label=r"PC")
        ax3.grid()
        ax3.axes.set_xlabel(r"Calibration Factor")
        ax3.axes.set_ylabel(r"")
        ax3.legend(fontsize='small', loc=7)
        ax3.axes.set_xlim((0, 1))

        # Plot VLDR
        #(0.0848*dw["rs_T_%s" % ch_an] / dw["rs_R_%s" % ch_an]).plot(ax=ax4, y='range', color='k', ls=':')
        #(0.1295*dw["rs_T_%s" % ch_pc] / dw["rs_R_%s" % ch_pc]).plot(ax=ax4, y='range', color='r', ls=':')

        dw["vldr_%s" % ch_an].plot(ax=ax4, y='range', c='g',
                                   label=r"AN. %5.3f $\pm$ %5.3f" %
                                         (dw["vldr_avg_%s" % ch_an], dw["vldr_std_%s" % ch_an]))
        dw["vldr_%s" % ch_pc].plot(ax=ax4, y='range', c='m',
                                   label=r"PC. %5.3f $\pm$ %5.3f" %
                                         (dw["vldr_avg_%s" % ch_pc], dw["vldr_std_%s" % ch_pc]))
        (dw["vldr_%s" % ch_pc]*0.0 + 0.00355).plot(ax=ax4, y='range', c='r', label=r"Molecular: 0.0035")
        # pdb.set_trace()
        # dfLVD = pd.read_excel('C:/datos/Code/VolumeDepolarizationRatio.xlsx',index_col = 0)
        # ax4.plot(dfLVD.values,dfLVD.index/1000,color='k',lw=2)
        ax4.grid()
        ax4.axes.set_xlabel(r"VLDR")
        ax4.axes.set_ylabel(r"")
        ax4.legend(fontsize='small', loc=7)
        ax4.axes.set_xscale('log')
        ax4.axes.set_xlim((1e-3, 1))

        plot_fn = os.path.join(output_dn, "plot_depolarization_%s_%s.png" % (cal_label, wv_str))
        plt.savefig(plot_fn, dpi=200, bbox_inches="tight")

    return dp_ds


def plot_depolarization(ds, output_dir=None, plot_label=None):
    """

    """

    if output_dir is None:
        output_dir = os.getcwd()
    if plot_label is None:
        plot_label = 'xxx'

    # Range. Limits
    if ds["range"].attrs["units"] == 'm':
        ds["range"] = ds["range"].values*1e-3
        ds["range"].attrs['units'] = 'km'
    y_lim = (0, 12)

    # PLOT
    font = {'size': 12}
    matplotlib.rc('font', **font)

    # times
    time_start = pd.to_datetime(ds.time_start.values)
    time_end = pd.to_datetime(ds.time_end.values)

    # pbs orientation parameter
    pbs_orientation_param = ds["pbs_orientation"].values
    if pbs_orientation_param == -1:
        r_id = "p"
        t_id = "c"
    else:  # +1
        r_id = "c"
        t_id = "p"

    # loop over wavelengths
    for wv in np.unique(ds.wavelength.values):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15, 10), sharey=True)

        # colorbar = matplotlib.cm.get_cmap('jet', len(rf_ds.sectors))
        # colors = colorbar(np.linspace(0, 1, len(rf_ds.sectors)))

        # Title
        fig_title_1 = r"Calibration depolarization analysis: %s. %s - %s" % (time_start.strftime("%Y%m%d"), time_start.strftime("%H%M"), time_end.strftime("%H%M"))
        fig_title_2 = r"Analog: Calib. factor [%3.1f - %3.1f km] = %4.2f $\pm$ %6.4f ; $\epsilon_{misalig.}$ ($^\circ$) = %4.2f $\pm$ %4.2f" % (ds.min_calib_height_a*1e-3, ds.max_calib_height_a*1e-3, ds["gain_ratio_%ixa_avg"%wv], ds["gain_ratio_%ixa_std"%wv], ds["epsilon_%ixa"%wv], ds["epsilon_error_%ixa"%wv])
        fig_title_3 = r"Photoncount: Calib. factor [%3.1f - %3.1f km] = %4.2f $\pm$ %6.4f ; $\epsilon_{misalig.}$ ($^\circ$) = %4.2f $\pm$ %4.2f" % (ds.min_calib_height_p*1e-3, ds.max_calib_height_p*1e-3, ds["gain_ratio_%ixp_avg"%wv], ds["gain_ratio_%ixp_std"%wv], ds["epsilon_%ixp"%wv], ds["epsilon_error_%ixp"%wv])
        plt.suptitle("%s \n %s \n %s" % (fig_title_1, fig_title_2, fig_title_3))

        # Plot RCS
        ds["corrected_rcs_%ix%sa_P45" % (wv, t_id)].plot(ax=ax1, y='range', lw=2, c='g', label=r"RCS$^T_{+45}$. AN")
        ds["corrected_rcs_%ix%sa_P45" % (wv, r_id)].plot(ax=ax1, y='range', lw=1, c='g', label=r"RCS$^R_{+45}$. AN")
        ds["corrected_rcs_%ix%sa_N45" % (wv, t_id)].plot(ax=ax1, y='range', lw=2, c='r', label=r"RCS$^T_{-45}$. AN")
        ds["corrected_rcs_%ix%sa_N45" % (wv, r_id)].plot(ax=ax1, y='range', lw=1, c='r', label=r"RCS$^R_{-45}$. AN")
        ds["corrected_rcs_%ix%sp_P45" % (wv, t_id)].plot(ax=ax1, y='range', lw=2, c='b', label=r"RCS$^T_{+45}$. PC")
        ds["corrected_rcs_%ix%sp_P45" % (wv, r_id)].plot(ax=ax1, y='range', lw=1, c='b', label=r"RCS$^R_{+45}$. PC")
        ds["corrected_rcs_%ix%sp_N45" % (wv, t_id)].plot(ax=ax1, y='range', lw=2, c='m', label=r"RCS$^T_{-45}$. PC")
        ds["corrected_rcs_%ix%sp_N45" % (wv, r_id)].plot(ax=ax1, y='range', lw=1, c='m', label=r"RCS$^R_{-45}$. PC")
        ds["RS_corrected_rcs_%ix%sa" % (wv, t_id)].plot(ax=ax1, y='range', lw=2, c='k', label=r"RCS$^T_{Rayleigh}$. AN")
        ds["RS_corrected_rcs_%ix%sa" % (wv, r_id)].plot(ax=ax1, y='range', lw=1, c='k', label=r"RCS$^R_{Rayleigh}$. AN")
        ds["RS_corrected_rcs_%ix%sp" % (wv, t_id)].plot(ax=ax1, y='range', lw=1, c='c', label=r"RCS$^T_{Rayleigh}$. PC")
        ds["RS_corrected_rcs_%ix%sp" % (wv, r_id)].plot(ax=ax1, y='range', lw=1, c='c', label=r"RCS$^R_{Rayleigh}$. PC")
        ax1.grid()
        ax1.axes.set_xlabel(r"RCS [a.u.]")
        ax1.axes.set_ylabel(r"range [km, agl]")
        ax1.axes.set_ylim(y_lim)
        ax1.axes.set_xscale('log')
        ax1.legend(fontsize='small', loc=2)

        # Plot Signal Ratio
        ds["gain_ratio_P45_%ixa" % wv].plot(ax=ax2, y='range', c='b', label=r"AN at +45")
        ds["gain_ratio_N45_%ixa" % wv].plot(ax=ax2, y='range', c='r', label=r"AN at -45")
        ds["gain_ratio_P45_%ixp" % wv].plot(ax=ax2, y='range', c='g', label=r"PC at +45")
        ds["gain_ratio_N45_%ixp" % wv].plot(ax=ax2, y='range', c='k', label=r"PC at -45")
        ax2.grid()
        ax2.axes.set_xlabel(r"Signal Ratio [R / T]")
        ax2.axes.set_ylabel(r"")
        ax2.legend(fontsize='small', loc=7)
        ax2.axes.set_xlim((0, 1))

        # Plot Calibration Factor
        ds["gain_ratio_%ixa" % wv].plot(ax=ax3, y='range', c='g', label=r"AN")
        ds["gain_ratio_%ixp" % wv].plot(ax=ax3, y='range', c='m', label=r"PC")
        ax3.grid()
        ax3.axes.set_xlabel(r"Calibration Factor")
        ax3.axes.set_ylabel(r"")
        ax3.legend(fontsize='small', loc=7)
        ax3.axes.set_xlim((0, 1))
        
        # Plot LVDR
        ds["LVDR_%ixa" % wv].plot(ax=ax4, y='range', c='g', label=r"AN. %5.3f $\pm$ %5.3f" % (ds["LVDR_%ixa_avg" % wv].values, ds["LVDR_%ixa_std" % wv].values))
        ds["LVDR_%ixp" % wv].plot(ax=ax4, y='range', c='m', label=r"PC. %5.3f $\pm$ %5.3f" % (ds["LVDR_%ixp_avg" % wv].values, ds["LVDR_%ixp_std" % wv].values))
        (ds["LVDR_%ixp" % wv]*0.0 + 0.00355).plot(ax=ax4, y='range', c='r', label=r"LVDR Molecular")
        ax4.grid()
        ax4.axes.set_xlabel(r"LVDR")
        ax4.axes.set_ylabel(r"")
        ax4.legend(fontsize='small', loc=7)
        # ax4.axes.set_xscale('log')
        ax4.axes.set_xlim((0, 1))

        # Save Fig
        fig_fn = os.path.join(output_dir, "plot_depolarization_%s_%i.png" % (plot_label, wv))
        plt.savefig(fig_fn, dpi=200, bbox_inches="tight")


"""
Setup Functions
"""


def setup_rayleigh_fit(parser):

    # function for pass parser as arguments for rayleigh_fit
    def rayleigh_fit_from_args(parsed):
        rayleigh_fit(rs_fl=parsed.rs_fl, dc_fl=parsed.dc_fl, date_str=parsed.date_str,
                     hour_ini=parsed.hour_ini, hour_end=parsed.hour_end, duration=parsed.duration,
                     lidar_name=parsed.lidar_name, meas_type=parsed.meas_type, channels=parsed.channels,
                     z_min=parsed.z_min, z_max=parsed.z_max, smooth_range=parsed.smooth_range,
                     range_min=parsed.range_min, range_max=parsed.range_max,
                     meteo=parsed.meteo, pressure_prf=parsed.pressure_prf, temperature_prf=parsed.temperature_prf,
                     data_dn=parsed.data_dn, ecwmf_dn=parsed.ecmwf_dn, level_1a_dn=parsed.level_1a_dn,
                     output_dn=parsed.output_dn, save_fig=parsed.save_fig)

    def datetime_fmt(arg):
        if re.match(r'\d{4}\d{2}\d{2}T\d{2}:\d{2}:\d{2}.\d{1}', arg):
            return arg
            raise argparse.ArgumentTypeError("Date must be in format 'yyyymmddTHH:MM:SS.S'")
    def positive_val(arg):
        arg = float(arg)
        if arg >= 0:
            return arg
        else:
            raise argparse.ArgumentTypeError("Input argument must be >= 0")
    def none_val(arg):
        if arg is None:
            raise argparse.ArgumentTypeError("Input argument must be something")
        else:
            return arg

    # Add Input Arguments into Parser
    parser.add_argument("-rs_fl", "--rs_fl", help="wildcard of measurement files", default=None)  # type=none_val
    parser.add_argument("-dc_fl", "--dc_fl", help="wildcard of dark current files", default=None)
    parser.add_argument("-date_str", "--date_str", help="date (YYYYMMDD)", default=None)  #, type=datetime_fmt)
    parser.add_argument("-hour_ini", "--hour_ini", help="start time (HH, HHMM, HHMMSS)", default=None)  #, type=datetime_fmt)
    parser.add_argument("-hour_end", "--hour_end", help="end time (HH, HHMM, HHMMSS)", default=None)  #, type=datetime_fmt)
    parser.add_argument("-duration", "--duration", help="minutes to accumulate measurements for rayleigh fit", type=positive_val, default=None)
    parser.add_argument("-lidar_name", "--lidar_name", help="lidar name (VELETA, MULHACEN)", default="MULHACEN")
    parser.add_argument("-meas_type", "--meas_type", help="measurement type (RS, OT, ...)", default="RS")
    parser.add_argument("-channels", "--channels", help="select channels: 'all', 0, 1, ...", default='all') #, choices=['all']
    parser.add_argument("-z_min", "--z_min", help="minimum altitude we use as base for our rayleigh fit", type=positive_val, default=5000)
    parser.add_argument("-z_max", "--z_max", help="maximum altitude we use as base for our rayleigh fit", type=positive_val, default=6000)
    parser.add_argument("-smooth_range", "--smooth_range", help="range resolution window for smoothing profiles", type=positive_val, default=250)
    parser.add_argument("-range_min", "--range_min", help="minimum range", type=positive_val, default=15)
    parser.add_argument("-range_max", "--range_max", help="maximum range", type=positive_val, default=30000)
    parser.add_argument("-meteo", "--meteo", help="identifier for hydrostatic (P, T) data source (ecmwf (default), user, lidar, grawmet)", choices=['ecmwf', 'lidar', 'grawmet', 'user'], default="ecmwf")
    parser.add_argument("-pressure_prf", "--pressure_prf", help="Pressure profile given by user", type=positive_val, default=None)
    parser.add_argument("-temperature_prf", "--temperature_prf", help="Temperature profile given by user", type=positive_val, default=None)
    parser.add_argument("-data_dn", "--data_dn", help="disk directory for all data", default="GFATserver")
    parser.add_argument("-ecmwf_dn", "--ecmwf_dn", help="ecmwf data directory (ecmwf_dn/yyyy/)", default="GFATserver")
    parser.add_argument("-level_1a_dn", "--level_1a_dn", help="level 1a data directory", default="GFATserver")
    parser.add_argument("-output_dn", "--output_dn", help="directory to store rayleigh fit results", default="GFATserver")
    parser.add_argument("-save_fig", "--save_fig", help="save figure in png format", default=True)

    # Set Function to Be executed
    parser.set_defaults(execute=rayleigh_fit_from_args)


def setup_telecover(parser):
    #(date_str, level_1a_dn='GFATserver', rayleigh_fit_dn='GFATserver', savefig=True):
    # function for pass parser as arguments for rayleigh_fit
    def telecover_from_args(parsed):
        telecover(parsed.date_str, zmin=parsed.zmin, zmax=parsed.zmax, input_directory=parsed.in_dir, output_directory=parsed.out_dir, savefig=parsed.savefig)

    # opciones para los argumentos
    def datetime_fmt(arg):
        if re.match(r'\d{4}\d{2}\d{2}', arg):
            return arg
        else:
            raise argparse.ArgumentTypeError("Date must be in format 'yyyymmdd'")
    def positive_val(arg):
        arg = float(arg)
        if arg >= 0:
            return arg
        else:
            raise argparse.ArgumentTypeError("Input argument must be >= 0")

    # Add Input Arguments into Parser
    parser.add_argument("-date_str", "--date_str", help="initial date", type=datetime_fmt)
    parser.add_argument("-zmin", "--zmin", help="min height for normalization", type=positive_val, default=2000)
    parser.add_argument("-zmax", "--zmax", help="Filter for only the selected station", type=positive_val, default=3000)
    parser.add_argument("-savefig", "--savefig", help="save figure in disk", default=True)
    parser.add_argument("-in_dir", "--in_dir", help="disk directory to store output files/figures", default='GFATserver')
    parser.add_argument("-out_dir", "--out_dir", help="disk directory to store output files/figures", default='GFATserver')

    # Set Function to Be executed
    parser.set_defaults(execute=telecover_from_args)


def setup_depolarization(parser):
    # function for pass parser as arguments for depolarization
    def depolarization_from_args(parsed):
        depolarization_cal(p45_fn=parsed.p45_fn, n45_fn=parsed.n45_fn,
                           rs_fn=parsed.rs_fn, dc_fn=parsed.dc_fn,
                           cal_height_an=parsed.cal_height_an, cal_height_pc=parsed.cal_height_pc,
                           pbs_orientation_parameter=parsed.pbs_orientation_parameter,
                           cal_type=parsed.cal_type, channels=parsed.channels,
                           range_min=parsed.range_min, range_max=parsed.range_max,
                           alpha=parsed.alpha, epsilon=parsed.epsilon, ghk_tpl_fn=parsed.ghk_tpl_fn,
                           output_dn=parsed.output_dn)

    # opciones para los argumentos
    def datetime_fmt(arg):
        if re.match(r'\d{4}\d{2}\d{2}', arg):
            return arg
        else:
            raise argparse.ArgumentTypeError("Date must be in format 'yyyymmdd'")
    def positive_val(arg):
        arg = float(arg)
        if arg >= 0:
            return arg
        else:
            raise argparse.ArgumentTypeError("Input argument must be >= 0")

    # Add Input Arguments into Parser
    parser.add_argument("-p45_fn", "--p45_fn", help="file name for P45 measurement (netcdf)")
    parser.add_argument("-n45_fn", "--n45_fn", help="file name for N45 measurement (netcdf)")
    parser.add_argument("-rs_fn", "--rs_fn", help="file name for RS complementary measurement (netcdf)")
    parser.add_argument("-dc_fn", "--dc_fn", help="filename of lidar dark current signal level 1a files (*.nc)")
    parser.add_argument("-cal_height_an", "--cal_height_an", help="height range for averaging calibrations for analog signals")
    parser.add_argument("-cal_height_pc", "--cal_height_pc", help="height range for averaging calibrations for photoncounting signals")
    parser.add_argument("-pbs_orientation_parameter", "--pbs_orientation_parameter", help="polarising beam splitter orientation parameter, y [Freudenthaler, 2016, sec3.3]")
    parser.add_argument("-cal_type", "--cal_type", help="type of calibration: rotator (rot), polarizer (pol)")
    parser.add_argument("-channels", "--channels", help="list of channels to process")
    parser.add_argument("-range_min", "--range_min", help="minimum range")
    parser.add_argument("-range_max", "--range_max", help="maximum range")
    parser.add_argument("-alpha", "--alpha", help="rotational misalignment of the polarizing plane of the laser light respect to the incident plane of the PBS")
    parser.add_argument("-epsilon", "--epsilon", help="misalignment angle of the rotator")
    parser.add_argument("-ghk_tpl_fn", "--ghk_tpl_fn", help="ghk file template for running GHK")
    parser.add_argument("-output_dn", "--output_dn", help=" directory to store depolarization calibration results")

    # Set Function to Be executed
    parser.set_defaults(execute=depolarization_from_args)


# TODO: settings file
#def settings_from_path(config_file_path):
#    """ Read the configuration file.
#
#    The file should be in YAML syntax."""
#
#    if not os.path.isfile(config_file_path):
#        raise argparse.ArgumentTypeError("Wrong path for configuration file (%s)" % config_file_path)
#
#    with open(config_file_path) as yaml_file:
#        try:
#            settings = yaml.safe_load(yaml_file)
#            logger.debug("Read settings file(%s)" % config_file_path)
#        except Exception:
#            raise argparse.ArgumentTypeError("Could not parse YAML file (%s)" % config_file_path)
#
#    # YAML limitation: does not read tuples
#    settings['basic_credentials'] = tuple(settings['basic_credentials'])
#    settings['website_credentials'] = tuple(settings['website_credentials'])
#    return settings


def main():

    # Define the command line arguments.
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # Functions
    # TODO: crear una opcion default
    rf_parser = subparsers.add_parser("RF", help="Rayleigh Fit")
    tc_parser = subparsers.add_parser("TC", help="Telecover")
    dp_parser = subparsers.add_parser("DP", help="Depolarization")

    # Setup Functions
    #setup_void(void_parser)
    setup_rayleigh_fit(rf_parser)
    setup_telecover(tc_parser)
    setup_depolarization(dp_parser)

    # Logging: Verbosity settings from http://stackoverflow.com/a/20663028
    parser.add_argument('-d', '--debug', help="Print debugging information.", action="store_const",
                        dest="loglevel", const=logging.DEBUG, default=logging.INFO,
                        )  # DEFAULT value for loglevel
    parser.add_argument('-s', '--silent', help="Show only warning and error messages.", action="store_const",
                        dest="loglevel", const=logging.WARNING
                        )

    # Configuration File. TODO
    """
    default_config_location = os.path.abspath(os.path.join(home, ".scc_access.yaml"))
    parser.add_argument("-c", "--config", help="Path to the config file.", type=settings_from_path)
                        default=default_config_location)
    """

    # Parse Args
    args = parser.parse_args()

    # Get the logger with the appropriate level
    logging.basicConfig(format='%(levelname)s: %(message)s', level=args.loglevel)

    # Dispatch to appropriate function
    args.execute(args)
    
    
if __name__== "__main__":
    main()
