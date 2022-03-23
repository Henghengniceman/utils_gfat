import os
import sys
import glob
import numpy as np
import xarray as xr
import datetime as dt
import logging
import time
import pdb

from .ceilometer_utils import *

""" logging  """
log_formatter = logging.Formatter('%(levelname)s: %(funcName)s(). L%(lineno)s: %(message)s')
logger = logging.getLogger(__name__)
if (logger.hasHandlers()):
    logger.handlers.clear()
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(log_formatter)
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.propagate = False


""" Module for Ceilometer Data Processing
"""

def reader_chm15k(fn, fields=None):
    """ Reader Function for NetCDF files generated at CHM15K

    Parameters
    ----------
    fn : str
        filename
    fields: list(str)
        field names to extract
        fields=['beta_raw', 'cbh', 'altitude', 'latitude', 'longitude', 'wavelength']

    Returns
    -------

    """

    try:
        if os.path.isfile(fn):
            with xr.open_dataset(fn, chunks={}) as dx:
                if fields is not None:
                    dx = dx[fields]
        else:
            print("File %s does not exist" % fn)
    except Exception as e:
        logger.error(str(e))
        logger.error("File %s Not Read. Return None" % fn)
        dx = None

    return dx


def preprocessing(cei_fl, cal_factor, cal_factor_err, fields=None, h_min=0, h_max=25000):
    #, window_for_snr=200): #, height_for_overlap_correction=300):
    """
    preprocess_ceilometer [summary]
    + Preprocess a list of files
    + Concatenate preprocesssed files by time

    Parameters
    ----------
    cei_fl : str
        wildcard of ceilometer nc files
    cal_factor : float
        [description]
    cal_factor_err : float
        [description]
    fields: list
        fields to be read from nc files
    h_min: float
        min output height above sea level
    h_max: float
        max output height above sea level

    Returns
    -------
    ps_ds: xarray.Dataset
        preprocessed signal
    """

    logger.info("Start Preprocessing")
    t_i = time.time()
    ps_ds = None
    try:
        cei_fns = glob.glob(cei_fl)
        if len(cei_fns) > 0:
            for cei_fn in cei_fns:
                rs_ds = reader_chm15k(cei_fn, fields=fields)
                if rs_ds is not None:
                    # Beta Raw and Ranges
                    ranges = rs_ds['range']
                    beta_raw = rs_ds['beta_raw']

                    # Output Ranges
                    idx_range = np.logical_and(ranges >= h_min, ranges <= h_max)
                    # Create Output Dataset
                    ps_ds_i = rs_ds.sel(range=ranges[idx_range])

                    # Preprocess Ceilometer Raw Signal
                    #beta_att , rcs, signal, bg, bg_err, raw_signal = \
                    beta_att = xr.apply_ufunc(preprocess, beta_raw, ranges, cal_factor, dask='allowed', \
                        input_core_dims=[['time', 'range'], ['range'], []], \
                        output_core_dims=[['time', 'range']])
                        #, ['time', 'range'], \
                        #    ['time', 'range'], ['time'], ['time'], ['time', 'range']])

#                    """ New variables in Dataset """
#                    # Raw Signal
#                    attrs = {'units': 'm-3 sr-1', 'name': 'raw signal', 'long_name': 'raw_signal'}
#                    ps_ds_i['signal_raw'] = raw_signal[:, idx_range].assign_attrs(attrs)
#                    # Background
#                    attrs = {'units': 'm-3 sr-1', 'name': 'background', 'long_name': 'background'}
#                    ps_ds_i['background'] = bg.assign_attrs(attrs)
#                    # Background Error
#                    attrs = {'units': 'm-3 sr-1', 'name': 'background error', 'long_name': 'background_error'}
#                    ps_ds_i['background_error'] = bg_err.assign_attrs(attrs)
#                    # Corrected Signal
#                    attrs = {'units': 'm-3 sr-1', 'name': 'corrected signal', 'long_name': 'corrected_signal'}
#                    ps_ds_i['signal'] = signal.assign_attrs(attrs)
#                    # Corrected Range Corrected Signal
#                    attrs = {'units': 'm-1 sr-1', 'name': 'corrected rcs', 'long_name': 'corrected_rcs'}
#                    ps_ds_i['rcs'] = rcs.assign_attrs(attrs)

                    # Attenuated Backscatter
                    attrs = {'units': 'm-1 sr-1', 'name': 'attenuated backscatter coefficient', 'long_name': r'$\beta_{att}$'}
                    ps_ds_i['beta_att'] = beta_att.assign_attrs(attrs)

                    # Height Above Sea Level
                    height = ps_ds_i['range'] + ps_ds_i['altitude']
                    attrs = ps_ds_i['range'].attrs
                    attrs['name'] = 'height above sea level'
                    attrs['long_name'] = 'height (asl)'
                    ps_ds_i['height'] = height.assign_attrs(attrs)
                    # range -> height [above mean sea level]
                    #cei_ds = cei_ds.assign_coords(height=("range", cei_ds.range + cei_ds.altitude))
                    #cei_ds = cei_ds.swap_dims({"range": "height"})
                    #cei_ds = cei_ds.drop('range')

                    # Calibration Factor
                    ps_ds_i['calibration_factor'] = cal_factor
                    ps_ds_i['calibration_factor'].attrs['long_name'] = 'calibration'
                    ps_ds_i['calibration_factor_error'] = cal_factor_err
                    ps_ds_i['calibration_factor_error'].attrs['long_name'] = 'calibration error'

                    # Concatenate Preprocessed Signals
                    if ps_ds is None:
                        ps_ds = ps_ds_i
                    else:
                        ps_ds = xr.concat([ps_ds, ps_ds_i], 'time')#, coords='minimal', vars='minimal')
        else:
            logger.error("Preprocessing not performed for %s. Return None" % cei_fl)

    except Exception as e:
        logger.error(str(e))
        logger.error("Preprocessing not performed for %s. Return None" % cei_fl)
    t_e = time.time()
    logger.info("End Preprocessing. [%f sec]" % (t_e - t_i))

    return ps_ds


        
        
        
    
    """
    # Set Value of First Ranges



    # Signal To Noise Ratio 
    #cei_ds['snr'] = xr.apply_ufunc(signal_to_noise_ratio, \
    #    cei_ds['signal'], cei_ds['background'], dask='allowed', \
    #        input_core_dims=[['time', 'range'], ['time']], \
    #            output_core_dims=[['time', 'range']])
    #cei_ds['snr'].attrs = cei_ds['beta_raw'].attrs
    #cei_ds['snr'].attrs['units'] = ''
    #cei_ds['snr'].attrs['name'] = 'snr'
    #cei_ds['snr'].attrs['long_name'] = 'signal to noise ratio'
    w = int(window_for_snr / cei_ds['range'].diff(dim='range').mean())
    cei_ds['snr'] = cei_ds['signal'].rolling(range=w).mean() / \
        cei_ds['signal'].rolling(range=w).std()
    cei_ds['snr'].attrs = cei_ds['beta_raw'].attrs
    cei_ds['snr'].attrs['units'] = ''
    cei_ds['snr'].attrs['name'] = 'snr'
    cei_ds['snr'].attrs['long_name'] = 'signal to noise ratio'

    # RCS
    cei_ds['rcs'] = cei_ds['signal'] * cei_ds['range']**2
    cei_ds['rcs'].attrs = cei_ds['beta_raw'].attrs
    cei_ds['rcs'].attrs['units'] = 'm-1 sr-1'
    cei_ds['rcs'].attrs['name'] = 'rcs'
    cei_ds['rcs'].attrs['long_name'] = 'range corrected signal'

    # Correct Signal with Calibration Factor
    cei_ds['beta_att'] = cei_ds["rcs"] / cal_factor
    cei_ds['beta_att'].attrs = cei_ds['rcs'].attrs
    cei_ds['beta_att'].attrs['units'] = 'm-1 sr-1'
    cei_ds['beta_att'].attrs['name'] = 'attenuated backscatter coefficient'
    cei_ds['beta_att'].attrs['long_name'] = 'attenuated backscatter coefficient'
    cei_ds['calibration_factor'] = cal_factor
    cei_ds['calibration_factor'].attrs['long_name'] = 'Calibration'
    cei_ds['calibration_factor_error'] = cal_factor_err
    cei_ds['calibration_factor_error'].attrs['long_name'] = 'Calibration Error'

    # Keep First layer of CBH and drop otherwise (closer to cloudnet cbh values)
    #cei_ds['cbh'] = cei_ds.cbh.isel(layer=0).drop('layer')


    return cei_ds
    """