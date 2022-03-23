import os, sys, glob
import numpy as np
import pandas as pd
import xarray as xr
import datetime as dt
import logging
import pdb

# MODULE DIR
MODULE_DIR = os.path.dirname(sys.modules[__name__].__file__)

import lidar, solar

""" Functions for reading L1a Files
"""
### Funcion para obtener la ruta del directorio del dia
def get_1a_day_dn(date_str, data_dn, lidar_name=None):
    """[summary]

    Args:
        date_str ([type]): [description]
        data_dn ([type]): [description]

    Returns:
        [type]: [description]
    """
    if lidar_name is None:
        lidar_name = "MULHACEN"
    return os.path.join(data_dn, lidar_name, "1a", date_str[:4], date_str[4:6], date_str[6:])


### Funcion para construir la ruta de un archivo 1a
def get_1a_fn(date_str, m_type, data_dn, lidar_name=None):
    """[summary]

    Args:
        date_str ([type]): [description]
        m_type ([type]): [description]
        data_dn ([type]): [description]

    Returns:
        [type]: [description]
    """
    day_dn = get_1a_day_dn(date_str, data_dn, lidar_name=lidar_name)
    rs_fn = glob.glob(os.path.join(day_dn, '*%s*.nc' % m_type))
    if len(rs_fn) > 0:
        rs_fn = rs_fn[0]
    else:
        rs_fn = None
        print("No File found for %s and %s" % (date_str, m_type))
    return rs_fn


def build_lidar_fn_wildcard(date_str, meas_mode, data_dn, lidar_name=None):
    """[summary]

    Args:
        date_str ([type]): [description]
        meas_mode ([type]): [description]
        data_dn ([type]): [description]
        lidar_name ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    if lidar_name is None:
        lidar_name = "MULHACEN"

    date_dt = dt.datetime.strptime(date_str, "%Y%m%d")
    dn_1a = os.path.join(data_dn, lidar_name, "1a", "%04d" % date_dt.year,
                         "%02d" % date_dt.month, "%02d" % date_dt.day)
    fl = os.path.join(dn_1a, "*%s*%s*.nc" % (meas_mode, date_str))

    return fl


def read_1a_lidar(date_str, meas_mode, data_dn):
    """[summary]

    Args:
        date_str ([type]): [description]
        meas_mode ([type]): [description]
        data_dn ([type]): [description]

    Raises:
        ValueError: [description]

    Returns:
        [type]: [description]
    """

    # measurement mode
    if meas_mode == 'RS':
        meas_mode_str = 'Prs'
    elif meas_mode == 'DC':
        meas_mode_str = 'Pdc'
    elif meas_mode == 'OT':
        meas_mode_str = 'Pot'
    elif meas_mode == 'DP-P45':
        meas_mode_str = 'Pdp-P45'
    elif meas_mode == 'DP-N45':
        meas_mode_str = 'Pdp-N45'
    else:
        raise ValueError("Measurement Mode not implemented")

    fl_wc = build_lidar_fn_wildcard(date_str, meas_mode_str, data_dn)
    try:
        ds = lidar.reader_xarray(fl_wc)
        if len(ds) == 0:
            print("ERROR. No measurements")
            ds = None
    except Exception as e:
        print("ERROR. %s" % str(e))
        ds = None

    return ds


def get_prepared_data(date_str, channel, don, max_range, data_dn, time_range=None):
    """
    Input:
    date_str = "20200528"
    channel = '532p' # '532p', '532c', '355'
    don = 1 # 0 (all), 1 (day), 2 (ngt)
    max_height = 15000
    time_range = (h_ini, h_end). floats

    Output:

    """

    # Options
    if time_range is not None:
        don = 0

    # ------ DATA READING
    rs_a_ds = read_1a_lidar(date_str, "RS", data_dn)
    dc_a_ds = read_1a_lidar(date_str, "DC", data_dn)

    data = {}
    if np.logical_and(rs_a_ds is not None, dc_a_ds is not None):
        # ------ Illumination Conditions
        if rs_a_ds.lon.shape:
            lat = float(rs_a_ds.lon.values[0])
            lon = float(rs_a_ds.lat.values[0])
            alt = float(rs_a_ds.altitude.values[0])
        else:
            lat = float(rs_a_ds.lon.values)
            lon = float(rs_a_ds.lat.values)
            alt = float(rs_a_ds.altitude.values)
        times_a = np.sort(np.append(rs_a_ds.time.values, dc_a_ds.time.values))
        sun = solar.SUN(times_a, lon, lat, elev=alt)
        csza = sun.get_csza()
        if don == 1:  # DAY
            idx_ilu = np.where(csza > 0.01)  # (np.logical_and(csza>0.8,csza<1))
        elif don == 2:  # NGT
            idx_ilu = np.where(csza < -0.01)  #(np.logical_and(csza>-0.2,csza<0))
        elif don == 0:  # ALL
            idx_ilu = np.where(csza > -9999)
        else:
            times_aux = pd.to_datetime(times_a)
            idx_ilu = np.where(np.logical_and(times_aux.hour >= time_range[0],
                                              times_aux.hour < time_range[1]))
        times_ilu = times_a[idx_ilu]
        # Data Selection by illumination conditions: RS, DC
        rs_ds = rs_a_ds.sel(time=slice(times_ilu[0], times_ilu[-1]), range=slice(max_range))
        if np.logical_and((csza[idx_ilu] > 0).any(), not (csza[idx_ilu] < 0).any()):
            idx_dc = np.where(csza > 0.01)
            times_dc = times_a[idx_dc]
        elif np.logical_and(not (csza[idx_ilu] > 0).any(), (csza[idx_ilu] < 0).any()):
            idx_dc = np.where(csza < 0.01)
            times_dc = times_a[idx_dc]
        else:
            times_dc = times_a
        dc_ds = dc_a_ds.sel(time=slice(times_dc[0], times_dc[-1]), range=slice(max_range))

        # ------ CHANNELS
        # correspondence channel <-> number
        #                00        01        02        03        04        05
        channels = ['532xpa', '532xpp', '532xca', '532xcp', '355xna', '355xnp']

        # Channel selection
        if channel == '532p':
            an_ch = "00"
            pc_ch = "01"
            wv = 532
            pol = 1
        elif channel == '532c':
            an_ch = "02"
            pc_ch = "03"
            wv = 532
            pol = 2
        elif channel == '355':
            an_ch = "04"
            pc_ch = "05"
            wv = 355
            pol = 0

        # ----- PREPARE DATA
        h = rs_ds['range']
        times = rs_ds['time']
        an = rs_ds["signal_%s" % an_ch]
        pc = rs_ds["signal_%s" % pc_ch]
        an_rcs = rs_ds["rcs_%s" % an_ch]
        pc_rcs = rs_ds["rcs_%s" % pc_ch]
        an_dc = dc_ds["signal_%s" % an_ch]
        bg_an = rs_ds['bckgrd_signal_%s' % an_ch]
        bg_pc = rs_ds['bckgrd_signal_%s' % pc_ch]
        bg_range = (rs_ds.BCK_MIN_ALT, rs_ds.BCK_MAX_ALT)
        rs_range = (0, max_range)

        # ------ altitude ranges
        idx_bg_range = np.logical_and(h >= bg_range[0], h <= bg_range[1])
        idx_rs_range = np.logical_and(h >= rs_range[0], h <= rs_range[1])

        data['analog'] = an
        data['photoncounting'] = pc
        data['analog_rcs'] = an_rcs
        data['photoncounting_rcs'] = pc_rcs
        data['analog_dc'] = an_dc
        data['background_analog'] = bg_an
        data['background_photoncounting'] = bg_pc
        data['range'] = h
        data['times'] = times
        data['background_range'] = bg_range
        data['signal_range'] = rs_range
        data['indices_background_range'] = idx_bg_range
        data['indices_signal_range'] = idx_rs_range
        #data['signal_dataset'] = rs_ds
        #data['dark_current_dataset'] = dc_ds
        data['analog_channel'] = an_ch
        data['photoncounting_channel'] = pc_ch
        data['wavelength'] = wv
        data['polarization'] = pol
    else:
        print("data not found")
    
    return data