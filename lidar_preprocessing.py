import numpy as np
import pandas as pd
import datetime as dt
import xarray as xr
import scipy as sp
from scipy.signal import savgol_filter
import dask.array as da
import collections
import time
import pdb

""" AD-HOC CORRECTIONS
+ pc peak correction: MULHACEN
"""
def mulhacen_pc_peak_correction(signal):
    """
    Correction of the PC peaks in the PC channels caused by PMT degradation of MULHACEN
    Parameters
    ----------
    signal: array numpy
        lidar signal uncorrected from pc peaks 1D, 2D array (time, range)
    Returns
    -------
    signal_new: array numpy
        lidar signal pc peaks corrected. 2D array (time, range) [numpy.array]
    """

    print("INFO. Start PC Peak Correction")

    # force to 2D
    is_1d = False
    if signal.ndim == 1:
        signal = signal[np.newaxis, :]
        is_1d = True

    # threshold for delta bin:
    threshold = 1000

    new_signal = signal.copy()
    for i in range(signal.shape[0]):
        try:
            profile = np.squeeze(signal[i, :])
            # Obsolete
            #if (profile > 1000).any():
            #    indexes = np.arange(profile.size)
            #    idx_diff = indexes[profile > 200]
            #    for idx_ in idx_diff:
            #        if idx_ < 2:
            #            datacorr = 0.8 * np.median(profile[:10], axis=0)
            #        else:
            #            datacorr = 2 * profile[idx_ - 1] - profile[idx_ - 2]
            #        dif = profile[idx_] - datacorr
            #        profile[idx_] = datacorr
            #        if (idx_ + 1) in idx_diff:
            #            k = 1
            #            while (idx_ + k) in idx_diff:
            #                profile[idx_ + k] = profile[idx_ + k] - dif
            #                k += 1
            #        else:
            #            k = 0
            #        idx_ += k

            diff = np.diff(profile)
            if (diff > threshold).any():
                indexes = np.arange(diff.size)
                idx_diff = indexes[diff>threshold] #With respect to diff array
                idx_profile = idx_diff + 1  #With respect to profile array
                for idx_ in idx_profile:
                    datacorr = np.mean(profile[idx_ - 3 : idx_ - 1])
                    profile[idx_] = datacorr
                    k = 1
                    while (idx_ + k) < profile.size and (profile[idx_ + k] - profile[idx_]) > threshold:
                        datacorr = np.mean(profile[idx_ + k - 3 : idx_ + k - 1])
                        profile[idx_ + k] = datacorr
                        k += 1
            profile[0:3] = 0  # TODO: Preguntar a JABA el beneficio de poner los primeros bines a 0
            new_signal[i, :] = profile
        except Exception as e:
            print("pc peak correction not performed for profile %d-th" % i)
    # if 1d:
    if is_1d:
        new_signal = new_signal.ravel()

    print("INFO. End PC Peak Correction")

    return new_signal


""" DARK SIGNAL
"""
def subtract_dark_current(signal, dc):
    """

    """
    # same length of range dimension
    do = False
    # DC must be numpy
    dc = np.float64(dc)
    """
    if isinstance(dc, int):
        dc = np.int64(dc)
    elif isinstance(dc, float):
        dc = np.float64(dc)
    """

    if not np.isnan(dc).all():
        if signal.ndim == 1:
            if dc.ndim == 1:
                if len(signal) == len(dc):
                    do = True
            elif dc.ndim == 0:
                do = True
            else:
                print("ERROR. Wrong dimensions")
        elif signal.ndim == 2:
            if dc.ndim == 2:
                if signal.shape == dc.shape:
                    do = True
            elif dc.ndim == 1:
                if signal.shape[1] == len(dc):
                    do = True
            elif dc.ndim == 0:
                do = True
            else:
                print("ERROR. Wrong dimensions")
        elif signal.ndim == 0:
            if dc.ndim == 0:
                do = True
        else:
            print("ERROR. Wrong dimensions")

    if do:
        signal = signal - dc
    else:
        print("WARNING. DC not subtracted")

    return signal


""" BACKGROUND
"""
def estimate_background(rs, idx_min, idx_max, dc=None):
    """
    Background Signal is estimated by considering values in the very far range,

    Parameters
    ----------
    rs: numpy.array, xarray.Dataarray 
        Signal: 1D, 2D array (time, range)
    idx_min: Index of Min Height for Background. int
    idx_max: Index of Max Height for Background. int
    dc: numpy.array, xarray.Dataarray 
        DC Signal. 1D array (range)

    Returns
    -------
    bg: Background value. 1D array (time)
    """
    
    print("Start Estimate Background")
    if rs.ndim == 1:
        rs = rs[np.newaxis, :]
    try:
        # Method: Average RS values over BG height range for each profile
        if dc is None:
            bg = np.nanmean(rs[:, idx_min:idx_max+1], axis=1)
        else:
            bg = np.nanmean(rs[:, idx_min:idx_max+1] - dc[idx_min:idx_max+1], axis=1)
    except Exception as e:
        bg = None
        print(str(e))
        print("background cannot be estimated")
    print("End Estimate Background")

    return bg


def subtract_background(signal, bg):
    """ Subtract Background signal to Raw signal

    Parameters
    ----------
    signal : array
        raw signal
    bg : float
        background value

    Returns
    -------
    [type]
        [description]
    """

    do = False

    # BG must be numpy.float64 (it has ndim)
    if isinstance(bg, int):
        bg = np.int64(bg)
    elif isinstance(bg, float):
        bg = np.float64(bg)

    if signal.ndim == 1:
        if bg.ndim == 0:
            do = True
        else:
            print("ERROR. Wrong dimensions")
    elif signal.ndim == 2:
        if bg.ndim == 1:
            if signal.shape[0] == len(bg):
                do = True
        elif bg.ndim == 0:
            do = True
        else:
            print("ERROR. Wrong dimensions")
    elif signal.ndim == 0:
        if bg.ndim == 0:
            do = True
    else:
        print("ERROR. Wrong dimensions")

    if do:
        signal = (signal.T - bg).T
    else:
        print("ERROR. BG not subtracted")

    return signal


def apply_bin_zero_correction(sg, delay):
    """
    TODO: desacoplar dependencia con tipo de variable (numpy/dask array)

    Parameters
    ----------
    sg: array
        signal (range), (time, range)
    delay: float
        position of bin zero (>0, <0)

    Returns
    -------
    sg_c: array
        signal corrected from bin zero

    """

    if not isinstance(delay, int):
        delay = int(delay)

    if isinstance(sg, np.ndarray):
        is_1d = False
        if sg.ndim == 1:
            sg = sg[np.newaxis, :]
            is_1d = True

        sg_c = sg.copy()
        try:
            if delay > 0:
                sg_c[:, :-delay] = sg[:, delay:]
                sg_c[:, -delay:] = np.nan
            elif delay < 0:
                sg_c[:, -delay:] = sg[:, :delay]
                sg_c[:, :-delay] = np.nan
        except Exception as e:
            print("ERROR. In apply_bin_zero_correction. %s" % str(e))
        if is_1d:
            sg_c = sg_c.ravel()
    else:
        if delay > 0:
            sg_c = da.concatenate([sg[:, delay:], da.zeros((sg.shape[0], delay))*np.nan], axis=1)
        elif delay < 0:
            sg_c = da.concatenate([da.zeros((sg.shape[0], abs(delay)))*np.nan, sg[:, :delay]], axis=1)
        else:
            sg_c = sg

    return sg_c


""" DEAD TIME
"""
def apply_dead_time_correction(pc, tau, system=0):
    """
    Application of DEAD TIME correction over PC signal

    Parameters
    ----------
    pc: array
        photoncounting signal in MHz
    tau: float
        dead time in ns
    system: int
        paralyzable (1), nonparalyzable (0)

    Returns
    -------
    c: array
        pc signal corrected from dead time. 1D (2D) array (range (time, range))

    """

    try:
        # tau from ns to us
        tau_us = tau * 1e-3
        # tau_us = 
        if system == 0:  # NON-PARALYZABLE
            # Eq 4 [D'Amico et al., 2016]
            c = pc / (1 - pc * tau_us)
        elif system == 1:  # PARALYZABLE
            # To be derived from Eq (2) [D'Amico et al., 2016]. Non-analytic
            c = pc
            print("WARNING: PARALYZABLE NOT IMPLEMENTED. No correction is applied")
        else:
            c = pc
            print("WARNING: wrong system for dead time correction. None is applied")
        # No infinites nor negative values
        c = np.where(np.logical_or(np.isinf(c), c < 0), np.nan, c)

    except Exception as e:
        print("ERROR. In apply_dead_time_correction %s" % str(e))
        c = pc * np.nan

    return c


""" PREPROCESSING SIGNAL
"""
def average_dc_signal(dc):
    """
    Provided a set of DC profiles, a time averaged and vertical smoothed DC profile
    is calculated, which might be broadcasted in a given new time dimension

    Parameters
    ----------
    dc: array (1d, 2d)
        Dark Current (time (dc), range)
    Returns
    -------
    dc_pp: array
        Dark Current Smoothed Profile. 1D (range) or 2D array (time, range)
    """

    print("Start DC Averaging")
    try:
        # force to 2D
        if dc.ndim == 1:  # (range) -> (1, dim_range)
            dc = dc[np.newaxis, :]

        # average dc over time
        dc_avg_t = np.nanmean(dc, axis=0)

        # vertical smoothing of time-averaged profile
        weights = np.array([1, 2, 4, 6, 8, 6, 4, 2, 1], dtype=np.float)
        weights /= np.sum(weights)
        dc_pp = sp.ndimage.filters.convolve(dc_avg_t, weights, mode='constant')

    except Exception as e:
        dc_pp = np.zeros(dc.size)
        print(str(e))
        print("ERROR: in preprocessing dc. DC set to 0")

    print("End DC Averaging")
    return dc_pp 


def preprocessing_analog_signal(signal, dc, bz, idx_min, idx_max, \
    dc_flag=True, zerobin_flag=True, workflow=0):
    """

    Parameters
    ----------
    signal: Raw Signal. Measured. 1D, 2D array (time (rs), range)
    dc: Dark Current Signal. Measured. 1D, 2D array (time (dc), range)
    bz: Bin Zero. scalar.
    bg: Background. Pre processed. 1D array (time (rs))
    zerobin_flag: activate/desactivate zero-bin correction. bool
    workflow: type of workflow. Scalar. 0: SCC; 1: BG before ZB

    Returns
    -------
    signal: Preprocessed Signal. 2D array (time (rs), range)

    """
    
    print("Start Analog Preprocessing")
    try:
        # force to 2D
        is_1d = False
        if signal.ndim == 1:
            signal = signal[np.newaxis, :]
            is_1d = True

        # Type of workflow
        if np.logical_or(workflow < 0, workflow > 1):
            workflow = 0  # SCC

        # Workflow: ZERO_BIN[(RAW - DC)] -  BG
        # 0. Estimate Background
        bg = estimate_background(signal, idx_min, idx_max, dc=dc)

        # 1. Subtract DC from AN and BG
        if dc_flag:
            signal = subtract_dark_current(signal, dc)
        if workflow == 0:  # SCC
            if zerobin_flag:
                # 2. Apply Trigger Delay
                signal = apply_bin_zero_correction(signal, bz)
            # 3. Subtract Background (which has been subtracted from DC)
            signal = subtract_background(signal, bg)
        else:  # Indistinguible de SCC
            # 2. Subtract Background (which has been subtracted from DC)
            signal = subtract_background(signal, bg)
            if zerobin_flag:
                # 3. Apply Trigger Delay
                signal = apply_bin_zero_correction(signal, bz)

    except Exception as e:
        print(str(e))
        print("Signal not pre-processed.")

    if is_1d:
        signal = signal.ravel()

    print("End Analog Preprocessing")
    return signal


def preprocessing_photoncounting_signal(signal, tau, bz, idx_min, idx_max, \
    deadtime_flag=True, zerobin_flag=True, workflow=0):
    """

    Parameters
    ----------
    signal: Raw Signal. 2D array (time (rs), range)
    tau: Dead Time (ns). Scalar
    bz: zero bin for photoncounting (delay_an + bin_shift). Scalar.
    bg: Background. 1D array (time (rs))
    peak_correction: Apply peak correction. Bool
    workflow: type of workflow. Scalar. 0: SCC; 1: BG before ZB

    Returns
    -------
    signal: Preprocessed Signal. 2D array (time (rs), range)
    """
    # pdb.set_trace()
    try:
        # Force to 2D
        is_1d = False
        if signal.ndim == 1:
            signal = signal[np.newaxis, :]
            is_1d = True

        # Type of workflow
        if np.logical_or(workflow < 0, workflow > 1):
            workflow = 0  # SCC

        # Workflow SCC: ZERO_BIN[DT(PK(RAW))] -  BG
        # 0. Estimate Background
        bg = estimate_background(signal, idx_min, idx_max)

        if deadtime_flag:
            # Dead Time Correction
            # 1. Apply Dead Time Correction
            # pass
            signal = apply_dead_time_correction(signal, tau)

        if workflow == 0:  # SCC
            if zerobin_flag:
                # pass
                # 2. Apply Trigger Delay  No delay for photoncounting channel 
                signal = apply_bin_zero_correction(signal, bz)
            # 3. Subtract Background as estimated from Raw Signal
            signal = subtract_background(signal, bg)
        else:
            # 2. Subtract Background as estimated from Raw Signal
            signal = subtract_background(signal, bg)
            if zerobin_flag:
                # 3. Apply Trigger Delay
                signal = apply_bin_zero_correction(signal, bz)

    except Exception as e:
        print(str(e))
        print("signal not pre-processed.")

    if is_1d:
        signal = signal.ravel()

    return signal