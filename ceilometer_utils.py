import sys
import numpy as np
import logging


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


""" Module for Ceilometer Operations

"""

#def estimate_background(signal, ranges, r_min=12000, r_max=15000):
#    """
#    estimate_background [summary]
#
#    Heese, B., Flentje, H., Althausen, D., Ansmann, A. and Frey, S.: Ceilometer lidar comparison: Backscatter coefficient retrieval and signal-to-noise ratio determination, Atmos. Meas. Tech., 3(6), 1763–1770, doi:10.5194/amt-3-1763-2010, 2010.
#
#    Parameters
#    ----------
#    signal : [type]
#        [description]
#    ranges: [type]
#        [description]
#    r_min : int, optional
#        [description], by default 12000
#    r_max : int, optional
#        [description], by default 15000
#
#    Returns
#    -------
#    bg: [type]
#        [description]
#    bg_error
#    """
#
#    # Signal must have dimensions: (time, height)
#    if signal.ndim == 1:
#        signal = signal[np.newaxis, :]
#
#    idr = np.logical_and(ranges >= r_min, ranges <= r_max)
#
#    bg_prf = signal[:, idr]
#    bg = abs(np.nanmean(bg_prf, axis=1))
#    bg_error = np.nanstd(bg_prf, axis=1) / np.sqrt(bg_prf.shape[1])
#
#    return bg, bg_error


#def subtract_background(signal, bg):
#    """[summary]
#
#    Parameters
#    ----------
#    signal : [type]
#        [description]
#    bg : [type]
#        [description]
#    """
#    error_msg = "Background Not Subtracted"
#    try:
#        if np.logical_or(signal.ndim >=1, signal.ndim <= 2):
#            # BG must be numpy.float64 (it has ndim)
#            bg = np.float64(bg)
#            if bg.ndim == 0:
#                p = signal - bg
#            elif bg.ndim == 1:
#                if signal.ndim == 2:
#                    p = (signal.T - bg).T
#                else:
#                    logger.error(error_msg)
#        else:
#            logger.error(error_msg)
#    except Exception as e:
#        logger.error(str(e))
#        logger.error(error_msg)
#        p = signal
#    return p
#
#def signal_to_noise_ratio(signal, background):
#    """
#    signal_to_noise_ratio [summary]
#
#    [extended_summary]
#    Heese, B., Flentje, H., Althausen, D., Ansmann, A. and Frey, S.: Ceilometer lidar comparison: Backscatter coefficient retrieval and signal-to-noise ratio determination, Atmos. Meas. Tech., 3(6), 1763–1770, doi:10.5194/amt-3-1763-2010, 2010.
#
#    Parameters
#    ----------
#    signal : [type]
#        [description]
#    background : [type]
#        [description]
#    """
#    if signal.ndim == 2:
#        background = background[:, np.newaxis]
#    return signal / np.sqrt(signal + 2*background)


def preprocess(beta_raw, ranges, cal_factor, r_min=None, r_max=None):
    """
    preprocess [summary]

    [extended_summary]

    Parameters
    ----------
    beta_raw : [type]
        Normalized Range Corrected Signal
    ranges : [type]
        [description]
    cal_factor : [type]
        [description]

    """

    # Ranges for Estimate Background
    if r_min is None:
        r_min = 12000
    if r_max is None:
        r_max = 15000

    # Attenuated Backscatter
    beta_att = beta_raw / cal_factor

    """
    # Raw Signal
    raw_signal = beta_raw / (ranges**2)

    # Background
    bg, bg_err = estimate_background(raw_signal, ranges, r_min=r_min, r_max=r_max)

    # Signal
    signal = subtract_background(raw_signal, bg)

    # Range Corrected Signal
    rcs = signal * (ranges**2)
    """

    return beta_att #, rcs, signal, bg, bg_err, raw_signal

