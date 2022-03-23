import os
import sys
import numpy as np
from scipy.signal import savgol_filter
import pdb

MODULE_DIR = os.path.dirname(sys.modules[__name__].__file__)
sys.path.insert(0, MODULE_DIR)
import utils
from lidar_linear_response_region import estimate_linear_response_region


""" GLUING

D'Amico, G., Amodeo, A., Mattis, I., Freudenthaler, V., and Pappalardo, G.: EARLINET Single Calculus Chain -
technical - Part 1: Pre-processing of raw lidar data, Atmos. Meas. Tech., 9, 491-507, 
https://doi.org/10.5194/amt-9-491-2016, 2016.
"""


def _estimate_first_range(analog_signal, photon_signal, z, photon_threshold,
                          adc_range, adc_bits, n_res, correlation_threshold,
                          min_points, full_overlap_range):
    """

    :param analog_signal:
    :param photon_signal:
    :param z:
    :param photon_threshold:
    :param adc_range:
    :param adc_bits:
    :param n_res:
    :param correlation_threshold:
    :param min_points:
    :param full_overlap_range:
    :return:
    """

    # estimation is performed if a threshold for pc is achieved
    pc_min = np.nanmin(photon_signal)
    if pc_min < photon_threshold:
        # smooth pc signal to avoid selecting noise as peak
        idx_finite = np.isfinite(photon_signal)
        smoothed_pc_signal = photon_signal*0.0
        smoothed_pc_signal[idx_finite] = savgol_filter(photon_signal[idx_finite], 21, polyorder=2)

        # estimate of height of full overlap begins

        # with PC, lower height is estimated
        idx_fo_up, z_fo_up = utils.find_nearest_1d(z, full_overlap_range[1])
        idx_fo_bo_1, z_fo_bo = utils.find_nearest_1d(z, full_overlap_range[0])
        idx_fo_bo_2 = np.nanargmax(smoothed_pc_signal[:idx_fo_up])
        if idx_fo_bo_1 > idx_fo_bo_2:
            idx_fo_bo = idx_fo_bo_1
        else:
            idx_fo_bo = idx_fo_bo_2
        lower_idx = idx_fo_bo + np.nanargmax(photon_signal[idx_fo_bo:] < photon_threshold)

        # with AN, upper height is estimated
        analog_threshold = n_res * adc_range / ((2 ** adc_bits) - 1)
        upper_idx = idx_fo_bo + np.nanargmax(analog_signal[idx_fo_bo:] < analog_threshold)

        if (upper_idx - lower_idx) < min_points:
            print("No suitable region found. Lower gluing idx: {0}. Upper gluing idx: {1}."
                  .format(lower_idx, upper_idx))
            lower_idx, upper_idx = False, False

        correlation = np.corrcoef(analog_signal[lower_idx:upper_idx],
                                  photon_signal[lower_idx:upper_idx])[0, 1]

        if correlation < correlation_threshold:
            print("Correlation test not passed. Correlation: {0}. Threshold {1}"
                  .format(correlation, correlation_threshold))
            lower_idx, upper_idx = False, False

    else:  # wrong initial guess for photon_threshold
        print("wrong initial guess for photon_threshold")
        lower_idx, upper_idx = False, False

    return lower_idx, upper_idx


def _calculate_gluing_values(lower_gluing_region, upper_gluing_region, use_upper_as_reference):
    """
    Calculate the multiplicative calibration constants for gluing the two signals.

    Parameters
    ----------
    lower_gluing_region: array
       The low-range signal to be used. Can be either 1D or 2D with dimensions (time, range).
    upper_gluing_region: array
       The high-range signal to be used. Can be either 1D or 2D with dimensions (time, range).
    use_upper_as_reference: bool
       If True, the upper signal is used as reference. Else, the lower signal is used.

    Returns
    -------
    c_lower: float
       Calibration constant of the lower signal. It will be equal to 1, if `use_upper_as_reference` argument
       is False.
    c_upper: float
       Calibration constant of the upper signal. It will be equal to 1, if `use_upper_as_reference` argument
       is True.
    """
    lower_gluing_region = lower_gluing_region.ravel()  # Ensure we have an 1D array using ravel
    upper_gluing_region = upper_gluing_region.ravel()

    # Find their linear relationship, using least squares
    slope_zero_intercept, _, _, _ = np.linalg.lstsq(lower_gluing_region[:, np.newaxis], upper_gluing_region)

    # Set the calibration constants
    if use_upper_as_reference:
        c_upper = 1
        c_lower = slope_zero_intercept
    else:
        c_upper = 1 / slope_zero_intercept
        c_lower = 1

    return c_lower, c_upper


def _calculate_residual_slope(analog_segment, photon_segment, use_photon_as_reference):
    """

    """
    c_analog, c_photon = _calculate_gluing_values(analog_segment, photon_segment, use_photon_as_reference)

    residuals = c_analog * analog_segment - c_photon * photon_segment

    fit_values, cov = np.polyfit(np.arange(len(residuals)), residuals, 1, cov=True)

    k = fit_values[0]  # Slope
    dk = np.sqrt(np.diag(cov)[0])  # Check here: https://stackoverflow.com/a/27293227

    return k, dk


def _calculate_slope(analog_segment, photon_segment, use_photon_as_reference):
    """

    """
    if use_photon_as_reference:
        fit_values, cov = np.polyfit(analog_segment, photon_segment, 1, cov=True)
    else:
        fit_values, cov = np.polyfit(photon_segment, analog_segment, 1, cov=True)

    k = fit_values[0]  # Slope
    dk = np.sqrt(np.diag(cov)[0])  # Check here: https://stackoverflow.com/a/27293227

    return k, dk


def _optimize_with_slope_test(analog_signal, photon_signal, low_idx_start,
                              up_idx_start, slope_threshold, step,
                              use_photon_as_reference):
    """

    """
    low_idx, up_idx = low_idx_start, up_idx_start

    glue_found = False
    first_round = True

    N = up_idx - low_idx

    while not glue_found and (N > 5):

        analog_segment = analog_signal[low_idx:up_idx]
        photon_segment = photon_signal[low_idx:up_idx]

        if N <= 30:
            k, dk = _calculate_residual_slope(analog_segment, photon_segment, use_photon_as_reference)

            if np.abs(k) < slope_threshold * dk:
                glue_found = True
            else:
                # print("Changing indices")
                # update indices for next loop
                if first_round:
                    up_idx -= step
                else:
                    low_idx += step

                N = up_idx - low_idx

                # If first round finished without result, start the second round, increasing the lower bound.
                if (N <= 5) and first_round:
                    first_round = False
                    low_idx, up_idx = low_idx_start, up_idx_start  # Start searching from the start
                    N = up_idx - low_idx
        else:
            mid_idx = (up_idx - low_idx) // 2

            k1, dk1 = _calculate_residual_slope(analog_segment[:mid_idx], photon_segment[:mid_idx],
                                                use_photon_as_reference)
            k2, dk2 = _calculate_residual_slope(analog_segment[mid_idx:], photon_segment[mid_idx:],
                                                use_photon_as_reference)

            # print("Slope iteration: %s - %s. N > 30. K1: %s, DK1: %s, K2: %s, DK2: %s" % (low_idx, up_idx, k1, dk1, k2, dk2))

            if np.abs(k2 - k1) < slope_threshold * np.sqrt(dk1 ** 2 + dk2 ** 2):
                # print("Slope test passed!!!!")
                glue_found = True
            else:
                # print("Changing indices")
                # update indices for next loop
                if first_round:
                    up_idx -= step
                else:
                    low_idx += step

                N = up_idx - low_idx

                # If first round finished without result, start the second round, increasing the lower bound.
                if (N <= 5) and first_round:
                    first_round = False
                    low_idx, up_idx = low_idx_start, up_idx_start  # Start searching from the start
                    N = up_idx - low_idx

    if not glue_found:
        low_idx, up_idx = False, False
        print("No suitable region found. Lower gluing idx: {0}. Upper gluing idx: {1}.".format(low_idx, up_idx))

    return low_idx, up_idx


def _optimize_with_stability_test(analog_signal, photon_signal, low_idx_start, up_idx_start,
                                  stability_threshold, step, use_photon_as_reference):
    """

    """

    # print('entering stability test')

    low_idx, up_idx = low_idx_start, up_idx_start

    glue_found = False
    N = up_idx - low_idx

    while not glue_found and (N > 5):

        analog_segment = analog_signal[low_idx:up_idx]
        photon_segment = photon_signal[low_idx:up_idx]

        mid_idx = (up_idx - low_idx) // 2

        k1, dk1 = _calculate_slope(analog_segment[:mid_idx], photon_segment[:mid_idx], use_photon_as_reference)
        k2, dk2 = _calculate_slope(analog_segment[mid_idx:], photon_segment[mid_idx:], use_photon_as_reference)

        # print("Stability iteration:  K1: %s, DK1: %s, K2: %s, DK2: %s" % (k1, dk1, k2, dk2))

        if np.abs(k2 - k1) < stability_threshold * np.sqrt(dk1 ** 2 + dk2 ** 2):
            glue_found = True
        else:
            # update
            low_idx += step
            up_idx -= step
            N = up_idx - low_idx

    if not glue_found:
        low_idx, up_idx = False, False
        print("No suitable region found. Lower gluing idx: {0}. Upper gluing idx: {1}.".format(low_idx, up_idx))

    return low_idx, up_idx


def _glue_signals_at_bins(lower_signal, upper_signal, min_bin, max_bin, c_lower, c_upper):
    """
    Glue two signals at a given bin range.

    The signal can be either a 1D array or a 2D array with dimensions (time, range).

    Both signals are assumed to have the same altitude grid. The final glued signal is calculated
    performing a linear fade-in/fade-out operation in the glue region.

    Parameters
    ----------
    lower_signal: array
       The low-range signal to be used. Can be either 1D or 2D with dimensions (time, range).
    upper_signal: array
       The high-range signal to be used. Can be either 1D or 2D with dimensions (time, range).
    min_bin: int
       The lower bin to perform the gluing
    max_bin: int
       The upper bin to perform the gluing
    c_lower: float
       Calibration constant of the lower signal. It will be equal to 1, if `use_upper_as_reference` argument
       is False.
    c_upper: float
       Calibration constant of the upper signal. It will be equal to 1, if `use_upper_as_reference` argument
       is True.

    Returns
    -------
    glued_signal: array
       The glued signal array, same size as lower_signal and upper_signal.
    """
    # Ensure that data are 2D-like
    if lower_signal.ndim == 1:
        lower_signal = lower_signal[np.newaxis, :]  # Force 2D
        upper_signal = upper_signal[np.newaxis, :]  # Force 2D
        axis_added = True
    else:
        axis_added = False

    gluing_length = max_bin - min_bin

    lower_weights = np.zeros_like(lower_signal)

    lower_weights[:, :min_bin] = 1
    lower_weights[:, min_bin:max_bin] = 1 - np.arange(gluing_length) / float(gluing_length)

    upper_weights = 1 - lower_weights

    # Calculate the glued signal
    glued_signal = c_lower * lower_weights * lower_signal + c_upper * upper_weights * upper_signal

    # Remove dummy axis, if added
    if axis_added:
        glued_signal = glued_signal[0, :]

    return glued_signal, c_lower, c_upper


def gluing(an, pc, ranges, adc_range, adc_bits, n_res=None, pc_threshold=None,
           correlation_threshold=None, range_threshold=None, min_points=None,
           slope_threshold=2, stability_threshold=1, step=5,
           use_photon_as_reference=True):
    """

    :param an: array 1d
    :param pc: array 1d
    :param ranges: array 1d
    :param adc_range:
    :param adc_bits:
    :param n_res:
    :param pc_threshold:
    :param correlation_threshold:
    :param range_threshold:
    :param min_points:
    :return:
    """

    # Defaults
    if n_res is None:
        F = 5000
        n_res = (2**adc_bits - 1)/F  # D'Amico et al., 2016, Eq.12
    if pc_threshold is None:
        pc_threshold = 20
    if correlation_threshold is None:
        correlation_threshold = 0.8
    if range_threshold is None:
        range_threshold = (1000, 6000)
    if min_points is None:
        min_points = 15

    try:
        # First Estimation of Gluing Range
        first_lower_idx, first_upper_idx = _estimate_first_range(an, pc, ranges, pc_threshold,
                                                                 adc_range, adc_bits, n_res,
                                                                 correlation_threshold, min_points,
                                                                 range_threshold)
        if np.logical_and(first_lower_idx, first_upper_idx):
            # Slope Test
            slope_lower_idx, slope_upper_idx = _optimize_with_slope_test(an, pc, first_lower_idx,
                                                                         first_upper_idx,
                                                                         slope_threshold, step,
                                                                         use_photon_as_reference)
            if np.logical_and(slope_lower_idx, slope_upper_idx):
                # Stability Test
                lower_idx, upper_idx = _optimize_with_stability_test(an, pc, slope_lower_idx,
                                                                     slope_upper_idx, stability_threshold, step,
                                                                     use_photon_as_reference)
                if np.logical_and(lower_idx, upper_idx):
                    # Calculate Gluing Values
                    c_analog, c_photon = _calculate_gluing_values(an[lower_idx:upper_idx],
                                                                  pc[lower_idx:upper_idx], use_photon_as_reference)
                    glued = True
                    glued_signal, _, _ = _glue_signals_at_bins(an, pc, lower_idx, upper_idx, c_analog, c_photon)

                    all_indices = [first_lower_idx, first_upper_idx, slope_lower_idx, slope_upper_idx,
                                   lower_idx, upper_idx]
                else:
                    glued = False
            else:
                glued = False
        else:
            glued = False
    except Exception as e:
        glued = False
        print("ERROR. Signal not glued. %s" % str(e))
    if not glued:
        glued_signal, c_analog, c_photon, all_indices = \
            ranges*np.nan, np.nan, np.nan, []

    return glued_signal, c_analog, c_photon, all_indices, glued


def gluing_dbp(analog, photon, ranges, analog_bg, analog_bg_threshold=0.02, pc_threshold=15,
               region_range=(1500, 10000), min_region=1000, subregion=200, n_step=3,
               correlation_threshold=0.85):
    """
    analog: 1d
    photon: 1d
    ranges: 1d
    analog_bg: float
    """
    # estimate linear region
    h_lr_min, h_lr_max, r2, slope, intercept, linear_region = estimate_linear_response_region(
        ranges, analog, photon, analog_bg, an_bg_thres=analog_bg_threshold,
        pc_thres=pc_threshold, region_range=region_range, min_region=min_region,
        subregion=subregion, n_step=n_step, correlation_thres=correlation_threshold)
    if linear_region:
        # indices
        lower_idx = int(np.argwhere(ranges == h_lr_min))
        upper_idx = int(np.argwhere(ranges == h_lr_max))
        indices = [lower_idx, upper_idx]

        # Calculate Gluing Values
        c_analog, c_photon = _calculate_gluing_values(analog[lower_idx:upper_idx],
                                                      photon[lower_idx:upper_idx],
                                                      use_upper_as_reference=True)

        glued_signal, _, _ = _glue_signals_at_bins(analog, photon, lower_idx, upper_idx,
                                                   c_analog, c_photon)

        glued = True
    else:
        glued_signal, c_analog, c_photon, indices, glued = \
            ranges*np.nan, np.nan, np.nan, [], False
        print("ERROR. Signal not glued")

    return glued_signal, c_analog, c_photon, indices, glued

