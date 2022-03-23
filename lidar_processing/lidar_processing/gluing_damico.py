"""
Gluing procedure as implemented in the SCC, following D'Amico et al. 2016.

D'Amico, G., Amodeo, A., Mattis, I., Freudenthaler, V., and Pappalardo, G.: EARLINET Single Calculus Chain -
technical - Part 1: Pre-processing of raw lidar data, Atmos. Meas. Tech., 9, 491-507, 
https://doi.org/10.5194/amt-9-491-2016, 2016.
"""

import numpy as np
from scipy.signal import savgol_filter

from .pre_processing import calculate_gluing_values, glue_signals_at_bins


def glue_analog_photon(analog_signal, photon_signal, z, photon_background, adc_range, adc_bits, n_res,
                       photon_threshold=20.,
                       correlation_threshold=0.8, slope_threshold=2, stability_threshold=1, step=5,
                       min_initial_points=15, min_points=5, full_overlap=5000, use_photon_as_reference=True):
    """ Glue two signals using the procedure described in D'Amico et al. 2016. """

    analog_signal_w_rc = analog_signal / z ** 2
    photon_signal_w_rc = photon_signal / z ** 2 + photon_background

    first_lower_idx, first_upper_idx = estimate_first_range(analog_signal_w_rc, photon_signal_w_rc, z, photon_threshold,
                                                            adc_range, adc_bits, n_res, correlation_threshold,
                                                            min_initial_points, step, full_overlap)

    # print("First gluing region: %s - %s" % (first_lower_idx, first_upper_idx))

    slope_lower_idx, slope_upper_idx = optimize_with_slope_test(analog_signal, photon_signal, first_lower_idx,
                                                                first_upper_idx, slope_threshold, step,
                                                                use_photon_as_reference)

    # print("Slope test finished: %s - %s" % (slope_lower_idx, slope_upper_idx))

    lower_idx, upper_idx = optimize_with_stability_test(analog_signal, photon_signal, slope_lower_idx,
                                                        slope_upper_idx, stability_threshold, step,
                                                        use_photon_as_reference)

    c_analog, c_photon = calculate_gluing_values(analog_signal[lower_idx:upper_idx],
                                                 photon_signal[lower_idx:upper_idx], use_photon_as_reference)

    glued_signal, _, _ = glue_signals_at_bins(analog_signal, photon_signal, lower_idx, upper_idx, c_analog, c_photon)

    all_indices = [first_lower_idx, first_upper_idx, slope_lower_idx, slope_upper_idx]

    return glued_signal, lower_idx, upper_idx, c_analog, c_photon, all_indices


def estimate_first_range(analog_signal, photon_signal, z, photon_threshold, adc_range, adc_bits, n_res,
                         correlation_threshold, min_points, step, full_overlap):
    """
    Return the lower and upper altitudes (if they are exist) as the first estimation of the gluing range of
    lidar signals.

    Parameters
    ----------
    analog_background_and_bin_shift_corr: (N, ) array
        Signal from analog channel which should be had dark noise, background and trigger delay corrections.
    photon_dead_time_corrected_signal: (N, ) array
        Signal from photon counting channel, dead time corrected and convert to MHz.
        Be careful is not background corrected.
    adc_range: float [mVolt]
        Full scale voltage (for calculation of the Least Significant bit).
    adc_bits: integer
        Number of bits (for calculation of the Least Significant bit).
    n_res: integer or float
        Number of times above the LSB that we trust the analog signal. c.f. equation (12) of D'Amico et al.
    full_overlap : float
        Full overlap range. Used to limit the search region of the maximum signal.

    Returns
    -------
    lower_gluing_range: float
            The lower gluing altitude, not optimal byt from the first estimation.
    upper_gluing_range: float
            The higher gluing altitude, not optimal byt from the first estimation.

    References
    ----------
    D'Amico, G., Amodeo, A., Mattis, I., Freudenthaler, V., and Pappalardo, G.: EARLINET Single Calculus Chain –
    technical – Part 1: Pre-processing of raw lidar data, Atmos. Meas. Tech., 9, 491–507,
    https://doi.org/10.5194/amt-9-491-2016, 2016.
    """
    analog_threshold = n_res * adc_range / ((2 ** adc_bits) - 1)

    min_photon = np.min(photon_signal)
    if min_photon > photon_threshold:
        raise ValueError('Wrong initials guess. Photon signal above threshold value: {0}. The minimum value of photon '
                         'counting signal was: {1}'.format(photon_threshold, min_photon))

    idx_full_overlap = _idx_at_range(z, full_overlap)

    smoothed_signal = savgol_filter(photon_signal, 21, polyorder=2)  # Smooth to avoid selecting noise as peak

    idx_peak = np.argmax(smoothed_signal[:idx_full_overlap])  # Search up to the altitude of full overlap

    lower_idx = idx_peak + np.argmax(photon_signal[idx_peak:] < photon_threshold)

    upper_idx = idx_peak + np.argmax(analog_signal[idx_peak:] < analog_threshold)

    if (upper_idx - lower_idx) < min_points:
        raise TooFewPointsError(
            "No suitable region found. Lower gluing idx: {0}. Upper gluing idx: {1}.".format(lower_idx, upper_idx))

    correlation = np.corrcoef(analog_signal[lower_idx:upper_idx], photon_signal[lower_idx:upper_idx])[0, 1]

    if correlation < correlation_threshold:
        raise TestFailedError(
            "Correlation test not passed. Correlation: {0}. Threshold {1}".format(correlation, correlation_threshold))

    return lower_idx, upper_idx


def optimize_with_slope_test(analog_signal, photon_signal, low_idx_start, up_idx_start, m_stddevs, step,
                             use_photon_as_reference):
    low_idx, up_idx = low_idx_start, up_idx_start

    glue_found = False
    first_round = True

    N = up_idx - low_idx

    while not glue_found and (N > 5):

        analog_segment = analog_signal[low_idx:up_idx]
        photon_segment = photon_signal[low_idx:up_idx]

        if N <= 30:
            k, dk = _calculate_residual_slope(analog_segment, photon_segment, use_photon_as_reference)

            # print("Slope iteration: %s - %s. N < 30. K: %s, DK: %s" % (low_idx, up_idx, k, dk))

            if np.abs(k) < m_stddevs * dk:
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
        else:
            mid_idx = (up_idx - low_idx) // 2

            k1, dk1 = _calculate_residual_slope(analog_segment[:mid_idx], photon_segment[:mid_idx],
                                                use_photon_as_reference)
            k2, dk2 = _calculate_residual_slope(analog_segment[mid_idx:], photon_segment[mid_idx:],
                                                use_photon_as_reference)

            # print("Slope iteration: %s - %s. N > 30. K1: %s, DK1: %s, K2: %s, DK2: %s" % (low_idx, up_idx, k1, dk1, k2, dk2))

            if np.abs(k2 - k1) < m_stddevs * np.sqrt(dk1 ** 2 + dk2 ** 2):
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
        raise TooFewPointsError(
            "No suitable region found. Lower gluing idx: {0}. Upper gluing idx: {1}.".format(low_idx, up_idx))

    return low_idx, up_idx


def optimize_with_stability_test(analog_signal, photon_signal, low_idx_start, up_idx_start, m_stddevs,
                                 step, use_photon_as_reference):
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

        if np.abs(k2 - k1) < m_stddevs * np.sqrt(dk1 ** 2 + dk2 ** 2):
            glue_found = True
        else:
            # update
            low_idx += step
            up_idx -= step
            N = up_idx - low_idx

    if not glue_found:
        raise TooFewPointsError(
            "No suitable region found. Lower gluing idx: {0}. Upper gluing idx: {1}.".format(low_idx, up_idx))

    return low_idx, up_idx


def _calculate_residual_slope(analog_segment, photon_segment, use_photon_as_reference):
    c_analog, c_photon = calculate_gluing_values(analog_segment, photon_segment, use_photon_as_reference)

    residuals = c_analog * analog_segment - c_photon * photon_segment

    fit_values, cov = np.polyfit(np.arange(len(residuals)), residuals, 1, cov=True)

    k = fit_values[0]  # Slope
    dk = np.sqrt(np.diag(cov)[0])  # Check here: https://stackoverflow.com/a/27293227

    return k, dk


def _calculate_slope(analog_segment, photon_segment, use_photon_as_reference):
    if use_photon_as_reference:
        fit_values, cov = np.polyfit(analog_segment, photon_segment, 1, cov=True)
    else:
        fit_values, cov = np.polyfit(photon_segment, analog_segment, 1, cov=True)

    k = fit_values[0]  # Slope
    dk = np.sqrt(np.diag(cov)[0])  # Check here: https://stackoverflow.com/a/27293227

    return k, dk


def _idx_at_range(z, z_ref):
    """ Find the index closest to the specified range. """
    return np.argmin(np.abs(z - z_ref))


class TestFailedError(RuntimeError):
    pass


class TooFewPointsError(RuntimeError):
    pass
