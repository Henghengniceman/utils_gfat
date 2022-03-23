"""
Function for pre-processing lidar signals.

.. todo::
   Decide how to handle error propagation. Current plan is to estimate errors using end-to-end Monte Carlo
   simulations (not implemented yet), instead of  calculating errors in every step.
"""
import numpy as np
from scipy.optimize import fmin_l_bfgs_b

from . import fit_checks
from . import constants


def apply_range_correction(signal, altitude):
    """
    Apply range correction, i.e. multiply with z^2.

    The signal can be either a 1D array or a 2D array with dimensions (time, range).
    The input signal should have already been corrected for background offset.

    Parameters
    ----------
    signal: array
       The input signal array
    altitude: array
       The altitude array. It should be the same size as the signal.

    Returns
    -------
    rcs: array
       The range corrected signal.

    """
    rcs = signal * altitude ** 2
    return rcs


def correct_dead_time_nonparalyzable(signal, measurement_interval, dead_time):
    """ Apply non-paralizable dead time correction.
        
    The signal can be either a 1D array or a 2D array with dimensions (time, range).

    Parameters
    ----------
    signal: integer array
       The measured number on photons
    measurement_interval: float
       The total measurement interval in ns
    dead_time: float
       The detector system dead time in ns
    
    Returns
    -------
    corrected_signal: float array
       The true number of photons arriving at the detector    
    """

    if dead_time == 0:
        return signal  # Nothing to correct for 0 dead time.

    # Maximum number of photons are limited by dead time
    max_photons = measurement_interval / dead_time

    if np.any(signal > max_photons):
        raise ValueError(
            "Signal contains photons above the maximum number permitted by the dead time value. Max: {0}. Found: {1}".format(
                max_photons, np.max(signal)))

    corrected_signal = signal / (1 - signal * dead_time / float(measurement_interval))
    return corrected_signal


def correct_dead_time_paralyzable(signal, measurement_interval, dead_time):
    """ Apply paralyzable dead time correction.

    For a paralyzable system, each measurement corresponds to two possible real
    values. The solution offered here corresponds to the first branch [0, measurement_interval / dead_time]

    The signal can be either a 1D array or a 2D array with dimensions (time, range).

    The solution uses a L-BFGS-B algorithm for optimization.

    Parameters
    ----------
    signal: integer array
       The measured number on photons
    measurement_interval: float
       The total measurement interval in ns
    dead_time: float
       The detector system dead time in ns
    
    Returns
    -------
    corrected_signal: float array
       The true number of photons arriving at the detector    

    Notes
    -----
    If you use this routine you should cite one of the papers of the original authors.
    See the `scipy <http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html>`_ website
    for details.
    """

    if dead_time == 0:
        return signal  # Nothing to correct for 0 dead time

    # Maximum number of photons are limited by dead time + probability from Poisson
    max_photons = measurement_interval / dead_time / np.e

    if np.any(signal > max_photons):
        raise ValueError('Signal contains photons above the maximum number permitted by \
                            the dead time value. Max: {0}. Found: {1}'.format(max_photons, np.max(signal)))

    # The above equation has two solutions. We need to constrain the solutions
    # in the first branch [0, measurement_interval / dead_time]
    signal_maximum = measurement_interval / dead_time

    # Get the first estimate by non-paralyzable correction, for faster convergence
    first_estimate = correct_dead_time_nonparalyzable(signal, measurement_interval, dead_time)
    bounds = ((0, signal_maximum),) * len(signal)
    corrected_signal, _, _ = fmin_l_bfgs_b(_paralyzable_objective_function, first_estimate,
                                           args=(signal, measurement_interval, dead_time),
                                           fprime=_paralyzable_jacobian, bounds=bounds)
    return corrected_signal


def _paralyzable_objective_function(corrected_signal, signal, measurement_interval, dead_time):
    """
    Helper function that calculates the objective function of the paralyzable dead-time correction case.

    Parameters
    ----------
    corrected_signal: array
       The estimate for the corrected signal.
    signal: array
       The measured signal.
    measurement_interval: float
       The total measurement interval in ns
    dead_time: float
       The detector system dead time in ns

    Returns
    -------
    :float
       The sum of squared residuals.
    """
    return np.sum((signal - (corrected_signal * np.exp(-corrected_signal * dead_time / measurement_interval))) ** 2)


def _paralyzable_jacobian(corrected_signal, signal, measurement_interval, dead_time):
    """
    Helper function that calculates the jacobian (derivative) of the objective function
    in respect with each parameter.

    Parameters
    ----------
    corrected_signal: array
       The estimate for the corrected signal.
    signal: array
       The measured signal.
    measurement_interval: float
       The total measurement interval in ns
    dead_time: float
       The detector system dead time in ns

    Returns
    -------
    :array
       The derivative of the objective function.

    """
    factor = corrected_signal * dead_time / measurement_interval
    diagonal_elements = 2 * np.exp(-2 * factor) * (factor - 1) * (signal * np.exp(factor) - corrected_signal)
    return diagonal_elements


def correct_dead_time_polynomial_count_rate(signal, coefficients):
    """ Correct dead time using a polynomial.

    The coefficients will be passed to numpy.pollyval, so they should be order from higher to lower like
    p[0]*x**(N-1) + p[1]*x**(N-2) + ... + p[N-2]*x + p[N-1]

    Parameters
    ----------
    signal: integer array
       The measured count rate in MHz
    coefficients: list
       A list of polynomial coefficients for the correction

    Returns
    -------
    corrected_signal: float array
       The true number of photons arriving at the detector

    Notes
    -----
    The method is described in
    Engelmann, R. et al.: The automated multiwavelength Raman polarization and water-vapor lidar PollyXT:
    the neXT generation, Atmos. Meas. Tech., 9, 1767-1784, doi:10.5194/amt-9-1767-2016, 2016.
    """

    corrected_signal = np.polyval(coefficients, signal)
    return corrected_signal


def correct_dead_time_polynomial_counts(signal, coefficients, measurement_interval):
    """ Correct dead time using a polynomial.

    The coefficients will be passed to numpy.pollyval, so they should be order from higher to lower like
    p[0]*x**(N-1) + p[1]*x**(N-2) + ... + p[N-2]*x + p[N-1]

    Parameters
    ----------
    signal: integer array
       The measured number on photons
    coefficients: list
       A list of polynomial coefficients for the correction
    measurement_interval: float
       The total measurement interval in ns

    Returns
    -------
    corrected_signal: float array
       The true number of photons arriving at the detector

    Notes
    -----
    The method is described in
    Engelmann, R. et al.: The automated multiwavelength Raman polarization and water-vapor lidar PollyXT:
    the neXT generation, Atmos. Meas. Tech., 9, 1767-1784, doi:10.5194/amt-9-1767-2016, 2016.
    """
    count_rate = signal / float(measurement_interval) * 1000.  # Convert to count rate
    corrected_count_rate = correct_dead_time_polynomial_count_rate(count_rate, coefficients)
    corrected_signal = corrected_count_rate * float(measurement_interval) / 1000.  # Covert back to photons
    return corrected_signal


def subtract_background(signal, idx_min, idx_max):
    """ Subtracts the background level from the signal.

    The signal can be either a 1D array or a 2D array with dimensions (time, range).

    Parameters
    ----------
    signal: float array
       The lidar signal vertical profile.
    idx_min: integer
       The minimum index to calculate the background level
    idx_max: integer
       The maximum index to calculate the background level
       
    Returns
    -------
    corrected_signal: float array
       The signal without the background level
    background_mean: float or float array
       The mean value of the background
    background_std: float or float array
       The standard deviation of the background   
    """
    # Convert to 2D with dimensions (time, range) if needed.
    if signal.ndim == 1:
        signal = signal[np.newaxis, :]
        axis_added = True
    else:
        axis_added = False

    if idx_min >= signal.shape[1]:
        raise ValueError('Background subraction index {0} larger than signal length {1}.'.format(idx_min,
                                                                                                 signal.shape[1]))

    background_mean = np.mean(signal[:, idx_min:idx_max], axis=1)
    background_std = np.std(signal[:, idx_min:idx_max], axis=1)

    corrected_signal = (signal.T - background_mean).T

    # Remove extra axis, if added
    if axis_added:
        corrected_signal = corrected_signal[0]
        background_mean = background_mean[0]
        background_std = background_std[0]

    return corrected_signal, background_mean, background_std


def subtract_electronic_background(signal, background_signal):
    """
    Subtract the electronic background profile from the signal.

    The signal can be either a 1D array or a 2D array with dimensions (time, range).

    Parameters
    ----------
    signal: int or float array
       The lidar signal profile.
    background_signal: int or float array
       Electronic background profile.

    Returns
    -------
    output_singal: array
       The array with the corrected signal.
    """
    corrected_signal = signal - background_signal
    return corrected_signal


def correct_overlap(signal, overlap_array=None, full_overlap_idx=None):
    """
    Corrects the signal for a give overlap function.

    Parameters
    ----------
    signal: int or float array
       The lidar signal profile.
    overlap_array: int or float array
       An array containing the overlap function (from 0 to 1)
    full_overlap_idx: int
       Index of full overlap. If specified, all values above this index are considered 1.

    -------
    Returns
    corrected_signal: array
       The overlap-corrected signal.
    """
    overlap_copy = overlap_array[:]  # Make a copy to avoid changing the data

    # Set overlap function to 1 above the full overlap.
    if full_overlap_idx is not None:
        overlap_copy[full_overlap_idx:] = 1

    corrected_signal = signal / overlap_copy
    return corrected_signal


def trigger_delay_to_bins(trigger_delay_ns, altitude_resolution):
    """
    Calculates the integer and fractional part of the trigger delay altitude shift.
    
    Parameters
    ----------
    trigger_delay_ns: float
        Trigger delay value of the channel [ns]
    altitude_resolution: float
        Altitude resolution of the lidar [m]
    
    Returns
    -------
    integer_bins: int
        Part of the trigger delay in meters that is equal to a multiple of
        the altitude resolution
    fraction_bins: float
        The fractional part of the trigger delay in meters
    
    Notes
    -----
    This function can be used to divide the trigger delay so that the integer
    part can be solved with correct_trigger_delay_bins() and the fractional part
    with correct_trigger_delay_ns(). It is required for trigger delay error 
    propagation in the case of interpolation.
    """
    c = constants.c

    trigger_delay_length = trigger_delay_ns * 10 ** -9 * c / 2
    trigger_delay_bins = trigger_delay_length / altitude_resolution
    integer_bins = np.floor(trigger_delay_bins)
    fraction_bins = trigger_delay_bins - integer_bins

    return integer_bins, fraction_bins


def correct_trigger_delay_bins(signal, trigger_delay_bins):
    """
    Shifts the signal for one channel by a specified number of bins.

    Parameters
    ----------
    signal: float array
        The lidar signal profile. Can be either 1D or 2D with dimensions (time, range).
    trigger_delay_bins: int
        Number of bins to shift the signal

    Returns
    -------
    signal: float array
        Corrected lidar signal profile (shifted)
    
    Notes
    -----
    The number of bins to be shifted should be positive. For negative values
    the shift will be done in reverse.
    """
    # Force 2D data
    if signal.ndim == 1:
        signal = signal[np.newaxis, :]
        dimension_added = True
    else:
        dimension_added = False

    signal = np.roll(signal, trigger_delay_bins, axis=1)

    # Remove rolled bins
    if trigger_delay_bins >= 0:
        signal[:, :trigger_delay_bins] = np.nan
    else:
        signal[:, trigger_delay_bins:] = np.nan

    signal = np.ma.masked_invalid(signal)

    # Reduce back to 1D if required
    if dimension_added:
        signal = signal[0, :]

    return signal


def correct_trigger_delay_ns(signal, altitude, trigger_delay_ns):
    """
    Corrects the trigger delay for one channel for trigger delay smaller than
    the difference between 2 neighboring bins.

    Parameters
    ----------
    signal: int or float array
        The lidar signal profile
    altitude: float array
        Altitude array [m]
    trigger_delay_ns : float 
       Trigger delay value of the channel [ns]

    Returns
    -------
    corrected_signal: int or float array
       Corrected signal (shifted and interpolated to the initial altitude values)
    """
    c = constants.c

    corrected_altitude = altitude + trigger_delay_ns * c * 1e-9 / 2.
    corrected_signal = np.interp(altitude, corrected_altitude, signal)
    return corrected_signal


def glue_signals_at_bins(lower_signal, upper_signal, min_bin, max_bin, c_lower, c_upper):
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


def calculate_gluing_values(lower_gluing_region, upper_gluing_region, use_upper_as_reference):
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


def glue_signals_1d(lower_signal, upper_signal, window_length=200, correlation_threshold=0.95,
                 intercept_threshold=0.5, gaussian_threshold=0.2, minmax_threshold=0.5,
                 min_idx=None, max_idx=None, use_upper_as_reference=True):
    """
    Automatically glue two signals.

    Parameters
    ----------
    lower_signal: array
       The low-range signal to be used.
    upper_signal: array
       The high-range signal to be used.
    window_length: int
       The number of bins to be used for gluing
    correlation_threshold: float
       Threshold for the correlation coefficient
    intercept_threshold:
       Threshold for the linear fit intercept
    gaussian_threshold:
       Threshold for the Shapiro-Wilk p-value.
    minmax_threshold:
       Threshold for the min/max ratio
    min_idx, max_idx: int
       Minimum and maximum index to search for a gluing region.
    use_upper_as_reference: bool
       If True, the upper signal is used as reference. Else, the lower signal is used.

    Returns
    -------
    glued_signal: array
       The glued signal array, same size as lower_signal and upper_signal.
    gluing_center_idx: int
       Index choses to perform gluing.
    gluing_score: float
       The gluing score at the chosen point.
    c_lower, c_upper: floats
       Calibration constant of the lower and upper signal. One of them will be 1, depending on the
       value of `use_upper_as_reference` argument.
    """
    lower_signal_cut = lower_signal[min_idx:max_idx]
    upper_signal_cut = upper_signal[min_idx:max_idx]

    gluing_score = get_sliding_gluing_score(lower_signal_cut, upper_signal_cut, window_length, correlation_threshold,
                                            intercept_threshold, gaussian_threshold, minmax_threshold)

    gluing_center_idx = np.argmax(gluing_score) + min_idx  # Index of the original, uncut signals

    min_bin = int(gluing_center_idx - window_length // 2)
    max_bin = int(gluing_center_idx + window_length // 2)

    # Extract the gluing region
    lower_gluing_region = lower_signal[:, min_bin:max_bin]
    upper_gluing_region = upper_signal[:, min_bin:max_bin]

    # Calculate weights to fade-in/fade-out signals in gluing region.
    c_lower, c_upper = calculate_gluing_values(lower_gluing_region, upper_gluing_region, use_upper_as_reference)

    # Perform gluing
    glued_signal = glue_signals_at_bins(lower_signal, upper_signal, min_bin, max_bin, c_lower, c_upper)

    return glued_signal, gluing_center_idx, gluing_score[gluing_center_idx], c_lower, c_upper


def glue_signals_2d(lower_signal, upper_signal, correlation_threshold=0.95,
                 intercept_threshold=0.5, gaussian_threshold=0.2, minmax_threshold=0.5,
                 min_idx=None, max_idx=None, use_upper_as_reference=True):
    """
    Automatically glue two signals.

    Parameters
    ----------
    lower_signal: array
       The low-range signal to be used.
    upper_signal: array
       The high-range signal to be used.
    window_length: int
       The number of bins to be used for gluing
    correlation_threshold: float
       Threshold for the correlation coefficient
    intercept_threshold:
       Threshold for the linear fit intercept
    gaussian_threshold:
       Threshold for the Shapiro-Wilk p-value.
    minmax_threshold:
       Threshold for the min/max ratio
    min_idx, max_idx: int
       Minimum and maximum index to search for a gluing region.
    use_upper_as_reference: bool
       If True, the upper signal is used as reference. Else, the lower signal is used.

    Returns
    -------
    glued_signal: array
       The glued signal array, same size as lower_signal and upper_signal.
    gluing_center_idx: int
       Index choses to perform gluing.
    gluing_score: float
       The gluing score at the chosen point.
    c_lower, c_upper: floats
       Calibration constant of the lower and upper signal. One of them will be 1, depending on the
       value of `use_upper_as_reference` argument.
    """
    lower_signal_cut = lower_signal[:, min_idx:max_idx]
    upper_signal_cut = upper_signal[:, min_idx:max_idx]

    profile_number = lower_signal.shape[0]

    gluing_mask = []
    for profile_idx in range(profile_number):
        gluing_possible = check_gluing_possible(lower_signal_cut[profile_idx, :], upper_signal_cut[profile_idx, :],
                                            correlation_threshold, intercept_threshold, gaussian_threshold, minmax_threshold)
        gluing_mask.append(gluing_possible)

    gluing_mask = np.array(gluing_mask)

    # Extract the gluing region
    lower_gluing_region = lower_signal_cut[gluing_mask, :]
    upper_gluing_region = upper_signal_cut[gluing_mask, :]

    # Calculate weights to fade-in/fade-out signals in gluing region.
    c_lower, c_upper = calculate_gluing_values(lower_gluing_region, upper_gluing_region, use_upper_as_reference)

    # Perform gluing
    glued_signal = glue_signals_at_bins(lower_signal, upper_signal, min_idx, max_idx, c_lower, c_upper)

    return glued_signal, c_lower, c_upper


def get_sliding_gluing_score(lower_signal, upper_signal, window_length, correlation_threshold, intercept_threshold,
                             gaussian_threshold, minmax_threshold):
    """ Get gluing score.

    Parameters
    ----------
    lower_signal : array
       The low-range signal to be used.
    upper_signal : array
       The high-range signal to be used.
    window_length : int
       The number of bins to be used for gluing
    correlation_threshold : float
       Threshold for the correlation coefficient
    intercept_threshold : float
       Threshold for the linear fit intercept
    gaussian_threshold : float
       Threshold for the Shapiro-Wilk p-value.
    minmax_threshold : float
       Threshold for the min/max ratio.

    Returns
    -------
    gluing_score : masked array
       A score indicating regions were gluing is better. Regions were gluing is not possible are masked.
    """

    # Get values of various gluing tests
    intercept_values, correlation_values = fit_checks.sliding_check_linear_fit_intercept_and_correlation(lower_signal, upper_signal, window_length)
    gaussian_values = fit_checks.sliding_check_residuals_not_gaussian(lower_signal, upper_signal, window_length)
    minmax_ratio_values = fit_checks.sliding_check_min_max_ratio(lower_signal, upper_signal, window_length)

    # Find regions where all tests pass
    correlation_mask = correlation_values > correlation_threshold
    intercept_mask = intercept_values < intercept_threshold
    not_gaussian_mask = gaussian_values < gaussian_threshold
    minmax_ratio_large_mask = minmax_ratio_values > minmax_threshold

    gluing_possible = correlation_mask & intercept_mask & ~not_gaussian_mask & minmax_ratio_large_mask

    if not np.any(gluing_possible):
        raise RuntimeError("No suitable gluing regions found.")

    # Calculate a (arbitrary) cost function to deside which region is best
    intercept_scale_value = 40.
    intercept_values[intercept_values > intercept_scale_value] = intercept_scale_value
    intercept_score = 1 - intercept_values / intercept_scale_value

    gluing_score = correlation_values * intercept_score * minmax_ratio_values

    # Mask regions were gluing is not possible
    gluing_score = np.ma.masked_where(~gluing_possible, gluing_score)

    return gluing_score


def check_gluing_possible(lower_signal, upper_signal, correlation_threshold, intercept_threshold,
                             gaussian_threshold, minmax_threshold):
    """ Get gluing score.

    Parameters
    ----------
    lower_signal : array
       The low-range signal to be used.
    upper_signal : array
       The high-range signal to be used.
    correlation_threshold : float
       Threshold for the correlation coefficient
    intercept_threshold : float
       Threshold for the linear fit intercept
    gaussian_threshold : float
       Threshold for the Shapiro-Wilk p-value.
    minmax_threshold : float
       Threshold for the min/max ratio.

    Returns
    -------
    gluing_score : float or nan
       A score indicating regions were gluing is better. If gluing not possible, retunrs nan
    """

    # Get values of various gluing tests
    intercept_value, correlation_value = fit_checks.check_linear_fit_intercept_and_correlation(lower_signal, upper_signal)
    gaussian_value = fit_checks.check_residuals_not_gaussian(lower_signal, upper_signal)
    minmax_ratio_value = fit_checks.check_min_max_ratio(lower_signal, upper_signal)

    # Find regions where all tests pass
    correlation_mask = correlation_value > correlation_threshold
    intercept_mask = intercept_value < intercept_threshold
    not_gaussian_mask = gaussian_value < gaussian_threshold
    minmax_ratio_large_mask = minmax_ratio_value > minmax_threshold

    gluing_possible = correlation_mask & intercept_mask & ~not_gaussian_mask & minmax_ratio_large_mask

    return gluing_possible


def get_array_gluing_score(lower_signal, upper_signal, correlation_threshold, intercept_threshold,
                             gaussian_threshold, minmax_threshold):
    """ Get gluing score for 2D array.

    Parameters
    ----------
    lower_signal : array
       The low-range signal to be used.
    upper_signal : array
       The high-range signal to be used.
    correlation_threshold : float
       Threshold for the correlation coefficient
    intercept_threshold : float
       Threshold for the linear fit intercept
    gaussian_threshold : float
       Threshold for the Shapiro-Wilk p-value.
    minmax_threshold : float
       Threshold for the min/max ratio.

    Returns
    -------
    gluing_score : masked array
       A score indicating regions were gluing is better. Regions were gluing is not possible are masked.
    """

    # Get values of various gluing tests
    intercept_values, correlation_values = fit_checks.check_linear_fit_intercept_and_correlation(lower_signal, upper_signal)
    gaussian_values = fit_checks.check_residuals_not_gaussian(lower_signal, upper_signal)
    minmax_ratio_values = fit_checks.check_min_max_ratio(lower_signal, upper_signal)

    # Find regions where all tests pass
    correlation_mask = correlation_values > correlation_threshold
    intercept_mask = intercept_values < intercept_threshold
    not_gaussian_mask = gaussian_values < gaussian_threshold
    minmax_ratio_large_mask = minmax_ratio_values > minmax_threshold

    gluing_possible = correlation_mask & intercept_mask & ~not_gaussian_mask & minmax_ratio_large_mask

    if not np.any(gluing_possible):
        raise RuntimeError("No suitable gluing regions found.")

    # Calculate a (arbitrary) cost function to deside which region is best
    intercept_scale_value = 40.
    intercept_values[intercept_values > intercept_scale_value] = intercept_scale_value
    intercept_score = 1 - intercept_values / intercept_scale_value

    gluing_score = correlation_values * intercept_score * minmax_ratio_values

    # Mask regions were gluing is not possible
    gluing_score = np.ma.masked_where(~gluing_possible, gluing_score)

    return gluing_score


def interpolate():
    raise NotImplementedError()


def bin_array(data, time, altitude, bins, is_photon_counting=True):
    """
    Performs a binning (addition or averaging) of a signal array.

    If the array is photon counting, the photon counts are summed.
    If the array is analog signals, the signals are averaged.

    Parameters
    ----------
    data : ndarray
       Input data array in (time, altitude).
    time : 1D datetime or float array
       Time array. It can be either a list of datetime objects or floats
    altitude : 1D array
       Altitude array
    bins:
       A (time_bins, altitude_bins) tuple.
    is_photon_counting : bool
       If True, the data are summed, if not they are averaged.

    Returns
    ----------
    binned_data : arrays
       Binned data array
    std_data : arrays
       Standard deviation of each beaned array
    binned_time : 1D array
       Binned time array
    binned_altitude : 1D array
       Binned altitude array
    """

    if data.ndim == 1:
        data = data[np.newaxis, :]
        axis_added = True
    else:
        axis_added = False

    t_bins, z_bins = bins
    t_size, z_size = data.shape

    # Make the new size a multiple of the bins
    new_t_size = t_size // int(t_bins)
    new_z_size = z_size // int(z_bins)

    t_idx_max = new_t_size * t_bins
    z_idx_max = new_z_size * z_bins

    # Reshape the data and make the average by axis
    reshaped_data = np.reshape(data[:t_idx_max, :z_idx_max],
                               (new_t_size, t_bins, new_z_size, z_bins))

    if is_photon_counting:
        binned_data = np.sum(reshaped_data, axis=(1, 3))
    else:
        binned_data = np.mean(reshaped_data, axis=(1, 3))

    std_data = np.std(reshaped_data, axis=(1, 3))

    if axis_added:
        binned_data = binned_data[0, :]
        std_data = std_data[0, :]

    binned_altitude = np.mean(np.reshape(altitude[:z_idx_max], (new_z_size, z_bins)), axis=1)

    # Regrid time
    time_array = np.array(time)

    if isinstance(time[0], (int, float)):  # If time is given in seconds etc.
        binned_time = np.mean(np.reshape(time_array[:t_idx_max], (new_t_size, t_bins)), axis=1)
    else:  # Assume time is an array of datetime object
        min_t = np.min(time_array)
        dts = time_array - min_t

        mean_dts = np.sum(np.reshape(dts[:t_idx_max], (new_t_size, t_bins)), axis=1) / t_bins
        binned_time = min_t + mean_dts

    return binned_data, std_data, binned_time, binned_altitude


def bin_axis(data, bins):
    """ Bin 1D array of data.

    The method can be used if you need to bin altitude or time array without binning the data.

    Parameters
    ----------
    data : 1D np.array
       1D array of floats
    bins : int
       The number of bins to average

    Returns
    -------
    binned_data : 1D np.array
       The binned array
    """

    if data.ndim != 1:
        raise ValueError('This method is used only for 1D arrays.')

    data_size = len(data)

    # Make the new size a multiple of the bins
    new_size = data_size // int(bins)

    idx_max = new_size * bins
    binned_data = np.mean(np.reshape(data[:idx_max], (new_size, bins)), axis=1)

    return binned_data



def bin_profile(data, data_error, altitude, altitude_bins, is_photon_counting=True):
    """
    Performs a binning (addition or averaging) of a signal profile.

    If the profile is photon counting, the photon counts are summed.
    If the profile is analog signals, the signals are averaged.

    Parameters
    ----------
    data : 1D array
       Input data array in (altitude).
    data_error : 1D array
       Uncertainty of the input data.
    altitude : 1D array
       Altitude array
    altitude_bins:
       A altitude bins to merge.
    is_photon_counting : bool
       If True, the data are summed, if not they are averaged.

    Returns
    ----------
    binned_data : arrays
       Binned data array
    binned_altitude : 1D array
       Binned altitude array
    """

    if data.ndim != 1:
        raise ValueError("Bin_profile supports only 1D profiles. Use bin_array function instead.")

    # Create 2D array data
    bins = (1, altitude_bins)
    time = np.array([0, ])
    
    binned_data, _, _, binned_altitude = bin_array(data, time, altitude, bins, is_photon_counting=is_photon_counting)

    # Calculate the sum of square error
    squared_sum, _, _, _ = bin_array(data_error ** 2, time, altitude, bins, is_photon_counting=True)

    if is_photon_counting:
        binned_error = np.sqrt(squared_sum)
    else:
        binned_error = np.sqrt(squared_sum) / altitude_bins

    return binned_data, binned_error, binned_altitude


def counts_to_rate(signal, measurement_interval):
    """
    Transform signal from counts to count rate.

    Parameters
    ----------
    signal: array
       The input signal array
    measurement_interval: float
       The total measurement interval in s

    Returns
    -------
    rate: array
       Count rate signal
    """
    rate = signal / measurement_interval
    return rate


def rate_to_counts(signal, measurement_interval=1.5 * 10 ** -4):
    """
    Transform signal from count rate to counts.

    Parameters
    ----------
    signal: array
       The input signal array
    measurement_interval: float
       The total measurement interval in s

    Returns
    -------
    counts: array
       Counts signal
    """
    counts = signal * measurement_interval
    return counts
