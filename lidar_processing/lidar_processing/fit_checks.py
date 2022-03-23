""" This file contains functions that check if two signals fit or not. They can be used to check a
gluing or molecular fit regions.
"""
import numpy as np
from scipy.stats import shapiro
from scipy.stats.mstats import normaltest
from scipy.stats import linregress


def check_correlation(first_signal, second_signal, threshold=None):
    """
    Returns the correlation coefficient between the two signals.

    The signals can be either 1D arrays or 2D arrays containing the rolling slices
    of the input signals. In the 2D case, the function returns the sliding correlation
    between the original signals.

    If a threshold is provided, returns True if
    the correlation is above the specified threshold.

    Parameters
    ----------
    first_signal: array
       The first signal array
    second_signal: array
       The second signal array
    threshold: float or None
       Threshold for the correlation coefficient.

    Returns
    -------
    correlation: float or boolean
       If threshold is None, then the function returns an the correlation coefficient.
       If a threshold is provided, the function returns True if the correlation value is
       above the threshold.
    """
    correlation_matrix = np.corrcoef(first_signal, second_signal)

    # This is needed to make the function work both for 1D vectors and rolling-window 2D matrices
    if first_signal.ndim == 1:
        number_of_slices = 1
    else:
        number_of_slices = first_signal.shape[0]

    first_signal_idxs = np.arange(number_of_slices)
    second_signal_idxs = first_signal_idxs + number_of_slices

    correlation = correlation_matrix[first_signal_idxs, second_signal_idxs]

    if threshold:
        correlation = correlation > threshold

    return correlation


def sliding_check_correlation(first_signal, second_signal, window_length=11, threshold=None):
    """
    Returns the sliding correlation coefficient between the two signals.

    If a threshold is provided, returns True if
    the correlation is above the specified threshold.

    Parameters
    ----------
    first_signal: array
       The first signal array
    second_signal: array
       The second signal array
    window_length: int
       The length of the window. It should be an odd number.
    threshold: float or None
       Threshold for the correlation coefficient.

    Returns
    -------
    correlation: float or boolean
       If threshold is None, then the function returns an the correlation coefficient.
       If a threshold is provided, the function returns True if the correlation value is
       above the threshold.
    """

    # Make window length odd
    if window_length % 2:
        window_length += 1

    sliding_first_signal = _rolling_window(first_signal, window_length)
    sliding_second_signal = _rolling_window(second_signal, window_length)

    correlation = check_correlation(sliding_first_signal, sliding_second_signal, threshold)
    correlation = _restore_array_length(correlation, window_length)

    return correlation


def check_linear_fit_intercept_and_correlation(first_signal, second_signal):
    """
    Check if the intercept of a linear fit is near zero,
    and the correlation coefficient of the two signals.

    Performs a linear fit to the data, assuming y = ax + b, with x the first_signal
    and y the second_signal. It will return the value np.abs(b / np.mean(y) * 100)

    If the intercept is far from zero, it indicates that the two signals
    do not differ from a multiplication constant.


    Parameters
    ----------
    first_signal : array
       The first signal array
    second_signal : array
       The second signal array

    Returns
    -------
    intercept_percent : float or boolean
       The value of the intercept b, relative to the mean value of the second_signal.
    correlation : float
       Correlation coefficient between the two samples
    """
    a, b, correlation, _, _ = linregress(first_signal, second_signal)  # Fit the function y = ax + b

    intercept_percent = np.abs(b / np.mean(second_signal) * 100)

    return intercept_percent, correlation


def sliding_check_linear_fit_intercept_and_correlation(first_signal, second_signal, window_length=11):
    """
    Check if the intercept of a linear fit is near zero.

    Performs a linear fit to the data, assuming y = ax + b, with x the first_signal
    and y the second_signal.

    It will return the value np.abs(b / np.mean(y) * 100) and the correlation of the two signals.


    Parameters
    ----------
    first_signal: array
       The first signal array
    second_signal: array
       The second signal array
    window_length: int
       The length of the window. It should be an odd number.

    Returns
    -------
    intercepts : float or boolean
       The value of the intercept b, relative to the mean value of the second_signal.
    correlations : float
       Correlation coefficient between the two samples
    """
    # Make window length odd
    if window_length % 2:
        window_length += 1

    results = _apply_sliding_check(check_linear_fit_intercept_and_correlation, first_signal, second_signal, window_length)

    intercepts = results[:, 0]
    correlations = results[:, 1]

    intercepts = _restore_array_length(intercepts, window_length)
    correlations = _restore_array_length(correlations, window_length)

    return intercepts, correlations


def check_residuals_not_gaussian(first_signal, second_signal, threshold=None):
    """
    Check if the residuals of the linear fit are not from a normal distribution.

    The function uses a Shapiro-Wilk test on the residuals of a linear fit. Specifically,
    the function performs a linear fit to the data, assuming y = ax, and then calculates the residuals
    r = y - ax. It will return the p value of the Shapiro-Wilk test on the residuals.

    If a threshold is provided, returns True if the p value is below the specified threshold, i.e. if
    the residuals are probably not gaussian.

    Parameters
    ----------
    first_signal: array
       The first signal array
    second_signal: array
       The second signal array
    threshold: float or None
       Threshold for the Shapiro-Wilk p-value.

    Returns
    -------
    p_value: float or boolean
       If threshold is None, then the function returns the p-value of the Shapiro-Wilk test on the residuals.
       If a threshold is provided, the function returns True if p-value is below the threshold.
    """
    # Estimate the residual from a linear fit.
    residuals = _residuals(first_signal, second_signal)

    # Get p-value
    _, p_value = shapiro(residuals)

    if threshold:
        p_value = p_value < threshold

    return p_value


def sliding_check_residuals_not_gaussian(first_signal, second_signal, window_length, threshold=None):
    """
    Check if the residuals of the linear fit are not from a normal distribution.

    The function uses a Shapiro-Wilk test on the residuals of a linear fit. Specifically,
    the function performs a linear fit to the data, assuming y = ax, and then calculates the residuals
    r = y - ax. It will return the p value of the Shapiro-Wilk test on the residuals.

    If a threshold is provided, returns True if the p value is below the specified threshold, i.e. if
    the residuals are probably not gaussian.

    Parameters
    ----------
    first_signal: array
       The first signal array
    second_signal: array
       The second signal array
    window_length: int
       The length of the window. It should be an odd number.
    threshold: float or None
       Threshold for the Shapiro-Wilk p-value.

    Returns
    -------
    p_value: array
       If threshold is None, then the function returns the p-value of the Shapiro-Wilk test on the residuals.
       If a threshold is provided, the function returns True if p-value is below the threshold.
    """
    # Make window length odd
    if window_length % 2:
        window_length += 1

    p_values = _apply_sliding_check(check_residuals_not_gaussian, first_signal, second_signal, window_length, threshold)
    p_values = _restore_array_length(p_values, window_length)

    return p_values


def check_min_max_ratio(first_signal, second_signal, threshold=None):
    """
    Returns the ration between minimum and maximum values (i.e. min / max).

    The operation is performed for both signals and the minimum is returned. The
    aim is to detect regions of large variation e.g. edges of clouds. Similar
    large values will be returned when the signals are near 0, so the relative difference is
    large. Consequently, this test should be used in parallel with checks e.g. about
    signal to noise ratio.

    If a threshold is provided, returns True if the reltio  is above the specified threshold.

    Parameters
    ----------
    first_signal: array
       The first signal array
    second_signal: array
       The second signal array
    threshold: float or None
       Threshold for the correlation coefficient.

    Returns
    -------
    minmax: float or boolean
       If threshold is None, then the function returns the min/max ratio.
       If a threshold is provided, the function returns True if the correlation value is
       above the threshold.
    """
    minmax_first = np.min(first_signal, axis=-1) / np.max(first_signal, axis=-1)
    minmax_second = np.min(second_signal, axis=-1) / np.max(second_signal, axis=-1)

    combined_minmax = np.min([minmax_first, minmax_second], axis=0)

    if threshold:
        combined_minmax = combined_minmax > threshold

    return combined_minmax


def sliding_check_min_max_ratio(first_signal, second_signal, window_length=11, threshold=None):
    """
    Returns the sliding min/max ratio for both signals

    If a threshold is provided, returns True if
    the min/max ratio is above the specified threshold.

    Parameters
    ----------
    first_signal: array
       The first signal array
    second_signal: array
       The second signal array
    window_length: int
       The length of the window. It should be an odd number.
    threshold: float or None
       Threshold for the correlation coefficient.

    Returns
    -------
    correlation: float or boolean
       If threshold is None, then the function returns an the correlation coefficient.
       If a threshold is provided, the function returns True if the correlation value is
       above the threshold.
    """

    # Make window length odd
    if window_length % 2:
        window_length += 1

    sliding_first_signal = _rolling_window(first_signal, window_length)
    sliding_second_signal = _rolling_window(second_signal, window_length)

    minmax_ratio = check_min_max_ratio(sliding_first_signal, sliding_second_signal, threshold)
    minmax_ratio = _restore_array_length(minmax_ratio, window_length)

    return minmax_ratio


def check_residuals_not_gaussian_dagostino(first_signal, second_signal, threshold=None):
    """
    Check if the residuals of the linear fit are not from a normal distribution.

    The function uses a D'agostino - Pearsons's test on the residuals of a linear fit. Specifically,
    the function performs a linear fit to the data, assuming y = ax, and then calculates the residuals
    r = y - ax. It will return the p value of the D'agostino - Pearsons's omnibus test on the residuals.

    If a threshold is provided, returns True if the p value is below the specified threshold, i.e. if
    the residuals are probably not gaussian.

    Parameters
    ----------
    first_signal: array
       The first signal array
    second_signal: array
       The second signal array
    threshold: float or None
       Threshold for the Shapiro-Wilk p-value.

    Returns
    -------
    p_value: float or boolean
       If threshold is None, then the function returns the p-value of the D'agostino - Pearsons's test on the residuals.
       If a threshold is provided, the function returns True if p-value is below the threshold.
    """
    residuals = _residuals(first_signal, second_signal)

    # Get p-value
    _, p_value = normaltest(residuals)

    if threshold:
        p_value = p_value < threshold

    return p_value


def sliding_check_residuals_not_gaussian_dagostino(first_signal, second_signal, window_length, threshold=None):
    """
    Check if the residuals of the linear fit are not from a normal distribution.

    The function uses a Shapiro-Wilk test on the residuals of a linear fit. Specifically,
    the function performs a linear fit to the data, assuming y = ax, and then calculates the residuals
    r = y - ax. It will return the p value of the Shapiro-Wilk test on the residuals.

    If a threshold is provided, returns True if the p value is below the specified threshold, i.e. if
    the residuals are probably not gaussian.

    Parameters
    ----------
    first_signal: array
       The first signal array
    second_signal: array
       The second signal array
    window_length: int
       The length of the window. It should be an odd number.
    threshold: float or None
       Threshold for the Shapiro-Wilk p-value.

    Returns
    -------
    p_value: array
       If threshold is None, then the function returns the p-value of the Shapiro-Wilk test on the residuals.
       If a threshold is provided, the function returns True if p-value is below the threshold.
    """
    # Make window length odd
    if window_length % 2:
        window_length += 1

    p_values = _apply_sliding_check(check_residuals_not_gaussian_dagostino, first_signal, second_signal, window_length, threshold)
    p_values = _restore_array_length(p_values, window_length)

    return p_values


def _residuals(first_signal, second_signal):
    # Fit y = ax
    a, _, _, _ = np.linalg.lstsq(first_signal[:, np.newaxis], second_signal)
    # Calculate residuals
    y_predicted = first_signal * a
    residuals = y_predicted - second_signal
    return residuals


def _rolling_window(a, window):
    """
    Return a rolling window view of the input array. This can be used to calculate moving window statistics
    efficiently.

    The code is from http://www.rigtorp.se/2011/01/01/rolling-statistics-numpy.html

    Parameters
    ----------
    a: array
       Input array
    window: int
       Window length

    Returns
    -------
    : array
       Rolling window view of the array. One dimension larger than the input array.
    """
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)

    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)  # Maybe add writeable=False in the future


def _apply_sliding_check(check_function, first_signal, second_signal, window_length, threshold=None):
    """ Apply the callable check function to a sliding window.
    """
    signal_length = len(first_signal)
    output_list = []

    for min_idx in range(signal_length - window_length + 1):
        max_idx = min_idx + window_length
        first_slice = first_signal[min_idx:max_idx]
        second_slice = second_signal[min_idx:max_idx]

        if threshold:
            check_output = check_function(first_slice, second_slice, threshold)
        else:
            check_output = check_function(first_slice, second_slice)
        output_list.append(check_output)

    output_array = np.array(output_list)

    return output_array


def _restore_array_length(input_array, window_length):
    """
    Pad an array with the missing values.

    It is used to restore the shape of the array after applying sliding window operation.

    Parameters
    ----------
    input_array: array
       The input 1D numpy array.
    window_length:
       Size of the window used for the sliding operation/

    Returns
    -------
    output_array: array
       The padded array.
    """
    prepend_region_length = window_length // 2
    append_region_length = window_length - prepend_region_length - 1

    # Create nan arrays fill the begining and end of the array
    prepend_array = np.ma.masked_all(prepend_region_length, dtype=input_array.dtype)
    append_array = np.ma.masked_all(append_region_length, dtype=input_array.dtype)
    output_array = np.ma.append(prepend_array, input_array)
    output_array = np.ma.append(output_array, append_array)

    output_array = np.ma.masked_invalid(output_array)
    return output_array

