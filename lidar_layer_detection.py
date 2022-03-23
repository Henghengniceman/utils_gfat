import numpy as np
from scipy.signal import savgol_filter, find_peaks
from utils import normalize

""" LAYER DETECTION

    Description of Algorithm:
    -------------------------



"""

def layer_detection(raw_signal, ranges, min_range=None, max_range=None):
    """[summary]

    Args:
        raw_signal ([type]): [description]
        ranges ([type]): [description]
        min_range ([type], optional): [description]. Defaults to None.
        max_range ([type], optional): [description]. Defaults to None.
    """

    layer = False

    return layer






#def layer_detection(ranges, signal, range_limits=None, window_width=None,
#                    threshold=None):
#    """
#    Find layers in the signal profile.
#
#    Returns a reference altitude where the layer might be.
#    TODO: return the complete altitude range for the layer
#
#    Input:
#        signal
#        ranges
#        range_limits
#        window_width: [m]
#
#    """
#
#    # Default Optional values
#    if range_limits is None:
#        range_limits = (1500, 10000)
#    if window_width is None:
#        window_width = 50
#    if threshold is None:
#        threshold = 0.1
#
#    # step range
#    step_range = ranges[1] - ranges[0]
#    # window width in ranges
#    window_range = int(window_width / step_range)
#
#    # useful range
#    idx = np.logical_and(ranges >= range_limits[0], ranges <= range_limits[1])
#    ranges = ranges[idx]
#    signal = signal[idx]
#
#    # algorithm to detect layer, based on:
#    #   1. detection of sustained increase in signal along a window of ranges
#    #   2. decision about if the increase is significant
#    ref_range = []
#    i = 0
#    while i < len(ranges):
#        seek = 1
#        j = 0
#        aux = []
#        while seek == 1:
#            if (i + j) < len(ranges):
#                if np.logical_and(signal[i + j] - signal[i - 1] > 0,
#                                  signal[i + j] - signal[i + j - 1] > 0):
#                    aux.append(signal[i+j])
#                    j += 1
#                else:
#                    seek = 0
#            else:
#                seek = 0
#        if j > window_range:
#            delta = (aux[-1] - signal[i - 1]) / signal[i - 1]
#            if delta > threshold:
#                ref_range.append(ranges[i])
#            i += j
#        else:
#            if j == 0:
#                i += 1
#            else:
#                i += j
#
#    # layers reference height
#    if len(ref_range) > 0:
#        layer = True
#    else:
#        layer = False
#
#    return layer, ref_range
