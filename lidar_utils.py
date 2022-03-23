import numpy as np
from scipy.signal import savgol_filter

""" MODULE For General Lidar Utilities
"""

def smooth_signal(signal, method='savgol', savgol_kwargs=None):
    """ Smooth Lidar Signal

    Args:
        signal ([type]): [description]
        method (str, optional): [description]. Defaults to 'savgol'.
    """

    if method == 'savgol':
        if savgol_kwargs is None:
            savgol_kwargs = {'window_length': 21, 'polyorder': 2}
        smoothed_signal = savgol_filter(signal, **savgol_kwargs)
    else:
        smoothed_signal = signal
        print("signal not smoothed")
    
    return smoothed_signal


def estimate_snr(signal, window=5):
    """[summary]

    Args:
        signal ([type]): [description]
    """

    # ventana: numero impar
    if window%2 == 0:
        window += 1
    subw = window//2

    n = len(signal)
    avg = np.zeros(n)*np.nan
    std = np.zeros(n)*np.nan

    for i in range(n):
        ind_delta_min = i - subw if i - subw >= 0 else 0
        ind_delta_max = i + subw if i + subw < n else n - 1

        si = signal[ind_delta_min:(ind_delta_max+1)]
        avg[i] = np.nanmean(si)
        std[i] = np.nanstd(si)

        #print("%i, %i, %i" % (i, ind_delta_min, ind_delta_max + 1))
        #print(signal[ind_delta_min:(ind_delta_max+1)])
    snr = avg / std

    return snr, avg, std
