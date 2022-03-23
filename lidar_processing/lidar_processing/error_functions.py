import numpy as np    

def error_dead_time_paralyzable(avg_signal, std_signal, avg_DT, std_DT, avg_MT, std_MT):
    """
    Calculates the STD for dead time nonparalyzable correction.
    
    Parameters
    ----------
    avg_signal: float array
        Lidar signal measured value
    std_signal: float
        STD for measured lidar signal
    avg_DT: float
        Average of dead time [s]
    std_DT: float
        STD of dead time [s]
    avg_MT: float
        Average of measurement time [s]
    std_MT:
        STD of measurement time [s]
    
    Returns
    -------
    std: float array
        STD of the dead time paralyzable corrected signal
        
    Notes
    -----
    The input parameters for dead time and measurement time (avg and std) 
    usually are a constant number. The function will check if the input is a 
    number or an array and if it is a number it will generate an array with the same 
    length as avg_signal. It will be assumed that np.size(avg_DT) = np.size(std_DT),
    np.size(avg_MT) = np.size(std_MT).
    
    The calculation for std is based on the STD formula for the measured signal
    in which appears std for corrected signal. All the parameters are known 
    expect STD for corrected signal.
        
    If the signal is in count rate, the folowing values must be set as arguments:
    std_MT =  0 and avg_MT = 1
    """
    
    if (np.size(avg_DT)==1 & np.size(avg_MT)==1):
        avg_DT = np.ones_like(avg_signal) * avg_DT
        avg_MT = np.ones_like(avg_signal) * avg_MT
        std_DT = np.ones_like(avg_signal) * std_DT
        std_MT = np.ones_like(avg_signal) * std_MT    
        
    std = np.sqrt((std_signal**2 - (-avg_signal**2 / avg_MT * np.exp(-avg_signal * avg_DT / avg_MT) * std_DT)**2 - (avg_signal**2 * avg_DT / avg_MT**2 * np.exp(-avg_signal * avg_DT / avg_MT) * std_MT)**2))/np.abs(np.exp(-avg_signal * avg_DT / avg_MT) * (1 - avg_signal * avg_DT / avg_MT))
    return std

def error_dead_time_nonparalyzable(avg_signal, std_signal, avg_DT, std_DT, avg_MT, std_MT):
    """
    Calculates the STD for dead time nonparalyzable correction.
    
    Parameters
    ----------
    avg_signal: float array
        Lidar signal measured value
    std_signal: float
        STD for measured lidar signal
    avg_DT: float
        Average of dead time [s]
    std_DT: float
        STD of dead time [s]
    avg_MT: float
        Average of measurement time [s]
    std_MT:
        STD of measurement time [s]
    
    Returns
    -------
    std: float array
        STD of the dead time nonparalyzable corrected signal
        
    Note
    ----
    The input parameters for dead time and measurement time (avg and std) 
    usually are a constant number. The function will check if the input is a 
    number or an array and if it is a number it will generate an array with the same 
    length as avg_signal. It will be assumed that np.size(avg_DT) = np.size(std_DT),
    np.size(avg_MT) = np.size(std_MT).
    
    If the signal is in count rate, the folowing values must be set as arguments:
    std_MT =  0 and avg_MT = 1
    """
    
    if (np.size(avg_DT)==1 & np.size(avg_MT)==1):
        avg_DT = np.ones_like(avg_signal) * avg_DT
        avg_MT = np.ones_like(avg_signal) * avg_MT
        std_DT = np.ones_like(avg_signal) * std_DT
        std_MT = np.ones_like(avg_signal) * std_MT
        
    std = np.sqrt((avg_MT**4 * std_signal**2 + avg_signal**4 * avg_MT**2 * std_DT**2 + avg_signal**4 * avg_DT**2 * std_MT**2)/(avg_MT - avg_signal * avg_DT)**4)

    return std
    
def error_range_correction(std_signal, altitude):
    """
    Calculates the STD for range correction.
    
    Parameters
    ----------
    std_signal: float array
        STD for lidar signal
    altitude: float array
        Array with altitude values
    
    Returns
    -------
    std: float array
        STD of the range corrected signal
    """
    
    std = std_signal * altitude ** 2

    return std
 
def error_overlap(std_signal, overlap_array = None):
    """
    Calculates the STD for overlap correction.
    
    Parameters
    ----------
    std_signal: float array
        STD for lidar signal
    overlap_array: float array
        Overlap function values
    
    Returns
    -------
    std: float array
        STD of the overlap corrected signal
        
    Note
    ----
    The NaN or Inf values have been set to 0 for the moment. This was done in 
    order to allow the use of the function without errors regarding number type. 
    """
    
    std = std_signal / overlap_array
    std[np.isnan(std)] = 0
    std[np.isinf(std)] = 0

    return std
    
def error_trigger_delay_ns(altiude, avg_signal, std_signal, avg_trigger_delay_m, std_trigger_delay_m):
    """
    Calculates the STD for trigger delay correction, interpolation case.
    
    Parameters
    ----------
    altitude: float array
        Altitude array [m]
    avg_signal: float array
        Lidar signal array
    std_signal: float array
        Lidar STD for signal array
    avg_trigger_delay_m: float
        Fractional part of trigger delay in meters from trigger_delay_to_bins()
        (suggested) [m]
    std_trigger_delay_m:
        Trigger delay STD for fractional part [m]
    
    Returns
    -------
    std: float array
        STD of the trigger delay interpolation correction
    
    Notes
    -----
    trigger_delay_m is supposed to be the fractional part from trigger_delay_to_bins()
    function from pre_processing.py file. Therefore, the trigger_delay_m value is 
    in meters and is smaller than the altitude resolution of the system.
    
    Input signal must be shifted with the number of bins that results from
    trigger_delay_to_bins().    
    """   
    
    if avg_trigger_delay_m == 0:
        return std_signal
    
    altitude_shift = np.roll(altitude, 1)
    corrected_altitude = altitude + avg_trigger_delay_m     
    signal_shift = np.roll(avg_signal, 1)
    std_signal_shift = np.roll(std_signal, 1)
    
    #terms in the error propagation function
    signal_term = (signal_shift - avg_signal) / (altitude_shift - altitude) * std_trigger_delay_m
    upper_term = (altitude_shift - corrected_altitude) / (altitude_shift - altitude) * std_signal_shift
    lower_term = (corrected_altitude - altitude) / (altitude_shift - altitude) * std_signal
    
    std = np.sqrt(signal_term ** 2 + upper_term ** 2 + lower_term ** 2)
    
    return std

def error_photon_counting(signal, measurement_interval = 1.5 * 10 ** -5):
    """
    Calculates the photon counting standard deviation.

    Parameters
    ----------
    signal: int/float array
        The lidar signal profile
    measurement_interval: float
        Integration time of the signal
    
    Returns
    -------
    signal_error: float array
       Signal standard deviation.
    
    Note
    ----
    measurement_interval is used only when the signal is in count rate. Otherwise
    it must be set 1.
    """
    #signal = signal * measurement_interval
    std = np.sqrt(signal)

    return std
    
def error_analog(signal, nr_profiles_mean = 10):
    """
    Calculates the analog standard deviation.

    Parameters
    ----------
    signal: float array
        The lidar signal profile
    nr_profiles_mean: int
        Number of profiles to be averaged
    
    Returns
    -------
    avg_signal: float array
        Signal average
    std_signal: float array
        Signal standard deviation
        
    Notes
    -----
    signal[X,Y,Z] must have X for channel, Y for profile/file index and Z is altitude     
    """
    avg_signal = []
    std_signal = []
    avg_signal_size = int(np.size(signal, 1) / nr_profiles_mean)
    signal = signal[:,:(avg_signal_size * nr_profiles_mean),:]

    for i in range(avg_signal_size):
        avg_signal.append(np.sum(signal[:,i*nr_profiles_mean:(i+1)*nr_profiles_mean,:], 1)/nr_profiles_mean)
        std_signal.append(np.std(signal[:,i*nr_profiles_mean:(i+1)*nr_profiles_mean,:], 1))

    avg = np.array(avg_signal)
    std = np.array(std_signal)

    return avg, std