""" Routines to process High Spectral Resolution Lidar (HSRL) signals.

.. todo::
   Fix molecular scattering calculations. Currently the bacskcatter and extinction
   calculations are done using slightly different formulas.

   Fix cabannes line calculations.

.. warning::
   Functions under development, still untested.
"""
import numpy as np
from scipy.signal import savgol_filter

from helper_functions import molecular_backscatter, molecular_extinction, number_density_at_pt

from lidar_processing.pre_processing import apply_range_correction


def hsr_calibration_constant(signal_total, signal_mol, Cmm, Cmt, Cam, Cat, scattering_ratio=0):
    """ Calculate the calibration constant for two HSR channels. 
        
    Parameters
    ----------
    signal_total: array
       The signal of the total channel.
    signal_mol: array
       The signal in the molecular channel
    Cmm, Cmt, Cam, Cat: float or array
       The cross-talk coefficients for the two channels. Cmm->Molecular signal to molecular channel, etc.
    scattering_ratio: float
       The ratio of aerosol to molecular backscatter at the specific altitude.
           
    Returns
    -------
    calibration_constant: float
       The relative calibration constant eta_rho between the two channels (total / molecular).
    calibration_error: float
       The uncertainty (standard deviation of the mean) for the constant.
    """

    R = scattering_ratio  # Shorthand

    # Calculate calibration for each point
    calibration_profile = (Cmm + R * Cam) / (Cmt + R * Cat) * (signal_total / signal_mol)

    # Find mean and standard deviation of the mean
    calibration_constant = np.mean(calibration_profile)

    n = len(calibration_profile)  # The number of samples
    calibration_error = np.std(calibration_profile) / np.sqrt(n)

    return calibration_constant, calibration_error


def signal_unmixing(signal_total, signal_mol, Cmm, Cmt, Cam, Cat, calibration_constant):
    """ Calculate the molecular and aerosol photons arriving at the lidar. 
        
    Parameters
    ----------
    signal_total: array
       The signal of the total channel.
    signal_mol: array
       The signal in the molecular channel
    Cmm, Cmt, Cam, Cat: float or array
       The cross-talk coefficients for the two channels. Cmm->Molecular signal to molecular channel, etc.
    calibration_constant: float
       The relative calibration constant eta_rho between the two channels (total / molecular).
           
    Returns
    -------
    N_a, N_m: arrays
       The aerosol and molecular photons arriving at the detector.
    """
    S_t = signal_total  # Shorthands
    S_m = signal_mol
    denominator = Cam * Cmt - Cmm * Cat

    N_a = (calibration_constant * Cmt * S_m - Cmm * S_t) / denominator
    N_m = (Cam * S_t - calibration_constant * Cat * S_m) / denominator

    return N_a, N_m


def aerosol_backscatter(signal_aerosol, signal_mol, temperature, pressure, wavelength):
    """ Calculates the aerosol backscatter coefficient. 

    The profiles of temperature and pressure are used, together with CO2 and 
    water vaport concentrations, to calculate the scattering properties of 
    the molecular atmosphere.
    
    Parameters
    ----------
    signal_aerosol: array
       The range-corrected aerosol photons arriving at the lidar.
    signal_mol: array
       The range_corrected molecular photons arriving at the lidar
    temperature: array
       The temperature profile for each measurement point [Kelvin]
    pressure: array
       The pressure profile for each measurement point [Pa]
    wavelength: float
       Emission wavelength [nm]
               
    Returns
    -------
    beta_aer: array
       The aerosol backscatter coefficient [m-1 sr-1]
    """

    R = signal_aerosol / signal_mol  # Scattering ratio
    beta_pi = molecular_backscatter(wavelength, pressure, temperature, component='cabannes')

    beta_aer = R * beta_pi

    return beta_aer


def aerosol_extinction(signal, dz, temperature, pressure, wavelength,window_size=11, order=2):
    """ Calculates the aerosol extinction coefficient. 

    The profiles of temperature and pressure are used, together with CO2 and 
    water vaport concentrations, to calculate the scattering properties of 
    the molecular atmosphere.
    
    The derivative is calculated using a Savitzky-Golay filter.
    
    Parameters
    ----------
    signal:
       The range corrected molecular photons arriving at the lidar
    dz: float
       Altitude step, used in the derivative [m]
    temperature: array
       The temperature profile for each measurement point [Kelvin]
    pressure: array
       The pressure profile for each measurement point [Pa]
    wavelength: float
       Emission wavelength [nm]
    window_size : int
        the length of the smoothing window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.       
    
    Returns
    -------
    alpha_aer: arrays
       The aerosol extinction coefficient [m-1]
    """

    # Molecular parameters
    alpha_m = molecular_extinction(wavelength, pressure, temperature)
    n = number_density_at_pt(pressure, temperature)

    # Ratio to apply derivative
    ratio = np.ma.log(n / signal)

    derivative = savgol_filter(ratio, window_size, order, deriv=1, delta=dz, mode='nearest')  # Calculate 1st derivative

    alpha_aer = 0.5 * derivative - alpha_m

    return alpha_aer


def retrieve_channel(signal_total, signal_mol, z, temperature, pressure,
                     Cmm, Cmt, Cam, Cat, wavelength,
                     cal_idx_min=800, cal_idx_max=1000, eta_rho=None,
                     window_size=11, order=2, ):
    """ Retrieve the optical parameters from channel data. 
    
    Parameters
    ----------
    signal_total, signal_mol: array
       The total and molecular signals received from the system
    z: array
       The altitude of each range bin [m]
    temperature: array
       The temperature profile for each measurement point [Kelvin]
    pressure: array
       The pressure profile for each measurement point [hPa]
    Cmm, Cmt, Cam, Cat: float or array
       The cross-talk coefficients for the two channels. Cmm->Molecular signal 
       to molecular channel, etc.
    wavelength: float
       Emission wavelength [nm]
    cal_idx_min, cal_idx_max: int
       Array index for the calibration region
    eta_rho: float
       Calibration constant. If provided, the calibration indexes are ignored.
    window_size : int
        the length of the smoothing window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.      
    
    Returns
    -------
    alpha_aer, beta_aer: array
       Aerosol extinction coefficients [m-1] and bacsckatter coefficients[m-1 sr-1]
    """

    # Get step
    dz = np.diff(z)[0]

    # Calibrate
    if not eta_rho:
        eta_rho, eta_error = hsr_calibration_constant(signal_total[cal_idx_min:cal_idx_max],
                                                      signal_mol[cal_idx_min:cal_idx_max],
                                                      Cmm, Cmt, Cam, Cat)

    # Unmix signals
    Na, Nm = signal_unmixing(signal_total, signal_mol, Cmm, Cmt, Cam, Cat, eta_rho)

    # Range correct
    Pm = apply_range_correction(Nm, z)
    Pa = apply_range_correction(Na, z)

    # Optical preperties                                                         
    alpha_aer = aerosol_extinction(Pm, dz, temperature, pressure, wavelength, window_size=window_size, order=order)
    beta_aer = aerosol_backscatter(Pa, Pm, temperature, pressure, wavelength)

    return alpha_aer, beta_aer
