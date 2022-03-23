"""
Calculation of volume and particle depolarization coefficient.
"""

import numpy as np
from scipy import stats


def calibration_constant_cross_total_profile(signal_cross_plus45, signal_cross_minus45,
                                             signal_total_plus45, signal_total_minus45,
                                             r_cross, r_total):
    r"""
    Calculate the calibration constant profile, in a lidar system that is able
    to detect the cross-to-total depolarization ratio.
    
    Parameters
    ----------
    signal_cross_plus45: vector
       The input vertical profile from the cross channel. Calibrator angle phi=45.
    signal_total_plus45: array
       The input vertical profile from the total channel. Calibrator angle phi=45.
    signal_cross_minus45: vector
       The input vertical profile from the cross channel. Calibrator angle phi=-45.
    signal_total_minus45: vector
       The input vertical profile from the total channel. Calibrator angle phi=-45.
    r_cross: float
       The transmission ratio of the cross channel (Rc).
    r_total: float
       The transmission ratio of the total channel (Rt).
    
    Returns
    -------
    c_profile: vector
       The vertical profile of the calibration constant.
    
    Notes
    -----
    The calibration constant is calculated by the following formula:
    
    .. math::
       C = \frac{1 + R_t}{1 + R_c} \cdot \sqrt{\delta'_{+45} \cdot \delta'_{-45}}
    
    
    References
    ----------
    Engelmann, R. et al. The automated multiwavelength Raman polarization and water-vapor lidar
    Polly XT: the neXT generation. Atmos. Meas. Tech., 9, 1767-1784 (2016)
    """
    #   Calculate the signal ratio for the +45 position.
    delta_v45_plus = signal_cross_plus45 / signal_total_plus45
    
    #   Calculate the signal ratio for the -45 position.
    delta_v45_minus = signal_cross_minus45 / signal_total_minus45
    
    #   Calculate the calibration constant vertical profile.
    c_profile = ((1 + r_total) / (1 + r_cross)) * np.sqrt(delta_v45_plus * delta_v45_minus)
    return c_profile


def calibration_constant_cross_parallel_profile(signal_cross_plus45, signal_cross_minus45,
                                                signal_parallel_plus45, signal_parallel_minus45,
                                                t_cross, t_parallel, r_cross, r_parallel):
    r"""
    Calculate the calibration constant in a lidar system that is able to 
    detect the cross-to-parallel depolarization ratio.
    
    Parameters
    ----------
    signal_cross_plus45: vector
       The input vertical profile from the cross channel. Calibrator angle phi=45.
    signal_parallel_plus45: vector
       The input vertical profile from the total channel. Calibrator angle phi=45.
    signal_cross_minus45: vector
       The input vertical profile from the cross channel. Calibrator angle phi=-45.
    signal_parallel_minus45: vector
       The input vertical profile from the total channel. Calibrator angle phi=-45.
    t_cross: float
       Transmittance of cross component through transmitted path.
    t_parallel: float
       Transmittance of parallel component through transmitted path.
    r_cross: float
       Transmittance of cross component through reflected path.
    r_parallel: float
       Transmittance of parallel component through reflected path.
    
    Returns
    -------
    v_star_mean: float
       Calibration constant's mean value (vertical axis).
    v_star_sem: float
       Calibration constant's standard error of the mean (vertical axis).
    
    Notes
    -----
    The calibration constant is calculated by the following formula:
    
    .. math::
       V^* = \frac{[1 + \delta^V tan^2 (\phi)]T_p + [tan^2 (\phi) + \delta^V]T_s}
       {[1 + \delta^V tan^2 (\phi)]R_p + [tan^2 (\phi) + \delta^V]R_s} \cdot \delta^* (\phi)
       
    References
    ----------
    Freudenthaler, V. et al. Depolarization ratio profiling at several wavelengths in pure
    Saharan dust during SAMUM 2006. Tellus, 61B, 165-179 (2008)
    """
    #   Calculate the signal ratio for the +45 position.    
    delta_v45_plus = signal_cross_plus45 / signal_parallel_plus45
    
    #   Calculate the signal ratio for the -45 position.
    delta_v45_minus = signal_cross_minus45 / signal_parallel_minus45
    
    #   Calculate the calibration constant vertical profile.
    v_star_profile = ((t_parallel + t_cross) / (r_parallel + r_cross)) * np.sqrt(delta_v45_plus * delta_v45_minus)
    return v_star_profile


def calibration_constant_value(calibration_constant_profile, first_bin,
                               bin_length, lower_limit, upper_limit):
    r"""
    Calculate the mean calibration constant and its standard error of the mean,
    from the calibration constant profile.
    
    Parameters
    ----------
    c_profile: vector
       The vertical profile of the calibration constant.
    first_bin: integer
       The first bin of the system.
    bin_length: float
       The length of each bin. (in meters)
    lower_limit: float
       The lower vertical limit for the calculation. (in meters)
    upper_limit: float
       The lower vertical limit for the calculation. (in meters)
    
    Returns
    -------
    c_mean: float
       Calibration constant's mean value (vertical axis).
    c_sem: float
       Calibration constant's standard error of the mean (vertical axis).
    """
    #   Convert the lower and upper limit from meters to bins.
    lower_limit = int(first_bin + (lower_limit // bin_length))
    upper_limit = int(first_bin + (upper_limit // bin_length))
    
    #   Select the area of interest.
    c_profile = calibration_constant_profile[lower_limit:(upper_limit+1)]
    
    #   Calculate statistics.    
    c_mean = np.mean(c_profile)
    c_sem = stats.sem(c_profile)
    
    #   Return the statistics.
    return (c_mean, c_sem)
    

def volume_depolarization_cross_total(signal_cross, signal_total, r_cross, r_total, c):
    r"""    
    Calculate the linear volume depolarization ratio in a lidar system that is
    able to detect the cross-to-total depolarization ratio.
    The calibration factor from the delta-90 calibration is being used.
    
    Parameters
    ----------
    signal_cross: vector
       The input vertical profile from the cross channel. Normal measurement (phi=0).
    signal_total: vector
       The input vertical profile from the total channel. Normal measurement (phi=0).
    r_cross: float
       The transmission ratio of the cross channel (Rc).
    r_total: float
       The transmission ratio of the total channel (Rt).
    c: float
       The calibration constant.
    
    Returns
    -------
    delta_v: vector
       The linear volume depolarization.

    Notes
    -----
    The linear volume depolarization ratio is calculated by the formula:
    
    .. math::
       \delta^V = \frac{1 - \frac{\delta'}{C}}{\frac{\delta'R_t}{C} - R_C}
    
    References
    ----------
    Engelmann, R. et al. The automated multiwavelength Raman polarization and water-vapor
    lidar Polly XT: the neXT generation. Atmos. Meas. Tech., 9, 1767-1784 (2016)    
    """
    delta_quote = signal_cross / signal_total
    delta_v = (1 - (delta_quote / c)) / ((delta_quote * r_total / c) - r_cross)
    return delta_v


def volume_depolarization_cross_parallel(signal_cross, signal_parallel,
                                         t_cross, t_parallel, r_cross, r_parallel,
                                         v_star):
    r"""    
    Calculate the linear volume depolarization ratio in a lidar system that is
    able to detect the cross-to-parallel depolarization ratio.
    The calibration factor from the delta-90 calibration is being used.
    
    Parameters
    ----------
    signal_cross: vector
       The input vertical profile from the cross channel. Normal measurement (phi=0).
    signal_parallel: vector
       The input vertical profile from the parallel channel. Normal measurement (phi=0).
    t_cross: float
       Transmittance of cross component through transmitted path.
    t_parallel: float
       Transmittance of parallel component through transmitted path.
    r_cross: float
       Transmittance of cross component through reflected path.
    r_parallel: float
       Transmittance of parallel component through reflected path.
    v_star: float
       The calibration constant.
       
    Returns
    -------
    delta_v: vector
       The linear volume depolarization.
    
    Notes
    -----
    The linear volume depolarization ratio is calculated by the formula:
    
    .. math::
       \delta^V = \frac{\frac{\delta^*}{V^*}T_p - R_p}{R_s - \frac{\delta^*}{V^*}T_s}
    
    References
    ----------
    Freudenthaler, V. et al. Depolarization ratio profiling at several wavelengths in pure
    Saharan dust during SAMUM 2006. Tellus, 61B, 165-179 (2008)
    """
    delta_star = signal_cross / signal_parallel
    delta_v = ((delta_star * t_parallel / v_star) - r_parallel) / (r_cross - (delta_star * t_cross / v_star))
    return delta_v


def particle_depolarization(delta_m, delta_v, molecular_backscatter, particle_backscatter):
    r"""
    Calculate the linear particle depolarization ratio.
    
    Parameters
    ----------
    delta_m: vector
       The linear molecular depolarization ratio.
    delta_v: vector
       The linear volume depolarization ratio.
    molecular_backscatter: vector
       The molecular component of the total backscatter coefficient.
    particle_backscatter: vector
       The particle component of the total backscatter coefficient.
    
    Returns
    -------
    delta_p: vector
       The linear particle depolarization ratio.

    Notes
    -----
    The linear particle depolarization ratio is calculated by the formula:
    
    .. math::
       \delta^p = \frac{(1 + \delta^m)\delta^V \mathbf{R} - (1 + \delta^V)\delta^m}
       {(1 + \delta^m)\mathbf{R} - (1 + \delta^V)}
    
    References
    ----------
    Freudenthaler, V. et al. Depolarization ratio profiling at several wavelengths in pure
    Saharan dust during SAMUM 2006. Tellus, 61B, 165-179 (2008)
    """
    r = (molecular_backscatter + particle_backscatter) / molecular_backscatter
    delta_p = ((1 + delta_m) * delta_v * r - (1 + delta_v) * delta_m) \
              / ((1 + delta_m) * r - (1 + delta_v))
    return delta_p


def backscatter_ratio(beta_mol, beta_aer):
   """
   Backscatter Ratio

   Parameters
   ----------
   beta_mol : [type]
       [description]
   beta_aer : [type]
       [description]
   """

   return (beta_mol + beta_aer) / beta_mol
   
   
def backscatter_ratio_error(beta_mol, error_beta_aer):
   """
   Error Backscatter Ratio

   Parameters
   ----------
   beta_mol : [type]
       [description]
   error_beta_aer : [type]
       [description]
   """

   return error_beta_aer / beta_mol

   
def particle_depolarization_error(beta_aer, beta_mol, delta_mol, delta_v, \
   error_beta_aer, error_delta_v):
   """
   Compute Error for PLDR as estimated above, based on Error Propagation
   Rodríguez-Gómez, A., Sicard, M., Granados-Muñoz, M. J., Chahed, E. Ben, Muñoz-Porcar, C., Barragán, R., … Vidal, E. (2017). An architecture providing depolarization ratio capability for a multi-wavelength raman lidar: Implementation and first measurements. Sensors (Switzerland), 17(12). https://doi.org/10.3390/s17122957
   [Eq. A8]

   Parameters
    ----------
   beta_aer : [type]
       [description]
   beta_mol : [type]
       [description]
   delta_mol : [type]
       [description]
   delta_v : [type]
       [description]
   error_delta_v : [type]
       [description]
   """
   
   # Backscatter Ratio
   rho = backscatter_ratio(beta_mol, beta_aer)
   error_rho = backscatter_ratio_error(beta_mol, error_beta_aer)
   
   # Error PLDR
   num_R = (1 + delta_mol)*delta_v*rho - (1 + delta_v)*delta_mol
   den_R = (1 + delta_mol)*rho - (1 + delta_v)
   x1 = ( ((1 + delta_mol)*rho - delta_mol)*den_R + num_R ) / (den_R**2)
   x2 = ( (1 + delta_mol)*delta_v*den_R - (1 + delta_mol)*num_R ) / (den_R**2)
   error_delta_p = np.sqrt( (x1**2)*(error_delta_v**2) + (x2**2)*(error_rho**2))
   
   return error_delta_p