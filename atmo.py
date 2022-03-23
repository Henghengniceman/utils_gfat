"""
Tomado prestado de lidar_processing.lidar_processing.helper_functions
"""

import numpy as np
import scipy
from scipy import interpolate

#from .constants import k_b
from constants import k_b

# Scattering parameters according to Freudenthaler V., 2015.
Fk = {308: 1.05574,    
      351: 1.05307,
      354.717: 1.05209,
      355: 1.05288,
      386.890: 1.05166,
      400: 1.05125,
      407.558: 1.05105,
      510.6: 1.04922,
      532: 1.04899,
      532.075: 1.04899,
      607.435: 1.04839,
      710: 1.04790,
      800: 1.04763,
      1064: 1.04721,
      1064.150: 1.04721}

epsilon = {308: 0.25083,    
      351: 0.238815,
      354.717: 0.234405,
      355: 0.23796,
      386.890: 0.23247,
      400: 0.230625,
      407.558: 0.229725,
      510.6: 0.22149,
      532: 0.220455,
      532.075: 0.220455,
      607.435: 0.217755,
      710: 0.21555,
      800: 0.214335,
      1064: 0.212445,
      1064.150: 0.212445}

Cs = {308: 3.6506E-5,   # K/hPa/m
      351: 2.0934E-5,
      354.717: 2.0024E-5,
      355: 1.9957E-5,
      386.890: 1.3942E-5,
      400: 1.2109E-5,
      407.558: 1.1202E-5,
      510.6: 4.4221E-6,
      532: 3.7382E-6,
      532.075: 3.7361E-6,
      607.435: 2.1772E-6,
      710: 1.1561E-6,
      800: 7.1364E-7,
      1064: 2.2622E-7,
      1064.150: 2.2609E-7}

BsT = {308: 4.2886E-6,
       351: 2.4610E-6,
       354.717: 2.3542E-6,
       355: 2.3463E-6,
       400: 1.4242E-6,
       510.6: 5.2042E-7,
       532: 4.3997E-7,
       532.075: 4.3971E-7,
       710: 1.3611E-7,
       800: 8.4022E-8,
       1064: 2.6638E-8,
       1064.150: 2.6623E-8}

BsC = {308: 4.1678E-6,
       351: 2.3949E-6,
       354.717: 2.2912E-6,
       355: 2.2835E-6,
       400: 1.3872E-6,
       510.6: 5.0742E-7,
       532: 4.2903E-7,
       532.075: 4.2878E-7,
       710: 1.3280E-7,
       800: 8.1989E-8,
       1064: 2.5999E-8,
       1064.150: 2.5984E-8}

BsC_parallel = {308: 4.15052184e-6,
       351:  2.38547616e-06,
       354.717: 2.28368241e-06,
       355: 2.27451222e-06,
       400: 1.38198631e-06,
       510.6: 5.05563542e-07,
       532: 4.27459520e-07,
       532.075: 4.27219387e-07,
       710: 1.32322062e-07,
       800: 8.16989147e-08,
       1064: 2.59074156e-08,
       1064.150: 2.58925276e-08}

BsC_perpendicular = {308: 1.72550768e-08,
       351:  9.44466863e-09,
       354.717: 8.87554356e-09,
       355:  8.97326485e-09,
       400: 5.28492464e-09,
       510.6: 1.85714694e-09,
       532: 1.56293632e-09,
       532.075: 1.56205831e-09,
       710: 4.73100855e-10,
       800: 2.90465461e-10,
       1064: 9.13006514e-11,
       1064.150: 9.12481844e-11}

KbwT = {308: 1.01610,
        351: 1.01535,
        354.717: 1.01530,
        355: 1.01530,
        400: 1.01484,
        510.6: 1.01427,
        532: 1.01421,
        532.075: 1.01421,
        710: 1.01390,
        800: 1.01383,
        1064: 1.01371,
        1064.150: 1.01371}

KbwC = {308: 1.04554,
        351: 1.04338,
        354.717: 1.04324,
        355: 1.04323,
        400: 1.04191,
        510.6: 1.04026,
        532: 1.04007,
        532.075: 1.04007,
        710: 1.03919,
        800: 1.03897,
        1064: 1.03863,
        1064.150: 1.03863}

# Create interpolation function once, to avoid re-calculation (does it matter?)
f_ext = interpolate.interp1d(list(Cs.keys()), list(Cs.values()), kind='cubic')
f_bst = interpolate.interp1d(list(BsT.keys()), list(BsT.values()), kind='cubic')
f_bsc = interpolate.interp1d(list(BsC.keys()), list(BsC.values()), kind='cubic')
f_bsc_parallel = interpolate.interp1d(list(BsC_parallel.keys()), list(BsC_parallel.values()), kind='cubic')
f_bsc_perpendicular = interpolate.interp1d(list(BsC_perpendicular.keys()), list(BsC_perpendicular.values()), kind='cubic')

# Splines introduce arifacts due to limited input resolution
f_kbwt = interpolate.interp1d(list(KbwT.keys()), list(KbwT.values()), kind='linear')
f_kbwc = interpolate.interp1d(list(KbwC.keys()), list(KbwC.values()), kind='linear')


def standard_atmosphere(altitude, temperature_surface=288.15, pressure_surface = 101325.):
   """
   Calculation of Temperature and Pressure Profiles in Standard Atmosphere.

   Parameters
   ----------
   altitude: float
      The altitude above sea level. (m)

   Returns
   -------
   pressure: float
      The atmospheric pressure. (N * m^-2 or Pa)
   temperature: float
      The atmospheric temperature. (K)
   density: float
      The air density. (kg * m^-3)

   References
   ----------
   http://home.anadolu.edu.tr/~mcavcar/common/ISAweb.pdf
   """

   # Dry air specific gas constant. (J * kg^-1 * K^-1)
   R = 287.058

   g = 9.8 #m/s^2
    
   # Temperature calculation.
   if altitude < 11000:
      temperature = temperature_surface - 6.5 * altitude / 1000.
   else:
      temperature = temperature_surface - 6.5 * 11000 / 1000.
   # Pressure calculation.
   if altitude < 11000:
      pressure = pressure_surface * (1 - (0.0065 * altitude / temperature_surface)) ** 5.2561
   else:
      # pressure = pressure_surface*((temperature/scaled_T[idx])**-5.2199))\
      #                       *np.exp((-0.034164*(_height - z_tmp))/scaled_T[idx])
      tropopause_pressure = pressure_surface * (1 - (0.0065 * 11000 / temperature_surface)) ** 5.2561
      tropopause_temperature = temperature
      pressure = tropopause_pressure*np.exp(-(altitude-11000)*(g/(R*tropopause_temperature)))

   # Density calculation.
   density = pressure / (R * temperature)

   return pressure, temperature, density


def molecular_backscatter(wavelength, pressure, temperature, component='total'):
    """
    Molecular backscatter calculation.

    Parameters
    ----------
    wavelength : float
       The wavelength of the radiation in air. From 308 to 1064.15
    pressure : float
       The atmospheric pressure. (Pa)
    temperature : float
       The atmospheric temperature. (K)
    component : str
       One of 'total' or 'cabannes'.

    Returns
    -------
    beta_molecular: float
       The molecular backscatter coefficient. (m^-1 * sr^-1)

    References
    ----------
    Freudenthaler, V. Rayleigh scattering coefficients and linear depolarization
    ratios at several EARLINET lidar wavelengths. p.6-7 (2015)
    """
    if component not in ['total', 'cabannes', 'cabannes_parallel', 'cabannes_perpendicular']:
        raise ValueError("Molecular backscatter available only for 'total' or 'cabannes' component.")

    if component == 'total':
       bs_function = f_bst
    elif component == 'cabannes':
       bs_function = f_bsc
    elif  component == 'cabannes_parallel':
       bs_function = f_bsc_parallel         
    elif  component == 'cabannes_perpendicular':
       bs_function = f_bsc_perpendicular

    Bs = bs_function(wavelength)

    # Convert pressure to correct units for calculation. (Pa to hPa)
    pressure = pressure / 100.

    # Calculate the molecular backscatter coefficient.
    beta_molecular = Bs * pressure / temperature

    return beta_molecular


def molecular_lidar_ratio(wavelength, component='total'):
    """
    Molecular lidar ratio.

    Parameters
    ----------
    wavelength : float
       The wavelength of the radiation in air. From 308 to 1064.15
    component : str
       One of 'total' or 'cabannes'.

    Returns
    -------
    lidar_ratio_molecular : float
       The molecular backscatter coefficient. (m^-1 * sr^-1)

    References
    ----------
    Freudenthaler, V. Rayleigh scattering coefficients and linear depolarization
    ratios at several EARLINET lidar wavelengths. p.6-7 (2015)
    """
    if component not in ['total', 'cabannes']:
        raise ValueError("Molecular lidar ratio available only for 'total' or 'cabannes' component.")

    if component == 'total':
        k_function = f_kbwt
    else:
        k_function = f_kbwc

    Kbw = k_function(wavelength)

    lidar_ratio_molecular = 8 * np.pi / 3. * Kbw

    return lidar_ratio_molecular


def molecular_extinction(wavelength, pressure, temperature):
    """
    Molecular extinction calculation.

    Parameters
    ----------
    wavelength : float
       The wavelength of the radiation in air. From 308 to 1064.15
    pressure : float
       The atmospheric pressure. (Pa)
    temperature : float
       The atmospheric temperature. (K)

    Returns
    -------
    alpha_molecular: float
       The molecular extinction coefficient. (m^-1)

    References
    ----------
    Freudenthaler, V. Rayleigh scattering coefficients and linear depolarization
    ratios at several EARLINET lidar wavelengths. p.6-7 (2015)
    """
    cs = f_ext(wavelength)

    # Convert pressure to correct units for calculation. (Pa to hPa)
    pressure = pressure / 100.

    # Calculate the molecular backscatter coefficient.
    alpha_molecular = cs * pressure / temperature

    return alpha_molecular


def transmittance(alpha, heights):
    """
    Transmittance = exp[-integral{alpha*dz}_[0,z]]

    Parameters
    ----------
    alpha: array
        extinction profile
    heights: array
        heights profile

    Returns
    -------
    transmittance: array
         transmittance

    """

    delta_height = np.median(np.diff(heights))
    integrated_extinction = scipy.integrate.cumtrapz(alpha, initial=0, dx=delta_height)
    return np.exp(-integrated_extinction)


def attenuated_backscatter(backscatter, transmittance):
   """ Calculate Attenuated Backscatter

   Args:
       backscatter ([type]): [description]
       transmittance ([type]): [description]
   """

   return backscatter*transmittance**2   


def number_density_at_pt(pressure, temperature):
    """ Calculate the number density for a given temperature and pressure.

    This method does not take into account the compressibility of air.

    Parameters
    ----------
    pressure: float or array
       Pressure in Pa
    temperature: float or array
       Temperature in K

    Returns
    -------
    n: array or array
       Number density of the atmosphere [m-3]
    """
    #p_pa = pressure * 100.  # Pressure in pascal

    n = pressure / (temperature * k_b)

    return n
