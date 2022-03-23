
import os
import sys
import time
import importlib
import numpy as np
import pandas as pd
import xarray as xr
import datetime as dt
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from cycler import cycler
plt.ion()
plt.close('all')

from scipy.signal import savgol_filter
from scipy import integrate

import pdb

""" import utils_gfat modules """
MODULE_DIR = os.path.dirname(sys.modules[__name__].__file__)
sys.path.insert(0, MODULE_DIR)
import utils
import plot
import solar
import lidar_processing.lidar_processing.elastic_retrievals as elastic_retrievals
import lidar_processing.lidar_processing.helper_functions as helper_functions

""" BACKSCATTER RETRIEVAL Functions """
def get_ref_height_index(height, z_min, z_max):
    """

    Returns index position for ref heights
    """
    z_ref = 0.5*(z_min + z_max)
    idx_ref, _ = utils.find_nearest_1d(height, z_ref)
    idx_min, _ = utils.find_nearest_1d(height, z_min)
    idx_max, _ = utils.find_nearest_1d(height, z_max)
    ref_range = np.max((idx_ref - idx_min, idx_max - idx_ref))
    return idx_ref, ref_range

def integrate_from_reference(integrand, x, reference_index):
    """

    at x[ref_index], the integral equals = 0
    """
    # integrate above reference
    int_above_ref = integrate.cumtrapz(integrand[reference_index:], x=x[reference_index:])

    # integrate below reference
    int_below_ref = integrate.cumtrapz(integrand[:reference_index+1][::-1], x=x[:reference_index+1][::-1])[::-1]

    return np.concatenate([int_below_ref, np.zeros(1), int_above_ref])


def snr_filter(snr, height, threshold=1.0, min_height=1000):
    """
    snr_filter 

    [extended_summary]

    Parameters
    ----------
    snr : [type]
        [description]
    threshold : float, optional
        [description], by default 1.0

    Returns
    -------
    [type]
        [description]
    """
    n_z = len(snr)
    
    # get index of min height to start snr filter
    i = 0
    stop = 0
    while stop == 0:
        if i < len(height):
            if (height[i] < min_height):
                i += 1
            else:
                stop = 1
        else:
            stop = 1
    
    # find index for snr filter
    c = i
    stop = 0
    while stop == 0:
        if c < n_z:
            if snr[c] < threshold:
                stop = 1
            else:
                c += 1
        else:
            stop = 1
    return c


def negative_filter(beta_att):
    """
    negative_filter [summary]

    Turns positive those negative values at lowest altitudes

    Parameters
    ----------
    beta_att : [type]
        [description]
    """
    c0 = 0
    stop = 0
    while stop == 0:
        if c0 < 10:
            if beta_att[c0] > 0:
                c0 += 1
            else:
                stop = 1
        else:
            stop = 1
    if np.logical_and(c0 == 10, beta_att[c0] > 0):
      c0 = 0
    c = c0
    while beta_att[c] < 0:
        c += 1
    y = beta_att[1:c+1]
    if c > 0:
        y2 = np.concatenate([np.zeros(1)+1e-9, y])
        not_nan = np.logical_not(y2 < 0)
        indices = np.arange(len(y2))
        beta_att_int = np.concatenate([np.interp(indices, indices[not_nan], y2[not_nan]), beta_att[c+1:] ])
    else:
        beta_att_int = beta_att

    return beta_att_int


def optical_depth(extinction, height, ref_index=0):
    """
    Integrate extinction profile along height: r'$\tau(z) = \int_0^z d\dseta \alpha(\dseta)$'
    """

    return integrate_from_reference(extinction, height, reference_index=ref_index)

def refill_overlap(beta, height, fulloverlap_height = 600):    
    """
    Fill overlap region with the first valid value according to fulloverlap_height. 
    """    
    idx_overlap = (np.abs(height-fulloverlap_height)).argmin()
    beta[:idx_overlap] = beta[idx_overlap]
    return beta

def attenuated_backscatter(beta_mol, beta_aer, transmission_mol, transmission_aer):
    return (beta_mol + beta_aer)*(transmission_mol*transmission_aer)**2

def find_lidar_ratio(rcs, height, beta_mol, lr_mol, reference_aod, mininum_height= 0, lr_initial=50, lr_resol =1, max_iterations=100, rel_diff_aod_percentage_threshold=1, debugging=False,**kwargs):

    """ Get Input Arguments """
    ymin = kwargs.get("ymin", 7000)
    ymax = kwargs.get("ymax", 8000)   
    
    range_resolution = np.median(np.diff(height)) 
    
    idx_min = np.abs(height-mininum_height).argmin()

    #Initialize loop
    lr_, iter_, run, success = lr_initial, 0, True, False
    while run:
        iter_ = iter_ + 1
        beta_ = klett(rcs, height, beta_mol, lr_mol, lr_aer = lr_, ymin = ymin, ymax = ymax) 
        beta_ = refill_overlap(beta_, height)        
        aod_ = integrate.simps(beta_[idx_min:]*lr_,dx=range_resolution)    
        rel_diff_aod =100*(aod_ - reference_aod)/reference_aod
        if debugging:
            print('lidar_aod: %.3f| reference_aod: %.3f | relative_difference: %.1f%%' % (aod_, reference_aod, rel_diff_aod))
        if np.abs(rel_diff_aod) > rel_diff_aod_percentage_threshold:
            if rel_diff_aod > 0:
                if lr_ < 10:                    
                    run = False
                    print('No convergence. LR goes too low.')
                else:
                    lr_ = lr_ - 1
            else: 
                if lr_ > 150:                    
                    run = False
                    print('No convergence. LR goes too high.')
                else:
                    lr_ = lr_ + 1
        else:
            print('LR found: %f' % lr_)
            run = False
            success = True
        if iter_ == max_iterations:
            run = False
            print('No convergence. Too many iterations.')
    return lr_, rel_diff_aod, success

def klett(rcs, height, beta_mol, lr_mol, lr_aer =45, ymin = 8000, ymax = 8500, aerosol_backscatter_at_reference= 0):
    '''
    Klett clásico verificado con Fernald,  F.  G.:, Appl. Opt., 23, 652–653, 1984
    Input:
        rcs: numpy.ndarray
        height: numpy.ndarray
        beta_mol: numpy.ndarray        
        lr_mol: numpy.ndarray
        lr_aer: float/int sr
        ymin: float/int m
        ymax: float/int m
    Output:
        particle_beta: numpy.ndarray
    '''             
    particle_beta = np.zeros(len(height))
    ytop = (np.abs(height-ymax)).argmin()
    range_resolution = np.median(np.diff(height))
    idx_ref = np.logical_and(height>=ymin, height<=ymax)    
    if idx_ref.any():
        calib = np.nanmean(rcs[idx_ref]/(beta_mol[idx_ref] + aerosol_backscatter_at_reference))
        # print(np.nanmean(beta_mol[idx_ref]))
        #Calculo de backscatter cuando i<Z0    
        # integer1 = np.flip(-integrate.cumtrapz(np.flip(beta_mol[:ytop]),dx=range_resolution,initial=0))        
        integer1 = np.flip(-integrate.cumtrapz(np.flip(beta_mol[:ytop]),dx=range_resolution,initial=0))
        integrando = rcs[:ytop]*np.exp(-2*(lr_aer-lr_mol)*integer1)
        # integer3 = np.flip(-integrate.cumtrapz(np.flip(integrando),dx=range_resolution,initial=0))
        integer3 = np.flip(-integrate.cumtrapz(np.flip(integrando),dx=range_resolution,initial=0))
        particle_beta[:ytop] = (rcs[:ytop]*np.exp(-2*(lr_aer-lr_mol)*integer1))/(calib-2*lr_aer*integer3) - beta_mol[:ytop]
        
    else:
        print('ERROR: Range [ymin,ymax] out of rcs size.')
    return particle_beta

def quasi_backscatter_aerosol(beta_att, height, beta_mol, lr_mol, lr_aer, \
    alpha_aer_0=0.0, max_iterations=10, rms_threshold=1.0e-2, ymax_rms=4000, div_iterations=15, debugging=False):
    """
    
    Baars, H., Seifert, P., Engelmann, R. and Wandinger, U.: Target categorization of aerosol and clouds by continuous multiwavelength-polarization lidar measurements, Atmos. Meas. Tech., 10(9), 3175–3201, doi:10.5194/amt-10-3175-2017, 2017.
    CONTINUOUS PROFILES PLEASE


    """
    # num of points of profile
    n_p = len(beta_att)
    
    # Initial Guess for Aerosol Extinction Profile
    if isinstance(alpha_aer_0, float) or isinstance(alpha_aer_0, int):
        alpha_aer_0 = np.zeros(n_p) + alpha_aer_0
    elif isinstance(alpha_aer_0, np.ndarray) or isinstance(alpha_aer_0, list):
        if not len(alpha_aer_0) == n_p:
            print("error")
            return beta_att*np.nan
    else:
        print("error")
        return beta_att*np.nan

    # transmission molecular
    transm_mol = np.exp(-optical_depth(lr_mol*beta_mol, height))

    # transmission aerosol initial guess
    transm_aer_0 = np.exp(-optical_depth(alpha_aer_0, height))

    # iteration products
    alpha_aer_arr = []
    alpha_aer_arr.append(alpha_aer_0)
    transm_aer_arr = []
    transm_aer_arr.append(transm_aer_0)
    beta_aer_arr = []
    beta_aer_arr.append(np.zeros(n_p))
    rms_arr = []
    rms_arr.append(1)
    beta_att_arr = []
    beta_att_arr.append(beta_att)

    # iteration    
    end_loop = False
    converged = False
    diverged = 0
    i = 1
    transm_aer_it = transm_aer_0
    alpha_aer_it = alpha_aer_0    
    while not end_loop:        
        #print("Iteration %i-th" % i)

        """
        # beta aerosol
        beta_aer_it = (beta_att / (transm_aer_it*transm_mol)) - beta_mol

        # new transmission aerosol
        alpha_aer_it = lr_aer*beta_aer_it
        transm_aer_it = np.exp(-2*optical_depth(alpha_aer_it, height))

        # beta attenuated iteration
        beta_att_it = attenuated_backscatter(beta_mol, beta_aer_it, transm_mol, transm_aer_it)
        """

        logbeta = np.log(beta_att) + 2*optical_depth(lr_mol*beta_mol, height) + 2*optical_depth(alpha_aer_it, height)
        beta_aer_it = np.exp(logbeta) - beta_mol
        alpha_aer_it = lr_aer*beta_aer_it
        transm_aer_it = np.exp(-optical_depth(alpha_aer_it, height))
        beta_att_it = attenuated_backscatter(beta_mol, beta_aer_it, transm_mol, transm_aer_it)

        # normalised rms
        idx_max = np.abs(height-ymax_rms).argmin()
        rms = np.sqrt(np.nansum( (beta_att_it[:idx_max] - beta_att[:idx_max])**2 )/idx_max) / np.nanmean(beta_att[:idx_max])
        if debugging:
            print("iter: %i | RMS=%.2e" % (i, rms))
                
        # append iteration results
        # if debugging:
        #     alpha_aer_arr.append(alpha_aer_it)        
        #     transm_aer_arr.append(transm_aer_it)            
        #     beta_att_arr.append(beta_att_it)
        beta_aer_arr.append(beta_aer_it)
        rms_arr.append(rms)
        
        if debugging:
            print(np.shape(beta_aer_arr))

        # check convergence        
        if rms < rms_threshold:
            if debugging:
                print("CONVERGED: %.2e < %.2e" % (rms, rms_threshold))
            converged = True
        else:
            if i > 1:                      
                if np.logical_or.reduce((rms > rms_arr[i-1], np.isnan(rms), np.isinf(rms))):
                    diverged += 1
        if converged:
            end_loop = True
        if i >= max_iterations:
            end_loop = True
            #print("NON-CONVERGED: Max. Iterations Reached without convergence!!!")
        if diverged > div_iterations:
            end_loop = True
            #print("NON-CONVERGED: Method does not seem to converge!!!")
        i += 1
    
    # if debugging:
    #     alpha_aer_arr = np.array(alpha_aer_arr)
    #     transm_aer_arr = np.array(transm_aer_arr)        
    #     beta_att_arr = np.array(beta_att_arr)
    beta_aer_arr = np.array(beta_aer_arr)        
    rms_arr = np.array(rms_arr)
    
    if converged:
        beta_aer = beta_aer_it
    else:
        beta_aer = beta_aer_arr[-1, :]

    return beta_aer, converged, beta_aer_arr # beta_att_arr, alpha_aer_arr, transm_aer_arr, rms_arr)

def bottom_up_backscatter_aerosol(beta_att, height, beta_mol, lr_mol, lr_aer=45, fulloverlap_height=0):
    """
    bottom_up_backscatter_aerosol [summary]

    [extended_summary]

    Parameters
    ----------
    beta_att : [type]
        [description]
    height : [type]
        [description]
    beta_mol : [type]
        [description]
    lr_mol : [type]
        [description]
    lr_aer : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    def rel_diff(a,b):
        return (a-b)/b

    # definitions    
    height_res = np.diff(height).mean()
    n_z = len(height)
    alpha_aer = np.zeros(n_z)
    od_aer = np.zeros(n_z)
    beta_aer = np.zeros(n_z)
    t_aer = np.ones(n_z)
    r_diff = np.zeros(n_z)
    od_aer_calc = np.zeros(n_z)
    beta_att_calc = np.zeros(n_z)

    alpha_mol = beta_mol*lr_mol

    #Refill overlap
    beta_att = refill_overlap(beta_att,height, fulloverlap_height=fulloverlap_height)

    # initialize
    # od_mol = optical_depth(alpha_mol, height)
    od_mol = integrate.cumtrapz(alpha_mol, dx=height_res, initial=0)
    t_mol = np.exp(-od_mol)
    logbeta = np.log(beta_att[0]) + 2*od_aer[0] + 2*od_mol[0]
    beta_aer[0] = np.exp(logbeta) - beta_mol[0]
    alpha_aer[0] = lr_aer*beta_aer[0]
    beta_att_calc[0] = beta_att[0]
    od_aer[0] = alpha_aer[0]*height_res
    
    # loop bottom-up
    for i in np.arange(1,n_z):
        od_aer[i] = od_aer[i-1] + alpha_aer[i]*height_res
        t_aer[i] = np.exp(-od_aer[i])
        beta_aer[i] =  beta_att[i]/ (t_aer[i]*t_mol[i])**2 - beta_mol[i]
        alpha_aer[i] = lr_aer*beta_aer[i]
        # od_aer_calc[i] = optical_depth(alpha_aer[:i+1], height[:i+1])[i]
    beta_att_calc = (beta_mol + beta_aer)*np.exp(-2*(od_aer + od_mol))
        # r_diff[i] = rel_diff(np.log(beta_att[i]), np.log(beta_att_calc[i]))

    # beta_aer = xr.DataArray(beta_aer, dims=['range'], coords={'range': height})

    return beta_aer, beta_att_calc, od_aer #, (alpha_aer, t_aer, od_aer, od_aer_calc, beta_att_calc)


def compute_klett_retrieval(rcs, height, beta_mol, beta_aer_ref, lr_mol, lr_aer, h_ref_min, h_ref_max):
    """[summary]

    Parameters
    ----------
    rcs : [type]
        [description]
    height : [type]
        [description]
    beta_mol : [type]
        [description]
    beta_aer_ref : [type]
        [description]
    lr_mol : [type]
        [description]
    lr_aer : [type]
        [description]
    h_ref_min : [type]
        [description]
    h_ref_max : [type]
        [description]
    """
    
    # Get Reference Height Index
    idx_ref, ref_range = get_ref_height_index(height, h_ref_min, h_ref_max)
    h_ref = height[idx_ref]

    # Height Resolution
    height_res = np.diff(height).mean()

    # Klett Retrieval
    print('WARNING: DO NOT USE KLETT INVERSION FROM IOANIS''s CODE')
    beta_aer_klett = elastic_retrievals.klett_backscatter_aerosol(\
        rcs, lr_aer, beta_mol, idx_ref, ref_range, beta_aer_ref, height_res, lr_mol)
    
    return beta_aer_klett


def compute_quasi_backscatter_retrieval(beta_att, height, beta_mol, lr_mol, lr_aer, \
    snr=None, snr_threshold=1, alpha_aer_0=0.0, max_iterations=20, rms_threshold=1.0e-2, div_iterations=5):
    """[summary]

    Parameters:
    ----------
    beta_att : [type]
        [description]
    height : [type]
        [description]
    beta_mol : [type]
        [description]
    lr_mol : [type]:
        [description]
    lr_aer : [type]
        [description]
    alpha_aer_0 : float, optional
        [description], by default 0.0
    max_iterations : int, optional
        [description], by default 10
    rms_threshold : [type], optional
        [description], by default 1.0e-2
    div_iterations : int, optional
        [description], by default 3
    """

    n_z = len(height)

    # deal with negative values of beta att at first heights
    # supposing first levels must be > 0
    beta_att = negative_filter(beta_att)

    # compute beta_aer
    if snr is not None:
        ids = snr_filter(snr, height, threshold=snr_threshold)
        beta_aer_quasi, converged, retrieval_result = quasi_backscatter_aerosol(\
            beta_att[:ids], height[:ids], beta_mol[:ids], lr_mol, lr_aer, \
            alpha_aer_0=alpha_aer_0, max_iterations=max_iterations, \
                rms_threshold=rms_threshold, div_iterations=div_iterations)
        beta_aer_quasi = np.concatenate([beta_aer_quasi, np.zeros(n_z -ids)*np.nan])
        retrieval_result_raw = retrieval_result
        retrieval_result = ()
        for x in retrieval_result_raw:
            if x[0].size == ids:
                y = []
                for xx in x:
                    y.append(np.concatenate([xx, np.zeros(n_z - ids)*np.nan]))
                y = np.array(y)
                retrieval_result += (y,)
            else:
                retrieval_result += (x,)
    else:
        beta_aer_quasi, converged, retrieval_result = quasi_backscatter_aerosol(beta_att, height, beta_mol, lr_mol, lr_aer, \
            alpha_aer_0=alpha_aer_0, max_iterations=max_iterations, \
                rms_threshold=rms_threshold, div_iterations=div_iterations)

    return beta_aer_quasi, converged, retrieval_result


def compute_bottom_up_backscatter_retrieval(beta_att, height, beta_mol, lr_mol, lr_aer):
    """
    compute_bottom_up_backscatter_retrieval [summary]

    [extended_summary]

    Parameters
    ----------
    beta_att : [type]
        [description]
    height : [type]
        [description]
    beta_mol : [type]
        [description]
    lr_mol : [type]
        [description]
    lr_aer : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """

    # deal with negative values of beta att at first heights
    # supposing first levels must be > 0
    beta_att = negative_filter(beta_att)

    # Compute beta_aer
    beta_aer, retrieval_result = bottom_up_backscatter_aerosol(beta_att, height, beta_mol, lr_mol, lr_aer)
    return beta_aer, retrieval_result