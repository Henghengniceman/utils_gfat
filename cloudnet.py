#!/usr/bin/env python

"""
function for reading files from CLOUDNET

Categorice_reader: line 17
Classification_reader: line 65
cloudnetQuicklook: 131
"""

import os
import warnings
import numpy as np
import netCDF4 as nc
import xarray as xr
import datetime as dt
import matplotlib as mpl
import matplotlib.pyplot as plt
import logging
from . import utils
import pdb


__author__ = "Bravo-Aranda, Juan Antonio"
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Bravo-Aranda, Juan Antonio"
__email__ = "jabravo@ugr.es"
# __status__ = "Production"

""" logging """
log_format = '%(filename)s, L%(lineno)d [%(funcName)s]: %(message)s'
logging.basicConfig(format=log_format, level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.handlers.clear()
#logger.addHandler(logging.StreamHandler(sys.stdout))


CLOUDNET_KINDS = ["categorize", "classification"]


def get_cloudnet_fn(station_name, date_str, kind, cloudnet_dn=None):
    """ Get file name full path for a day cloudnet data

    Parameters
    ----------
        station_name: str
            name of station
        date_str: str
            date (day) YYYYMMDD
        kind: str
            categorize, classification
        cloudnet_dn: str
            absolute path for cloudnet data
    Returns
    -------
        cldnt_fn: str
            full path cloudnet file name
    """

    try:
        if cloudnet_dn is None:  # GFAT NAS
            cloudnet_dn = r"/mnt/NASGFAT/datos/CLOUDNET"

        if kind in CLOUDNET_KINDS:
            date_year = dt.datetime.strptime(date_str, '%Y%m%d').strftime('%Y')
            cldnt_fn = os.path.join(cloudnet_dn, station_name, kind, date_year,
                                    "%s_%s_%s.nc" % (date_str, station_name, kind))
            if not os.path.isfile(cldnt_fn):
                logger.error("File %s not found" % cldnt_fn)
                cldnt_fn = None
        else:
            logger.error("Cloudnet File kind %s does not exist" % kind)
            cldnt_fn = None
    except Exception as e:
        logger.error(str(e))
        cldnt_fn = None

    return cldnt_fn


def cloudnet_reader(cloudnet_fn, fields=None):
    """ Read Cloudnet File

    Parameters
    ----------
    cloudnet_fn: str
        full path of cloudnet file name
    fields: list
        list of variables within the cloudnet dataset

    Returns
    -------
    ds: xarray
        Cloudnet dataset
    """
    try:
        if os.path.isfile(cloudnet_fn):
            with xr.open_dataset(cloudnet_fn, chunks={}) as ds:
                if fields is not None:
                    ds = ds[fields]
        else:
            logger.error("File %s not found" % cloudnet_fn)
            ds = None
    except Exception as e:
        logger.error(str(e))
        ds = None

    return ds


def categorice_reader(list_files):
    """read categorice data from netCDF files

    Designed to work with netcdf list_files associated to one single day
    """
    logger.warning("to be deprecated")
    print('Reading netCDF files: %s' % list_files)    
    data = {}
    
    if list_files:
        var2load = ['lwp', 'Z', 'beta', 'mean_zbeta', 'temperature', 'v',
                    'pressure', 'specific_humidity', 'rainrate']
    
        # open all files
        nc_ids = [nc.Dataset(file_) for file_ in list_files]

        # localization of instrument
        data['lat'] = nc_ids[0].variables['latitude'][:]
        data['lon'] = nc_ids[0].variables['longitude'][:]
        data['alt'] = nc_ids[0].variables['altitude'][:]
        data['location'] = nc_ids[0].location
        
        # read alt (no need to concatenate)
        data['height'] = nc_ids[0].variables['height'][:]
        data['model_height'] = nc_ids[0].variables['model_height'][:]
    
        # wavelength (no need to concatenate)
        tmp = nc_ids[0].variables['lidar_wavelength'][:]
        data['wavelength'] = tmp 
        data['wavelength_units'] = nc_ids[0].variables['lidar_wavelength'].units  
    
        # read time 
        units = nc_ids[0].variables['time'].units  # HOURS SINCE %Y-%m-%d 00:00:00 +0:00
        tmp = [nc.num2date(nc_id.variables['time'][:], units) for nc_id in nc_ids]
        data['raw_time'] = np.concatenate(tmp)
    
        # check if any data available
        # print(date)
        # time_filter = (data['raw_time'] >= date) & (data['raw_time'] < date + dt.timedelta(days=1))
        # if not np.any(time_filter):
        #     return None    
        
        for var in var2load:
            tmp = [nc_id.variables[var][:] for nc_id in nc_ids]
            data[var] = np.ma.filled(np.concatenate(tmp, axis=0), fill_value=np.nan)
            # It is assumed that properties do not change along netcdf files for the same day
            data[var][data[var] == nc_ids[0][var].missing_value] = np.nan
               
        # Change name to fit real array content
        data['dBZe'] = data['Z'].copy()
        
        # close all files
        [nc_id.close() for nc_id in nc_ids]

    return data    


def classification_reader(list_files):
    """read data from netCDF files

    Designed to work with netcdf list_files associated to one single day
    """
    """
    0: Clear sky
    1: Cloud liquid droplets only
    2: Drizzle or rain
    3: Drizzle or rain coexisting with cloud liquid droplets
    4: Ice particles
    5: Ice coexisting with supercooled liquid droplets
    6: Melting ice particles
    7: Melting ice particles coexisting with cloud liquid droplets
    8: Aerosol particles, no cloud or precipitation
    9: Insects, no cloud or precipitation
    10: Aerosol coexisting with insects, no cloud or precipitation
    """
    logger.warning("to be deprecated")
    print('Reading netCDF files: %s' % list_files)

    data = {}
    
    if list_files:
        var2load = ['cloud_base_height', 'cloud_top_height', 'target_classification']
        # open all files
        nc_ids = [nc.Dataset(file_) for file_ in list_files]
    
        # localization of instrument
        data['lat'] = nc_ids[0].variables['latitude'][:]
        data['lon'] = nc_ids[0].variables['longitude'][:]
        data['alt'] = nc_ids[0].variables['altitude'][:]
        data['location'] = nc_ids[0].location
        
        # read alt (no need to concantenate)
        data['height'] = nc_ids[0].variables['height'][:]
    
        # read time 
        units = nc_ids[0].variables['time'].units #'days since %s' % dt.datetime.strftime(date, '%Y-%m-%d %H:%M:%S')
        tmp = [nc.num2date(nc_id.variables['time'][:], units) for nc_id in nc_ids]
        data['raw_time'] = np.concatenate(tmp)
    
        # check if any data available
        # print(date)
        # time_filter = (data['raw_time'] >= date) & (data['raw_time'] < date + dt.timedelta(days=1))
        # if not np.any(time_filter):
        #     return None    
        
        for var in var2load:        
            tmp = [nc_id.variables[var][:] for nc_id in nc_ids]
            data[var] = np.ma.filled(np.concatenate(tmp, axis=0), fill_value=np.nan)
            # It is assumed that properties do not change along netcdf files for the same day
            if 'missing_value' in nc_ids[0][var].ncattrs():
                data[var][data[var] == nc_ids[0][var].missing_value] = np.nan
                
        # close all files
        [nc_id.close() for nc_id in nc_ids]

    return data 


def CBH_attenuated(cat, threshold=2.5e-6, cloud_height_maximum=4000,plot_flag=False):
    """
    CBH_attenuated finds the CBH from attenuated backscatter in CLUODNET categorize files
    Input:
    cat: xarray from  CLUODNET categorize file    
    threshold: minimum value of cloud attenuated backscatter (default: 2.5e-6 m^1*sr^-1
    cloud_height_maximum: maximxum height at which CBH will be search (Default: 4000 m),
    plot_flag: it plots the CBH temporal evolution (Default: False)
    Output:
    cat: xarray categorize input but with CBH.
    """
    
    if plot_flag:
        fig, axes = plt.subplots(1,1)
        fig.set_figheight(15)
        fig.set_figwidth(15)
    top_idx, _ = utils.find_nearest_1d(cat.height.values, cloud_height_maximum)
    height = cat.height[0:top_idx].values

    CBH = np.nan*np.ones(len(cat.time))
    for i_ in np.arange(len(cat.time)):
        profile = cat.beta[i_,0:top_idx].values
        try:
            candidate = height[profile>threshold][0]
        except:
            candidate = height[profile>threshold]
        if candidate < cloud_height_maximum:
            CBH[i_] = candidate
            if plot_flag:
                axes.plot(profile, height)
    if plot_flag:
        axes.vlines(threshold,0,cloud_height_maximum)
        plt.xlim(0,threshold*3)            
    
    cat['CBH_attenuated_beta'] = ('time',CBH)
    
    return cat

def cloud_edges(cat, threshold_split = 200):
    """
    CTH_finder finds the CTH from radar reflectivity in CLUODNET categorize files
    Input:
    cat: xarray from CLUODNET categorize file        
    cloud_height_maximum: maximxum height at which CTH will be search (Default: 5000 m),    
    Output:
    cat: xarray categorize input but with CTH.
    """     
    cat['Zboolean'] = cat.Z.where(np.isnan(cat.Z),other=1)
    cat['Zboolean'] = cat.Zboolean.where(~np.isnan(cat.Z),other=0)
    cat['Zbooleandiff_CBH'] = (('time', 'height'),np.diff(cat.Zboolean, axis=1, prepend=0))
    cat['Zbooleandiff_CTH'] = (('time', 'height'),np.diff(cat.Zboolean, axis=1, prepend=0))

    #CBH candidates    
    final_CBH = np.nan*np.ones(np.shape(cat['Z']))
    final_CTH = np.nan*np.ones(np.shape(cat['Z']))
    CBH0 = np.nan*np.ones(len(cat['time']))
    CTH0 = np.nan*np.ones(len(cat['time']))
    single_layer = np.ones(len(cat['time']))
    idx_height = np.arange(len(cat['height']))
    for idx_ in np.arange(len(cat['time'])):
        #CBH candidates    
        booleanCBH = cat['Zbooleandiff_CBH'][idx_,:]==1
        CBHcandidates = cat['height'].where(booleanCBH)    
        #CTH candidates
        booleanCTH = cat['Zbooleandiff_CTH'][idx_,:]==-1        
        CTHcandidates = cat['height'].where(booleanCTH)    

        #Glue clouds too near
        CBHs = CBHcandidates[booleanCBH]
        CTHs = CTHcandidates[booleanCTH]
        if len(CBHs)>0:
            if np.logical_and.reduce((len(CBHs) > 1, len(CBHs) == len(CTHs))):
                distance = CBHs.values[1:] - CTHs.values[:-1]
                if np.logical_and.reduce((len(distance) > 0, distance.any())):
                    idx_badDistance = distance < threshold_split #CTH(n) and CBH(n+1) should be far enough to be consider different cloud
                    goodCBH = CBHs[np.append(True, ~idx_badDistance)]
                    goodCTH = CTHs[np.append(~idx_badDistance,True)]            
                    if np.logical_and.reduce((goodCBH.any(), goodCTH.any())):
                        for CBH_ in goodCBH:
                            final_CBH[idx_,idx_height[cat['height']==CBH_]] = CBH_
                        for CTH_ in goodCTH:
                            final_CTH[idx_,idx_height[cat['height']==CTH_]] = CTH_
                    CBH0[idx_] =final_CBH[idx_,:][~np.isnan(final_CBH[idx_,:])][0]
                    CTH0[idx_] = final_CTH[idx_,:][~np.isnan(final_CTH[idx_,:])][0]
                single_layer[idx_] = 0
            else:            
                try:
                    CBH0[idx_] = CBHs
                    CTH0[idx_] = CTHs
                except:
                        pdb.set_trace()
    cat['CBH'] = (('time', 'height'), final_CBH)    
    cat['CTH'] = (('time', 'height'), final_CTH)    
    cat['CBH_first'] = (('time'), CBH0)    
    cat['CTH_first'] = (('time'), CTH0)
    cat['cloud_depth'] = (('time'), CTH0-CBH0)
    cat['single_layer'] = (('time'), single_layer)
    return cat

def CTH_finder(cat, cloud_height_maximum=5000):    
    """
    CTH_finder finds the CTH from radar reflectivity in CLUODNET categorize files
    Input:
    cat: xarray from CLUODNET categorize file        
    cloud_height_maximum: maximxum height at which CTH will be search (Default: 5000 m),    
    Output:
    cat: xarray categorize input but with CTH.
    """    
    top_idx, _ = utils.find_nearest_1d(cat.height.values, cloud_height_maximum)
    height = cat.height[0:top_idx].values
    CTH = np.nan*np.ones(len(cat.time))
    for i_ in np.arange(len(cat.time)):
        profile = cat.Z[i_,0:top_idx].values
        CBH = cat.CBH[i_]
        if np.logical_and(~np.isnan(CBH),(~np.isnan(profile)).any()):
            CTH[i_] = height[~np.isnan(profile)][-1]
            if CTH[i_] <= CBH:
               CTH[i_] = np.nan 
            
    cat['CTH'] = ('time',CTH)
    
    return cat

def filter_Z(cat, cbh_threshold=2.5e-6, plot_flag=False):        
    """
    filter_Z cleans the Z matrix of low aerosol layers. It requires CLUODNET categorize files with CBH. 
    Input:
    cat: xarray from CLUODNET categorize file        
    cbh_threshold: attenuated beta threshold value to detect the CBH using cloudnet.CBH_attenuated()
    
    Output:
    cat: xarray categorize input but with cleaned Z.
    """    
    
    if not 'CBH' in cat:
        cat = CBH_attenuated(cat, threshold=cbh_threshold)        
        
    cat['Z'] = cat.Z.where(cat.height > cat.CBH_attenuated_beta)
    if plot_flag:            
        fig, axes = plt.subplots(1,1)
        fig.set_figheight(4)
        fig.set_figwidth(15)
        cmap = mpl.cm.jet        
        cat.Z.where(cat.height > cat.CBH_first).plot(x='time',cmap=cmap,vmin=-50,vmax=20, ax= axes)
        cat.CBH_first.plot(c='r',ax=axes)
    return cat

def plotQuicklook(data, plt_conf, saveImageFlag):
    logger.warning("to be deprecated. Use cloudnetpy instead")
    """
    Inputs: 
    - data: from categorice_reader()
    - plt_conf: plot configuration dictionary as follows:
        plt_conf =  {
            'mainpath': "Y:\\datos\\CLOUDNET\\juelich\\quicklooks",
            'coeff': COEFF,
            'gapsize': HOLE_SIZE,
            'dpi': dpi,
            'fig_size': (16,5),
            'font_size': 16,
            'y_min': 0,
            'y_max': range_limit,
            'rcs_error_threshold':1.0, }
    - saveImageFlag [Boolean]: to save png-figure or print in command screen.
    """          
    var2plot = {0: 'dBZe', 1: 'v', 2: 'beta'} #, 2: 'sigma' , 3: 'sigma', 4: 'kurt'                        
    #Dictionary for the vmax and vmin of the plot
    Vmin = {0: -55, 1: -1.5, 2:0} # , 2: 0 , 3: -3, 4: -3
    Vmax = {0: -20, 1: 1.5, 2: 10} #, 2: 5 , 3: 3, 4: 3
    Vn = {0: 16, 1: 7, 2: 10}
    scale = {0: 1, 1: 1, 2: 1e6}
    titleStr = {0: 'Reflectivity', 1: 'Vertical mean velocity', 2: 'Backscatter coeff.'} #, 2: 'spectral width'
    cblabel = {0: '$Z_e, dBZe$', 1: '$V_m, m/s$', 2: r'$\beta$, $Mm^-1$'} #, 2: 'spectral width'
    datestr = data['raw_time'][0].strftime('%Y%m%d')
    for idx in var2plot.keys():            
        _var = var2plot[idx]
        #print(idx)
        print(_var)        
        _fig, _axes = plt.subplots(nrows=1, figsize=(15,6))
        _axes.set_facecolor('whitesmoke')
        cmap = mpl.cm.jet
        bounds = np.linspace(Vmin[idx],Vmax[idx], Vn[idx])
#        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        range_km = data['height']/1000.
        q = _axes.pcolormesh(data['raw_time'], range_km, scale[idx]*data[_var].T,
                            cmap=cmap,
                            vmin=Vmin[idx],
                            vmax=Vmax[idx],)
        q.cmap.set_over('white')
        q.cmap.set_under('darkblue')
        cb = plt.colorbar(q, ax=_axes,
                          ticks=bounds,
                          extend='max')
        cb.set_label(cblabel[idx])        
        _axes.set_xlabel('Time, UTC')
        _axes.set_ylabel('Height, Km asl')
        _axes.set_title('%s on  %s' % (titleStr[idx], datestr))
        xmin = dt.datetime.strptime(datestr, '%Y%m%d')
        xmax = xmin + dt.timedelta(days=1)
        _axes.set_xlim(xmin, xmax)
        _axes.set_ylim(0, 2)
        if saveImageFlag:            
            figstr = '%s_%s_%s.png' % ('cloudnet', var2plot[idx], datestr)
            finalpath = os.path.join(plt_conf['mainpath'], _var, figstr)              
            print('Saving %s' % finalpath)
            final_dir_path = os.path.split(finalpath)[0]
            if not os.path.exists(final_dir_path):
                os.makedirs(final_dir_path)
            plt.savefig(finalpath, dpi=100, bbox_inches='tight')
            if os.path.exists(finalpath):
                print('Saving %s...DONE!' % finalpath)
            else:
                print('Saving %s... error!' % finalpath)
            plt.close(_fig)
        else:
            plt.show()


def plotLWP(data, mainpath, saveImageFlag):   
    """
    Inputs: 
    - data: from categorice_reader()
    - mainpath: "Y:\\datos\\CLOUDNET\\juelich\\quicklooks",
    - saveImageFlag [Boolean]: to save png-figure or print in command screen.
    """
    logger.warning("to be deprecated. Use cloudnetpy instead")
    datestr = data['raw_time'][0].strftime('%Y%m%d')
    fig, axes = plt.subplots(nrows=1, figsize=(15,6))
    axes.set_facecolor('whitesmoke')
    plt.plot(data['raw_time'], data['lwp'])
    axes.set_xlabel('Time, UTC')
    axes.set_ylabel('LWP, $g/m^2$')
    axes.set_title('LWP on  %s' % datestr)
    xmin = dt.datetime.strptime(datestr, '%Y%m%d')
    xmax = xmin + dt.timedelta(days=1)
    axes.set_xlim(xmin, xmax)
    if saveImageFlag:
        figstr = '%s_%s_%s.png' % ('cloudnet', 'lwp', datestr)
        finalpath = os.path.join(mainpath, 'LWP', figstr)              
        print('Saving %s' % finalpath)
        final_dir_path = os.path.split(finalpath)[0]
        if not os.path.exists(final_dir_path):
            os.makedirs(final_dir_path)
        plt.savefig(finalpath, dpi=100, bbox_inches='tight')
        if os.path.exists(finalpath):
            print('Saving %s...DONE!' % finalpath)
        else:
            print('Saving %s... error!' % finalpath)
        plt.close(fig)
    else:
        plt.show()

def HM_model_retrieval(Z_linear, delta_h, model='knist_N', lwp=[], nu=8.7, sigma = 0.29, CDNC = None):
    """
    HM (Cloud Model): Radar Reflectivity-Homogeneous Mixing
    
        Inputs:
    - Z_linear, 1D array (dentro de la nube)     
    - delta_h: Resolution (m)
    - model: approach used. Options: 'Knist' [Default] | 'frisch'
    - nu: parameter to determine the shape of the log normal distrubion [Required for Knist''s approach]
    - sigma: parameter to determine the width of the the log normal distrubion [Required for Frisch''s approach]
    - lwp: Liquid Water Path, Scalar [Required for Knist''s approach]
    - CDNC: cloud-droplet number concenctration [cm^{-3}] [Required for Frisch''s approach]
        Outputs
    - re: Effective radius. 1D array. (in um)
    - N: Droplet concentration. Scalar. (in cm-3). It is CDNC for Frisch''s approach
    - Opt_Depth: Optical Depth (Adimensional). It is empty for Frisch''s approach    
    """    
    if model =='knist_lwp':        
        if len(lwp)>0:        
            WATER_DENSITY = 1e6  # g/m3
            kre = np.power(((nu+2)**3) / ((nu+3)*(nu+4)*(nu+5)), 1./3.)
            knt = ((nu+3)*(nu+4)*(nu+5)) / (nu*(nu+1)*(nu+2))

            Z_linear_root = np.sqrt(Z_linear)
            sumZroot = Z_linear_root.sum(dim='height',skipna=True,min_count=1)
            # effective radius in um
            try:
                re = kre*(Z_linear**(1./6.))*np.power((np.pi*WATER_DENSITY*sumZroot*delta_h)/(48*lwp), 1./3.) #(Eq. 2.57 Knist PhD dissertation) Identical to Frisch et al. 2002 with sigma_x = 0.29 (value from marine Sc, Martin et al 1994)                 
                re *= 1e6
                #re = substitute_nan(re)
            except:
                re = np.empty(Z_linear.shape)
                re[:] = np.nan
            # droplet concentration in cm-3
            try:
                #N = knt*((6*lwp)/(np.pi*WATER_DENSITY*delta_h*np.nansum(np.sqrt(Z_linear))))**2 #Andrea's version                
                N = knt*((6*lwp)/np.power(np.pi*WATER_DENSITY*delta_h*sumZroot,2))
                N *= 1e-6        
            except:
                N = np.nan        
            #COD
            try:                 
                Opt_Depth=(3./(2.*kre)) * ((48/np.pi)**(1./3.)) *(lwp/np.power(WATER_DENSITY*delta_h*sumZroot),4./3.)*delta_h*np.power(sumZroot,1./3.)
            except:
                Opt_Depth = np.nan
        else: 
            print('ERROR: Knist''s approximation requires lwp array.')        
    elif model =='knist_N':
        if CDNC:
            WATER_DENSITY = 1e6  # g/m3
            kre = np.power( np.power(nu+2,5) / (nu*(nu+1)*(nu+3)*(nu+4)*(nu+5)) , 1./6.)                        
            # effective radius in um
            try:
                re = 0.5*kre*np.power(Z_linear/(CDNC*1e6),1./6.) #(Eq. 2.56 Knist PhD dissertation) Identical to Frisch et al. 2002 with sigma_x = 0.29 (value from marine Sc, Martin et al 1994)
                re *= 1e6                
            except:
                re = np.empty(Z_linear.shape)
                re[:] = np.nan
            # droplet concentration in cm-3
            N = CDNC*np.ones(len(re))
            #COD
            Opt_Depth = np.nan*np.ones(len(re))
        else: 
            print('ERROR: Knist_N approximation requires CDNC.')        
    elif model =='frisch':
        if CDNC:            
            Opt_Depth = []
            # effective radius in um
            try:                
                re = 0.5*np.exp(-0.5*sigma**2)*np.power(Z_linear/(CDNC*1e6),1./6.) #(Frisch et al., 2002)
                #re = 0.5*np.power(Z_linear/(CDNC*1e6),1./6.) #(Frisch et al., 2002)
                #re = 0.5*np.exp(-0.5*sigma**2)*np.power(Z_linear/(CDNC*1e6),1./6.) #(Frisch et al., 2002)
                #re = 0.5*sigma*np.power(Z_linear/(CDNC*1e6),1./6.) #(Frisch et al., 2002)
                re *= 1e6
                #re = substitute_nan(re)
            except:
                re = np.empty(Z_linear.shape)
                re[:] = np.nan                        
            N = CDNC*np.ones(len(re))
        else: 
            print('ERROR: Frisch''s approximation requires lwp array.')
    return re, N, Opt_Depth

def LWC_model_HM(Z_linear, LWP, delta_h):
    """
    HM (Cloud Model): Radar Reflectivity-Homogeneous Mixing    
    Inputs:
    - Z_linear, 1D array (dentro de la nube)             
    - lwp: Liquid Water Path, Scalar [Required for Knist''s approach]
    - delta_h: Resolution (m)
        Outputs
    - LWC: Liquid Water Content. 1D array. (in g/m^2)
    """        

    #Retrieval LWC
    Z_linear_root = np.sqrt(Z_linear)
    sumZroot = Z_linear_root.sum(dim='height',skipna=True,min_count=1)
    LWC = LWP *(Z_linear_root/(sumZroot*delta_h))
    return LWC

def VU_model_retrieval(Z_avg_linear, lwp, delta_H, nu=8.7):
    """
    Vertical Uniform Model.
    Z_avg_linear, lwpi, delta_H: arrays of same size

        Inputs:
    - Z_avg_linear: Average Z (m^3)
    - lwp: Liquid Water Path (g m-2)
    - delta_H: cloud height (m)
    - nu: parameter default 8.7 (Knist, 2014)
        Outputs:
    - re: Effective radius (in um)
    - N: Droplet concentration (in cm-3)
    - Opt_Depth: Optical Depth (Adimensional)

    """

    WATER_DENSITY = 1e6  # g/m3
    kre = ((nu+2)**3 / ((nu+3)*(nu+4)*(nu+5)))**(1./3.)
    knt = ((nu+3)*(nu+4)*(nu+5) / (nu*(nu+1)*(nu+2)))

    # effective radius in um
    try:
        re = kre*((np.pi*WATER_DENSITY*delta_H*Z_avg_linear)/(48*lwp))**(1./3.)
        re *= 1e6
    except:
        re = np.nan

    # droplet concentration in cm-3
    try:
        N = (knt/Z_avg_linear)*((6*lwp)/(np.pi*WATER_DENSITY*delta_H))**2
        N *= 1e-6
    except:
        N = np.nan

    # optical depth (adimensional)
    try:
        Opt_Depth = (3./2.)*(lwp/(WATER_DENSITY*(re/1000000)))
        
    except:
        Opt_Depth = np.nan
        
    return re, N, Opt_Depth

def SAS_model_retrieval (Z_avg_linear, lwp, height, delta_H, nu=8.7):
    
    """
    SCALED ADIABATIC STRATIFIED CLOUD MODEL
        Inputs:
    - Z_avg_linear:  Mean (linear) Z in cloud for each profile 
    - lwp: values of lwp for each selected profile (filtering function). 
    - height: heights in cloud for each profile
    - delta_H: Total height (cth.values-cbh.values)
    - nu: parameter
    
        Outputs:
    - re: Effective radius (in um)
    - N: Droplet concentration (in cm-3)
    - Opt_Depth: Optical Depth (adimensional)
    
    """
    
    WATER_DENSITY = 1000000  # g/m3
    kre = ((nu+2)**3 / ((nu+3)*(nu+4)*(nu+5)))**(1./3.)
    knt = ((nu+3)*(nu+4)*(nu+5) / (nu*(nu+1)*(nu+2)))
    
     # effective radius in um
    
    try:
        re = height**(1./3.)*kre*((np.pi*WATER_DENSITY*Z_avg_linear)/(32*lwp))**(1./3.)#um
        re *= 1e6
    except:
        re = np.nan
    
    # Maximum radius
    re_max = np.nanmax(re)
    
    #Optical_Depth (adimensional)
    try:
        Opt_Depth = (9./5.)*(lwp/(WATER_DENSITY*(re_max/1000000)))
    except:
        Opt_Depth = np.nan
    
    # droplet concentration in cm-3
    try:
        N = (4./3.)*(knt/Z_avg_linear)*((6*lwp)/(np.pi*WATER_DENSITY*delta_H))**2
        N *= 1e-6
    except:
        N = np.nan
                
    return re, N, Opt_Depth

def compute_avg_cloud(Z, v, height, cbh, cth, avg_cloud=-1):
    """
    avg: calcula el promedio en el tramo que estamos indicando. 
    """
    if avg_cloud == -1:  # Average Over Cloud
        idc = np.logical_and(height > cbh, height < cth)
        Z_avg = np.nanmean(Z[idc])
        v_avg = np.nanmean(v[idc])
    elif avg_cloud > 0:  # Average Over CBH and a height above CBH (and below
        # CTH)
        _, hi = utils.find_nearest_1d(height, cbh + avg_cloud)
        idx = np.logical_and.reduce((height >= cbh, height <= hi,
                                     height <= cth))
        Z_avg = np.nanmean(Z[idx])
        v_avg = np.nanmean(v[idx])
    else:
        print("proxy_cloud must be -1 or >0")
        Z_avg = Z*np.nan
        v_avg = v*np.nan

    return Z_avg, v_avg
       
def ZdB_to_Z(Z_avg):
    """
    Z (dBZ) to linear Z
    Z_linear = 10**(Z_log/10) x 1e-18 [mm6/m3 -> m6/m3 ]

        Inputs:
    - Z_avg: units mm^6/m^3

        Outputs:
    - Linear Z: units m^3
    """

    return 10**(Z_avg/10-18)

def filtering(dx_in, liquid_drop=True, cbh_range=[450, 4000],
              lwp_range=[50, 150], lwp_rel_error_threshold=50,
              similar_weather_conditions=True, persistence=True,
              force_liquid_drop_incloud=True, preserve_time=False,
              cbh_weather_conditions=None):
    
    """

        Inputs:
    - dx_in: categorize-classification file (merge)
    - liquid_drop: 
    - cbh_range: heights where the cbh can be situated
    - lwp_range: range of values for the lwp
    - lwp_rel_error_threshold: error for lwp
    - force_liquid_drop_incloud:
    - preserve_time: 
    - cbh_weather_conditions: 

        Outputs:
    - dx_out: files that satify the conditions
    """
    # filtrado
    # - nubes liquidas
    # - rango cbh
    # - lwp>0 (e_lwp)
    # - condiciones meteo
    # - persistencia 30min
    # - si despues de filtrar, quedan menos de 10 perfiles, elimino el dia

    t = pd.to_datetime(dx_in.time.values)
    #print(t)
    h = dx_in.height.values
    #print(h)
    mh = dx_in.model_height.values
    #print(mh)
    tc = dx_in.target_cla.values
    #print(tc)
    cbh = dx_in.cbh.values
    #print(cbh)
    cth = dx_in.cth.values
    #print(cth)
    lwp = dx_in.lwp.values
    #print(lwp)
    elwp = dx_in.lwp_error.values
    #print(elwp)
    tk = dx_in.temperature.values
    #print(tk)
    p = dx_in.pressure.values
    #print(p)
    q = dx_in.specific_humidity.values
    #print(q)
    rr = dx_in.rainrate.values
    #print(rr)

    # ADD Tk, P, Q at CLOUD BASE HEIGHT
    tk_cbh, p_cbh, q_cbh = weather_conditions_at_cbh(cbh, mh, tk, p, q)
    dx_in['tk_cbh'] = (['time'], tk_cbh)
    dx_in['p_cbh'] = (['time'], p_cbh)
    dx_in['q_cbh'] = (['time'], q_cbh)

    # index for filtering
    idx = []
    try:
        # LIQUID DROPS
        # For each profile (time), filtering is performed
        # Use TARGET CLASSIFICATION to filter out all profiles that have
        # water components other than liquid drops that may contribute to LWP:
        # drizzle, rain (2,3), ice+liquid drops (5), melting ice (6), melting
        # ice + liquid drops (7)
        # In addition, only profiles with, at least, one value for liquid
        # drop, are considered.
        idx_ld = np.ones(t.shape[0])
        if liquid_drop:  # select profiles
            for i, _t in enumerate(t):
                x = tc[:, i]
                if np.logical_and.reduce(((not np.logical_or.reduce((
                        x == 2, x == 3, x == 5, x == 6, x == 7)).any())
                                          and ((x == 1).any()))):
                    idx_ld[i] = 1
                else:
                    idx_ld[i] = 0
                del x
        # In addition, ensure no precipitation with rainrate variable
        idx_ld[rr > 0] = 0
        
        # MODIFY CTH



        # PERSISTENCE: if liquid_drop == TRUE
        # Se buscan perfiles consecutivos que satisfacen la condicion
        # de gota liquida durante, al menos, media hora.
        # Aflojo un poco la condicion y permito perfiles no consecutivos si
        # el lapso entre 2 no consecutivos es menor que 5 minutos.
        # obviamente, los perfiles que se intercalan entre estos dos no
        # consecutivos que, por razones obvias, no satisfacen la condición de
        # gota líquida, no se incluyen
        idx_per = idx_ld.copy()
        if persistence and liquid_drop:
            count = 1
            lp = []
            while count < t.shape[0]:
                # print("count=%i"%count)
                cp = count
                _lp = []
                if idx_ld[cp] == 1:  # si liquid drop
                    # print("comienza subset")
                    si = 1
                    last_1 = cp
                    # print("last_1=%i"%last_1)
                    while (si == 1) and (cp < t.shape[0]):  # Liquid Drop
                        if idx_ld[cp] == 1:  # si liquid drop
                            _lp.append(cp)
                            cp += 1
                            last_1 = cp
                            # print("update last_1=%i" % last_1)
                        else:  # si no liquid drop
                            if (t[cp] - t[last_1]) < np.timedelta64(300, 's'):
                                cp += 1
                            else:
                                si = 0
                    # print("termina subset")
                    lp.append(_lp)
                    count = cp + 1
                else:  # avance
                    # print("avance")
                    count += 1

            # Los subconjuntos generados deben durar, al menos, 10 minutos
            for _lp in lp:
                t_lapse = t[_lp[-1]] - t[_lp[0]]
                if t_lapse < np.timedelta64(10, 'm'):
                    idx_per[_lp] = 0

        # CLOUD BASE HEIGHT RANGE
        idx_cbh = np.ones(t.shape[0])
        if isinstance(cbh_range, list):
            if len(cbh_range) == 2:
                idx = ~np.logical_and(cbh >= cbh_range[0], cbh <= cbh_range[1])
                idx_cbh[idx] = 0

        # LWP, LWP_ERROR THRESHOLDS
        idx_lwp = np.ones(t.shape[0])
        if isinstance(lwp_range, list):
            if len(lwp_range) == 2:
                idx = ~np.logical_and.reduce((lwp >= lwp_range[0],
                                              lwp <= lwp_range[1],
                                              100*elwp/lwp <
                                              lwp_rel_error_threshold))
                idx_lwp[idx] = 0

        # ONLY LIQUID DROP INSIDE THE CLOUD
        idx_ld_incloud = np.ones(t.shape[0])
        if force_liquid_drop_incloud:
            for i, _t in enumerate(t):
                x = tc[:, i]
                ix = np.logical_and(h >= cbh[i], h <= cth[i])
                if x[ix].any():  # Existen datos incloud
                    if not (x[ix][1:] == 1).all():
                        idx_ld_incloud[i] = 0
                else:
                    idx_ld_incloud[i] = 0

        # SIMILAR METEO CONDITIONS
        # usamos los perfiles que satisfacen las condiciones de gota liquida
        idx_met = np.ones(t.shape[0])
        if similar_weather_conditions:
            xx = idx_ld == 1
            idx = ~ np.logical_and(
                np.nanstd(tk_cbh[xx])/np.nanmean(tk_cbh[xx]) < 0.1,
                np.nanstd(p_cbh[xx])/np.nanmean(p_cbh[xx]) < 0.1)
            idx_met[idx] = 0

        # similar meteo conditions based on whole period
        # TODO: investigar si esta condicion es interesante
        """
        if cbh_weather_conditions is not None:
            # Tk, P, Q avg, sd at CBHs
            tk_cbh_avg, tk_cbh_sd = cbh_weather_conditions[0]
            p_cbh_avg, p_cbh_sd = cbh_weather_conditions[1]
            q_cbh_avg, q_cbh_sd = cbh_weather_conditions[2]

            #
            tk = dx_in.temperature.values
            p = dx_in.pressure.values
            q = dx_in.specific_humidity.values
            tk_cbh, p_cbh, q_cbh = weather_conditions_at_cbh(cbh, mh, tk, p, q)
        """

        # FILTER BY IDX
        idx = idx_ld * idx_per * idx_cbh * idx_lwp * idx_ld_incloud * idx_met
        # Hay, al menos, 10 elementos
        if len(np.argwhere(idx == 1)) >= 10:
            dx_out = dx_in.sel(time=dx_in.time[idx == 1])

            # DROP NON-NECESSARY VARIABLES
            dx_out = dx_out.drop(['temperature', 'pressure',
                                  'specific_humidity', 'model_height'])
            # REINDEX TO ORIGINAL TIME
            if preserve_time:
                dx_out = dx_out.reindex({'time': dx_in.time})
        else:
            dx_out = None
    except:
        print("SOMETHING WENT WRONG IN FILTERING")
        dx_out = None
        
    return dx_out
