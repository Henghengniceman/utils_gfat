#!/usr/bin/env python
"""
Functions to retrieve ACI indexes.
- filterCloudLiquid
- meanZ
- upsampler
- cloudMicrophysics
- aci
"""
import os
#import sys
#import glob
#import xarray as xr
import numpy as np
import pandas as pd
#import netCDF4 as nc
import datetime as dt
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt

__author__ = "Bravo-Aranda, Juan Antonio"
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Bravo-Aranda, Juan Antonio"
__email__ = "jabravo@ugr.es"
__status__ = "Production"


def filterCloudLiquid(cat_data, clas_data, debugger, debugger_saveImageFlag):
    """
    Inputs: 
    - cat_data: categorice data from categorice_reader()
    - clas_data: categorice data from classification_reader()
    """
    #Cloud liquid droplets only
    #Copy the data dictionary
    cat_data['Zcl'] = cat_data['Z'].copy()
    #Finde CBH and CTH
    clas_data['CBH'] = clas_data['cloud_base_height'].copy()
    clas_data['CTH'] = clas_data['cloud_top_height'].copy()
    #For each profile (time)
    for idx in np.arange(clas_data['raw_time'].shape[0]):
        #Targer classification in the profile
        column=clas_data['target_classification'][idx,:]        
#         if idx == 100:        
#             plt.figure()
#             plt.plot(clas_data['height'], column)
#             plt.show()
        if np.logical_or(column == 2, column == 3).any():    
#             cat_data['Zcl'][idx,:] = np.nan
            clas_data['CBH'][idx] = np.nan
            clas_data['CTH'][idx] = np.nan
            cat_data['lwp'][idx] = np.nan
        if not (column == 1).any():
            clas_data['CBH'][idx] = np.nan
            clas_data['CTH'][idx] = np.nan
            cat_data['lwp'][idx] = np.nan    
    cat_data['Zcl'][clas_data['target_classification']!=1] = np.nan
    debugger = True
    if debugger:
        #Only liquid water clouds
        Vmin = {0: -55} #, 1: -3, 2:0 , 2: 0 , 3: -3, 4: -3
        Vmax = {0: 20} # , 1: 3, 2: 0.9, 2: 5 , 3: 3, 4: 3
        Vn = {0: 16} #, 1: 7, 2: 10
        scale = {0: 1} #, 1: 1, 2: 1e6
        titleStr = {0: 'Filtered Liquid-Cloud Reflectivity'} #, 1: 'Vertical mean velocity', 2: 'Backscatter coeff.', 2: 'spectral width'
        cblabel = {0: '$Z_e, dBZe$'} #, 1: '$V_m, m/s$', 2: r'$\beta$, $Mm^-1$', 2: 'spectral width'

        idx = 0
        var = 'Zcl'
        fig, axes = plt.subplots(nrows=1, figsize=(15,6))
        cmap = mpl.cm.jet
        bounds = np.linspace(Vmin[idx],Vmax[idx], Vn[idx])
#        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        range_km = cat_data['height']/1000.
        q = axes.pcolormesh(cat_data['raw_time'], range_km, scale[idx]*cat_data[var].T,
                            cmap=cmap,
                            vmin=Vmin[idx],
                            vmax=Vmax[idx],)
        q.cmap.set_over('white')
        q.cmap.set_under('darkblue')
        cb = plt.colorbar(q, ax=axes,
                        ticks=bounds,
                        extend='max')
        cb.set_label(cblabel[idx])

        axes.set_xlabel('Time, UTC')
        axes.set_ylabel('Height, Km asl')        
        datestr = cat_data['raw_time'][0].strftime('%Y%m%d')
        axes.set_title('%s on  %s' % (titleStr[idx] , datestr))
        axes.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
        xmin = dt.datetime.strptime(datestr, '%Y%m%d')
        xmax = xmin + dt.timedelta(days=1)
        axes.set_xlim(xmin,xmax)
        axes.set_facecolor('whitesmoke')
        if debugger_saveImageFlag:            
            figstr = '%s_%s_%s_%s.png' % ('cloudnet', 'jue', 'Zcl', datestr)                        
            mainpath = os.path.join('Y:\\datos\\CLOUDNET', 'juelich', 'aci')
            finalpath = os.path.join(mainpath, figstr)              
            print('Saving %s' % finalpath)
            plt.savefig(finalpath, dpi=100, bbox_inches = 'tight')
            if os.path.exists(finalpath):
                print('Saving %s...DONE!' % finalpath)
            else:
                print('Saving %s... error!' % finalpath)
            plt.close(fig)
        else:
            plt.show()
    return cat_data

def meanZ(data, CBH, CTH):
    """
    Inputs:
    - data: categorice data from filterCloudLiquid() and categorice_reader()
    - CBH: Cloud Base Height from categorice-data from classification_reader()
    - CTH: Cloud Top Height from categorice-data from classification_reader()
    """    
    #Define new variables
    data['meanZ'] = np.ones(np.shape(data['raw_time']))*np.nan
    data['meanv'] = np.ones(np.shape(data['raw_time']))*np.nan
    data['mean_beta'] = np.ones(np.shape(data['raw_time']))*np.nan
    #Making average by profile
    for idx, _CBH in enumerate(CBH):        
        if not np.isnan(_CBH):            
            #Cloud Top Heigth
            _CTH = CTH[idx]        
            #Find cloud region
            idx_Z = np.logical_and(data['height']>_CBH, data['height']<_CTH)            
            
            #Z mean
            if ~(np.isnan(data['Z'][idx, idx_Z])).all(): #If not all values are NaN
                data['meanZ'][idx] = np.nanmean(data['Z'][idx, idx_Z])        
                data['meanv'][idx] = np.nanmean(data['v'][idx, idx_Z])        
                        
            #Top aerosol region
            top_limit = _CBH - 300
            #Bottom aerosol region
            bottom_limit = top_limit - 200            
            idx_beta = np.logical_and(data['height']>bottom_limit, data['height']<top_limit )
            #beta Mean
            if (~np.isnan(data['beta'][idx, idx_beta])).any(): #If not all values are NaN
                data['mean_beta'][idx] = np.nanmean(data['beta'][idx, idx_beta])               
    return data

def upsampler(data, vars2resample, timeStep, limit4AvoidNan):
    """
    Inputs:
    - data: categorice data from categorice_reader()>filterCloudLiquid()>meanZ()
    - vars2resample: list of variable in data to be resample.  Variable 'raw_time' is mandatory: e.g., ('raw_time', 'mean_beta', 'meanZ', 'lwp')
    - timeStep: time resolution to upsample the categorice data according to Pandas (e.g., 'T' for a minute-resolution)
    - limit4AvoidNan: interpolation skip all NaN till find non-nan to make the avarage. 'limit4AvoidNan' limits how far the interpolarization looks for non-nan. 
    """
    data4panda = {}
    data4panda['raw_time'] = data['raw_time']
    for _var in vars2resample:
        if _var in data: 
            data4panda[_var] = data[_var]
        else:
            print('Variable: %s is not in dictionary.' % _var)
    df = pd.DataFrame.from_dict(data4panda)
    df.set_index('raw_time',inplace=True)
    upsampled = df.resample(timeStep).mean()
    interpolated_df = upsampled.interpolate(method='linear',limit=limit4AvoidNan)
    debugger = False
    if debugger:
        example = vars2resample[0]
        plt.figure(figsize=[15,5])
        plt.scatter(data['raw_time'], data[example])
        plt.scatter(interpolated_df.index, interpolated_df[example].values)
        axes = plt.gca()
        axes.set_xlim(-5,50)
        axes.set_facecolor('whitesmoke')
        plt.show()
    return interpolated_df

def cloudMicrophysics(height, df, nu, debugger, debugger_saveImageFlag):
    """
    Inputs:    
    - height: profile of height    
    - vars2resample: list of variable in data to be resample.  Variable 'raw_time' is mandatory: e.g., ('raw_time', 'mean_beta', 'meanZ', 'lwp')
    - timeStep: time resolution to upsample the categorice data according to Pandas (e.g., 'T' for a minute-resolution)
    - limit4AvoidNan: interpolation skip all NaN till find non-nan to make the avarage. 'limit4AvoidNan' limits how far the interpolarization looks for non-nan. 
    """
    if not nu:
        print("Shape factor not provided by the user. Default value = 8.")
        nu = 8.

    kre = ( (nu+2)**3 / ((nu+3)*(nu+4)*(nu+5)  )   )**(1./3.)    
    knt = ((nu+3)*(nu+4)*(nu+5) / (nu*(nu+1)*(nu+2)) )
    water_density = 1e6 #g/m3
    resol = np.diff(height).mean()
    meanZ = df['meanZ'].values
    linearZ = np.empty(np.shape(meanZ))
    idxZnan = np.isnan(meanZ) 
    #Convert to linear reflectivity
    linearZ[~idxZnan] = 1e-18 * 10.**(meanZ[~idxZnan]/10.)
    
    #Remove negative LWP
    LWP = df['lwp'].values
    LWP[np.isnan(LWP)] = -1.
    LWP[LWP<0] = np.nan
    idxLWPnan = np.isnan(LWP) 
    
    #LWP and Z nan positions
    idxnanT = np.logical_or(idxLWPnan, idxZnan)

    #Effective radius retrieval
    effective_radius = np.empty(np.shape(meanZ))
    effective_radius.fill(np.nan)
    effective_radius[~idxnanT] = kre * ( (np.pi*water_density*resol*linearZ[~idxnanT]) / (48.*LWP[~idxnanT])  )**(1./3.)
    effective_radius[ np.logical_or(np.isnan(effective_radius), np.isinf(effective_radius)) ] = -1.
    effective_radius[effective_radius < 0] = np.nan
    #Total droplet number
    total_droplet_number = np.empty(np.shape(meanZ))
    total_droplet_number.fill(np.nan)
    total_droplet_number[~idxnanT] = (knt/linearZ[~idxnanT]) * ( (6.*LWP[~idxnanT])/(np.pi*water_density*resol) )**2.
    total_droplet_number[np.logical_or(np.isnan(total_droplet_number), np.isinf(total_droplet_number))] = -1.
    total_droplet_number[total_droplet_number < 0] = np.nan
    
    df['re'] = effective_radius*1e6 #micrometers
    df['Nd'] = total_droplet_number*1e-6 #cm^-3      
       
    if debugger:
        print('DEBUGGER: plotting figures')
        #Plot Effective Radius
        fig, axes = plt.subplots(nrows=1, figsize=(15,6))
        cmap = mpl.cm.jet
        Vmin, Vmax, Vn = 0, 150, 11
        bounds = np.linspace(Vmin,Vmax, Vn)
#        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        pp = plt.scatter(df.index, df['re'], c=LWP, vmin=Vmin, vmax=Vmax, cmap=cmap)
        cb = plt.colorbar(pp, ax=axes,
                        ticks=bounds,
                        extend='max')
        cb.set_label('LWP, $g/m^2$')
        axes.set_facecolor('whitesmoke')
        axes.set_xlabel('Time, UTC')
        axes.set_ylabel(r'$r_e$, $\mu$m')        
        axes.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
        datestr = df.index[0].strftime('%Y%m%d')
        axes.set_title('%s on  %s' % ('Effective radius' , datestr))       
        xmin = dt.datetime.strptime(datestr, '%Y%m%d')
        xmax = xmin + dt.timedelta(days=1)
        ymin = 0
        if ~np.isnan(df['re']).all():
            ymax = 1.2*df['re'].max()
        else:
            ymax = 1.0
        axes.set_xlim(xmin, xmax)
        axes.set_ylim(ymin, ymax)        
        if debugger_saveImageFlag:            
            figstr = '%s_%s_%s_%s.png' % ('cloudnet', 'jue', 're', datestr)                        
            mainpath = os.path.join('Y:\\datos\\CLOUDNET', 'juelich', 'aci')
            finalpath = os.path.join(mainpath, figstr)              
            print('Saving %s' % finalpath)
            plt.savefig(finalpath, dpi=100, bbox_inches = 'tight')
            if os.path.exists(finalpath):
                print('Saving %s...DONE!' % finalpath)
            else:
                print('Saving %s... error!' % finalpath)
            plt.close(fig)
        else:
            plt.show()

        #Plot Total number droplets
        fig, axes = plt.subplots(nrows=1, figsize=(15,6))
        cmap = mpl.cm.jet
        Vmin,Vmax, Vn = 0, 150, 11
        bounds = np.linspace(Vmin,Vmax, Vn)
#        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        pp = plt.scatter(df.index, df['Nd'], c=LWP, vmin=Vmin, vmax=Vmax, cmap=cmap)
        cb = plt.colorbar(pp, ax=axes,
                        ticks=bounds,
                        extend='max')
        cb.set_label('LWP, $g/m^2$')
        axes.set_facecolor('whitesmoke')
        axes.set_xlabel('Time, UTC')
        axes.set_ylabel(r'$Nd$, $part./cm^3$')        
        axes.set_title('%s on  %s' % ('Droplet total number' , datestr))
        axes.set_xlim(xmin, xmax)
        if debugger_saveImageFlag:            
            figstr = '%s_%s_%s_%s.png' % ('cloudnet', 'jue', 'Nd', datestr)                        
            mainpath = os.path.join('Y:\\datos\\CLOUDNET', 'juelich', 'aci')
            finalpath = os.path.join(mainpath, figstr)              
            print('Saving %s' % finalpath)
            plt.savefig(finalpath, dpi=100, bbox_inches = 'tight')
            if os.path.exists(finalpath):
                print('Saving %s...DONE!' % finalpath)
            else:
                print('Saving %s... error!' % finalpath)
            plt.close(fig)
        else:
            plt.show()        
    return df

def aciRetrieval(idf, LWPmin, LWPmax, LWPstep, fitThreshold, debugger, debugger_saveImageFlag):
    """
    Inputs:
    - data: categorice data from categorice_reader() > filterCloudLiquid() > meanZ()
    - idf: interpolated dataFrame from upsampler()
    - LWPmin, LWPmax, LWPstep: to define the LWP array.
    - fitThreshold: 
    """    
    #Define LWP range 
    LWPrange = np.arange(LWPmin, LWPmax-LWPstep, LWPstep)

    #Create ACI matrix with statistics   
    ACI = {}
    #ACI - re     
    ACI['re'] = np.nan*np.empty([np.size(LWPrange), 8])
    for idx, _lwp in enumerate(LWPrange):
        #Define LWP range t obe ~constant
        lwp_min = _lwp
        lwp_max = _lwp + LWPstep            
        idxLWP = np.logical_and(idf['lwp'] >= lwp_min,  idf['lwp'] < lwp_max)            
            #Select data in the LWP range
            #Select data in the LWP range
        idxgtzero = np.logical_and(idf['re'] > 0, idf['mean_beta'] > 0)
#            print('%d points found in idxgtzero' % idxgtzero.sum())
        idxnan = np.logical_or(np.isnan(idf['mean_beta']), np.isnan(idf['re']))
#            print('%d points found in idxnan' % idxnan.sum())
        idxtotal1 = np.logical_and(~idxnan, idxgtzero)
#            print('%d points found in idxtotal1' % idxtotal1.sum())
        idxtotal2 = np.logical_and(idxtotal1, idxLWP) 
#            print('%d points found in idxtotal2' % idxtotal2.sum())                
        idxtotal = idxtotal2
        if idxtotal.sum() >= fitThreshold:                    
            LWP = idf['lwp'][idxtotal]
            log_beta=np.log(idf['mean_beta'][idxtotal])
            log_re =np.log(idf['re'][idxtotal])                    
            if np.logical_and( np.isnan(log_beta).sum()!=len(log_beta), np.isnan(log_beta).sum()!=len(log_re)):
                
                #ACI index        
                _slope, _intercept, _R, _p, _std_err = stats.linregress(log_beta, log_re)
                ACI['re'][idx] = np.asarray([lwp_min, lwp_max, idxtotal.sum(), _slope, _intercept, _R, _p, _std_err])

                if debugger:                    
                    line = _slope*log_beta+_intercept
                    _fig = plt.figure(figsize=(8,5))
                    cmap = plt.get_cmap('jet')
                    plt.scatter(log_beta, log_re, 15, c=LWP, cmap=cmap)
                    plt.plot(log_beta, line, 'r')
                    plt.title(r"ACI$_{re}$=%3.2f, r_value=%3.2f, N=%d, LWP=[%d,%d] $g/m^2$" % (_slope, _R, idxtotal.sum(), lwp_min, lwp_max))
                    plt.xlabel("log(beta)")
                    plt.ylabel("log(re)")
                    cbar= plt.colorbar()
                    cbar.set_label(r"$LWP (g/m^2)$", labelpad=+1)
                    ax = plt.gca()
                    ax.set_facecolor('whitesmoke')
                    if debugger_saveImageFlag:          
                        datestr = idf.index[0].strftime('%Y%m%d')
                        figstr = '%s_%03d-%03d_%s_%s_%s.png' % ('cloudnet', lwp_min, lwp_max, 'jue', 'aci-re', datestr)
                        mainpath = os.path.join('Y:\\datos\\CLOUDNET', 'juelich', 'aci')
                        finalpath = os.path.join(mainpath, figstr)              
                        print('Saving %s' % finalpath)
                        plt.savefig(finalpath, dpi=100, bbox_inches = 'tight')
                        if os.path.exists(finalpath):
                            print('Saving %s...DONE!' % finalpath)
                        else:
                            print('Saving %s... error!' % finalpath)
                        plt.close(_fig)
                    else:
                        plt.show()        
            else:
                print('NaN array found in ACI-re index.')            
        else:
            print('Conditions not accomplished for ACI retrieval.')    
    # ACi - Nd
    ACI['Nd'] = np.nan*np.empty([1, 8])    
    idxLWP = np.logical_and(idf['lwp'] > LWPmin,  idf['lwp'] < LWPmax)            
    if idxLWP.size > fitThreshold:
        idxgtzero = np.logical_and(idf['Nd'] > 0, idf['mean_beta'] > 0)
        idxnan = np.logical_and(np.isnan(idf['mean_beta']), np.isnan(idf['Nd']))
        idxtotal = np.logical_and(~idxnan, idxgtzero)        
        LWP = idf['lwp'][idxtotal]

        #Select data in the LWP range
        log_beta=np.log(idf['mean_beta'][idxtotal])
        log_Nd =np.log(idf['Nd'][idxtotal])
        LWP = idf['lwp'][idxtotal]
        
        if np.logical_and( np.isnan(log_beta).sum()!=len(log_beta), np.isnan(log_beta).sum()!=len(log_Nd)):
            #ACI index        
            _slope, _intercept, _R, _p, _std_err = stats.linregress(log_beta, log_Nd)
            ACI['Nd'][0] = np.asarray([lwp_min, lwp_max, idxtotal.sum(), _slope, _intercept, _R, _p, _std_err])                        
            if debugger:
                line = _slope*log_beta+_intercept
                _fig = plt.figure(figsize=(8,5))
                cmap = plt.get_cmap('jet')
                plt.scatter(log_beta, log_Nd, 15, c=LWP, cmap=cmap)
                plt.plot(log_beta, line, 'r')
                plt.title(r"ACI$_{Nd}$=%3.2f, r_value=%3.2f, N=%d, LWP=[%d,%d] $g/m^2$" % (_slope, _R, idxtotal.sum(), LWPmin, LWPmax))
                plt.xlabel("log(beta)")
                plt.ylabel("log(Nd)")
                cbar= plt.colorbar()
                cbar.set_label(r"$LWP (g/m^2)$", labelpad=+1)
                ax = plt.gca()
                ax.set_facecolor('whitesmoke')
                if debugger_saveImageFlag:          
                    datestr = idf.index[0].strftime('%Y%m%d')
                    figstr = '%s_%03d-%03d_%s_%s_%s.png' % ('cloudnet', LWPmin, LWPmax, 'jue', 'aci-Nd', datestr)
                    mainpath = os.path.join('Y:\\datos\\CLOUDNET', 'juelich', 'aci')
                    finalpath = os.path.join(mainpath, figstr)              
                    print('Saving %s' % finalpath)
                    plt.savefig(finalpath, dpi=100, bbox_inches = 'tight')
                    if os.path.exists(finalpath):
                        print('Saving %s...DONE!' % finalpath)
                    else:
                        print('Saving %s... error!' % finalpath)
                    plt.close(_fig)
                else:
                    plt.show()        
        else:
            print('NaN array found in ACI-Nd index.')            
    return ACI

