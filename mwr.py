# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 22:40:59 2020

@author: Marta
"""

import os
import sys
import glob
import argparse
import matplotlib
import numpy as np
import pandas as pd
import xarray as xr
import netCDF4 as nc
import datetime as dt
#from utils_gfat import plot
import plot
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from distutils.dir_util import mkpath
from matplotlib.dates import DateFormatter

__author__ = "Jiménez-Martín, Marta María"
__license__ = "GPL"
__version__ = "1.1"
__maintainer__ = "Jiménez-Martín, Marta María"
__email__ = "mmjimenez@ugr.es"
__status__ = "Production"

def reader(filelist, filelistmet):
    """
    RPG Microwave Radiometer data reader. 
    Inputs:
    - filelist: List of radar files (i.e, '/drives/c/ZENITH*.TMP.NC') (str)
    Output:
    - mwr: Xarray dataframe (xarray) or 'None' in case of error.
    """
    
    def include_potential_temperature(mwr, filelistmet):
        """
        Potential temperature in terms of RPG temperature and pressure data
        
        """    
        #Read meteo data
        metfiles2load = glob.glob(filelistmet)    
        if metfiles2load:     
            mwrmet = xr.open_mfdataset(metfiles2load)
            if mwrmet:            
                meanday_press = mwrmet.env_pressure.values.mean()
                meanday_temp = mwrmet.env_temperature.values.mean()
                if not meanday_press:
                    meanday_press = 960.
                if not meanday_temp :
                    meanday_temp  = 960.            
            else:
                meanday_press = 960.
                meanday_temp = 20.

            # Parámetros de atmósfera standard escalada
            T = np.ones(mwr.altitude_layers.values.size)*np.nan
            P = np.ones(mwr.altitude_layers.values.size)*np.nan
            for i,_height in enumerate(mwr.altitude_layers):                
                sa = helper_functions.standard_atmosphere(_height, meanday_temp, meanday_press)                
                P[i]  = sa[0]
                T[i]  = sa[1] 

            pressure_ratio = np.power((1000./P),0.286)
            pot_temp = (mwr.temperature_profiles+273.15)*np.tile(pressure_ratio, (mwr.temperature_profiles.shape[0],1)) - 273.15

            mwr['potential_temperature_profiles'] = pot_temp
            mwr['potential_temperature_profiles'].attrs['units'] = '$^{\circ}$C'
            mwr['potential_temperature_profiles'].attrs['long_name'] = 'Potential temperature'
            mwr['potential_temperature_profiles'].attrs['pressure_ratio'] = pressure_ratio
            mwr['potential_temperature_profiles'].attrs['pressure'] = P
        #     print('Potential temperature has been added.')
        return mwr
        
    #Date format conversion
    files2load = glob.glob(filelist)    

    if files2load:        
        print(files2load)

        #Extract information from filename   
#         parts = os.path.basename(files2load[0]).split('_')
#         print(parts)
#         infofile = {'radarNick': parts[0], 'stationNick': parts[1]}        

        #Load data
        mwr = []
        tmp_mwr = xr.open_mfdataset(files2load)
        if tmp_mwr:
            mwr0 = tmp_mwr.assign_coords(number_altitude_layers=tmp_mwr.altitude_layers.values)        
            mwr = mwr0.rename({'number_altitude_layers': 'altitude'})        
            #Encoding time as python datetime
            mwr['time'].attrs['long_name'] = 'Date, UTC'
            mwr['time'].attrs['units'] = 'Datetime'        
            mwr = mwr.sortby(mwr['time'])                    
            if not 'location' in mwr.attrs.keys():
                mwr.attrs['location'] = 'Granada'
                print('Granada added as location attribute.')

            mwr['temperature_profiles'] = mwr['temperature_profiles'] - 273.15
            mwr['temperature_profiles'].attrs['units'] = '$^{\circ}$C'
            infofile = {'stationNick': 'gr','mwrNick': 'mwr'} #Creo que esto no hace falta y complica la función así que lo quitaría.

            mwr = include_potential_temperature(mwr, filelistmet)
        else:
            mwr = None
    else:
        mwr = None
        infofile = None
#     mwr.close()
    return mwr, infofile




def mean_time(filelist, start_time, stop_time, variables='Ze'):
    """

    """
    


    return datamean

def mean_space(filelist, start_time, stop_time, variables='Ze'):
    """

    """
    


    return datamean


def plot_mwr_variables(filelist, filelistmet, variables2plot, plt_conf, figdirectory):
    """
    Quicklook maker of RPG MWR measurements.    
    Inputs:
    - filelist: List of mwr files (i.e, '/drives/c/ZENITH*.TMP.NC') (str)
    - variables2plot: mwr variables. (integer)
    Outputs:
    - None
    """
    ###############################################################################
    #Define colorbar    
    bottom = matplotlib.cm.get_cmap('cool', 90)  #Cool
    middle = matplotlib.cm.get_cmap('winter', 30)  #Winter
    top = matplotlib.cm.get_cmap('inferno', 80) #'Spectral'    #Inferno
    lsbottom = np.flip(bottom(np.linspace(0, 1, 90)),axis=0)
#     lsbottom = bottom(np.linspace(0, 1, 90))
    lsmiddle = middle(np.linspace(0, 1, 30))
#     lsmiddle = np.flip(middle(np.linspace(0, 1, 30)),axis=0)
    lstop = np.flip(top(np.linspace(0, 1, 80)),axis=0)
#     lstop = top(np.linspace(0, 1, 80))
    newcolors = np.vstack((lsbottom, lsmiddle , lstop ))    
    mwr_cmap = matplotlib.colors.ListedColormap(newcolors, name='OrangeBlue')
    ###############################################################################
       
    #Dictionaries
    #Variables that can be plotted     
    var2plot = {0: 'temperature_profiles', 1: 'potential_temperature_profiles'} #, 3: 'sigma', 4: 'kurt'
    #Mininum value on the colorbar
    Vmin = {0: -60, 1: 0}#, 1: -5, 2: 0} #, 3: -3, 4: -3
    #Maximum value on the colorbar
    Vmax = {0: 40, 1: 50}#, 1: 5, 2: 5} #, 3: 3, 4: 3
    #Variable name to be written in the title of the figure
    titleStr = {0: 'Temperature', 1: 'Potential Temperature'}#, 1: 'vertical mean velocity', 2: 'spectral width'}
    #Variable name to be written in the ylabel of the colorbar
    ylabelbar = {0: 'Temperature,$\degree$$C$', 1: 'Temperature,$\degree$$C$'}#, 1: 'Vertical mean velocity, $[m/s]$', 2: 'Doppler spectral with, $[m/s]$'}
    cbar = {0:mwr_cmap, 1:matplotlib.cm.jet}
    #Font size of the letters in the figure
    matplotlib.rcParams.update({'font.size': 20})
    
    # Read the list of files to plot
    # --------------------------------------------------------------------
    mwr, infofile = reader(filelist, filelistmet)   

    if mwr != None:
        # One figure per variable
        # --------------------------------------------------------------------
        for idx in variables2plot:
            var_ = var2plot[idx]
#             print(mwr[var_])
            print('Current plot %s' % var_)
            # #Create Figure
            fig = plt.figure(figsize=(20,10))
            axes = fig.add_subplot(111)    
            #Plot        
            cmap = mwr_cmap
            cmap.set_under("crimson")
            cmap.set_over("w")
            bounds = np.linspace(Vmin[idx], Vmax[idx], 128)
            norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
            range_km = mwr['altitude_layers']/1000.
            q = axes.pcolormesh(mwr['time'], range_km, mwr[var_].T,
                                cmap=cbar[idx],
                                norm=norm,
                                vmin=Vmin[idx],
                                vmax=Vmax[idx],
                                shading='flat')

#             axes.pcolormesh(mwr['time'], range_km, mwr_zero.T - 273.)
            q.cmap.set_under('magenta')
            q.cmap.set_over('black')
            # q.update_ticks()
            cb = plt.colorbar(q, ax=axes, ticks = np.arange(Vmin[idx], Vmax[idx], 10),  extend='both')
            
            
            # search for gaps in data
            # --------------------------------------------------------------------     
            if plt_conf['gapsize'] == 'default':        
                dif_time = mwr['time'].values[1:] - mwr['time'].values[0:-1]                   
                GAP_SIZE = 2*int(np.ceil((np.median(dif_time).astype('timedelta64[s]').astype('float')/60))) #GAP_SIZE is defined as the median of the resolution fo the time array (in minutes)        
                print('GAP_SIZE parameter automatically retrieved to be %d.' % GAP_SIZE)         
            else:
                GAP_SIZE = int(plt_conf['gapsize'])
                print('GAP_SIZE set by the user: %d (in minutes)' % GAP_SIZE)     
            dttime = np.asarray([dt.datetime.utcfromtimestamp((time_ - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')) for time_ in  mwr['time'].values])
            plot.gapsizer(axes, dttime, range_km, GAP_SIZE, '#c7c7c7')

            # Setting axes 
            # --------------------------------------------------------------------
            mf = matplotlib.ticker.FuncFormatter(plot.tmp_f)
            axes.xaxis.set_major_formatter(mf)
            hours = mdates.HourLocator(range(0, 25, 3))
            date_fmt = mdates.DateFormatter('%H')
            axes.xaxis.set_major_locator(hours)
            axes.xaxis.set_major_formatter(date_fmt)        
            min_date = mwr['time'].values.min()
            max_date = mwr['time'].values.max()
            axes.set_xlim(min_date  -  np.timedelta64(minutes=30), max_date +  -  np.timedelta64(hours=23, minutes=59, seconds=59))            
            axes.xaxis.set_major_locator(matplotlib.dates.HourLocator(interval = 1))
            axes.set_ylim(plt_conf['y_min'], plt_conf['y_max'])
            plt.grid(True)
            axes.set_xlabel('Time, $[UTC]$')
            axes.set_ylabel('Altitude, $[km]$')
            cb.ax.set_ylabel(ylabelbar[idx])
        
            # title
            # ----------------------------------------------------------------------------
            plt_conf['title1'] = 'Microwave Radiometer %s' % var_ #titleStr[idx]
            plot.title1(plt_conf['title1'], plt_conf['coeff'])
            datestr = mwr['time'].values.min().astype('str').split('T')[0]
            print('Datestr:',datestr)
            plot.title2(datestr, plt_conf['coeff'])
            if not 'location' in plt_conf:
                loc = {'gr': 'Granada'} 
                plt_conf['location'] = loc[infofile['stationNick']]

            plot.title3('{} ({:.1f}N, {:.1f}E)'.format(plt_conf['location'], float(mwr.station_latitude[1:3]), float(mwr.station_longitude[1:2])), plt_conf['coeff'])

            # logo
            # ----------------------------------------------------------------------------
            plot.add_GFAT_logo([0.85, 0.01, 0.15, 0.15])
                                
            # Font Size
            # ----------------------------------------------------------------------------
            # for item in ([axes.title, axes.xaxis.label, axes.yaxis.label] +
            #              axes.get_xticklabels() + axes.get_yticklabels()):
            #     item.set_fontsize(20)    
            
            # Saver figure
            # ----------------------------------------------------------------------------           
            debugging = False
            
            if debugging:
                plt.show()
            else:            
                #create output folder
                # --------------------------------------------------------------------
                year = datestr.replace('-','')[0:4]
                fulldirpath = os.path.join(figdirectory, year, var_) 
                if np.logical_not(os.path.exists(fulldirpath)):
                    mkpath(fulldirpath)
                    print('fulldirpath created: %s' % fulldirpath)                
                figstr = '%s_%s_%s_%s.png' % (infofile['mwrNick'], infofile['stationNick'], var_, datestr.replace('-',''))
                finalpath = os.path.join(fulldirpath, figstr)
                print('Saving %s' % finalpath)
                plt.savefig(finalpath, dpi=100, bbox_inches = 'tight')
                if os.path.exists(finalpath):
                    print('Saving %s...DONE!' % finalpath)
                else:
                    print('Saving %s... error!' % finalpath)
        mwr.close()
    else:
        print('ERROR: files not found with format: %s' % filelist)
    return





def daily_quicklook(filelist, filelistmet, figdirectory):
    """
    Formatted daily quicklook of RPG Radiometer.
    Inputs:
    - filelist: List of mwr files (i.e, '/drives/c/*ZEN*.LC?') (str)
    - figdirectory: Array of numbers corresponding to the variables read by RPG. (integer)
    Outputs:
    - None
    """
    
    plt_conf = {}
    plt_conf['gapsize'] = 'default'
    plt_conf['y_min'] = 0.
    plt_conf['y_max'] = 10.
    plt_conf['coeff'] = 2.   
    variables2plot = (0,1)
    plot_mwr_variables(filelist, filelistmet, variables2plot, plt_conf, figdirectory)


def date_quicklook(dateini, dateend, path1a='GFATserver', figpath='GFATserver'):
    """
    Formatted daily quicklook of RPG Cloud Radar measurements for hierarchy GFAT data.
    Inputs:
    - path1a: path where 1a-level data are located.
    - figpath: path where figures are saved.
    - Initial date [yyyy-mm-dd] (str). 
    - Final date [yyyy-mm-dd] (str).
        
    Outputs: 
    - None
    """        

    if path1a == 'GFATserver':
        path1a = '/mnt/NASGFAT/datos/rpgradar/1a'

    if figpath == 'GFATserver':
        figserverpath = '/mnt/NASGFAT/datos/rpgradar/quicklooks'        

    inidate = dt.datetime.strptime(dateini, '%Y%m%d')
    enddate = dt.datetime.strptime(dateend, '%Y%m%d')

    period = enddate - inidate
    
    for _day in range(period.days + 1):
        current_date = inidate + dt.timedelta(days=_day)            
        filename = 'wradar_gr_%s*compact.nc' % (dt.datetime.strftime(current_date, '%y%m%d'))
        filelist = os.path.join(path1a, '%d' % current_date.year, '%02d' % current_date.month, '%02d' % current_date.day, filename)        
        if figpath == 'GFATserver':
            figdirectory = figserverpath

        daily_quicklook(filelist, figdirectory)

def send2web(filelist, remote_directory, overwrite):
    """
    Send quicklooks to the browser webpage.
    Inputs:
    - filelist: List of radar figures (i.e, '/drives/c/*ZEN*.png') (str)
    - remote_directory: directory in the ftp where radar figures are located. (integer)
    - overwrite: Flag to allow/avoid overwritting files. (binary) 
    Outputs:
    - None
    """

    import ftplib
    #Information to access to the server
    address = 'ftpwpd.ugr.es'
    port = 21
    user = 'gfat'
    pwd = '4evergfat' # password
    files2send = glob.glob(filelist)
    if files2send and files2send != None:        
        for tmp_file in files2send:
            print('Current file to send: %s' % tmp_file)
            ftp = ftplib.FTP(address)
            ftp.set_pasv(True)
            try:
                ftp.connect(address, port)
                ftp.login(user, pwd)
                ftp.cwd(remote_directory)                
                if overwrite:
                    file4tranfer=True
                else:
                    remote_lines=[]
                    ftp.retrlines('LIST', remote_lines.append)
                    remote_files = [line_.split(' ')[-1] for line_ in remote_lines]
                    file4tranfer = os.path.basename(tmp_file) not in remote_files
                if file4tranfer:
                    with open(tmp_file, "rb") as FILE:                    
                        ftp.storbinary('STOR ' + os.path.basename(tmp_file), FILE)
                    ftp.close()
                    code = 1
                else:
                    code = 2
            except:
                code = 0
                ftp.close()
            if code==1:
                print('Succesfully sent: %s' %  os.path.join(remote_directory, os.path.basename(tmp_file)))
            elif code == 2:
                print('Send aborted. File already exists in ftp: %s' %  os.path.join(remote_directory, os.path.basename(tmp_file)))
            else:
                print(': Unable to send file %s' % os.path.basename(tmp_file))
    else:
        print('No files as %s' % filelist)

def main():
    parser = argparse.ArgumentParser(description="usage %prog [arguments]")
    parser.add_argument("-i", "--initial_date",
        action="store",
        dest="dateni",
        required=True,
        help="Initial date [example: '20190131'].")         
    parser.add_argument("-e", "--final_date",
        action="store",
        dest="dateend",
        default=".",
        help="Final date [example: '20190131'].")
    parser.add_argument("-d", "--datadir",
        action="store",
        dest="path1a",
        default="GFATserver",
        help="Path where date-hierarchy files are located [example: '~/data/1a'].")
    parser.add_argument("-f", "--figuredir",
        action="store",
        dest="figpath",
        default="GFATserver",
        help="Path where figures will be saved [example: '~/radar/quicklooks'].")
    args = parser.parse_args()

    date_quicklook(args.dateini, args.dateend, path1a=args.path1a, figpath=args.figpath)

if __name__== "__main__":
    main()