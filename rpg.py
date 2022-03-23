#!/usr/bin/env python

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
import plot
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from distutils.dir_util import mkpath
from matplotlib.dates import DateFormatter
import pdb

__author__ = "Bravo-Aranda, Juan Antonio"
__license__ = "GPL"
__version__ = "1.1"
__maintainer__ = "Bravo-Aranda, Juan Antonio"
__email__ = "jabravo@ugr.es"
__status__ = "Production"

def reader(filelist):
    """
    RPG Cloud Radar data reader. 
    Inputs:
    - filelist: List of radar files (i.e, '/drives/c/*ZEN*.LC?') (str)
    Output:
    - rpg: Xarray dataframe (xarray) or 'None' in case of error.
    """
    def retrieve_dbZe(rpg):
        """
        It converts the linear reflectivity into decibel reflectivity.
        Inputs:
        - rpg: Xarray dataframe. (xarray)
        Output: 
        - rpg: Xarray dataframe. (xarray)
        """
        #Convert reflectivity to dB
        dbze = 10*np.log10(rpg['Ze'].values)
        
        #Converting array in DataArray
        dbze_tmp = xr.DataArray(dbze, coords=[('time', rpg['time'].values), ('range', rpg['range'].values)])
        #Meta data
        dbze_tmp.attrs['units'] = 'dBZe'
        dbze_tmp.attrs['long_name'] = 'Equivalent radar reflectivity factor Ze'
        dbze_tmp.name = 'dBZe'
        return dbze_tmp

    #Date format conversion
    files2load = glob.glob(filelist)    

    if files2load:        
        print(files2load)

        #Extract information from filename
        parts = os.path.basename(files2load[0]).split('_')
        infofile = {'radarNick': parts[0], 'stationNick': parts[1], 'datestr': parts[2], 'timestr': parts[3]}

        #Load data
        #rpg = []
        #rpg = xr.open_mfdataset(files2load, combine='by_coords')
        rpg = None
        for ff in files2load:
            with xr.open_dataset(ff) as ds:
                if ds.dims['chirp_sequences'] == 4:
                    if rpg is None:
                        rpg = ds
                    else:
                        try:
                            rpg = xr.concat([rpg, ds], 'time')#, coords='minimal', vars='minimal')
                        except:
                            pass
        rpg = rpg.sortby('time')
        #Encoding time as python datetime
        rpg['time'].attrs['units'] = 'seconds since 2001-01-01'
        rpg = xr.decode_cf(rpg)    
        rpg['time'].attrs['long_name'] = 'Date, UTC'
        rpg['time'].attrs['units'] = 'Datetime'        
        rpg = rpg.sortby(rpg['time'])
        #Convert Ze in dBZe and include the variable in the DataFrame
        dbze_tmp = retrieve_dbZe(rpg)                
        finalrpg = xr.merge([rpg, dbze_tmp])
        rpg.close()
        if not 'location' in finalrpg.attrs.keys():
            finalrpg.attrs['location'] = 'Granada'
            print('Granada added as location attribute.')
    else:
        finalrpg = None
        infofile = None
    return finalrpg, infofile

def mean_time(filelist, start_time, stop_time, variables='Ze'):
    """

    """
    


    return datamean

def mean_space(filelist, start_time, stop_time, variables='Ze'):
    """

    """
    


    return datamean


def plot_rpg_moments(filelist, moments2plot, plt_conf, figdirectory):
    """
    Quicklook maker of RPG Cloud Radar measurements.
    Inputs:
    - filelist: List of radar files (i.e, '/drives/c/*ZEN*.LC?') (str)
    - moments2plot: Array of numbers corresponding to the moment of the Doppler spectra. (integer)
    Outputs:
    - None
    """
    #Dictionaries
    #Variables that can be plotted     
    var2plot = {0: 'dBZe', 1: 'vm', 2: 'sigma'} #, 3: 'sigma', 4: 'kurt'
    #Mininum value on the colorbar
    Vmin = {0: -65, 1: -5, 2: 0} #, 3: -3, 4: -3
    #Maximum value on the colorbar
    Vmax = {0: 35, 1: 5, 2: 5} #, 3: 3, 4: 3
    #Variable name to be written in the title of the figure
    titleStr = {0: 'reflectivity', 1: 'vertical mean velocity', 2: 'spectral width'}
    #Variable name to be written in the ylabel of the colorbar
    ylabelbar = {0: 'Equivalent radar reflectivity, $[dBZe]$', 1: 'Vertical mean velocity, $[m/s]$', 2: 'Doppler spectral with, $[m/s]$'}

    #Font size of the letters in the figure
    matplotlib.rcParams.update({'font.size': 16})
    
    # Read the list of files to plot
    # --------------------------------------------------------------------
    rpg, infofile = reader(filelist)

    if rpg != None:
        # One figure per variable
        # --------------------------------------------------------------------
        for idx in moments2plot:
            var_ = var2plot[idx]
            print('Current plot %s' % var_)
            #Create Figure
            fig = plt.figure(figsize=(15,5))
            axes = fig.add_subplot(111)    
            #Plot        
            cmap = matplotlib.cm.jet
            bounds = np.linspace(Vmin[idx], Vmax[idx], 128)
            norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
            range_km = rpg['range']/1000.
            q = axes.pcolormesh(rpg['time'], range_km, rpg[var_].T,
                                cmap=cmap,
                                vmin=Vmin[idx],
                                vmax=Vmax[idx])
    #                             norm=norm)
            q.cmap.set_over('white')
            # q.cmap.set_under('darkblue')
            # q.update_ticks()
            cb = plt.colorbar(q, ax=axes,
                            #ticks=bounds,
                            extend='max')
            
            
            # search for gaps in data
            # --------------------------------------------------------------------     
            if plt_conf['gapsize'] == 'default':        
                dif_time = rpg['time'].values[1:] - rpg['time'].values[0:-1]                   
                GAP_SIZE = 2*int(np.ceil((np.median(dif_time).astype('timedelta64[s]').astype('float')/60))) #GAP_SIZE is defined as the median of the resolution fo the time array (in minutes)        
                print('GAP_SIZE parameter automatically retrieved to be %d.' % GAP_SIZE)         
            else:
                GAP_SIZE = int(plt_conf['gapsize'])
                print('GAP_SIZE set by the user: %d (in minutes)' % GAP_SIZE)     
            dttime = np.asarray([dt.datetime.utcfromtimestamp((time_ - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')) for time_ in  rpg['time'].values])
            plot.gapsizer(axes, dttime, range_km, GAP_SIZE, '#c7c7c7')

            # Setting axes 
            # --------------------------------------------------------------------
            mf = matplotlib.ticker.FuncFormatter(plot.tmp_f)
            axes.xaxis.set_major_formatter(mf)
            hours = mdates.HourLocator(range(0, 25, 3))
            date_fmt = mdates.DateFormatter('%H')
            axes.xaxis.set_major_locator(hours)
            axes.xaxis.set_major_formatter(date_fmt)        
            min_date = rpg['time'].values.min()
            max_date = rpg['time'].values.max()
            axes.set_xlim(min_date.astype('datetime64[D]'), max_date.astype('datetime64[D]') + np.timedelta64(1,'D'))
            axes.xaxis.set_major_locator(matplotlib.dates.HourLocator(interval = 1))
            axes.set_ylim(plt_conf['y_min'], plt_conf['y_max'])
            plt.grid(True)
            axes.set_xlabel('Time, $[UTC]$')
            axes.set_ylabel('Altitude, $[km]$')
            cb.ax.set_ylabel(ylabelbar[idx])
        
            # title
            # ----------------------------------------------------------------------------
            plt_conf['title1'] = 'W-band radar %s' % titleStr[idx]
            plot.title1(plt_conf['title1'], plt_conf['coeff'])
            datestr = rpg['time'].values.min().astype('str').split('T')[0]
            plot.title2(datestr, plt_conf['coeff'])
            if not 'location' in plt_conf:
                loc = {'gr': 'Granada'} 
                plt_conf['location'] = loc[infofile['stationNick']]

            plot.title3('{} ({:.1f}N, {:.1f}E)'.format(plt_conf['location'], float(rpg['Lat'].values[0]), float(rpg['Lon'].values[0])), plt_conf['coeff'])

            # logo
            # ----------------------------------------------------------------------------
            plot.watermark(fig, axes,alpha=0.5)
                                
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
                figstr = '%s_%s_%s_%s.png' % (infofile['radarNick'], infofile['stationNick'], var_, datestr.replace('-',''))
                finalpath = os.path.join(fulldirpath, figstr)
                print('Saving %s' % finalpath)
                plt.savefig(finalpath, dpi=100, bbox_inches = 'tight')
                if os.path.exists(finalpath):
                    print('Saving %s...DONE!' % finalpath)
                else:
                    print('Saving %s... error!' % finalpath)
                plt.close()
    else:
        print('ERROR: files not found with format: %s' % filelist)
    return

def daily_quicklook(filelist, figdirectory):
    """
    Formatted daily quicklook of RPG Cloud Radar measurements.
    Inputs:
    - filelist: List of radar files (i.e, '/drives/c/*ZEN*.LC?') (str)
    - figdirectory: Array of numbers corresponding to the moment of the Doppler spectra. (integer)
    Outputs:
    - None
    """
    
    plt_conf = {}
    plt_conf['gapsize'] = 'default'
    plt_conf['y_min'] = 0
    plt_conf['y_max'] = 14
    plt_conf['coeff'] = 2    
    moments2plot = (0, 1, 2)
    plot_rpg_moments(filelist, moments2plot, plt_conf, figdirectory)

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
        figpath= '/mnt/NASGFAT/datos/rpgradar/quicklooks'        

    inidate = dt.datetime.strptime(dateini, '%Y%m%d')
    enddate = dt.datetime.strptime(dateend, '%Y%m%d')

    period = enddate - inidate
    
    for _day in range(period.days + 1):
        current_date = inidate + dt.timedelta(days=_day)            
        filename = 'wradar_gr_%s*compact.nc' % (dt.datetime.strftime(current_date, '%y%m%d'))
        filelist = os.path.join(path1a, '%d' % current_date.year, '%02d' % current_date.month, '%02d' % current_date.day, filename)        

        daily_quicklook(filelist, figpath)

def send2web(filelist, remote_directory, overwrite):
    """
    Send quicklooks to the browser webpage.
    Inputs:
    - filelist: List of radar figures (i.e, '/drives/c/*ZEN*.png') (str)
    - remote_directory: directory in the ftp where radar figures are located. (string)
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
        dest="dateini",
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
