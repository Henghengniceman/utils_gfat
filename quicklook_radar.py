#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import glob
import datetime as dt
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib
from matplotlib.dates import DateFormatter
from optparse import OptionParser
from utils_gfat import readers

def main():
    parser = OptionParser(usage="usage %prog [options]",
        version="%prog 1.0")
    parser.add_option("-l", "--radar_nick",
        action="store",
        dest="radar_name",
        default="wradar",
        help="Radar nick name [default: %default].")
    parser.add_option("-d", "--date2plot",
        action="store",
        dest="date2plot",
        default="20190429",
        help="Date to plot [default: %default].")
    parser.add_option("-t", "--dir_out",
        action="store",
        dest="odir",
        default=".",
        help="Output folder [default: %default].")
    parser.add_option("-i", "--dir_in",
        action="store",
        dest="idir",
        default=".",
        help="Input folder [default: %default].")         
    (options, args) = parser.parse_args()

    radarNick=options.radar_name
    datestr=options.date2plot
    options.date2plot = dt.datetime.strptime(options.date2plot, '%Y%m%d')    
    #year=dt.datetime.strftime(options.date2plot, '%Y') 
    #month=dt.datetime.strftime(options.date2plot, '%m')
    #day=dt.datetime.strftime(options.date2plot, '%d')
    mainPath=options.idir
    figpath=options.odir

    # print(year)
    # print(month)
    # print(day)
    # yesterday = dt.datetime.today() - dt.timedelta(1)        
    # datestr = yesterday.strftime('%Y%m%d')

    # def readRPG(mainPath, radarNick, stationNick, datestr):
    #     #Date format conversion
    #     datestr = dt.datetime.strptime(datestr, '%Y%m%d')
    #     date2name = dt.datetime.strftime(datestr, '%y%m%d')
    #     year = dt.datetime.strftime(datestr, '%Y')
    #     month = dt.datetime.strftime(datestr, '%m')
    #     day = dt.datetime.strftime(datestr, '%d')
    #     #Datestr format yymmdd
    #     fileName=
    #     #fileName="wradar_gr_190407_210003_P06_ZEN_compact.nc"
    #     filePath = os.path.join(mainPath, fileName)
    #     print('Looking for files: %s' % filePath)
    #     fileList = sorted(glob.glob(filePath))
    #     control = 1
    #     #load the files
    #     rpg = []
    #     print('List of files founds:')
    #     print(fileList)        
    #     for tempFilePath in sorted(fileList):
    #         temp_rpg = xr.open_dataset(tempFilePath)       
    #         if temp_rpg.attrs['location'] == 'Granada':
    #             temp_rpg['Lat'] = 37.1637            
    #             temp_rpg['Lon'] = -3.60
    # #         try:        
    #         if control:
    #             rpg = temp_rpg            
    #             control = 0
    #         else:             
    #             rpg = xr.merge([rpg, temp_rpg], compat='no_conflicts')
    # #           rpg.combine_first(temp_rpg)
    # #             rpg = xr.merge([rpg, temp_rpg], compat='no_conflicts')                
    # #         except:
    # #             print('Error with %s file' % tempFilePath)
    #         rpg.attrs = temp_rpg.attrs
    #         temp_rpg.close()
    #     return rpg

    def retrieve_dbZe(rpg):
        #Convert reflectivity to dB
        dbze = 10*np.log10(rpg['Ze'].values)
        
        #Converting array in DataArray
        dbze_tmp = xr.DataArray(dbze, coords=[('time', rpg['time'].values), ('range', rpg['range'].values)])
        #Meta data
        dbze_tmp.attrs['units'] = 'dBZe'
        dbze_tmp.attrs['long_name'] = 'Equivalent radar reflectivity factor Ze'
        dbze_tmp.name = 'dBZe'
        return dbze_tmp

    def plot_rpg_moments(radarNick, new, rpg, datestr):
        #Adapt date string
        strdate = dt.datetime.strptime(datestr, '%Y%m%d')
        year = dt.datetime.strftime(strdate, '%Y')
        #Information required for plotting
        #Dictionary of the plots
        var2plot = {0: 'dBZe', 1: 'vm', 2: 'sigma'} #, 3: 'sigma', 4: 'kurt'
        #Dictionary for the vmax and vmin of the plot
        Vmin = {0: -65, 1: -5, 2: 0} #, 3: -3, 4: -3
        Vmax = {0: 30, 1: 5, 2: 5} #, 3: 3, 4: 3
        titleStr = {0: 'reflectivity', 1: 'vertical mean velocity', 2: 'spectral width'}
        #Vmax = {0: 50, 1: 5, 2: 5, 3: 3, 4: 3}
        matplotlib.rcParams.update({'font.size': 16})
        for idx in var2plot.keys():    
            print('Current plot %s' % var2plot[idx])
            #create output folder
            fullpath = os.path.join(figpath, year, var2plot[idx])    
            if np.logical_not(os.path.exists(fullpath)):
                os.mkdir(fullpath)    
            #Create Figure
            fig = plt.figure(figsize=(15,5))
            ax = fig.add_subplot(111)    
            #Plot
            new[var2plot[idx]].T.plot(cmap='jet', vmax=Vmax[idx], vmin=Vmin[idx], cbar_kwargs={'extend': 'max'})
            #Setting labels            
            if not rpg.attrs['location']:
                rpg.attrs['location'] = 'Granada'
            ax.set_title('W-band radar %s | %s | %s' % (titleStr[idx],rpg.attrs['location'], datestr))
        #     cb.formatter.set_powerlimits((0, 0))
        #     cb.update_ticks()
            ax.set_xlabel('Time, UTC') 
            formatter = DateFormatter('%H:%M')
            plt.gca().xaxis.set_major_formatter(formatter)
            figstr = '%s_%s_%s_%s.png' % (radarNick, 'gr', var2plot[idx], datestr)
            finalpath = os.path.join(fullpath, figstr)              
            print('Saving %s' % finalpath)
            plt.savefig(finalpath, dpi=100, bbox_inches = 'tight')
            if os.path.exists(finalpath):
                print('Saving %s...DONE!' % finalpath)
            else:
                print('Saving %s... error!' % finalpath)
                
    #Read the data
    rpg = readers.rpg(mainPath, radarNick, 'gr', datestr)

    # print(rpg)

    #Convert Ze into dB
    dbze_tmp = retrieve_dbZe(rpg)

    #Merge rpg dataset and the dataArray with Ze in dB
    new = xr.merge([rpg, dbze_tmp])

    #Plots moments
    plot_rpg_moments(radarNick, new, rpg, datestr)

if __name__== "__main__":
    main()
