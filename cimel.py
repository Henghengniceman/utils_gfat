#!/usr/bin/env python
# coding: utf-8

# In[264]:
import os
import sys
import glob
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 20})
# plt.switch_backend('agg')
from optparse import OptionParser

def reader(filename):
    """
    Reader for cimel files type *.lev15.
    Output in pandas format.      
    """    
    if os.path.isfile(filename):
        if filename.split('.')[-1] == 'lev15':
            dateparse = lambda x: pd.datetime.strptime(x, '%d:%m:%Y %H:%M:%S')
            data = pd.read_csv(filename, parse_dates={'datetime': ['Date(dd-mm-yy)', 'Time(hh:mm:ss)']}, date_parser=dateparse, index_col='datetime')
        else:
            print('ERROR: file does not match required format: %s' % '*.lev15')
            data = None
    else:
        data = None
        print('File not found!')
    return data

# def send2web():
#     import platform
#     #Information to access to the server
#     address = 'ftpwpd.ugr.es'
#     port = 21
#     user = 'gfat'
#     pwd = '4evergfat' # poner la contrase�a
#     server_figure_path='public_html/quicklooks/plots'

#     #Date
#     date2plot = dt.datetime.strptime(date2plot, '%Y%m%d')    
#     year=dt.datetime.strftime(date2plot, '%Y') 
#     month=dt.datetime.strftime(date2plot, '%m')
#     day=dt.datetime.strftime(date2plot, '%d')

#     #Main path
#     if platform.system() == 'Windows':        
#         mpath = 'Y:\\datos\\CLOUDNET'
#     else:
#         mpath = '/mnt/NASGFAT/datos/CLOUDNET'

#     #Full path
#     fullpath = os.path.join(path[field], filename[field])
    
#     for field in path.keys():    
#     fullpath = os.path.join(path[field], filename[field])
#     files2send = glob.glob(fullpath)
#     #print('files2send: %s' % files2send)
#     if files2send:
#         print('Files found as %s' % fullpath)
#         remoteDir = os.path.join(server_figure_path, station, rDir[field]) # poner el directorio correcto
#         print('remoteDir %s' % remoteDir)
#         for tmp_file in files2send:      
#             print('Current file to send: %s' % tmp_file)
#             ftp = ftplib.FTP(address)
#             ftp.set_pasv(True)
#             try:
#                 ftp.connect(address, port)
#                 ftp.login(user, pwd)
#                 ftp.cwd(remoteDir)
#                 print("pwd: %s" % ftp.pwd())                
#                 with open(tmp_file, "rb") as FILE:                    
#                     ftp.storbinary('STOR ' + os.path.basename(tmp_file), FILE)
#                 ftp.close()
#                 code = 1
#             except:
#                 code = 0
#                 ftp.close()                
#             if code:
#                 print('Succesfully sent: %s' %  os.path.join(remoteDir, os.path.basename(tmp_file)))
#             else:
#                 print(': Unable to send file %s' % os.path.basename(tmp_file))
#     else:
#         print('No files as %s' % fullpath)
        
def main():
    parser = OptionParser(usage="usage %prog [options]",
        version="%prog 1.0")
    parser.add_option("-s", "--station_name",
        action="store",
        dest="station",
        default="UGR",
        help="Aeronet nickname station [default: %default].")
    parser.add_option("-d", "--date2plot",
        action="store",
        dest="date2plot",
        default="20180118",
        help="Date to plot [default: %default].")
    parser.add_option("-t", "--dir_out",
        action="store",
        dest="dir_out",
        default=".",
        help="Output folder [default: %default].")        
    (options, args) = parser.parse_args()
    station=options.station    
    figpath=options.dir_out
    #nickLidar=options.lidar_name
    strdate=options.date2plot
    #month=dt.datime(options.date2plot, '%m')
    #day=dt.datime(options.date2plot, '%d')
    d=dt.datetime.strptime(strdate, '%Y%m%d')
    date2year = dt.datetime.strftime(d, '%y%m%d')
    year=dt.datetime.strftime(d, '%Y') 
    
    #Dictionary where include new stations of the icenet network
    stationDict={'UGR': 'Granada [UGR]', 'UGR-CP': 'Cerro Poyos', 'UGR-AU':'Albergue-UGR'}
    iniDirDict={'UGR': '/mnt/NASGFAT/datos/iCeNet/data/UGR/aeronet_Granada', 'UGR-CP': '/mnt/NASGFAT/datos/iCeNet/data/UGR/aeronet_Cerro_Poyos', 'UGR-AU':'/mnt/NASGFAT/datos/iCeNet/data/UGR/aeronet_Albergue_UGR'}
    outDirDict={'UGR': '/mnt/NASGFAT/datos/iCeNet/data/UGR/aeronet_Granada/quicklooks', 'UGR-CP': '/mnt/NASGFAT/datos/iCeNet/data/UGR/aeronet_Cerro_Poyos/quicklooks', 'UGR-AU':'/mnt/NASGFAT/datos/iCeNet/data/UGR/aeronet_Albergue_UGR/quicklooks'}
    
    #Current station
    location=stationDict[station]
    iniDir=iniDirDict[station]
    outDir=outDirDict[station]

    # In[291]:
    print(os.path.join(iniDir, '%s*.lev15' % date2year ))
    fileList = glob.glob(os.path.join(iniDir, year, '%s*.lev15' % date2year ))        
    if fileList:
        filename = fileList[0]  
    else:        
        sys.exit('No file found on %s' % strdate)
        
    # In[292]: Reading filename    
    df = reader(filename)
    if df != None:
        # In[295]: Plot figure
        fig = plt.figure(figsize=(20,7.5))
        var2plot =['AOT_340', 'AOT_380', 'AOT_440', 'AOT_500', 'AOT_675', 'AOT_870', 'AOT_1020', 'AOT_1640']
        colours={'AOT_340':'darkviolet', 'AOT_380': 'violet', 'AOT_440': 'blue', 'AOT_500': 'green', 'AOT_675': 'orange', 'AOT_870': 'red', 'AOT_1020':'darkred', 'AOT_1640':'saddlebrown'}
        ax1 = plt.gca()
        for var in var2plot:    
            df.plot(kind='line', y=var,color=colours[var], ax=ax1)    
        plt.ylim(0,1)
        plt.yticks(np.linspace(0., 1.1, num=11, endpoint=False))
        nextday = dt.datetime.strftime(d+dt.timedelta(days=1), '%Y%m%d')
        plt.xlim(dt.datetime.strptime(strdate, '%Y%m%d'), dt.datetime.strptime(nextday, '%Y%m%d'))
        plt.legend( prop={'size': 10})

        #twinx makes a new axes in the right side
        ax2 = ax1.twinx()
        df.plot(kind='line', y='440-870Angstrom',color='black', ax=ax2)    
        plt.legend(loc=2, prop={'size': 12})
        plt.ylim(0,1)
        ax2.set_ylim(0, 2)
        plt.yticks(np.linspace(0, 2.2, num=11, endpoint=False))

        #Properties of the figure
        ax1.grid()
        ax1.set_title('Aerosol Optical Tickness and\nAOT-related Angstrom exponent at 440-840nm on %s at %s' % (strdate, location), fontdict={'fontsize': 20})
        ax1.set_ylabel('AOT', color='k')
        ax1.set_xlabel('Time, HH:MM', color='k')
        ax2.set_ylabel('Angstrom exponent 440-870', color='k')
        formatter = matplotlib.dates.DateFormatter('%H:%M')
        ax2.xaxis.set_major_formatter(formatter)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0)
        
        debugging = False

        if debugging:
            plt.show()
        else:
            # In[296]: Save the figure
            figname=os.path.join(outDir,'%s_aot_ae_%s.png' % (station.lower(), strdate))
            fig.savefig(figname, bbox_inches='tight', dpi=100)

            if figname:
                print('Figure succesfully created on %s' % strdate)
            else:
                print('Figure NOT created on %s' % strdate)
    
if __name__== "__main__":
    main()