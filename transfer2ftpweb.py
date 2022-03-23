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

def send2web(files2send, remote_directory, overwrite=False):
    """
    Send quicklooks to the browser webpage.
    Inputs:
    - files2send: List of radar figures (i.e, ['/c/figure1.png','/c/figure2.png']) (list)
    - remote_directory: directory in the ftp where figures will be located. (string)
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
    if overwrite == 'True':
        print('Files will be overwritten.')
        file2transfer=True                    
    elif overwrite == 'False':
        print('Files won''t be overwritten.')
    print(files2send)
    if files2send and files2send != None:        
        for tmp_file in files2send:            
            ftp = ftplib.FTP(address)
            ftp.set_pasv(True)
            try:
                ftp.connect(address, port)
                ftp.login(user, pwd)
                ftp.cwd(remote_directory)                
                if overwrite == 'True':                    
                    file2transfer=True                    
                elif overwrite == 'False':                    
                    remote_lines=[]
                    ftp.retrlines('LIST', remote_lines.append)
                    remote_files = [line_.split(' ')[-1] for line_ in remote_lines]                    
                    file2transfer = os.path.basename(tmp_file) not in remote_files                    
                if file2transfer:
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
                print('Skipped. File already exists in ftp: %s' %  os.path.join(remote_directory, os.path.basename(tmp_file)))
            else:
                print('Error. Unable to send file %s' % os.path.basename(tmp_file))
    else:
        print('No file found as %s' % files2send)

def main():
    parser = argparse.ArgumentParser(description="usage %prog [arguments]")
    parser.add_argument("-n", "--nickname_instrument",
        action="store",
        dest="instrument",
        required=True,
        help="Nickname of the instrument [example: 'rpg' for the cloud radar].")
    parser.add_argument("-o", "--overwrite",
        action="store",
        dest="overwrite",
        required=False,
        default='False',
        help="Permission to overwrite figures already in the FTP [values: 'True' or 'False']. Default: False")                 
    parser.add_argument("-y", "--year",
        action="store",
        dest="year",
        required=False,
        default='all',
        help="Only figures of the provided year will the transfer. To transfer the whole data set, use 'all' [values: 'yyyy' or 'all']. Default: 'all'")        
    # parser.add_argument("-i", "--initial_date",
    #     action="store",
    #     dest="date_ini,
    #     required=True,
    #     help="Initial date [example: '20190131'].")         
    # parser.add_argument("-e", "--final_date",
    #     action="store",
    #     dest="dateend",
    #     default=".",
    #     help="Final date [example: '20190131'].")
    # parser.add_argument("-d", "--datadir",
    #     action="store",
    #     dest="path1a",
    #     default="GFATserver",
    #     help="Path where date-hierarchy files are located [example: '~/data/1a'].")
    # parser.add_argument("-f", "--figuredir",
    #     action="store",
    #     dest="figpath",
    #     default="GFATserver",
    #     help="Path where figures will be saved [example: '~/radar/quicklooks'].")
    args = parser.parse_args()

    print('Start transfer2ftpweb')

    dic2send = {}
    rdir = {}

    if args.year == 'all':
        year = '*'
    else:
        year = args.year

    if args.instrument == 'all':
        instruments = ['rpg', 'mhc', 'ceilo', 'mwr']
    else:
        instruments = [args.instrument, ]

    print(instruments)

    for _instrument in instruments:                
        if _instrument == 'rpg':                
            dic2send['dBZe'] = glob.glob('/mnt/NASGFAT/datos/rpgradar/quicklooks/%s/dBZe/*.png' % year)
            rdir['dBZe'] = '/public_html/quicklooks/plots/granada/radar/dbZe'
            dic2send['vm'] = glob.glob('/mnt/NASGFAT/datos/rpgradar/quicklooks/%s/vm/*.png' % year)
            rdir['vm'] = '/public_html/quicklooks/plots/granada/radar/vm'
            dic2send['sigma'] = glob.glob('/mnt/NASGFAT/datos/rpgradar/quicklooks/%s/sigma/*.png' % year)
            rdir['sigma'] = '/public_html/quicklooks/plots/granada/radar/sigma'
        elif _instrument == 'mhc':
            dic2send['rcs'] = glob.glob('/mnt/NASGFAT/datos/MULHACEN/quicklooks/532xpa/%s/*.png' % year)
            rdir['rcs'] = '/public_html/quicklooks/plots/granada/mulhacen/quicklooks'
            # dic2send['depolarization'] = glob.glob('/mnt/NASGFAT/datos/MULHACEN/quicklooks/depolarization/%s/*.png' % year)
            # rdir['depolarization'] = '/public_html/quicklooks/plots/granada/mulhacen/quicklooks'            
            # dic2send['inversiones'] = glob.glob('/mnt/NASGFAT/datos/MULHACEN/quicklooks/depolarization/*/*.png')
            # rdir['inversiones'] = '/public_html/quicklooks/plots/granada/mulhacen/inversiones'
        elif _instrument == 'ceilo':
            dic2send['rcs'] = glob.glob('/mnt/NASGFAT/datos/iCeNet/data/UGR/quicklooks/%s/*.jpg' % year)
            rdir['rcs'] = '/public_html/quicklooks/plots/granada/mwr/quicklooks'
        elif _instrument == 'mwr':
            dic2send['CMP_TPC'] = glob.glob('/mnt/NASGFAT/datos/RPG-HATPRO/quicklooks/COMP_TPC/%s/*.png' % year)
            rdir['CMP_TPC'] = '/public_html/quicklooks/plots/granada/ceilo/'
        elif _instrument == 'noaaRT':
            dic2send['noaaRT'] = glob.glob('/mnt/NASGFAT/datos/IN-SITU/Quicklooks/noaaRT/noaaRT_*%s*.png' % year)
            rdir['noaaRT'] = '/public_html/quicklooks/plots/granada/noaaRT/'

        # elif instrument == 'doppler':
        # elif instrument == 'aeronet_granada':
        # elif instrument == 'aeronet_Cerro_Poyos':
        #      
        if bool(dic2send) and  bool(rdir):
            for _key in dic2send.keys():
                print(_key)
                if dic2send[_key]:
                    print(dic2send[_key])
                    send2web(dic2send[_key], rdir[_key], args.overwrite)

if __name__== "__main__":
    main()
