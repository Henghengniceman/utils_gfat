#!/usr/bin/env python
# coding: utf-8

import os
import sys
import argparse
import datetime as dt
from optparse import OptionParser
import disdrometer as dd

__version__ = '1.0.0'
__author__ = 'Juan Antonio Bravo-Aranda'


# script description
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PROG_NAME = 'noaaRT_plot'
PROG_DESCR = 'create plots from NOAA Real Time files'

def main():
    parser = OptionParser(usage="usage %prog [options]",
        version="%prog 1.0")
    parser.add_option("-s", "--station_name",
        action="store",
        dest="station",
        default="UGR",
        help="Aeronet nickname station [default: %default].")
    parser.add_option("-i", "--inidate",
        action="store",
        dest="inidate",
        default="20191121",
        help="Initial date to plot in format yyymmdd [default: %default].")
    parser.add_option("-e", "--enddate",
        action="store",
        dest="enddate",
        default="20191122",
        help="Final date to plot in format yyymmdd [default: %default].")
    parser.add_option("-t", "--dir_out",
        action="store",
        dest="dir_out",
        default=".",
        help="Output folder [default: %default].")        
    (options, args) = parser.parse_args()
    station=options.station
    figpath=options.dir_out
    #nickLidar=options.lidar_name
    inidate=options.inidate
    enddate=options.enddate
    dateini=dt.datetime.strptime(inidate, '%Y%m%d')
    dateend=dt.datetime.strptime(enddate, '%Y%m%d')    
    #Dictionary where include new stations 
    stationDict={'UGR': 'Granada [UGR]'} #, 'UGR-CP': 'Cerro Poyos', 'UGR-AU':'Albergue-UGR'}
    iniDir='/mnt/NASGFAT/datos/parsivel/1b'
    accuDSDDir='/mnt/NASGFAT/datos/parsivel/figures/accuDSD'
    quicklookDSDDir='/mnt/NASGFAT/datos/parsivel/figures/quicklooks'
    
    #Current station
    location=stationDict[station]
    figuredir=os.path.join(accuDSDDir, station)

    daterange = [dt.datetime.strftime(dateini,'%Y-%m-%d %H:%M:%S'), dt.datetime.strftime(dateend,'%Y-%m-%d %H:%M:%S')]    

    #Accumulated spectra
    print('Plotting accumulated DSD...')
    figuredir=os.path.join(accuDSDDir, station)
    dd.accumulatedSpectrum(iniDir, figuredir, 'UGR', daterange)
    print('Plotting accumulated DSD... done!')
    
    #Quicklook
    print('Plotting DSD quicklook...')
    figuredir=os.path.join(quicklookDSDDir, station)
    dd.quicklook(['diameter', 'velocity'], iniDir, figuredir, 'UGR', daterange)
    print('Plotting DSD quicklook... done!')

if __name__== "__main__":
    sys.exit(main())
    