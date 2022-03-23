#!/usr/bin/env python
# coding: utf-8

# In[52]:

import os
import glob
import sys
import argparse
import datetime as dt
from optparse import OptionParser
import disdrometer as dd
import parsivelConverter1b as pc1b

__version__ = '1.0.0'
__author__ = 'Juan Antonio Bravo-Aranda'

# script description
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PROG_NAME = 'disdrometerConverter'
PROG_DESCR = 'converting raw data to netCDF files.'

def main():
    parser = OptionParser(usage="usage %prog [options]",
        version="%prog 1.0")
    parser.add_option("-s", "--station_name",
        action="store",
        dest="station",
        default="UGR",
        help="Measurement station [default: %default].")
    parser.add_option("-d", "--date2plot",
        action="store",
        dest="date2plot",
        default="20191121",
        help="Date to plot [default: %default].")
    parser.add_option("-o", "--dir_out",
        action="store",
        dest="dir_out",
        default=".",
        help="Output folder [default: %default].")
    parser.add_option("-i", "--dir_in",
        action="store",
        dest="dir_in",
        default=".",
        help="Input folder [default: %default].")                
    (options, args) = parser.parse_args()
    dir0 = options.dir_in    
    dir2 = options.dir_out   
    strdate = options.date2plot
    station = options.station
    date_ = dt.datetime.strptime(strdate,'%Y%m%d')
    year = dt.datetime.strftime(date_,'%Y')
    month = dt.datetime.strftime(date_,'%m')
    file0a = os.path.join(dir0, year, month, '%s_%s.mis' % (dt.datetime.strftime(date_,'%Y%m%d'), station))
    if os.path.isfile(file0a):
        print('%s found!' % file0a)    
        file1a = file0a.replace('0a', '1a')
        pc1b.cleanMIS(file0a, file1a)
        
        if os.path.isfile(file1a):
            print('%s found!' % file1a)            
            file1b = file1a.replace('1a', '1b')
            file1b = file1b.replace('mis', 'nc')
            dd.to_nc(file1a, file1b)    
        
            if os.path.isfile(file1b):    
                print('%s successfully converted!' % file1b)
            else:
                print('%s conversion to 1b-level FAILED!' % file0a)
        else:    
            print('%s conversion to 1a-level FAILED!' % file0a)
    else:    
        print('%s not found.' % file0a)        

if __name__== "__main__":
    main()
