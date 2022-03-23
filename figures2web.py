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

__author__ = "Bravo-Aranda, Juan Antonio"
__license__ = "GPL"
__version__ = "1.1"
__maintainer__ = "Bravo-Aranda, Juan Antonio"
__email__ = "jabravo@ugr.es"
__status__ = "Production"

def main():
    parser = argparse.ArgumentParser(description="usage %prog [arguments]")
    parser.add_argument("-n", "--instrument",
        action="store",
        dest="instrument",
        required=True,
        help="Instrument [example: 'wradar'].")
    parser.add_argument("-i", "--input_files",
        action="store",
        dest="filelist",
        required=True,
        help="Input files [example: '/home/radar*.nc'].")
    parser.add_argument("-o", "--remote_directory",
        action="store",
        dest="rdir",
        default=".",
        help="Remote directory [default: '.'].")
    parser.add_argument("-w", "--overwrite",
        action="store",
        dest="overwrite",
        type=int,
        default=0,
        help="Overwrite flag number [default: 0].")        
    args = parser.parse_args()

    if args.instrument == 'wradar':    
        import rpg
        rpg.send2web(args.filelist, args.rdir, args.overwrite)
    elif args.instrument == 'mhc':
        import lidar
        lidar.send2web(args.instrument, args.filelist, args.rdir, args.overwrite)
    elif args.instrument == 'disdrometer':
        import disdrometer
        disdrometer.send2web(args.filelist, args.rdir, args.overwrite)
    elif args.instrument == 'disdrometer':
        import disdrometer
        disdrometer.send2web(args.filelist, args.rdir, args.overwrite)
        
if __name__== "__main__":
    main()
