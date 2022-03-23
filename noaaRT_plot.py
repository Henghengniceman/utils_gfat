#!/usr/bin/env python
# coding: utf-8

# In[52]:

import os
import glob
import sys
import platform
import argparse
import matplotlib
import pandas as pd
import numpy as np
import datetime as dt
from optparse import OptionParser
matplotlib.use('Agg') # set the backend before importing pyplot
import matplotlib.pyplot as plt
import plot #./lib/python2.7/site-packages/utils_gfat
import readers
import args_parser as ag

__version__ = '1.0.0'
__author__ = 'Juan Antonio Bravo-Aranda'


# script description
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PROG_NAME = 'noaaRT_plot'
PROG_DESCR = 'create plots from NOAA Real Time files'


def local_argparser():
    """function to parse input arguments
    return in put arguments as a dictionnary"""

    # argparser default arguments (logs)
    parser = ag.init_argparser(PROG_DESCR, BASE_DIR, PROG_NAME, __version__)

    # arguments specific to the script
    # ------------------------------------------------------------------------

    parser.add_argument("--dpi",
                        action="store",
                        dest="dpi",
                        default=100,
                        type=int,
                        help="Input folder [default: %default].")
    parser.add_argument("--coeff",
                        action="store",
                        dest="coeff",
                        default=2.,
                        type=float,
                        help="Parameter to change figure size [default: %default].")
    return parser

def parse_args(input_args):
    """parse inputs arguments and return dict"""

    # init args
    parser = local_argparser()

    # parse args
    try:
        args = parser.parse_args(input_args)
    except argparse.ArgumentError as exc:
        print('\n', exc)
        sys.exit(1)

    return vars(args)

def noaaRT(filepath):
    """This program can be used for reading NOAA-RT files. Example:
       readers.noaaRT(Y:\\datos\\IN-SITU\\Data\\noaaRT\\20191801.csv)
       Output is in pandas format
    """
    # Reading file
    # ----------------------------------------------------------------------------
    if os.path.isfile(filepath):
        dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M')
        df=pd.read_csv(filepath, names=['DateTime', 'pm25', 'eb'], date_parser=dateparse, index_col='DateTime')
    else:
        df=''
        print('File not found!')
    
    # cleaning file
    # ----------------------------------------------------------------------------
    for key in df.keys():
        df[key][df[key]>9000.]=np.nan        
    return df

def plot_noaaRT(data, datenum, plt_conf):       
    colours={'eb':'black', 'pm25': 'red'}

    # Figure
    # ----------------------------------------------------------------------------
    fig = plt.figure(figsize= plt_conf['fig_size'], )

    # AXIS 1
    # ----------------------------------------------------------------------------
    ax1 = plt.gca()
    if not np.sum(np.isnan(data.eb))==np.size(data.eb):
        data.plot(kind='line', y='eb', color=colours['eb'], linewidth=2.0, ax=ax1, label='EBC')    

    #Properties X-axis
    plt.xlim(datenum, datenum+dt.timedelta(days=1))
    ax1.set_xlabel('Time, HH:MM', color='k', fontdict={'fontsize':  plt_conf['font_size']},)

    #Properties Y-axis
    plt.ylim(0,20)
    ax1.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
    ax1.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(1))
    ax1.set_ylabel(r'Black carbon, $\mu$g/$m^3$', color='k', fontdict={'fontsize':  plt_conf['font_size']})

    #General properties axis
    ax1.tick_params(which='major', length=5, direction='out', labelsize=15)
    ax1.tick_params(which='minor', length=4, direction='in')
    ax1.set_facecolor('xkcd:white')
    ax1.legend(prop={'size': 20}, loc=2)

    ax2 = ax1.twinx()
    if not np.sum(np.isnan(data.pm25))==np.size(data.pm25):
        data.plot(kind='line', y='pm25',color=colours['pm25'], linewidth=3.0, ax=ax2, label='PM2.5')    

    #General properties axis
    plot.title1(plt_conf['title1'], plt_conf['coeff'])
    plot.title2(datenum.strftime('%Y/%m/%d'), plt_conf['coeff'])
    plot.title3('{} ({:.1f}N, {:.1f}E)'.format(plt_conf['location'], plt_conf['lat'], plt_conf['lon']), plt_conf['coeff'])
    ax2.legend(prop={'size': 20})
    ax2.tick_params(which='major', length=5, direction='out', labelsize=15)
    ax2.tick_params(which='minor', length=4, direction='in')
    ax2.grid()

    #Properties X-axis
    formatter = matplotlib.dates.DateFormatter('%H:%M')
    ax2.xaxis.set_major_formatter(formatter)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0)
    
    #Properties Y-axis
    plt.ylim(0,1)
    ax2.set_ylim(0, 40)
    ax2.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
    ax2.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(1))
    ax2.set_ylabel(r'PM2.5, $\mu$g/$m^3$', color='k', fontdict={'fontsize':  plt_conf['font_size']})
    
    # matplotlib.rc('ytick', labelsize=20) 

    # logo
    # ----------------------------------------------------------------------------
    plot.add_GFAT_logo([0.9, 0.06, 0.10, 0.10])

    plt.show()    
    return fig

# In[44]:

def main(raw_args):
    print('Starting code')
    print(raw_args)
    # Inputs
    # ----------------------------------------------------------------------------
    # parse inputs arguments
    print('Reading arguments...')
    args = parse_args(raw_args)
    print('Reading arguments...done!')

    #Reading inputs
    # -----------------------------------------------------------
    # print('args input: %s' % args['input'][0])
    print('Converting arguments...')
    file=args['input'][0][0]
    print('file to plot %s' % file)
    print('type file to plot %s' % type(file))
    figpath=args['output_dir']
    figname=args['output_name']
    # nickLidar=args['lidar_name']
    dpi=int(args['dpi'])
    COEFF = float(args['coeff'])
    datenum=args['date']
    year=dt.datetime.strftime(datenum, '%Y') 
    month=dt.datetime.strftime(datenum, '%m')
    day=dt.datetime.strftime(datenum, '%d')    
    print(year)
    print(month)
    print(day)
   
    # date4file = dt.datetime.strftime(datenum, '%Y%d%m')
    # file='Y:\\datos\\IN-SITU\\Data\\noaaRT\\%s.csv' % date4file
    # print(file)

    # Read File
    # ----------------------------------------------------------------------------
    data=readers.noaaRT(file)

    # Figure properties values
    # ----------------------------------------------------------------------------
    plt_conf =  {
    'title1': 'Elemental black carbon and PM2.5',
    'coeff': COEFF,
    'location': 'Granada, UGR-IISTA, GFAT',
    'lat': -3.2,
    'lon': 37.2,    
    'dpi': dpi,
    'fig_size': (15,10),
    'font_size': 20,
    'y_min': 0,
    'ymax_ebc': 20.,
    'ymax_pm25': 40.} 

    # Figure properties values
    # ----------------------------------------------------------------------------
    fig = plot_noaaRT(data, datenum, plt_conf)   

    # Save figure
    # ----------------------------------------------------------------------------
    if figname=='auto':
        figname='noaaRT_%s.png' % dt.datetime.strftime(datenum, '%Y%m%d') 
        print(figname)    
    fig.savefig(os.path.join(figpath, figname), bbox_inches='tight')

if __name__== "__main__":
    sys.exit(main(sys.argv[1:]))
