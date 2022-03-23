#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os 
import sys
import glob
import argparse
import numpy as np
import netCDF4 as nc
import datetime as dt
import pdb

import matplotlib as mpl
mpl.use('Agg') # set the backend before importing pyplot
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from matplotlib.dates import DateFormatter
from optparse import OptionParser
from utils_gfat import plot
from utils_gfat import args_parser as ag

__version__ = '1.0.0'
__author__ = 'Juan Antonio Bravo-Aranda'

# script description
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PROG_NAME = 'quicklook_lidar'
PROG_DESCR = 'flux to create quicklooks for lidar 1a'

def local_argparser():
    """function to parse input arguments
    return in put arguments as a dictionnary"""

    # argparser default arguments (logs)
    parser = ag.init_argparser(PROG_DESCR, BASE_DIR, PROG_NAME, __version__)

    # arguments specific to the script
    # ------------------------------------------------------------------------

    # load configuration file
    parser.add_argument('-c', '--conf',
                        type=ag.check_conf_file,
                        help='configuration file (*.py)')
    # parser.add_argument("-l", "--inst_name",
    #                     action="store",
    #                     dest="inst_name",
    #                     required=True,
    #                     help="Lidar name [e.g.,: mhc].")
    parser.add_argument("--dpi",
                        action="store",
                        dest="dpi",
                        default=100,
                        type=int,
                        help="Input folder [default: %default].")
    parser.add_argument("-g", "--gapsize",
                        action="store",
                        dest="gapsize",
                        default="default",                        
                        help="Temporal gap in minutes [default: %default].")                               
    parser.add_argument("--coeff",
                        action="store",
                        dest="coeff",
                        default=2.,
                        type=float,
                        help="Parameter to change figure size [default: %default].")
    parser.add_argument("-a", "--altitude_max",
                        action="store",
                        dest="range_limit",
                        default=14,
                        type=int,
                        help="Maximum height plot [im km.] [default: %default].")        
    parser.add_argument("--channel",
                        action="store",
                        dest="channel",
                        default=1,
                        type=int,
                        help="Channel ID in licel file measurement [default: %default].")    

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

def read_data(list_files, date, channel):
    """read data from netCDF files"""
    print('Reading netCDF files')

    print(list_files)

    data = {}

    # open all files
    nc_ids = [nc.Dataset(file_) for file_ in list_files]

    # localization of instrument
    data['lat'] = nc_ids[0].variables['lat'][:]
    data['lon'] = nc_ids[0].variables['lon'][:]
    data['alt'] = nc_ids[0].variables['altitude'][:]
    data['location'] = nc_ids[0].site_location.split(',')[0]
    data['instr'] = nc_ids[0].system

    # read alt (no need to concantenate)
    data['range'] = nc_ids[0].variables['range'][:]

    # wavelength (no need to concantenate)
    tmp = nc_ids[0].variables['wavelength'][:]
    data['wavelength'] = tmp 
    data['wavelength_units'] = nc_ids[0].variables['wavelength'].units  

    # detection_mode (no need to concantenate)
    tmp = nc_ids[0].variables['detection_mode'][:]
    data['detection_mode'] = tmp

    # polarization (no need to concantenate)
    tmp = nc_ids[0].variables['polarization'][:]
    data['polarization'] = tmp

    # read time 
    units = nc_ids[0].variables['time'].units #'days since %s' % dt.datetime.strftime(date, '%Y-%m-%d %H:%M:%S')
    tmp = [nc.num2date(nc_id.variables['time'][:], units) for nc_id in nc_ids]
    data['raw_time'] = np.concatenate(tmp)

    # check if any data available
    # print(date)
    time_filter = (data['raw_time'] >= date) & (data['raw_time'] < date + dt.timedelta(days=1))
    if not np.any(time_filter):
        return None

    # RCS
    tmp = [nc_id.variables['rcs_%02d' % channel][:] for nc_id in nc_ids]
    data['rcs'] = np.ma.filled(np.concatenate(tmp, axis=0))    

    # Background
    tmp = [nc_id.variables['bckgrd_rcs_%02d' % channel][:] for nc_id in nc_ids]
    data['background'] = np.ma.filled(np.concatenate(tmp, axis=0), fill_value=np.nan)       
    
    # close all files
    [nc_id.close() for nc_id in nc_ids]

    return data

# In[4]:
def bckgCorrection(data, channel, range_limit):
    range2plot = data['range'][ data['range']<=range_limit*1000. ] 
    lrcs = data['rcs'][ :, data['range']<=range_limit*1000. ]
    lbck = data['background']
    mrange = np.tile(np.square(range2plot), (data['raw_time'].size,1))    
    tmp_signal = lrcs / mrange
    mbckg = np.tile(lbck, (np.size(range2plot),1)).T
    rcs2plot = (tmp_signal - mbckg) * (mrange)
    return rcs2plot, range2plot

def plot_rcs(data, date, plt_conf, f_png):
    """plot RCS"""
    
    print('Creating figure %s' % f_png)
    # vwm = np.ma.masked_where(data['w_error'] > plt_conf['vw_error_threshold'], data['w'])    
    fig, axes = plt.subplots(nrows=1, figsize=(15,6))
    cmap = mpl.cm.jet
    bounds = np.linspace(plt_conf['v_min'], plt_conf['v_max'], plt_conf['v_n'])
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    range_km = data['range2plot']/1000.
    q = axes.pcolormesh(data['raw_time'], range_km, data['rcs2plot'].T,
                        cmap=cmap,
                        vmin=plt_conf['v_min'],
                        vmax=plt_conf['v_max'],
                        norm=norm
                       )
    q.cmap.set_over('white')
    # q.cmap.set_under('darkblue')
    # q.update_ticks()
    cb = plt.colorbar(q, ax=axes,
                      #ticks=bounds,
                      extend='max', 
                      format=plot.OOMFormatter(plt_conf['power_plot'], mathText=True))
    # cb.formatter.set_powerlimits((0, 0))

    # search for holes in data
    # --------------------------------------------------------------------
    plot.gapsizer(axes, data['raw_time'], range_km, plt_conf['gapsize'], '#c7c7c7')

    # Setting axes 
    # --------------------------------------------------------------------
    mf = mpl.ticker.FuncFormatter(plot.tmp_f)
    axes.xaxis.set_major_formatter(mf)
    hours = mdates.HourLocator(range(0, 25, 3))
    date_fmt = mdates.DateFormatter('%H')
    axes.xaxis.set_major_locator(hours)
    axes.xaxis.set_major_formatter(date_fmt)

    min_date = dt.datetime.strptime(dt.datetime.strftime(data['raw_time'].min().date(), '%Y%m%d'), '%Y%m%d') 
    max_date = dt.datetime.strptime(dt.datetime.strftime(data['raw_time'].max().date(), '%Y%m%d'), '%Y%m%d') 
    # print(dt.datetime.strftime(min_date, '%Y%m%d-%H%M%S'))
    # print(dt.datetime.strftime(max_date + dt.timedelta(hours=23, minutes=59, seconds=59), '%Y%m%d-%H%M%S'))

    axes.set_xlim(min_date - dt.timedelta(minutes=30), max_date + dt.timedelta(hours=23, minutes=59, seconds=59))
    axes.set_ylim(plt_conf['y_min'], plt_conf['y_max'])
    plt.grid(True)

    axes.set_xlabel('Time, $[UTC]$')
    axes.set_ylabel('Altitude, $[km]$')
    cb.ax.set_ylabel('Range corrected signa,l $[a.u.]$')

    # title
    # ----------------------------------------------------------------------------
    plot.title1(plt_conf['title1'], plt_conf['coeff'])
    plot.title2(date.strftime('%Y/%m/%d'), plt_conf['coeff'])
    plot.title3('{} ({:.1f}N, {:.1f}E)'.format(plt_conf['location'], float(data['lat']), float(data['lon'])), plt_conf['coeff'])

    # Font Size
    # ----------------------------------------------------------------------------
    # for item in ([axes.title, axes.xaxis.label, axes.yaxis.label] +
    #              axes.get_xticklabels() + axes.get_yticklabels()):
    #     item.set_fontsize(20)    
    
    # logo
    # ----------------------------------------------------------------------------
    plot.add_GFAT_logo([0.85, 0.01, 0.15, 0.15])

    # save
    # ----------------------------------------------------------------------------
    print('Saving %s' % f_png)
    #ignore by message
    fig.savefig(f_png, dpi=plt_conf['dpi'])
    # plt.savefig(f_png, dpi=dpi, bbox_inches = 'tight')
    plt.close()
    

# In[59]:
def main(raw_args):

    # parse inputs arguments
    args = parse_args(raw_args)

    #Reading inputs
    # -----------------------------------------------------------
    # print('args input: %s' % args['input'][0])
    
    file_list=args['input'][0]
    figpath=args['output_dir']
    figname=args['output_name']
    # nickLidar=args['lidar_name']
    dpi=int(args['dpi'])
    gapsize = args['gapsize']
    COEFF = float(args['coeff'])
    range_limit = int(args['range_limit'])
    print('This is range limit %d' % range_limit)
    channel=int(args['channel'])
    # date2plot=dt.datetime.strftime(args['date'], '%Y%m%d')
    datenum=args['date']
    year=dt.datetime.strftime(datenum, '%Y') 
    month=dt.datetime.strftime(datenum, '%m')
    day=dt.datetime.strftime(datenum, '%d')    
    print(year)
    print(month)
    print(day)
    
    # Dictionary of detection mode and polarization channels 
    # -----------------------------------------------------------    
    mode={0:'a', 1: 'p'}
    pol={0:'t', 1: 'p', 2:'c'}
    nick={'MULHACEN': 'mhc', 'VELETA': 'vlt'}
        
    # Color bar limits
    # -----------------------------------------------------------    
    clims = {'355xpa': (0,3e6),'532xpa': (0,8e6),'532xpp': (0,2.5e8), '532xca': (0,8e6), '1064xta': (0,8e7)}
    power_plot = {'355xpa': 6, '532xpa': 6,'532xpp': 8, '532xca': 6, '1064xta': 7}

    #Read data
    # -----------------------------------------------------------    
    data = read_data(file_list, datenum, channel)

    #Background correction   
    # -----------------------------------------------------------        
    print('Background substraction...')         
    data['rcs2plot'], data['range2plot'] = bckgCorrection(data, channel, range_limit)    

    # Setting the parameter gapsize
    # -----------------------------------------------------------        
    if gapsize == 'default':        
        dif_time = data['raw_time'][1:] - data['raw_time'][0:-1]
        dif_seconds = [tmp.seconds for (i, tmp) in enumerate(dif_time)]                        
        # print(dif_seconds)
        HOLE_SIZE = 2*int(np.ceil((np.median(dif_seconds)/60))) #HOLE_SIZE is defined as the median of the resolution fo the time array (in minutes)        
        print('HOLE_SIZE parameter automatically retrieved to be %d.' % HOLE_SIZE)         
    else:
        HOLE_SIZE = int(gapsize)
        print('HOLE_SIZE set by the user: %d (in minutes)' % HOLE_SIZE)

    #Channel name
    # -----------------------------------------------------------    
    modestr = mode[data['detection_mode'][channel]]
    polstr = pol[data['polarization'][channel]]       
    channelstr = '%sx%s%s' % (int(data['wavelength'][channel]), polstr, modestr)
    
    # Creating figure name
    # -----------------------------------------------------------    
    if figname=='auto':
        figname='%s_1a_rcs-%s_%s%s%s.png' % (nick[data['instr']], channelstr,year, month, day)         
        print(figname)
    f_png = os.path.join(figpath, figname) #create output filename

    # Setting figure configuration
    # -----------------------------------------------------------        
    plt_conf =  {
    'title1': '%s range corrected signal at %d %s' % (data['instr'], data['wavelength'][channel], data['wavelength_units']),
    'location': 'Granada, UGR-IISTA, GFAT',
    'coeff': COEFF,
    'gapsize': HOLE_SIZE,
    'dpi': dpi,
    'fig_size': (16,5),
    'font_size': 16,
    'y_min': 0,
    'y_max': range_limit,
    'power_plot': power_plot[channelstr],
    'v_min': clims[channelstr][0],
    'v_max': clims[channelstr][1],
    'rcs_error_threshold':1.0,
    'v_n': 64, }
    
    #Plot quicklook
    # -----------------------------------------------------------        
    plot_rcs(data, datenum, plt_conf, f_png)

    #Check created quicklook 
    # -----------------------------------------------------------        
    if os.path.exists(f_png):
        print('Saving %s...DONE!' % f_png)
    else:
        print('Saving %s... error!' % f_png)

if __name__== "__main__":
    sys.exit(main(sys.argv[1:]))

#%%
