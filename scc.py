#!/usr/bin/env python
"""
SCC GFAT

Functionality: Chain Process of Lidar Measurements from Raw format to SCC server processing
    1. Convert Lidar Raw Data (0a) into SCC format
    2. Make some plots of the SCC format data
    3. Upload SCC format data to SCC server where data is processed
    4. Download Data processed in SCC server
    5. Make some plots of the processed data downloaded from SCC server

Usage:
run scc.py
    -i YYYYMMDD
    -d DATA_PATH [DATA_PATH/LIDAR/...]
    -c CAMPAIGN CONFIG FILE JSON

TODO:
    + Add input arguments for hour_ini, hour_end, hour_res, timestamp, slot_name_type in case no campaign_file is used. Default values are currently set in get_campaign_config function.
"""

import os
import sys
import glob
import platform
import importlib
import shutil
import re
import argparse
from distutils.dir_util import mkpath
import itertools
import numpy as np
import xarray as xr
import pandas as pd
import datetime as dt
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import time
import signal
import json
import logging
from multiprocessing import Pool
import pickle
import smtplib, ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import COMMASPACE, formatdate
import pdb

""" 3rd party code dependencies """
MODULE_DIR = os.path.dirname(sys.modules[__name__].__file__)
sys.path.insert(0, MODULE_DIR)
import solar
from atmospheric_lidar.scripts import licel2scc # copied to utils_gfat
import scc_access as sa  # copied to utils_gfat
from lidar_preprocessing import mulhacen_pc_peak_correction
import config

"""
Functions for deriving SCC info directly from measurement folder/files
(Currently Not Used)
"""
def get_scc_code_from_measurement_folder(meas_folder, campaign):
    """
    get scc_code from measurement folder.
    Para todas las scc posibles (2) de la campaña, se busca si los canales que las definen existen en la medida.
    En caso de que existan las 2, la scc correcta es la que tiene más canales.
    Input:
    - measurement folder
    - campaign name
    """

    assert isinstance(meas_folder, str), "meas_folder must be String Type"
    assert isinstance(campaign, str), "campaign must be String Type"

    if campaign is not None:
        campaign_info, campaign_scc_fn = get_campaign_info(campaign)
        scc_cfg = campaign_info.scc_cfg
        scc_codes = campaign_info.scc_codes
        CustomLidarMeasurement = licel2scc.create_custom_class(campaign_scc_fn, use_id_as_name=True)
        rm_files = glob.glob(os.path.join(meas_folder, "R*"))
        if len(rm_files)>0:
            rm_file = [rm_files[0]]
            measurement = CustomLidarMeasurement(rm_file)
            channels_in_rm = list(measurement.channels.keys())
            sccs = len(scc_codes)
            exist_scc = [False]*sccs
            channels_in_scc = [0]*sccs
            for i, i_scc in enumerate(scc_cfg):
                exist_scc[i] = all(j in channels_in_rm for j in scc_cfg[i_scc]['channels'])
                channels_in_scc[i] = len(scc_cfg[i_scc]["channels"])
            scc_code = [b for a, b in zip(exist_scc, scc_codes) if a]
            if len(scc_code)==0:  # no estan los scc posibles en la medida
                scc_code = None
            elif len(scc_code)==1: # hay 1.
                scc_code = int(scc_code[0])
            elif len(scc_code)==2:  # los dos son posibles. elegimos el que tiene mas canales
                max_chan = channels_in_scc == np.max(channels_in_scc)
                scc_code = [b for a, b in zip(max_chan, scc_codes) if a]
                scc_code = int(scc_code[0])
        else:
            scc_code = None
    else:
        scc_code = None
    return scc_code


def get_campaign_info(campaign, scc_config_directory=None):
    """

    """
    # TODO: make scc_config_directory optional ¿?. Enlazar con crear archivo de configuracion
    try:
        if scc_config_directory is None:
            scc_config_directory = os.path.join(os.path.abspath(MODULE_DN), "scc_configFiles")
        if campaign=="covid":  # there is only one campaign so far. 
            campaign_scc_fn = "scc_channels_covid19.py"
            campaign_scc_fn = os.path.join(scc_config_directory, campaign_scc_fn)
            campaign_info = import_campaign_scc(campaign_scc_fn)
            campaign_info.scc_codes = [*campaign_info.scc_cfg]
        else:
            campaign_info = None
            campaign_scc_fn = None
    except:
        campaign_info = None
        campaign_scc_fn = None

    return campaign_info, campaign_scc_fn


def import_campaign_scc(campaign_scc_fn):
    """

    """
    # TODO: darle una vuelta
    try:
        sys.path.append(os.path.dirname(campaign_scc_fn))
        campaign_scc = importlib.import_module(os.path.splitext(os.path.basename(campaign_scc_fn))[0])
    except:
        logger.warning("ERROR. importing scc-channels info from %s" % campaign_scc_fn)
        campaign_scc = None

    return campaign_scc


""" """
__author__ = "Bravo-Aranda, Juan Antonio"
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Bravo-Aranda, Juan Antonio"
__email__ = "jabravo@ugr.es"
__status__ = "Production"


""" directories of interest  """
# Root Directory (in NASGFAT)  according to operative system
DATA_DN = config.DATA_DN

# Source Code Dir
MODULE_DN = os.path.dirname(sys.modules[__name__].__file__)

# SCC Config Directory
SCC_CONFIG_DN = os.path.join(os.path.abspath(MODULE_DN), "scc_configFiles")

# Run Dir
RUN_DN = os.getcwd()


""" Common Variables along the module """
STATION_ID = "gra"
LIDAR_SYSTEMS = {
    "ALHAMBRA": {'nick': 'alh',
                 'code': ''},
    "MULHACEN": {'nick': 'mhc',
                 'code': ''},
    "VELETA": {'nick': 'vlt',
               'code': '02_'}
    }
MEASUREMENT_TYPES = ["RS", "HF"]


""" SCC Server Connection Settings """
SCC_SERVER_SETTINGS = {
    "basic_credentials": tuple(['scc_user', 'sccforever!']),
    "website_credentials": tuple(['juan.antonio.aranda', '2ZU3ntER']),
    "base_url": 'https://scc.imaa.cnr.it/',
    "output_dir": None  # It has to be filled later in the code according to scc_access rules
}


""" logging  """
log_formatter = logging.Formatter('%(levelname)s: %(funcName)s(). L%(lineno)s: %(message)s')
debug = True
logger = logging.getLogger(__name__)
if (logger.hasHandlers()):
    logger.handlers.clear()
if debug:
    handler = logging.StreamHandler(sys.stdout)
else:
    log_fn = os.path.join(RUN_DN, "scc_%s.log" % dt.datetime.utcnow().strftime("%Y%m%dT%H%M"))
    handler = logging.FileHandler(log_fn)
handler.setFormatter(log_formatter)
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.propagate = False


""" Email 
    From a gmail account. Permission must be granted previously.
    At manage account: less secure apps, enable
"""
email_sender = {
                "server": "smtp.gmail.com", #"smtp.ugr.es",
                "port": 587,
                "sender_email": "controlador.mulhacen@gmail.com",
                "password": "lidarceama12345"
                }
#email_sender = {
#                "server": "smtp.ugr.es", #"smtp.gmail.com", #"smtp.ugr.es",
#                "port": 587,
#                "sender_email": "icenet@ugr.es", #"controlador.mulhacen@gmail.com",
#                "password": "ib3r1c0", # "lidarceama12345"
#                }
email_receiver = ["dbp@ugr.es", "mjgranados@ugr.es", "rascado@ugr.es", "jabravo@ugr.es"]
#email_receiver = ["dbp@ugr.es"]

def send_email(email_sender, email_receiver, email_content):
    """

    """
    # Check Input
    if isinstance(email_receiver, str):
        email_receiver = [email_receiver]

    logger.info("Send Email")
    date = dt.datetime.utcnow()
    return_code = 1
    try:
        # Prepare Message
        message = MIMEMultipart()
        message["From"] = email_sender["sender_email"]
        message["To"] = COMMASPACE.join(email_receiver)
        message["Date"] = formatdate(localtime=True)
        message["Subject"] = "SCC ERROR [%s UTC]]" % date.strftime("%Y%m%d_%H%M")
        message.attach(MIMEText(email_content))

        # Connect to email server and send email
        #ssl_context = ssl.create_default_context()
        #conn = smtplib.SMTP_SSL(email_sender['server'], email_sender['port'], context=ssl_context)
        conn = smtplib.SMTP(email_sender['server'], email_sender['port'])
        conn.ehlo()
        conn.starttls()
        conn.ehlo()
        conn.login(email_sender['sender_email'], email_sender['password'])
        conn.sendmail(email_sender['sender_email'], email_receiver, message.as_string())
        logger.info("Email Sent")
    except:
        logger.error("Email not sent")
        return_code = 0
    finally:
        conn.quit()   
    return return_code


"""
Helper Functions
"""

""" Handling Exceedance of Execution Time.
    Inspired in: https://stackoverflow.com/questions/51712256/how-to-skip-to-the-next-input-if-time-is-out-in-python """
# Maximum Allowed Execution Time (seconds)
MAX_EXEC_TIME = int(1200)  # 3600

# Class for timeout exception
class TimeoutException(Exception):
    pass

# Handler function to be called when SIGALRM is received 
def sigalrm_handler(signum, frame):
    # We get signal!
    raise TimeoutException()


""" get date from raw lidar files name """
def date_from_filename(filelist):
    """
    It takes the date from the file name of licel files.
    Parameters
    ----------
    filelist: list, str
        list of licel-formatted files

    Returns
    -------
    datelist: list, datetime
        list of datetimes for each file in input list
    """

    datelist = []
    if filelist:            
        for _file in filelist:        
            body = _file.split('.')[0]
            tail = _file.split('.')[1]
            year = int(body[-7:-5]) + 2000      
            month = body[-5:-4]
            try:
                month = int(month)
            except Exception as e:
                if month == 'A':
                    month = 10
                elif month == 'B':
                    month = 11
                elif month == 'C':
                    month = 12
            day = int(body[-4:-2])
            hour = int(body[-2:])
            minute = int(tail[0:2])
            #print('from body %s: the date %s-%s-%s' % (body, year, month, day))      
            cdate = dt.datetime(year, month, day, hour, minute)
            datelist.append(cdate)
    else:
        print('Filelist is empty.')
        datelist = None
    return datelist


def move2odir(source_filename, destination, overwrite=True):
    """
    It moves the file named source_filename in local directory to the absolute path file 'moved_filepath'.
    Input:
    source_filename: full path filename (string). 
    destination: destination path (string). 
    Output:
    sent: control variable. True=sent; False=not sent. (boolean).
    """
    sent = False    
    logger.info('Moving %s to %s' % (source_filename, destination))
    try:
        # create directory if it does not exist
        if not os.path.exists(destination):
            os.makedirs(destination)

        # delete file if exists in destination
        if os.path.exists(os.path.join(destination, os.path.basename(source_filename))):
            os.remove(os.path.join(destination, os.path.basename(source_filename)))

        # copyfile (shutil da problemas de incoherencia entre ejecucion (bien) y mensaje (error))
        sent = shutil.copy(source_filename, destination)
        if sent:
            sent = True
        logger.info('Moving file %s to user-defined output directory: %s... DONE!' % (source_filename, destination))

        # delete original
        os.remove(source_filename)
    except:
        if os.path.exists(os.path.join(destination, os.path.basename(source_filename))):
            logger.info('Moving file %s to user-defined output directory: %s... DONE!' % (source_filename, destination))
            sent = True
            # delete original
            os.remove(source_filename)
        else:
            logger.warning('Moving file %s to user-defined output directory: %s... ERROR!' % (source_filename, destination))
    return sent


def getTP(filepath):
    """
    Get temperature and pressure from header of a licel-formatted binary file.
    Inputs:
    - filepath: path of a licel-formatted binary file (str)
    Output:
    - temperature: temperature in celsius (float).
    - pressure: pressure in hPa (float).
    """
    # This code should evolve to read the whole header, not only temperature and pressure.
    if os.path.isfile(filepath):
        with open(filepath, mode="rb") as f:  #
            filename = f.readline().decode('utf-8').rstrip()[1:]
            second_line = f.readline().decode('utf-8')
            f.close()
        second_line_list = second_line.split(' ')
        if len(second_line_list) == 14:
            temperature = float(second_line_list[12].rstrip())
            pressure = float(second_line_list[13].rstrip())
        else:
            logger.warning('Cannot find temperature, pressure values. set to None')
            temperature = None
            pressure = None
    else:
        logger.warning('File not found.')
        temperature = None
        pressure = None
    return temperature, pressure


def apply_pc_peak_correction(filelist, scc_pc_channels):
    """
    Correction of the PC peaks in the PC channels caused by PMT degradation.

    Parameters
    ----------
    filelist: list(str)
        File list (e.g., /c/*.nc') (list)

    Returns
    -------
    outputlist: list(str)
        NetCDF file [file]
    """
    outputlist = list()
    threshold = 1000

    #scc_pc_channels = [1047, 1048, 1090, 1093, 1094]  # TODO: esto aqui a fuego ...
    if np.logical_and(len(filelist) > 0, len(scc_pc_channels) > 0):
        for file_ in filelist:
            try:
                lxarray = xr.open_dataset(file_)
                output_directory = os.path.dirname(file_)
                filename = os.path.basename(file_)
                for channel_ in scc_pc_channels:
                    idx_channel = np.where(lxarray.channel_ID == channel_)[0]
                    if idx_channel.size > 0:
                        # Call pc_peak_correction from utils_gfat
                        profile_raw = lxarray.Raw_Lidar_Data[:, idx_channel,:].values
                        shape_raw = profile_raw.shape
                        profile = np.squeeze(profile_raw)
                        new_profile = mulhacen_pc_peak_correction(profile)
                        lxarray.Raw_Lidar_Data[: , idx_channel,:] = np.reshape(new_profile.astype('int'), shape_raw)

                        profile_raw = lxarray.Background_Profile[:, idx_channel,:].values
                        shape_raw = profile_raw.shape
                        profile = np.squeeze(profile_raw)
                        new_profile = mulhacen_pc_peak_correction(profile)
                        lxarray.Background_Profile[: , idx_channel,:] = np.reshape(new_profile.astype('int'), shape_raw)

                # save corrected data in the same file
                auxfilepath = os.path.join(output_directory, 'aux')
                lxarray.to_netcdf(path=auxfilepath, mode='w')
                os.remove(file_)
                os.rename(auxfilepath, file_)
            except Exception as e:
                logger.warning(str(e))
                logger.warning("PC peak correction not performed")
            outputlist.append(file_)
    else:
        print('Files not found.')
    return outputlist


"""
PLOTTING FUNCTIONS
"""
def plot_scc_input(filelist):
    """
    Plot SCC input files.
    Input: 
    filelist: File string format (e.g., /c/*.nc') [str]    
    Output: 
    png figure files. Figures are saved in the same file directory.
    """
    MAX_ALT = 1000 #Points [bins]

    if len(filelist) > 0:
        for file_ in filelist:
            if os.path.isfile(file_):
                lxarray = xr.open_dataset(file_)
                output_directory = os.path.dirname(file_)
                filename = os.path.basename(file_)    

                colorbar = matplotlib.cm.get_cmap('jet', lxarray.time.size)  #Cool        
                colors = colorbar(np.linspace(0, 1, lxarray.time.size))
                try:
                    for idx_channel in np.arange(lxarray.channel_ID.size):                  
                        channel = lxarray.channel_ID.values[idx_channel]
                        fig = plt.figure(figsize=(9,5))
                        try:
                            for idx_time in np.arange(lxarray.time.size):
                                lxarray['Raw_Lidar_Data'][idx_time,idx_channel,0:MAX_ALT].plot(c=colors[idx_time], label='%d' % idx_time)
                        except:
                            logger.warning("Error: in plot_scc_input for %s. Plotting Raw Lidar Data. Try to continue plotting" % file_)
                        try:
                            for idx_time in np.arange(lxarray.time_bck.size):
                                lxarray['Background_Profile'][idx_time,idx_channel,0:MAX_ALT].plot(c=colors[idx_time], label='%d' % idx_time)
                        except:
                            logger.warning("Error: in plot_scc_input for %s. Plotting Background. Try to continue plotting" % file_)
                        plt.title('SCC-channel %d | %s' % (channel, filename))
                        plt.ylabel('Raw_Lidar_Data channel %d [counts/mV]' % channel)
                        plt.xlim(0,MAX_ALT)
                        ymin_value = 0.9*lxarray['Raw_Lidar_Data'][:,idx_channel,0:MAX_ALT].min()
                        ymax_value = 1.1*lxarray['Raw_Lidar_Data'][:,idx_channel,0:MAX_ALT].max()
                        plt.ylim(ymin_value,ymax_value)        
                        plt.savefig(os.path.join(output_directory, '%s_Raw_Lidar_Data_%d.png' % (filename.split('.')[0], channel)), dpi=200)
                        plt.close(fig)
                        logger.info("Plot SCC Raw Data in file %s" % file_)
                except:
                    logger.warning("Error: No plot for %s" % file_)
    else:
        logger.warning('Plot SCC INPUT not Done. Slot File List Empty')


def plot_scc_output(output_dir, scc_code=None):
    """
    Plot SCC optical output products, if available:
    backstter, extinction, angstrom exponent, lidar ratio, particle and volume depolarization

    Parameters:
    ----------
    output_dir: str
        output directory where scc processing products are downloaded
    scc_code: int
        system id

    """

    # config
    font = {'size': 12}
    matplotlib.rc('font', **font)

    # TODO: products id associated to beta, alfa, depo will be read from config file
    #beta_products = [1203, 838, 845, 1199, 760, 669, 839, 863]
    #alfa_products = [850, 1200, 838, 669, 839, 863]
    #depo_products = [760, 1203, 838, 669, 839, 863]
    #inversion_id = {760: 'K', 838: 'K', 845: 'R', 850: 'R', 1199: 'K',
    #                1200: 'R', 1203: 'R', 669: 'K', 839: 'K', 863: 'K'}

    # Plot x,y ranges
    y_lim = (0.68, 10)
    x_lim = {'beta': (-1e-2, 10), 
             'alfa': (1e-2, 500), 
             'angstrom': (-1, 3),
             'depo': (0, 0.40), 
             'lr': (10, 100)}
    # Colors according to Wavelength and Type of retrieval (Klett, K, or Iterative, R)
    colors = {'355': 'tab:blue', '355K': 'dodgerblue', '355I': 'dodgerblue', '355R': 'darkblue', '355U': 'tab:blue',
              '532': 'tab:green', '532K': 'limegreen', '532I': 'limegreen', '532R': 'darkgreen', '532U': 'tab:green',
              '1064': 'tab:red', '1064K': 'red', '1064I': 'red', '1064U': 'tab:red'}

    # necessary variables to be found in scc_optical/pid*nc
    vars_ = ["backscatter", "extinction", "particledepolarization", "volumedepolarization"]

    # scc
    if scc_code is None:
        scc_code = int(
            re.search(r"scc\d{3}", output_dir).group().split("scc")[1])

    logger.info('Plot SCC output. Processing folder: %s' % output_dir)

    # we need all nc files in the directory scc_optical
    nc_files = glob.glob(os.path.join(output_dir, 'scc_optical', '*.nc'))
    if nc_files:
        scc_slot_date_dt = dt.datetime.strptime(re.search(r"/\d{4}/\d{2}/\d{2}/", nc_files[0]).group(), '/%Y/%m/%d/') 
        # Take unique dates in the folder to know the number of different inversions
        def find_dates_in_file(fn, STATION_ID):
            if "pid" in os.path.basename(fn):
                date_pattern = r"\d{2}\d{2}\d{2}\d{2}\d{2}"
                pid_type = 0
            else:  # >= 2021
                date_pattern = r"\d{4}\d{2}\d{2}" + STATION_ID + "\d{2}\d{2}"
                pid_type = 1
            try:
                if pid_type == 0:
                    dates_str = re.search(date_pattern, os.path.basename(fn).split('_')[1].split('.')[0]).group()
                elif pid_type == 1:
                    dates_str = re.search(date_pattern, os.path.basename(fn).split('_')[6]).group()
                    #dates_str = dt.datetime.strptime(dates_str, "%Y%m%d"+STATION_ID+"%H%M").strftime("%Y%m%d%H%M")
                else:
                    dates_str = None
            except Exception as e:
                logger.error("Something went wrong finding date pattern in pid files")
                dates_str = None
            return dates_str
        dates_str = [find_dates_in_file(fn, STATION_ID) for fn in nc_files]
        if dates_str is not None:
            dates_str = np.unique(dates_str).tolist()
        # Loop over dates
        for date_ in dates_str:
            # pid files for the date
            pid_files = glob.glob(os.path.join(output_dir, 'scc_optical', '*%s*.nc' % date_))
            if len(pid_files) > 0:
                # define fig
                fig = plt.figure(figsize=(15, 10))  # , constrained_layout=True)

                """ Store Products in a dictionary """
                profile = {}
                for pid_fn in pid_files:
                    # product id
                    if "pid" in pid_fn:
                        product_id = int(os.path.basename(pid_fn).split('_')[0].replace('pid', ''))
                    else:  # >= scc version 2021
                        product_id = int(os.path.basename(pid_fn).split('_')[3])

                    with xr.open_dataset(pid_fn) as aux_ds:
                        pid_ds = None
                        for i_var in vars_:
                            try:
                                var_ds = aux_ds[i_var].squeeze()  # remove dimensions of length=1
                                # change dimensions for backscatter and extinction
                                if np.logical_or(i_var == "backscatter",
                                                 i_var == "extinction"):
                                    var_ds *= 1e6
                                    var_ds.attrs['units'] = "1/(Mm*sr)"
                                # add to pid dataset
                                if pid_ds is None:
                                    pid_ds = var_ds.to_dataset(name=i_var)
                                else:
                                    pid_ds[i_var] = var_ds
                                # add inversion method as attribute
                                # TODO: particularize for rest of variables: extinction, particledepolarization, volumedepolarization
                                try:
                                    if i_var == "backscatter":
                                        method = aux_ds["backscatter_evaluation_method"].values
                                        if method == 0:
                                            algorithm = aux_ds["raman_backscatter_algorithm"].values
                                        elif method == 1:
                                            algorithm = aux_ds["elastic_backscatter_algorithm"].values
                                        else:
                                            algorithm = 0
                                    else: #if i_var == "extinction":
                                        method = -1 # aux_ds["backscatter_evaluation_method"].values
                                        algorithm = -1 # aux_ds["elastic_backscatter_algorithm"].values
                                    if method == 0:  # Raman
                                        inversion_method = "R"
                                    elif method == 1:  # Elastic Backscatter
                                        if algorithm == 0:
                                            inversion_method = "K"   # Klett-Fernald
                                        elif algorithm == 1:
                                            inversion_method = "I"  # Iterative
                                        else:
                                            inversion_method = ""  # Unknown
                                    else:
                                        inversion_method = ""  # Unknown
                                except Exception as e:
                                    logger.error(str(e))
                                    logger.error("Inversion Method not found")
                                    inversion_method = None
                                pid_ds[i_var].attrs["inversion_method"] = inversion_method
                            except Exception as e:
                                logger.warning("%i does not have %s profile" % (product_id, i_var))
                    # altitude in km
                    pid_ds["altitude"] = pid_ds["altitude"] * 1e-3
                    pid_ds["altitude"].attrs['units'] = "km"

                    # save in profile dictionary
                    profile[product_id] = pid_ds.squeeze()  # remove dimensions of length=1

                """ Plot Backscatter """
                plot_code = 'beta'
                ax = fig.add_subplot(151)
                # loop over pids
                for pid in profile.keys():
                    #if pid in beta_products:  # realmente, ¿hace falta saber si pid está en beta_products?
                    if 'backscatter' in profile[pid].keys():
                        try:
                            wave = int(profile[pid]["wavelength"])
                            inv_met = profile[pid]['backscatter'].attrs["inversion_method"]
                            beta_id = '%d%s' % (wave, inv_met)
                            try:
                                color_beta = colors[beta_id]
                            except:
                                color_beta = colors["%s" % wave]
                            profile[pid]['backscatter'].plot(y='altitude',
                                                             ax=ax, linewidth=2,
                                                             c=color_beta,
                                                             label=beta_id)
                        except Exception as e:
                            logger.warning("Backscatter Not Plotted for %i" % wave)
                ax.xaxis.set_minor_locator(MultipleLocator(1))
                ax.xaxis.grid(b=True, which='minor', linestyle='--')
                ax.set_ylabel(r'Altitude, km asl', fontsize='large')
                ax.set_xlabel(r'$\beta_{a}, Mm^{-1} sr^{-1}$', fontsize='large')
                ax.set_xlim(x_lim[plot_code])
                #ax.set_xscale('log')
                if len(ax.get_lines()) > 0:
                    plt.legend(loc=1, fontsize='medium')

                """ Plot Extinction """
                plot_code = 'alfa'
                ax = fig.add_subplot(152)
                # loop over pids
                for pid in profile.keys():
                    if 'extinction' in profile[pid].keys():
                        try:  # if pid in alfa_products:
                            wave = int(profile[pid]["wavelength"])
                            #inv_met = profile[pid]['extinction'].attrs["inversion_method"]
                            inv_met = ""
                            alfa_id = '%d%s' % (wave, inv_met)
                            try:
                                color_alfa = colors[alfa_id]
                            except:
                                color_alfa = colors["%i" % wave]
                            profile[pid]['extinction'].plot(y='altitude', ax=ax,
                                                            linewidth=2,
                                                            c=color_alfa,
                                                            label=alfa_id)
                        except Exception as e:
                            logger.warning("Extinction Not Plotted for %i" % wave)
                ax.xaxis.set_minor_locator(MultipleLocator(50))
                ax.xaxis.grid(b=True, which='minor', linestyle='--')
                ax.set_xlabel(r'$ \alpha _{a}, Mm^{-1} $', fontsize='large')
                ax.set_xlim(x_lim[plot_code])
                if len(ax.get_lines()) > 0:
                    plt.legend(loc=1, fontsize='medium')

                """ Plot Angstrom """
                ax = fig.add_subplot(153)
                plot_code = 'angstrom'
                angstrom = {}
                if profile != {}:
                    coefficients = ['backscatter', 'extinction']
                    coef_id = {'backscatter': 'beta', 'extinction': 'alpha'}
                    for coef_ in coefficients:
                        for pid_1 in profile.keys():
                            for pid_2 in profile.keys():
                                wave_1 = int(profile[pid_1]['wavelength'])
                                wave_2 = int(profile[pid_2]['wavelength'])
                                #inv_met_1 = profile[pid_1][coef_].attrs["inversion_method"]
                                #inv_met_2 = profile[pid_2][coef_].attrs["inversion_method"]
                                inv_met_1 = ""
                                inv_met_2 = ""
                                if wave_1 < wave_2:
                                    if np.logical_and(
                                            coef_ in profile[pid_1].keys(),
                                            coef_ in profile[pid_2].keys()):
                                        try:
                                            profile[pid_1][coef_][profile[pid_1][coef_] <= 0] = np.nan
                                            profile[pid_2][coef_][profile[pid_2][coef_] <= 0] = np.nan
                                            angstrom_id = r'$\%s$ (%d%s-%d%s)' % (coef_id[coef_], 
                                                    wave_1, inv_met_1, wave_2, inv_met_2)
                                            angstrom[angstrom_id] = (-1) * np.log(
                                                profile[pid_1][coef_] / profile[pid_2][coef_]) / np.log(wave_1 / wave_2)
                                            angstrom[angstrom_id].name = angstrom_id
                                            angstrom[angstrom_id].plot(y='altitude', ax=ax,
                                                                       linewidth=2, label=angstrom_id)
                                        except Exception as e:
                                            logger.warning("Angstrom for %s, %s, %s not plotted. %s" 
                                                    % (coef_, pid_1, pid_2,str(e)))
                    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
                    ax.xaxis.grid(b=True, which='minor', linestyle='--')
                    ax.set_ylim(y_lim)
                    ax.set_xlim(x_lim[plot_code])
                    ax.set_xlabel('Angstrom exponent, #', fontsize='large')
                    if len(ax.get_lines()) > 0:
                        plt.legend(loc=1, fontsize='medium')

                """ Lidar ratio """
                ax = fig.add_subplot(154)
                plot_code = 'lr'
                lr = {}
                if profile != {}:
                    for pid in profile.keys():
                        if np.logical_and('backscatter' in profile[pid].keys(),
                                          'extinction' in profile[pid].keys()):
                            try:
                                wave = int(profile[pid]["wavelength"])
                                #lr_id = '%d%s' % (wave, inversion_id[pid])
                                lr_id = '%d' % (wave,)
                                profile[pid]['extinction'][profile[pid]['extinction'] <= 0] = np.nan
                                profile[pid]['backscatter'][profile[pid]['backscatter'] <= 0] = np.nan
                                lr[lr_id] = profile[pid]['extinction'] / profile[pid]['backscatter']
                                lr[lr_id].name = lr_id
                                lr[lr_id].plot(y='altitude', ax=ax, linewidth=2,
                                               c=colors[lr_id], label=lr_id)
                            except Exception as e:
                                logger.warning("Lidar Ratio for %s not plotted. %s" % (pid, str(e)))
                    ax.xaxis.set_minor_locator(MultipleLocator(10))
                    ax.xaxis.grid(b=True, which='minor', linestyle='--')
                    ax.set_ylim(y_lim)
                    ax.set_xlim(x_lim[plot_code])
                    ax.set_xlabel('Lidar ratio, sr', fontsize='large')
                    if len(ax.get_lines()) > 0:
                        plt.legend(loc=1, fontsize='medium')

                """ Depolarization """
                plot_code = 'depo'
                ax = fig.add_subplot(155)
                for pid in profile.keys():
                    if np.logical_and(
                            "particledepolarization" in profile[pid].keys(),
                            "volumedepolarization" in profile[pid].keys()):# pid in depo_products:
                        wave = int(profile[pid]["wavelength"])
                        #depo_id = '%d%s' % (wave, inversion_id[pid])
                        depo_id = '%d' % (wave,)
                        try:
                            profile[pid]['particledepolarization'].plot(
                                y='altitude', ax=ax, linewidth=2,
                                c=colors[depo_id], label=depo_id)
                            profile[pid]['volumedepolarization'].plot(
                                y='altitude', ax=ax, linewidth=2,
                                linestyle='dashed', c=colors[depo_id], label=depo_id)
                        except Exception as e:
                            logger.warning("Not Plotted Depolarization for %i" % wave)
                ax.xaxis.set_minor_locator(MultipleLocator(0.05))
                ax.xaxis.grid(b=True, which='minor', linestyle='--')
                ax.set_xlabel(r'$\delta_{a}$,#', fontsize='large')
                ax.set_xlim(x_lim[plot_code])
                if len(ax.get_lines()) > 0:
                    plt.legend(loc=1, fontsize='medium')

                # Fig Format and details
                fig_title = '%s' % np.datetime_as_string(profile[pid].time.values, unit='s')
                plt.suptitle(fig_title.replace('T', ' '),verticalalignment='baseline')
                plt.subplots_adjust(bottom=0.25, top=0.95)
                for ax in fig.get_axes():
                    ax.yaxis.set_major_locator(MultipleLocator(1))
                    ax.tick_params(axis='both', labelsize=14)
                    ax.set_ylim(y_lim)
                    ax.set_title('')
                    ax.grid()
                    ax.label_outer()

                # Save Fig
                if profile != {}:
                    fig_dir = os.path.dirname(output_dir)
                    fig_fn = os.path.join(fig_dir, 'scc%d_%s.png' %
                                          (scc_code, fig_title.replace('-', '').replace(':', '').replace('T', '_')))
                    plt.savefig(fig_fn, dpi=200, bbox_inches="tight")
                    plt.close(fig)
                    if os.path.isfile(fig_fn):
                        logger.info('Figure successfully created: %s' % fig_fn)
                        #os.path.basename(fig_fn))
    else:
        logger.warning('ERROR. No NC Files/scc_optical in Folder: %s ' % os.path.basename(output_dir))


"""
Processor Functions
"""
def get_info_from_measurement_file(mea_fn, scc_config_fn):
    """
    From a R File, extract information

    Parameters
    ----------
    mea_fn: str
        full path of measurement file
    scc_config_fn: str
        py file of scc configuration

    Returns
    -------
    time_ini_i: datetime.datetime
        initial time
    time_end_i: datetime.datetime
        end time
    channels_in_rm: collections.OrderedDict
        channels info

    """
    CustomLidarMeasurement = licel2scc.create_custom_class(scc_config_fn, use_id_as_name=True)
    mea = CustomLidarMeasurement([mea_fn])  # MUY LENTO
    time_ini_i = mea.info['start_time'].replace(tzinfo=None) 
    time_end_i = mea.info['stop_time'].replace(tzinfo=None) 
    # channels: Object LicelChannel (atmospheric_lidar/licel.py, L516)
    channels_in_rm = mea.channels
    del CustomLidarMeasurement, mea 
    return time_ini_i, time_end_i, channels_in_rm


def get_info_from_measurement_files(meas_files, scc_config_fn):
    """
    given a list of measurement files, information about:
        - scc configuration
        - time lapse of measurement is taken
    a scc_config.py file is needed to read file contents. [mhc_parameters_scc_xxx.py]

    Parameters
    ----------
    meas_files : [type]
        [description]
    scc_config_fn : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    try:
        # Parallelization: reduces near 1 minute when normal takes 2 minutes.
        with Pool(os.cpu_count()) as pool:
            x = [( mea_fn, scc_config_fn, ) for mea_fn in meas_files]
            res = np.array(pool.starmap(get_info_from_measurement_file, x))
        times_ini = [x[0] for x in res]
        times_end = [x[1] for x in res]
        channels_in_rm = [x[2] for x in res][0]
        time_start = times_ini[0].replace(second=0)
        time_end = times_end[-1] + dt.timedelta(minutes=1)
        time_end = time_end.replace(second=0)

        """ 
        # Non-parallel
        t1 = time.time()
        times_ini_s = []
        times_end_s = []
        for i, m_file in enumerate(meas_files):
            # times ini, end
            time_ini_i, time_end_i, channels_in_rm = get_info_from_measurement_file(scc_config_fn, m_file)
            if i == 0:
                # time of first measurement starts
                time_start = time_ini_i.replace(second=0)
            if i == len(meas_files) - 1:
                # time of last measurement ends
                time_end = time_end_i + dt.timedelta(minutes=1)
                time_end = time_end.replace(second=0)
            times_ini_s.append(time_ini_i)
            times_end_s.append(time_end_i)
        print(time.time() - t1)
        """
    except Exception as e:
        logger.error(str(e))
        logger.error("Measurement files not read")
        return

    return times_ini, times_end, time_start, time_end, channels_in_rm


def get_scc_config(scc_config_fn):
    """

    Parameters
    ----------
    scc_config_fn: str
        scc configuration file

    Returns
    -------
    scc_config_dict: dict
        Dictionary with scc configuration info from scc config file


    """
    try:
        sys.path.append(os.path.dirname(scc_config_fn))
        module = importlib.import_module(os.path.splitext(os.path.basename(scc_config_fn))[0])
        scc_config_dict = dict()
        scc_config_dict['general_parameters'] = module.general_parameters
        scc_config_dict['channel_parameters'] = module.channel_parameters
        scc_config_dict['channels'] = [*scc_config_dict['channel_parameters']]
    except:
        logger.warning("ERROR. importing scc parameteres from %s" % scc_config_fn)
        scc_config_dict = None

    return scc_config_dict


def get_campaign_config(campaign_cfg_fn=None, scc_lidar_cfg=None, hour_ini=None, hour_end=None,
        hour_resolution=None, timestamp=0, slot_name_type=0):
    """
    Get Campaign Info from file into a dictionary
    If not campaign_cfg_fn is given, a campaign_cfg is built, using, 
    if scc_lidar_cfg is given 
    
    Campaign config file is a json with different configurations as keys

    Parameters
    ----------
    campaign_cfg_fn : str
        campaign config file. Default: GFATserver
    scc_lidar_cfg: int
        scc lidar configuration number
    hour_ini: float
    hour_end: float
    hour_resolution: float
    timestamp: int
        0: timestamp at beginning of interval
        1: timestamp at center of interval
    slot_name_type: int
        0: earlinet: YYYYMMDD+station+slot_number.
        1: scc campaigns: YYYYMMDD+station+HHMM(Timestamp). 
        2: alternative: YYYYMMDDHHMM(Timestamp)+station+scc

    Returns
    -------
    scc_campaign_cfg: dict

    """

    if campaign_cfg_fn is None:
        campaign_cfg_fn = "GFATserver"

    if campaign_cfg_fn == "GFATserver":  # Default Campaign. Dictionary as template
        scc_campaign_cfg = {
            "name": "operational",
            "lidar_config":{
                "operational":{
                    "scc": scc_lidar_cfg,
                    "hour_ini": hour_ini,
                    "hour_end": hour_end,
                    "hour_res": hour_resolution,
                    "timestamp": timestamp,
                    "slot_name_type": slot_name_type
                }
            }
        }
    else:  # If Campaign File is given
        if os.path.isfile(campaign_cfg_fn):
            with open(campaign_cfg_fn) as f:
                scc_campaign_cfg = json.load(f)
        else:
            logger.error("Campaign File %s Does Not Exist. Exit program" % campaign_cfg_fn)
            sys.exit()
    return scc_campaign_cfg


def check_scc_output_inlocal(scc_output_slot_dn):
    """
    Check if Products have been downloaded from SCC server for a given slot output directory

    Parameters
    ----------
    scc_output_slot_dn: str
        full path local directory of scc slot
    Returns
    -------
    scc_output_inlocal: bool
        False if something has not been downloaded from SCC server
    """

    expected_dns = ['hirelpp', 'cloudmask', 'scc_preprocessed', 'scc_optical', 'scc_plots']
    exist_dns = [os.path.isdir(os.path.join(scc_output_slot_dn, i)) for i in expected_dns]
    if not all(exist_dns):
        scc_output_inlocal = False
    else:
        if len(glob.glob(os.path.join(scc_output_slot_dn, "scc_optical"))) == 0:
            scc_output_inlocal = False
        if len(glob.glob(os.path.join(scc_output_slot_dn, "scc_plots"))) == 0:
            scc_output_inlocal = False
        else:
            scc_output_inlocal = True

    return scc_output_inlocal
    

def process_day(lidar_name, date_str, meas_type, scc_campaign_config, 
                process, mode, data_dn, scc_dn):
    """[summary]

    Parameters
    ----------
    date_str: str, YYYYMMDD

    Returns
    -------
    status: bool
        True: all ok; False: something went totally or partially (slot) wrong
    msg: str
        output message. Especially useful if errors

    """
    logger.info("Start Process day: %s" % date_str)

    msg = ''

    # TODO: check input variables

    # date info
    date_dt = dt.datetime.strptime(date_str, "%Y%m%d")
    i_year, i_month, i_day = [date_dt.strftime(x) for x in ["%Y", "%m", "%d"]]
    r_year = date_dt.strftime('%y')
    if date_dt.month < 10:
        r_month = "%i"%date_dt.month
    elif date_dt.month == 10:
        r_month = "A"
    elif date_dt.month == 11:
        r_month = "B"
    elif date_dt.month == 12:
        r_month = "C"
    else:
        msg = "Error Naming Month For RS filename"
        logger.warning(msg)
        return False, msg
    r_day = date_dt.strftime('%d')
    r_date_str = "%s%s%s" % (r_year, r_month, r_day)

    """ Measurement Directories """
    lidar_0a_dn = os.path.join(data_dn, lidar_name, "0a")
    day_0a_dn = os.path.join(lidar_0a_dn, i_year, i_month, i_day)
    # Measurements Directories: RS
    rs_dns = glob.glob(os.path.join(day_0a_dn, "%s_%s_*" % (meas_type, date_str)))
    # Include Yesterday Directories
    date_ytd_dt = date_dt - dt.timedelta(days=1)
    date_ytd_str = date_ytd_dt.strftime("%Y%m%d")
    y_year, y_month, y_day = [date_ytd_dt.strftime(x) for x in ["%Y", "%m", "%d"]]
    ytd_0a_dn = os.path.join(lidar_0a_dn, y_year, y_month, y_day)
    rs_dns.extend(glob.glob(os.path.join(ytd_0a_dn, "%s_%s_*" % (meas_type, date_ytd_str))))

    """ Run process for every configuration in the scc campaign config """
    campaign_name = scc_campaign_config["name"]
    lidar_configs = scc_campaign_config["lidar_config"]
    # Loop over lidar configurations within the campaign
    for lidar_config in lidar_configs:
        # Get Lidar Configuration Info
        ldr_cfg = lidar_configs[lidar_config]
        scc = ldr_cfg["scc"]
        hour_ini = ldr_cfg["hour_ini"]
        hour_end = ldr_cfg["hour_end"]
        hour_res = ldr_cfg["hour_res"]
        timestamp = ldr_cfg["timestamp"]
        slot_name_type = ldr_cfg["slot_name_type"]

        # SCC Config File
        if scc is None:
            # TODO: integrar deduccion de SCC a partir de la medida
            msg = "SCC cannnot be None"
            logger.error(msg)
            return False, msg
        else:
            scc_config_fn = os.path.join(SCC_CONFIG_DN,'%s_parameters_scc_%d.py' 
                                         % (LIDAR_SYSTEMS[lidar_name]["nick"], scc))
        scc_config_dict = get_scc_config(scc_config_fn)

        logger.info("SCC = %d" % scc)
        # Measurement Files within the day
        rs_files = list(itertools.chain.from_iterable(
            [glob.glob(os.path.join(rs_dn, "R*%s*" % r_date_str)) for rs_dn in rs_dns]))
        rs_files.sort()
        if len(rs_files) > 0:
            # Time Period of Measurements within the day (Bottleneck)
            times_ini_rs, times_end_rs, time_start_rs, time_end_rs, meas_channels \
               = get_info_from_measurement_files(rs_files, scc_config_fn)
            # This is only for testing purposes
            #with open(os.path.join(MODULE_DN, "times.pickle"), "wb") as f:
            #   pickle.dump([times_ini_rs, times_end_rs, time_start_rs, time_end_rs, meas_channels], f)
            #with open(os.path.join(MODULE_DN, "times.pickle"), "rb") as f:
            #    times_ini_rs, times_end_rs, time_start_rs, time_end_rs, meas_channels = pickle.load(f)
            
            # More than 10 minutes of measurements
            if (time_end_rs - time_start_rs).total_seconds()/60 < 10:
                msg = "Less than 10 minutes of measurements. Exit Program"
                logger.error(msg)
                return False, msg

            # Build Slots:
            logger.info("Building Slots...")
            if hour_ini is None:
                hour_ini = float(time_start_rs.hour + time_start_rs.minute/60. + time_start_rs.second/3600.)
            else:
                hour_ini = float(hour_ini)
            if hour_end is None:
                if time_end_rs.day == time_start_rs.day:
                    hour_end = float(time_end_rs.hour + time_end_rs.minute/60. + time_end_rs.second/3600.)
                else:
                    hour_end = 23.99
            else:
                hour_end = float(hour_end)
            if hour_res is None:
                hour_res = float(1)
            else:
                hour_res = float(hour_res)
            if timestamp is None:
                timestamp = 1
            if slot_name_type is None:
                slot_name_type = 0
            if hour_end == hour_ini:
                hour_end = hour_ini + hour_res
            slots_ini = [date_dt + dt.timedelta(hours=x) 
                         for x in np.arange(hour_ini, hour_end, hour_res)]
            slots_end = [s + dt.timedelta(hours=hour_res) for s in slots_ini]
            slots_stp = [s + dt.timedelta(hours=(timestamp/2)*hour_res) for s in slots_ini]
            logger.info("Done")

            # SCC input directory for the day:
            if isinstance(hour_res, int) or hour_res.is_integer():
                input_pttn = 'input_%02dhoras' % hour_res 
            else:
                input_pttn = 'input_%dminutes' % int(hour_res*60)
            if scc_dn == 'GFATserver':
                scc_dn = os.path.join(data_dn, lidar_name)
            scc_input_dn = os.path.join(scc_dn, 'scc%d' % scc, input_pttn, 
                                        i_year, i_month, i_day)
            if not os.path.isdir(scc_input_dn):
                mkpath(scc_input_dn)
                logger.info("Local Directory %s CREATED" % scc_input_dn)

            # SCC Output Directory for the day
            scc_output_dn = scc_input_dn.replace("input", "output")

            # SCC Object for Interaction withh SCC Server
            logger.info("Creating SCC object...")
            SCC_SERVER_SETTINGS["output_dir"] = scc_output_dn  # se tiene que llamar asi por scc_access
            scc_obj = sa.SCC(SCC_SERVER_SETTINGS['basic_credentials'], 
                                SCC_SERVER_SETTINGS['output_dir'], 
                                SCC_SERVER_SETTINGS['base_url'])
            logger.info("Done")

            # Loop over Slots
            # TODO: paralelizar?
            for s in range(len(slots_stp)):
                """ Name Of Slot """
                slot_number = s
                # Slot ID. For the Name of the Netcdf
                if slot_name_type == 0:  # operational slot_id
                    slot_id = '%s%s%s%02d' \
                        % (date_str, STATION_ID, LIDAR_SYSTEMS[lidar_name]["code"], 
                        slot_number)
                elif slot_name_type == 1:  # timestamp
                    slot_id = '%s%s%s%02d%02d' \
                    % (date_str, STATION_ID, LIDAR_SYSTEMS[lidar_name]["code"], 
                    slots_stp[s].hour, slots_stp[s].minute)
                elif slot_name_type == 2: # timestamp + station + scc
                    slot_id = '%s%02d%02d%s%d%s' \
                    % (date_str, slots_stp[s].hour, slots_stp[s].minute, 
                       STATION_ID, scc, LIDAR_SYSTEMS[lidar_name]["code"])
                else:
                    logger.warning("slot name type not defined. Set to operational")
                    slot_id = '%s%s%s%02d' \
                        % (date_str, STATION_ID, LIDAR_SYSTEMS[lidar_name]["code"], 
                        slot_number)

                """ In Operational Mode, Check if Slot is completed so it can be processed """
                do_process_slot = True
                if mode == 0:
                    now_dt = dt.datetime.utcnow()
                    if now_dt < slots_end[s]:
                        do_process_slot = False

                """ Process Slot if Possible """
                if do_process_slot:
                    """ Check Slot Exists in Local and in SCC server """
                    # Local File for Slot
                    slot_fn = "%s.nc" % slot_id
                    scc_slot_fn = os.path.join(scc_input_dn, slot_fn)
                    if os.path.isfile(scc_slot_fn):
                        slot_in_local = True
                        logger.info("Slot %s Already exists in Local" % slot_id)
                    else:
                        slot_in_local = False
                        logger.info("Slot %s does not exist in Local" % slot_id)
                    # Slot in Server
                    scc_obj.login(SCC_SERVER_SETTINGS['website_credentials'])
                    meas_obj, _ = scc_obj.get_measurement(slot_id)
                    scc_obj.logout()
                    if meas_obj is not None:
                        slot_in_scc = True 
                        logger.info("Slot %s Already exists in SCC" % slot_id)
                    else:
                        slot_in_scc = False
                        logger.info("Slot %s Does Not Exist in SCC" % slot_id)

                    # Local Output SCC Slot
                    scc_output_slot_dn = os.path.join(scc_output_dn, slot_id)

                    """ 0a to SCC """
                    if np.logical_or.reduce((process == 0, process == 10)):
                        #not slot_in_local, 
                        # Subset RS Files for Slot
                        ids = []
                        for (i, t_i), (j, t_e) in zip(enumerate(times_ini_rs), enumerate(times_end_rs)):
                            if np.logical_and(t_i >= slots_ini[s], t_e <= slots_end[s]):
                                ids.append(i)
                        if len(ids) > 0:  # IF there are files within the slot
                            logger.info("Creating SCC format for slot %s" % slot_id)
                            # RS Files and DC Pattern
                            rs_files_slot = [rs_files[i] for i in ids]
                            dc_files_patt = os.path.join(
                                os.path.dirname(rs_files_slot[0]).replace(meas_type, 'DC'), "R*")
                            # if there is no operational DC, it is searched
                            if len(glob.glob(dc_files_patt)) == 0:
                                try:
                                    logger.warning("No DC files for day %s" % date_str)
                                    logger.warning("Search DC files within the previous 30 days")
                                    for b_day in range(1, 30):
                                        b_date_dt = date_dt + dt.timedelta(days=-b_day)
                                        b_year, b_month, b_day = [b_date_dt.strftime(x) for x in ["%Y", "%m", "%d"]]
                                        b_day_0a_dn = os.path.join(lidar_0a_dn, b_year, b_month, b_day)
                                        dc_files_patt = os.path.join(b_day_0a_dn, "DC*", "R*")
                                        if len(glob.glob(dc_files_patt)) > 0:
                                            break
                                    if len(glob.glob(dc_files_patt)) > 0:
                                        dc_dns = np.unique([os.path.basename(os.path.dirname(x)) 
                                            for x in glob.glob(dc_files_patt)]).tolist()
                                        dc_hours = np.asarray(
                                                [dt.datetime.strptime(i, "DC_%Y%m%d_%H%M").hour for i in dc_dns])
                                        # Daytime or Nighttime slot
                                        limit_hour = 17
                                        if time_start_rs.hour <= limit_hour:  # Daytimeslot
                                            ix = np.argwhere(dc_hours <= limit_hour)[0]
                                        else:  # Nighttime Slot
                                            ix = np.argwhere(dc_hours > limit_hour)[0]
                                        if len(ix) > 0: 
                                            dc_dn = dc_dns[ix[0]]
                                            dc_files_patt = os.path.join(b_day_0a_dn, dc_dn, "R*")
                                except Exception as e:
                                    logger.error("No DC files found for slot %s" % slot_id)

                            # Take temperature and pressure from the first file:
                            temperature, pressure = getTP(rs_files_slot[0]) 
                            # Prepare Licel2SCC input:
                            CustomLidarMeasurement = licel2scc.create_custom_class(
                                scc_config_fn, use_id_as_name=True, 
                                temperature=temperature, pressure=pressure)
                            # Convert from Raw to SCC Format:
                            try:
                                licel2scc.convert_to_scc(CustomLidarMeasurement, rs_files_slot, 
                                        dc_files_patt, slot_id, slot_number)
                            except Exception as e:
                                logger.error("%s. Conversion from 0a to SCC not possible for slot %s" % (str(e), slot_id))
                                msg += "Slot %s not created. " % slot_id

                            """ Name of Netcdf File """
                            # With reverse engineering, we know that the name of the
                            # netcdf file is:
                            scc_slot_ini_fn = os.path.join(RUN_DN, slot_fn)

                            """ PC peak correction if MULHACEN """
                            if lidar_name == "MULHACEN":
                                if os.path.isfile(scc_slot_ini_fn):
                                    logger.info("PC peak correction")
                                    pc_channels = []
                                    for c in scc_config_dict["channels"]:
                                        if c in meas_channels.keys():
                                            if meas_channels[c].analog_photon_string == 'ph':
                                                pc_channels.append(scc_config_dict["channel_parameters"][c]["channel_ID"])
                                    if len(pc_channels) > 0:
                                        scc_slot_ini_fn = apply_pc_peak_correction([scc_slot_ini_fn], pc_channels)  # se traga una lista de archivos
                                        # Devuelve una lista de archivos. TODO: Arreglar esto
                                        scc_slot_ini_fn = scc_slot_ini_fn[0]

                            """ Move File to NAS """
                            if os.path.isfile(scc_slot_ini_fn):
                                scc_saved = move2odir(scc_slot_ini_fn, scc_input_dn)
                                if scc_saved:
                                    logger.info("Created SCC Format for slot %s" % slot_id)
                                else:
                                    logger.warning("Error: SCC Format for slot %s not created" % slot_id)
                            else:
                                logger.error("Slot %s not created" % slot_id)

                        else:
                            logger.warning("No RS Files within the slot: [%s, %s]"
                                        % (slots_ini[s].strftime("%H%M"),
                                            slots_end[s].strftime("%H%M")))

                    """ PLOT SCC INPUT """
                    if np.logical_or.reduce((process == 0, process == 20)):
                        #not slot_in_local,
                        logger.info("Plot Input SCC")
                        plot_scc_input([scc_slot_fn])  # Se traga una lista de archivos en vez de uno. TODO: arreglar esto
                    else:
                        logger.info("Plot Input SCC already done")

                    """ SCC UPLOAD AND PROCESS """
                    if np.logical_or.reduce((process == 0, process == 1, process == 30)):
                        #not slot_in_scc, 
                        """ UPLOAD SLOT TO SCC SERVER """
                        if slot_in_scc:  # Delete Slot in SCC if exists: DOES NOT WORK YET
                            try:
                                logger.info("Start Delete Slot %s from SCC server" % slot_id)
                                scc_obj.login(SCC_SERVER_SETTINGS['website_credentials'])
                                deleted = scc_obj.delete_measurement(slot_id)
                                scc_obj.logout()
                                if deleted:
                                    logger.info("Slot %s deleted from SCC server" % slot_id)
                                else:
                                    logger.error("Slot %s NOT deleted from SCC server" % slot_id)
                                    logger.error("Slot %s must be deleted manually from SCC server" % slot_id)
                            except Exception as e:
                                logger.error(str(e))
                                logger.error("Slot %s not deleted from SCC server")
                        try:
                            # Wrap for skip in case of upload takes longer than MAX_EXEC_TIME
                            logger.info("Start Upload Slot %s" % slot_id)
                            old_handler = signal.signal(signal.SIGALRM, sigalrm_handler)
                            signal.alarm(MAX_EXEC_TIME)
                            if os.path.isfile(scc_slot_fn):
                                scc_obj.login(SCC_SERVER_SETTINGS['website_credentials'])
                                scc_obj.upload_file(scc_slot_fn, scc)
                                meas_obj, _ = scc_obj.get_measurement(slot_id)
                                scc_obj.logout()
                                if meas_obj is not None:
                                    logger.info("End Upload Slot %s" % slot_id)
                                else:
                                    logger.info("Slot %s not uploaded" % slot_id)
                            else:
                                logger.error("Slot %s cannot be uploaded because it does not exist" % slot_id)
                        except TimeoutException:
                            logger.error("Upload file took longer than %d s" % MAX_EXEC_TIME)
                        except Exception as e:
                            logger.error(str(e))
                            logger.error("Slot %s not uploaded" % slot_id)
                        finally:
                            # Turn off timer, Restore handler to previous value
                            signal.alarm(0)
                            signal.signal(signal.SIGALRM, old_handler)

                        """ Check, afterall, if Slot is at SCC """
                        scc_obj.login(SCC_SERVER_SETTINGS['website_credentials'])
                        meas_obj, _ = scc_obj.get_measurement(slot_id)
                        scc_obj.logout()
                        if meas_obj is None:
                            logger.error("Slot %s has not been uploaded to SCC server" % slot_id)
                    else:
                        logger.info("Slot %s already in SCC" % slot_id)

                    scc_output_inlocal = check_scc_output_inlocal(scc_output_slot_dn)

                    """ PROCESS AND DOWNLOAD SCC-PROCESSED """
                    if np.logical_or.reduce((process == 0, process==1, process == 2, process == 40, process == 50)):
                                            #not scc_output_inlocal,                        
                                            #not slot_in_local, not slot_in_scc,
                        try:
                            # Wrap for skip in case of upload takes longer than MAX_EXEC_TIME
                            old_handler = signal.signal(signal.SIGALRM, sigalrm_handler)
                            signal.alarm(MAX_EXEC_TIME)

                            # connect to scc
                            scc_obj.login(SCC_SERVER_SETTINGS['website_credentials'])

                            # get measurement object
                            meas_obj, status = scc_obj.get_measurement(slot_id)

                            if meas_obj:
                                """ PROCESSING """
                                if np.logical_or(process == 2, process == 40):
                                    logger.info("Start Processing Slot %s" % slot_id)
                                    request = scc_obj.session.get(meas_obj.rerun_all_url, stream=True)
                                    if request.status_code != 200:
                                        logger.error("Could not rerun pre processing for %s. Status code: %s" %
                                                     (slot_id, request.status_code))
                                """ DOWNLOADING """
                                logger.info("Start Download Processed Slot %s" % slot_id)
                                meas_obj = scc_obj.monitor_processing(slot_id)
                                logger.info("End Download Processed Slot %s" % slot_id)
                            else:
                                logger.error("Measurement Slot %s not found." % slot_id)
                            scc_obj.logout()
                        except TimeoutException:
                            logger.error("Upload file took longer than %d s" % MAX_EXEC_TIME)
                        except Exception as e:
                            logger.error(str(e))
                            logger.error("Slot %s not downloaded" % slot_id)
                        finally:
                            # Turn off timer, Restore handler to previous value
                            signal.alarm(0)
                    else:
                        logger.info("Slot %s already downloaded from SCC server" % slot_id)

                    """ PLOT OUTPUT SCC SERVER """
                    if np.logical_or.reduce((process == 0, process == 1, process == 2, process == 60)):
                        #not slot_in_local, not slot_in_scc,
                        #                    not scc_output_inlocal,
                        logger.info("Plot Output SCC")
                        plot_scc_output(scc_output_slot_dn, scc_code=scc)
                    else:
                        logger.info("Plot Output SCC already done")
                else:
                    logger.warning("Too Early to process Slot %s: %s" % (slot_id, now_dt.strftime("%Y%m%d_%H%M")))
                    logger.warning("Slot %s ends at %s" % (slot_id, slots_end[s].strftime("%Y%m%d_%H%M")))
        else:
            msg = "No Measurement Files found."
            logger.warning(msg)
            return False, msg

    logger.info("End Process day: %s" % date_str)
    if msg == '':
        status = True
    else:
        status = False
    return status, msg


def process_scc(**kwargs):
    """[summary]

    Parameters
    ----------

    Returns
    -------

    """

    """ Start SCC Processing """
    logger.info("Start SCC")

    try:
        """ Get Input Arguments """
        # Dates
        date_ini_str = kwargs.get("date_ini_str", None)
        if date_ini_str is None:
            date_ini_dt = dt.datetime.utcnow()
        else:
            date_ini_dt = dt.datetime.strptime(date_ini_str, "%Y%m%d")
        date_end_str = kwargs.get("date_end_str", None)
        if date_end_str is None:
            date_end_dt = date_ini_dt
        else:
            date_end_dt = dt.datetime.strptime(date_end_str, "%Y%m%d")

        # Lidar System
        lidar_name = kwargs.get("lidar_name", "MULHACEN")
        if lidar_name is None:
            lidar_name = "MULHACEN"
        if lidar_name not in LIDAR_SYSTEMS.keys():
            logger.error("Lidar %s does not exist. Exit program" % lidar_name)
            return

        # Measurement Type
        meas_type = kwargs.get("meas_type", "RS")
        if meas_type is None:
            meas_type = "RS"
        if meas_type not in MEASUREMENT_TYPES:
            logger.error("Measurement Type %s does not exist. Exit program" % meas_type)
            return

        # Campaign Config
        campaign_cfg_fn = kwargs.get("campaign_cfg_fn", "GFATserver")
        if campaign_cfg_fn is None:
            campaign_cfg_fn = "GFATserver"

        # SCC lidar config: Necessary if No Campaign Config File is given
        scc_lidar_cfg = kwargs.get("scc_lidar_cfg", None)

        # Start Hour: 
        hour_ini = kwargs.get("hour_ini", None)

        # End Hour: 
        hour_end = kwargs.get("hour_end", None)

        # Hour Resolution: If no campaign config file is given and different from default
        hour_resolution = kwargs.get("hour_resolution", None)
        if hour_resolution is None:
            hour_resolution = 1.0

        # Timestamp: 
        timestamp = kwargs.get("timestamp", None)
        if timestamp is None:
            timestamp = 0

        # Slot Name Type: 
        slot_name_type = kwargs.get("slot_name_type", None)
        if slot_name_type is None:
            slot_name_type = 1

        # Type of Processing
        process = kwargs.get("process", None)
        if process is None:
            process = 0

        # Mode of Processing: Operational/Offline
        mode = kwargs.get("mode", None)
        if mode is None:
            mode = 0
        if mode == 1:
            do_send_email = False
        else:
            do_send_email = True

        # Data directory: data_dn  [data_dn]/LIDAR/[0a, 1a, SCCxxx]
        data_dn = kwargs.get("data_dn", None)
        if np.logical_or(data_dn is None, data_dn == "GFATserver"):
            data_dn = DATA_DN
        if not os.path.isdir(data_dn):
            logger.error("Data Directory %s not found. Exit program" % data_dn)
            return

        # SCC data format directory: [data_dn/LIDAR/SCCxxx] / user-defined
        scc_dn = kwargs.get("scc_dn", "GFATserver")
        if scc_dn is None:
            scc_dn = "GFATserver"


        logger.info("LIDAR: %s" % lidar_name)

        """ Set SCC Campaign Configuration """
        scc_campaign_config = get_campaign_config(campaign_cfg_fn=campaign_cfg_fn, 
                                                  scc_lidar_cfg=scc_lidar_cfg,
                                                  hour_ini=hour_ini, hour_end=hour_end,
                                                  hour_resolution=hour_resolution,
                                                  timestamp=timestamp,
                                                  slot_name_type=slot_name_type)

        """ Process SCC workflow: loop along days """
        if date_end_dt.date() > date_ini_dt.date():
            logger.info("process period: %s - %s" % (date_ini_str, date_end_str))
            date_range = pd.date_range(date_ini_dt, date_end_dt)
            for i_date in date_range:
                i_date_str = i_date.strftime("%Y%m%d")
                result, msg_day = process_day(lidar_name, i_date_str, meas_type, scc_campaign_config, 
                        process, mode, data_dn, scc_dn)
                if not result:
                    msg = "ERROR: Lidar %s. Day %s. %s " % (lidar_name, i_date_str, msg_day)
                    if do_send_email:
                        send_email(email_sender, email_receiver, msg)
        elif date_end_dt.date() == date_ini_dt.date():
            logger.info("process single day")
            date_str = date_ini_dt.strftime("%Y%m%d")
            result, msg_day = process_day(lidar_name, date_str, meas_type, scc_campaign_config, 
                                 process, mode, data_dn, scc_dn)
            if not result:
                msg = "ERROR: Lidar %s. Day %s. %s " % (lidar_name, date_str, msg_day)
                if do_send_email:
                    send_email(email_sender, email_receiver, msg)
        else:
            msg = "Error in input dates. Exit program"
            logger.error(msg)
            if do_send_email:
                send_email(email_sender, email_receiver, msg)
            return
    except Exception as e:
        logger.error("%s. Exit program" % str(e))
        msg = "Exception Error in process_scc"
        logger.error(msg)
        if do_send_email:
            send_email(email_sender, email_receiver, msg)
        return


def parse_args():
    """ Parse Input Arguments
    python -u scc.py -arg1 v1 -arg2 v2 ...

    Parameters
    ----------
    ()

    Returns
    -------
    args: dict
        Dictionary 'arg':value for input

    """
    # TODO: incluir input sobre resolucion de slot y timestamp

    logger.info("Parse Input")

    parser = argparse.ArgumentParser(description="usage %prog [arguments]")    

    parser.add_argument("-i", "--initial_date",
        action="store", 
        dest="date_ini_str", 
        help="Initial date [example: '20190131'].")         
    parser.add_argument("-e", "--final_date",
        action="store",
        dest="date_end_str", # required=True,        
        help="Final date [example: '20190131'].")
    parser.add_argument("-l", "--lidar_name",
        action="store",
        dest="lidar_name",
        default="MULHACEN",
        help="Name of lidar system ['MULHACEN', 'VELETA']. Default: 'MULHACEN'.")
    parser.add_argument("-t", "--measurement_type",
        action="store",
        dest="meas_type",
        default="RS",
        help="Type of measurement [example: 'RS', 'HF'].")
    parser.add_argument("-c", "--campaign_cfg_fn",
        action="store",
        dest="campaign_cfg_fn",
        default="GFATserver",
        help="campaign config file in JSON format (full path), including name and scc lidar configurations.\
              Default: GFATserver means ACTRIS standardized format. \
              File must include the following fields: \
              name: str")
    parser.add_argument("-s", "--scc_lidar_cfg",
        type=int,
        action="store",
        dest="scc_lidar_cfg",
        help="SCC lidar configuration [example: 436, 403]\
              IF NO CAMPAIGN CONFIG FILE IS GIVEN.")
    parser.add_argument("-hi", "--hour_ini",
        type=float,
        action="store",
        dest="hour_ini",
        help="Start Hour (HH.H) for creation of slots.\
              DEFAULT: First Time of Measurement within the day")
    parser.add_argument("-he", "--hour_end",
        type=float,
        action="store",
        dest="hour_end",
        help="End Hour (HH.H) for creation of slots.\
              DEFAULT: Last Time of Measurement within the day")
    parser.add_argument("-r", "--hour_resolution",
        action="store",
        type=float,
        dest="hour_resolution",
        default=1,
        help="time resolution measurement slot. Lidar measurements will be \
              splitted in time slots in number of hours (default: 1 hour) \
              IF NO CAMPAIGN CONFIG FILE IS GIVEN.")
    parser.add_argument("-ts", "--timestamp",
        type=int,
        action="store",
        dest="timestamp",
        help="Timestamp for slot: 0, beginning of interval; 1: center of interval.\
              DEFAULT: 0")
    parser.add_argument("-snt", "--slot_name_type",
        type=int,
        action="store",
        dest="slot_name_type",
        help="Type of slot naming:\
                0: Earlinet:  YYYYMMDD+station+slot_number.\
                1: SCC Campaigns: YYYYMMDD+station+HHMM(Timestamp).\
                2: Alternative: YYYYMMDDHHMM(Timestamp)+station+scc.\
                DEFAULT: 1")
    parser.add_argument("-p", "--process",
        action="store",
        type=int,
        dest="process",
        default=0,
        help="what steps to process [-1: whole chain skipping steps if possible, \
                                0 (DEFAULT): whole chain forcing to process all steps, \
                                1: skip 0a->scc format, plot input. do: upload, process, download, plot output\
                                2: skip 0a->scc, plot input, upload to scc. do: process, download, plot_output\
                               10: only 0a->scc format, \
                               20: only plot input scc format, \
                               30: only upload&process to scc, \
                               40: only process&download from scc, \
                               50: only download from scc, \
                               60: only plot downloaded from scc")
    parser.add_argument("-m", "--mode",
        action="store",
        type=int,
        dest="mode",
        default=0,
        help="[0: operational (real-time. checks if the slot period has ended before processing it), 1: off-line")
    parser.add_argument("-d", "--data_dn", 
        action="store",
        dest="data_dn",
        default="GFATserver",
        help="disk directory for all data")
    parser.add_argument("-o", "--scc_dn",
        action="store",
        dest="scc_dn",
        default='GFATserver',
        help="directory where scc files transformed from 0a are saved. \
              Default: 'GFATserver' means copy files to the GFAT NAS.")

    args = parser.parse_args()

    return args.__dict__


def main():
    """
    To be called from terminal / shell:
    python -u scc.py -arg1 v1 -arg2 v2 ...

    """

    logger.info("Start SCC: %s" % dt.datetime.utcnow().strftime("%Y%m%d_%H%M"))

    """ parse args """
    args = parse_args()

    """ process scc workflow """
    process_scc(**args)

    logger.info("End SCC: %s" % dt.datetime.utcnow().strftime("%Y%m%d_%H%M"))


if __name__== "__main__":

    """ main """
    main()
