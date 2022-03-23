#!/usr/bin/env python
# coding: utf-8

import re
import os
import pdb
import argparse
import numpy as np
import pandas as pd
import xarray as xr
import datetime as dt
__author__ = "Bravo-Aranda, Juan Antonio"
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Bravo-Aranda, Juan Antonio"
__email__ = "jabravo@ugr.es"
__status__ = "Production"

# script description
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PROG_NAME = 'grawmet'
PROG_DESCR = 'Manage GRAWMET radisonde data.'

def reader_typeA(filepath, location='Granada'):
    '''
    file type: Y:\datos\radiosondes_spain\RS_20210625_135200UTC\202106242218048076.txt
    '''    
    if os.path.isfile(filepath):
        #Create variables
        time=[] 
        pressure = [] 
        temperature = [] 
        relative_humidity = []
        wind_speed = []
        wind_direction = []
        height = []
        dew_temperature = []
        varname = {'Tiempo': 'time', 'T': 'temperature', 'P': 'pressure', 'Hu': 'relative_humidity',
                  'Ws': 'wind_speed', 'Wd':'wind_direction','Geopot': 'range', 'Dewp.': 'dew_temperature'}
        #Open file
        f =open(filepath)
        #Read first line: useless
        f.readline()
        #Read second line: column names
        second = f.readline().split('\t')
        rs = {}
        for tmp_ in second:    
            rs[tmp_.strip()] = []  
        #Read third line: column units
        third = f.readline().split('\t')
        rs_units = rs.copy()
        for idx, tmp_ in enumerate(third):    
            rs_units[second[idx].strip()] = tmp_.strip()[1:-1]
        #Read following lines till ''Tropopausas:''
        for idx_data, line in enumerate(f.readlines()):
            if line.find('Tropopausas:') == -1:        
                line2 = line.replace(' ', '')
                values = np.asarray(line2.split('\t'),dtype='float')    
                time.append(values[0])
                pressure.append(values[1])
                temperature.append(values[2])
                relative_humidity.append(values[3])
                wind_speed.append(values[4])
                wind_direction.append(values[5])
                height.append(values[6])
                dew_temperature.append(values[7])        
            else:
                time = np.asarray(time,dtype='int')
                pressure = np.asarray(pressure,dtype='float')
                temperature = np.asarray(temperature,dtype='float')
                relative_humidity = np.asarray(relative_humidity,dtype='float')
                wind_speed = np.asarray(wind_speed,dtype='float')
                wind_direction = np.asarray(wind_direction,dtype='float')
                height = np.asarray(height,dtype='float')
                dew_temperature = np.asarray(dew_temperature,dtype='float')
                break
        #Keep ascending profile        
        idx_min = np.squeeze(np.where(np.min(height)==height))
        idx_max = np.squeeze(np.where(np.max(height)==height))
        time = time[idx_min:idx_max]
        pressure = pressure[idx_min:idx_max]
        temperature = temperature[idx_min:idx_max]
        relative_humidity = relative_humidity[idx_min:idx_max]        
        wind_speed = wind_speed[idx_min:idx_max]
        wind_direction = wind_direction[idx_min:idx_max]
        height = height[idx_min:idx_max]
        dew_temperature = dew_temperature[idx_min:idx_max]        

        f.close()
        #Open file again to read the last part of the file
        f =open(filepath)
        for idx, line in enumerate(f.readlines()):
            if idx > idx_data + 3:                 
                try:#Tropopauses                    
                    if line.find('Tropopause') != -1: 
                        tropopauses = None
                        fields = line.split('\t')
                        for tropo in fields:
                            candidates = re.findall(r'\d+',tropo)
                            if len(candidates) == 2:
                                tropopauses.append(candidates[-1])
                except:
                    tropopauses = None
                try:
                    if line.find('LCL') != -1: #LCL, LI, K-index
                        LCL, LI, K  = np.nan, np.nan, np.nan
                        fields = line.split('\t')            
                        for var in fields:    
                            if var.find('LCL') != -1:
                                candidate = re.findall(r"[-+]?\d+.?\d+",var)
                                if candidate:
                                    LCL = np.float(candidate[0])
                            elif var.find('LI') != -1:
                                candidate = re.findall(r"[-+]?\d+.?\d+",var)
                                if candidate:
                                    LI = np.float(candidate)
                            elif var.find('K-Index') != -1:
                                candidate = re.findall(r"[-+]?\d+.?\d+",var)
                                if candidate:
                                    K = np.float(candidate[0])
                except:
                    LCL, LI, K  = np.nan, np.nan, np.nan
                try:
                    if line.find('LFC') != -1: #LFC, SI, S-Index
                        LFC, SI, S  = np.nan, np.nan, np.nan
                        fields = line.split('\t')            
                        for var in fields:    
                            if var.find('LFC') != -1:
                                candidate = re.findall(r"[-+]?\d+.?\d+",var)
                                if candidate:
                                    LFC = np.float(candidate[0])
                            elif var.find('SI') != -1:
                                candidate = re.findall(r"[-+]?\d+.?\d+",var)
                                if candidate:
                                    SI = np.float(candidate[0])
                            elif var.find('S-Index') != -1:
                                candidate = re.findall(r"[-+]?\d+.?\d+",var)
                                if candidate:
                                    S = np.float(candidate[0])
                except:
                    LFC, SI, S  = np.nan, np.nan, np.nan
                try:
                    if line.find('CCL') != -1: #CCL, CAPE, TT-Index
                        CCL, CAPE, TT  = np.nan, np.nan, np.nan
                        fields = line.split('\t')            
                        for var in fields:    
                            if var.find('CCL') != -1:
                                candidate = re.findall(r"[-+]?\d+.?\d+",var)
                                if candidate:
                                    CCL = np.float(candidate[0])
                            elif var.find('CAPE') != -1:
                                candidate = re.findall(r"[-+]?\d+.?\d+",var)
                                if candidate:
                                    CAPE = np.float(candidate[0])
                            elif var.find('TT-Index') != -1:
                                candidate = re.findall(r"[-+]?\d+.?\d+",var)
                                if candidate:
                                    TT = np.float(candidate[0])
                except:
                    CCL, CAPE, TT  = np.nan, np.nan, np.nan
                try:
                    if  line.find('CINH') != -1: #EL, CINH, Ko-Index
                        EL, CINH, Ko  = np.nan, np.nan, np.nan
                        fields = line.split('\t')
                        for var in fields:    
                            if var.find('EL') != -1:
                                candidate = re.findall(r"[-+]?\d+.?\d+",var)
                                if candidate:
                                    EL = np.float(candidate[0])
                            elif var.find('CINH') != -1:
                                candidate = re.findall(r"[-+]?\d+.?\d+",var)
                                if candidate:
                                    CINH = np.float(candidate[0])
                            elif var.find('Ko-Index') != -1:
                                candidate = re.findall(r"[-+]?\d+.?\d+",var)                    
                                if candidate:
                                    Ko = np.float(candidate[0])
                except:
                    EL, CINH, Ko  = np.nan, np.nan, np.nan
                    
        #Global attribution
        launch_date = dt.datetime.strftime(dt.datetime.strptime(os.path.basename(filepath)[0:12],'%Y%m%d%H%M'), '%Y-%m-%dT%H:%M:00.0')

        #Creation of xarray Dataset 
        ds = xr.Dataset({'temperature':('range',temperature + 273.15),'pressure':('range',pressure),
                        'relative_humidity':('range',relative_humidity),'wind_speed':('range',wind_speed),
                        'wind_direction':('range',wind_direction), 'dew_temperature':('range',dew_temperature),
                        'time':('range',time)}, coords={'range':height}, 
                        attrs={'Location': location, 'Date': launch_date})
        #Variable attributes
        for var in rs_units.keys():
            ds[varname[var]].attrs['units'] = rs_units[var]
        ds['temperature'].attrs['units'] = 'K'

        #Additional information        
        ds['LCL'] = LCL
        ds['LCL'].attrs['units'] = 'hPa'
        ds['LCL'].attrs['long_name'] = 'Lifting Condensation Level'
        ds['LI'] = LI
        ds['LI'].attrs['units'] = '#'
        ds['LI'].attrs['long_name'] = 'LI'
        ds['K'] = K
        ds['K'].attrs['units'] = '#'
        ds['K'].attrs['long_name'] = 'K'
        ds['LFC'] = LFC
        ds['LFC'].attrs['units'] = 'hPa'
        ds['LFC'].attrs['long_name'] = 'Level of Free Convection'
        ds['SI'] = SI
        ds['SI'].attrs['units'] = '#'
        ds['SI'].attrs['long_name'] = 'SI'
        ds['S'] = S
        ds['S'].attrs['units'] = '#'
        ds['S'].attrs['long_name'] = 'S-Index'
        ds['CCL'] = CCL
        ds['CCL'].attrs['units'] = 'hPa'
        ds['CCL'].attrs['long_name'] = 'Convective Condensation Level'
        ds['CAPE'] = CAPE
        ds['CAPE'].attrs['units'] = 'J/Kg'
        ds['CAPE'].attrs['long_name'] = 'Convective Available Potential Energy'
        ds['CAPE'].attrs['comments'] = 'Instability: <1000J/kg: weak | 1000-2500J/kg: moderate | 2500-4000J/kg: strong | >4000J/kg: extreme'
        ds['TT'] = TT
        ds['TT'].attrs['units'] = '#'
        ds['TT'].attrs['long_name'] = 'TT-Index'
        ds['EL'] = EL
        ds['EL'].attrs['units'] = 'hPa'
        ds['EL'].attrs['long_name'] = 'EL'
        ds['CINH'] = CINH
        ds['CINH'].attrs['units'] = 'J/kg'
        ds['CINH'].attrs['long_name'] = 'CINH'
        ds['Ko'] = Ko
        ds['Ko'].attrs['units'] = '#'
        ds['Ko'].attrs['long_name'] = 'Ko-Index'
    else:
        ds = None
        print('File not found: %s' % filepath)
    return ds

def reader_typeB(file, location='Granada'):     
    '''
    type file: Y:\datos\radiosondes_spain\RS_20210625_135200UTC\202106242218048076.txt
    '''

    if os.path.isfile(file):
        #Create variables        
        units = {'time': 's', 'Time[UTC]': '', 'pressure': 'hPa', 'temperature': 'ºC','relative_humidity': '%', 'wind_speed': 'm/s',
                 'wind_direction': 'º', 'u': 'm/s', 'v': 'm/s', 'range': 'm', 'dew_temperature': 'º'}        
        
        df = pd.read_csv(file,sep='\t',header=1, names =['time', 'Time[UTC]', 'pressure','temperature','relative_humidity', 'wind_speed', 'wind_direction', 'u', 'v', 'range', 'dew_temperature'])        
        launch_date = os.path.basename(file)[0:8]
        try:
            dt.datetime.strptime(launch_date,'%Y%m%d')
        except:
            print("Filename format incorrect. It should start by yyyymmdd*.*")
        df['Date'] = launch_date
        df['Datetime_str'] = df['Date'] + ' ' + df['Time[UTC]']
        datetime = pd.to_datetime(df['Datetime_str'], format='%Y%m%d %H:%M:%S')
        df.insert(0,'Datetime',datetime)
        df = df.drop(labels=['Datetime_str', 'Date', 'Time[UTC]'], axis=1)
        df = df.set_index('Datetime')
        ds = df.to_xarray()
        ds = ds.swap_dims({'Datetime':'range'})
        ds['temperature'] = ds['temperature'] + 273.15
        ds['range'] = ds['range'] - ds['range'][0]
        rs_units = {'time': 's', 'Time[UTC]': '', 'pressure': 'hPa', 'temperature': 'K','relative_humidity': '%', 'wind_speed': 'm/s',
                         'wind_direction': 'º', 'u': 'm/s', 'v': 'm/s', 'range': 'm', 'dew_temperature': 'º'}        
        for key_ in ds.keys():
            ds[key_].attrs['units'] = rs_units[key_]
        ds.attrs={'Location': location, 'Date': dt.datetime.strptime(launch_date,'%Y%m%d')}
    else:
        ds = []
        print('File not found.')    
    return ds

def main():
    parser = argparse.ArgumentParser(description="usage %prog [arguments]")
    parser.add_argument("-f", "--filepath",
        action="store",
        dest="filepath",        
        help="Filepath")
    parser.add_argument("-o", "--outputfilepath",
        action="store",
        dest="outputfilepathfilepath",        
        help="output filepath")
             
    args = parser.parse_args()
    filepath = args.filepath
    outputfilepath = args.filepath



if __name__== "__main__":
    main()