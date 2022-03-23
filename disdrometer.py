#!/usr/bin/env python
# coding: utf-8

#import csv

import os
import sys
import pdb
import glob
import time
import argparse
import numpy as np
import pandas as pd
import xarray as xr
import datetime as dt
import netCDF4 as nc
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from optparse import OptionParser

__version__ = '1.0.0'
__author__ = 'Juan Antonio Bravo-Aranda'

# script description
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PROG_NAME = 'disdrometerConverter'
PROG_DESCR = 'converting raw data to netCDF files.'

###############################################################################################################
###############################################################################################################
###############################################################################################################

def mis2nc(MIS_FILE,NC_OUT_FILE):
    """
    Esta funcion convierte un archivo .mis preprocesado (limpio) a un archivo .nc
    Variables de entrada:
    1. MIS_FILE= Ruta del archivo de entrada .mis (str)
    2. NC_OUT_FILE= Ruta del archivo de salida .nc (str)
    """

    dat = pd.read_csv(MIS_FILE, header=None).values


    # CREA NC
    ncout = nc.Dataset(NC_OUT_FILE, "w", format="NETCDF4")

    # DEFINE DIMENSIONES
    time = ncout.createDimension('time', None)       #filelen, set='none' if unlimited dimension
    dclasses = ncout.createDimension('dclasses',32)          #sorting into velocity classes = bins
    vclasses = ncout.createDimension('vclasses',32)          #sorting into diameter classes = bins
    

    # DEFINE ATRIBUTOS GLOBALES:
    ncout.Title = "Parsivel disdrometer data"
    ncout.Institution = 'ANDALUSIAN INSTITUTE FOR EARTH SYSTEM RESEARCH (Granada,Spain)'
    ncout.Contact_person = 'Dr. Juan Bravo (jabravo@ugr.es )'
    ncout.Source = 'NETCDF4'
    ncout.History = 'Data processed on python.'
    ncout.Dependencies = 'external'
    ncout.Conventions = "CF-1.6 where applicable"
    ncout.Processing_date = dt.datetime.today().strftime('%Y-%m-%d,%H:%m:%S')
    ncout.Author = 'Irving Juanico (iejuv@ier.unam.mx)'
    ncout.Comments = ''
    ncout.Licence = 'For non-commercial use only. These data are the property of IISTA, their use is strictly prohibited without the authorization of the institute'


    # DEFINE VARIABLES
    # Define dimensiones como variables
    time = ncout.createVariable('time', np.float64, ('time',))
    time.fill_value = np.nan
    time.units = 'hours since 2000-01-01'
    time.long_name = 'time'
    
    dclasses = ncout.createVariable('dclasses', np.float64, ('dclasses',))
    dclasses.fill_value = np.nan
    dclasses.units = 'mm'
    dclasses.long_name = 'volume equivalent diameter class center'

    vclasses = ncout.createVariable('vclasses', np.float64, ('vclasses',))
    vclasses.fill_value = np.nan
    vclasses.units = 'm s-1'
    vclasses.long_name = 'velocity class center'


    # Definir exclusivamente variables dependientes de tiempo, clases v y clases d
    N = ncout.createVariable('droplet_number', np.float64, ('time','dclasses','vclasses'))
    N.fill_value = np.nan
    N.units = '-'
    N.long_name = 'number of droplets per volume equivalent diameter class and velocity class center'

    # Definir variables exclusivamente dependientes del tiempo.

    #'RECORD'
    inpr = ncout.createVariable('record_number', np.float32, ('time',))
    inpr.fill_value = np.nan
    inpr.units = '#'
    inpr.long_name = 'File record number.'

    #'rainIntensity'
    inpr = ncout.createVariable('precipitation_rain', np.float32, ('time',))
    inpr.fill_value = np.nan
    inpr.units = 'mm h-1'
    inpr.long_name = 'Intensity rain precipitation'

    #'snowIntensity'
    snowprec = ncout.createVariable('precipitation_snow', np.float32, ('time',))
    snowprec.fill_value = np.nan
    snowprec.units = 'mm h-1'
    snowprec.long_name = 'Intensity snow precipitation'    
    
    #'accPrec'
    pss = ncout.createVariable('precipitacion_accumulated', np.float32, ('time',))
    pss.fill_value = np.nan
    pss.units = 'mm'
    pss.long_name = 'Accumulated precipitation since start'

    #'weatherCodeWaWa'
    WaWa = ncout.createVariable('wmo_code_WaWa', np.float32, ('time',))
    WaWa.fill_value = np.nan
    WaWa.units = '-'
    WaWa.long_name = 'weather code according to WMO SYNOP 4680'

    #'radarReflectivity'
    rr = ncout.createVariable('radar_reflectivity', np.float32, ('time',))
    rr.fill_value = np.nan
    rr.units = 'dbZ'
    rr.long_name = 'Radar reflectivity'

    #'morVisibility'
    MOR = ncout.createVariable('visibility_MOR', np.float32, ('time',))
    MOR.fill_value = np.nan
    MOR.units = 'm'
    MOR.long_name = 'MOR visibility in the precipitation'

    #'signalAmplitude'
    sal= ncout.createVariable('signal_amplitude', np.float32, ('time',))
    sal.fill_value = np.nan
    sal.units = '#'
    sal.long_name = 'Signal amplitude of Laserband'

    #'numberParticles'
    Ndp = ncout.createVariable('detected_droplets', np.float32, ('time',))
    Ndp.fill_value = np.nan
    Ndp.units = '#'
    Ndp.long_name = 'Number of detected particles'

    #'sensorTemperature'
    Ts = ncout.createVariable('sensor_temperature', np.float32, ('time',))
    Ts.fill_value = np.nan
    Ts.units = 'ºC'
    Ts.long_name = 'Temperature in sensor'

    #'heatingCurrent'    
    Ih = ncout.createVariable('heating_current', np.float32, ('time',))
    Ih.fill_value = np.nan
    Ih.units = 'A'
    Ih.long_name = 'Heating current'

    #'sensorVoltage'
    Vs = ncout.createVariable('sensor_voltage', np.float32, ('time',))
    Vs.fill_value = np.nan
    Vs.units = 'V'
    Vs.long_name = 'Sensor Voltage'

    #'kineticEnergy'
    Ek = ncout.createVariable('kinetic_energy', np.float32, ('time',))
    Ek.fill_value = np.nan
    Ek.units = 'J'
    Ek.long_name = 'Kinetic energy'

    #'sensorStatus'
    Ss = ncout.createVariable('sensor_status', np.float32, ('time',))
    Ss.fill_value = np.nan
    Ss.units = '0: OK/ON  and   1: FUCK  2: OFF'
    Ss.long_name = 'Sensor status'

    # EXTRAYENDO LOS DATOS (archivos .mis)
        
    date_list = dat[:,0]
    date_nc = nc.date2num([ dt.datetime.strptime(date, '%Y-%m-%d %H:%M:%S') for date in date_list],time.units)
   
    spectrum_list = dat[:,14:-1]
    spectrum_matrix = np.array([np.split(spectrum,32) for spectrum in spectrum_list])

    # ESCRIBIENDO NC

    dclasses[:] =  np.array([0.062, 0.187, 0.312, 0.437, 0.562, 0.687, 0.812, 0.937, 1.062, 1.187,1.375, 1.625,1.875, 2.125,2.375, 2.750, 3.250, 3.750, 4.250, 4.750,5.500, 6.500, 7.500, 8.500,9.500, 11, 13, 15, 17, 19, 21.5, 24.5])
    vclasses[:] =  np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.10,1.30, 1.50, 1.70,1.90, 2.20, 2.60, 3.00, 3.40, 3.80, 4.40, 5.20,6.00, 6.80, 7.60, 8.80, 10.4, 12.0,13.6, 15.2, 17.6, 20.8])
    time[:]     =  date_nc
    N[:]        =  spectrum_matrix
    inpr[:]     =  dat[:,1]
    pss[:]      =  dat[:,2]
    WaWa[:]     =  dat[:,3]
    rr[:]       =  dat[:,4]
    MOR[:]      =  dat[:,5]
    sal[:]      =  dat[:,6]
    Ndp[:]      =  dat[:,7]
    Ts[:]       =  dat[:,8]
    Ih[:]       =  dat[:,9]
    Vs[:]       =  dat[:,10]
    Ek[:]       =  dat[:,11]
    Ss[:]       =  dat[:,12]

    ncout.close()
    
    return

def dat2nc(DAT_FILE,NC_OUT_FILE):
    """
    Esta funcion convierte un archivo .dat a un archivo .nc
    Variables de entrada:
    1. DAT_FILE= Ruta del archivo de entrada .dat (str)
    2. NC_OUT_FILE= Ruta del archivo de salida .nc (str)
    """

    if os.path.isfile(DAT_FILE):
        df = pd.read_csv(DAT_FILE,skiprows=0,header=1,parse_dates=["TIMESTAMP"],low_memory=False)
        df = df.drop([0, 1])
        df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
        df.set_index('TIMESTAMP',inplace=True) 

        # CREA NC            
        ncout = nc.Dataset(NC_OUT_FILE, "w", format="NETCDF4")
        
        # DEFINE DIMENSIONES
        time = ncout.createDimension('time', None)       #filelen, set='none' if unlimited dimension
        dclasses = ncout.createDimension('dclasses',32)          #sorting into velocity classes = bins
        vclasses = ncout.createDimension('vclasses',32)          #sorting into diameter classes = bins


        # DEFINE ATRIBUTOS GLOBALES:
        ncout.Title = "Parsivel disdrometer data"
        ncout.Institution = 'ANDALUSIAN INSTITUTE FOR EARTH SYSTEM RESEARCH (Granada,Spain)'
        ncout.Contact_person = 'Dr. Juan Bravo (jabravo@ugr.es )'
        ncout.Source = 'NETCDF4'
        ncout.History = 'Data processed on python.'
        ncout.Dependencies = 'external'
        ncout.Conventions = "CF-1.6 where applicable"
        ncout.Processing_date = dt.datetime.today().strftime('%Y-%m-%d,%H:%m:%S')
        ncout.Author = 'Dr. Juan A. Bravo-Aranda (jabravo@ugr.es)'
        ncout.Comments = ''
        ncout.Licence = 'For non-commercial use only. These data are the property of IISTA, their use is strictly prohibited without the authorization of the institute'


        # DEFINE VARIABLES
        # Define dimensiones como variables
        time = ncout.createVariable('time', np.float64, ('time',))
        time.fill_value = np.nan
        time.units = 'hours since 2000-01-01 0:0:0'
        time.long_name = 'time'

        dclasses = ncout.createVariable('dclasses', np.float64, ('dclasses',))
        dclasses.fill_value = np.nan
        dclasses.units = 'mm'
        dclasses.long_name = 'volume equivalent diameter class center'

        vclasses = ncout.createVariable('vclasses', np.float64, ('vclasses',))
        vclasses.fill_value = np.nan
        vclasses.units = 'm s-1'
        vclasses.long_name = 'velocity class center'


        # Definir exclusivamente variables dependientes de tiempo, clases v y clases d
        N = ncout.createVariable('droplet_number', np.float64, ('time','dclasses','vclasses'))
        N.fill_value = np.nan
        N.units = '-'
        N.long_name = 'number of droplets per volume equivalent diameter class and velocity class center'

        # Definir variables exclusivamente dependientes del tiempo.

        #'RECORD'
        inpr = ncout.createVariable('record_number', np.float32, ('time',))
        inpr.fill_value = np.nan
        inpr.units = '#'
        inpr.long_name = 'File record number.'

        #'rainIntensity'
        inpr = ncout.createVariable('precipitation_rain', np.float32, ('time',))
        inpr.fill_value = np.nan
        inpr.units = 'mm h-1'
        inpr.long_name = 'Intensity rain precipitation'

        #'snowIntensity'
        snowprec = ncout.createVariable('precipitation_snow', np.float32, ('time',))
        snowprec.fill_value = np.nan
        snowprec.units = 'mm h-1'
        snowprec.long_name = 'Intensity snow precipitation'            

        #'accPrec'
        pss = ncout.createVariable('precipitacion_accumulated', np.float32, ('time',))
        pss.fill_value = np.nan
        pss.units = 'mm'
        pss.long_name = 'Accumulated precipitation since start'

        #'weatherCodeWaWa'
        WaWa = ncout.createVariable('wmo_code_WaWa', np.float32, ('time',))
        WaWa.fill_value = np.nan
        WaWa.units = '-'
        WaWa.long_name = 'weather code according to WMO SYNOP 4680'

        #'radarReflectivity'
        rr = ncout.createVariable('radar_reflectivity', np.float32, ('time',))
        rr.fill_value = np.nan
        rr.units = 'dbZ'
        rr.long_name = 'Radar reflectivity'

        #'morVisibility'
        MOR = ncout.createVariable('visibility_MOR', np.float32, ('time',))
        MOR.fill_value = np.nan
        MOR.units = 'm'
        MOR.long_name = 'MOR visibility in the precipitation'

        #'signalAmplitude'
        sal= ncout.createVariable('signal_amplitude', np.float32, ('time',))
        sal.fill_value = np.nan
        sal.units = '#'
        sal.long_name = 'Signal amplitude of Laserband'

        #'numberParticles'
        Ndp = ncout.createVariable('detected_droplets', np.float32, ('time',))
        Ndp.fill_value = np.nan
        Ndp.units = '#'
        Ndp.long_name = 'Number of detected particles'

        #'sensorTemperature'
        Ts = ncout.createVariable('sensor_temperature', np.float32, ('time',))
        Ts.fill_value = np.nan
        Ts.units = 'ºC'
        Ts.long_name = 'Temperature in sensor'

        #'heatingCurrent'    
        Ih = ncout.createVariable('heating_current', np.float32, ('time',))
        Ih.fill_value = np.nan
        Ih.units = 'A'
        Ih.long_name = 'Heating current'

        #'sensorVoltage'
        Vs = ncout.createVariable('sensor_voltage', np.float32, ('time',))
        Vs.fill_value = np.nan
        Vs.units = 'V'
        Vs.long_name = 'Sensor Voltage'

        #'kineticEnergy'
        Ek = ncout.createVariable('kinetic_energy', np.float32, ('time',))
        Ek.fill_value = np.nan
        Ek.units = 'J'
        Ek.long_name = 'Kinetic energy'

        #'sensorStatus'
        Ss = ncout.createVariable('sensor_status', np.float32, ('time',))
        Ss.fill_value = np.nan
        Ss.units = '0: OK/ON  and   1: FUCK  2: OFF'
        Ss.long_name = 'Sensor status'


        #'rightTemperature'
        Tr = ncout.createVariable('temperature_right', np.float32, ('time',))
        Tr.fill_value = np.nan
        Tr.units = 'ºC'
        Tr.long_name = 'Temperature of the right arm.'

        #'leftTemperature'
        Tl = ncout.createVariable('temperature_left', np.float32, ('time',))
        Tl.fill_value = np.nan
        Tl.units = 'ºC'
        Tl.long_name = 'Temperature of the left arm.'    

        #'pbcTemperature'
        Tpbc = ncout.createVariable('temperature_pbc', np.float32, ('time',))
        Tpbc.fill_value = np.nan
        Tpbc.units = 'ºC'
        Tpbc.long_name = 'pbc Temperature.'    

        #'errorCode'
        Ec = ncout.createVariable('error_code', np.float32, ('time',))
        Ec.fill_value = np.nan
        Ec.units = 'ºC'
        Ec.long_name = 'Error code.'    


        # EXTRAYENDO LOS DATOS (archivos .mis)                
        date_nc = nc.date2num(df.index.to_pydatetime(),time.units)

        spectrum_array = df.iloc[:,-1024:].to_numpy()
        spectrum_matrix = np.array([np.split(spectrum,32) for spectrum in spectrum_array])

        # ESCRIBIENDO NC
        time[:]     =  date_nc
        N[:]        =  spectrum_matrix
        inpr[:]     =  df['rainIntensity'].to_numpy()
        pss[:]      =  df['accPrec'].to_numpy() 
        WaWa[:]     =  df['weatherCodeWaWa'].to_numpy() 
        rr[:]       =  df['radarReflectivity'].to_numpy()
        MOR[:]      =  df['morVisibility'].to_numpy()
        sal[:]      =  df['signalAmplitude'].to_numpy()
        Ndp[:]      =  df['numberParticles'].to_numpy()
        Ts[:]       =  df['sensorTemperature'].to_numpy()
        Ih[:]       =  df['heatingCurrent'].to_numpy()
        Vs[:]       =  df['sensorVoltage'].to_numpy()
        Ek[:]       =  df['kineticEnergy'].to_numpy()
        Ss[:]       =  df['sensorStatus'].to_numpy()
        Tr[:]       =  df['rightTemperature'].to_numpy()
        Tl[:]       =  df['leftTemperature'].to_numpy()
        Tpbc[:]     =  df['pbcTemperature'].to_numpy()
        Ec[:]       =  df['errorCode'].to_numpy()

        dclasses[:] =  np.array([0.062, 0.187, 0.312, 0.437, 0.562, 0.687, 0.812, 0.937, 1.062, 1.187,1.375, 1.625,1.875, 2.125,2.375, 2.750, 3.250, 3.750, 4.250, 4.750,5.500, 6.500, 7.500, 8.500,9.500, 11, 13, 15, 17, 19, 21.5, 24.5])
        vclasses[:] =  np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.10,1.30, 1.50, 1.70,1.90, 2.20, 2.60, 3.00, 3.40, 3.80, 4.40, 5.20,6.00, 6.80, 7.60, 8.80, 10.4, 12.0,13.6, 15.2, 17.6, 20.8])
        
        ncout.close()
        print('Done!')
    return

def rawfile2nc(INPUT_PATH, OUTPUT_PATH):
    '''
    rawfile2nc(INPUT_PATH,FILE_OUT_NC)
        This function converts a single *.mis or *.dat file in netCDF files.
    Input:
        1. INPUT_PATH= file path *.mis or *.dat (str)
        2. FILE_OUT_NC= (optional) output file path.(str)
    '''        
    print(INPUT_PATH)
    extension = INPUT_PATH.split('.')[-1]
    if extension == 'mis':            
        print('Converting mis-file: %s' % INPUT_PATH)
        mis2nc(INPUT_PATH,OUTPUT_PATH)
    elif extension == 'dat':
        print('Converting dat-file: %s' % INPUT_PATH)
        dat2nc(INPUT_PATH,OUTPUT_PATH)
    else:
        print('File extension unknown: %s' % INPUT_PATH)
    return


def raw2nc(INPUT_PATH):
    '''
    disdro2nc(INPUT_PATH)
        This function converts *.mis or *.dat files in netCDF files.
    Input:
        1. INPUT_PATH= file paths *.mis or *.dat (str)        
    '''

    file_list = glob.glob(INPUT_PATH)  # lista de archivos 
    if len(file_list)>=1:
        file_list.sort() # Ordenada alfabéticamente los ficheros del directorio
        for file_ in file_list:
            print('Convirtiendo archivos...')
            if file_.find('1a') == -1:
                #parsivel_20200205.dat                
                dtfile_ = dt.datetime.strptime(file_.split('_')[1].split('.')[0], '%Y%m%d')
                year = dtfile_.year
                month = dtfile_.month
                outputdir = os.path.join(os.path.dirname(os.path.dirname(file_)), '1a', '%d' % year, '%02d' % month)
                filename = os.path.basename(file_).replace('.dat', '.nc')
                output_filepath = os.path.join(outputdir,filename)
            else:
                output_filepath = file_.replace('0a','1b').replace(file_.split('.')[-1], 'nc')
            if not os.path.exists(os.path.dirname(output_filepath)):                
                os.makedirs(os.path.dirname(output_filepath))
            rawfile2nc(file_, output_filepath)
    return
###############################################################################################################

def spectrumPlot(station, spectrum_list, figurePath, daterange, plotrange=((0,6.1),(0,12)), size=17):
    """
    spectrumPlot(spectrum_list, daterange=tuple, plotrange=tuple, size):   
        Esta función grafica la distribución de goteo del espectro medido. 

    Variables de entrada son la siguientes:
    1. spectrum_list = Puede ser una lista de 1024 elementos o una matriz de [32,32] (np.array)
    2. daterange = Es una tupla definida como (date_0,date_f) donde date_o y date_son el rango
       inical y final de graficación respectivamente   (tuple of str)
    3. plotrange = Es una tupla definida como ((xmin,xmax),(ymin,ymax)) que permite modificar el 
       rango de visualización en eje coordenado (x,y) del grafico, dicho rango se encuentra 
       predefinido como plotrange=((0,6.1),(0,12))      (tuple of int/float)
    4. size = Tamaño de los encabezados (7 POR DEFECTO) (int/float)
    5. figurePath = ruta completa de la figura a guardar. (string)
        NOTA: las variables daterange[:] deben de ser escrita de la forma YYYY-MM-DD hh:mm:ss
           YYYY: Año escrito con 4 digitos
           MM: Numero del mes escrito con 2 digitos
           DD: Dia del mes escrito con 2 digitos
           hh: Hora del dia escrito con 2 digitos
           mm: Minuto de la hora escrito con 2 digitos
           ss: Segundos del minuto escrito con 2 digitos           
    """
    if len(spectrum_list) == 1024:
        spectrum_list = np.split(spectrum_list,32)
    else:
        spectrum_list = spectrum_list
       
    # Tamaño de la fuente de las etiquetas
    font = {'size': size}    
    mpl.rc('font', **font)
    
    # Tamaño de la figura
    fig, axes = plt.subplots(nrows=1, figsize=(18,10))   

    # Colormap
    bounds = [0,1,5,25,50,100,250,500,1000,2000,4000,8000,16000,50000] # Rangos del colormap discreto
    colors = ['#ffffff','#0015ff','#0051ff','#0095ff','#00ebfc','#9ff8fc','#e8fc60','#f2ff00','#fcac00','#fc7e00','#fc4700','#fa0000','#9c000f']  # colores de cada rango en formato HEX
    cm = mpl.colors.ListedColormap(colors)
    norm = mpl.colors.BoundaryNorm(bounds, cm.N)

    # Mallado del colormap
    dclasses = [0.062, 0.187, 0.312, 0.437, 0.562, 0.687, 0.812, 0.937, 1.062, 1.187,1.375, 1.625,1.875, 2.125, 2.375, 2.750, 3.250, 3.750, 4.250, 4.750,5.500, 6.500, 7.500, 8.500, 9.500, 11, 13, 15, 17, 19, 21.5, 24.5] # Clases diametro
    vclasses = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.10,1.30, 1.50, 1.70, 1.90,2.20, 2.60, 3.00, 3.40, 3.80, 4.40, 5.20,6.00, 6.80, 7.60, 8.80, 10.4, 12.0, 13.6, 15.2, 17.6, 20.8] # Clases velocidad   
    
    colormap = axes.pcolormesh(dclasses, vclasses, spectrum_list,cmap=cm,norm=norm,shading='gouraud') # genera el colormap

    # Etiquetas y rango de visualización
    axes.set_xlabel('Rain-droplet diameter, $m$$m$')
    axes.set_ylabel('Fall velocity, $m/s$')
    axes.set_xlim(plotrange[0][0],plotrange[0][1])
    axes.set_ylim(plotrange[1][0],plotrange[1][1])

    # Dibuja color map
    PLOT = plt.colorbar(colormap, ax=axes)
    PLOT.ax.set_ylabel('Rain-droplet number concentration, $\#/m^3$')   

    # Dibuja el mallado del colormap
    xi, yi = np.meshgrid(dclasses, vclasses)
    axes.plot(xi, yi, 'k-', alpha=0.3) # Dibuja la lineas verticales del mallado
    axes.plot(xi.T, yi.T, 'k-', alpha=0.3) # Dibuja la lineas horizontales del mallado
    
    if daterange[0][:-9] == daterange[1][:-9]:
        date = daterange[0][:-9]
    else:
        date = daterange[0][:-9] + ' to ' + daterange[1][:-9]
    
    datetemp0 = dt.datetime.strptime(daterange[0], '%Y-%m-%d %H:%M:%S')
    datetemp1 = dt.datetime.strptime(daterange[1], '%Y-%m-%d %H:%M:%S')
    year0 = dt.datetime.strftime(datetemp0, '%Y')
    year1 = dt.datetime.strftime(datetemp1, '%Y') 
    if year0 == year1:
        date0 = dt.datetime.strftime(datetemp0, '%H:%M %d/%m')
        date1 = dt.datetime.strftime(datetemp1, '%H:%M %d/%m/%y')                 
    else:
        date0 = dt.datetime.strftime(datetemp0, '%H:%M %d/%m/%y')
        date1 = dt.datetime.strftime(datetemp1, '%H:%M %d/%m/%y')
        
    axes.set_title('Disdrometer DSD in %s | Period: %s - %s ' % (station, date0,date1)) #

    # Dibuja la linea teórica
    gunnKinzerX = [0.0783, 0.0913, 0.1064, 0.1241, 0.1447, 0.1687, 0.1966, 0.229, 0.267,0.312, 0.363, 0.424, 0.494, 0.576, 0.671, 0.783, 0.913, 1.064, 1.241,                   1.447, 1.687, 1.966, 2.29, 2.67, 3.12, 3.63, 4.24, 4.94, 5.76]
    gunnKinzerY = [0.18, 0.25, 0.32, 0.4, 0.47, 0.57, 0.7, 0.87, 1.03, 1.21, 1.46,1.7, 2.03, 2.36, 2.74, 3.22, 3.72, 4.24, 4.76, 5.24, 5.90, 6.42,                   7.08, 7.65, 8.18, 8.62, 8.93, 9.08, 9.17]
    plt.plot(gunnKinzerX,gunnKinzerY, 'gray') 

    fig.savefig(figurePath, bbox_inches='tight', dpi=100)    
    if os.path.isfile(figurePath):
        print('Figure succesfully created: %s' % figurePath)
        control = True
    else:
        print('Figure NOT created')
        control = False    
    return control

###############################################################################################################
###############################################################################################################
###############################################################################################################

def accumulatedSpectrum(mainpath, figuredir, daterange, station='UGR', delay=0, plotrange=((0,6.1),(0,12)), size=17):
    """
    accumulatedSpectrum(PATH, daterange=tuple, delay=2, plotrange=tuple, size=7):
    Esta función grafica la distribución de goteo acumulada en un intervalo de tiempo [dat_0,date_f]

    NOTA 1: El gráficador considera la horas de retraso de medición del disdrometro (2 horas)

    Variables de entrada:
        1. mainpath = Ruta principal de los fichero *.nc  (str)
        2. figuredir = Ruta principal de las figuras *.png  (str)
        3. daterange = Es una tupla definida como (date_0,date_f) donde date_o y date_son el rango
           inical y final de graficación respectivamente   (tuple of str, format '%Y-%m-%d %H:%M:%S')
        4. delay= Es el tiempo de retraso de medición en horas (2 HORAS POR DEFECTO) (int/float)
        5. plotrange = Es una tupla definida como ((xmin,xmax),(ymin,ymax)) que permite modificar el 
           rango de visualización en eje coordenado (x,y) del grafico, dicho rango se encuentra 
           predefinido como plotrange=((0,6.1),(0,12))      (tuple of int/float)
        6. size = Tamaño de los encabezados (7 POR DEFECTO) (int/float)
    NOTA 2: las variables daterange[:] deben de ser escrita de la forma YYYY-MM-DD hh:mm:ss
    YYYY: Año escrito con 4 digitos
    MM: Numero del mes escrito con 2 digitos
    DD: Dia del mes escrito con 2 digitos
    hh: Hora del dia escrito con 2 digitos
    mm: Minuto de la hora escrito con 2 digitos
    ss: Segundos del minuto escrito con 2 digitos    
    """
    control = False
    TIME_DELAY = delay #HOURS
    dateini = dt.datetime.strptime(daterange[0], '%Y-%m-%d %H:%M:%S') - dt.timedelta(hours= TIME_DELAY)
    dateend = dt.datetime.strptime(daterange[1], '%Y-%m-%d %H:%M:%S') - dt.timedelta(hours= TIME_DELAY)  
    inidate = dt.datetime.strftime(dateini, '%Y%m%d-%H%M%S')    
    enddate = dt.datetime.strftime(dateend, '%Y%m%d-%H%M%S')

    date_ = dateini
    PATH = []
    while date_ < dateend:
        year = dt.datetime.strftime(date_,'%Y')
        month = dt.datetime.strftime(date_,'%m')
        filename = '%s_%s.nc' % (dt.datetime.strftime(date_,'%Y%m%d'), station)
        testfile = os.path.join(mainpath, year, month, filename)        
        if os.path.isfile(testfile):
            PATH.append(testfile)            
            print('Existing file append!: %s' % testfile)
        else:
            filename = 'parsivel_%s.nc' % (dt.datetime.strftime(date_,'%Y%m%d'))
            testfile = os.path.join(mainpath, year, month, filename)                    
            if os.path.isfile(testfile):
                PATH.append(testfile)            
                print('Existing file append!: %s' % testfile)            

        date_ = date_ + dt.timedelta(days=1)
    
    if PATH:
        #Read files
        data_nc = xr.open_mfdataset(PATH)
        
        #Select region to plot
        data_dates = data_nc.sel(time=slice(*[dateini,dateend]))
        
        spectrum_accumulate_matrix = np.nansum(data_dates.droplet_number.values,axis=0)

        #Figure output path    
        figurename = 'accuDSD_%s_%s_%s.png' % (station, inidate, enddate)
        figurepath = os.path.join(figuredir, figurename) 

        #Check folder exists
        if not os.path.isdir(os.path.dirname(figurepath)):
            os.mkdir(os.path.dirname(figurepath))

        #Figure plot
        control = spectrumPlot(station, spectrum_accumulate_matrix, figurepath, daterange=daterange,plotrange=plotrange,size=size)

    return control

###############################################################################################################
###############################################################################################################
###############################################################################################################

def quicklook(variables2plot, mainpath, figuredir, daterange, station='UGR', delay=0, plotrange=((0,6),(0,12)),axesTime='Default',size=16):
    """
     quicklook(variables2plot, mainpath, figuredir, station, daterange, delay=0, plotrange=((0,6),(0,12)),axesTime='Default',size=16):

    Esta función realiza las siguientes graficas del disdrometro:           
        1. Distribución de goteo por diametro
        2. Distribución de goteo por velocidad

    NOTA 1: El gráficador puede considerar las horas de retraso de medición del disdrometro (2 horas)

    Variables de entrada:
        1. variables2plot = Variable to make the quicklook. Options: 'diameter' and 'velocity' (str)
        2. PATH= Ruta principal del fichero netcdf (str)
        2. daterange = Es una tupla definida como (date_0,date_f) donde date_o y date_son el rango
           inical y final de graficación respectivamente   (tuple of str)
        3. plotrange = Es una tupla definida como ((dmin,dmax),(vmin,vmax)) que permite modificar el 
           rango de visualización de cada distribución, (dmin,dmax) para modificar el rango de la 
           distribución de goteo por diametro y (vmin,vmax) para modificar el rango de la distribución 
           de goteo por velocidad.
           predefinido como plotrange=((0,6),(0,12))      (tuple of int/float)
        4. delay= Es el tiempo de retraso de medición en horas (0 HORAS POR DEFECTO) (int/float)
        5. size = Tamaño de los encabezados (16 POR DEFECTO) (int/float)

    NOTA 2: las variables date_0 y date_f deben de ser escrita de la forma YYYY-MM-DD hh:mm:ss
           YYYY: Año escrito con 4 digitos
           MM: Numero del mes escrito con 2 digitos
           DD: Dia del mes escrito con 2 digitos
           hh: Hora del dia escrito con 2 digitos
           mm: Minuto de la hora escrito con 2 digitos
           ss: Segundos del minuto escrito con 2 digitos    
    """
    
    def quicklook_DSD(var_, tclasses, data_dates, daterange, plotrange, station, figurepath):
        # FIGURA 1: DISTRIBUCIÓN DE GOTEO POR DIAMETRO
        # Tamaño de la figura
        control = False
        dclasses = [0.062, 0.187, 0.312, 0.437, 0.562, 0.687, 0.812, 0.937, 1.062, 1.187,1.375, 1.625,1.875, 2.125, 2.375, 2.750, 3.250, 3.750, 4.250, 4.750,5.500, 6.500, 7.500, 8.500,9.500, 11, 13, 15, 17, 19, 21.5, 24.5] # Clases diametro
        vclasses = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.10,1.30, 1.50, 1.70, 1.90,2.20, 2.60, 3.00, 3.40, 3.80, 4.40, 5.20,6.00, 6.80, 7.60, 8.80, 10.4, 12.0, 13.6, 15.2,17.6, 20.8]# Clases velocidad                
        var2plot = {'diameter': dclasses, 'velocity': vclasses}
        ylabelstr = {'diameter': 'Rain-droplet diameter, $m$$m$', 'velocity': 'Rain-droplet fall velocity, $m/s$'}
        ylimDict = {'diameter': plotrange[0], 'velocity': plotrange[1]}
                #Extrayendo matriz del espectro por todo el lapso de tiempo
        spectrum_matrix = data_dates.droplet_number.values
        # Caculando distribución de goteo por diametro y por velocidad
        spectrum_per_diameter = np.nansum(spectrum_matrix,1)
        spectrum_per_velocity = np.nansum(spectrum_matrix,2)
        I = spectrum_per_diameter.shape[1]
        J = spectrum_per_velocity.shape[1]
    
        # Artilugio para poder graficar las distribuciones 
        droplet_per_diameter = np.array([ list( spectrum_per_diameter[:,i] ) for i in range( I ) ])
        droplet_per_velocity = np.array([ list( spectrum_per_velocity[:,i] ) for i in range( J ) ])
        
        DSD = {'diameter': droplet_per_diameter, 'velocity': droplet_per_velocity}
        
        # Tamaño de la fuente de las etiquetas
        font = {'size': size} 
        mpl.rc('font', **font)        
        
        fig, axes = plt.subplots(nrows=1, figsize=(18,10)) 
        
        # Colormap
        bounds = [0,1,2,5,10,20,50,100,200,500,700,1000,1200,1500] # Rangos del colormap discreto
        colors = ['#ffffff','#0015ff','#0051ff','#0095ff','#00ebfc','#9ff8fc','#e8fc60','#f2ff00','#fcac00','#fc7e00','#fc4700','#fa0000','#9c000f']  # colores de cada rango en formato HEX
        cm = mpl.colors.ListedColormap(colors)
        norm = mpl.colors.BoundaryNorm(bounds, cm.N)
        colormap = axes.pcolormesh(tclasses,var2plot[var_],DSD[var_],cmap=cm,norm=norm,shading='gouraud') # genera el colormap

        # Dibuja color map
        PLOT = plt.colorbar(colormap, ax=axes)
        PLOT.ax.set_ylabel('Rain-droplet number concentration, $m^3$')

        # Dibuja el mallado del colormap
#         xi, yi = np.meshgrid(tclasses, dclasses)        
#         axes.plot(xi.T, yi.T, 'k-', alpha=0.3) # Dibuja la lineas horizontales del mallado    
        # Etiquetas 
        plt.grid(which='both')
        axes.set_ylabel(ylabelstr[var_])

        if axesTime == 'Default':
            axes.set_xlabel('Time, $HH:MM$')
            # intervalo tiempo (step) en eje del tiempo (step de 1 hora)
            hours = mdates.HourLocator(interval = 1) 
            axes.xaxis.set_major_locator(hours)
            #unidades del eje de tiempo
            h_fmt = mdates.DateFormatter('%H:%M') 
            axes.xaxis.set_major_formatter(h_fmt) 
        elif axesTime == 'Automatic':
            axes.set_xlabel('Time')
        
        # rango de visualiación
        datetemp0 = dt.datetime.strptime(daterange[0], '%Y-%m-%d %H:%M:%S')
        datetemp1 = dt.datetime.strptime(daterange[1], '%Y-%m-%d %H:%M:%S')

        axes.set_xlim(datetemp0,datetemp1)
        axes.set_ylim(ylimDict[var_][0],ylimDict[var_][1])
                
        diff = datetemp1-datetemp0        
        if diff <= dt.timedelta(hours=24):
            hour_resolution = 1
            hour_fmt= '%H:%M'
        elif np.logical_and(diff > dt.timedelta(hours=24), diff <= dt.timedelta(hours=48)):    
            hour_resolution = 3
            hour_fmt= '%H:%M'
        elif np.logical_and(diff > dt.timedelta(hours=48), diff <= dt.timedelta(hours=72)): 
            hour_resolution = 6
            hour_fmt= '%d %H:%M'
        else:
            hour_resolution = 12
            hour_fmt= '%d/%m %Hh'
        xdates = np.arange(datetemp0,datetemp1,dt.timedelta(hours=hour_resolution))
        plt.xticks(xdates)
        axes.xaxis.set_major_formatter(DateFormatter(hour_fmt))
        fig.autofmt_xdate()
        
        year0 = dt.datetime.strftime(datetemp0, '%Y')
        year1 = dt.datetime.strftime(datetemp1, '%Y') 
        if year0 == year1:
            date0 = dt.datetime.strftime(datetemp0, '%H:%M %d/%m')
            date1 = dt.datetime.strftime(datetemp1, '%H:%M %d/%m/%y')                 
        else:
            date0 = dt.datetime.strftime(datetemp0, '%H:%M %d/%m/%y')
            date1 = dt.datetime.strftime(datetemp1, '%H:%M %d/%m/%y')
        axes.set_title('%s-DSD quicklook | %s Period: %s - %s ' % (var_, station, date0,date1)) #

        fig.savefig(figurepath, bbox_inches='tight', dpi=100)        
        if os.path.isfile(figurepath):
            print('Figure succesfully created: %s' % figurepath)
            control = True
        else:
            print('Figure NOT created')
            control = False
        plt.close(fig)

        return control
        
    #Main code
    control = False
    TIME_DELAY = delay #HOURS
    dateini = dt.datetime.strptime(daterange[0], '%Y-%m-%d %H:%M:%S') - dt.timedelta(hours= TIME_DELAY)
    dateend = dt.datetime.strptime(daterange[1], '%Y-%m-%d %H:%M:%S') - dt.timedelta(hours= TIME_DELAY)  
    inidate = dt.datetime.strftime(dateini, '%Y%m%d-%H%M%S')    
    enddate = dt.datetime.strftime(dateend, '%Y%m%d-%H%M%S')

    date_ = dateini
    PATH = []
    while date_ < dateend:
        year = dt.datetime.strftime(date_,'%Y')
        month = dt.datetime.strftime(date_,'%m')
        filename = '%s_%s.nc' % (dt.datetime.strftime(date_,'%Y%m%d'), station)
        testfile = os.path.join(mainpath, year, month, filename)        
        if os.path.isfile(testfile):
            PATH.append(testfile)            
            print('Existing file append!: %s' % testfile)
        else:
            filename = 'parsivel_%s.nc' % (dt.datetime.strftime(date_,'%Y%m%d'))
            testfile = os.path.join(mainpath, year, month, filename)                    
            if os.path.isfile(testfile):
                PATH.append(testfile)            
                print('Existing file append!: %s' % testfile)            
                
        date_ = date_ + dt.timedelta(days=1)
    
    if PATH:
        #Read files
        data_nc = xr.open_mfdataset(PATH)

        #Select region to plot
        data_dates = data_nc.sel(time=slice(*[dateini,dateend]))
    
        # Mallado del colormap
        tclasses = data_dates.time.values + TIME_DELAY*60*60000000000 # Clases tiempo   

        for var_ in variables2plot:                  
            #Figure output path    
            figurename = 'quicklook_%s_%s_%s_%s.png' % (var_, station, inidate, enddate)
            figurepath = os.path.join(figuredir, figurename) 
        
            #Check folder exists
            if not os.path.isdir(os.path.dirname(figurepath)):
                os.makedirs(os.path.dirname(figurepath))

            print('Plotting %s' % var_)
            control = quicklook_DSD(var_, tclasses, data_dates, daterange, plotrange, station, figurepath)
            if not control:
                print('ERROR: figure not created: %s' % figurepath)                            
    return control

###############################################################################################################
###############################################################################################################
###############################################################################################################


def main():
    # parser = OptionParser(usage="usage %prog [options]",
    #     version="%prog 1.0")
    # parser.add_option("-s", "--station_name",
    #     action="store",
    #     dest="station",
    #     default="UGR",
    #     help="Measurement station [default: %default].")
    # parser.add_option("-d", "--date2plot",
    #     action="store",
    #     dest="date2plot",
    #     default="20191121",
    #     help="Date to plot [default: %default].")
    # parser.add_option("-o", "--dir_out",
    #     action="store",
    #     dest="dir_out",
    #     default=".",
    #     help="Output folder [default: %default].")
    # parser.add_option("-i", "--dir_in",
    #     action="store",
    #     dest="dir_in",
    #     default=".",
    #     help="Input folder [default: %default].")                
    # (options, args) = parser.parse_args()
    # dir0 = options.dir_in    
    # dir2 = options.dir_out   
    # strdate = options.date2plot
    # station = options.station
    # date_ = dt.datetime.strptime(strdate,'%Y%m%d')
    # year = dt.datetime.strftime(date_,'%Y')
    # month = dt.datetime.strftime(date_,'%m')
    # file0a = os.path.join(dir0, year, month, '%s_%s.mis' % (dt.datetime.strftime(date_,'%Y%m%d'), station))
    # if os.path.isfile(file0a):
    #     print('%s found!' % file0a)    
    #     file1a = file0a.replace('0a', '1a')
    #     disdroconverter1b.cleanMIS(file0a, file1a)
        
    #     if os.path.isfile(file1a):
    #         print('%s found!' % file1a)            
    #         file1b = file1a.replace('1a', '1b')
    #         file1b = file1b.replace('mis', 'nc')
    #         to_nc(file1a, file1b)
        
    #         if os.path.isfile(file1b):    
    #             print('%s successfully converted!' % file1b)
    #         else:
    #             print('%s conversion to 1b-level FAILED!' % file0a)
    #     else:    
    #         print('%s conversion to 1a-level FAILED!' % file0a)
    # else:    
    #     print('%s not found.' % file0a)        
    return

if __name__== "__main__":
    main()