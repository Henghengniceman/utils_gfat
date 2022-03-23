#!/usr/bin/env python
# coding: utf-8



import csv
import glob
import numpy as np
import os
import pandas as pd
import datetime as dt


TO = '00:00:00'
TF = '23:59:00'


#---------------------------------------------------------------------------------------#
# Esta función elimina listas vacías de un archivo .mis
def deleteSpaceWhite(file):
    archive = open(file)                     # file es la ruta del archivo .mis
    archive_list = list(csv.reader(archive)) # Escribe el archivo .mis
    DEL = []
    for i in archive_list:
        if i!=[]:                            # Elimina listas vacías en el archivo
            DEL.append(i)
    return DEL                               # Regresa el archivo .mis en un lista sin LISTAS VACÌAS


#---------------------------------------------------------------------------------------#
# Esta función detecta si  existe un error de medición en archivo .mis, principalmente errores de falta y/o 
#excedente de elementos de la medición (si sobra o falta algùn dato), imprime las lineas de erros del archivo.
def checkmis(list_mis):
    lens = list(map(len,list_mis))           # longitud de los datos de cada medición
    check = []                               # lista vacía
    linea = 0                                # contador para saber línea de error

    for i in lens:
        linea = linea + 1 
        if i==15 or i==1039:                 # i=15 significa una medición sin lluvia, i=1039 una medición con 
                                             # lluvia
            check.append(1)
        else:
            check.append(0)
            print('ERROR en linea',linea,'  <--------------------------------------------------------- E R R O R')
                                                          # Imprime error y el numero de línea del error
    if sum(check)/len(check)==1:
        print('Los archivos estan completos \n ...')
    else:
        print('Los archivos NO estan completos \n ...')
    

#---------------------------------------------------------------------------------------#
# Esta función detecta si existe algun archivo .mis (pertenciente a un directorio) con errores de escritura e
# imprime los erros de linea de cada archivo .mis (si los hay).

def checkParsivel_directory(file):           # file es el directorio de la carpeta de archivos .mis
    print('PATH =', file, '\n')
    print('Buscando errores... \n ...')
    entries = os.listdir(file)               # lista de archivos .mis contenidos en el directorio(carpeta)
    entries.sort()
    files_mis = [file+i for i in entries]    # lista de directorios de cada archivo .mis
    for i in files_mis:
        print(i)
        mis = deleteSpaceWhite(i)            # Aplica función deleteSpaceWhite()
        checkmis(mis)                        # Aplica función checkmis()
    print('No hay ningun error!\n')
    return 


#---------------------------------------------------------------------------------------#
# Esta función convierte la fecha de medición al formato YYYY-MM-DD hh:mm:ss
def joinDate(date_list):
    YYMMDD = date_list[0][6:] + date_list[0][2] + date_list[0][3:6] + date_list[0][0:2] 
    hhmmss = date_list[1]
    JOIN = [YYMMDD.replace('.','-') + ' ' + hhmmss]
    return JOIN


#---------------------------------------------------------------------------------------#
def cleanVariables(list_variables):
# Esta función cambia el tipo de dato de la varibles medidas, las cuáles se encuentran en un tipo de dato 'str' 
# y lo reescribe a valores de tipo 'float'
    VAR = [float(i) for i  in list_variables]
    return VAR


#---------------------------------------------------------------------------------------#
# Esta función cambia el espectro medido, el cuál se encuentra en un tipo de dato 'str' y lo reescribe 
# a valores de tipo 'float'
def cleanSpectrum(spectrum_list):
    SPT = []
    for i in spectrum_list:
        if i=='':
            SPT.append(0.)
        else:
            SPT.append(float(i))
    return SPT


#---------------------------------------------------------------------------------------#
# Esta funcòn aplana una lista de listas una de una dimensión a un lista de una dimensión
def flatten(list_of_list):
    FLAT = []
    for i in list_of_list:
        for j in i:
            FLAT.append(j)
    return FLAT

################################################################################################################
################################################################################################################
################################################################################################################
DATA = flatten([[0]*11,[2],['<SPECTRUM>'],[0]*1024,['<SPECTRUM>']])

def generateDatelist(dates,TYPE='datetime.datetime',step=1):
    DATES = []
    if type(dates[0]) == type(dates[1]) == dt.datetime:
        date_0 = dates[0]
        date_f = dates[1]
    elif type(dates[0]) == type(dates[1]) == str:
        date_0 = dt.datetime.strptime(dates[0],'%Y-%m-%d %H:%M:%S')
        date_f = dt.datetime.strptime(dates[1],'%Y-%m-%d %H:%M:%S')
    else:
        print('Error: Input the elements of the tuple "dates" as str type or as datetime.datetime type.')

    date_difference = date_f -date_0
    date_difference = int(date_difference.seconds/60)

    DATES.append(date_0)
    while date_0 < date_f:
        date_i = date_0 + dt.timedelta(minutes=step)
        DATES.append(date_i)
        date_0 = date_i

    if TYPE == 'datetime.datetime':
        OUTPUT = DATES
    elif TYPE == 'str':
        OUTPUT = [ dt.datetime.strftime(i,'%Y-%m-%d %H:%M:%S') for i in DATES ]
              
    return OUTPUT

################################################################################################################
################################################################################################################
################################################################################################################

def complete_between_T0andTF(mis,date_time):
    mis = [ list(i) for i in mis]
    
    CLEAN = []
    CLEAN.append(mis[0])
    for i in range(len(mis)-1):
        date_difference = date_time[i+1] - date_time[i]
        DT = date_difference.seconds/60

        if DT == 1:
                CLEAN.append(mis[i+1])
        elif DT > 1:
            date_unknown = generateDatelist((date_time[i],date_time[i+1]),TYPE='str')[1:-1]
            for j in date_unknown:
                add = [j] + DATA
                CLEAN.append(add)

            CLEAN.append(mis[i+1])
            
    mis = np.array(CLEAN)
        
    return mis

################################################################################################################
################################################################################################################
################################################################################################################

def complete_noTF(mis,date_mis):
    
    CLEAN = []
    
    date_mis.append(date_mis[0][:10] + ' ' + TF)
    date_time = [ dt.datetime.strptime(i,'%Y-%m-%d %H:%M:%S') for i in date_mis]

    date_difference = date_time[-1] - date_time[-2]
    DT = date_difference.seconds/60
    date_unknown = generateDatelist((date_time[-2],date_time[-1]),TYPE='str')[1:]
    for i in date_unknown:
        add = [i] + DATA
        CLEAN.append(add)
    CLEAN = [list(i) for i in mis] + CLEAN

    mis = pd.DataFrame(CLEAN).values
    return mis


################################################################################################################
################################################################################################################
################################################################################################################

def complete_noTO(mis,date_mis):
    
    CLEAN = []
    
    date_mis = [date_mis[0][:10] + ' ' + TO] + date_mis
    date_time = [ dt.datetime.strptime(i,'%Y-%m-%d %H:%M:%S') for i in date_mis]
    
    date_difference = date_time[1] - date_time[0]
    DT = date_difference.seconds/60
    date_unknown = generateDatelist((date_time[0],date_time[1]),TYPE='str')[:-1]
    for i in date_unknown:
        add = [i] + DATA
        CLEAN.append(add)
    CLEAN = CLEAN + [list(i) for i in mis]
    
    mis = pd.DataFrame(CLEAN).values
    return mis

################################################################################################################
################################################################################################################
################################################################################################################

def complete_noTOandTF(mis,date_mis):
    
    mis = complete_noTO(mis,date_mis)
    date_mis = list(mis[:,0])
    
    mis = complete_noTF(mis,date_mis)
    
    return mis

################################################################################################################
################################################################################################################
################################################################################################################

def complete_data(mis,TO='00:00:00',TF='23:59:00',DATA=flatten([[0]*11,[2],['<SPECTRUM>'],[0]*1024,['<SPECTRUM>']])):

    date_mis = list(mis[:,0])

    #STRING = ['<SPECTRUM>']
    #TO = '00:00:00'
    #TF = '23:59:00'
    #print(len(mis),'\n',mis.shape) 
    #DATA = flatten([[0]*11,[2],STRING,[0]*1024,STRING])

    if date_mis[0][11:] == TO and date_mis[-1][11:] == TF:
        date_time = [ dt.datetime.strptime(i,'%Y-%m-%d %H:%M:%S') for i in date_mis]
        #print(1)
    elif date_mis[0][11:] != TO and date_mis[-1][11:] == TF:

        mis = complete_noTO(mis,date_mis)
        date_mis = list(mis[:,0])
        date_time = [ dt.datetime.strptime(i,'%Y-%m-%d %H:%M:%S') for i in date_mis]

        #print(2)
    elif date_mis[0][11:] == TO and date_mis[-1][11:] != TF:

        mis = complete_noTF(mis,date_mis)
        date_mis = list(mis[:,0])
        date_time = [ dt.datetime.strptime(i,'%Y-%m-%d %H:%M:%S') for i in date_mis]

        #print(3)
    elif date_mis[0][11:] != TO and date_mis[-1][11:] != TF:

        mis = complete_noTOandTF(mis,date_mis)
        date_mis = list(mis[:,0])
        date_time = [ dt.datetime.strptime(i,'%Y-%m-%d %H:%M:%S') for i in date_mis]

        #print(4)

    mis = complete_between_T0andTF(mis,date_time)

    #print(len(mis),'\n',mis.shape) 
    
    return mis

################################################################################################################
################################################################################################################
################################################################################################################

#---------------------------------------------------------------------------------------#
# Esta función limpia un archivo .mis dado su fichero y lo envia (almacena) en FILE_OUT 
def cleanMIS(file,file_out):
    #FILE_OUT = file.replace('0a','1a') # <---------------
                                       # Para cambiar la dirección donde se va guardar el archivo, escriba aqui la 
                                       # ruta, de lo contrario se guardara a la carpeta /1a
        
    mis = deleteSpaceWhite(file) # llamando función 'deleteSpaceWhite()' 
    MIS = []
    for i in mis:
        dates = joinDate(i[:2]) # llamando función 'joinDate()'
        variables = cleanVariables(i[2:14]) # llamando función 'cleanVariables()' 
        string0 = ['<SPECTRUM>']
        if i[-1]=='<SPECTRUM>ZERO</SPECTRUM>':
            spectrum = [0]*1024 # lista de ceros con longitud de 1024 elementos

        else:
            spectrum = cleanSpectrum(i[15:-1]) # llamando función 'cleanSpectrum()'
            spectrum.append(0.)

        measurement = flatten([dates,variables,string0,spectrum,string0]) # llamando función 'flatten()'
        MIS.append(measurement)
    
    if len(MIS) == 1440:
    	df_mis = pd.DataFrame(MIS)
    	df_mis.to_csv(file_out,header=None,index=None)
    elif len(MIS) != 1440:
    	mis = pd.DataFrame(MIS).values
    	MIS = complete_data(mis)
    	df_mis = pd.DataFrame(MIS)
    	df_mis.to_csv(file_out,header=None,index=None) # exporta archivo a .mis

    return 


#------------------------------------------------------------------------------------------#
# Esta función devuelve los archivos de un directorio .mis limpios a un formato de archivo 
# especifico(.txt, .dat, .csv, ..., etc)
def cleanParsivel(PATH,PATH_OUT,formate):
    print('Limpiado archivos .mis ... \n ...')
    file_list = os.listdir(PATH)               # lista de archivos .mis contenidos en el directorio(carpeta)
    file_list.sort()                           # Ordenada alfabéticamente los ficheros del directorio
    for file in file_list:
        i = PATH + file                        # ruta del fichero
        print(i)
        j = PATH_OUT + file.replace(file[-4:],formate)
        cleanMIS(i,j)                            # Aplica función cleanMIS()
        print(j)
        print('Archivo limpio \n ...')
    print('\n PROCESO TERMINADO!')
        
        
        
        
#**************************************************************************************************#
# for i in ['12']:
#     PATH     = '/run/user/1000/gvfs/smb-share:server=synologynas.ugr.es,share=gfatnas/datos/parsivel/0a/2018/' + i +'/'
#     PATH_OUT = '/run/user/1000/gvfs/smb-share:server=synologynas.ugr.es,share=gfatnas/datos/parsivel/1a/2018/' + i +'/'
#     cleanParsivel(PATH,PATH_OUT,'.mis')
    