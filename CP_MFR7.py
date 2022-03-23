# -*- coding: utf-8 -*-
"""
Created on 25/01/2016
@author: Roberto Román
@e-mail: robertor@ugr.es
ENVIA LOS DATOS DE MFR7 EN EL DISCO MFR7FTP/ AL NASGFAT/
"""

import os
import time
import datetime
import math

import shutil


def hacerCOPY(dir2): #main():

    Output_Directory = '/mnt/NASGFAT/datos/MFR7/'+dir2 #Directorio al que van los datos 
    direc="/mnt/MFR7FTP/"+dir2  #Directorio donde se encuentran los datos a copiar
    lista=os.listdir(direc) #Listado de los ficheros que hay en nuestro directorio donde se encuentran los datos a subir
    lonlist=len(lista) 
    listaFTP=os.listdir(Output_Directory) #Listado de los ficheros que hay en el directorio seleccionado del ftp 
    salir=0

    for i in (range(lonlist)): #Se hace un barrido para comporbar cada fichero de datos
        nam=lista[i];
 #       print nam
        if  os.path.isdir(direc+nam):
            continue
        byte= os.path.getsize(direc+ lista[i]) #Tamanno del fichero a subir
        for n in (range(len(listaFTP))):  #Se hace un barrido para comprobar si el fichero de datos a subir ya existe y de ser así, si tiene el mismo tamanno
            namFTP=listaFTP[n];
            if nam==namFTP:
                byteFTP=os.path.getsize(Output_Directory+ listaFTP[n])
  #              try:
  #                  byteFTP=ftp.size(listaFTP[n])
  #              except: 
  #                  byteFTP=-1.0 

                if byteFTP==byte:
                    salir=1
                else:
                    salir=0
                    #ftp.delete(nam) #Existe el fichero pero no tiene el mismo tamaño, así que se borra el del ftp 
                    #time.sleep(10)
                break
            else:
                continue     
            
        if salir==1: 
            salir=0 #Ya existe el fichero y tiene el mismo tamanno
            continue
        else:
            File2Send = direc+ nam
            shutil.copy(File2Send, Output_Directory)




def main():

    dt = datetime.datetime.utcnow()
    print "Connection",  dt.day, "/",dt.month,  "/",dt.year,  " ",dt.hour, ":", dt.minute




    
    hacerCOPY("ZOUT/")
    hacerCOPY("ZOUT/AOD/")
    hacerCOPY("ZLOG/")
    hacerCOPY("ZDATA/")
    hacerCOPY("SOL_CAL/")
    hacerCOPY("scripts/")
    hacerCOPY("junk/")
    hacerCOPY("")

    return 0

if __name__ == '__main__':
    main()






