# -*- coding: utf-8 -*-
"""
Created on 17/07/2015
@author: Roberto Román
@e-mail: robertor@goa.uva.es
#ESTE PROGRAMA ACCEDE A LA CARPETA HOME/CEILOMETRO/DATA Y MANDA LOS DATOS DE ESTA CARPETA AL FTP 
#DE LA RED DE CEILOMETROS SI ESTE ARCHIVO NO EXISTE EN EL FTP
#PARA VER SI EXISTE EL MISMO ARCHIVO COMPARA EL NOMBRE DEL FICHERO Y EL TAMAÑO
#SE VA A EJECUTAR TODOS LOS DÍAS A LAS 03:07 INCLUYENDO EN LA CRONTAB LA SENTENCIA:
7 3 * * * python /home/ceilometro/FTPredCeilometros.py >> /home/ceilometro/log_ftpREDceilometros.txt
#Almacenando los logs en "/home/ceilometro/log_ftpREDceilometros.txt"
"""

import os
import time
import datetime
import math

from ftplib import FTP 


def main():
    dt = datetime.datetime.utcnow()
    print "Connection",  dt.day, "/",dt.month,  "/",dt.year,  " ",dt.hour, ":", dt.minute
    for yy in range(2016, dt.year+1): #Se hace un barrido para comporbar cada año


        Output_Directory = "/UGR/L0/%04d/" % (yy) #Directorio al que van los datos dentro del ftp
        direc='/home/roberto/provisional/DatosCeilometro/' #Directorio donde se encuentran los datos del ceilometro a subir al ftp
        lista=os.listdir(direc) #Listado de los ficheros que hay en nuestro directorio donde se encuentran los datos a subir
        lonlist=len(lista) 
        ftp = FTP() 
  #  ftp.close()
        ftp.set_pasv(False)
        try:
	    ftp.connect('aire.ugr.es', 54840, -999)
	    ftp.login('ceilometer', 'iberian')
	    ftp.cwd(Output_Directory)
	    listaFTP=ftp.nlst() #Listado de los ficheros que hay en el directorio seleccionado del ftp 
        except: 
	    print "Connection failure" 
	    ftp.close()
	    lonlist=0 #Si no puede conectar se pone la longitud de lista igual a cero para que no haga el bucle y directamente se acabe el programa
	#    try:
	#        listaFTP=ftp.nlst() #Listado de los ficheros que hay en el directorio seleccionado del ftp 
	#    except: 
	#        print "Connection failure" 
	#        lonlist=0 #Si no puede conectar se pone la longitud de lista igual a cero para que no haga el bucle y directamente se acabe el programa

	 #   listaFTP=ftp.nlst() #Listado de los ficheros que hay en el directorio seleccionado del ftp 

        salir=0

        for i in (range(lonlist)): #Se hace un barrido para comporbar cada fichero de datos
	    nam=lista[i];
	    if nam.startswith("%04d" % (yy)):
	        bien=1  #El fichero tiene que acabar con la extensión "nc"
	    else:
	        continue
	    if nam.endswith('nc'):
	        bien=1  #El fichero tiene que acabar con la extensión "nc"
	    else:
	        continue
	    if nam[9]=='G': #El fichero tiene que contener la letra V de Valladolid.
	        bien=1
	    else:
	        continue
	    byte= os.path.getsize(direc+ lista[i]) #Tamanno del fichero a subir
	    for n in (range(len(listaFTP))):  #Se hace un barrido para comprobar si el fichero de datos a subir ya existe y de ser así, si tiene el mismo tamanno
	        namFTP=listaFTP[n];
	        if nam==namFTP:
	            byteFTP=ftp.size(listaFTP[n])
	            if byteFTP==byte:
	                salir=1
	            else:
	                salir=0
	                ftp.delete(nam) #Existe el fichero pero no tiene el mismo tamaño, así que se borra el del ftp 
	                time.sleep(10)
	            break
	        else:
	            continue     
	    
	    if salir==1: 
	        salir=0 #Ya existe el fichero y tiene el mismo tamanno
	        continue
	    else:
	        File2Send = direc+ nam
	        FILE = open(File2Send, "rb")
	        ftp.storbinary('STOR ' + nam, FILE) 
	        print "File transfered: " +File2Send
	        time.sleep(10)
        ftp.close()

if __name__ == '__main__':
    main()
