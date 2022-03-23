# -*- coding: utf-8 -*-
"""
Created on 18/11/2015
@author: Roberto Román
@e-mail: robertor@ugr.es

@version: 1.0
Hace una copia en el disco NAS de los datos del MFR7 

"""
import os
import time
import datetime
import subprocess
import ephem
import math
from optparse import OptionParser



def main():
    dir_out="/mnt/NASGFAT/datos/MFR7/"
    subdir = '/mnt/NASFTP/MFR7/ZOUT' 
  
			
    command1 = 'sudo cp -r %s %s' % (subdir, dir_out)
    eje1=subprocess.Popen(command1, shell=True)
 
    dir_out="/mnt/NASGFAT/datos/MFR7/"
    subdir = '/mnt/NASFTP/MFR7/ZLOG'
  

    command2 = 'sudo cp -r %s %s' % (subdir, dir_out)
    eje2=subprocess.Popen(command2, shell=True)

    dir_out="/mnt/NASGFAT/datos/MFR7/"
    subdir = '/mnt/NASFTP/MFR7/ZDATA'
  

    command3 = 'sudo cp -r %s %s' % (subdir, dir_out)
    eje3=subprocess.Popen(command3, shell=True)



if __name__ == '__main__':
    main()
