# -*- coding: utf-8 -*-
"""
Created on 18/11/2015
@author: Roberto Román
@e-mail: robertor@ugr.es

@version: 1.0
Hace una copia en el disco NAS de los datos del datalogger del albergue

"""
import os
import time
import datetime
import subprocess
import ephem
import math
from optparse import OptionParser



def main():
    dir_out="/mnt/NASGFAT/datos/DATALOGGER_SNS/Datos/"
    subdir = '/mnt/NASFTP/albergue' 
  
			
    command1 = 'sudo cp -r %s %s' % (subdir, dir_out)
    eje1=subprocess.Popen(command1, shell=True)
 

if __name__ == '__main__':
    main()
