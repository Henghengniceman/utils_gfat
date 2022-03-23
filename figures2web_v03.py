#!/usr/bin/env python
# coding: utf-8

"""
Move figures to the ftp of the quicklook website
python figures2web_v03.py wradar 20190520
"""

# In[31]:
import os
import sys
import platform
import glob
import datetime as dt
import ftplib
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib
from matplotlib.dates import DateFormatter
from optparse import OptionParser

# In[4]:
#Inputs: Instrument name and date
try: 
    station = sys.argv[1]
except:    
    station = 'granada'

print('Station chosen: %s' % station)

try: 
    instrument = sys.argv[2]
except:
    instrument = 'mhc'

print('Instrument chosen: %s' % instrument)

try: 
    date2plot = sys.argv[3]
except:
    yesterday = dt.datetime.today() - dt.timedelta(1)
    date2plot = yesterday.strftime('%Y%m%d')

print('date2plot chosen: %s' % date2plot)

#Information to access to the server
address = 'ftpwpd.ugr.es'
port = 21
user = 'gfat'
pwd = '4evergfat' # poner la contrase�a
server_figure_path='public_html/quicklooks/plots'


# In[33]:
date2plot = dt.datetime.strptime(date2plot, '%Y%m%d')    
year=dt.datetime.strftime(date2plot, '%Y') 
month=dt.datetime.strftime(date2plot, '%m')
day=dt.datetime.strftime(date2plot, '%d')

# In[13]:
if platform.system() == 'Windows':
    mpathWin = {'mhc': 'Y:\\datos\\MULHACEN\\quicklooks', \
                'wradar': 'Y:\\datos\\rpgradar\\quicklooks', \
                'ceilo': 'Y:\\datos\\iCeNet\\data\\UGR\\quicklooks', \
                'smps': 'Y:\\datos\\IN-SITU\\Quicklooks\\SMPS', \
                'noaaRT': 'Y:\\datos\\IN-SITU\\Quicklooks\\noaaRT', \
                'cloudnet': 'Y:\\datos\\CLOUDNET'}    
    mpath = mpathWin
else:
    mpathLin = {'mhc': '/mnt/NASGFAT/datos/MULHACEN/quicklooks', \
                'wradar': '/mnt/NASGFAT/datos/rpgradar/quicklooks', \
                'ceilo': '/mnt/NASGFAT/datos/iCeNet/data/UGR/quicklooks', \
                'smps': '/mnt/NASGFAT/datos/IN-SITU/Quicklooks/SMPS', \
                'noaaRT': '/mnt/NASGFAT/datos/IN-SITU/Quicklooks/noaaRT', \
                'cloudnet': '/mnt/NASGFAT/datos/CLOUDNET'}
    mpath = mpathLin

# In[14]:
path={}
filename={}
rDir={}
if station == 'granada':
    if instrument == 'mhc':    
        #Extension
        extension = '.png'
        #Main part of the path
        path['rcs'] = os.path.join(mpath[instrument], year, '532xpa')
        path['scc436'] = os.path.join(mpath[instrument], 'scc436', 'output', year, month, day, 'daily')
        #Name structure (admit regular expressions)
        filename['rcs'] = ''.join(['rcs532xpa_', year, month, day, extension])
        filename['scc436'] = ''.join(['beta*', year, month, day, extension])        
        #Remote folder in the server
        rDir['rcs'] = 'mulhacen/quicklooks'
        rDir['scc436'] = 'mulhacen/inversiones'
    elif instrument == 'wradar':
        extension = '.png'
        path['dbze'] = os.path.join(mpath[instrument], year, 'dbZe')
        path['vm'] = os.path.join(mpath[instrument], year, 'vm')
        path['sigma'] = os.path.join(mpath[instrument], year, 'sigma')
        filename['dbze'] = ''.join(['wradar_gr_dBZe_', year, month, day, extension])
        filename['vm'] = ''.join(['wradar_gr_vm_', year, month, day, extension]) 
        filename['sigma'] = ''.join(['wradar_gr_sigma_', year, month, day, extension])
        rDir['dbze'] = 'radar/dbZe'
        rDir['vm'] = 'radar/vm'
        rDir['sigma'] = 'radar/sigma'                        
    elif instrument == 'ceilo':
        extension = '.jpg'
        path['rcs'] = os.path.join(mpath[instrument], year)
        filename['rcs'] = ''.join(['CHM_QL_', year, month, day, extension])
        rDir['rcs'] = 'ceilo'
    elif instrument == 'smps':
        extension = '.png'
        path['psd'] = mpath[instrument]
        filename['psd'] = ''.join([year, '*', extension])
        rDir['psd'] = 'smps'
    elif instrument == 'noaaRT':
        extension = '.png'
        path['noaaRT'] = mpath[instrument]
        filename['noaaRT'] = ''.join(['noaaRT_', year, month, day, extension])
        rDir['noaaRT'] = 'noaaRT'
elif station == 'juelich':
    if instrument == 'cloudnet':
        extension = '.png'
        path['dbze'] = os.path.join(mpath[instrument], station, 'quicklooks', 'dbZe')
        path['vm'] = os.path.join(mpath[instrument], station, 'quicklooks', 'v')
        path['beta'] = os.path.join(mpath[instrument], station, 'quicklooks', 'beta')
        path['lwp'] = os.path.join(mpath[instrument], station, 'quicklooks', 'LWP')
        path['Nd'] = os.path.join(mpath[instrument], station, 'aci')
        path['re'] = os.path.join(mpath[instrument], station, 'aci')
        path['Zcl'] = os.path.join(mpath[instrument], station, 'aci')
        
        filename['dbze'] = ''.join(['cloudnet_jue_dBZe_', year, month, day, extension])
        filename['vm'] = ''.join(['cloudnet_jue_v_', year, month, day, extension]) 
        filename['beta'] = ''.join(['cloudnet_jue_beta_', year, month, day, extension]) 
        filename['lwp'] = ''.join(['cloudnet_jue_lwp_', year, month, day, extension]) 
        filename['Nd'] = ''.join(['cloudnet_jue_Nd_', year, month, day, extension]) 
        filename['re'] = ''.join(['cloudnet_jue_re_', year, month, day, extension]) 
        filename['Zcl'] = ''.join(['cloudnet_jue_Zcl_', year, month, day, extension]) 
        
        rDir['dbze'] = 'cloudnet/dBZe'
        rDir['vm'] = 'cloudnet/vm'
        rDir['beta'] = 'cloudnet/attBeta'
        rDir['lwp'] = 'cloudnet/LWP'
        rDir['Nd'] = 'ACI/Nd'
        rDir['re'] = 'ACI/re'
        rDir['Zcl'] = 'ACI/Zcl'        
        
print('Instrument chosen: %s' % instrument)

# In[28]:
code = 0
for field in path.keys():    
    #print('Current field: %s' % field)
    #print(path[field])
    #print(filename[field])    
    fullpath = os.path.join(path[field], filename[field])
    #print('Current fullpath: %s' % fullpath)
    files2send = glob.glob(fullpath)
    #print('files2send: %s' % files2send)
    if files2send:
        print('Files found as %s' % fullpath)
        remoteDir = os.path.join(server_figure_path, station, rDir[field]) # poner el directorio correcto
        print('remoteDir %s' % remoteDir)
        for tmp_file in files2send:      
            print('Current file to send: %s' % tmp_file)
            ftp = ftplib.FTP(address)
            ftp.set_pasv(True)
            try:
                ftp.connect(address, port)
                ftp.login(user, pwd)
                ftp.cwd(remoteDir)
                print("pwd: %s" % ftp.pwd())                
                with open(tmp_file, "rb") as FILE:                    
                    ftp.storbinary('STOR ' + os.path.basename(tmp_file), FILE)
                ftp.close()
                code = 1
            except:
                code = 0
                ftp.close()                
            if code:
                print('Succesfully sent: %s' %  os.path.join(remoteDir, os.path.basename(tmp_file)))
            else:
                print(': Unable to send file %s' % os.path.basename(tmp_file))
    else:
        print('No files as %s' % fullpath)
print('###END###')