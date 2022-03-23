# -*- coding: utf-8 -*-
"""
#
#
#
"""
import os
import glob
import numpy as np
import datetime as dt
import cloudnet, ACI 

#Debugger
debugger = True

#Folder Paths
path = {}
path['cloudnet'] = 'Y:\\datos\\CLOUDNET'
filename = {}
filepath = {}
station = 'juelich'

#Period to be analyzed
datestr = {}
datestr['min'] = '20180114'
datestr['max'] = '20180115'
datemin = dt.datetime.strptime(datestr['min'], '%Y%m%d')
datemax = dt.datetime.strptime(datestr['max'], '%Y%m%d')

datelist = [datemin + dt.timedelta(days=x) for x in range(0, (datemax-datemin).days)]

for _date in datelist:    
    _datestr = dt.datetime.strftime(_date, '%Y%m%d')
    
    #Categorice Data
    path['cat'] = os.path.join(path['cloudnet'], 'juelich', 'categorize', dt.datetime.strftime(_date,'%Y'))
    filename['cat'] = "%s_%s_categorize.nc" % (_datestr, station)
    filepath['cat'] = os.path.join(path['cat'], filename['cat'])
    print(filepath['cat'])
    
    #Classification Data
    path['cla'] = os.path.join(path['cloudnet'], 'juelich', 'classification', dt.datetime.strftime(_date,'%Y'))
    filename['cla'] = "%s_%s_classification.nc" % (_datestr, station)
    filepath['cla'] = os.path.join(path['cla'], filename['cla'])
    
    print(filepath['cla'])
    
    data = {}
    data['cat'] = cloudnet.categorice_reader(glob.glob(filepath['cat']))
    data['cla'] = cloudnet.classification_reader(glob.glob(filepath['cla']))
    
    if data['cat'] and data['cla']:
        #Debugger plot    
        debugger = True
        if debugger:
            dpi=100
            gapsize= 'default'
            range_limit = 20
            channel=1
            COEFF=2.
            
            # Setting the parameter gapsize
            # -----------------------------------------------------------        
            if gapsize == 'default':        
                dif_time = data['cat']['raw_time'][1:] - data['cat']['raw_time'][0:-1]
                dif_seconds = [tmp.seconds for (i, tmp) in enumerate(dif_time)]                        
                # print(dif_seconds)
                HOLE_SIZE = 2*int(np.ceil((np.median(dif_seconds)/60))) #HOLE_SIZE is defined as the median of the resolution fo the time array (in minutes)        
                print('HOLE_SIZE parameter automatically retrieved to be %d.' % HOLE_SIZE)         
            else:
                print('HOLE_SIZE set by the user: %d (in minutes)' % HOLE_SIZE)
            
            #Plot configuration parameters
            plt_conf =  {
            'mainpath': os.path.join(path['cloudnet'], station, 'quicklooks'),
            'location': station,
            'coeff': COEFF,
            'gapsize': HOLE_SIZE,
            'dpi': dpi,
            'fig_size': (16,5),
            'font_size': 16,
            'y_min': 0,
            'y_max': range_limit,
            'rcs_error_threshold':1.0, }
            
            #CLOUDNET Quicklook         
            cloudnet.plotQuicklook(data['cat'], plt_conf, True)
            
            cloudnet.plotLWP(data['cat'], os.path.join(path['cloudnet'], station, 'quicklooks'), True)
    
        #Filter liquid cloud
        data['filtered'] = ACI.filterCloudLiquid(data['cat'], data['cla'], True, True)
        
        if ~np.isnan(data['filtered']['Zcl']).all():        
            #Full-cloud Mean Reflectivity
            data['average'] = ACI.meanZ(data['filtered'], data['cla']['CBH'], data['cla']['CTH'])
               
        #    print(data['average'])
            
            #Upsampler to 1 minute
            vars2resample = ('lwp', 'meanZ', 'mean_beta')
            timeStep = 'T'
            limit4AvoidNan = 2
            data['upsample'] = ACI.upsampler(data['average'],  vars2resample, timeStep, limit4AvoidNan)
            
            #Microphysical properties
            nu = 8.
            data['mp'] = ACI.cloudMicrophysics(data['cat']['height'], data['upsample'], nu, True, True)
            
            #ACI indexes
            LWPmin, LWPmax, LWPstep, = 0, 150, 10
            minNumber2fit = 15    
            ACIstats = ACI.aciRetrieval(data['mp'], LWPmin, LWPmax, LWPstep, minNumber2fit, True, True)
            
            #Save Aci values
            indexes=('Nd', 're')
            for _index in indexes:
                dateMatrix = np.matlib.repmat([_date.year, _date.month, _date.day], ACIstats[_index].shape[0], 1)
                array2save = np.concatenate((np.asarray(dateMatrix), np.asarray(ACIstats[_index])),axis=1)                 
                acifile = 'aci_%s_%s.csv' % (_index, _datestr)
                path['csv'] = os.path.join(path['cloudnet'], 'juelich','aci_csv')
                np.savetxt(os.path.join(path['csv'], acifile), array2save, delimiter=",", fmt='%2.4f', header='year,month,day,hour,minute,second,Npoints,slope,intercept,R,p,std_err')
            del data
        else:
            print('No liquid cloud on %s' % _date)            
    else:
        print('No data on %s' % _date)            