#!/bin/bash

PYTHON=/usr/local/software/venvs/gfat/bin/python

while getopts l:i:e:h: option
do
case "${option}"
in
l) instrument_name=${OPTARG};; #instrument nick name (e.g., mhc)
i) inidate=$OPTARG;;   #First date to print
e) enddate=${OPTARG};; #Last date to print
h)
  echo "Arguments:"
  echo "-l : instrument nick name (e.g.,mhc, vlt, rpg, psv)."
  echo "-i : First date to plot."
  echo "-e : Last date to plot."
  ;;
esac
done

echo "Initializing auto_quicklook_v02"

case "${instrument_name}"
in
mhc)
  fullname='MULHACEN'
  PROGRAM=/usr/local/software/gfat/utils_gfat/quicklook_lidar.py
  #Main directory
  mdir="/mnt/NASGFAT/datos/${fullname}/1a"
  #Type filename
  fileformat="mhc_1a_Prs_"
  #Ouput directory
  odir="/mnt/NASGFAT/datos/${fullname}/quicklooks"
  ;;  
vlt)
  fullname='VELETA'
  PROGRAM=/usr/local/software/gfat/utils_gfat/quicklook_lidar.py
  #Main directory
  mdir="/mnt/NASGFAT/datos/${fullname}/1a"
  
  #Type filename
  fileformat="vlt_1a_Prs_"
  
  #Ouput directory
  odir="/mnt/NASGFAT/datos/${fullname}/quicklooks"
  ;;
wradar)
  fullname='rpgradar'
  #Main directory
  mdir="/mnt/NASGFAT/datos/${fullname}/1a"

  #Ouput directory
  odir="/mnt/NASGFAT/datos/${fullname}/quicklooks"
  PROGRAM=/usr/local/software/gfat/utils_gfat/quicklook_radar.py
  ;;
halo)
  fullname='DOPPLER'
  #Main directory
  mdir="/mnt/NASGFAT/datos/${fullname}/Data/Granada/products/windvad75"

  #Ouput directory
  odir="/mnt/NASGFAT/datos/${fullname}/Data/Granada/quicklooks"
  PROGRAM=
  
  ;;
psv)
  fullname='parsivel'
  PROGRAM=quicklook_parsivel.py
  ;;

noaaRT)   
  #Main directory
  mdir="/mnt/NASGFAT/datos/IN-SITU/Data/noaaRT"
  
  #Ouput directory
  odir="/mnt/NASGFAT/datos/IN-SITU/Quicklooks/noaaRT"
  PROGRAM=/usr/local/software/gfat/utils_gfat/noaaRT_plot.py
  ;;  
*)
  echo "Fatal error. Instrument, ${instrument_name}, does not considered in the program."
  exit 0
  ;;
esac

startdate=$(date -I -d "$inidate") || exit -1
enddate=$(date -I -d "$enddate") || exit -1

echo "startdate: " ${startdate}
echo "enddate: " ${enddate}

d="$startdate"
while [ "$d" != "$enddate" ]; do
  echo "Processing day: " $d
  cdate=$(date -d $d +"%Y%m%d")  
  cyear=${cdate:0:4}
  cmonth=${cdate:4:2}
  cday=${cdate:6:2} 

  case "${instrument_name}"
  in
  mhc)
    echo "${instrument_name} quicklook"
    idir="$mdir/$cyear/$cmonth/$cday/${fileformat}*${cdate}.nc"
    
    #Create folder for 532xpa
    odir_tmp="$odir/532xpa/$cyear"
    if [[ ! -e ${odir_tmp} ]]; then    
      echo "Trying to create folder..."
      sudo mkdir -p ${odir_tmp}
      if [[ -e ${odir_tmp} ]]; then
        echo "Folder created:" ${odir_tmp}
      else
        echo "ERROR: impossible to create folder."  
      fi
    else
      echo "${odir_tmp} already exists."  
    fi    
    #Launch plot 532xpa
    echo "sudo ${PYTHON} ${PROGRAM} -i $idir -t $odir_tmp -d $cdate --channel 0"
    sudo ${PYTHON} ${PROGRAM} -i $idir -t $odir_tmp -d $cdate --channel 0

    #Create folder for 1064xta
    odir_tmp="$odir/1064xta/$cyear"    
    if [[ ! -e ${odir_tmp} ]]; then    
      echo "Trying to create folder..."
      sudo mkdir -p ${odir_tmp}
      if [[ -e ${odir_tmp} ]]; then
        echo "Folder created:" ${odir_tmp}
      else
        echo "ERROR: impossible to create folder."  
      fi
    else
      echo "${odir_tmp} already exists."  
    fi    
    #Launch plot 1064xta
    echo "sudo ${PYTHON} ${PROGRAM} -i $idir -t $odir_tmp -d $cdate --channel 6"
    sudo ${PYTHON} ${PROGRAM} -i $idir -t $odir_tmp -d $cdate --channel 6
    
    #Create folder for 532xpp
    odir_tmp="$odir/532xpp/$cyear"    
    if [[ ! -e ${odir_tmp} ]]; then    
      echo "Trying to create folder..."
      sudo mkdir -p ${odir_tmp}
      if [[ -e ${odir_tmp} ]]; then
        echo "Folder created:" ${odir_tmp}
      else
        echo "ERROR: impossible to create folder."  
      fi
    else
      echo "${odir_tmp} already exists."  
    fi    
    echo "sudo ${PYTHON} ${PROGRAM} -i $idir -t $odir_tmp -d $cdate --channel 1 --altitude_max=25"
    sudo ${PYTHON} ${PROGRAM} -i $idir -t $odir_tmp -d $cdate --channel 1 --altitude_max=25
    ;;
  vlt)
    echo "${instrument_name} quicklook"
    idir="$mdir/$cyear/$cmonth/$cday/${fileformat}*${cdate}.nc"
    
    #Create folder for 355xpa
    odir_tmp="$odir/355xpa/$cyear"
    if [[ ! -e ${odir_tmp} ]]; then    
      echo "Trying to create folder..."
      sudo mkdir -p ${odir_tmp}
      if [[ -e ${odir_tmp} ]]; then
        echo "Folder created:" ${odir_tmp}
      else
        echo "ERROR: impossible to create folder."  
      fi
    else
      echo "${odir_tmp} already exists."  
    fi    
    #Launch plot 355xpa
    echo "sudo ${PYTHON} ${PROGRAM} -i $idir -t $odir_tmp -d $cdate --channel 0"
    sudo ${PYTHON} ${PROGRAM} -i $idir -t $odir_tmp -d $cdate --channel 0
    ;;
  noaaRT)
    noaadate=$(date -d $d +"%Y%d%m")
    idir="$mdir/${noaadate}.csv"
    echo "sudo ${PYTHON} ${PROGRAM} -i $idir -t $odir -d $cdate"
    sudo ${PYTHON} ${PROGRAM} -i $idir -t $odir -d $cdate
    ;;  
  wradar)
    idir="$mdir/$cyear/$cmonth/$cday"
    echo "sudo ${PYTHON} ${PROGRAM} -i $idir -t $odir -d $cdate"
    sudo ${PYTHON} ${PROGRAM} -i $idir -t $odir -d $cdate
    ;;   
  *)
    echo "sudo ${PYTHON} ${PROGRAM} -i $idir -t $odir -d $cdate"
    sudo ${PYTHON} ${PROGRAM} -i $idir -t $odir -d $cdate
    ;;  
  esac
  
  d=$(date -I -d "$d + 1 day")
done
