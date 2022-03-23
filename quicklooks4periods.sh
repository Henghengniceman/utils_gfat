#!/bin/bash

while getopts n:l:s:e:h: option
do
case "${option}"
in
n) instrument=${OPTARG};; #instrument nick name (e.g., wradar)
l) location= ${OPTARG};; #station nickname (e.g, gr)
s) firstdate=$OPTARG;;   #First date to print
e) lastdate=${OPTARG};; #Last date to print
h)
  echo "Arguments:"
  echo "-n : instrument nickname (e.g.,mhc, wradar, psv)."
  echo "-s : First date to plot."
  echo "-e : Last date to plot."
  echo "-l : Location to plot."
  ;;
esac
done

scriptname="quicklook4periods.sh"

now=$(date +"%Y%m%d%H%M%S")

#Program information
PYTHON=/usr/local/software/venvs/gfat/bin/python
UTILS_GFAT_DN=/usr/local/software/gfat/utils_gfat

if [ $instrument = "rpg" ]; then
  SCRIPT="${UTILS_GFAT_DN}/rpg.py"  
elif [ $instrument = "mhc" ]; then
  SCRIPT="${UTILS_GFAT_DN}/lidar.py"
elif [ $instrument = "vlt" ]; then
  SCRIPT="${UTILS_GFAT_DN}/lidar.py"
  mdir="/mnt/NASGFAT/datos/     /1a"
  figdir="/mnt/NASGFAT/datos/   /quicklooks"
elif [ $instrument = "mwr" ]; then
  SCRIPT="${UTILS_GFAT_DN}/    .py"
  mdir="/mnt/NASGFAT/datos/    /1a"
  figdir="/mnt/NASGFAT/datos/    /quicklooks"
elif [ $instrument = "doppler" ]; then
  SCRIPT="${UTILS_GFAT_DN}/   .py"
  mdir="/mnt/NASGFAT/datos/      /1a"
  figdir="/mnt/NASGFAT/datos/    /quicklooks"
fi

echo "$PYTHON $SCRIPT -i "$firstdate" -e "$lastdate"" #>> ${logpath}
$PYTHON $SCRIPT -i "$firstdate" -e "$lastdate" #>> ${logpath} 2>&1  
