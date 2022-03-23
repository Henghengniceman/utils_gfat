#!/bin/bash

PYTHON=/usr/local/software/venvs/gfat/bin/python
PROGRAM=/usr/local/software/gfat/utils_gfat/plot_disdrometer.py

while getopts s:i:e:h: option
do
case "${option}"
in
s) station=${OPTARG};; #instrument nick name (e.g., mhc)
i) inidate=$OPTARG;;   #First date to print
e) enddate=${OPTARG};; #Last date to print
h)
  echo "Arguments:"
  echo "-s : station (e.g.,UGR, UGR-CP, UGR-AU)."
  echo "-i : First date to plot. Date format: yyyymmdd"
  echo "-e : Last date to plot. Date format: yyyymmdd"
  ;;
esac
done

echo "Initializing auto_plot_disdrometer"

startdate=$(date -I -d "$inidate") || exit -1
enddate=$(date -I -d "$enddate") || exit -1

echo "startdate: " ${startdate}
echo "enddate: " ${enddate}

d="$startdate"
while [ "$d" != "$enddate" ]; do
  echo "Processing day: " $d
  cdate=$(date +"%Y%m%d" -d $d)  
  dateplustmp=$(date -I -d "$d + 1 day")  
  dateplus=$(date -d $dateplustmp +"%Y%m%d")  
  echo "sudo ${PYTHON} ${PROGRAM} -s ${station} -i $cdate -e $dateplus"
  sudo ${PYTHON} ${PROGRAM} -s ${station} -i $cdate  -e $dateplus 
  d=$dateplustmp
done
