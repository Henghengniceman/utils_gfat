#!/bin/bash

PYTHON=/usr/local/software/venvs/gfat/bin/python
PROGRAM=/usr/local/software/gfat/utils_gfat/figures2web_v03.py
now=$(date +"%Y%m%d_%H%M")
echo "Initializing $PROGRAM on $now"

while getopts s:l:i:e:h: option
do
case "${option}"
in
s) station_name=${OPTARG};; #instrument nick name (e.g., mhc)
l) instrument_name=${OPTARG};; #instrument nick name (e.g., mhc)
i) inidate=$OPTARG;;   #First date to print
e) enddate=${OPTARG};; #Last date to print
h)
  echo "Arguments:"
  echo "-s : station (e.g.,granada, juelich)."
  echo "-l : instrument nick name (e.g.,mhc, vlt, wradar, ceilo, smps)."
  echo "-i : First date to plot."
  echo "-e : Last date to plot."
  ;;
esac
done

startdate=$(date -I -d "$inidate") || exit -1
enddate=$(date -I -d "$enddate") || exit -1

echo "startdate: " ${startdate}
echo "enddate: " ${enddate}

d="$startdate"
while [ "$d" != "$enddate" ]; do
  echo "Processing day: " $d
  cdate=$(date -d $d +"%Y%m%d")
  echo sudo ${PYTHON} ${PROGRAM} ${station_name} ${instrument_name} $cdate
  sudo ${PYTHON} ${PROGRAM} ${station_name} ${instrument_name} $cdate
  d=$(date -I -d "$d + 1 day")
done
