#!/bin/bash

now=$(date +"%Y%m%d_%H%M")
ftpdir="/mnt/NASFTP/noaaRT"
nasdir="/mnt/NASGFAT/datos/IN-SITU/Data/noaaRT"

echo "Initializing FTP2NAS on $now"

while getopts i:e:h: option
do
case "${option}"
in
i) inidate=$OPTARG;;   #First date to print
e) enddate=${OPTARG};; #Last date to print
h)
  echo "Arguments:"
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
  echo "###############################"
  echo "### Processing day: $d"
  echo "###############################"
  cdate=$(date -d $d +"%Y%d%m")
  file2copy="${cdate}.csv"
  path2copy="${ftpdir}/$file2copy"  
  nasfile="${nasdir}/$file2copy"  
  if [[ -e $path2copy ]]; then
    echo "File found..."
    echo "Moving $path2copy to NAS..."
    cp -v $path2copy $nasfile
    if [[ -e $nasfile ]]; then
      echo "Succesfully copied."
    else
      echo "ERROR: file not copied."
    fi
  else
    echo "File not found: $path2copy"
  fi
  d=$(date -I -d "$d + 1 day")
  echo " "
done
