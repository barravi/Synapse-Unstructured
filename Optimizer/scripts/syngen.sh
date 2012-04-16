#!/bin/bash

# 2012-04-08, Riccardo Loti (loti@di.unito.it)
#
# This script generates iteratively each configuration file for a give nummber of overlays (<ovls>, 2-20) and s1 values
# (<sval>, 0.80-0.20), where each resultant sk, where k!=1, is (1-s1)/(ovls-1).

FILE_DIR=../confs
NAME_HEAD=syn
NAME_TAIL=.txt

for sval in 8 6 4 2 # for each 0.x0 in s1
do
  for ovls in 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 # for each number of overlays between 2 and 20
  do
    file_name=${FILE_DIR}/${NAME_HEAD}${sval}o${ovls}${NAME_TAIL} # the output file name and url
    echo "# $ovls" > $file_name
    echo "1 0.${sval}0" >> $file_name
    
    full_sval=0.${sval}0
    rem_part=$(echo "scale=5; 1.00 - $full_sval" | bc)
    div_part=$(echo "scale=5; $rem_part / ($ovls - 1)" | bc)
    
    for (( i = 2 ; i <= ovls ; i++ ))
    do
      echo "$i 0$div_part" >> $file_name
    done
  done
done
