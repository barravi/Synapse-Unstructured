#!/bin/sh

# 2012-04-08, Riccardo Loti (loti@di.unito.it)
#
# This script generates for <routing> routing types, <sval> s1 values (ex.: 8 means 0.80 value for s1) and <ovls> number of overlays
# the optimizer result for the specific configuration, expressed on the conf files (see syngen.sh).

SYNOPT_BIN=../synopt
DEG_FILE=../degree1.txt
TTL_PAR=3
ALPHA_PAR=0.0001
SYN_DIR=../confs
RES_DIR=../results/unlim/single_res

for routing in 2
do
  for sval in 8 6 4 2
  do
    for ovls in 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
    do
      $SYNOPT_BIN -d $DEG_FILE -o $ovls -c ${SYN_DIR}/syn${sval}o${ovls}.txt -t $TTL_PAR -a $ALPHA_PAR -r $routing -w ${RES_DIR}/s0${sval}_d1o${ovls}t${TTL_PAR}a1e-4r${routing}.txt
    done
   done
done