#!/bin/bash

BIN=../../synopt
DEGREE_FILE=../../degree2.txt
RESULTS_DIR=../../results
PAR_RES_ALPHA=0.1
PAR_DX=1e-9
PAR_GRANULARITY=20
ROUTING_STRATEGY=3

echo "Degree file: $DEGREE_FILE, Overlays: 2 to 20, Resource alpha: $PAR_RES_ALPHA, Granularity: $PAR_GRANULARITY, Routing: $ROUTING_STRATEGY"

# Degree degree1.txt file, (2-20) overlays, TTL 2, alpha 0.1, 20 granularity
echo "Modeling for TTL: 2"
$BIN -d $DEGREE_FILE -o 1 -w ${RESULTS_DIR}/d2_o1_t2_a1e-1_3.txt -t 2 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 2 -w ${RESULTS_DIR}/d2_o2_t2_a1e-1_3.txt -t 2 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 3 -w ${RESULTS_DIR}/d2_o3_t2_a1e-1_3.txt -t 2 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 4 -w ${RESULTS_DIR}/d2_o4_t2_a1e-1_3.txt -t 2 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 5 -w ${RESULTS_DIR}/d2_o5_t2_a1e-1_3.txt -t 2 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 6 -w ${RESULTS_DIR}/d2_o6_t2_a1e-1_3.txt -t 2 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 7 -w ${RESULTS_DIR}/d2_o7_t2_a1e-1_3.txt -t 2 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 8 -w ${RESULTS_DIR}/d2_o8_t2_a1e-1_3.txt -t 2 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 9 -w ${RESULTS_DIR}/d2_o9_t2_a1e-1_3.txt -t 2 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 10 -w ${RESULTS_DIR}/d2_o10_t2_a1e-1_3.txt -t 2 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 11 -w ${RESULTS_DIR}/d2_o11_t2_a1e-1_3.txt -t 2 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 12 -w ${RESULTS_DIR}/d2_o12_t2_a1e-1_3.txt -t 2 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 13 -w ${RESULTS_DIR}/d2_o13_t2_a1e-1_3.txt -t 2 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 14 -w ${RESULTS_DIR}/d2_o14_t2_a1e-1_3.txt -t 2 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 15 -w ${RESULTS_DIR}/d2_o15_t2_a1e-1_3.txt -t 2 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 16 -w ${RESULTS_DIR}/d2_o16_t2_a1e-1_3.txt -t 2 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 17 -w ${RESULTS_DIR}/d2_o17_t2_a1e-1_3.txt -t 2 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 18 -w ${RESULTS_DIR}/d2_o18_t2_a1e-1_3.txt -t 2 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 19 -w ${RESULTS_DIR}/d2_o19_t2_a1e-1_3.txt -t 2 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 20 -w ${RESULTS_DIR}/d2_o20_t2_a1e-1_3.txt -t 2 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY

# Degree degree1.txt file, (2-20) overlays, TTL 3, alpha 0.1, 20 granularity
echo "Modeling for TTL: 3"
$BIN -d $DEGREE_FILE -o 1 -w ${RESULTS_DIR}/d2_o1_t3_a1e-1_3.txt -t 3 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 2 -w ${RESULTS_DIR}/d2_o2_t3_a1e-1_3.txt -t 3 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 3 -w ${RESULTS_DIR}/d2_o3_t3_a1e-1_3.txt -t 3 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 4 -w ${RESULTS_DIR}/d2_o4_t3_a1e-1_3.txt -t 3 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 5 -w ${RESULTS_DIR}/d2_o5_t3_a1e-1_3.txt -t 3 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 6 -w ${RESULTS_DIR}/d2_o6_t3_a1e-1_3.txt -t 3 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 7 -w ${RESULTS_DIR}/d2_o7_t3_a1e-1_3.txt -t 3 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 8 -w ${RESULTS_DIR}/d2_o8_t3_a1e-1_3.txt -t 3 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 9 -w ${RESULTS_DIR}/d2_o9_t3_a1e-1_3.txt -t 3 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 10 -w ${RESULTS_DIR}/d2_o10_t3_a1e-1_3.txt -t 3 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 11 -w ${RESULTS_DIR}/d2_o11_t3_a1e-1_3.txt -t 3 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 12 -w ${RESULTS_DIR}/d2_o12_t3_a1e-1_3.txt -t 3 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 13 -w ${RESULTS_DIR}/d2_o13_t3_a1e-1_3.txt -t 3 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 14 -w ${RESULTS_DIR}/d2_o14_t3_a1e-1_3.txt -t 3 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 15 -w ${RESULTS_DIR}/d2_o15_t3_a1e-1_3.txt -t 3 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 16 -w ${RESULTS_DIR}/d2_o16_t3_a1e-1_3.txt -t 3 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 17 -w ${RESULTS_DIR}/d2_o17_t3_a1e-1_3.txt -t 3 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 18 -w ${RESULTS_DIR}/d2_o18_t3_a1e-1_3.txt -t 3 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 19 -w ${RESULTS_DIR}/d2_o19_t3_a1e-1_3.txt -t 3 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 20 -w ${RESULTS_DIR}/d2_o20_t3_a1e-1_3.txt -t 3 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY

# Degree degree1.txt file, (2-20) overlays, TTL 4, alpha 0.1, 20 granularity
echo "Modeling for TTL: 4"
$BIN -d $DEGREE_FILE -o 1 -w ${RESULTS_DIR}/d2_o1_t4_a1e-1_3.txt -t 4 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 2 -w ${RESULTS_DIR}/d2_o2_t4_a1e-1_3.txt -t 4 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 3 -w ${RESULTS_DIR}/d2_o3_t4_a1e-1_3.txt -t 4 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 4 -w ${RESULTS_DIR}/d2_o4_t4_a1e-1_3.txt -t 4 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 5 -w ${RESULTS_DIR}/d2_o5_t4_a1e-1_3.txt -t 4 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 6 -w ${RESULTS_DIR}/d2_o6_t4_a1e-1_3.txt -t 4 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 7 -w ${RESULTS_DIR}/d2_o7_t4_a1e-1_3.txt -t 4 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 8 -w ${RESULTS_DIR}/d2_o8_t4_a1e-1_3.txt -t 4 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 9 -w ${RESULTS_DIR}/d2_o9_t4_a1e-1_3.txt -t 4 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 10 -w ${RESULTS_DIR}/d2_o10_t4_a1e-1_3.txt -t 4 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 11 -w ${RESULTS_DIR}/d2_o11_t4_a1e-1_3.txt -t 4 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 12 -w ${RESULTS_DIR}/d2_o12_t4_a1e-1_3.txt -t 4 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 13 -w ${RESULTS_DIR}/d2_o13_t4_a1e-1_3.txt -t 4 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 14 -w ${RESULTS_DIR}/d2_o14_t4_a1e-1_3.txt -t 4 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 15 -w ${RESULTS_DIR}/d2_o15_t4_a1e-1_3.txt -t 4 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 16 -w ${RESULTS_DIR}/d2_o16_t4_a1e-1_3.txt -t 4 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 17 -w ${RESULTS_DIR}/d2_o17_t4_a1e-1_3.txt -t 4 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 18 -w ${RESULTS_DIR}/d2_o18_t4_a1e-1_3.txt -t 4 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 19 -w ${RESULTS_DIR}/d2_o19_t4_a1e-1_3.txt -t 4 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 20 -w ${RESULTS_DIR}/d2_o20_t4_a1e-1_3.txt -t 4 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY

# Degree degree1.txt file, (2-20) overlays, TTL 5, alpha 0.1, 20 granularity
echo "Modeling for TTL: 5"
$BIN -d $DEGREE_FILE -o 1 -w ${RESULTS_DIR}/d2_o1_t5_a1e-1_3.txt -t 5 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 2 -w ${RESULTS_DIR}/d2_o2_t5_a1e-1_3.txt -t 5 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 3 -w ${RESULTS_DIR}/d2_o3_t5_a1e-1_3.txt -t 5 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 4 -w ${RESULTS_DIR}/d2_o4_t5_a1e-1_3.txt -t 5 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 5 -w ${RESULTS_DIR}/d2_o5_t5_a1e-1_3.txt -t 5 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 6 -w ${RESULTS_DIR}/d2_o6_t5_a1e-1_3.txt -t 5 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 7 -w ${RESULTS_DIR}/d2_o7_t5_a1e-1_3.txt -t 5 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 8 -w ${RESULTS_DIR}/d2_o8_t5_a1e-1_3.txt -t 5 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 9 -w ${RESULTS_DIR}/d2_o9_t5_a1e-1_3.txt -t 5 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 10 -w ${RESULTS_DIR}/d2_o10_t5_a1e-1_3.txt -t 5 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 11 -w ${RESULTS_DIR}/d2_o11_t5_a1e-1_3.txt -t 5 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 12 -w ${RESULTS_DIR}/d2_o12_t5_a1e-1_3.txt -t 5 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 13 -w ${RESULTS_DIR}/d2_o13_t5_a1e-1_3.txt -t 5 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 14 -w ${RESULTS_DIR}/d2_o14_t5_a1e-1_3.txt -t 5 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 15 -w ${RESULTS_DIR}/d2_o15_t5_a1e-1_3.txt -t 5 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 16 -w ${RESULTS_DIR}/d2_o16_t5_a1e-1_3.txt -t 5 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 17 -w ${RESULTS_DIR}/d2_o17_t5_a1e-1_3.txt -t 5 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 18 -w ${RESULTS_DIR}/d2_o18_t5_a1e-1_3.txt -t 5 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 19 -w ${RESULTS_DIR}/d2_o19_t5_a1e-1_3.txt -t 5 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 20 -w ${RESULTS_DIR}/d2_o20_t5_a1e-1_3.txt -t 5 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
