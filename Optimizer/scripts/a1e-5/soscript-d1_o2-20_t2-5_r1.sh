#!/bin/bash

BIN=../../synopt
DEGREE_FILE=../../degree1.txt
RESULTS_DIR=../../results
PAR_RES_ALPHA=0.00001
PAR_DX=1e-9
PAR_GRANULARITY=20
ROUTING_STRATEGY=1

echo "Degree file: $DEGREE_FILE, Overlays: 2 to 20, Resource alpha: $PAR_RES_ALPHA, Granularity: $PAR_GRANULARITY, Routing: $ROUTING_STRATEGY"

# Degree degree1.txt file, (2-20) overlays, TTL 2, alpha 0.00001, 20 granularity
echo "Modeling for TTL: 2"
$BIN -d $DEGREE_FILE -o 1 -w ${RESULTS_DIR}/d1_o1_t2_a1e-5_1.txt -t 2 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 2 -w ${RESULTS_DIR}/d1_o2_t2_a1e-5_1.txt -t 2 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 3 -w ${RESULTS_DIR}/d1_o3_t2_a1e-5_1.txt -t 2 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 4 -w ${RESULTS_DIR}/d1_o4_t2_a1e-5_1.txt -t 2 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 5 -w ${RESULTS_DIR}/d1_o5_t2_a1e-5_1.txt -t 2 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 6 -w ${RESULTS_DIR}/d1_o6_t2_a1e-5_1.txt -t 2 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 7 -w ${RESULTS_DIR}/d1_o7_t2_a1e-5_1.txt -t 2 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 8 -w ${RESULTS_DIR}/d1_o8_t2_a1e-5_1.txt -t 2 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 9 -w ${RESULTS_DIR}/d1_o9_t2_a1e-5_1.txt -t 2 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 10 -w ${RESULTS_DIR}/d1_o10_t2_a1e-5_1.txt -t 2 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY

# Degree degree1.txt file, (2-20) overlays, TTL 3, alpha 0.00001, 20 granularity
echo "Modeling for TTL: 3"
$BIN -d $DEGREE_FILE -o 1 -w ${RESULTS_DIR}/d1_o1_t3_a1e-5_1.txt -t 3 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 2 -w ${RESULTS_DIR}/d1_o2_t3_a1e-5_1.txt -t 3 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 3 -w ${RESULTS_DIR}/d1_o3_t3_a1e-5_1.txt -t 3 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 4 -w ${RESULTS_DIR}/d1_o4_t3_a1e-5_1.txt -t 3 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 5 -w ${RESULTS_DIR}/d1_o5_t3_a1e-5_1.txt -t 3 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 6 -w ${RESULTS_DIR}/d1_o6_t3_a1e-5_1.txt -t 3 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 7 -w ${RESULTS_DIR}/d1_o7_t3_a1e-5_1.txt -t 3 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 8 -w ${RESULTS_DIR}/d1_o8_t3_a1e-5_1.txt -t 3 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 9 -w ${RESULTS_DIR}/d1_o9_t3_a1e-5_1.txt -t 3 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 10 -w ${RESULTS_DIR}/d1_o10_t3_a1e-5_1.txt -t 3 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY

# Degree degree1.txt file, (2-20) overlays, TTL 4, alpha 0.00001, 20 granularity
echo "Modeling for TTL: 4"
$BIN -d $DEGREE_FILE -o 1 -w ${RESULTS_DIR}/d1_o1_t4_a1e-5_1.txt -t 4 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 2 -w ${RESULTS_DIR}/d1_o2_t4_a1e-5_1.txt -t 4 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 3 -w ${RESULTS_DIR}/d1_o3_t4_a1e-5_1.txt -t 4 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 4 -w ${RESULTS_DIR}/d1_o4_t4_a1e-5_1.txt -t 4 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 5 -w ${RESULTS_DIR}/d1_o5_t4_a1e-5_1.txt -t 4 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 6 -w ${RESULTS_DIR}/d1_o6_t4_a1e-5_1.txt -t 4 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 7 -w ${RESULTS_DIR}/d1_o7_t4_a1e-5_1.txt -t 4 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 8 -w ${RESULTS_DIR}/d1_o8_t4_a1e-5_1.txt -t 4 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 9 -w ${RESULTS_DIR}/d1_o9_t4_a1e-5_1.txt -t 4 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 10 -w ${RESULTS_DIR}/d1_o10_t4_a1e-5_1.txt -t 4 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY

# Degree degree1.txt file, (2-20) overlays, TTL 5, alpha 0.00001, 20 granularity
echo "Modeling for TTL: 5"
$BIN -d $DEGREE_FILE -o 1 -w ${RESULTS_DIR}/d1_o1_t5_a1e-5_1.txt -t 5 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 2 -w ${RESULTS_DIR}/d1_o2_t5_a1e-5_1.txt -t 5 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 3 -w ${RESULTS_DIR}/d1_o3_t5_a1e-5_1.txt -t 5 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 4 -w ${RESULTS_DIR}/d1_o4_t5_a1e-5_1.txt -t 5 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 5 -w ${RESULTS_DIR}/d1_o5_t5_a1e-5_1.txt -t 5 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 6 -w ${RESULTS_DIR}/d1_o6_t5_a1e-5_1.txt -t 5 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 7 -w ${RESULTS_DIR}/d1_o7_t5_a1e-5_1.txt -t 5 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 8 -w ${RESULTS_DIR}/d1_o8_t5_a1e-5_1.txt -t 5 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 9 -w ${RESULTS_DIR}/d1_o9_t5_a1e-5_1.txt -t 5 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
$BIN -d $DEGREE_FILE -o 10 -w ${RESULTS_DIR}/d1_o10_t5_a1e-5_1.txt -t 5 -a $PAR_RES_ALPHA -x $PAR_DX -g $PAR_GRANULARITY -r $ROUTING_STRATEGY
