#!/bin/sh
# USAGE:   time runMultiSyn.sh degreeFile synapseFile outPrefix num_nodes num_ovl alpha alphaPost TTL number_of_generations counter_offset num_processes sampling_factor 2>&2 >> logFile
# EXAMPLE: time runMultiSyn.sh degree1.txt ./d1_o5_t2_a1e-4/d1_o5_t2_a1e-4_l2_synTopo ./d1_o5_t2_a1e-4/d1_o5_t2_n1000000 1000000 5 0.0001 4 2 20 0 8 2 >> lastLog

degreeFile=$1
synapseFile=$2
outputPrefix=$3
numNodes=$4
numOvl=$5
alpha=$6
alphaPost=$7
TTL=$8
numGenerations=$9
counterOffset=${10}
numProc=${11}
samplingFactor=${12}

topoPost="_topo"
topoErlPost="_erl"
statPost="_stat"
logPost="_log"

#alpha=(0.0001 0.000001)
#TTLa1=(2 3 4)
#for t in "${TTL[@]}"

echo "$0 $*"

for i in $(seq $((1+$counterOffset)) $(($numGenerations+$counterOffset)))
do
  tempTopoFile=${outputPrefix}_${numNodes}_${alphaPost}_${TTL}_${i}${topoPost}
  tempTopoErlFile=${tempTopoFile}${topoErlPost}
  tempLogFile=${outputPrefix}_${numNodes}_${alphaPost}_${TTL}_${i}${logPost}
  tempStatFile=${outputPrefix}_${numNodes}_${alphaPost}_${TTL}_${i}${statPost}
  
  if [ -f $tempLogFile ]
  then
    rm $tempLogFile
  fi

  echo "$(date +%Y%m%d:%H%M%S) Command: $0 $*" > $tempLogFile
  
  if [ -f $tempTopoErlFile ]
  then
    echo "$(date +%Y%m%d:%H%M%S) $tempTopoErlFile exists."
  else
    echo "$(date +%Y%m%d:%H%M%S) Generating random graph $i from Pk file $degreeFile and Sk file $synapseFile to file $tempTopoErlFile" &&
    ../topology/generateSynapseTopology.py --fs=$synapseFile --fo=$degreeFile --no=$numOvl --nn=$numNodes --oe=$tempTopoErlFile -j $numProc >> $tempLogFile
  fi
  if [ -f $tempStatFile ]
  then
    echo "$(date +%Y%m%d:%H%M%S) $tempStatFile exists. Moving on to the next simulation"
  else
    echo "$(date +%Y%m%d:%H%M%S) Ready to run simulation $i with alpha=$alpha (ref $alphaPost) and TTL $TTL!" &&
    echo "command: time ./runSynapseSimulation.erl $tempTopoErlFile $tempTopoStatFile $alpha $TTL 1.0 $samplingFactor" &&
    time ./runSynapseSimulation.erl $tempTopoErlFile $tempStatFile $alpha $TTL 1.0 $samplingFactor 2>&1 >> $tempLogFile
  fi
#  echo $tempTopoFile, $tempTopoErlFile, $tempLogFile, $tempStatFile
done

