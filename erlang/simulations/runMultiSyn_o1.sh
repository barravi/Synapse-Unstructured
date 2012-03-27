#!/bin/sh
# USAGE: time runMultiSyn_o1.sh degreeFile outPrefix alpha alphaPost TTL number_of_generations counter_offset sampling_factor 2>&1 >> log_file
# EXAMPLE: time runMultiSyn_o1.sh degree1n1000000.txt ./d1_o1/o1_n1000000 0.0001 4 3 40 10 1 2>&1 >> lastLog

degreeFile=$1
outputPrefix=$2
alpha=$3
alphaPost=$4
TTL=$5
numGenerations=$6
counterOffset=$7
samplingFactor=$8

topoPost="_topo"
topoErlPost="_erl"
statPost="_stat"
logPost="_log"

#alpha=(0.0001 0.000001)
#TTLa1=(2 3 4)
#for t in "${TTL[@]}"

for i in $(seq $((1+$counterOffset)) $(($numGenerations+$counterOffset)))
do
  tempTopoFile=${outputPrefix}_${alphaPost}_${TTL}_${i}${topoPost}
  tempTopoErlFile=${tempTopoFile}${topoErlPost}
  tempLogFile=${outputPrefix}_${alphaPost}_${TTL}_${i}${logPost}
  tempStatFile=${outputPrefix}_${alphaPost}_${TTL}_${i}${statPost}

  echo "Generating random graph $i from degree file $degreeFile to file $tempTopoFile" &&
  time ./graph $degreeFile > $tempTopoFile 2>$tempLogFile &&
  echo "Converting graph $i to Erlang input $tempTopoErlFile" &&
  ./convertTopologyToErlang.awk $tempTopoFile > $tempTopoErlFile &&
  echo "Reading number of lines from $tempTopoFile" &&
  numNodes=`cat $tempTopoFile | wc -l` &&
  echo "Updating file $tempTopoErlFile with header $numNodes lines" &&
  sed -i '1 i '"$numNodes". $tempTopoErlFile &&
  echo "Ready to run simulation $i with alpha=$alpha (ref $alphaPost) and TTL $TTL!" &&
  echo "command: time ./runSynapseSimulation.erl $tempTopoErlFile $tempTopoStatFile $alpha $TTL 1.0 | tee $tempLogFile" &&
  time ./runSynapseSimulation.erl $tempTopoErlFile $tempStatFile $alpha $TTL 1.0 $samplingFactor 2>&1 > $tempLogFile
#  echo $tempTopoFile, $tempTopoErlFile, $tempLogFile, $tempStatFile
done

