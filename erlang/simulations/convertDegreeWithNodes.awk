#!/usr/bin/awk -f
# USAGE: convertDegreeWithNodes.awk -v numNodes=1000000 degree1.txt > newDegree.txt
BEGIN {
	$OFS=" "
	arcsCounter = 0
} 

{
	numDegNodes = int($2*numNodes)
	arcsCounter += int($1) * numDegNodes
	if (numDegNodes != 0)
		print $1, numDegNodes;
}

END {
	if (arcsCounter % 2 != 0)
		print "ODD ARCS NUMBER! Add 1 node with the least odd degree (e.g. 1)" > "/dev/stderr"
}
