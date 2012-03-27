#!/bin/awk -f

BEGIN {
	numSynapseEntries = 0;
#	print line;
#	for (i = 0; i < ARGC; i++)
#		printf "%s ", ARGV[i]
}

(NR == 1) {
	for (i=1; i<NF; i++) {
		if ($i ~ /s[0-9]+/) {
			numSynapseEntries++;
		}
	}
}

(NR == line+1) {
	for (i=1; i<=numSynapseEntries; i++) {
		print i,$i
	}
}

#END { 
#	print numSynapseEntries; 
#}
