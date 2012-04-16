#!/usr/bin/awk -f
# USAGE: for f in $(ls -d1 d1_o1/o1_n1000000_4_2_*_stat); do ./extractStatData.awk $f >> stats_4_2_erl; done

BEGIN {
#	FS=""
	RS=","
}
(NR==14){
	m = $1
}
(NR==18){
#	sub(/[ \t]+/, "");
	phit = $1
}
END {
	print "{"phit", "m"}." 
}
