#!/usr/bin/python

import re
from math import sqrt
from optparse import OptionParser

reMString = "\{m,(?P<m>\d+\.\d+)\}"
mRegex = re.compile(reMString)

rePHitString1 = "\{pHit,\[(\{.+?\},?)+\]\}"
rePHitString2 = "\{(?P<alpha>\d+),(?P<pHit>\d+\.\d+)\}"

pHitRegex1 = re.compile(rePHitString1, re.DOTALL)
pHitRegex2 = re.compile(rePHitString2, re.DOTALL)

usage = "Usage: %prog list_of_stat_files"
parser = OptionParser(usage)
(options, args) = parser.parse_args()
if (len(args) == 0):
	parser.error("No stat file to parse")

pHit = {}
m = []

for fileName in args:
	print "# Processing ", fileName

	f = open(fileName, 'r')
	statData = f.read()

	res = mRegex.search(statData)
	m.append(float(res.group('m')))

#	print m

	res = pHitRegex1.search(statData)
	if (res != None):
#		print res.group(1)
		pHitList = res.group(1)
#		res = pHitRegex2.search(pHitList)

		for data in pHitRegex2.finditer(pHitList):
			alpha = int(data.group('alpha'))
			if not pHit.has_key(alpha):
				pHit[alpha] = []
			pHit[alpha].append(float(data.group('pHit')))
		
#		print pHit
N = len(m)
# Calcolo media
AvgM = reduce(lambda acc, new: acc + (new/N), m, .0)
AvgPHit = {key:0 for key in pHit.keys()}
#print "Average M", AvgM

for alpha in AvgPHit.keys():
	AvgPHit[alpha] = reduce(lambda acc, new: acc + (new/N), pHit[alpha], .0)

#print "Average pHit", AvgPHit

# Calcolo varianza
Np = N-1
VarM = reduce(lambda acc, new: acc + (AvgM - new)**2, m, .0)

VarPHit = {key:0 for key in pHit.keys()}
for alpha in VarPHit.keys():
	VarPHit[alpha] = reduce(lambda acc, new: acc + (AvgPHit[alpha] - new)**2, pHit[alpha], .0)

#print VarM
#print VarPHit

# Calcolo deviazione standard
StdDevM = sqrt(VarM)
StdDevPHit = {key:0 for key in pHit.keys()}
for alpha in StdDevPHit.keys():
	StdDevPHit[alpha] = sqrt(VarPHit[alpha])

#print StdDevM, StdDevPHit	

# Calcolo Errore medio standard / CI 68%
SN = sqrt(N)
SEMM = StdDevM / SN

SEMPHit = {key:0 for key in pHit.keys()}
for alpha in SEMPHit.keys():
	SEMPHit[alpha] = StdDevPHit[alpha] / SN

#print SEMM, SEMPHit

# Calcolo CI 95 %
Z = 1.96
CI95M = SEMM * Z
CI95PHit = {key:0 for key in pHit.keys()}
for alpha in CI95PHit.keys():
	CI95PHit[alpha] = SEMPHit[alpha] * Z

# PRINT
print "STATS:"
print "M:\tAvg:", AvgM
print "\tCI 68%:", SEMM
print "\tCI 95%:", CI95M
print
print "PHIT:"
for k, val in AvgPHit.items():
	print "Alpha:", k
	print "\tAvg:", AvgPHit[k]
	print "\tCI 68%:", SEMPHit[k]
	print "\tCI 95%:", CI95PHit[k]
