#!/usr/bin/python
# 1) creo la topologia di sinapsi synapseO200 DISTRIBUZIONE include anche nodi di grado 1!!!
# 2) Numero totale di nodi = SUM(Vicini(N)) (nodo non sinapsi = 1 solo vicino)
# 3) IGNORO il node number, considero solo i vicini
# 4) Ogni vicino nella topologia = 1 overlay a cui appartiene
# 5) VNum % NumOv per sapere a quale overlay
# 6) Se 2 overlay uguali nello stesso vicinato, ne tiro a caso un altro tra i NumOv
# 7) Una volta fatta la topologia di overlay, pesco nodi e mergio vicinati

# DATA STRUCTURES:
# topologies = {ovlID1:{nodeID1:[nID1,nID2...], nodeID2:[...] ...}, olvID2:{...}, ...}
# 
# synapseTopology = [[ovlID1, ovlID2...],[...], ...]
#
# finalTopology = {nodeID1:{ovlID1:[nID1, nID2], ovlID2:[...]}, nodeID2:{...}, ...}

from __future__ import print_function
import random
import os
import tokenize
from optparse import OptionParser
import sys
import copy
import multiprocessing
import threading

graphCommand = "./graph"
degreeCommand = "./distrib"

synDegreeFile = "tmpSynDegree"
ovlDegreeFile = "tmpOvlDegree"

topologyPost = "_topo"
synapseNodeMapPost = "_nm"

usage = "Usage: %prog --fs=synapse_degree_file --fo=overlay_degree_file --no=num_overlays --nn=num_nodes --oe erlangOutputFile [-g -v]"

parser = OptionParser(usage)

parser.add_option("--go", action="store_true", dest="generateOverlay", default=False, help="Generate the overlay degree instead of reading them form file")
parser.add_option("--gs", action="store_true", dest="generateSynapse", default=False, help="Generate the synapse degree instead of reading them form file")
parser.add_option("-v", action="store_true", dest="verbose", default=False, help="VERBOSE: show status messages while generating")
parser.add_option("--fo", type="string", dest="overlayDegreeFile", default="", help="Specify the overlay degree file name")
parser.add_option("--fs", type="string", dest="synapseDegreeFile", default="", help="Specify the synapse degree file name")
parser.add_option("--nn", type="int", dest="numNodes", default=1000, help="Number of physical nodes in the system")
parser.add_option("--no", type="int", dest="numOverlay", default=10, help="Number of overlays in the system")
parser.add_option("--gr", type="float", dest="synGranularity", default="0.05", help="Granularity of synapse degree file (default 0.05).")
parser.add_option("--oe", type="string", dest="outputErlangFile", default="", help="Filename where to store the topology (Erlang format)")
parser.add_option("-j", type="int", dest="numProcesses", default="1", help="Number of parallel generator processes")

#parser.add_option("-v", action="store_true", dest="verbose", default=false, help="VERBOSE: show status messages while generating")

(options, args) = parser.parse_args()


def generate_topology(degree_file, isSynapse=False, debug=False) :
	graphOutput = os.popen(graphCommand + " " + degree_file)
	topology = dict()
	key = 0
	value = []
	for t in tokenize.generate_tokens(graphOutput.readline):
		if t[2][1] == 0 and t[0] != tokenize.ENDMARKER: #first token, node number --> key
			key = int(t[1])
			if isSynapse: # for synapse topologies the node number IS a neightbor, a connected overlay
				value.append(int(t[1]))
		elif t[0] == tokenize.NEWLINE: # last token, newline --> store set
			topology[key] = value
			value = []
			key = 0
		elif t[0] == tokenize.NUMBER: # neightbor
			value.append(int(t[1]))

	if debug: 
		print(topology)
	
	graphOutput.close()
	return topology

def generate_topology_list(degree_file, isSynapse=False, debug=False, resultList=None) :
	graphOutput = os.popen(graphCommand + " " + degree_file)
	topology = []
	key = 0
	value = []
	for t in tokenize.generate_tokens(graphOutput.readline):
		if t[2][1] == 0 and t[0] != tokenize.ENDMARKER: #first token, node number --> key
			key = int(t[1])
			if isSynapse: # for synapse topologies the node number IS a neightbor, a connected overlay
				value.append(int(t[1]))
		elif t[0] == tokenize.NEWLINE: # last token, newline --> store set
			if (isSynapse):
				topology.append(value)
			else:
				topology.append((key, value))
			value = []
			key = 0
		elif t[0] == tokenize.NUMBER: # neightbor
			value.append(int(t[1]))
#	topology = list(token for token in tokenize.generate_tokens(graphOutput.readline))

#	topology = list(token[1] for token in tokenize.generate_tokens(graphOutput.readline) if token[0] == tokenize.NUMBER)
	if debug: 
		print(topology)
	
	graphOutput.close()

	if (resultList):
		resultList.extend(topology)
	else:
		return topology

def generate_topology_list_parallel((degree_file, isSynapse, numIteration, totalIterations)) :
	initMsg = "Synapse" if isSynapse else "Overlay"
	if (numIteration % 100 == 0): print(initMsg + " topology", numIteration, "/", totalIterations)

	graphOutput = os.popen(graphCommand + " " + degree_file)
	topology = []
	key = 0
	value = []
	for t in tokenize.generate_tokens(graphOutput.readline):
		if t[2][1] == 0 and t[0] != tokenize.ENDMARKER: #first token, node number --> key
			key = int(t[1])
			if isSynapse: # for synapse topologies the node number IS a neightbor, a connected overlay
				value.append(int(t[1]))
		elif t[0] == tokenize.NEWLINE: # last token, newline --> store set
			if (isSynapse):
				topology.append(value)
			else:
				topology.append((key, value))
			value = []
			key = 0
		elif t[0] == tokenize.NUMBER: # neightbor
			value.append(int(t[1]))
#	topology = list(token for token in tokenize.generate_tokens(graphOutput.readline))

#	topology = list(token[1] for token in tokenize.generate_tokens(graphOutput.readline) if token[0] == tokenize.NUMBER)
	
	graphOutput.close()

	return topology

def read_degree_file(inFileName):
	fIn = open(inFileName, 'r') #format: degree nPerc

	key = 0
	value = 0
	degree = dict()
	for t in tokenize.generate_tokens(fIn.readline):
		if t[2][1] == 0 and t[0] != tokenize.ENDMARKER:
			key = int(t[1])
		elif t[0] == tokenize.NEWLINE: # last token, newline --> store set
			degree[key] = value
		elif t[0] == tokenize.NUMBER: # neightbor
			value = [float(t[1])]

	fIn.close()
	
	return degree

def write_degree_file(degreeArray, outFileName):
	fOut = open(outFileName, 'w')
	for deg, perc in degreeArray.items():
#		print("%d %d" % (deg, int(round(perc * numNodes)))) 
		fOut.write("%d %d\n" % (deg, perc[1]))
	
	fOut.close()

#oldKey, neightbors = 
def pick_random_node(topologies, ovlId): #pick a random node in the overlay
	#print(ovlId)
	nodeSet = topologies[ovlId]
	rndIdx = int(random.uniform(0, len(nodeSet)))
	#print(len(nodeSet))
	rndKey = nodeSet.items()[rndIdx][0]
	rndNeight  = nodeSet.items()[rndIdx][1]

	del nodeSet[rndKey]

	return (rndKey, rndNeight)

def pick_random_node_list(topologies, ovlId): #pick a random node in the overlay
	#print(ovlId)
	nodeList = topologies[ovlId]
	try:
		rndIdx = random.randint(0, len(nodeList)-1)
	except ValueError:
		print(ovlId, nodeList)

	#print(len(nodeSet))
	rndKeyNeight = nodeList[rndIdx]

	del nodeList[rndIdx]

	return rndKeyNeight

def pick_random_node_list_topo(topology):
	rndIdx = random.randint(0, len(topology)-1)
	#print(len(nodeSet))
	rndKeyNeight = topology[rndIdx]

	del nodeList[rndIdx]

	return rndKeyNeight
	
def pick_first_node(topologies, ovlId): #pick the first node in an overlay
	nodeSet = topologies[ovlId]
	rndKey = nodeSet.items()[0][0]
	rndNeight  = nodeSet.items()[0][1]

	del nodeSet[rndKey]

	return (rndKey, rndNeight)


def update_topologies(topologies, ovlId, oldKey, nodeIndex): # update all the entries in the remainig ovl
	nodeSet = topologies[ovlId]
	#print(nodeSet)
	for _, neightborhood in nodeSet.items():
		for i in xrange(0, len(neightborhood)):
			if neightborhood[i] == oldKey:
				neightborhood[i] = nodeIndex

def update_synapse_topology(finalTopology, ovlId, oldKey, nodeIndex): # update all the entries in the topology
	for node, nSet in finalTopology.items():
		#print(finalTopology)
		for k, nodes in nSet.items():
			if k == ovlId:
				for i in xrange(0, len(nodes)):
					if nodes[i] == oldKey:
						nodes[i] = nodeIndex


def normalize_synapse_topology(topo, normFactor, numOvl):
	tempNorm = dict()
	for ovl in topo:
		newOvl = int(ovl // normFactor)
		if newOvl not in tempNorm:
			tempNorm[newOvl] = True
		else: # if key is already there, I have to pick another radom overlay
			allOverlays = dict.fromkeys(xrange(0, numOvl)) #create an overlay list
			for k in tempNorm.keys(): #remove all elements already in the neightborhood
				if k in allOverlays: 
					del allOverlays[k]
			allOverlayList = allOverlays.keys() 
			#pick one random of the remaining
			replaceOvl = allOverlayList[random.randint(0,len(allOverlayList)-1)]
			tempNorm[replaceOvl] = True
	return tempNorm.keys()			


# MATH FUNCTIONS TO COMPUTE LOWEST COMMON MULTIPLE AND
# GREATEST COMMON DENOMINATOR
def lcm(a, b):
	return ( a * b ) / gcd(a, b)

def gcd(a, b):
	a = int(a)
	b = int(b)
	while b:
		a,b = b,a%b
	return a

################################
# SCRIPT STARTS HERE          
################################

# 0 calcola il numero di nodi minimo per la synapse topology
# onde poter rispettare la granularita' della distribuzione
# (e.g. granularity=0.5 --> almeno 20 nodi)
minGranularity = int((1/options.synGranularity) + 0.5)
print("minGranularity", minGranularity)

if (options.outputErlangFile == ""):
	parser.error("Output ERLANG file unspecified.")

# 1 leggi file con degree sinapsi normalizzato
if (options.synapseDegreeFile == ""):
	parser.error("Synapse degree file unspecified.")

numNodes = options.numNodes
numOvl = options.numOverlay
#numNodesOvl = int((numNodes/float(numOvl)) + 0.5)
numProc = options.numProcesses

print("numNodes, numOvl",numNodes, numOvl)

def generateSynapseTopology(numNodes, numOvl):
	# Check that we have enough nodes at least to have 1 node for each synapse degree
	if (minGranularity > numNodes/numOvl):
		parser.error("Too few nodes to generate a synapse topology. Need at least %d" % minGranularity)

	synapseDegree = read_degree_file(options.synapseDegreeFile) #FORMAT: {1:[0.03]}
	print("1. synapseDegree", synapseDegree)

	# 1b Calcola numero di nodi sinapsi e di nodi normali
	numNormalNodes = int((synapseDegree[1][0] * numNodes) + 0.5) # NB ARROTONDAMENTO
	numSynapseNodes = numNodes - numNormalNodes
	print("numNormalNodes, numSynapseNodes", numNormalNodes, numSynapseNodes)

	# 1c (calcoliamo il surplus di nodi sinapsi, tanto per divertirci
	numSynapseSurplus = 0
	for deg, perc in synapseDegree.items():
		numSynapseSurplus += int((numNodes * perc[0]) * (deg-1))
	numSynapseSurplusOvl = int((numSynapseSurplus / numOvl) + 0.5) #NB ARROTONDAMENTO
	print("numSynapseSurplus, numSynapseSurplusOvl", numSynapseSurplus, numSynapseSurplusOvl)

	## NUOVA GENERAZIONE DELLE CONFIGURAZIONI SDI SINAPSI
	#numNormalNodesOvl = int((numNormalNodes / float(numOvl)) + 0.5) #NB ARROTONDAMENTO
	# numero di nodi TOTALI per overlay (incluse istanze virtuali di nodi sinapse)
	#numRealNodesOvl = int((numNodes / float(numOvl)) + 0.5) #NB ARROTONDAMENTO
	numTotalNodes = numNodes + numSynapseSurplus
	numTotalNodesOvl = int((numTotalNodes/float(numOvl))+0.5) #NB ARROTONDAMENTO

	numNodesOvl = int((numNodes / float(numOvl)) + 0.5) #NB ARROTONDAMENTO

	print("numTotalNodes, numTotalNodesOvl, numNodesOvl",numTotalNodes, numTotalNodesOvl, numNodesOvl)

	for deg in synapseDegree.keys():
		# Numero di nodi synapse di grado deg IN TOTALE
		synapseDegree[deg].append(int((synapseDegree[deg][0] * numNodes) + 0.5)) #NB ARROTONDAMENTO
		# Numero di nodi synapse di grado deg PER OVERLAY
		synapseDegree[deg].append(int((synapseDegree[deg][0] * numNodesOvl) + 0.5)) #NB ARROTONDAMENTO

	print("2. synapseDegree", synapseDegree)

	# Inizialilzziamo il numero di nodi da assegnare per ogni overlay
	remainingNodes = [{} for o in xrange(numOvl)]
	for deg in synapseDegree.keys():
		for remNodesOvl in remainingNodes:
			remNodesOvl[deg] = synapseDegree[deg][2]

	# synapseDegree format: {deg:[perc, numNodes, numNodesOvl]}
	# overlays = index [0...numOvl-1]
	synapseTopology = [] #FORMAT: [[O1, O2...],[...],...]

	# 1. La parte facile: aggiungiamo N topologie con tutti gli overlay per quanti nodi con deg=numOvl
	if synapseDegree.has_key(numOvl):
		for nT in xrange(synapseDegree[numOvl][2]):
			synapseTopology.append([o for o in xrange(numOvl)])
			for remNodesOvl in remainingNodes:
				remNodesOvl[numOvl] -= 1

	print("1. synapseTopology len", len(synapseTopology))
	#print(synapseTopology)
	print("remainingNodes", remainingNodes)

	# trova il massimo degree delle sinapsi
	maxSynapseDegree = 0
	for deg in synapseDegree.keys():
		if deg > maxSynapseDegree:
			maxSynapseDegree = deg
	# Se il massimo degree 'e il numero di overlay, saltiamo che l' abbiamo gia' fatto
	maxSynapseDegree += 0 if maxSynapseDegree == numOvl else 1

	print("maxSynapseDegree", maxSynapseDegree)

	def updateRemainingNodes(remainingNodes, combination):
		degree = len(combination)
		for c in combination:
			remainingNodes[c][degree] -= 1
	#		print(c, degree, remainingNodes[c][degree])

	def isConfigurationAvailable(values, indexes, remainingNodes):
		degree = len(indexes)
		hasZeroNodes = False
	#	print("CHECK")
		for i in xrange(degree):
	#		print("\t", remainingNodes[values[i][indexes[i]]][degree])
			hasZeroNodes = hasZeroNodes or (remainingNodes[values[i][indexes[i]]][degree] == 0)
		return not hasZeroNodes


	# creiamo delle combinazioni di ovl per ogni degree
	for deg in xrange(2, maxSynapseDegree):

	# genero sufficienti conbinazioni affinche', se mi servono X nodi di grado deg per ogni overlay
	# l'overlay X compaia almeno X volte nelle combinazioni
	#	numConfig = int((numOvl/float(deg)) + 0.5) * synapseDegree[deg][2] #NB ARROTONDAMENTO
	# PER ORA NO, GENERO COMBINAZIONI FINCHE' HO NODI DISPONIBILI
	#	numConfig = int(round(numOvl/float(deg))) * synapseDegree[deg][2]
	#	print("numConfig", deg, numConfig)
		print("Combinations for degree", deg)
	# Ora contiamo tutte le possibili combinazioni di deg cifre tutte diverse l' una dall'altra
		# Inizializziamo un array di indici per ogni cifra
		indexes = [0 for i in xrange(deg)]
	#	for c in xrange(numConfig):
		hasConfig = True

		while (hasConfig):
		# Inizializziamo una tabella con tutti i valori possibili per ogni cifra
			values = [[o for o in xrange(numOvl)] for d in xrange(deg)]
		# Rimuoviamo da ogni cifra i valori scelti nelle cifre precedenti
			remValues = []
			for i in xrange(len(indexes)):
				for r in remValues:
					values[i].remove(r)
				remValues.append(values[i][indexes[i]])
		# Generiamo una configurazione prendendo cifre con i valori puntati dagli indici
			synapseConfig = []
			for i in xrange(len(indexes)):
				synapseConfig.append(values[i][indexes[i]])
		# Se la configurazione e' disponibile, la aggiungo e aggiorno i contatori
			if (isConfigurationAvailable(values, indexes, remainingNodes)):
				synapseTopology.append(synapseConfig)
				updateRemainingNodes(remainingNodes, synapseConfig)
				firstUnavailableConfig = []
			else:
		# Se non e' disponibile, posso 
		# 1) segnarmela se e' la prima non disponibile
		# 3) interrompere tutto se la configurazione e' uguale a quella marcata 
		#    (vuol dire che ho fatto il giro completo e non ho piu' configurazioni)
				if firstUnavailableConfig == []:
					firstUnavailableConfig = synapseConfig
				elif firstUnavailableConfig == synapseConfig:
					hasConfig = False

		# Incrementiamo gli indici, se modulo incrementiamo il precedente
			carry=1
			for i in reversed(xrange(len(indexes))):
				indexes[i] += carry
				if indexes[i] % len(values[i]) == 0:
					indexes[i] = 0
					carry=1
				else:
					carry=0

	#print(synapseTopology)
	print("2. synapseTopology len", len(synapseTopology))

	# TEST: calcoliamo il grado Sk all' inverso, partendo dalle configurazioni
	reverseSynapseDegreeOvl = [{deg:[0] for deg in xrange(1, len(synapseDegree)+1)} for o in xrange(numOvl)]
	totals = [synapseDegree[1][2] for i in xrange(numOvl)]

	for sT in synapseTopology:
		deg = len(sT)
		for ovl in sT:
	#		print(deg, ovl)
			reverseSynapseDegreeOvl[ovl][deg][0] += 1
			totals[ovl] += 1

	for osd in xrange(len(reverseSynapseDegreeOvl)):
		for d in xrange(len(reverseSynapseDegreeOvl[osd])):
	#		reverseSynapseDegreeOvl[osd][d+1].append(reverseSynapseDegreeOvl[osd][d+1][0]/float(totals[osd]))
			reverseSynapseDegreeOvl[osd][d+1].append(reverseSynapseDegreeOvl[osd][d+1][0]/float(numNodesOvl))

	for ovlD in xrange(len(reverseSynapseDegreeOvl)):
		print("OVERLAY", ovlD)
		for k, v in reverseSynapseDegreeOvl[ovlD].items():
			print("\t", k, v)

	#print(reverseSynapseDegreeOvl)
	print("remainingNodes", remainingNodes)

	return synapseTopology, remainingNodes

# Calculate the first topology
synapseTopology, remainingNodes = generateSynapseTopology(numNodes, numOvl)

def computeRevNumNodes(synapseTopology, remainingNodes):
	# Set every remaining node as a non-synapse
	# Calculate new total
	revNumNodes = 0
	for remNodesOvl in remainingNodes:
		for deg in remNodesOvl.keys():
			revNumNodes += remNodesOvl[deg]
			if deg > 1:
				remNodesOvl[1] += remNodesOvl[deg]
				remNodesOvl[deg] = 0

	print("2. remainingNodes", remainingNodes)
	# Add the synapse nodes to the count
	revNumNodes += len(synapseTopology)
	print("revNumNodes", revNumNodes)

	return revNumNodes

revNumNodes = computeRevNumNodes(synapseTopology, remainingNodes)

# Increase the number of nodes
newNumNodes = int((pow(numNodes,2) / float(revNumNodes)) + 0.5) #NB ARROTONDAMENTO
print("numNodes, revNumNodes, newNumNodes", numNodes, revNumNodes, newNumNodes)

# Recalculate topology
synapseTopology, remainingNodes = generateSynapseTopology(newNumNodes, numOvl)

revNumNodes = computeRevNumNodes(synapseTopology, remainingNodes)
print("2. newNumNodes, revNumNodes", newNumNodes, revNumNodes)

numNodesOvl = [0 for i in xrange(numOvl)]
for ri in xrange(len(remainingNodes)):
	for rid in remainingNodes[ri].keys():
		numNodesOvl[ri] += remainingNodes[ri][rid]
for sConfig in synapseTopology:
	for ovl in sConfig:
		numNodesOvl[ovl] += 1

print("numNodesOvl", numNodesOvl)

# OVERLAY TOPOLOGIES GENERATION
# 5 leggi il file degree overlay normalizzato
if (options.overlayDegreeFile == ""):
	parser.error("Overlay degree file unspecified.")

overlayDegree = read_degree_file(options.overlayDegreeFile)

# 6 conta gli effettivi nodi virtuali per ogni topologia
#   Genera le topologie
#numNormalNodesPerOverlay = int((numNormalNodes / numOvl) + 0.5) # NB ARROTONDAMENTO
#numOvlNodes = dict()
#topologies = dict()
tempOverlayDegree = list()
for i in xrange(numOvl):
	tempOverlayDegree.append(copy.deepcopy(overlayDegree))

def prepare_and_generate_overlay_topology((numTotOvlNodes, tempOverlayDegree, fileName, ovl, numOvl)):
	print("Topology generation n. ", ovl, "/", numOvl-1)
# 7 Calcola il numero di nodi per ogni overlay a partire dalle istanze synapse 

# 8 converti con numOvlVirtualNodes
	for deg, value in tempOverlayDegree.items():
		tempOverlayDegree[deg].append(int((value[0] * numTotOvlNodes)+0.5)) #NB: ARROTONDAMENTO

# 8b FIX: se il numero di arachi e' dispari, aggiungi un nodo di grado 1 per renderlo pari
#    FIX2: rimuovi le entries con 0 nodi
	totArcs=0
	for deg, value in tempOverlayDegree.items():
		totArcs += deg * value[1]
		if (value[1] == 0): # togli gradi senza nodi
			try:
				del tempOverlayDegree[deg]
			except KeyError:
				print("NEEEEEH!", ovl)

	if totArcs % 2 == 1: # if odd number of arcs
		tempOverlayDegree[1][1]+=1 #add one node to make it even

# 9 scrivi file degree overlay
	write_degree_file(tempOverlayDegree, fileName)

# 10 genera numOverlay topologie, stora in array
	return generate_topology_list(fileName)

po = multiprocessing.Pool(numProc)
topologies = po.map(prepare_and_generate_overlay_topology, \
		[(numNodesOvl[i], tempOverlayDegree[i], \
		ovlDegreeFile + str(i), i, numOvl) for i in xrange(numOvl)])
po.close()
po.join()

for i in xrange(0, len(topologies)):
	print("%d, len(topo)" % i, len(topologies[i]))

def downsample_lists(synapseTopology, topologies, numWorkers): #idWorker = {1..numWorkers}
	dsSynapseTopology = [[] for i in xrange(numWorkers)]
	dsTopologies = [[] for i in xrange(numWorkers)]
	dsOffSets = [0]
	nodeOverlayDiff = [[0 for o in xrange(len(topologies))] for w in xrange(numWorkers)]

	synSize = len(synapseTopology)
	for synIdx in xrange(synSize):
		if (synIdx % 1000 == 0): 
			print("DS: Syn", synIdx, "/", synSize)
		if ((synIdx != 0) and (synIdx % numWorkers  == 0)):
			for idWorker in xrange(1, numWorkers+1):
				dsSynapseTopology[idWorker-1].append(synapseTopology[synIdx - idWorker])
		if (synIdx == synSize - 1):
			for idWorker in xrange((synIdx % numWorkers)+1):
				dsSynapseTopology[idWorker].append(synapseTopology[synIdx - idWorker])
	for tIdx in xrange(len(topologies)):
		for workerTopo in dsTopologies:
			workerTopo.append([])
		
		tSize = len(topologies[tIdx])
		for nIdx in xrange(tSize):
			if (nIdx % 1000 == 0): 
				print("DS: Ovl", tIdx, "Node", nIdx, "/", tSize)
			if ((nIdx != 0) and (nIdx % numWorkers) == 0):
				for idWorker in xrange(numWorkers):
					dsTopologies[idWorker][tIdx].append(topologies[tIdx][nIdx - idWorker - 1])
					nodeOverlayDiff[idWorker][tIdx] += 1
			if (nIdx == tSize - 1):
				for idWorker in xrange((nIdx % numWorkers) + 1):
					dsTopologies[idWorker][tIdx].append(topologies[tIdx][nIdx - idWorker])
					nodeOverlayDiff[idWorker][tIdx] += 1

	# subtract every node required by a synapse
	for wIdx in xrange(len(dsSynapseTopology)):
		for synTopo in dsSynapseTopology[wIdx]:
			for oIdx in synTopo:
				nodeOverlayDiff[wIdx][oIdx] -= 1
	print("Node differences BEFORE", nodeOverlayDiff)
	# loop in all the overlays, for every negative value, look away for the additoinal nodes
	#allGood = False
	#while not allGood:
	#	allGood = True
	for wIdx in xrange(len(nodeOverlayDiff)):
		for oIdx in xrange(len(nodeOverlayDiff[wIdx])):
			if nodeOverlayDiff[wIdx][oIdx] < 0:
	#			allGood = False
				seek_and_swap(nodeOverlayDiff, dsTopologies, wIdx, oIdx)
	print("Node differences AFTER", nodeOverlayDiff)
# compute the offset for each subset
	partial = 0
	for pSyn in xrange(len(dsSynapseTopology)-1):
		partial += len(dsSynapseTopology[pSyn])
		dsOffSets.append(partial)

	return (dsSynapseTopology, dsTopologies, dsOffSets)

def seek_and_swap(nodeOverlayDiff, dsTopologies, wIdx, oIdx):
	needed = 0-nodeOverlayDiff[wIdx][oIdx] # amount of needed nodes from the overlay (negative if needed)
	
	nSize = len(nodeOverlayDiff)
	for sIdx in xrange(1, nSize):
		actualIdx = (wIdx + sIdx) % nSize

		if (needed > 0):
			if (nodeOverlayDiff[actualIdx][oIdx] < 0):
				available = 0 # no available nodes
			elif (nodeOverlayDiff[actualIdx][oIdx] >= needed):
				available = needed #enough nodes
			else:
				available = nodeOverlayDiff[actualIdx][oIdx]; #not enough but some
		
			for i in xrange(available):
				newNode = pick_random_node_list(dsTopologies[actualIdx], oIdx)
				nodeOverlayDiff[actualIdx][oIdx] -= 1
				dsTopologies[wIdx][oIdx].append(newNode)
				nodeOverlayDiff[wIdx][oIdx] += 1
				needed -= 1
	
	if (needed > 0): print("DIOFA!")

disjointSets = downsample_lists(synapseTopology, topologies, numProc)


def merge_synapses_disjoints((partialSynTopo, partialTopologies, nodeIdOffset, synTot)):
	print("Mergin synapse part", nodeIdOffset)
	synNodeMap = {}
	synapseMap = {}

	pSynSize = len(partialSynTopo)
#Merge the synapses
	for synIdx in xrange(pSynSize):
		synTopo = partialSynTopo[synIdx]
		nList = {}
		nodeId = synIdx + nodeIdOffset
		for ovlId in synTopo:
			(nId, nNeigh) = pick_random_node_list(partialTopologies, ovlId)
			nList[ovlId] = (nId, nNeigh) # andrebbe aggiunto qui nId
			synNodeMap[(ovlId, nId)] = nodeId
		synNode = (nodeId, nList)
		synapseMap[nodeId] = synNode

	return (synapseMap, synNodeMap, partialTopologies)

synTot = len(synapseTopology)
pm = multiprocessing.pool.ThreadPool(numProc)
finalNodeLists = pm.map(merge_synapses_disjoints, \
		[(disjointSets[0][i], disjointSets[1][i], disjointSets[2][i], synTot) \
		for i in xrange(numProc)])
pm.close()
pm.join()

nlC = 0
for l in finalNodeLists:
	print("FinalNodeList len: ", nlC, len(l[0]))
	nlC += 1

#print(len(synapseTopology))

## UPDATE SELETTIVO
# 1) prendere 1 nodo con vicinati da synapseTopology
# 	2) per ogni overlay connesso
# 		3) per ogni nodo nel vicinato dell'overlay
#				4) recuperare il nodo corrispondente dalla synapseNodeMap -> entry nella SynTopology
#				   altrimenti recuperarlo dai nodi restanti -> entry nella partialTopologies
#				5) aggiornare la sua entry per il nodo sinapsi con il nuovo ID
def selective_update(finalNodeLists):
	synapseMaps = []
	synNodeMaps = []
	partialTopo = []
	for (synMap, synNodeMap, pTopo) in finalNodeLists:
		synapseMaps.append(synMap)
		synNodeMaps.append(synNodeMap)
		partialTopo.append(pTopo)

	print("Start selective update")
	displayCounter = 0
	for sMap in synapseMaps:
		for _, (synNodeId, onList) in sMap.items():
			displayCounter += 1	
			if (displayCounter % 1000 == 0):
				print ("Updating node:", displayCounter)
			
			for ovlId, (oldId, nList) in onList.items():
				for nNode in nList:
					updateMeSynId = synNodeMap_multiseek(synNodeMaps, (ovlId, nNode))	
					if (updateMeSynId != None):
						# vado nella synapseMap
						update_synMaps(synapseMaps, updateMeSynId, ovlId, oldId, synNodeId)
					else:
						# vado nella pTopologies
						update_pTopologies(partialTopo, ovlId, nNode, oldId, synNodeId)

def selective_update_merge(finalNodeLists):
	print("Merging lists")
	synapseMap = {}
	synNodeMap = [{} for o in xrange(numOvl)]
	partialTopo = [{} for o in xrange(numOvl)]
	finalSynapseMap = {}
	finalPartialTopo = [{} for o in xrange(numOvl)]

	counter=0
	for (synMap, sNodeMap, pTopo) in finalNodeLists:
		counter+=1
		print("Batch ", counter)
		for synId, (rSynId, nList) in synMap.items():
			synapseMap[synId] = nList
			finalSynapseMap[synId] = copy.deepcopy(nList)
		for (ovlId, nId), synId in sNodeMap.items():
			synNodeMap[ovlId][nId] = synId
		for pOvl in xrange(len(pTopo)):
			for (nodeId, nList) in pTopo[pOvl]:
				partialTopo[pOvl][nodeId] = nList
				finalPartialTopo[pOvl][nodeId] = copy.deepcopy(nList)

	print("Start selective update")
	nodeCounter = 0
	totalNodes = len(synapseMap)
	for oTopo in partialTopo:
		totalNodes += len(oTopo)

	for synNodeId, onList in synapseMap.items():
		nodeCounter += 1	
		if (nodeCounter % 1000 == 0):
			print ("Updating node:", nodeCounter, "/", totalNodes)
			
		for ovlId, (oldId, nList) in onList.items():
			for nNode in nList:
			#	if (synNodeMap[ovlId].has_key(nNode)):
					# aggiorno la synNodeMap
				try:
					newNNode = synNodeMap[ovlId][nNode]
					for i in xrange(len(finalSynapseMap[newNNode][ovlId][1])): #lista vicini
						if finalSynapseMap[newNNode][ovlId][1][i] == oldId:
							finalSynapseMap[newNNode][ovlId][1][i] = synNodeId							
				except KeyError:
				#	print("nNode not in synMap", nNode)
					# vado nella pTopologies
					try:
						for i in xrange(len(finalPartialTopo[ovlId][nNode])):
							if (finalPartialTopo[ovlId][nNode][i] == oldId): 
								finalPartialTopo[ovlId][nNode][i] = synNodeId
					except KeyError:
						print("KeyError", ovlId, nNode)
	
	nodeCounter -= 1

	for ovlId in xrange(len(partialTopo)):
		onList = partialTopo[ovlId]
		for oldId, nList in onList.items():
			nodeCounter += 1	
			if (nodeCounter % 1000 == 0):
				print ("Updating node:", nodeCounter, "/", totalNodes)
			
			# aggiungi nodo semplice alla final map
			# Aggiungi reference a nuovo index nella synapseNodeMap
			finalSynapseMap[nodeCounter] = {ovlId:(oldId, finalPartialTopo[ovlId][oldId])}
			synNodeMap[ovlId][oldId] = nodeCounter
			# Aggiorna il vicinato
			for nNode in nList:
			#	if (synNodeMap[ovlId].has_key(nNode)):
					# aggiorno la synNodeMap
				try:
					newNNode = synNodeMap[ovlId][nNode]
					for i in xrange(len(finalSynapseMap[newNNode][ovlId][1])): #lista vicini
						if finalSynapseMap[newNNode][ovlId][1][i] == oldId:
							finalSynapseMap[newNNode][ovlId][1][i] = nodeCounter					
				except KeyError:
					#print("nNode not in synMap", nNode)
					# vado nella pTopologies
					try:
						for i in xrange(len(finalPartialTopo[ovlId][nNode])):
							if (finalPartialTopo[ovlId][nNode][i] == oldId): 
								finalPartialTopo[ovlId][nNode][i] = nodeCounter
					except KeyError:
						print("KeyError", ovlId, nNode)
	
	return finalSynapseMap, synNodeMap, finalPartialTopo

def update_pTopologies(partialTopo, ovlId, nNode, oldId, synNodeId):
	for pT in partialTopo:
		ovlNodes = pT[ovlId]
		for (nId, nList) in ovlNodes:
			if nId == nNode:
				for nIdx in xrange(len(nList)):
					if nList[nIdx] == oldId:
						nList[nIdx] = synNodeId
				return

def update_synMaps(synapseMaps, updateSynId, ovlId, oldId, synNodeId):
	for sMap in synapseMaps:
		if sMap.has_key(updateSynId):
			(uSId, noList) = sMap[updateSynId]
			(oId, nList) = noList[ovlId]
			for nIdx in xrange(len(nList)):
				if nList[nIdx] == oldId:
					nList[nIdx] = synNodeId
			return

def synNodeMap_multiseek(synNodeMaps, key):
	for synNodeMap in synNodeMaps:
		if synNodeMap.has_key(key):
			return synNodeMap[key]
	return None

# TOO LONG TO RUN
# selective_update(finalNodeLists)
finalSynMap, finalSynNodeMap, finalPartialTopo = selective_update_merge(finalNodeLists)

## RICALCOLIAMO IL SYNAPSE DEGREE PER TUTTI GLI OVERLAY, PER ESSERE SICURI
## CHE LA DISTRIBUZIONE SIA BILANCIATA IN TUTTI GLI OVERLAY

###############################################################
###############################################################
def reverse_syn_degree(finalNnodeLists, numOvl):
	synapseMaps = []
	synNodeMaps = []
	partialTopo = []
	for (synMap, synNodeMap, pTopo) in finalNodeLists:
		synapseMaps.append(synMap)
		synNodeMaps.append(synNodeMap)
		partialTopo.append(pTopo)

	revSynDegree = [{1:0} for i in xrange(numOvl)]
	totals = [0 for i in xrange(len(partialTopo[0]))]

# Prendiamo i gradi 1 dalla partialTopo, che contiene i nodi non sinapsati
	for wTopo in partialTopo:
		for o in xrange(len(wTopo)):
			revSynDegree[o][1] += len(wTopo[o])
			totals[o] += len(wTopo[o])

	for wSynMap in synapseMaps:
		for (sId, noList) in wSynMap.values():
			sDegree = len(noList) #a quanti overlay e' collegata
			for (oId, nList) in noList.items():
				revSynDegree[oId][sDegree] = revSynDegree[oId][sDegree]+1 if revSynDegree[oId].has_key(sDegree) else 1
				totals[oId] += 1
	
	ovlTotals = [0 for i in xrange(len(totals))]
	degTotals = [0 for i in xrange(len(revSynDegree[0]))]
	
	for odI in xrange(len(revSynDegree)):
	#	print(ovlDeg)
		for d, n in revSynDegree[odI].items():
			ovlTotals[odI] += n
			#print(d, len(degTotals))
			degTotals[d-1] += n/float(d)
	
	print("revSynDegree with nodes", revSynDegree)
	print("totals", totals)
	print("ovlTotals", ovlTotals)
	print("degTotals", degTotals)

	for o in xrange(len(revSynDegree)):
		for k in revSynDegree[o].keys():
			revSynDegree[o][k] /= float(totals[o])
	
	return revSynDegree

revDeg = reverse_syn_degree(finalNodeLists, numOvl)

print("revSynDegree normalized", revDeg)

def print_erlang_topology_merge(finalSynMap, finalPartialTopo, outputFileName):
	print("Writing files...")

	fOutTopo = open(outputFileName, 'w') #format: {nId, [{oId, [n1, n2, ...]}, {oId2, [...]}, ...]}.
	
	# Write the first line with the total number of phisical nodes (synapses + remaining nodes)
	totalEntries = len(finalSynMap)
	totalOvl = len(finalPartialTopo)
#	for o in finalPartialTopo:
#		totalEntries += len(o)
	print("Total entries", totalEntries)
	print("Total overlays", totalOvl)
	fOutTopo.write("%d.\n" % totalEntries)
	# Write on the second line the number of overlays
	# fOutTopo.write("%d.\n" % totalOvl) # number of overlays
	
	# Write the synapse topology on file
	for nId, topo in finalSynMap.iteritems():
		fOutTopo.write("{%d, " % (int(nId)+1)) # write the node index
		if len(topo) > 0:
			fOutTopo.write("[") # open the square brackets if there are neightbors sets
			for it, (ovlId, (oldId, nSet)) in enumerate(topo.iteritems()): # for each neightbors set
				fOutTopo.write("{%d, [" % ovlId) #Write a tuple {ovlId, [neightbors list]}
				for i in xrange(0, len(nSet)):
					fOutTopo.write("%d" % (int(nSet[i])+1))
					if i < len(nSet)-1: # if it's not the last one, we add a comma
						fOutTopo.write(",")
				fOutTopo.write("]}") # Close list, close tuple
				if it < len(topo)-1: # add a comma if not the last one
					fOutTopo.write(",")
			fOutTopo.write("]")
		fOutTopo.write("}.\n")

	nodeCounter = len(finalSynMap)+1

	## Write the remaining nodes
	#for ovlId in xrange(len(finalPartialTopo)):
		#for nId, nSet in finalPartialTopo[ovlId].iteritems():
			## write the entry in the topology file
			#fOutTopo.write("{%d, " % nodeCounter) # write the node index
			##if len(topo) > 0:
			#fOutTopo.write("[") # open the square brackets if there are neightbors sets
			##for it, (ovlId, nSet) in enumerate(topo.iteritems()): # for each neightbors set
			#fOutTopo.write("{%d, [" % ovlId) #Write a tuple {ovlId, [neightbors list]}
			#for i in xrange(0, len(nSet)):
				#fOutTopo.write("%d" % nSet[i])
				#if i < len(nSet)-1: # if it's not the last one, we add a comma
					#fOutTopo.write(",")
			#fOutTopo.write("]}") # Close list, close tuple
			#fOutTopo.write("]")
			#fOutTopo.write("}.\n")
			#nodeCounter += 1

	fOutTopo.close()

def print_erlang_topology(finalNodeLists, outputFileName):
	print("Writing files...")

	fOutTopo = open(outputFileName + topologyPost, 'w') #format: {nId, [{oId, [n1, n2, ...]}, {oId2, [...]}, ...]}.
	fOutSynMap = [open(outputFileName + synapseNodeMapPost + str(i), 'w') for i in xrange(numOvl)]
	
	# Write the first line with the total number of phisical nodes (synapses + remaining nodes)
	totalEntries = 0
	for wG in finalNodeLists:
		totalEntries += len(wG[0])
		totalOvl = len(wG[2])
		for o in wG[2]:
			totalEntries += len(o)
	print("Total entries", totalEntries)
	print("Total overlays", totalOvl)
	fOutTopo.write("%d.\n" % totalEntries)
	# Write on the second line the number of overlays
	# fOutTopo.write("%d.\n" % totalOvl) # number of overlays
	
	nodeCounter = 1
	for wGroup in finalNodeLists: # for each worker result
		# Write the synapse topology on file
		pSynMap = wGroup[0] # get the partial list of synapse nodes
		nodeCounter += len(pSynMap)
		for nId, (_, topo) in pSynMap.iteritems():
			#(_, nList) = nodeStruct
			fOutTopo.write("{%d, " % nId) # write the node index
			if len(topo) > 0:
				fOutTopo.write("[") # open the square brackets if there are neightbors sets
				for it, (ovlId, (oldId, nSet)) in enumerate(topo.iteritems()): # for each neightbors set
					fOutTopo.write("{%d, [" % ovlId) #Write a tuple {ovlId, [neightbors list]}
					for i in xrange(0, len(nSet)):
						fOutTopo.write("%d" % nSet[i])
						if i < len(nSet)-1: # if it's not the last one, we add a comma
							fOutTopo.write(",")
					fOutTopo.write("]}") # Close list, close tuple
					if it < len(topo)-1: # add a comma if not the last one
						fOutTopo.write(",")
				fOutTopo.write("]")
			fOutTopo.write("}.\n")

#	fOutSynMap.write("%d.\n" % totalOvl)
	for wGroup in finalNodeLists: # format: [{ovlId, nId}, synId].
		# Write the synapseNodeMap in the second file
		pSynNodeMap = wGroup[1] # partial synapse node map
		for (ovlId, nId), synId in pSynNodeMap.iteritems(): # for each entry in the map
			fOutSynMap[ovlId].write("{%d, %d}.\n" % (nId, synId))
	
	for wGroup in finalNodeLists:
		# Write the remaining nodes
		pTopo = wGroup[2] # get the partial topology dictionary
		for ovlId in xrange(len(pTopo)):
			for (nId, nSet) in pTopo[ovlId]:
				# write the entry in the synapseNodeMap
				fOutSynMap[ovlId].write("{%d, %d}.\n" % (nId, nodeCounter))
				# write the entry in the topology file
				fOutTopo.write("{%d, " % nodeCounter) # write the node index
				#if len(topo) > 0:
				fOutTopo.write("[") # open the square brackets if there are neightbors sets
				#for it, (ovlId, nSet) in enumerate(topo.iteritems()): # for each neightbors set
				fOutTopo.write("{%d, [" % ovlId) #Write a tuple {ovlId, [neightbors list]}
				for i in xrange(0, len(nSet)):
					fOutTopo.write("%d" % nSet[i])
					if i < len(nSet)-1: # if it's not the last one, we add a comma
						fOutTopo.write(",")
				fOutTopo.write("]}") # Close list, close tuple
				#if it < len(topo)-1: # add a comma if not the last one
				#	fOutTopo.write(",")
				fOutTopo.write("]")
				fOutTopo.write("}.\n")
				nodeCounter += 1

	fOutTopo.close()
	for f in fOutSynMap:
		f.close()

#print_erlang_topology(finalNodeLists, options.outputErlangFile)
print_erlang_topology_merge(finalSynMap, finalPartialTopo, options.outputErlangFile)

print("DONE!")

#def print_erlang_topology(finalTopology, outputFileName):
	#fOut = open(outputFileName, 'w') #format: {nId, [{oId, [n1, n2, ...]}, {oId2, [...]}, ...]}.

	#fOut.write("%d.\n" % len(finalTopology))
	#for nId, topo in finalTopology.iteritems():
		#fOut.write("{%d, " % nId) # write the node index
		#if len(topo) > 0:
			#fOut.write("[") # open the square brackets if there are neightbors sets
		#for it, (ovlId, nSet) in enumerate(topo.iteritems()): # for each neightbors set
			#fOut.write("{%d, [" % ovlId) #Write a tuple {ovlId, [neightbors list]}
			#for i in xrange(0, len(nSet)):
				#fOut.write("%d" % nSet[i])
				#if i < len(nSet)-1: # if it's not the last one, we add a comma
					#fOut.write(",")
			#fOut.write("]}") # Close list, close tuple
			#if it < len(topo)-1: # add a comma if not the last one
				#fOut.write(",")
		#if len(topo) > 0:
			#fOut.write("]")
		#fOut.write("}.\n")

