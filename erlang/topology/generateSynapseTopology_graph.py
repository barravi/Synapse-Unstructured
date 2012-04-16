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
print("numNodes, numOvl",numNodes, numOvl)

# Check that we have enough nodes at least to have 1 node for each synapse degree
if (minGranularity > numNodes):
	parser.error("Too few nodes to generate a synapse topology. Need at least %d" % minGranularity)

synapseDegree = read_degree_file(options.synapseDegreeFile)
print("synapseDegree", synapseDegree)

# 1b Calcola numero di nodi sinapsi e di nodi normali
numNormalNodes = int((synapseDegree[1][0] * numNodes) + 0.5) # NB ARROTONDAMENTO
numSynapseNodes = numNodes - numNormalNodes
print("numNormalNodes, numSynapseNodes", numNormalNodes, numSynapseNodes)

# 1c (calcoliamo il surplus di nodi sinapsi, tanto per divertirci
numSynapseSurplus = 0
for deg, perc in synapseDegree.items():
	numSynapseSurplus += (numNodes * perc[0]) * (deg-1)
numSurplusPerOverlay = int((numSynapseSurplus / numOvl) + 0.5)
print("numSynapseSurplus, numSurplusPerOverlay", numSynapseSurplus, numSurplusPerOverlay)

# 3 GENERAZIONE CONFIGURAZIONI SINAPSI
#
# 3a) newSynDegree = synDegree-1 per tutti i degree
# 3b) numNodes = MCM(numOverlays, granularity)
# 3c) genero N = numSynapse//numOverlays volte delle topologie
# 3d) overlays d'appartenenza = nodo + vicini (per quello riduco di 1
# 3e) per l'ultimo tiro, se numSynRestanti < numOverlay, genero una topo piena, e prendo numSynrestanti
#    configurazioni a caso dalla lista.
# 3f) rinormalizzo topologie, link doppi li metto su altri overlay
# ---------------------------------------------------------------------

# 3a crea nuovo synDegree con grado -1, percentuale rinormalizzata senza contare i gradi 1  E NUOVO NUMERO DI NODI
# moltiplicato per numOvl invece che numNodes
newSynapseDegree = dict()
for deg, percArray in synapseDegree.items():
	if (deg > 1): 
		newSynapseDegree[deg-1] = percArray[:]
	#del(newSynapseDegree[deg-1][1])
print("newSynapseDegree",newSynapseDegree)

# 3b We compute the least common multiple between the number of overlay
# and the minimum number of nodes required to generate a synapse topology
# that respects the granularity
numNodesSynTopo = lcm(numOvl, minGranularity)
print("numNodesSynTopo", numNodesSynTopo)

# 3c normalizziamo il degree per avere le percentuali SENZA i gradi 1
normFactor = 1 - synapseDegree[1][0] #FIRST INDEX IS LABEL (1 = DEGREE 1)
for pa in newSynapseDegree.itervalues():
	pa[0] = pa[0] / normFactor
	pa.append(int((pa[0] * numNodesSynTopo) + 0.5)) #NB: ARROTONDAMENTO	
print("newSynapsesDegree: ", newSynapseDegree) #TEST OK!

# 3d calcola il numSynTopo di volte che van generate le topologie di sinapsi
numSynTopo = numSynapseNodes // numNodesSynTopo
remSynTopo = numSynapseNodes % numNodesSynTopo
print("numSynTopo, remSynTopo: ", numSynTopo, remSynTopo) #TEST OK!

# 3d scrivi file degree sinapsi con numero nodi
write_degree_file(newSynapseDegree, synDegreeFile)
# 3d genera numSynTopo topologie IN PARALLELO
numProc = options.numProcesses
tempSynapseTopology = []

ps = multiprocessing.Pool(numProc)
res = ps.map(generate_topology_list_parallel, [(synDegreeFile, True, i, numSynTopo) for i in xrange(numSynTopo)])
ps.close()
ps.join()

for l in res:
	tempSynapseTopology.extend(l)

#for i in xrange(0, numSynTopo):
	#if (i % 100 == 0): print("Synapse topology:", i, "/", numSynTopo)
	#tempTopo = generate_topology_list(synDegreeFile, True)
	#tempSynapseTopology.extend(tempTopo)

print("1. tempSynapseTopology, lenght: ", len(tempSynapseTopology)) #TEST OK!

# 3e se remSyn > 0, generane ancora una e tira remSyn entries
if remSynTopo > 0:
	print("Synapse topology: last")
	tempTopo = generate_topology_list(synDegreeFile, True)
#	tempValues = list(tempTopo.values())
	for i in xrange(0, remSynTopo):
		rndIdx = random.randint(0, len(tempTopo))
		tempSynapseTopology.append(tempTopo[rndIdx])
		del(tempTopo[rndIdx])
print("2. tempSynapseTopology, lenght: ", len(tempSynapseTopology)) #TEST OK!

# 3f dobbiamo normalizzare da Nodo E {1..numNodesSynTopo} --> Nodo E {1..numOvl}
normFactor = numNodesSynTopo / numOvl
synapseTopology = []
for topo in tempSynapseTopology:
	synapseTopology.append(normalize_synapse_topology(topo, normFactor, numOvl))
print("3. synapseTopology, lenght: ", len(synapseTopology)) #TEST OK!
#print(synapseTopology) #TEST OK

# 4 fai statistica del numero di vicini e verifica che siamo ancora buoni come degree
# Conta l'effettivo numero di nodi virtuali synapse per ogni overlay (altrimenti non va)
synStats = dict()
ovlVirtualNodesMap = dict.fromkeys(xrange(0, numOvl), 0)
for topo in synapseTopology:
	l = len(topo)
	if (synStats.has_key(l)):
		synStats[l][0] += 1
	else:
		synStats[l] = [1]
	for ovl in topo:
		ovlVirtualNodesMap[ovl] += 1

total = numNormalNodes
for v in synStats.values():
	total += v[0]
for v in synStats.values():
	v.append(float(float(v[0])/total))
print("synStats", synStats)
print("ovlVirtualNodesMap", ovlVirtualNodesMap)

effectiveSynapseSurplus=0
for _,v in ovlVirtualNodesMap.items():
	effectiveSynapseSurplus+=v
print("effectiveSynapseSurplus", effectiveSynapseSurplus)

# END SYNAPSE TOPOLOGY GENERATION

#sys.exit(0)

# OVERLAY TOPOLOGIES GENERATION
# 5 leggi il file degree overlay normalizzato
if (options.overlayDegreeFile == ""):
	parser.error("Overlay degree file unspecified.")

overlayDegree = read_degree_file(options.overlayDegreeFile)

# 6 conta gli effettivi nodi virtuali per ogni topologia
#   Genera le topologie
numNormalNodesPerOverlay = int((numNormalNodes / numOvl) + 0.5) # NB ARROTONDAMENTO
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

#topologies = prepare_and_generate_overlay_topology( \
#		(ovlVirtualNodesMap[4] + numNormalNodesPerOverlay, tempOverlayDegree[4], \
#		ovlDegreeFile + str(4), 4, numOvl))
po = multiprocessing.Pool(numProc)
topologies = po.map(prepare_and_generate_overlay_topology, \
		[(ovlVirtualNodesMap[i] + numNormalNodesPerOverlay, tempOverlayDegree[i], \
		ovlDegreeFile + str(i), i, numOvl) for i in xrange(numOvl)])
po.close()
po.join()

for i in xrange(0, len(topologies)):
	print("%d, len(topo)" % i, len(topologies[i]))

# TOPOLOGY MERGE PARALLEL
# 0) create disjoint sets to feed the workers
# 1) For each connected overlay, we get 1 random node in the overlay
# 2) Add the Overlay-Node:Synapse in a dictionary
# 3) Write 2 files, 1 with the synapses and 1 with the overlay topologies
# (the validator will have to be adapted
# DOWNSAMPLING LISTS
#def downsample_lists_parallel((synapseTopology, topologies, idWorker, numWorkers)): #idWorker = {1..numWorkers}
	#print("Downsampling, id:", idWorker, "/", numWorkers)
	#dsSynapseTopology = []
	#dsTopologies = []
	#counter = 0

	#synSize = len(synapseTopology)
	#for synIdx in xrange(synSize):
		#if (synIdx % 1000 == 0): 
			#print("DS:", idWorker,"/", numWorkers, ": Syn", synIdx)
		#if ((synIdx != 0) and (synIdx % numWorkers) == 0):
			#dsSynapseTopology.append(synapseTopology[synIdx - idWorker])
		#if (synIdx == synSize - 1) and (synIdx % numWorkers >= idWorker):
			#dsSynapseTopology.append(synapseTopology[synIdx - idWorker])
	
	#for tIdx in xrange(len(topologies)):
		#topo = topologies[tIdx]
		#tempTopology = []
		#tSize = len(topo)
		#for nIdx in xrange(tSize):
			#if (nIdx % 1000 == 0): 
				#print("DS:", idWorker,"/", numWorkers, ": Ovl", tIdx, "Node", nIdx)
			#if (nIdx != 0) and (nIdx % numWorkers) == 0:
				#tempTopology.append(topo[nIdx - idWorker])
			#if (nIdx == tSize - 1) and (nIdx % numWorkers >= idWorker):
				#tempTopology.append(topo[nIdx - idWorker])

		#dsTopologies.append(tempTopology)
	
	#return (dsSynapseTopology, dsTopologies)

#pd = multiprocessing.Pool(numProc)
#disjointSets = pd.map(downsample_lists, \
		#[(synapseTopology, topologies, w, numProc) 
		#for w in xrange(numProc)])
#pd.close()
#pd.join()

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

#print(len(synapseTopology))
#totNodes = [0 for i in xrange(numOvl)]
#for workerTopo in disjointSets[1]:
	#print("WORKER: len", len(workerTopo))
	#ovlC = 0
	#for topo in workerTopo:
		#print("OVL", ovlC)
		#print(len(topo))
		#totNodes[ovlC] += len(topo)
		#ovlC += 1

#for tot in range(len(totNodes)):
	#print(totNodes[tot], len(topologies[tot]))

#print(totDSynSet)

#sys.exit(0)

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
			nList[ovlId] = nNeigh
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

#for pRes in finalNodeLists:
	#print("Worker")
	#for pOvl in pRes[2]:
		#print("Ovl")
		#print(len(pOvl))
#sys.exit(0)


def print_header(fOutTopo, (pSynMap, _, pTopo), totalNodes=0):
	if totalNodes > 0:
		fOutTopo.write("%d.\n" % totalNodes)
	totalEntries = len(pSynMap)
	for o in pTopo:
		totalEntries += len(o)
	totalOvl = len(pTopo)
	fOutTopo.write("%d.\n" % totalEntries)
	# Write on the second line the number of overlays
	# fOutTopo.write("%d.\n" % totalOvl) # number of overlays

def print_syn_nodes(fOutTopo, (pSynMap, _, __)):
	for nId, (_, topo) in pSynMap.iteritems():
		#(_, nList) = nodeStruct
		fOutTopo.write("{%d, " % nId) # write the node index
		if len(topo) > 0:
			fOutTopo.write("[") # open the square brackets if there are neightbors sets
			for it, (ovlId, nSet) in enumerate(topo.iteritems()): # for each neightbors set
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
	
def print_remaining_nodes(fOutTopo, (_, __, pTopo), nodeCounterOffset):
	nodeCounter = nodeCounterOffset
	for ovlId in xrange(len(pTopo)):
		for (_, nSet) in pTopo[ovlId]:
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

def print_syn_node_map(fOutSynMap, (_, pSynNodeMap, __)):
	for (ovlId, nId), synId in pSynNodeMap.iteritems(): # for each entry in the map
		fOutSynMap.write("[{%d, %d}, %d].\n" % (ovlId, nId, synId))

def print_erlang_topology_parallel((totalNodesSubset, index, nodeOffset, totalNodes, outputFileName)):
	print("Writing files...")

	#format: {nId, [{oId, [n1, n2, ...]}, {oId2, [...]}, ...]}.
	fOutTopo = open(outputFileName + topologyPost + "_" + str(index), 'w') 
	# format: [{ovlId, nId}, synId].	
	fOutSynMap = open(outputFileName + synapseNodeMapPost + "_" + str(index), 'w')

	print_header(fOutTopo, totalNodesSubset, totalNodes if index==0 else 0)
	print_syn_nodes(fOutTopo, totalNodesSubset)
	print_remaining_nodes(fOutTopo, totalNodesSubset, nodeOffset)
	print_syn_node_map(fOutSynMap, totalNodesSubset)

	fOutTopo.close()
	fOutSynMap.close()

### PARALLEL FILE OUTPUT - USELESS (loads too much the system I/O)
# compute the remaining nodes offset
#nodeOffset = [0]
#noCounter = 0
#for wG in finalNodeLists:
	#for ovl in wG[2]: #pTopo
		#noCounter += len(ovl)
	#nodeOffset.append(noCounter)

## compute the overall number of entries
#totalEntries = 0
#for wG in finalNodeLists:
	#totalEntries += len(wG[0])
	#totalOvl = len(wG[2])
	#for o in wG[2]:
		#totalEntries += len(o)

#pf = multiprocessing.pool.ThreadPool(numProc)
#pf.map(print_erlang_topology_parallel, \
	#[(finalNodeLists[i], i, nodeOffset[i], totalEntries, options.outputErlangFile) \
	#for i in xrange(numProc)])
#pf.close()
#pf.join()

#sys.exit(0)

def print_erlang_topology(finalNodeLists, outputFileName):
	print("Writing files...")

	fOutTopo = open(outputFileName + topologyPost, 'w') #format: {nId, [{oId, [n1, n2, ...]}, {oId2, [...]}, ...]}.
	fOutSynMap = open(outputFileName + synapseNodeMapPost, 'w')
	
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
	
	nodeCounter = 0
	for wGroup in finalNodeLists: # for each worker result
		# Write the synapse topology on file
		pSynMap = wGroup[0] # get the partial list of synapse nodes
		nodeCounter += len(pSynMap)
		for nId, (_, topo) in pSynMap.iteritems():
			#(_, nList) = nodeStruct
			fOutTopo.write("{%d, " % nId) # write the node index
			if len(topo) > 0:
				fOutTopo.write("[") # open the square brackets if there are neightbors sets
				for it, (ovlId, nSet) in enumerate(topo.iteritems()): # for each neightbors set
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

	fOutSynMap.write("%d.\n" % totalOvl)
	for wGroup in finalNodeLists: # format: [{ovlId, nId}, synId].
		# Write the synapseNodeMap in the second file
		pSynNodeMap = wGroup[1] # partial synapse node map
		for (ovlId, nId), synId in pSynNodeMap.iteritems(): # for each entry in the map
			fOutSynMap.write("{{%d, %d}, %d}.\n" % (ovlId, nId, synId))
	
	for wGroup in finalNodeLists:
		# Write the remaining nodes
		pTopo = wGroup[2] # get the partial topology dictionary
		for ovlId in xrange(len(pTopo)):
			for (nId, nSet) in pTopo[ovlId]:
				# write the entry in the synapseNodeMap
				fOutSynMap.write("{{%d, %d}, %d}.\n" % (ovlId, nId, nodeCounter))
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
	fOutSynMap.close()

print_erlang_topology(finalNodeLists, options.outputErlangFile)

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

