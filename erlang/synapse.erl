				% TODO:
% 1 - l'inizializzazione della lista, farla a blocchi di 10000/100000 nodi
%     che fa piu' in fretta FATTO VERIFICA OK.
% 1b- usare un dict per storare la lista di processi (piu' veloce in lettura e per tanti dati).
%     FATTO VERIFICA OK.
% 2 - una volta inizializzati tutti, mandare a tutti un messaggio con il vicinato
%     (le sinapsi ricevono + set di vicini, no distinzione tra overlay)
%     FATTO VERIFICA OK.
% 2b- Mandare messaggio a nodi perche' si registrino (forse non serve).
%     FATTO VERIFICA OK.
% 3 - ricevere risposta per vicinato messaggio e settare nodo come pronto
%     FATTO VERIFICA OK.
% 4 - fare il seed di chiavi a caso OK.
% 5 - fare ricerca delle chiavi inserite partendo da nodo a caso:
%     - 1 processo manda richieste OK
%     - 1 processo riceve risposte e logga statistsiche OK
%     - coordinator riceve parziali, fa merge statistiche  OK
% 6 - Scrivere statistiche su file DA FARE!!
%     AGGIUNGERE ALLE STATISTICHE ANCHE IL NUMERO TOTALE DI MESSAGGIO INVIATI
%     (quelli che continuano a cercare dopo che la chiave e' stata trovata)!
%
% 7 - Leggi topologia da file OK
% 8 - Synapse sceglie vicini usando dado con probabilita' i/k (k = numero overlay) OK!
% 9 - Synapse sceglie vicini usando metodo con MAX-MESSAGES. DA FARE
% 10- Implementa con high order functions OK
%     (passa la funzione di probabilita', parametrizza in fase di compilazione) OK

% 1.5 Gb di memoria per 1.000.000 di nodi semplici (erl +P 2000000)
% 5> DNNV = synapse:insertValue(0.01, synapse:assignNeightbors(synapse:initNodeList(10000))).
% 5> Res = synapse:searchCoordinatorVT(5, 10, 3, 3, 1, DNNV).

% c(synapse, {d, debug}).
% NTree = synapse:initNodeListFromFile("./simulations/d1_o10_t3_a1e-4/topology_l2_topo").
% NTree = synapse:initNodeListFromFile("./simulations/topology_o1_n10000_erl").
% NH = synapse:initSynapseNodeDictionaryParallel("./simulations/d1_o10_t3_a1e-4/topology_l2_nm", 2, NTree).
% SynNodeDict = synapse:initSynapseNodeDictionary("./simulations/d1_o10_t3_a1e-4/topology_l2_nm", NTree).
% DN = synapse:initTopologyFromFileSNDict("./simulations/d1_o10_t3_a1e-4/topology_l2_topo", NH, NTree).
% DN = synapse:initTopologyFromFile("./simulations/topology_o1_n10000_erl", NTree).
% [N1|_] = dict:fetch(1, DN).
% DNK = synapse:insertValue(0.001, NTree).
% [NT|_] = dict:fetch(400, DNK).
% synapse:searchValueMan(NT).
% Stats = synapse:searchCoordinatorVT(1, 5, DNK, "testOut.erl").
% Stats = synapse:searchCoordinatorVT(5,3,1,2,0,2,DNK,"test").
-module(synapse).
-export([pingNode/1, boingNode/2, synapseNode/1, initNodeList/1, registerNodes/1, assignNeightbors/1,
	 insertValue/3, insertValue/2, insertValueRandom/3, insertValueRandom/2, searchValueMan/4, 
	 searchValueMan/1, searchValueMan/2, computeNodesDegree/1,
	 searchCoordinatorVT/8, searchCoordinatorVT/5, searchCoordinatorVT/4, searchListener/4, 
	 searchWorker/9, initSynapseNodeDictionary/2, initTopologyFromFile/2, initTopologyFromFileSNDict/3, 
	 initSynapseNodeDictionaryParallel/3, initSynapseNodeDictionaryProcess/4,
	 initNodeListFromFile/1, synapseNodeDictionaryHandler/1, neightborsMonitor/2, retrieveNeightborsReceiver/5,
   computeMsgStats/1 ]).
%-import(dict).
%-import(random).
%-import(timer).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   ROUTING STRATEGY
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Which probability function should we use to select the neightbors?
-define(ROUTING_STRATEGY, one_over_k).
-define(DEFAULT_TTL, 3).
-define(DEFAULT_QUERYINT_MILLIS, 50).

-define(B_DISPLAY_OUTPUT, 100000). %% every how many nodes to display a progress message.
-define(B_NODES, 10000). %% Nodes to be created in parallel
-define(DEFAULT_TIMEOUT, 10000).
-define(INT_RETRY, 2000).
-define(B_GARBAGE_COLLECTOR, 1000).

-define(DEFAULT_VALUE, 5).
-define(NODE_PREFIX, "node").

-define(DEFAULT_NUM_NEIGHBORS, 1). %number of neightbors to select during a flood.
-define(MIN_NBOR, 5).
-define(MAX_NBOR, 100).

-define(PARALLEL_REQUESTS, 1). % number of parallel processes requesting a key.

-record(stat, {value=0, ttl=0, iteration=0, sent=0, response=0, found=0, notFound=0, foundAfter=0, timeout=0, m=0, m2=0, pHit=0, numMessages=[]}).
% found2, notFound2, timeout2, numMessages2 record all the results we get AFTER we receive the first one (used in flooding)

%% ?LOG logs debug messages, these should disappear during the official simulation.
-ifdef(debug).
-define(LOG(X), io:format("{~p,~p,~p} DEBUG: ~p~n", [?MODULE,?LINE,self(),X])).
-else.
-define(LOG(X), true).
-endif.

%% ?LOGROUTE logs routing messages, to be used to test that the routing works correctly
%% (another form of debug, but we might wanna keep normal debug messages while suppressing the routing one)
-ifdef(logRoute).
-define(LOGROUTE(X), io:format("{~p,~p,~p} DEBUG: ~p~n", [?MODULE,?LINE,self(),X])).
-else.
-define(LOGROUTE(X), true).
-endif.

%% ?LOGNEIGHTBORS logs the neightbors selection process, to be used to test that the neightbors are chosen
%% according to the probability function.
-ifdef(logNeightbors).
-define(LOGNEIGHTBORS(), 
	NumSelected = lists:foldl(fun(_,A)->
		A+1 
	end, 0, NL),
	TotalNumNeight = dict:fold(fun(_,V,A1)->
		lists:foldl(fun(_,A2)-> 
			A2+1 
		end, A1, V) 
	end, 0, NeightborhoodsDict),
	?LOG(["Conn Ovl, Tot NBors, Sel NBors: ", dict:size(NeightborhoodsDict), TotalNumNeight, NumSelected])).
-else.
-define(LOGNEIGHTBORS(), true).
-endif.

%% ?MSG is an application output, it can appear even when not in debug mode. To be used carefully, 
%% we cannot generate too many when we are simulating with 10e6 nodes
-ifdef(noMessages).
-define(MSG(X), true).
-else.
-define(MSG(X), io:format("{~p,~p,~p}: ~p~n", [?MODULE,?LINE,self(),X])).
-endif.

%% ?WARN should display an erroneous condition in the simulation, therefore they should 
%% be displayed also when not in debug mode. They should mean the simulation results are not good.
-ifdef(noWarnings).
-define(WARN(X), true).
-else.
-define(WARN(X), io:format("{~p,~p,~p} WARNING!: ~p~n", [?MODULE,?LINE,self(),X])).
-endif.

-ifdef(noSeed).
-define(SEED(), true).
-else.
-define(SEED(), random:seed(erlang:now())).
-endif.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SYNAPSE NODE PROCESS FUNCTION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
-record(synapseState, {nodeId, neightbors, neightborsEntry, synNodeHandlersDict}).
synapseNode(SynapseState) -> %% Id must be in the form "Node2"
	receive
	{garbageCollect} -> erlang:garbage_collect();

	{registerSelf, From} ->
		Id = SynapseState#synapseState.nodeId,
		{ok,Token,_} = erl_scan:string(Id++"."),
		{ok,Term} = erl_parse:parse_term(Token),
		case register(Term, self()) of
			true ->	From ! {okInit, Id, self()};
			false -> From ! {noInit, Id, self()}
		end,
		synapseNode(SynapseState);

	{setNeightbors, NDict, From} ->
		Id = SynapseState#synapseState.nodeId,
		DSize = dict:size(NDict),
		?LOG(["Received neightbors.", Id, DSize]),
		if DSize > 0 ->
			From ! {okNeightbors, Id, self()};
		true ->
			From ! {noNeightbors, Id, self()}
		end,
		synapseNode(SynapseState#synapseState{neightbors=NDict});

	{getNumNeightbors, From} ->
		NDict = SynapseState#synapseState.neightbors,
		{NumOvl, NumNeightbors} = dict:fold(fun(_,NList,{OCount, NCount}) ->
			NewNCount = lists:foldl(fun(_, NC) ->
				NC + 1
			end, NCount, NList),
			{OCount + 1, NewNCount}
		end, {0, 0}, NDict),
		From ! {numNeightbors, NumOvl, NumNeightbors};
%% NEW IMPLEMENTATION
%% Set the node entry
	{nodeEntry, NodeEntry, SynNodeHandlersDict} ->
		synapseNode(SynapseState#synapseState{neightborsEntry=NodeEntry, synNodeHandlersDict=SynNodeHandlersDict});
%% Start retrieveing the neightbors PID
	{retreveNeightbors, NMonPID} ->
		Id = SynapseState#synapseState.nodeId,
		NodeEntry = SynapseState#synapseState.neightborsEntry,
		SynNodeHandlersDict = SynapseState#synapseState.synNodeHandlersDict,
%		?LOG(["Node retrieving neightbors.", Id]),
		retrieveNeightborsWorker(self(), NodeEntry, SynNodeHandlersDict, NMonPID),
		synapseNode(SynapseState);
%% start a worker that broadcasts and collects all the neightbors
%% add also remaining nodes no the synapse node dictionary (python)
	{putValue, Value, From} ->
		Id = SynapseState#synapseState.nodeId,
		case get(Value) of
		undefined ->
			put(Value, Value),
			From ! {okPut, Id, self()};
		_ -> 
			From ! {noPutDuplicate, Id, self()}
		end,
		synapseNode(SynapseState);
	
	{searchValue, Value, TTL, NumNeigh, NumMessages, ReqId, Originator, From} ->
		?LOGROUTE("Received search for value " ++ integer_to_list(Value) ++ 
				" from " ++ pid_to_list(From)),
		Id = SynapseState#synapseState.nodeId,
		Neightbors = SynapseState#synapseState.neightbors,
%% If the node has the value, send a foundValue to the requester	
		case get(Value) of 
			undefined -> %Value not found
				ValueFound = false;
			_ ->
				?LOGROUTE("Value " ++ integer_to_list(Value) ++ " found at node " ++ Id),
				Originator ! {foundValue, Value, NumMessages, ReqId, Id, self()},
				ValueFound = true
		end,
%% If there is still TTL, propagate the message
		if TTL > 0 ->
			Dest = selectDestList(Neightbors, From),
			case length(Dest) of
				0 ->
					Originator ! {endQuery, Value, NumMessages, ReqId, Id, self()};
				_ ->
					[FirstNode|RestList] = Dest,
					FirstNode ! {searchValue, Value, TTL-1, NumNeigh, NumMessages+1, ReqId, Originator, self()}, 
					[X ! {searchValue, Value, TTL-1, NumNeigh, 1, ReqId, Originator, self()} || X <- RestList]
			end;
%% If no more TTL, we stop, and send a notFound message in case the node didn't find it
		true -> 
%			case ValueFound of
%				false ->
					?LOGROUTE("Query for value " ++ integer_to_list(Value) ++ " ended."),
					Originator ! {endQuery, Value, NumMessages, ReqId, Id, self()} %;
%				true ->
%					ok
%			end			
		end,			

%% OLD IMPLEMENTATION
		%case get(Value) of
		%undefined ->
			%case TTL of 
			%0 ->
				%?LOGROUTE("Value " ++ integer_to_list(Value) ++ " not found."),
				%Originator ! {notFound, Value, ReqId, Id, self()};
			%_ ->
				%?LOGROUTE("Forwarding request for value " ++ integer_to_list(Value)),
				%Dest = selectDestList(Neightbors),
				%[X ! {searchValue, Value, TTL-1, NumNeigh, NumMessages+1, ReqId, Originator, self()} || X <- Dest]
			%end;
		%_ -> 
			%?LOGROUTE("Value " ++ integer_to_list(Value) ++ " found at node " ++ Id),
			%Originator ! {foundValue, Value, NumMessages, ReqId, Id, self()},
			%if TTL > 0 ->
				%Dest = selectDestList(Neightbors),
				%[X ! {searchValue, Value, TTL-1, NumNeigh, NumMessages+1, ReqId, Originator, self()} || X <- Dest];
			%true -> 
				%ok
			%end			
		%end,
		synapseNode(SynapseState);
	{ping, From} ->
		Id = SynapseState#synapseState.nodeId,
		?MSG("PING! from: " ++ pid_to_list(From)),
		From ! {pong, Id, self()},
		synapseNode(SynapseState);
% Pid reference test. Bounce a message to another node THEN back to the original one.
	{boing, To, From} ->
		?MSG("BOING! from: " ++ pid_to_list(From) ++ " to: " ++ pid_to_list(To)),
		To ! {ping, From},
		synapseNode(SynapseState);
	{printState, _} ->
		?MSG(SynapseState#synapseState.nodeId),
		?MSG(get()),
		?MSG(SynapseState#synapseState.neightbors),
		?MSG(dict:to_list(SynapseState#synapseState.neightbors)),
		synapseNode(SynapseState);
	{kill, _} -> ok;
	Other -> ?WARN(["Received ", Other])
	end.	

%% Compute the node's degree statistics
computeNodesDegree(NodeDict) ->
	NumNodes = dict:size(NodeDict),
	DegreeDict = dict:fold(fun(Nid,[PID|_], DDict) ->
		?MSG("Sending message"),
		PID ! {getNumNeightbors, self()},
		receive
			{numNeightbors, NOvl, NNeight} ->
				?MSG("Received response"),
				PrevNumNodes = 
				case dict:is_key(NNeight, DDict) of
					true ->
						dict:fetch(NNeight, DDict);
					false ->
						0
				end,
				NewDDict = dict:store(NNeight, PrevNumNodes+1, DDict);
		_ ->
			?MSG("This SHOULD NOT be happening"),
			DDict
		end
	end, dict:new(), NodeDict),
	NDAverage = dict:fold(fun(Deg, NNodes, PrevA) ->
		PrevA + (Deg * (NNodes / NumNodes))
	end, 0, DegreeDict),
	{NDAverage, DegreeDict}.
				
%% Worker process to broadcast requests for node PIDs
retrieveNeightborsWorker(SynapsePID, NodeEntry, SynNodeHandlersDict, NMonPID) -> 
	RecPID = spawn(?MODULE, retrieveNeightborsReceiver, [SynapsePID, dict:new(), dict:new(), NMonPID, false]),
%	?LOG(["Spawned receiver", RecPID, self()]),
	{_, ONList} = NodeEntry,
	lists:map(fun({OvlID, NList}) ->
		OvlHandlersList = dict:fetch(OvlID, SynNodeHandlersDict),
		lists:map(fun(NodeID) ->
			sendNodeInfoRequest(RecPID, OvlID, NodeID, make_ref(), OvlHandlersList)
		end, NList)
	end, ONList),
	RecPID ! {waitResults, self()}.

sendNodeInfoRequest(RecPID, OvlID, NodeID, Ref, OvlHandlerList) ->
	RecPID ! {store, Ref, self()},
	waitForConfirmAndSend(RecPID, OvlID, NodeID, Ref, OvlHandlerList).

waitForConfirmAndSend(RecPID, OvlID, NodeID, Ref, OvlHandlerList) ->
	receive
		{okStored, Ref} ->
%			?LOG(["Sending request", Ref]),
			[H ! {get_key, OvlID, NodeID, Ref, RecPID} || H <- OvlHandlerList]
	after ?DEFAULT_TIMEOUT ->
		?WARN(["Receiver didn't respond within the default timeout, retrying", ?DEFAULT_TIMEOUT, NodeID]),
		waitForConfirmAndSend(RecPID, OvlID, NodeID, Ref, OvlHandlerList)
	end.

%% Listener process to receive requests for node PIDs
retrieveNeightborsReceiver(SynapsePID, ReqDict, NDict, NMonPID, WaitForResults) ->
%	?MSG(["Alive!", self()]),
	receive
		{store, Ref, From} ->
%			?MSG(["Storing ref", Ref, From]),
			NewReqDict = dict:store(Ref, Ref, ReqDict),
			From ! {okStored, Ref},
			retrieveNeightborsReceiver(SynapsePID, NewReqDict, NDict, NMonPID, WaitForResults);
		{found, Value, Ovl, Ref} ->
%			?LOG(["Found ref", Ref]),
			NewNDict = dict:append(Ovl, Value, NDict),
			NewReqDict = dict:erase(Ref, ReqDict),
			case WaitForResults of
				false ->
					retrieveNeightborsReceiver(SynapsePID, NewReqDict, NewNDict, NMonPID, WaitForResults);
				true ->
					case dict:size(NewReqDict) of
		  			0 ->
%							?LOG(["Finished, sending neightbors", SynapsePID]),
							SynapsePID ! {setNeightbors, NDict, NMonPID};
						_ ->				
%							?LOG(["Waiting for neightbors", dict:size(NewReqDict), SynapsePID]),
							%timer:sleep(?INT_RETRY),
							retrieveNeightborsReceiver(SynapsePID, NewReqDict, NewNDict, NMonPID, WaitForResults)
					end
			end;
		no_key ->
%			?WARN(["Couldn't find entry in SynapseNodeDict"]); %It's ok, we do nothing
			retrieveNeightborsReceiver(SynapsePID, ReqDict, NDict, NMonPID, WaitForResults);
		{waitResults, From} ->
			retrieveNeightborsReceiver(SynapsePID, ReqDict, NDict, NMonPID, true)
	end.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NEIGHTBORS SELECTION FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Params record for the 1/K probability function 
% TODO: add other params record for other probabilities
-record(k_params, {k=1}).

% Parameter initialization functions
% TODO: return different parameters, depending on the probability function
init_params(NeightborhoodsDict) ->
	case ?ROUTING_STRATEGY of
		one_over_k -> 
			#k_params{k=dict:size(NeightborhoodsDict)};
		max_messages ->
			?WARN("Not implemented yet!"),
			none
	end.

% Returns the dice roll. True for value < K
get_probability(Params) ->
	case ?ROUTING_STRATEGY of
		one_over_k ->
			fun(_) -> 
				random:uniform() < 1/Params#k_params.k 
			end;
		max_messages ->
			fun(_) ->
				?WARN("Not implemented yet!"),
				false
			end
	end.

% Returns the filtering function
get_filter(NeightborhoodsDict) ->
	Params = init_params(NeightborhoodsDict),
	fun({_, PIDList}) ->
%		PIDList = lists:map(fun({_, NPID}) -> NPID end, dict:to_list(NodeDict)),  
		lists:filter(get_probability(Params), PIDList) 
	end.

% For each neightbors set, return the subset according to the specified filtering function.
selectDestList(NeightborhoodsDict, ExcludePID) -> % {ovlID:[nPID, nPID, ...], ovlID:[nPID, nPID, ...], ...}
	?SEED(),
	FilteredNList = lists:map(get_filter(NeightborhoodsDict), dict:to_list(NeightborhoodsDict)),
	NL = lists:append(FilteredNList),
	?LOGNEIGHTBORS(),
	lists:filter(fun(El) -> 
		if El == ExcludePID -> 
		%	?MSG("OK FOUND"), 
			false; 
		true -> 
			true 
		end 
	end, NL).
	
selectDestListFLOOD(NDict) -> % {ovlID:[nPID, nPID, ...], ovlID:[nPID, nPID, ...], ...}
	?SEED(),
	FilteredNList = dict:to_list(NDict),
	NL = lists:append(FilteredNList),
	?LOGNEIGHTBORS(),
	NL.

%% SIMPLE PING METHOD, DISPLAY THE RESPONSE
pingNode(NodePID) ->
	NodePID ! {ping, self()},
	receive
	{pong, From} ->
		?MSG("PONG! from: " ++ pid_to_list(From))
	end.

boingNode(FirstDest, SecondDest) ->
	FirstDest ! {boing, SecondDest, self()},
	receive
	{pong, From} ->
		?MSG("BOING! from: " ++ pid_to_list(From))
	end.

%% CREATES ALL THE NODE PROCESSES AND INITIALIZES WITH AN ID.
initNodeList(0) -> [];
initNodeList(N) ->
	D = dict:new(),
%	D = gb_trees:empty(),
	initNodeList(N,D).

initNodeList(N, DictNodeList) ->
	DSize = dict:size(DictNodeList),
%	DSize = gb_trees:size(DictNodeList),
	initNodeList(N, DSize, DictNodeList).

initNodeList(0, _, DictNodeList) ->  DictNodeList;

initNodeList(N, IndexOffset, DictNodeList) when N > ?B_NODES ->
	?MSG(integer_to_list(N) ++ " nodes to instantiate."),
	
	PartialDict = initNodeList(?B_NODES, IndexOffset, dict:new()),
%	PartialDict = initNodeList(?B_NODES, IndexOffset, gb_trees:empty()),
	NewDict = dict:merge(fun(_,D1,_)-> D1 end, PartialDict, DictNodeList),
%	NewDict = gb_trees:map(fun(K, V)-> gb_trees:insert(K,V,DictNodeList) end, PartialDict),
	initNodeList(N - ?B_NODES, IndexOffset + ?B_NODES, NewDict);
	
initNodeList(N, IndexOffset, DictNodeList) ->
	NewN = integer_to_list(IndexOffset + 1),
	NewPID = spawn(?MODULE, synapseNode, [#synapseState{nodeId=?NODE_PREFIX ++ NewN}]),
	NewDict = dict:store(IndexOffset + 1, [NewPID], DictNodeList),
%	NewDict = gb_trees:insert(IndexOffset + 1, [NewPID], DictNodeList),
	initNodeList(N-1, IndexOffset+1, NewDict).


%% TELLS THE NODES TO REGISTER THEMSELVES
registerNodes(DictNodeList) ->
	registerNodes(dict:size(DictNodeList), DictNodeList).

registerNodes(0, DictNodeList) -> DictNodeList;

registerNodes(N, DictNodeList) ->
	[NodePID|_] = dict:fetch(N, DictNodeList),
	NodePID ! {registerSelf, self()},
	receive
	{okInit, Id, _} -> 
		?MSG("Node " ++ Id ++ " registered."), 
		NewDict = dict:append(N,reg,DictNodeList),
		registerNodes(N-1, NewDict);
	{noInit, Id, _} -> 
		?WARN("Node " ++ Id  ++ " not registered."), 
		registerNodes(N-1, DictNodeList)
	after ?DEFAULT_TIMEOUT ->
		?WARN("Node " ++ integer_to_list(N) ++ " registration timeout."), 
		registerNodes(N-1, DictNodeList)
	end.

%% Creates and assigns a random neighborhood to each node.
assignNeightbors(NodeDict) -> 
	DSize = dict:size(NodeDict),
	if ?MIN_NBOR > DSize ->
		MinSize = DSize;
	true -> MinSize = ?MIN_NBOR
	end,
	
	if ?MAX_NBOR > DSize ->
		MaxSize = DSize;
	true -> MaxSize = ?MAX_NBOR
	end,
	
	assignNeightbors(DSize, NodeDict, MinSize, MaxSize).

assignNeightbors(0, NodeDict, _, _) -> NodeDict;
% pass also the distribution as [{perc, numNeight},{...}]
assignNeightbors(N, NodeDict, MinSize, MaxSize) -> 
	case N rem ?B_NODES == 0 of
		true -> ?MSG("Neightbors for node " ++ integer_to_list(N));
		false -> true
	end,
	% This should be decided by the distribution
	NumNeigh = MinSize + random:uniform(MaxSize - MinSize),

	DictNeigh = getRandomNeightbors(NumNeigh, NodeDict),
	DSize = dict:size(DictNeigh),

	[Node|_] = dict:fetch(N, NodeDict),
	
	NewDict = sendNeightborsToNode(Node, DictNeigh, DSize, N, NodeDict),

	assignNeightbors(N-1, NewDict, MinSize, MaxSize).

%% Sends the nieghtborhood to the corresponding node
sendNeightborsToNode(Node, DictNeigh, DSize, N, NodeDict) ->
	Node ! {setNeightbors, DictNeigh, self()},
	receive
	{okNeightbors,Id, _} -> 
		?LOG("Node " ++ Id ++ " with " ++ integer_to_list(DSize) ++ " neightbors."),
		NewDict = dict:append(N, DSize, NodeDict);
	{noNeightbors,Id, _} -> 
		?WARN("Node " ++ Id ++ " without neightbors."),
		NewDict = dict:append(N, 0, NodeDict)
	end,
	NewDict.

%% Get random neightbors from the Nodes Dictionary
getRandomNeightbors(N, NodeDict) ->
	?SEED(),
	D = dict:new(),
	DSize = dict:size(NodeDict),
	if N > DSize ->
		NodeDict;
	true ->
		getRandomNeightbors(N, NodeDict, D)
	end.

getRandomNeightbors(0, _, NeighDict) -> NeighDict;
getRandomNeightbors(N, NodeDict, NeighDict) ->
	DSize = dict:size(NodeDict),
	RndKey = random:uniform(DSize),
	[RndNode|_] = dict:fetch(RndKey, NodeDict),
	getRandomNeightbors(N-1, NodeDict, dict:store(dict:size(NeighDict)+1, RndNode, NeighDict)).

%% Synapse Node Dictionary handler process
%% 1 process per overlay dictionary
synapseNodeDictionaryHandler(OPartialDict) ->
	receive
		{get_key, Ovl, NodeID, Ref, From} ->
			%case dict:find(NodeID, D) of
			%	{ok, Value} -> 
			%		From ! {found, Value, Ovl, Ref},
			%		synapseNodeDictionaryHandler(OPartialDict);
			%	error -> 
			%		From ! no_key
			%end;
			case gb_trees:lookup(NodeID, OPartialDict) of
				{value, Val} ->
%					?LOG(["Lookup OK!", NodeID, self()]),
					From ! {found, Val, Ovl, Ref},
					synapseNodeDictionaryHandler(OPartialDict);
				none ->
%					?LOG(["Lookup NO", NodeID, self()]),
					From ! no_key,
					synapseNodeDictionaryHandler(OPartialDict)
			end;
		{die} -> byebye;
		{die, From} -> From ! byebye
	end.
			
%% READ FILE: read the synapse node dictionary file
initSynapseNodeDictionaryParallel(Filename, NumWorkers, NodesGbTree) ->
				Filenames = [Filename ++ "_0" ++ integer_to_list(I) || I <- lists:seq(0, NumWorkers-1)],
				?MSG(Filenames),
				lists:foldl(fun(Filename, I) -> 
					spawn(?MODULE, initSynapseNodeDictionaryProcess, 
						[Filename, self(), I, NodesGbTree]), I+1 
					end, 0, Filenames),
				receive_spawn_handlers(0, NumWorkers, []).

receive_spawn_handlers(N, N, HandlersPIDList) -> 
%% Here I have: [[PidO1-1, PidO2-1...],[PidO1-2, PidO2-2...]...]
%% I Need it to become: {1:[PidO1-1, PidO1-2...], 2:[PidO2-1, PidO2-2...]...}
	lists:foldl(fun(PidList, Acc1Dict) ->  
		{_, HDict} = lists:foldl(fun(Pid, {OIndex, Acc2Dict}) ->
			{OIndex+1, dict:append(OIndex, Pid, Acc2Dict)}
			end, {0, Acc1Dict}, PidList),
		HDict
		end, dict:new(), HandlersPIDList);
receive_spawn_handlers(N, NumWorkers, HandlersList) ->
	receive
		{dict, Dict, _} ->
%% NO MORE MERGING, TOO TIME CONSUMING, WE HANDLE THESE IN PARALLEL
%%NewD = dict:merge(fun(_, D1, D2) -> dict:merge(fun(_, E1, E2) -> [E1, E2] end, D1, D2) end, D, Dict),
%% Decompose the dictionary in a list, spawn a process for each overlay dictionary
			PartialHList = [spawn(?MODULE, synapseNodeDictionaryHandler, [D]) || {_,D} <- dict:to_list(Dict)],
			receive_spawn_handlers(N+1, NumWorkers, [PartialHList|HandlersList])
	end.

initSynapseNodeDictionaryProcess(Filename, CallerPID, Index, NodesGbTree) ->
	?MSG(["Synapse dictionary reader process.", Index]),
	Dict = initSynapseNodeDictionary(Filename, NodesGbTree),
	CallerPID ! {dict, Dict, Index}.

initSynapseNodeDictionary(FileName, NodesGbTree) ->
	{ok, Device} = file:open(FileName, [read]),
	NumOvl = getNumberHeader(Device),
%	DIRECT GB_TREES IMPLEMENTATION
	DictList = dict:from_list([{I, gb_trees:empty()} || I <- lists:seq(0, NumOvl-1)]),

%	FIRST LISTS, THEN WE CREATE THE GB_TREE
%	DictList = dict:from_list([{I, []} || I <- lists:seq(0, NumOvl-1)]),

%	DICTIONARY IMPLEMENTATION
%	DictList = dict:from_list([{I, dict:new()} || I <- lists:seq(0, NumOvl-1)]),

%	DICT/GB_TREE
	FilledDictList = initSynapseNodeDictionary(Device, DictList, 0, NodesGbTree), % ENDS HERE
%	LIST -> GB_TREES ONLY
%	FillSynNodeDict = dict:to_list(initSynapseNodeDictionary(Device, DictList, 0)),
%	dict:from_list([gb_trees:from_orddict(X) || X <- FillSynNodeDict]).
	file:close(Device),
	FilledDictList.
	
initSynapseNodeDictionary(Device, SynNodeDict, N, NodesGbTree) ->
	case N rem ?B_DISPLAY_OUTPUT of
		0 -> ?MSG(["SynapseNodeDictionary entry:", N]);
		_ -> ok
	end,
	case io:get_line(Device, "") of
		eof  -> file:close(Device), SynNodeDict;
		Line -> NewSynNodeDict = parse_syn_node_entry(Line, SynNodeDict, NodesGbTree),
			initSynapseNodeDictionary(Device, NewSynNodeDict, N+1, NodesGbTree)
	end.

parse_syn_node_entry(Line, SynNodeDict, NodesGbTree) ->
	{ok,Token,_} = erl_scan:string(Line), % Line++"."
	{ok,Term} = erl_parse:parse_term(Token), % Term = {{ovl, ovlNode}, synNode}
	{{OvlID, NodeID}, SynNode} = Term,
	ODict = dict:fetch(OvlID, SynNodeDict),
	[NodePID|_] = dict:fetch(SynNode+1, NodesGbTree),
%	[NodePID|_] = gb_trees:get(SynNode+1, NodesGbTree),
%	DICTIONARY IMPLEMENTATION
%	NewODict = dict:store(NodeID, NodePID, ODict),

% 	GB_TREES IMPLEMENTATION
	NewODict = gb_trees:insert(NodeID, NodePID, ODict),

%	LIST -> GB_TREE IMPLEMENTATIO
%	NewODict = ODict ++ [{NodeID, SynNode}],

	dict:store(OvlID, NewODict, SynNodeDict).

initNodeListFromFile(Filename) ->
	{ok, Device} = file:open(Filename, [read]),
	NumNodes = getNumberHeader(Device),
	NTree = initNodeList(NumNodes),
	file:close(Device),
	NTree.

%% READ FILE: read the synapse topology file, init nodes, assign neightbors
initTopologyFromFileSNDict(FileName, SynNodeHandlersDict, NodesGbTree) ->
	{ok, Device} = file:open(FileName, [read]),
	NumNodes = getNumberHeader(Device),
%	NumOverlays = getNumberHeader(Device),
%	NodeDict = initNodeList(NumNodes),
	Res = assignNeightborsFromFileSNDict(Device, NodesGbTree, SynNodeHandlersDict, NumNodes),
	file:close(Device),
	Res.

initTopologyFromFile(FileName, NodesGbTree) ->
	{ok, Device} = file:open(FileName, [read]),
	NumNodes = getNumberHeader(Device),
%	NumOverlays = getNumberHeader(Device),
% NodeDict = initNodeList(NumNodes),
	Res = assignNeightborsFromFile(Device, NodesGbTree, 0, NumNodes),
	file:close(Device),
	Res.

getNumberHeader(Device) ->
	case io:get_line(Device, "") of
		eof  -> file:close(Device), error;
		Line -> parseNum(Line)
	end.

%% Read info from topology file: DEPRECATED!!! (it took too long to generate such file)
% NUM_NODES
% NEIGHTBORS
%initFromFile(FileName) -> % First iteration, read number of nodes
	%{ok, Device} = file:open(FileName, [read]),
	%case io:get_line(Device, "") of
		%eof  -> file:close(Device), ok;
		%Line -> NumNodes = parseNum(Line),
			%NodeDict = initNodeList(NumNodes),
			%assignNeightborsFromFile(Device, NodeDict)
	%end.

%% Parse the first line of the topology file, returns number of nodes.
parseNum(Line) ->
%	String = unicode:characters_to_list(Line),
	{ok,Token,_} = erl_scan:string(Line), % Line++"."
	{ok,Term} = erl_parse:parse_term(Token),
	Term.

neightborsMonitor(NumNodes, NumNodes) ->
	?MSG(["All nodes initialized!", NumNodes]), ok;
neightborsMonitor(N, NumNodes) ->
	case N rem ?B_DISPLAY_OUTPUT of
		0 ->
			?MSG(["Node init complete (neightbors assigned).", N, NumNodes]);
		_ -> ok
	end,
	receive
		{okNeightbors, Id, From} ->
			?LOG(["Node initialized.", Id, From]),
			neightborsMonitor(N+1, NumNodes);
		{noNeightbors, Id, From} ->
			?WARN(["Node with no neightbors!", Id, From]),
			neightborsMonitor(N+1, NumNodes)
	end.

%% Assign neightbors, reading them from file
assignNeightborsFromFileSNDict(FileDevice, NodesGbTree, SynNodeHandlersDict, NumNodes) ->
	NMoniPID = spawn(?MODULE, neightborsMonitor, [0, NumNodes]),
	assignNeightborsFromFileSNDict(FileDevice, NodesGbTree, SynNodeHandlersDict, 0, NumNodes, NMoniPID).

assignNeightborsFromFileSNDict(FileDevice, NodesGbTree, SynNodeHandlersDict, N, NumNodes, NMoniPID) ->
	case io:get_line(FileDevice, "") of
		eof  -> file:close(FileDevice), {done, N};
%		Line -> {Node, DictNeigh, DNSize} = parseNeightbors(Line, NodeDict, SynapseNodeDict),
		Line -> {Node, NodeEntry} = parseNeightbors(Line, NodesGbTree),
			case N rem ?B_DISPLAY_OUTPUT of  
				0 -> ?MSG(["Reading entry: ", N]);
				_ -> ok
			end,
			Node ! {nodeEntry, NodeEntry, SynNodeHandlersDict},
%			Node ! {retreveNeightbors, NMoniPID},
%			NewDict = sendNeightborsEntryToNode(Node, DictNeigh, DNSize, N, NodeDict),
			assignNeightborsFromFileSNDict(FileDevice, NodesGbTree, SynNodeHandlersDict, N+1, NumNodes, NMoniPID)
	end.

assignNeightborsFromFile(FileDevice, NodesGbTree, N, NumNodes) ->
	case io:get_line(FileDevice, "") of
		eof  -> 
			file:close(FileDevice), {done, N};
		Line -> 
			{Node, DictNeigh, DNSize} = parseNeightborsPID(Line, NodesGbTree),
			case N rem ?B_DISPLAY_OUTPUT of  
				0 -> ?MSG(["Reading entry: ", N]);
				_ -> ok
			end,
			?LOG(["Sending neightbors to node", Node]),
			Node ! {setNeightbors, DictNeigh, self()},
			receive
				{okNeightbors, _, _} -> 
					?LOG(["Received ok form node.", Node]),
					ok;
				{noNeightbors, Id, _} ->
					?WARN(["No neightbors for node.", Id])
			end,
			%NewDict = sendNeightborsEntryToNode(Node, DictNeigh, DNSize, N, NodeDict),
			assignNeightborsFromFile(FileDevice, NodesGbTree, N+1, NumNodes)
	end.

%% Parse the negihtbors line from the topology file S
parseNeightbors(Line, NodesGbTree) ->
%	String = unicode:characters_to_list(Line),
	{ok,Token,_} = erl_scan:string(Line), % ++"."
	{ok,Term} = erl_parse:parse_term(Token),
	%% Term = {NodeNum, [{ovlID, [N1, N2, N3...]}, ...]}.
%% OLD, SERIAL IMPLEMENTATION, DEPRECATED 
	{NodeNum, _} = Term, % OvlNBorList = [{ovlID, [n1, n2, n3...]}, ...].
	%RemappedOvlNBorList = lists:map(fun({OvlID, NList}) ->
				%{OvlID, lists:map(fun(NodeID) -> case dict:is_key({OvlID, NodeID}, SynapseNodeDict) of
									%true ->	dict:fetch({OvlID, NodeID}, SynapseNodeDict);
									%false -> NodeID
								%end 
						%end, NList)} end, OvlNBorList),
	%{OvlIdList, NBorhoodList} = lists:unzip(RemappedOvlNBorList), % {[ovl1, ovl2,...], [[n1, n2, ...],[...], ...]}
	%NBorhoodPIDList = lists:map(fun(NBorsList) -> 
					%dict:from_list(lists:map(fun(Key) -> 
									%[NPid|_] = dict:fetch(Key+1, NodeDict),
									%{Key, NPid} 
								%end, 
								%NBorsList)) 
					%end, 
					%NBorhoodList),
	%DictNeigh = dict:from_list(lists:zip(OvlIdList, NBorhoodPIDList)), % DictNeigh = {ovlId: {nId:nPID, ...}, ...}
%% END DEPRECATED
	[NodePID|_] = dict:fetch(NodeNum+1, NodesGbTree), 
%	[NodePID|_] = gb_trees:get(NodeNum+1, NodesGbTree), 
	{NodePID, Term}.
%% STILL DEPRECATED RETURN
%	{NodePID, DictNeigh, dict:size(DictNeigh)}.

parseNeightborsPID(Line, NodesGbTree) ->
	{ok,Token,_} = erl_scan:string(Line), % ++"."
	{ok,Term} = erl_parse:parse_term(Token),
	%% Term = {NodeNum, [{ovlID, [N1, N2, N3...]}, ...]}.
	{NodeNum, OvlNBorList} = Term, % OvlNBorList = [{ovlID, [n1, n2, n3...]}, ...].
	{OvlIdList, NBorhoodList} = lists:unzip(OvlNBorList), % {[ovl1, ovl2,...], [[n1, n2, ...],[...], ...]}
	NBorhoodPIDList = lists:map(fun(NBorsIDList) -> 
		lists:map(fun(NodeKey) -> 
			[NPid|_] = dict:fetch(NodeKey, NodesGbTree),
		 	NPid	
		end, NBorsIDList) 
	end, NBorhoodList),
	DictNeigh = dict:from_list(lists:zip(OvlIdList, NBorhoodPIDList)), % DictNeigh = {ovlId: {nId:nPID, ...}, ...}
	[NodePID|_] = dict:fetch(NodeNum, NodesGbTree), 
  	{NodePID, DictNeigh, dict:size(DictNeigh)}.

parseNeightborsPID(Line, NodesGbTree, SynapseNodeDict) ->
	{ok,Token,_} = erl_scan:string(Line), % ++"."
	{ok,Term} = erl_parse:parse_term(Token),
	%% Term = {NodeNum, [{ovlID, [N1, N2, N3...]}, ...]}.
	{NodeNum, OvlNBorList} = Term, % OvlNBorList = [{ovlID, [n1, n2, n3...]}, ...].
	RemappedOvlNBorList = 
		lists:map(fun({OvlID, NList}) ->
			{OvlID, lists:map(fun(NodeID) -> 
				case dict:is_key({OvlID, NodeID}, SynapseNodeDict) of
					true ->	
						dict:fetch({OvlID, NodeID}, SynapseNodeDict);
					false ->
						NodeID
				end 
			end, NList)} 
		end, OvlNBorList),
	{OvlIdList, NBorhoodList} = lists:unzip(RemappedOvlNBorList), % {[ovl1, ovl2,...], [[n1, n2, ...],[...], ...]}
	NBorhoodPIDList = lists:map(fun(NBorsIDList) -> 
		lists:map(fun(NodeKey) -> 
			[NPid|_] = dict:fetch(NodeKey, NodesGbTree),
		 	NPid	
		end, NBorsIDList) 
	end, NBorhoodList),
	DictNeigh = dict:from_list(lists:zip(OvlIdList, NBorhoodPIDList)), % DictNeigh = {ovlId: {nId:nPID, ...}, ...}
	[NodePID|_] = dict:fetch(NodeNum, NodesGbTree), 
  	{NodePID, DictNeigh, dict:size(DictNeigh)}.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Process to insert a key with popularity alpha
insertValue(Popularity, NodeDict) when is_float(Popularity) -> insertValue(?DEFAULT_VALUE, Popularity, NodeDict).

insertValue(Value, Popularity, NodeDict) when is_float(Popularity) ->
	DSize = dict:size(NodeDict),
	NValues = erlang:trunc((DSize * Popularity) + 0.5),
	insertValue(Value, Popularity, NodeDict, NValues).

insertValue(_, _, NodeDict, 0) -> NodeDict;
insertValue(Value, Popularity, NodeDict, N) ->
	[Node|_] = dict:fetch(N, NodeDict),
	Node ! {putValue, Value, self()},
	receive
	{okPut, Id, _} ->
		?LOG("Value " ++ integer_to_list(Value) ++ " stored in node " ++ Id),
		NewDict = dict:append(N, {value, Value}, NodeDict),
		insertValue(Value, Popularity, NewDict, N-1);
	{noPutDuplicate, Id, _} ->
		?WARN("Value " ++ integer_to_list(Value) ++ " already in " ++ Id),
		insertValue(Value, Popularity, NodeDict, N-1)
	end.

% Inserts a key to Random nodes (not the first N)
insertValueRandom(Popularity, NodeDict) when is_float(Popularity) -> 
	insertValueRandom(?DEFAULT_VALUE, Popularity, NodeDict).

insertValueRandom(Value, Popularity, NodeDict) when is_float(Popularity) ->
	DSize = dict:size(NodeDict),
	NValues = erlang:trunc((DSize * Popularity) + 0.5),
	?SEED(),
	insertValueRandom(Value, Popularity, NodeDict, NValues).

insertValueRandom(_, _, NodeDict, 0) -> NodeDict;
insertValueRandom(Value, Popularity, NodeDict, N) ->
	RndIdx = random:uniform(dict:size(NodeDict)),
	[Node|_] = dict:fetch(RndIdx, NodeDict),
	Node ! {putValue, Value, self()},
	receive
	{okPut, Id, _} ->
		?LOG("Value " ++ integer_to_list(Value) ++ " stored in node " ++ Id),
		NewDict = dict:append(N, {value, Value}, NodeDict),
		insertValueRandom(Value, Popularity, NewDict, N-1);
	{noPutDuplicate, Id, _} ->
		?LOG("Value " ++ integer_to_list(Value) ++ " already in " ++ Id),
		insertValueRandom(Value, Popularity, NodeDict, N)
	end.

% Searches for one value (used to test).
searchValueMan(StartNode) -> 
	searchValueMan(?DEFAULT_VALUE, StartNode, ?DEFAULT_TTL, ?DEFAULT_NUM_NEIGHBORS).

searchValueMan(StartNode, TTL) -> 
	searchValueMan(?DEFAULT_VALUE, StartNode, TTL, ?DEFAULT_NUM_NEIGHBORS).

searchValueMan(Value, StartNode, TTL, NumNeigh) ->
	StartNode ! {searchValue, Value, TTL, NumNeigh, 1, make_ref(), self(), self()},
	receive
	{endQuery, Value, _, _, _, _} ->
		?MSG("Value " ++ integer_to_list(Value) ++ " not found.");
	{foundValue, Value, NumMsg, _, _, _} ->
		?MSG("Value " ++ integer_to_list(Value) ++ " found after " ++ 
			integer_to_list(NumMsg) ++ " steps.")
	end.

%% Experiment coordinator (spawns the workers and collects the data)
searchCoordinatorVT(NumIterations, NumWorkers, NodeDict, OutputFile) ->
	searchCoordinatorVT(?DEFAULT_VALUE, ?DEFAULT_TTL, NumIterations, NumWorkers, ?DEFAULT_QUERYINT_MILLIS, 1, NodeDict, OutputFile).

searchCoordinatorVT(TTL, NumIterations, NumWorkers, NodeDict, OutputFile) ->
	searchCoordinatorVT(?DEFAULT_VALUE, TTL, NumIterations, NumWorkers, ?DEFAULT_QUERYINT_MILLIS, 1, NodeDict, OutputFile).

searchCoordinatorVT(Value, TTL, NumIterations, NumWorkers, QueryIntervalMillis, DownsampleFactor, NodeDict, OutputFile) ->
	DSize = dict:size(NodeDict),
	NumNodesPerWorker = (DSize div NumWorkers),
	RemainderLastWorker = (DSize rem NumWorkers) div DownsampleFactor,
	
	ListenersList = [spawn(?MODULE, searchListener, [self(), dict:new(), onGoing, none]) || _ <- lists:seq(1, NumWorkers)],
	
	OffsetsList = [X * NumNodesPerWorker || X <- lists:seq(0, NumWorkers-1)],
	NumNodesList = [NumNodesPerWorker div DownsampleFactor || _  <- lists:seq(1, NumWorkers-1)] ++ 
			[NumNodesPerWorker div DownsampleFactor + RemainderLastWorker],
	ParamList = lists:zip3(OffsetsList, NumNodesList, ListenersList),

	[spawn(?MODULE, searchWorker, [Value, TTL, {NumIterations, NumIterations},
	 QueryIntervalMillis, {I, I}, {N, N}, NodeDict, L, DownsampleFactor]) || {I, N, L} <- ParamList],
	
%	?MSG("Starting the stats collector"),
	Stats = collectStats(NumWorkers*NumIterations, dict:new(), NodeDict),

	dict:map(fun(K, [PID|_]) -> erlang:garbage_collect(PID) end, NodeDict),

	saveToFile(OutputFile, Stats).
%	receive
%	{stats, Stats} -> 
%		?MSG("COORDINATOR: Received stats. Simulation ends."), 
		% First version of stats saving. Will update when we know what data we need (i.e., generate a CSV or something...)
		%file:write(OutputFile, dict:to_list(Stats)),
		%dict:to_list(Stats)
%		Stats
%	end.

saveToFile(OutputFile, StatDict) ->
	file:write_file(OutputFile, io_lib:fwrite("", [])),
	dict:map(fun(_, StatRecord) ->
		NumMsgList = StatRecord#stat.numMessages,
		NumMsgLen = length(NumMsgList),
		NumMsgTot = lists:foldl(fun(NM, S) -> S+NM end, 0, NumMsgList),
		M2 = StatRecord#stat.response/StatRecord#stat.sent,
		NewStatRecord = StatRecord#stat{m=NumMsgTot/NumMsgLen, m2=M2, pHit=StatRecord#stat.found/StatRecord#stat.sent},
		file:write_file(OutputFile, io_lib:fwrite("~p.\n", [NewStatRecord]), [append]),
		NewStatRecord
	end, StatDict)
.

%% Collect statistics from workers and listeners.
collectStats(0, StatDict, NodeDict) -> 
	?MSG("COLLECTOR: All results collected."), 
%	Collector ! {stats, StatDict};
	erlang:garbage_collect(),
	StatDict;

collectStats(NumRemainingStats, StatDict, NodeDict) ->
	receive
	{stats, PartialStatDict, From} ->
		?MSG(["COLLECTOR: Received stats from: ", From]),
		dict:map(fun(K, [PID|_]) -> erlang:garbage_collect(PID) end, NodeDict),%;
		collectStats(NumRemainingStats-1, 
			dict:merge(fun(_, RS1, RS2) ->
				#stat{value = RS1#stat.value, ttl = RS1#stat.ttl, iteration = RS1#stat.iteration,
					sent = RS1#stat.sent + RS2#stat.sent,
			  	found = RS1#stat.found + RS2#stat.found,
					foundAfter = RS1#stat.foundAfter + RS2#stat.foundAfter,
			  	notFound = RS1#stat.notFound + RS2#stat.notFound,
			  	response = RS1#stat.response + RS2#stat.response,
			  	timeout = RS1#stat.timeout + RS2#stat.timeout,
			  	numMessages = RS1#stat.numMessages ++ RS2#stat.numMessages}
				end, 
			PartialStatDict, StatDict), NodeDict)
	end.

%% Workers to perform the queries
searchWorker(Value, TTL, {_, 1}, _, {InitIO, _}, {_, 0}, _, Listener, DSFactor) -> 
	?LOG(["WORKER: finished.", Value, TTL, InitIO]),
	Listener ! {done, self()},
	ok;

searchWorker(Value, TTL, {InitNI, NumIterations}, 
	     QueryIntervalMillis, {InitIO, _}, 
	     {InitNN,0}, NodeDict, Listener, DSFactor) ->
	?LOG(["WORKER: Iteration finished", Value, TTL, NumIterations]),
	erlang:garbage_collect(),
	Listener ! {waitingNextIteration, self()},
	receive
		startIteration ->
%			if InitIO == 0 -> % If I'm the first worker, I take care of cleaning the memory before a new iteration
				dict:map(fun(K, [PID|_]) -> erlang:garbage_collect(PID) end, NodeDict),%;
%			true -> 
%				ok
%			end,
			searchWorker(Value, TTL, {InitNI, NumIterations-1}, 
				QueryIntervalMillis, {InitIO, InitIO}, 
				{InitNN,InitNN}, NodeDict, Listener, DSFactor);
		die ->
			byebye
	end;
	
searchWorker(Value, TTL, {InitNI, NumIterations}, QueryIntervalMillis, {InitIO, IndexOffset}, 
	    {InitNN,NumNodes}, NodeDict, Listener, DSFactor) ->
	case IndexOffset rem ?B_DISPLAY_OUTPUT of
		0 -> 
			?MSG(["Worker querying node: ", IndexOffset]),
			?SEED();
		_ -> ok
	end,
	case NumNodes rem ?B_GARBAGE_COLLECTOR of
		0 ->
			?LOG(["Calling erlang:garbage_collect() on all nodes", NumNodes, "/", InitNN]),
			dict:map(fun(_,[PID|_]) -> erlang:garbage_collect(PID) end, NodeDict);
		_ ->
			none
	end,
	ReqId = make_ref(),
	Listener ! {newRequest, ReqId, Value, TTL, NumIterations, self()},
	receive
	{okGo, ReqId} -> 
		RndIdx = 
			if DSFactor /= 1 ->
				InitIO + random:uniform(InitNN);
			true ->
				IndexOffset + 1
			end,
		[StartNode|_] = dict:fetch(RndIdx, NodeDict),
		?LOG(["Query", self(), Value, TTL, NumIterations, IndexOffset]),
		StartNode ! {searchValue, Value, TTL, ?DEFAULT_NUM_NEIGHBORS, 0, ReqId, Listener, self()}
	after ?DEFAULT_TIMEOUT ->
		?WARN({"WORKER: Search worker didn't get OK from the listener! Next iteration...", 
			self(), Value, TTL, IndexOffset, NumIterations})
	end,
	timer:sleep(QueryIntervalMillis),
	
	searchWorker(Value, TTL, {InitNI,NumIterations}, QueryIntervalMillis, {InitIO, IndexOffset+1}, 
		    {InitNN, NumNodes-1}, NodeDict, Listener, DSFactor).
	
%% Listener processes for results.
searchListener(ExperimentCoordinator, StatDict, OnGoing, WorkerPID) ->
	receive
%% The worker is creating a new request, ask the lilstener to store 
%% a record of it with all the parameters (Value, TTL, Iteration).
	{newRequest, ReqId, Value, TTL, Iteration, From} ->
		put(ReqId, {erlang:now(), Value, TTL, Iteration, 0, false}),
		From ! {okGo, ReqId},
		case dict:find({Value, TTL, Iteration}, StatDict) of
			error -> 
				StatRec = #stat{value=Value, ttl=TTL, iteration=Iteration, sent=1};		
			{ok, ExistingRec} ->	
				PrevSent = ExistingRec#stat.sent,
				StatRec = ExistingRec#stat{sent=PrevSent+1}
		end,
		NewStatDict = dict:store({Value, TTL, Iteration}, StatRec, StatDict),
		searchListener(ExperimentCoordinator, NewStatDict, onGoing, From);

%% A search has completed the value has been found
	{foundValue, Value, NumMsg, ReqId, Id, From} ->
		case Rec = get(ReqId) of
			undefined -> 
				?WARN(["FOUND Response without a ReqId. This shoudln't happen", ReqId]),
				NewStatDict = StatDict;
			_ ->	{_, Value, TTL, Iteration, PrevNumMsg, AlreadyFound} = Rec,
				case dict:find({Value, TTL, Iteration}, StatDict) of
					error -> 
						StatRec = #stat{value=Value, ttl=TTL, iteration=Iteration, found=1},
						NewAlreadyFound = true; 
					{ok, ExistingRec} ->
						if AlreadyFound == true -> %if the value for this request has been already found, we ignore it for stats
							PrevFoundAfter = ExistingRec#stat.foundAfter,
							StatRec = ExistingRec#stat{foundAfter=PrevFoundAfter+1},
							NewAlreadyFound = AlreadyFound;
						true ->
							PrevFound = ExistingRec#stat.found,
							StatRec = ExistingRec#stat{found=PrevFound+1},
							NewAlreadyFound = true
						end
				end,
%				put(ReqId, {erlang:now(), Value, TTL, Iteration, PrevNumMsg+NumMsg, NewAlreadyFound}),
				put(ReqId, {erlang:now(), Value, TTL, Iteration, PrevNumMsg, NewAlreadyFound}),
				NewStatDict = dict:store({Value, TTL, Iteration}, StatRec, StatDict)
			% erase(ReqId)
		end,
		searchListener(ExperimentCoordinator, NewStatDict, OnGoing, WorkerPID);
%% A search has completed, value has not been found.
	{endQuery, Value, NumMessages, ReqId, Id, From} ->
		case Rec = get(ReqId) of
			undefined ->
				?WARN(["END QUERY	Response without a ReqId. This shoudln't happen", ReqId]),
				NewStatDict = StatDict;
			_ ->	
				{_, Value, TTL, Iteration, PrevNumMsg, AlreadyFound} = Rec,
				case dict:find({Value, TTL, Iteration}, StatDict) of
					error ->
						StatRec = #stat{value=Value, ttl=TTL, iteration=Iteration, response=1, notFound=1};
					{ok, ExistingRec} ->	
						PrevNotFound = ExistingRec#stat.notFound,
						PrevResponse = ExistingRec#stat.response,
						StatRec = ExistingRec#stat{response=PrevResponse+1, notFound=PrevNotFound+1}
				end,
				put(ReqId, {erlang:now(), Value, TTL, Iteration, PrevNumMsg+NumMessages, AlreadyFound}),
				NewStatDict = dict:store({Value, TTL, Iteration}, StatRec, StatDict)
	%			erase(ReqId)
		end,
		searchListener(ExperimentCoordinator, NewStatDict, OnGoing, WorkerPID);
	{done, From} -> 
		searchListener(ExperimentCoordinator, StatDict, waitEnd, From);
	{waitingNextIteration, From} ->
		searchListener(ExperimentCoordinator, StatDict, waitNextIteration, From);
	{kill} -> ?LOG("LISTENER: adieu."), ok
	after ?DEFAULT_TIMEOUT ->
%		case length(get()) of
%			0 -> 
		case OnGoing of
			waitNextIteration ->
				?MSG(["LISTENER: done receiving messages. Waiting for next iteration", self()]),
				if WorkerPID /= none ->
					FinalStatDict = lists:foldl(fun({_, {_, V, TTL, I, NumMsg, _}}, SDict) ->
						StatRec = dict:fetch({V, TTL, I}, SDict),
						if NumMsg > 0 ->
							PrevMessages = StatRec#stat.numMessages,
							dict:store({V, TTL, I}, StatRec#stat{numMessages = [NumMsg|PrevMessages]}, SDict);
						true ->
							PrevTimeout = StatRec#stat.timeout,
							dict:store({V, TTL, I}, StatRec#stat{timeout=PrevTimeout+1}, SDict)
						end
					end, StatDict, get()),
					erase(),
	
					ExperimentCoordinator ! {stats, FinalStatDict, self()},				
					erlang:garbage_collect(),
					WorkerPID ! startIteration, %We wait until the previous iteration has finished before new one
					searchListener(ExperimentCoordinator, dict:new(), OnGoing, WorkerPID);
				true ->
					{die, unused}
				end;
			waitEnd ->	
				?MSG(["LISTENER: No more queries, quitting listener", self()]),

				FinalStatDict = lists:foldl(fun({_, {_, V, TTL, I, NumMsg, _}}, SDict) ->
					StatRec = dict:fetch({V, TTL, I}, SDict),
					if NumMsg > 0 ->
						PrevMessages = StatRec#stat.numMessages,
						dict:store({V, TTL, I}, StatRec#stat{numMessages = [NumMsg|PrevMessages]}, SDict);
					true ->
						PrevTimeout = StatRec#stat.timeout,
						dict:store({V, TTL, I}, StatRec#stat{timeout=PrevTimeout+1}, SDict)
					end
				end, StatDict, get()),
				erase(),

				ExperimentCoordinator ! {stats, FinalStatDict, self()},
				erlang:garbage_collect(),
				ok;
			onGoing ->
				?WARN(["Receiver timing out, system probably under heavy load."]),
				searchListener(ExperimentCoordinator, StatDict, OnGoing, WorkerPID)
		end %;
%			_ -> 
%				?MSG(["LISTENER: Listener still waiting for responses. Retrying", self()]),
%				searchListener(ExperimentCoordinator, StatDict, OnGoing, WorkerPID)
%		end
	end.

computeMsgStats(MsgList) ->
	lists:foldl(fun(El, SD) ->
		PrevNum = 
			case dict:is_key(El, SD) of
				true ->
					dict:fetch(El, SD);
				false ->
					0
			end,
			dict:store(El, PrevNum+1, SD)
		end, dict:new(), MsgList).
