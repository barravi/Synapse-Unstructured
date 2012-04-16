#!/usr/bin/env escript
%% -*- erlang -*-
%%! -smp enable +P 1500000
%% -sname factorial -mnesia debug verbose

main([TopologyFile, OutputStatFile, AString, TTLString, PfString, SFactor]) ->
	Workers = 4,
	Value = 5,
	QueryInterval = 0,
	Values = [{1,0.1}, {2, 0.01}, {3, 0.001}, {4, 0.0001}, {5, 0.00001}, {6, 0.000001}],

%    try
	Alpha = list_to_float(AString),
	TTL = list_to_integer(TTLString),
	Pf = list_to_float(PfString),
	SamplingFactor = list_to_integer(SFactor),

	io:format("SynapseSimlation ~w ~w ~w ~p ~p ~p\n", [Alpha, TTL, Pf, SFactor, TopologyFile, OutputStatFile]),

%	compile:file(synapse, [{d, no_propagation}]),	
	compile:file(synapse),
%    rr(synapse),

	NTree = synapse:initNodeListFromFile(TopologyFile),
	synapse:initTopologyFromFile(TopologyFile, NTree),

	lists:map(fun({Val, Al}) ->
		synapse:insertValueRandom(Val, Al, NTree)			
	end, Values),
%synapse:insertValueRandom(Value, Alpha, NTree),

%Stats = 
%	synapse:searchCoordinatorVT(Value,TTL,1,Workers,QueryInterval,SamplingFactor, NTree, OutputStatFile)
	synapse:searchCoordinatorVT(0,TTL,1,Workers,QueryInterval,SamplingFactor, NTree, OutputStatFile)

%    catch
%        E1:E2 ->
%	    io:format("~p, ~p\n",[E1, E2]),
%		usage()
%    end
;

main(_) ->
    usage().

usage() ->
    io:format("usage: runSynapseSimulation.erl topologyFile outputFile alpha TTL Pf SamplingFactor\n"),
    halt(1).

