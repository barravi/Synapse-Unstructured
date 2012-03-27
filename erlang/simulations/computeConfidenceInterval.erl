#!/usr/bin/env escript
%% -*- erlang -*-
%%! -smp enable
%% -sname factorial -mnesia debug verbose

main([StatFile]) ->
	try
		{ok, Device} = file:open(StatFile, [read]),
		StatList = read_entry_list(Device, []),
% StatList format: [{PHit1, M1}, {PHit2, M2}...]
		file:close(Device),
% Compute the average
		N = length(StatList),
		{AvgPHit, AvgM} = lists:foldl(
			fun({PHit, M}, {AccPHit, AccM}) ->
				{AccPHit+(PHit/N), AccM+(M/N)}
			end,
			{0, 0}, StatList),
% Compute the variance
		Np = N-1,
		{VarPHit, VarM} = lists:foldl(
			fun({PHit, M}, {AccPHit, AccM}) ->
				{AccPHit+(math:pow(PHit-AvgPHit, 2)/Np), AccM+(math:pow(M-AvgM, 2)/Np)}
			end,
			{0, 0}, StatList),
% Compute the standard deviation
		{StdDevPHit, StdDevM} = {math:sqrt(VarPHit), math:sqrt(VarM)},
% Compute the stadard error of mean
		Nps = math:sqrt(N),
		{SEMPHit, SEMM} = {StdDevPHit/Nps, StdDevM/Nps},
% Compute the confidence interval
% 68 %
		{CI68PHit, CI68M} = {SEMPHit, SEMM},
% 95 %
		Z = 1.96,
		{CI95PHit, CI95M} = {SEMPHit * Z, SEMM * Z},
% Output results
		io:format("PHit 68% Confidence interval: ~f +/- ~f \n", [AvgPHit, CI68PHit]),
		io:format("PHit 95% Confidence interval: ~f +/- ~f \n", [AvgPHit, CI95PHit]),
		io:format("M 68% Confidence interval: ~f +/- ~f \n", [AvgM, CI68M]),
		io:format("M 95% Confidence interval: ~f +/- ~f \n", [AvgM, CI95M])
	catch
		E1:E2 ->
	    io:format("~p, ~p\n",[E1, E2]),
			usage()
	end;

%% TODO: add directlly here the message average computation
main(_) ->
	usage().

usage() ->
	io:format("usage: computeConfidenceInterval.erl statFile\n"),
	halt(1).

read_entry_list(Device, EntryList) ->
	case io:get_line(Device, "") of
		eof  -> 
			EntryList;
		Line -> 
			{ok,Token,_} = erl_scan:string(Line), % Line++"."
			{ok,Term} = erl_parse:parse_term(Token),
			NewEntryList = [Term | EntryList],
			read_entry_list(Device, NewEntryList)
	end.
