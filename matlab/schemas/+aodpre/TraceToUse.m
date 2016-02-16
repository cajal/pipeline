%{
aodpre.TraceToUse (computed) # my newest table
-> aodpre.TraceSetToUse
trace_id :  smallint   # trace number within scan
-----
trace : longblob  #  traces

%}

classdef TraceToUse < dj.Relvar
end