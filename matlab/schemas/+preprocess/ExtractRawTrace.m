%{
preprocess.ExtractRawTrace (imported) # raw trace, common to Galvo
-> preprocess.ExtractRaw
-> preprocess.Channel
trace_id        : smallint               # 
---
raw_trace                   : longblob                      # unprocessed calcium trace
%}


classdef ExtractRawTrace < dj.Relvar
    % it's a part table of ExtractRaw
end