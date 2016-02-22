%{
trk.SelectionProtocol (lookup) # groups of filtering steps to reject bad frames
filter_protocol_id: int                  # id of the filtering protocol
---
protocol_name               : char(50)                      # descriptive name of the protocol
%}


classdef SelectionProtocol < dj.Relvar
end