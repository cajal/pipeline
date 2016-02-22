%{
trk.ProtocolStep (lookup) # single filter in a protocol to accept frames
-> trk.FilterProtocol
-> trk.FrameFilter
priority        : int                    # priority of the filter step, the low the higher the priority
---
filter_param=null           : longblob                      # parameters that are passed to the filter
%}


classdef ProtocolStep < dj.Relvar
end