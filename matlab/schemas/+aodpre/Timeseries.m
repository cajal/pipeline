%{
aodpre.Timeseries (imported) # raw trace from each channel
-> aodpre.ScanPoint
-> aodpre.Channel
-----
trace : longblob   #  fluorescent trace from given channel
%}

classdef Timeseries < dj.Relvar
end