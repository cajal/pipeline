%{
experiment.ScanBehaviorFile (manual) # file with info about synchronization, wheel, timestamping
-> experiment.Scan
---
filename                    : varchar(50)                   # filename of the h5 file
%}


classdef ScanBehaviorFile < dj.Relvar
end