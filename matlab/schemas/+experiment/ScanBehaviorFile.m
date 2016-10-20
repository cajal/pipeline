%{
experiment.ScanBehaviorFile (manual) # name of the running wheel file
-> experiment.Scan
---
filename                    : varchar(50)                   # filename of the h5 file
%}


classdef ScanBehaviorFile < dj.Relvar
end