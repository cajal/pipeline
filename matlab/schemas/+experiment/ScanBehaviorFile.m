%{
experiment.ScanBehaviorFile (manual) # name of the running wheel file
-> experiment.Scan
---
filename                    : varchar(50)                   # filename of the video
%}


classdef ScanBehaviorFile < dj.Relvar
end