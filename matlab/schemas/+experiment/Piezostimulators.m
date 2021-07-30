%{
# list of piezostimulators in various setups
id                      : int               # some number that identifies the piezostimulator
rig                     : char              # setup name
location                : tinyint           # 1 = Left, 2 = Right
%}

classdef Piezostimulators < dj.Manual
end

