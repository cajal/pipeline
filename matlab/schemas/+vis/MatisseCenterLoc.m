%{
vis.MatisseCenterLoc (lookup) # location of center for Matisse scans.
id              : smallint               # one is default, 2 is online change.
---
x_loc                       : decimal(4,3)                  # 
y_loc                       : decimal(4,3)                  # 
r                           : decimal(4,3)                  # 
%}


classdef MatisseCenterLoc < dj.Relvar
end