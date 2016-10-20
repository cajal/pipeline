%{
experiment.BrainArea (lookup) # recording brain area
brain_area     : char(12)               # brain area short name
---
area_description             : varchar(2048)                 # 
%}


classdef BrainArea < dj.Relvar
end